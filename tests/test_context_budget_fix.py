"""Test suite for 32K Ollama context window expansion and tier-aware tool schema pruning."""

import asyncio
import json
import unittest
from unittest.mock import patch, MagicMock

import httpx

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.core.plugins import get_tool_schemas, filter_tool_schemas_by_tier, load_plugins


class TestContextBudgetFix(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        load_plugins()
        self.orchestrator = NativeOrchestrator()
        self.all_schemas = get_tool_schemas(0)

    def test_max_context_tokens_is_8192(self):
        """NativeOrchestrator max_context_tokens should default to 8192."""
        self.assertEqual(self.orchestrator.max_context_tokens, 8192)

    def test_tier1_tool_filtering(self):
        """Tier 1 queries should only include core tools and Tier 1 OS/desktop tools."""
        query = "Send a desktop notification saying task completed"
        tier = self.orchestrator.route_tier(query)
        self.assertEqual(tier, "tier1_ipc")

        filtered = filter_tool_schemas_by_tier(self.all_schemas, tier)
        filtered_names = {s["function"]["name"] for s in filtered}

        # Should be strictly smaller than full registry (~9 tools vs 36)
        self.assertLess(len(filtered), len(self.all_schemas))
        self.assertEqual(len(filtered), 9)

        # Must include core and Tier 1 tools
        expected_tier1 = {
            "execute_command",
            "manage_desktop_window",
            "manage_system_process",
            "manage_system_clipboard",
            "query_system_journal",
            "manage_media_playback",
            "manage_workspace_file",
            "send_desktop_notification",
            "inspect_network",
        }
        self.assertEqual(filtered_names, expected_tier1)
        self.assertNotIn("interact_with_browser", filtered_names)

    def test_tier2_tool_filtering(self):
        """Tier 2 queries should only include core tools and interact_with_browser."""
        query = "Scroll down in browser to read the web page"
        tier = self.orchestrator.route_tier(query)
        self.assertEqual(tier, "tier2_browser")

        filtered = filter_tool_schemas_by_tier(self.all_schemas, tier)
        filtered_names = {s["function"]["name"] for s in filtered}

        self.assertEqual(len(filtered), 5)
        expected_tier2 = {
            "execute_command",
            "manage_desktop_window",
            "interact_with_browser",
            "web_search",
            "fetch_web_page",
        }
        self.assertEqual(filtered_names, expected_tier2)
        self.assertNotIn("send_desktop_notification", filtered_names)
        self.assertNotIn("manage_media_playback", filtered_names)
        self.assertNotIn("manage_system_process", filtered_names)

    def test_tier3_full_registry_fallback(self):
        """Tier 3 or general canvas queries should preserve the full registry."""
        query = "Click the canvas pixel in the retro game with no dom"
        tier = self.orchestrator.route_tier(query)
        self.assertEqual(tier, "tier3_vision")

        filtered = filter_tool_schemas_by_tier(self.all_schemas, tier)
        self.assertEqual(len(filtered), len(self.all_schemas))

    async def test_simulated_dispatch_num_ctx_and_options_preservation(self):
        """Verify outgoing payload includes num_ctx: 8192, preserves options, and filters tools."""
        captured_requests = []

        class MockStreamResponse:
            def __init__(self):
                self.status_code = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

            async def aiter_lines(self):
                # Emit a streaming response chunk with a tool call
                chunk = {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "send_desktop_notification",
                                    "arguments": json.dumps({"title": "Test", "message": "Done"}),
                                }
                            }]
                        }
                    }]
                }
                yield f"data: {json.dumps(chunk)}"
                yield "data: [DONE]"

            async def aread(self):
                return b""

        class MockAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

            async def post(self, url, **kwargs):
                captured_requests.append(("POST", url, kwargs))
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                return mock_resp

            def stream(self, method, url, **kwargs):
                captured_requests.append((method, url, kwargs))
                return MockStreamResponse()

        payload = {
            "messages": [
                {"role": "user", "content": "Send a desktop notification saying hello"}
            ],
            "options": {
                "temperature": 0.3,
                "top_p": 0.85,
            },
        }

        with patch("httpx.AsyncClient", MockAsyncClient):
            chunks = []
            async for chunk in self.orchestrator.generate_stream(payload):
                chunks.append(chunk)

            # Check that streaming succeeded and emitted tool call
            self.assertGreater(len(chunks), 0)

            # Find the completions request
            comp_req = next(r for r in captured_requests if "/v1/chat/completions" in r[1])
            req_body = comp_req[2]["json"]

            # 1. num_ctx must be 8192
            self.assertEqual(req_body["options"]["num_ctx"], 8192)

            # 2. Existing temperature and top_p must be preserved
            self.assertEqual(req_body["options"]["temperature"], 0.3)
            self.assertEqual(req_body["options"]["top_p"], 0.85)

            # 3. Tool schemas must be pruned to Tier 1 (~9 tools instead of 36)
            self.assertEqual(len(req_body["tools"]), 9)
            tool_names = [t["function"]["name"] for t in req_body["tools"]]
            self.assertIn("send_desktop_notification", tool_names)
            self.assertNotIn("interact_with_browser", tool_names)


if __name__ == "__main__":
    unittest.main()
