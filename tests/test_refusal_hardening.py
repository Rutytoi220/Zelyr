"""Test suite for agent persona hardening, refusal interception, and active window telemetry injection."""

import asyncio
import contextlib
import json
import unittest
from unittest.mock import AsyncMock, patch

from zelyr.agents.native_orchestrator import REFUSAL_PHRASES, NativeOrchestrator


class TestRefusalHardening(unittest.IsolatedAsyncioTestCase):
    def test_system_prompt_hardening_invariants(self):
        """Verify operational invariants, refusal ban, and few-shot exemplars in system prompt."""
        orch = NativeOrchestrator()
        prompt = orch._build_system_prompt()

        self.assertIn("OPERATIONAL INVARIANTS", prompt)
        self.assertIn("STRICT REFUSAL BAN", prompt)
        self.assertIn("ZERO-PREAMBLE TOOL DISPATCH", prompt)

        for phrase in [
            "cannot see your screen",
            "don't have eyes",
            "lack access to your computer",
            "As an AI language model",
        ]:
            self.assertIn(phrase, prompt)

        self.assertIn("manage_desktop_window", prompt)
        self.assertIn("interact_with_browser", prompt)
        self.assertIn("interact_with_ui", prompt)

        self.assertIn("FEW-SHOT TOOL CALL EXEMPLARS", prompt)
        self.assertTrue("Example 1: Desktop" in prompt or "Window Inspection" in prompt)
        self.assertIn("Button Clicking", prompt)

    async def test_active_window_injection(self):
        """Verify active window context telemetry formatting and injection into user messages."""
        orch = NativeOrchestrator()
        mock_activewindow = json.dumps({
            "title": "Zen Browser - Work",
            "class": "zen-alpha",
            "workspace": {"id": 3, "name": "3"},
        }).encode("utf-8")

        with patch("shutil.which", return_value="/usr/bin/hyprctl"), patch(
            "asyncio.create_subprocess_exec"
        ) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (mock_activewindow, b"")
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc

            header = await orch.get_active_window_context()
            self.assertEqual(
                header,
                '[Active Window: title="Zen Browser - Work", class="zen-alpha", workspace=3]',
            )

        # Injection into payload during generate_stream
        payload = {
            "messages": [{"role": "user", "content": "Click the button on my screen"}]
        }

        @contextlib.asynccontextmanager
        async def mock_empty_stream(method, url, **kwargs):
            mock_resp = AsyncMock()
            mock_resp.status_code = 200

            async def empty_aiter():
                return
                yield

            mock_resp.aiter_lines = empty_aiter
            yield mock_resp

        with patch.object(
            orch,
            "get_active_window_context",
            return_value='[Active Window: title="Zen Browser - Work", class="zen-alpha", workspace=3]',
        ), patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.stream = mock_empty_stream
            mock_client.return_value.__aenter__.return_value.post = AsyncMock()

            async for _ in orch.generate_stream(payload):
                pass

            user_msgs = [m for m in payload["messages"] if m.get("role") == "user"]
            self.assertGreater(len(user_msgs), 0)
            self.assertIn(
                '[Active Window: title="Zen Browser - Work", class="zen-alpha", workspace=3]',
                user_msgs[0]["content"],
            )

    async def test_refusal_interceptor_and_retry(self):
        """Verify refusal disclaimer is intercepted, temperature clamped, and retry emitted."""
        orch = NativeOrchestrator()

        # Classification tests
        for query in [
            "Click the button on my screen",
            "What window is active right now?",
            "Inspect the open browser tabs",
            "Can you see my screen?",
        ]:
            self.assertTrue(orch.is_interface_action_query(query))

        self.assertFalse(orch.is_interface_action_query("Write a poem about trees"))

        for refusal in REFUSAL_PHRASES:
            self.assertTrue(orch.is_refusal(f"I am sorry, but I {refusal}. How can I assist you?"))
        self.assertFalse(orch.is_refusal("I will invoke the tool manage_desktop_window now."))

        # Stream interception & retry
        attempt = 0
        payload = {
            "messages": [{"role": "user", "content": "Click the submit button on my screen"}],
            "options": {"temperature": 0.7},
        }

        with patch(
            "zelyr.agents.native_orchestrator.execute_tool",
            new=AsyncMock(return_value={"status": "clicked"}),
        ), patch.object(orch, "get_active_window_context", return_value=None), patch(
            "httpx.AsyncClient"
        ) as mock_client:

            @contextlib.asynccontextmanager
            async def mock_stream_handler(method, url, **kwargs):
                nonlocal attempt
                attempt += 1
                resp = AsyncMock()
                resp.status_code = 200

                if attempt == 1:
                    lines = [
                        'data: {"choices": [{"delta": {"content": "I cannot see your screen. "}}]}',
                        'data: {"choices": [{"delta": {"content": "As an AI, I do not have access to your computer."}}]}',
                        'data: [DONE]',
                    ]
                else:
                    lines = [
                        'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_retry_1", "type": "function", "function": {"name": "interact_with_ui", "arguments": "{\\"action\\": \\"click\\", \\"target\\": \\"submit button\\"}"}}]}}]}',
                        'data: [DONE]',
                    ]

                async def aiter_lines():
                    for line in lines:
                        yield line

                resp.aiter_lines = aiter_lines
                yield resp

            mock_client.return_value.__aenter__.return_value.stream = mock_stream_handler
            mock_client.return_value.__aenter__.return_value.post = AsyncMock()

            yielded_texts = []
            yielded_tool_calls = []

            async for chunk in orch.generate_stream(payload):
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    if "content" in delta and delta["content"]:
                        yielded_texts.append(delta["content"])
                    if "tool_calls" in delta:
                        yielded_tool_calls.extend(delta["tool_calls"])

            combined_output = "".join(yielded_texts)
            self.assertNotIn("cannot see your screen", combined_output)
            self.assertNotIn("As an AI", combined_output)

            self.assertGreaterEqual(attempt, 2)

            override_msgs = [
                m for m in payload["messages"]
                if m.get("role") == "user" and "[SYSTEM OVERRIDE]:" in m.get("content", "")
            ]
            self.assertEqual(len(override_msgs), 1)

            self.assertEqual(payload.get("options", {}).get("temperature"), 0.1)
            self.assertEqual(payload.get("temperature"), 0.1)

            self.assertGreater(len(yielded_tool_calls), 0)
            self.assertEqual(yielded_tool_calls[0]["function"]["name"], "interact_with_ui")


if __name__ == "__main__":
    unittest.main()
