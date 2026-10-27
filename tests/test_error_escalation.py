"""Test suite for Closed-Loop Fallback & Error Escalation State Machine (Phase 2).

Verifies:
1. ToolResult envelope contract (success, output, error, remedy_hint, allowed_actions).
2. Failure envelopes for common browser failure modes:
   - click_element / fill_element missing element -> recovery: get_page_snapshot or scroll_page.
   - switch_tab missing query -> recovery: list_tabs or open_tab.
3. NativeOrchestrator autonomous closed-loop recovery:
   - Ingests ToolResult structured failure and formats into tool message with Recovery Hint.
   - Rejects conversational tutorial prose attempting to avoid actuator execution.
   - Automatically re-prompts Ollama and executes recovery tool call.
"""

import contextlib
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.tools.base import ToolResult
from zelyr.tools.browser_cdp import InteractWithBrowserTool
from zelyr.tools.browser_extension import BrowserResult


class TestErrorEscalation(unittest.IsolatedAsyncioTestCase):
    def test_tool_result_envelope_contract(self):
        """Verify ToolResult envelope properly encapsulates error and recovery metadata."""
        # Success envelope
        ok_res = ToolResult(success=True, output={"status": "clicked", "element_id": 42})
        self.assertTrue(ok_res.success)
        self.assertIsNone(ok_res.error)
        self.assertIsNone(ok_res.remedy_hint)
        self.assertEqual(ok_res.output["element_id"], 42)

        # Failure envelope with recovery hints and allowed actions
        err_res = ToolResult(
            success=False,
            error="Element ID 42 not found on page.",
            remedy_hint="Call get_page_snapshot to refresh element IDs, or scroll_page down if the element is below the viewport.",
            allowed_actions=["get_page_snapshot", "scroll_page"],
        )
        self.assertFalse(err_res.success)
        self.assertEqual(err_res.error, "Element ID 42 not found on page.")
        self.assertIn("Call get_page_snapshot", err_res.remedy_hint)
        self.assertIn("scroll_page", err_res.allowed_actions)
        self.assertIsNone(err_res.output)

    async def test_simulated_click_and_fill_element_failure_envelope(self):
        """Verify interact_with_browser emits structured failure envelopes with remedy hints."""
        tool = InteractWithBrowserTool()

        with patch("zelyr.tools.browser_extension.get_bridge") as mock_get_bridge:
            mock_bridge = MagicMock()
            mock_bridge.is_connected.return_value = True
            mock_bridge.server = object()
            mock_bridge.ensure_server = AsyncMock()

            # 1. click_element failure
            mock_bridge.send_command = AsyncMock(
                return_value=BrowserResult({
                    "success": False,
                    "action": "click_element",
                    "error": "Element not found for id #42",
                })
            )
            mock_get_bridge.return_value = mock_bridge

            res_click = await tool.execute({"action": "click_element", "element_id": 42})
            self.assertFalse(res_click.success)
            self.assertIn("Element ID 42 not found on page.", res_click.error)
            self.assertEqual(
                res_click.remedy_hint,
                "Call get_page_snapshot to refresh element IDs, or scroll_page down if the element is below the viewport.",
            )
            self.assertEqual(res_click.allowed_actions, ["get_page_snapshot", "scroll_page"])

            envelope_click = res_click.to_dict("interact_with_browser", {"action": "click_element", "element_id": 42})
            self.assertFalse(envelope_click["success"])
            self.assertIn("remedy_hint", envelope_click)

            # 2. fill_element failure
            mock_bridge.send_command = AsyncMock(
                return_value=BrowserResult({
                    "success": False,
                    "action": "fill_element",
                    "error": "Element not found for id #99",
                })
            )

            res_fill = await tool.execute({"action": "fill_element", "element_id": 99, "text": "zelyr"})
            self.assertFalse(res_fill.success)
            self.assertIn("Element ID 99 not found on page.", res_fill.error)
            self.assertEqual(
                res_fill.remedy_hint,
                "Call get_page_snapshot to refresh element IDs, or scroll_page down if the element is below the viewport.",
            )
            self.assertEqual(res_fill.allowed_actions, ["get_page_snapshot", "scroll_page"])

    async def test_simulated_switch_tab_failure_envelope(self):
        """Verify switch_tab failure envelope advises list_tabs or open_tab."""
        tool = InteractWithBrowserTool()

        with patch("zelyr.tools.browser_extension.get_bridge") as mock_get_bridge:
            mock_bridge = MagicMock()
            mock_bridge.is_connected.return_value = True
            mock_bridge.server = object()
            mock_bridge.ensure_server = AsyncMock()

            mock_bridge.send_command = AsyncMock(
                return_value=BrowserResult({
                    "success": False,
                    "action": "switch_tab",
                    "error": "No open tab matching query: 'reddit'",
                    "open_tabs": [{"id": 1, "title": "GitHub", "url": "https://github.com"}],
                })
            )
            mock_get_bridge.return_value = mock_bridge

            res = await tool.execute({"action": "switch_tab", "query": "reddit"})
            self.assertFalse(res.success)
            self.assertIn("No open tab matching query: 'reddit'", res.error)
            self.assertIn("GitHub", res.error)
            self.assertEqual(
                res.remedy_hint,
                "Call list_tabs to see valid targets, or open_tab with url='reddit' to launch it.",
            )
            self.assertEqual(res.allowed_actions, ["list_tabs", "open_tab"])

    async def test_orchestrator_ingests_recovery_hint_and_dispatches_snapshot(self):
        """Test orchestrator receives a ToolResult error envelope, injects recovery hint,
        rejects conversational advisory prose, and executes get_page_snapshot.
        """
        orch = NativeOrchestrator()
        executed_tools = []

        async def mock_execute(name, **kwargs):
            executed_tools.append((name, kwargs))
            action = kwargs.get("action")
            if action == "click_element":
                return ToolResult(
                    success=False,
                    error="Element ID 42 not found on page.",
                    remedy_hint="Call get_page_snapshot to refresh element IDs, or scroll_page down if the element is below the viewport.",
                    allowed_actions=["get_page_snapshot", "scroll_page"],
                )
            elif action == "get_page_snapshot":
                return {"status": "success", "elements": [{"id": 1, "text": "OK"}]}
            return {"status": "ok"}

        turn_counter = 0

        @contextlib.asynccontextmanager
        async def mock_stream_client(method, url, **kwargs):
            nonlocal turn_counter
            turn_counter += 1
            resp = AsyncMock()
            resp.status_code = 200

            if turn_counter == 1:
                lines = [
                    'data: {"choices": [{"delta": {"content": "{\\"name\\": \\"interact_with_browser\\", \\"arguments\\": {\\"action\\": \\"click_element\\", \\"element_id\\": 42}}"}}]}',
                    "data: [DONE]",
                ]
            elif turn_counter == 2:
                lines = [
                    'data: {"choices": [{"delta": {"content": "I apologize, element 42 was not found on the page. Please inspect the DOM manually or refresh the browser tab."}}]}',
                    "data: [DONE]",
                ]
            elif turn_counter == 3:
                lines = [
                    'data: {"choices": [{"delta": {"content": "{\\"name\\": \\"interact_with_browser\\", \\"arguments\\": {\\"action\\": \\"get_page_snapshot\\"}}"}}]}',
                    "data: [DONE]",
                ]
            else:
                lines = [
                    'data: {"choices": [{"delta": {"content": "I refreshed the page snapshot and found updated elements 1 and 2."}}]}',
                    "data: [DONE]",
                ]

            async def aiter_lines():
                for l in lines:
                    yield l

            resp.aiter_lines = aiter_lines
            yield resp

        payload = {
            "messages": [{"role": "user", "content": "Click button 42 on the active web page"}],
        }

        with patch("zelyr.agents.native_orchestrator.execute_tool", side_effect=mock_execute), \
             patch.object(orch, "get_active_window_context", return_value=None), \
             patch("httpx.AsyncClient") as mock_client:

            mock_client.return_value.__aenter__.return_value.stream = mock_stream_client
            mock_client.return_value.__aenter__.return_value.post = AsyncMock()

            streamed_content = []
            async for chunk in orch.generate_stream(payload):
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    if "content" in delta and delta["content"]:
                        streamed_content.append(delta["content"])

            final_text = "".join(streamed_content)

            self.assertEqual(len(executed_tools), 2)
            self.assertEqual(executed_tools[0][1]["action"], "click_element")
            self.assertEqual(executed_tools[0][1]["element_id"], 42)
            self.assertEqual(executed_tools[1][1]["action"], "get_page_snapshot")

            self.assertNotIn("I apologize, element 42 was not found", final_text)
            self.assertIn("I refreshed the page snapshot and found updated elements", final_text)

            tool_messages = [m for m in payload["messages"] if m.get("role") == "tool"]
            self.assertGreaterEqual(len(tool_messages), 1)
            error_msg = tool_messages[0]
            self.assertIn("[Tool Error]: Element ID 42 not found on page.", error_msg["content"])
            self.assertIn(
                "Recovery Hint: Call get_page_snapshot to refresh element IDs, or scroll_page down if the element is below the viewport.",
                error_msg["content"],
            )

    async def test_orchestrator_handles_switch_tab_failure_and_prompts_open_tab(self):
        """Test orchestrator handles switch_tab failure, rejects advisory prose,
        and executes open_tab recovery tool.
        """
        orch = NativeOrchestrator()
        executed_tools = []

        async def mock_execute(name, **kwargs):
            executed_tools.append((name, kwargs))
            action = kwargs.get("action")
            if action == "switch_tab":
                return ToolResult(
                    success=False,
                    error="No open tab matching query: 'reddit'",
                    remedy_hint="Call list_tabs to see valid targets, or open_tab with url='https://reddit.com' to launch it.",
                    allowed_actions=["list_tabs", "open_tab"],
                )
            elif action == "open_tab":
                return {"status": "success", "tab_id": 202, "url": kwargs.get("url")}
            return {"status": "ok"}

        turn_counter = 0

        @contextlib.asynccontextmanager
        async def mock_stream_client(method, url, **kwargs):
            nonlocal turn_counter
            turn_counter += 1
            resp = AsyncMock()
            resp.status_code = 200

            if turn_counter == 1:
                lines = [
                    'data: {"choices": [{"delta": {"content": "{\\"name\\": \\"interact_with_browser\\", \\"arguments\\": {\\"action\\": \\"switch_tab\\", \\"query\\": \\"reddit\\"}}"}}]}',
                    "data: [DONE]",
                ]
            elif turn_counter == 2:
                lines = [
                    'data: {"choices": [{"delta": {"content": "No reddit tab was found. You should open Firefox and navigate to reddit.com."}}]}',
                    "data: [DONE]",
                ]
            elif turn_counter == 3:
                lines = [
                    'data: {"choices": [{"delta": {"content": "{\\"name\\": \\"interact_with_browser\\", \\"arguments\\": {\\"action\\": \\"open_tab\\", \\"url\\": \\"https://reddit.com\\"}}"}}]}',
                    "data: [DONE]",
                ]
            else:
                lines = [
                    'data: {"choices": [{"delta": {"content": "I opened reddit.com in a new tab."}}]}',
                    "data: [DONE]",
                ]

            async def aiter_lines():
                for l in lines:
                    yield l

            resp.aiter_lines = aiter_lines
            yield resp

        payload = {
            "messages": [{"role": "user", "content": "Switch to my reddit tab"}],
        }

        with patch("zelyr.agents.native_orchestrator.execute_tool", side_effect=mock_execute), \
             patch.object(orch, "get_active_window_context", return_value=None), \
             patch("httpx.AsyncClient") as mock_client:

            mock_client.return_value.__aenter__.return_value.stream = mock_stream_client
            mock_client.return_value.__aenter__.return_value.post = AsyncMock()

            streamed_content = []
            async for chunk in orch.generate_stream(payload):
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    if "content" in delta and delta["content"]:
                        streamed_content.append(delta["content"])

            final_text = "".join(streamed_content)

            self.assertEqual(len(executed_tools), 2)
            self.assertEqual(executed_tools[0][1]["action"], "switch_tab")
            self.assertEqual(executed_tools[1][1]["action"], "open_tab")
            self.assertEqual(executed_tools[1][1]["url"], "https://reddit.com")

            self.assertNotIn("No reddit tab was found. You should open Firefox", final_text)
            self.assertIn("I opened reddit.com in a new tab.", final_text)

            tool_messages = [m for m in payload["messages"] if m.get("role") == "tool"]
            self.assertGreaterEqual(len(tool_messages), 1)
            self.assertIn("Recovery Hint: Call list_tabs to see valid targets", tool_messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
