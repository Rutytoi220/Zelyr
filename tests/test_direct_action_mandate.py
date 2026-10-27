"""Verification Test Suite: Direct Action Mandate & Tool Dispatch Schema Normalization.

Verifies:
1. Instruction and system prompt directives:
   - Tab action distinction: switch_tab for existing tabs vs open_tab for new website.
   - Anti-Consultant Mandate (no tutorials/plans; autonomous actuator, not advisor).
2. extract_first_tool_call tolerant parsing with preceding conversational prose + markdown fence.
3. extract_first_tool_call tolerant parsing with pre-nested tool call schema.
4. Dual-format normalization: both res["function"]["name"] and res["name"] are present.
5. generate_stream() execution when model output has prose preceding code block.
"""

import contextlib
import json
import unittest
from unittest.mock import AsyncMock, patch

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.core.instructions import InstructionManager
from zelyr.core.system_prompt import (
    HARDENED_SYSTEM_DIRECTIVES,
    SOM_REACT_SYSTEM_PROMPT,
)


class TestDirectActionMandate(unittest.IsolatedAsyncioTestCase):

    def test_instruction_text_tab_distinction_and_anti_consultant(self):
        compiled = InstructionManager().get_compiled_instructions()

        expected_switch = "To switch to an EXISTING tab, use action 'switch_tab'"
        expected_open = (
            "To open a NEW website or tab, ALWAYS use action 'open_tab' (or 'navigate_url' to change the current tab). "
            "NEVER use 'switch_tab' to open a new website."
        )

        for source_name, text in [
            ("compiled instructions", compiled),
            ("HARDENED_SYSTEM_DIRECTIVES", HARDENED_SYSTEM_DIRECTIVES),
            ("SOM_REACT_SYSTEM_PROMPT", SOM_REACT_SYSTEM_PROMPT),
        ]:
            self.assertIn(expected_switch, text, f"Missing switch_tab directive in {source_name}")
            self.assertIn(expected_open, text, f"Missing open_tab directive in {source_name}")
            self.assertIn("ANTI-CONSULTANT MANDATE", text, f"Missing ANTI-CONSULTANT MANDATE in {source_name}")
            self.assertIn("You are an autonomous actuator, not an advisor", text, f"Missing autonomous actuator in {source_name}")

    def test_extract_first_tool_call_with_prose_and_markdown(self):
        orch = NativeOrchestrator()
        known = {"manage_desktop_window", "interact_with_browser"}

        text = (
            "Please verify the current active window before proceeding:\n"
            "```json\n"
            '{"name": "manage_desktop_window", "arguments": {"action": "get_active"}}\n'
            "```"
        )
        res = orch.extract_first_tool_call(text, known)

        self.assertIsNotNone(res, "Failed to extract tool call from prose + markdown")
        self.assertEqual(res.get("type"), "function")
        self.assertEqual(res.get("function", {}).get("name"), "manage_desktop_window")
        self.assertEqual(res.get("name"), "manage_desktop_window")
        self.assertEqual(res.get("arguments", {}).get("action"), "get_active")
        self.assertEqual(res.get("function", {}).get("arguments", {}).get("action"), "get_active")

    def test_extract_first_tool_call_pre_nested_format(self):
        orch = NativeOrchestrator()
        known = {"open_tab", "interact_with_browser"}

        text = (
            "```json\n"
            '{"type": "function", "function": {"name": "open_tab", "arguments": {"url": "https://github.com"}}}\n'
            "```"
        )
        res = orch.extract_first_tool_call(text, known)

        self.assertIsNotNone(res, "Failed to extract pre-nested tool call")
        self.assertEqual(res.get("type"), "function")
        self.assertEqual(res.get("function", {}).get("name"), "open_tab")
        self.assertEqual(res.get("name"), "open_tab")
        self.assertEqual(res.get("arguments", {}).get("url"), "https://github.com")
        self.assertEqual(res.get("function", {}).get("arguments", {}).get("url"), "https://github.com")

    async def test_generate_stream_executes_tool_call_with_prose(self):
        orch = NativeOrchestrator()
        executed_tools = []

        async def mock_execute(name, **kwargs):
            executed_tools.append((name, kwargs))
            return {"status": "success", "window": "Zen Browser"}

        turn_counter = 0

        @contextlib.asynccontextmanager
        async def mock_stream_client(method, url, **kwargs):
            nonlocal turn_counter
            turn_counter += 1
            resp = AsyncMock()
            resp.status_code = 200

            if turn_counter == 1:
                content = (
                    "Please verify the current active window before proceeding:\n"
                    "```json\n"
                    '{"name": "manage_desktop_window", "arguments": {"action": "get_active"}}\n'
                    "```"
                )
                lines = [
                    f'data: {{"choices": [{{"delta": {{"content": {json.dumps(content)}}}}}]}}',
                    "data: [DONE]",
                ]
            else:
                lines = [
                    'data: {"choices": [{"delta": {"content": "The active window is Zen Browser."}}]}',
                    "data: [DONE]",
                ]

            async def aiter_lines():
                for line in lines:
                    yield line

            resp.aiter_lines = aiter_lines
            yield resp

        payload = {
            "messages": [{"role": "user", "content": "What window is open?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "manage_desktop_window",
                        "description": "Manage desktop windows",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        }

        with patch("zelyr.agents.native_orchestrator.execute_tool", side_effect=mock_execute), \
             patch.object(orch, "get_active_window_context", return_value=None), \
             patch("httpx.AsyncClient") as mock_client:

            mock_client.return_value.__aenter__.return_value.stream = mock_stream_client
            mock_client.return_value.__aenter__.return_value.post = AsyncMock()

            chunks = []
            async for chunk in orch.generate_stream(payload):
                chunks.append(chunk)

            self.assertEqual(len(executed_tools), 1)
            self.assertEqual(executed_tools[0][0], "manage_desktop_window")
            self.assertEqual(executed_tools[0][1], {"action": "get_active"})


if __name__ == "__main__":
    unittest.main()
