"""Comprehensive unit and protocol tests for FallbackToolParser and NativeOrchestrator integration."""

from __future__ import annotations

import json
from typing import AsyncGenerator
from unittest.mock import AsyncMock, patch

import pytest

from zelyr.agents.fallback_parser import ExtractedToolCall, FallbackToolParser
from zelyr.agents.native_orchestrator import NativeOrchestrator


class TestFallbackToolParserFormats:
    """Verify extraction across Markdown, XML, and embedded JSON formats."""

    def test_markdown_code_fence_with_json_specifier(self):
        text = """I will inspect the system uptime.
```json
{"name": "execute_command", "arguments": {"command": "uptime"}}
```"""
        calls = FallbackToolParser.parse_tool_calls(text)
        assert len(calls) == 1
        call = calls[0]
        assert isinstance(call, ExtractedToolCall)
        assert call.name == "execute_command"
        assert call.arguments == {"command": "uptime"}
        assert call.tool_call_id.startswith("synth_")
        assert call.cleaned_text == "I will inspect the system uptime."

    def test_markdown_code_fence_bare_and_flat_arguments(self):
        text = """```
{"tool": "execute_command", "command": "uptime"}
```"""
        calls = FallbackToolParser.parse_tool_calls(text)
        assert len(calls) == 1
        call = calls[0]
        assert call.name == "execute_command"
        assert call.arguments == {"command": "uptime"}
        assert call.tool_call_id.startswith("synth_")
        assert call.cleaned_text == ""

    def test_xml_pseudo_tags_tool_call(self):
        text = """<tool_call>{"name": "manage_workspace_file", "arguments": {"action": "read", "path": "test.txt"}}</tool_call>"""
        calls = FallbackToolParser.parse_tool_calls(text)
        assert len(calls) == 1
        call = calls[0]
        assert call.name == "manage_workspace_file"
        assert call.arguments == {"action": "read", "path": "test.txt"}
        assert call.tool_call_id.startswith("synth_")
        assert call.cleaned_text == ""

    def test_xml_pseudo_tags_function_call(self):
        text = """Reasoning before action: reading instructions.
<function_call>
{
  "name": "manage_workspace_file",
  "arguments": {
    "action": "read",
    "path": "instructions.md"
  }
}
</function_call>"""
        calls = FallbackToolParser.parse_tool_calls(text)
        assert len(calls) == 1
        call = calls[0]
        assert call.name == "manage_workspace_file"
        assert call.arguments == {"action": "read", "path": "instructions.md"}
        assert call.tool_call_id.startswith("synth_")
        assert call.cleaned_text == "Reasoning before action: reading instructions."

    def test_stringified_arguments_unpacked(self):
        text = """{"name": "execute_command", "arguments": "{\\"command\\": \\"pwd\\"}"}"""
        calls = FallbackToolParser.parse_tool_calls(text)
        assert len(calls) == 1
        call = calls[0]
        assert call.name == "execute_command"
        assert call.arguments == {"command": "pwd"}
        assert call.tool_call_id.startswith("synth_")
        assert call.cleaned_text == ""

    def test_text_cleanup_preserves_preceding_and_subsequent_commentary(self):
        text = """Step 1: Check network connectivity.
```json
{"name": "execute_command", "arguments": {"command": "ping -c 1 127.0.0.1"}}
```
End of plan."""
        calls = FallbackToolParser.parse_tool_calls(text)
        assert len(calls) == 1
        assert "```json" not in calls[0].cleaned_text
        assert "execute_command" not in calls[0].cleaned_text
        assert "Step 1: Check network connectivity." in calls[0].cleaned_text
        assert "End of plan." in calls[0].cleaned_text


class TestFallbackParserValidationAndDefenses:
    """Verify rejection of unknown tools, deterministic IDs, and ambiguity resolution."""

    def test_unrecognized_tool_rejected(self):
        text = """```json
{"message": "hello world", "status": "success"}
```"""
        calls = FallbackToolParser.parse_tool_calls(text)
        assert calls == []

    def test_deterministic_synthetic_id_generation(self):
        text = """```json
{"name": "execute_command", "arguments": {"command": "df -h"}}
```"""
        calls_1 = FallbackToolParser.parse_tool_calls(text)
        calls_2 = FallbackToolParser.parse_tool_calls(text)

        assert calls_1[0].tool_call_id == calls_2[0].tool_call_id
        assert calls_1[0].tool_call_id.startswith("synth_")
        assert len(calls_1[0].tool_call_id) == 18  # "synth_" (6) + 12 hex

    def test_conflicting_blocks_extracts_only_valid_registered_tool(self):
        text = """Status response:
```json
{"status": "in_progress", "code": 100}
```
Executing action:
```json
{"name": "execute_command", "arguments": {"command": "uptime"}}
```"""
        calls = FallbackToolParser.parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "execute_command"
        assert calls[0].arguments == {"command": "uptime"}
        # Verify the non-matching block remains in cleaned_text
        assert '{"status": "in_progress", "code": 100}' in calls[0].cleaned_text
        assert "uptime" not in calls[0].cleaned_text

    def test_multiple_valid_tools_extracted_sequentially(self):
        text = """First command:
```json
{"name": "execute_command", "arguments": {"command": "whoami"}}
```
Second command:
```json
{"name": "execute_command", "arguments": {"command": "hostname"}}
```"""
        calls = FallbackToolParser.parse_tool_calls(text, max_calls=3)
        assert len(calls) == 2
        assert calls[0].name == "execute_command"
        assert calls[0].arguments == {"command": "whoami"}
        assert calls[1].name == "execute_command"
        assert calls[1].arguments == {"command": "hostname"}
        assert "whoami" not in calls[0].cleaned_text
        assert "hostname" not in calls[0].cleaned_text
        assert "First command:" in calls[0].cleaned_text
        assert "Second command:" in calls[0].cleaned_text

    def test_unauthorized_ring_tool_injection_rejected(self):
        # Mock ToolRegistry where ring check fails
        with patch("zelyr.tool_registry.ToolRegistry._get_tool_ring", return_value=0):
            # caller_ring is 1, tool requires ring 0
            calls = FallbackToolParser.parse_tool_calls(
                """```json\n{"name": "execute_command", "arguments": {"command": "reboot"}}\n```""",
                caller_ring=1,
            )
            assert calls == []


class TestNativeOrchestratorFallbackIntegration:
    """Verify NativeOrchestrator recovers degraded text streams into wire-safe tool executions."""

    @pytest.mark.asyncio
    async def test_native_orchestrator_recovers_markdown_tool_call(self):
        orch = NativeOrchestrator()
        sample_text = """I will check the directory contents.
```json
{"name": "execute_command", "arguments": {"command": "ls -la"}}
```"""

        executed_tools = []

        async def mock_tool_executor(tool_name: str, args: dict):
            executed_tools.append((tool_name, args))
            return "total 42\n-rw-r--r-- 1 test test 100 file.txt"

        turn_count = 0

        async def mock_stream_provider(payload: dict) -> AsyncGenerator[dict, None]:
            nonlocal turn_count
            turn_count += 1
            if turn_count == 1:
                # First turn: model outputs Markdown tool call in prose
                yield {"choices": [{"delta": {"content": sample_text}}]}
            else:
                # Second turn: model synthesizes final answer after observing tool output
                yield {"choices": [{"delta": {"content": "Directory contents listed successfully."}}]}

        payload = {"messages": [{"role": "user", "content": "list directory"}]}

        events = []
        async for event in orch.generate_stream(
            payload,
            stream_provider=mock_stream_provider,
            tool_executor=mock_tool_executor,
        ):
            events.append(event)

        # 1. Assert tool actuator was invoked with exact normalized arguments
        assert len(executed_tools) == 1
        assert executed_tools[0] == ("execute_command", {"command": "ls -la"})

        # 2. Assert message history wire invariants
        messages = payload["messages"]
        # Expect: [system, user, assistant(with tool_calls), tool(observation)]
        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

        asst_msg = messages[2]
        assert asst_msg["role"] == "assistant"
        assert asst_msg["content"] == "I will check the directory contents."
        assert len(asst_msg["tool_calls"]) == 1

        synth_call = asst_msg["tool_calls"][0]
        assert synth_call["function"]["name"] == "execute_command"
        assert synth_call["function"]["arguments"] == {"command": "ls -la"}
        synth_id = synth_call["id"]
        assert synth_id.startswith("synth_")

        tool_msg = messages[3]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == synth_id
        assert tool_msg["name"] == "execute_command"
        assert tool_msg["status"] == "success"
        assert "file.txt" in tool_msg["content"]

        # 3. Assert final turn streamed synthesis output
        yielded_content = "".join(
            e.get("choices", [{}])[0].get("delta", {}).get("content", "")
            for e in events
        )
        assert "Directory contents listed successfully." in yielded_content
