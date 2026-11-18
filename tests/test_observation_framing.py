"""Unit and integration tests for ObservationFramingEngine, deterministic anchors, and compaction."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock
import pytest

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.engine.budget_manager import ContextBudgetEngine
from zelyr.engine.execution_ledger import TurnExecutionLedger
from zelyr.engine.observation_framing import ObservationFramingEngine
from zelyr.tools.core import ToolResult


class TestObservationFramingEngineUnit:
    """Dedicated unit tests for ObservationFramingEngine formatting rules."""

    def test_command_success_with_clean_stdout(self):
        """Clean command success outputs invariant 4-line anchor header with Status SUCCESS and Exit Code 0."""
        raw = "Exit code 0\npytest passed: 15 tests run, 0 failures."
        framed = ObservationFramingEngine.frame_observation("execute_command", raw)

        # Invariant 4-line header assertions
        lines = framed.splitlines()
        assert lines[0] == "[TOOL OBSERVATION: execute_command]"
        assert lines[1] == "- Status: SUCCESS"
        assert lines[2] == "- Exit Code: 0"
        assert lines[3] == "- Verified: N/A"
        assert "- Summary: Command executed successfully with exit code 0." in lines[4]

        # Output body assertion
        assert "[Output]:" in framed
        assert "pytest passed: 15 tests run, 0 failures." in framed
        assert "[Informational Diagnostic Stream" not in framed
        assert "[Error Diagnostic:" not in framed

        # Parse round-trip
        parsed = ObservationFramingEngine.parse_observation(framed)
        assert parsed["tool"] == "execute_command"
        assert parsed["status"] == "success"
        assert parsed["exit_code"] == 0
        assert parsed["verified"] is None
        assert "pytest passed" in parsed["output"]

    def test_command_success_with_noisy_stderr_git_checkout(self):
        """Command success with stderr (e.g. git checkout) formats stderr as informational without failing."""
        tool_result = ToolResult(
            success=True,
            output="Switched to branch 'feature/1.4-model-independence'",
            error=None,
            metadata={"stderr": "M  zelyr/engine/budget_manager.py\nAlready up to date."},
        )
        framed = ObservationFramingEngine.frame_observation("execute_command", tool_result)

        assert "[TOOL OBSERVATION: execute_command]" in framed
        assert "- Status: SUCCESS" in framed
        assert "- Exit Code: 0" in framed
        assert "[Output]:" in framed
        assert "Switched to branch 'feature/1.4-model-independence'" in framed

        # Explicit informational annotation
        assert "[Informational Diagnostic Stream (stderr)]:" in framed
        assert "Already up to date." in framed
        assert "(Note: Command executed successfully with exit code 0; this stream contains non-error progress info.)" in framed

        # Parse round-trip
        parsed = ObservationFramingEngine.parse_observation(framed)
        assert parsed["status"] == "success"
        assert parsed["exit_code"] == 0
        assert "Switched to branch" in parsed["output"]
        assert "Already up to date." in parsed["stderr"]

    def test_command_failure_exit_code_127(self):
        """Command failure (exit code 127) formats typed Error Diagnostic and recovery hint."""
        raw = "Exit code 127\nbash: ripgrep: command not found"
        framed = ObservationFramingEngine.frame_observation("execute_command", raw)

        assert "[TOOL OBSERVATION: execute_command]" in framed
        assert "- Status: ERROR" in framed
        assert "- Exit Code: 127" in framed
        assert "- Verified: N/A" in framed

        # Diagnostic block
        assert "[Error Diagnostic: COMMAND_NOT_FOUND]" in framed
        assert "- Failure Reason: bash: ripgrep: command not found" in framed
        assert "- Recovery Hint: Command not found in PATH." in framed
        assert "- Allowed Actions:" in framed
        assert "execute_command" in framed

        # Backward compatibility marker
        assert "[Tool Error]: bash: ripgrep: command not found. Recovery Hint: Command not found in PATH." in framed

        # Parse round-trip
        parsed = ObservationFramingEngine.parse_observation(framed)
        assert parsed["status"] == "error"
        assert parsed["exit_code"] == 127
        assert parsed["error_type"] == "COMMAND_NOT_FOUND"
        assert "Command not found in PATH" in parsed["remedy_hint"]
        assert "execute_command" in parsed["allowed_actions"]

    def test_verified_outcome_success(self):
        """Deterministic verification affirming machine state formats Verified TRUE and affirmative block."""
        verify_dict = {
            "verified": True,
            "status": "success",
            "check": "file_contains",
            "target": "/etc/hosts",
            "observed": "127.0.0.1 localhost",
            "output": "Content matched expected pattern",
        }
        framed = ObservationFramingEngine.frame_observation("verify_outcome", verify_dict)

        assert "[TOOL OBSERVATION: verify_outcome]" in framed
        assert "- Status: SUCCESS" in framed
        assert "- Verified: TRUE" in framed
        assert "- Summary: Outcome matched expected criteria and machine state verified." in framed
        assert "[Machine State Verified]: 127.0.0.1 localhost" in framed

        parsed = ObservationFramingEngine.parse_observation(framed)
        assert parsed["status"] == "success"
        assert parsed["verified"] is True

    def test_verification_failure_error(self):
        """Deterministic verification failure formats Verified FALSE and typed VERIFICATION_FAILED diagnostic."""
        verify_dict = {
            "verified": False,
            "status": "error",
            "check": "port_listening",
            "target": "8080",
            "observed": "Port 8080 is not listening",
            "error": "Port 8080 is not listening",
        }
        framed = ObservationFramingEngine.frame_observation("verify_outcome", verify_dict)

        assert "[TOOL OBSERVATION: verify_outcome]" in framed
        assert "- Status: ERROR" in framed
        assert "- Verified: FALSE" in framed
        assert "[Error Diagnostic: VERIFICATION_FAILED]" in framed
        assert "Observed state did not match expected criteria" in framed
        assert "[Tool Error]:" in framed
        assert "Recovery Hint:" in framed

        parsed = ObservationFramingEngine.parse_observation(framed)
        assert parsed["status"] == "error"
        assert parsed["verified"] is False
        assert parsed["error_type"] == "VERIFICATION_FAILED"

    def test_backward_compatibility_markers(self):
        """All error frames contain backward-compatible markers for legacy regex scrapers."""
        framed = ObservationFramingEngine.frame_observation(
            "manage_workspace_file",
            {"status": "error", "exit_code": 1, "error": "File /tmp/absent.txt does not exist"},
        )
        assert "[Tool Error]: File /tmp/absent.txt does not exist. Recovery Hint: Target file or directory does not exist." in framed

    def test_json_schema_conformance_and_dict_serialization(self):
        """Structured JSON schema serialization contains all canonical fields."""
        json_framed = ObservationFramingEngine.frame_observation(
            "execute_command",
            "Exit code 0\nsuccess message",
            as_json=True,
        )
        data = json.loads(json_framed)
        assert data["tool"] == "execute_command"
        assert data["status"] == "success"
        assert data["exit_code"] == 0
        assert data["verified"] is False or data["verified"] is None
        assert "summary" in data
        assert data["output"] == "success message"

        dict_data = ObservationFramingEngine.to_dict("execute_command", "Exit code 0\nsuccess message")
        assert dict_data["tool"] == "execute_command"
        assert dict_data["status"] == "success"


class TestObservationFramingBudgetCompaction:
    """Verify that ContextBudgetEngine preserves Anchor Header and Diagnostic blocks."""

    def test_context_compaction_preserves_anchor_header_and_remedy(self):
        """Compacting a 10,000-character framed observation preserves Anchor Header and Recovery Hint intact."""
        huge_stdout = "LOG_ENTRY_HEX_" * 600  # ~8,400 chars
        raw_error = {
            "status": "error",
            "exit_code": 1,
            "output": huge_stdout,
            "error": "Database lock timeout reached after 30 seconds",
        }
        framed = ObservationFramingEngine.frame_observation("execute_command", raw_error)
        assert len(framed) > 8500

        budget_engine = ContextBudgetEngine(tool_output_token_limit=250)
        compacted = budget_engine.compact_tool_output(framed, token_limit=250, keep_head_chars=500, keep_tail_chars=500)

        # 1. Anchor Header must be 100% intact
        lines = compacted.splitlines()
        assert lines[0] == "[TOOL OBSERVATION: execute_command]"
        assert lines[1] == "- Status: ERROR"
        assert lines[2] == "- Exit Code: 1"
        assert lines[3] == "- Verified: N/A"

        # 2. Output middle must be truncated
        assert "[... output truncated:" in compacted
        assert "[OUTPUT TRUNCATED FOR CONTEXT BUDGET]" in compacted

        # 3. Diagnostic block and Recovery Hint must remain completely intact
        assert "[Error Diagnostic:" in compacted
        assert "Database lock timeout reached" in compacted
        assert "Recovery Hint:" in compacted
        assert "[Tool Error]:" in compacted

        # 4. Total length should be dramatically reduced
        assert len(compacted) < 3000


class TestNativeOrchestratorLedgerObservationIntegration:
    """Verify NativeOrchestrator passes framed observation metadata to TurnExecutionLedger."""

    @pytest.mark.asyncio
    async def test_native_orchestrator_ledger_records_framed_success_and_error(self, monkeypatch):
        turn = 0
        ledger = TurnExecutionLedger()

        async def mock_stream_provider(payload: dict):
            nonlocal turn
            turn += 1
            if turn == 1:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "id": "call_ledger_1",
                                "index": 0,
                                "function": {
                                    "name": "execute_command",
                                    "arguments": json.dumps({"command": "which non_existent_binary"}),
                                },
                            }]
                        }
                    }]
                }
            elif turn == 2:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "id": "call_ledger_2",
                                "index": 0,
                                "function": {
                                    "name": "verify_outcome",
                                    "arguments": json.dumps({"check": "file_exists", "target": "/tmp/test"}),
                                },
                            }]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Workflow complete."}}]}

        async def mock_tool_executor(name: str, **kwargs):
            if name == "execute_command":
                return "Exit code 127\nwhich: no non_existent_binary in (/usr/bin)"
            elif name == "verify_outcome":
                return json.dumps({
                    "verified": True,
                    "status": "success",
                    "observed": "file exists",
                })
            return "Exit code 0\n"

        orch = NativeOrchestrator(stream_provider=mock_stream_provider, tool_executor=mock_tool_executor)
        monkeypatch.setattr(orch, "get_active_window_context", AsyncMock(return_value=None))

        payload = {"messages": [{"role": "user", "content": "Execute workflow"}]}

        async for _ in orch.generate_stream(payload, ledger=ledger):
            pass

        # Verify tool messages in payload are framed strings
        tool_msgs = [m for m in payload["messages"] if m.get("role") == "tool"]
        assert len(tool_msgs) == 2

        msg1 = tool_msgs[0]
        assert "[TOOL OBSERVATION: execute_command]" in msg1["content"]
        assert "- Status: ERROR" in msg1["content"]
        assert "- Exit Code: 127" in msg1["content"]
        assert msg1["status"] == "error"

        msg2 = tool_msgs[1]
        assert "[TOOL OBSERVATION: verify_outcome]" in msg2["content"]
        assert "- Status: SUCCESS" in msg2["content"]
        assert "- Verified: TRUE" in msg2["content"]
        assert msg2["status"] == "success"

        # Verify ledger recorded exact framed values
        actions = list(ledger._records.values())
        assert len(actions) == 2

        rec1 = actions[0]
        from zelyr.engine.execution_ledger import ActionState

        assert rec1.state == ActionState.FAILED
        assert rec1.exit_code == 127
        assert rec1.verified is False or rec1.verified is None

        rec2 = actions[1]
        assert rec2.state == ActionState.VERIFIED
        assert rec2.verified is True
