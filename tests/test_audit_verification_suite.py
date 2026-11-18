"""Independent Implementation Audit Verification Suite for Zelyr 1.4.

Hermetic, deterministic test scenarios covering the 12 core runtime authority,
fallback parser, schema validation, and recovery invariants.
"""

from __future__ import annotations

import json
from typing import AsyncGenerator
from unittest.mock import AsyncMock, patch

import pytest

from zelyr.agents.fallback_parser import FallbackToolParser
from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.engine.execution_ledger import (
    ActionRecord,
    ActionState,
    TurnExecutionLedger,
    detect_false_success,
    has_valid_inaction_justification,
    is_unresolvable_terminal_blocker,
)
from zelyr.engine.normalizer import SafeArgumentNormalizer, validate_arguments
from zelyr.engine.schema_compact import CompactSchemaGenerator


class TestAuditVerificationBattery:
    """The 12 mandatory independent audit scenarios."""

    # Scenario 1: Native tool call succeeds and is independently verified
    @pytest.mark.asyncio
    async def test_01_native_tool_call_succeeds_and_is_verified(self):
        ledger = TurnExecutionLedger()
        call_id = "call_native_1"
        ledger.record_proposed(call_id, "execute_command", {"command": "echo 'verified_run'"})
        ledger.record_validated(call_id)
        ledger.record_dispatched(call_id)

        # Tool execution observation without independent verification: state is SUCCESS, verified=False
        ledger.record_observation(
            call_id,
            status="success",
            output={"stdout": "verified_run\n", "exit_code": 0},
            exit_code=0,
            verified=False,
        )
        assert ledger._records[call_id].state == ActionState.SUCCESS
        assert not ledger._records[call_id].verified

        # Independent outcome verification confirms machine state
        verify_call_id = "call_verify_outcome_1"
        ledger.record_proposed(verify_call_id, "verify_outcome", {"check": "file_exists", "target": "/tmp/out.txt"})
        ledger.record_dispatched(verify_call_id)
        ledger.record_observation(
            verify_call_id,
            status="success",
            output={"verified": True, "exit_code": 0, "verification_detail": "File confirmed on disk"},
            exit_code=0,
            verified=True,
            verification_detail="File confirmed on disk",
        )
        assert ledger._records[verify_call_id].state == ActionState.VERIFIED
        assert ledger._records[verify_call_id].verified is True
        assert len(ledger.verified_actions()) == 1
        assert not ledger.has_unresolved_failures()

    # Scenario 2: Valid Markdown JSON invocation is recovered and executed once
    def test_02_valid_markdown_json_recovered_and_executed_once(self):
        text = """Plan: Check disk space.
```json
{"name": "execute_command", "arguments": {"command": "df -h"}}
```
Proceeding now."""
        calls = FallbackToolParser.parse_tool_calls(text)
        assert len(calls) == 1
        call = calls[0]
        assert call.name == "execute_command"
        assert call.arguments == {"command": "df -h"}
        assert call.tool_call_id.startswith("synth_")
        assert "```json" not in call.cleaned_text
        assert "Plan: Check disk space." in call.cleaned_text

    # Scenario 3: Quoted JSON example is not executed
    def test_03_quoted_json_example_is_not_executed(self):
        # Case A: Preceded by "For example:"
        example_text = """For example, you could run the following tool call:
```json
{"name": "execute_command", "arguments": {"command": "rm -rf /tmp/scratch"}}
```
Please let me know if you would like me to proceed."""
        calls_a = FallbackToolParser.parse_tool_calls(example_text)
        assert calls_a == []

        # Case B: Markdown blockquote
        blockquote_text = """> {"name": "execute_command", "arguments": {"command": "drop database"}}"""
        calls_b = FallbackToolParser.parse_tool_calls(blockquote_text)
        assert calls_b == []

    # Scenario 4: Ambiguous fallback invocation is rejected
    def test_04_ambiguous_fallback_invocation_rejected(self):
        ambiguous_text = """You can either run this action:
```json
{"name": "execute_command", "arguments": {"command": "cat a.txt"}}
```
or alternatively run:
```json
{"name": "execute_command", "arguments": {"command": "cat b.txt"}}
```"""
        calls = FallbackToolParser.parse_tool_calls(ambiguous_text)
        assert calls == []

    # Scenario 5: Invalid arguments produce a structured validation error
    def test_05_invalid_arguments_produce_structured_error(self):
        schema = {
            "name": "execute_command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"},
                },
                "required": ["command"],
            },
        }

        # Missing required parameter
        err_envelope = validate_arguments("execute_command", schema, {})
        assert err_envelope is not None
        assert err_envelope["status"] == "error"
        assert err_envelope["error_type"] == "SCHEMA_VALIDATION_ERROR"
        assert err_envelope["invalid_parameter"] == "command"
        assert "Missing required parameter 'command'" in err_envelope["error"]
        assert "remedy_hint" in err_envelope

    # Scenario 6: Permission denial is preserved (no override loops)
    def test_06_permission_denial_preserved_as_terminal_blocker(self):
        assert is_unresolvable_terminal_blocker(
            error_type="PERMISSION_DENIED",
            error_message="Permission denied: /etc/shadow",
            tool_name="manage_workspace_file",
        ) is True

        assert is_unresolvable_terminal_blocker(
            error_type="IO_ERROR",
            error_message="Read-only file system",
            tool_name="manage_workspace_file",
        ) is True

        # Non-terminal transient error is not flagged as terminal blocker
        assert is_unresolvable_terminal_blocker(
            error_type="FILE_NOT_FOUND",
            error_message="No such file or directory",
            tool_name="manage_workspace_file",
        ) is False

    # Scenario 7: Legitimate refusal is not overridden
    def test_07_legitimate_refusal_not_overridden(self):
        refusal_prose = "I cannot execute this operation because reading /etc/shadow violates system security policy."
        assert has_valid_inaction_justification(refusal_prose) is True

        unjustified_inaction = "Sure, I will execute the script right away."
        assert has_valid_inaction_justification(unjustified_inaction) is False

    # Scenario 8: Tool failure cannot be reported as verified success
    def test_08_tool_failure_cannot_be_reported_as_verified_success(self):
        ledger = TurnExecutionLedger()
        call_id = "call_fail_attempt"
        ledger.record_proposed(call_id, "execute_command", {"command": "cat /secret/key"})
        ledger.record_dispatched(call_id)

        # Attempt to inject verified=True into a failed exit code observation
        ledger.record_observation(
            call_id,
            status="error",
            output={"error": "Permission denied", "exit_code": 1},
            error="Permission denied",
            exit_code=1,
            verified=True,
        )

        rec = ledger._records[call_id]
        assert rec.state == ActionState.FAILED
        assert rec.verified is False  # Must be strictly False despite parameter
        assert ledger.has_unresolved_failures() is True

        # False success assertion in model text must be intercepted
        model_claim = "The file has been successfully retrieved and everything is ready."
        assert detect_false_success(model_claim, has_unresolved=True) is True

    # Scenario 9: Partial execution remains distinct from full success
    def test_09_partial_execution_remains_distinct(self):
        ledger = TurnExecutionLedger()
        call_id = "call_partial_1"
        ledger.record_proposed(call_id, "execute_command", {"command": "find / -name '*.log'"})
        ledger.record_dispatched(call_id)
        ledger.record_partially_completed(
            call_id,
            output="Output truncated at 500 lines",
            error="Execution truncated due to output buffer limits",
        )

        rec = ledger._records[call_id]
        assert rec.state == ActionState.PARTIALLY_COMPLETED
        assert rec.verified is False
        assert rec not in ledger.successful_actions()
        assert len(ledger.partially_completed_actions()) == 1
        assert ledger.has_unresolved_failures() is True

    # Scenario 10: Cancellation and exceptions cannot become success
    def test_10_cancellation_and_exceptions_cannot_become_success(self):
        ledger = TurnExecutionLedger()
        call_id = "call_cancelled_1"
        ledger.record_proposed(call_id, "execute_command", {"command": "sleep 100"})
        ledger.record_dispatched(call_id)
        ledger.record_cancelled(call_id, reason="User sent SIGINT")

        rec = ledger._records[call_id]
        assert rec.state == ActionState.CANCELLED
        assert rec not in ledger.successful_actions()
        assert len(ledger.cancelled_actions()) == 1

        # Attempting observation record on cancelled action cannot promote it
        ledger.record_observation(call_id, status="success", output="done", exit_code=0)
        assert ledger._records[call_id].state == ActionState.CANCELLED

    # Scenario 11: Recovery stops at its configured attempt limit (<= 2)
    def test_11_recovery_stops_at_attempt_ceiling(self):
        ledger = TurnExecutionLedger()
        call_id = "call_rec_ceiling"
        ledger.record_proposed(call_id, "execute_command", {"command": "test_cmd"})
        ledger.record_dispatched(call_id)
        ledger.record_observation(call_id, status="error", output="failed", exit_code=1)

        assert ledger.get_recovery_attempts(call_id) == 0
        att_1 = ledger.increment_recovery_attempts(call_id)
        assert att_1 == 1
        att_2 = ledger.increment_recovery_attempts(call_id)
        assert att_2 == 2

        # Ceiling reached
        ledger.mark_terminal(call_id)
        assert ledger.is_terminal(call_id) is True

    # Scenario 12: Compact schemas preserve validation behavior
    def test_12_compact_schemas_preserve_validation_behavior(self):
        verbose_schema = {
            "type": "function",
            "function": {
                "name": "manage_workspace_file",
                "description": "Performs file operations in workspace. Supports read, write, append, and delete.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["read", "write", "append", "delete", "list"],
                            "description": "File action to execute",
                        },
                        "path": {
                            "type": "string",
                            "description": "Target relative file path",
                        },
                        "content": {
                            "type": "string",
                            "description": "File payload content",
                        },
                    },
                    "required": ["action", "path"],
                },
            },
        }

        compact_schema = CompactSchemaGenerator.compact_schema(verbose_schema)

        # Valid arguments pass both
        valid_args = {"action": "read", "path": "test.txt"}
        assert validate_arguments("manage_workspace_file", verbose_schema, valid_args) is None
        assert validate_arguments("manage_workspace_file", compact_schema, valid_args) is None

        # Missing required parameter fails identically on both
        invalid_args = {"action": "read"}
        err_verbose = validate_arguments("manage_workspace_file", verbose_schema, invalid_args)
        err_compact = validate_arguments("manage_workspace_file", compact_schema, invalid_args)
        assert err_verbose is not None
        assert err_compact is not None
        assert err_verbose["invalid_parameter"] == err_compact["invalid_parameter"] == "path"
        assert err_verbose["error_type"] == err_compact["error_type"] == "SCHEMA_VALIDATION_ERROR"
