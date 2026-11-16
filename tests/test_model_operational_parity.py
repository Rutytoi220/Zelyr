"""Deterministic Cross-Model Operational Parity Regression Suite.

Verifies the 8 core operational parity invariants using deterministic mock providers:
1. Single file creation with exact independent verification.
2. Two sequential file creations using goal-aware replanning.
3. Failed tool call followed by valid recovery action.
4. Text-only completion claim with missing required artifact.
5. Mutation attempt against an already-verified target.
6. Multiple acceptance criteria with mixed outcomes.
7. Model response containing multiple tool proposals (one-tool-per-turn rule).
8. Malformed or non-native tool-call response via fallback parser.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.cli.main import _execute_headless_run
from zelyr.engine.execution_ledger import ActionState, TurnExecutionLedger
from zelyr.engine.verifier import OutcomeVerifier


class TestCrossModelOperationalParity:
    """Rigorous deterministic parity test suite across simulated model behaviors."""

    @pytest.mark.asyncio
    async def test_1_single_file_creation_exact_verification(self, tmp_path: Path):
        """Scenario 1: Single file creation with exact independent verification."""
        f_target = tmp_path / "parity_single.txt"
        expected_content = "PARITY_SECRET_42"

        executed_tools: List[Dict[str, Any]] = []

        async def mock_executor(name: str, **kwargs) -> Dict[str, Any]:
            executed_tools.append({"tool": name, "args": kwargs})
            p = Path(kwargs.get("path", ""))
            p.write_text(kwargs.get("content", ""), encoding="utf-8")
            return {"status": "success", "path": str(p)}

        async def mock_stream(payload: dict):
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_single_1",
                            "type": "function",
                            "function": {
                                "name": "manage_workspace_file",
                                "arguments": json.dumps({
                                    "action": "write",
                                    "path": str(f_target),
                                    "content": expected_content,
                                }),
                            },
                        }]
                    }
                }]
            }

        orch = NativeOrchestrator()
        ledger = TurnExecutionLedger()
        verify_spec = f"file_contains:{f_target}:{expected_content}"

        payload = {
            "messages": [{"role": "user", "content": f"Create {f_target} containing '{expected_content}'"}],
            "acceptance_criteria": verify_spec,
        }

        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=mock_executor,
            ledger=ledger,
        ):
            pass

        # 1. Tool execution occurred
        assert len(executed_tools) == 1
        assert executed_tools[0]["tool"] == "manage_workspace_file"
        assert executed_tools[0]["args"]["path"] == str(f_target)

        # 2. File written on disk with exact content
        assert f_target.exists()
        assert f_target.read_text(encoding="utf-8") == expected_content

        # 3. Independent verification succeeds
        v_res = OutcomeVerifier.verify("file_contains", str(f_target), expected=expected_content)
        assert v_res.success is True
        assert ledger.is_target_verified(str(f_target)) is True

    @pytest.mark.asyncio
    async def test_2_two_sequential_files_goal_aware_replanning(self, tmp_path: Path):
        """Scenario 2: Two sequential file creations using goal-aware replanning."""
        f_a = tmp_path / "parity_alpha.txt"
        f_b = tmp_path / "parity_beta.txt"

        executed_tools: List[str] = []

        async def mock_executor(name: str, **kwargs) -> Dict[str, Any]:
            p = Path(kwargs.get("path", ""))
            executed_tools.append(str(p.name))
            p.write_text(kwargs.get("content", ""), encoding="utf-8")
            return {"status": "success", "path": str(p)}

        turn = 0

        async def mock_stream(payload: dict):
            nonlocal turn
            turn += 1
            if turn == 1:
                # Turn 1: Proposes file A and file B
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "prop_a",
                                    "type": "function",
                                    "function": {
                                        "name": "manage_workspace_file",
                                        "arguments": json.dumps({"action": "write", "path": str(f_a), "content": "ALPHA"}),
                                    },
                                },
                                {
                                    "index": 1,
                                    "id": "prop_b_old",
                                    "type": "function",
                                    "function": {
                                        "name": "manage_workspace_file",
                                        "arguments": json.dumps({"action": "write", "path": str(f_b), "content": "BETA"}),
                                    },
                                },
                            ]
                        }
                    }]
                }
            elif turn == 2:
                # Turn 2: Premature text-only completion claim
                yield {"choices": [{"delta": {"content": "File A created successfully."}}]}
            elif turn == 3:
                # Turn 3: Goal-aware replanning directive received; model emits fresh call for B
                last_msg = payload["messages"][-1]["content"]
                assert "[SYSTEM DIRECTIVE: Goal-Aware Replanning]" in last_msg
                assert str(f_b) in last_msg
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "prop_b_fresh",
                                    "type": "function",
                                    "function": {
                                        "name": "manage_workspace_file",
                                        "arguments": json.dumps({"action": "write", "path": str(f_b), "content": "BETA"}),
                                    },
                                }
                            ]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Both files created."}}]}

        orch = NativeOrchestrator()
        ledger = TurnExecutionLedger()
        verify_spec = f"file_contains:{f_a}:ALPHA,file_contains:{f_b}:BETA"

        payload = {
            "messages": [{"role": "user", "content": f"Create {f_a} with ALPHA and create {f_b} with BETA"}],
            "acceptance_criteria": verify_spec,
        }

        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=mock_executor,
            ledger=ledger,
        ):
            pass

        assert executed_tools == ["parity_alpha.txt", "parity_beta.txt"]
        assert f_a.read_text(encoding="utf-8") == "ALPHA"
        assert f_b.read_text(encoding="utf-8") == "BETA"
        assert ledger._records["prop_b_old"].state == ActionState.CANCELLED
        assert ledger._records["prop_b_fresh"].state in (ActionState.SUCCESS, ActionState.VERIFIED)

    @pytest.mark.asyncio
    async def test_3_failed_tool_call_followed_by_valid_recovery(self):
        """Scenario 3: Failed tool call followed by valid recovery action."""
        executed_commands: List[str] = []

        async def mock_executor(name: str, **kwargs) -> Dict[str, Any]:
            cmd = kwargs.get("command", "")
            executed_commands.append(cmd)
            if cmd == "bad_command":
                return {"status": "error", "error": "command not found: bad_command", "exit_code": 127}
            return {"status": "success", "output": "recovered successfully", "exit_code": 0}

        turn = 0

        async def mock_stream(payload: dict):
            nonlocal turn
            turn += 1
            if turn == 1:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_fail",
                                "type": "function",
                                "function": {"name": "execute_command", "arguments": json.dumps({"command": "bad_command"})},
                            }]
                        }
                    }]
                }
            elif turn == 2:
                # Turn 2: Receives error observation and emits recovery command
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_recover",
                                "type": "function",
                                "function": {"name": "execute_command", "arguments": json.dumps({"command": "echo recovered"})},
                            }]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Recovery complete."}}]}

        orch = NativeOrchestrator()
        ledger = TurnExecutionLedger()
        payload = {"messages": [{"role": "user", "content": "Run tool with recovery"}]}

        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=mock_executor,
            ledger=ledger,
        ):
            pass

        assert executed_commands == ["bad_command", "echo recovered"]
        assert ledger._records["call_fail"].state == ActionState.FAILED
        assert ledger._records["call_recover"].state in (ActionState.SUCCESS, ActionState.VERIFIED)

    @pytest.mark.asyncio
    async def test_4_text_only_completion_claim_with_missing_artifact(self, tmp_path: Path):
        """Scenario 4: Text-only completion claim with missing required artifact terminates incomplete."""
        f_missing = tmp_path / "missing_artifact.txt"

        turn = 0

        async def mock_stream(payload: dict):
            nonlocal turn
            turn += 1
            # Model responds with text only, claiming it created the file without calling any tool
            yield {"choices": [{"delta": {"content": f"I have successfully created {f_missing}."}}]}

        orch = NativeOrchestrator()
        ledger = TurnExecutionLedger()
        verify_spec = f"file_exists:{f_missing}"

        payload = {
            "messages": [{"role": "user", "content": f"Create {f_missing}"}],
            "acceptance_criteria": verify_spec,
        }

        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            ledger=ledger,
        ):
            pass

        # File was never created
        assert not f_missing.exists()
        # Verifier confirms failure
        res = OutcomeVerifier.verify("file_exists", str(f_missing))
        assert res.success is False
        assert ledger.is_target_verified(str(f_missing)) is False

    @pytest.mark.asyncio
    async def test_5_mutation_attempt_against_already_verified_target(self, tmp_path: Path):
        """Scenario 5: Mutation attempt against an already-verified target is rejected."""
        f_target = tmp_path / "protected.txt"
        f_target.write_text("EMPTY_OR_OLD", encoding="utf-8")

        executed_writes: List[str] = []

        async def mock_executor(name: str, **kwargs) -> Dict[str, Any]:
            p = Path(kwargs.get("path", ""))
            content = kwargs.get("content", "")
            executed_writes.append(content)
            p.write_text(content, encoding="utf-8")
            return {"status": "success", "path": str(p)}

        turn = 0

        async def mock_stream(payload: dict):
            nonlocal turn
            turn += 1
            if turn == 1:
                # Turn 1: Valid initial write
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_init",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": json.dumps({"action": "write", "path": str(f_target), "content": "INITIAL_SAFE"}),
                                },
                            }]
                        }
                    }]
                }
            elif turn == 2:
                # Turn 2: Model attempts rogue mutation on verified file
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_rogue",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": json.dumps({"action": "write", "path": str(f_target), "content": "ROGUE_OVERWRITE"}),
                                },
                            }]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Finished."}}]}

        orch = NativeOrchestrator()
        ledger = TurnExecutionLedger()
        verify_spec = f"file_contains:{f_target}:INITIAL_SAFE"

        payload = {
            "messages": [{"role": "user", "content": f"Write INITIAL_SAFE to {f_target}"}],
            "acceptance_criteria": verify_spec,
        }

        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=mock_executor,
            ledger=ledger,
        ):
            pass

        # Target file content was preserved
        assert f_target.read_text(encoding="utf-8") == "INITIAL_SAFE"
        # Rogue call was rejected by mutation guard
        assert "call_rogue" in ledger._records
        assert ledger._records["call_rogue"].state in (ActionState.REJECTED, ActionState.FAILED)
        assert "already satisfies the verified acceptance criteria" in str(ledger._records["call_rogue"].error)

    @pytest.mark.asyncio
    async def test_6_multiple_acceptance_criteria_mixed_outcomes(self, tmp_path: Path, monkeypatch, capsys):
        """Scenario 6: Multiple acceptance criteria with mixed outcomes exits non-zero (exit 2)."""
        f_pass = tmp_path / "pass.txt"
        f_pass.write_text("VALID_DATA", encoding="utf-8")
        f_fail = tmp_path / "fail.txt"
        f_fail.write_text("WRONG_DATA", encoding="utf-8")

        async def mock_stream(payload: dict):
            yield {"choices": [{"delta": {"content": "Completed tasks."}}]}

        orch = NativeOrchestrator()
        monkeypatch.setattr(orch, "generate_stream", mock_stream)

        verify_spec = f"file_contains:{f_pass}:VALID_DATA,file_contains:{f_fail}:EXPECTED_CORRECT"

        exit_code = await _execute_headless_run(
            prompt="Do task",
            orchestrator=orch,
            verify_spec=verify_spec,
        )
        assert exit_code == 2
        captured = capsys.readouterr()
        assert f"[verified] file_contains on {f_pass}: PASS" in captured.out
        assert f"[verification failed] file_contains on {f_fail}: FAIL" in captured.out

    @pytest.mark.asyncio
    async def test_7_multiple_tool_proposals_preserves_one_tool_per_turn(self):
        """Scenario 7: Multiple proposals in one turn executes strictly one tool."""
        executed_tools: List[str] = []

        async def mock_executor(name: str, **kwargs) -> Dict[str, Any]:
            executed_tools.append(kwargs.get("command", ""))
            return {"status": "success", "exit_code": 0}

        turn = 0

        async def mock_stream(payload: dict):
            nonlocal turn
            turn += 1
            if turn == 1:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "tc_1",
                                    "type": "function",
                                    "function": {"name": "execute_command", "arguments": json.dumps({"command": "echo 1"})},
                                },
                                {
                                    "index": 1,
                                    "id": "tc_2",
                                    "type": "function",
                                    "function": {"name": "execute_command", "arguments": json.dumps({"command": "echo 2"})},
                                },
                                {
                                    "index": 2,
                                    "id": "tc_3",
                                    "type": "function",
                                    "function": {"name": "execute_command", "arguments": json.dumps({"command": "echo 3"})},
                                },
                            ]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Done."}}]}

        orch = NativeOrchestrator()
        ledger = TurnExecutionLedger()
        payload = {"messages": [{"role": "user", "content": "Run 1, 2, and 3"}]}

        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=mock_executor,
            ledger=ledger,
        ):
            pass

        # Exactly 1 tool was executed in Turn 1
        assert executed_tools == ["echo 1"]
        assert "tc_1" in ledger._records
        assert ledger._records["tc_1"].state in (ActionState.SUCCESS, ActionState.VERIFIED)
        assert ledger._records["tc_2"].state == ActionState.CANCELLED
        assert ledger._records["tc_3"].state == ActionState.CANCELLED

    @pytest.mark.asyncio
    async def test_8_degraded_markdown_tool_call_ingestion_and_fallback(self, tmp_path: Path):
        """Scenario 8: Degraded markdown tool call is parsed via FallbackToolParser and executed."""
        f_target = tmp_path / "fallback_target.txt"

        executed_tools: List[str] = []

        async def mock_executor(name: str, **kwargs) -> Dict[str, Any]:
            executed_tools.append(name)
            p = Path(kwargs.get("path", ""))
            p.write_text(kwargs.get("content", ""), encoding="utf-8")
            return {"status": "success", "path": str(p)}

        turn = 0

        async def mock_stream(payload: dict):
            nonlocal turn
            turn += 1
            if turn == 1:
                # Degraded 3B model returns markdown fenced tool call instead of native tool_calls
                content = (
                    "I will create the file now.\n```json\n"
                    f'{{"tool": "manage_workspace_file", "action": "write", "path": "{f_target}", "content": "FALLBACK_SUCCESS"}}\n'
                    "```"
                )
                yield {"choices": [{"delta": {"content": content}}]}
            else:
                yield {"choices": [{"delta": {"content": "File created via fallback."}}]}

        orch = NativeOrchestrator()
        ledger = TurnExecutionLedger()
        verify_spec = f"file_contains:{f_target}:FALLBACK_SUCCESS"

        payload = {
            "messages": [{"role": "user", "content": f"Create {f_target}"}],
            "acceptance_criteria": verify_spec,
        }

        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=mock_executor,
            ledger=ledger,
        ):
            pass

        assert len(executed_tools) == 1
        assert executed_tools[0] == "manage_workspace_file"
        assert f_target.exists()
        assert f_target.read_text(encoding="utf-8") == "FALLBACK_SUCCESS"
        assert ledger.is_target_verified(str(f_target)) is True
