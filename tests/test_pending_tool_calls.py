"""Adversarial regression test suite for pending tool proposal authorization semantics.

Verifies the 10 core authorization requirements:
1. Two calls proposed together, followed by a text-only response: the second call must NOT execute.
2. Two calls proposed together, followed by explicit re-emission of the second: it may execute after fresh validation.
3. A pending mutation followed by a text response claiming completion: no mutation occurs.
4. A pending mutation followed by explicit abandonment: it is discarded with abandoned_by_model.
5. A pending call followed by a different call: the old proposal is superseded and cannot execute later.
6. A pending call whose target becomes verified: it is rejected as redundant (outcome_already_verified).
7. A pending call after a failed preceding operation: reconsidered only through an explicit new proposal.
8. Cancellation and step-limit termination: no pending call survives.
9. A malformed or ambiguous response: no pending mutation executes (fail-closed).
10. Invariant: every execution strictly respects the one-tool-per-turn limit.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.engine.execution_ledger import ActionState, TurnExecutionLedger


class TestPendingProposalAuthorizationAudit:
    """Rigorous audit test battery verifying that pending proposals require explicit re-confirmation."""

    @pytest.mark.asyncio
    async def test_1_two_calls_proposed_together_followed_by_text_only_does_not_execute_second(self):
        """1. Two calls proposed together, followed by a text-only response: second call must NOT execute."""
        orch = NativeOrchestrator()
        executed_turns: List[Dict[str, Any]] = []

        async def mock_executor(name, **kwargs):
            executed_turns.append({"tool": name, "arguments": kwargs})
            return {"status": "success", "result": f"Executed {name}"}

        async def mock_stream(payload):
            msgs = payload.get("messages", [])
            tool_msgs = [m for m in msgs if m.get("role") == "tool"]
            if not tool_msgs:
                # Turn 1: Model emits two native tool calls
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c1",
                                    "type": "function",
                                    "function": {"name": "execute_command", "arguments": json.dumps({"command": "echo step1"})},
                                },
                                {
                                    "index": 1,
                                    "id": "c2",
                                    "type": "function",
                                    "function": {"name": "execute_command", "arguments": json.dumps({"command": "echo step2"})},
                                },
                            ]
                        }
                    }]
                }
            else:
                # Turn 2: Model receives feedback for c1, but produces text ONLY without tool calls
                yield {"choices": [{"delta": {"content": "I have executed step 1. Waiting for user input."}}]}

        ledger = TurnExecutionLedger()
        payload = {"messages": [{"role": "user", "content": "Execute step 1 and step 2"}]}

        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=mock_executor,
            ledger=ledger,
        ):
            pass

        # Exactly 1 call executed (c1). c2 was NOT executed merely because model emitted text!
        assert len(executed_turns) == 1
        assert executed_turns[0]["arguments"] == {"command": "echo step1"}

        # c2 was recorded as proposed, then unconfirmed and discarded
        assert "c2" in ledger._records
        rec_c2 = ledger._records["c2"]
        assert rec_c2.state == ActionState.CANCELLED
        assert "unconfirmed_by_model" in str(rec_c2.error)
        assert len(ledger.get_pending_proposals()) == 0

    @pytest.mark.asyncio
    async def test_2_two_calls_proposed_together_followed_by_explicit_reemission_executes(self):
        """2. Two calls proposed together, followed by explicit re-emission of second: executes after validation."""
        orch = NativeOrchestrator()
        executed_turns: List[Dict[str, Any]] = []

        async def mock_executor(name, **kwargs):
            executed_turns.append({"tool": name, "arguments": kwargs})
            return {"status": "success", "result": f"Executed {name}"}

        async def mock_stream(payload):
            msgs = payload.get("messages", [])
            tool_msgs = [m for m in msgs if m.get("role") == "tool"]
            if not tool_msgs:
                # Turn 1: Model proposes two calls
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c1",
                                    "type": "function",
                                    "function": {"name": "execute_command", "arguments": json.dumps({"command": "echo step1"})},
                                },
                                {
                                    "index": 1,
                                    "id": "c2",
                                    "type": "function",
                                    "function": {"name": "execute_command", "arguments": json.dumps({"command": "echo step2"})},
                                },
                            ]
                        }
                    }]
                }
            elif len(tool_msgs) == 1:
                # Turn 2: Model receives feedback for c1, explicitly re-emits c2
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c2_reconfirmed",
                                    "type": "function",
                                    "function": {"name": "execute_command", "arguments": json.dumps({"command": "echo step2"})},
                                }
                            ]
                        }
                    }]
                }
            else:
                # Turn 3: Final synthesis
                yield {"choices": [{"delta": {"content": "Both steps completed."}}]}

        ledger = TurnExecutionLedger()
        payload = {"messages": [{"role": "user", "content": "Execute steps"}]}

        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=mock_executor,
            ledger=ledger,
        ):
            pass

        assert len(executed_turns) == 2
        assert executed_turns[0]["arguments"] == {"command": "echo step1"}
        assert executed_turns[1]["arguments"] == {"command": "echo step2"}
        assert len(ledger.get_pending_proposals()) == 0

    @pytest.mark.asyncio
    async def test_3_pending_mutation_followed_by_text_claiming_completion_no_mutation(self):
        """3. A pending mutation followed by a text response claiming completion: no mutation occurs."""
        orch = NativeOrchestrator()
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            file_a = ws / "file_a.txt"
            file_b = ws / "file_b.txt"

            executed_ops = []

            async def mock_executor(name, **kwargs):
                executed_ops.append((name, kwargs))
                path = ws / kwargs.get("path", "")
                path.write_text(kwargs.get("content", ""), encoding="utf-8")
                return {"status": "success", "path": str(path)}

            async def mock_stream(payload):
                msgs = payload.get("messages", [])
                tool_msgs = [m for m in msgs if m.get("role") == "tool"]
                if not tool_msgs:
                    # Turn 1: Emits mutation for file_a and file_b
                    yield {
                        "choices": [{
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "mut_a",
                                        "type": "function",
                                        "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": "file_a.txt", "content": "A_CONTENT"})},
                                    },
                                    {
                                        "index": 1,
                                        "id": "mut_b",
                                        "type": "function",
                                        "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": "file_b.txt", "content": "B_CONTENT"})},
                                    },
                                ]
                            }
                        }]
                    }
                else:
                    # Turn 2: Model falsely claims both files are created without calling a tool
                    yield {
                        "choices": [{
                            "delta": {
                                "content": "I have created both file_a.txt and file_b.txt. Everything is done!"
                            }
                        }]
                    }

            ledger = TurnExecutionLedger()
            payload = {"messages": [{"role": "user", "content": "Create file_a and file_b"}]}

            async for _ in orch.generate_stream(
                payload,
                stream_provider=mock_stream,
                tool_executor=mock_executor,
                ledger=ledger,
            ):
                pass

            # file_a was written in turn 1
            assert file_a.exists()
            assert file_a.read_text(encoding="utf-8") == "A_CONTENT"

            # file_b MUST NOT EXIST on disk! Unconfirmed proposal did not execute!
            assert not file_b.exists()
            assert len(executed_ops) == 1

            # Ledger records mut_b as unconfirmed and cancelled
            rec_b = ledger._records["mut_b"]
            assert rec_b.state == ActionState.CANCELLED
            assert "unconfirmed_by_model" in str(rec_b.error)

    @pytest.mark.asyncio
    async def test_4_pending_mutation_followed_by_explicit_abandonment_discarded(self):
        """4. A pending mutation followed by explicit abandonment: discarded with abandoned_by_model."""
        orch = NativeOrchestrator()
        executed_calls = []

        async def mock_executor(name, **kwargs):
            executed_calls.append(name)
            return {"status": "success"}

        async def mock_stream(payload):
            msgs = payload.get("messages", [])
            tool_msgs = [m for m in msgs if m.get("role") == "tool"]
            if not tool_msgs:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "c1", "type": "function", "function": {"name": "execute_command", "arguments": json.dumps({"command": "echo 1"})}},
                                {"index": 1, "id": "c2", "type": "function", "function": {"name": "execute_command", "arguments": json.dumps({"command": "echo 2"})}},
                            ]
                        }
                    }]
                }
            else:
                yield {
                    "choices": [{
                        "delta": {
                            "content": "I will stop here. Aborting remaining actions due to safety policy."
                        }
                    }]
                }

        ledger = TurnExecutionLedger()
        payload = {"messages": [{"role": "user", "content": "Run commands"}]}

        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=mock_executor,
            ledger=ledger,
        ):
            pass

        assert len(executed_calls) == 1
        rec2 = ledger._records["c2"]
        assert rec2.state == ActionState.CANCELLED
        assert "abandoned_by_model" in str(rec2.error)

    @pytest.mark.asyncio
    async def test_5_pending_call_followed_by_different_call_is_superseded(self):
        """5. A pending call followed by a different call: old proposal is superseded and cannot execute later."""
        orch = NativeOrchestrator()
        executed_calls = []

        async def mock_executor(name, **kwargs):
            executed_calls.append((name, kwargs))
            return {"status": "success"}

        async def mock_stream(payload):
            msgs = payload.get("messages", [])
            tool_msgs = [m for m in msgs if m.get("role") == "tool"]
            if not tool_msgs:
                # Turn 1: Propose tool_A and tool_B
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "c_A", "type": "function", "function": {"name": "execute_command", "arguments": json.dumps({"command": "op_A"})}},
                                {"index": 1, "id": "c_B", "type": "function", "function": {"name": "execute_command", "arguments": json.dumps({"command": "op_B"})}},
                            ]
                        }
                    }]
                }
            elif len(tool_msgs) == 1:
                # Turn 2: Model receives feedback for op_A, but emits op_C instead of op_B
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "c_C", "type": "function", "function": {"name": "execute_command", "arguments": json.dumps({"command": "op_C"})}},
                            ]
                        }
                    }]
                }
            else:
                # Turn 3: Model finishes with text
                yield {"choices": [{"delta": {"content": "Completed with alternative path."}}]}

        ledger = TurnExecutionLedger()
        payload = {"messages": [{"role": "user", "content": "Execute workflow"}]}

        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=mock_executor,
            ledger=ledger,
        ):
            pass

        # op_A and op_C executed; op_B was never executed
        assert len(executed_calls) == 2
        assert executed_calls[0][1]["command"] == "op_A"
        assert executed_calls[1][1]["command"] == "op_C"

        rec_B = ledger._records["c_B"]
        assert rec_B.state == ActionState.CANCELLED
        assert "superseded_by_model" in str(rec_B.error)

    @pytest.mark.asyncio
    async def test_6_pending_call_whose_target_becomes_verified_rejected_as_redundant(self):
        """6. A pending call whose target becomes verified: rejected as redundant (outcome_already_verified)."""
        orch = NativeOrchestrator()
        executed_calls = []

        async def mock_stream(payload):
            msgs = payload.get("messages", [])
            tool_msgs = [m for m in msgs if m.get("role") == "tool"]
            if not tool_msgs:
                # Turn 1: Propose initial write and redundant overwrite
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "write_initial",
                                    "type": "function",
                                    "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": "state.txt", "content": "INITIAL_OK"})},
                                },
                                {
                                    "index": 1,
                                    "id": "write_overwrite",
                                    "type": "function",
                                    "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": "state.txt", "content": "OVERWRITTEN"})},
                                },
                            ]
                        }
                    }]
                }
            else:
                # Turn 2: Model re-emits write_overwrite
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "write_overwrite_re",
                                    "type": "function",
                                    "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": "state.txt", "content": "OVERWRITTEN"})},
                                }
                            ]
                        }
                    }]
                }

        ledger = TurnExecutionLedger()
        payload = {
            "messages": [{"role": "user", "content": "Create file state.txt with INITIAL_OK"}],
            "acceptance_criteria": "file_contains:state.txt:INITIAL_OK",
        }

        with patch("zelyr.engine.verifier.OutcomeVerifier.verify") as mock_v:
            from zelyr.engine.verifier import VerificationResult
            verified_state = [False]

            def verifier_side_effect(*args, **kwargs):
                if not verified_state[0]:
                    return VerificationResult(success=False, check="file_contains", target="state.txt", observed=None, error="not yet created")
                return VerificationResult(success=True, check="file_contains", target="state.txt", observed="INITIAL_OK")

            mock_v.side_effect = verifier_side_effect

            async def mock_executor_stateful(name, **kwargs):
                executed_calls.append(kwargs)
                verified_state[0] = True
                return {"status": "success", "verified": True, "detail": "target state written"}

            async for _ in orch.generate_stream(
                payload,
                stream_provider=mock_stream,
                tool_executor=mock_executor_stateful,
                ledger=ledger,
            ):
                pass

        # Only initial write executed on host
        assert len(executed_calls) == 1
        assert executed_calls[0]["content"] == "INITIAL_OK"

        # Overwrite was rejected under outcome_already_verified
        rec_re = ledger._records.get("write_overwrite_re") or ledger._records.get("write_overwrite")
        assert rec_re is not None
        assert rec_re.state in (ActionState.FAILED, ActionState.CANCELLED)
        assert "outcome_already_verified" in str(rec_re.error) or "already satisfies" in str(rec_re.error)

    @pytest.mark.asyncio
    async def test_7_failed_first_call_requires_explicit_new_proposal(self):
        """7. A failed first call: pending proposals are not executed on text, require explicit reconsideration."""
        orch = NativeOrchestrator()
        executed_calls = []

        async def mock_executor(name, **kwargs):
            executed_calls.append(kwargs)
            if kwargs.get("command") == "failing_cmd":
                return {"status": "error", "error": "Disk permission denied", "exit_code": 1}
            return {"status": "success"}

        async def mock_stream(payload):
            msgs = payload.get("messages", [])
            tool_msgs = [m for m in msgs if m.get("role") == "tool"]
            if not tool_msgs:
                # Turn 1: Model proposes failing command and second command
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "c1", "type": "function", "function": {"name": "execute_command", "arguments": json.dumps({"command": "failing_cmd"})}},
                                {"index": 1, "id": "c2", "type": "function", "function": {"name": "execute_command", "arguments": json.dumps({"command": "echo step2"})}},
                            ]
                        }
                    }]
                }
            else:
                # Turn 2: Preceding call failed. Model responds with explanatory text without tools.
                yield {
                    "choices": [{
                        "delta": {
                            "content": "The command failed due to disk permissions. Please grant write access."
                        }
                    }]
                }

        ledger = TurnExecutionLedger()
        payload = {"messages": [{"role": "user", "content": "Execute commands"}]}

        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=mock_executor,
            ledger=ledger,
        ):
            pass

        # Only failing_cmd was executed. c2 did NOT execute after the failure!
        assert len(executed_calls) == 1
        assert executed_calls[0]["command"] == "failing_cmd"

        rec_c2 = ledger._records["c2"]
        assert rec_c2.state == ActionState.CANCELLED
        assert "unconfirmed_by_model" in str(rec_c2.error)

    @pytest.mark.asyncio
    async def test_8_cancellation_and_step_limit_clear_all_pending_calls(self):
        """8. Cancellation and step-limit termination: no pending call survives in ledger."""
        # A: Task cancellation
        orch_a = NativeOrchestrator()
        executed_a = []

        async def mock_exec_a(name, **kwargs):
            executed_a.append(name)
            orch_a.cancel(reason="test_cancellation")
            return {"status": "success"}

        async def mock_stream_a(payload):
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "ca1", "type": "function", "function": {"name": "execute_command", "arguments": json.dumps({"command": "echo 1"})}},
                            {"index": 1, "id": "ca2", "type": "function", "function": {"name": "execute_command", "arguments": json.dumps({"command": "echo 2"})}},
                        ]
                    }
                }]
            }

        ledger_a = TurnExecutionLedger()
        payload_a = {"messages": [{"role": "user", "content": "Run"}]}
        async for _ in orch_a.generate_stream(payload_a, stream_provider=mock_stream_a, tool_executor=mock_exec_a, ledger=ledger_a):
            pass

        assert len(executed_a) == 1
        assert ledger_a._records["ca2"].state == ActionState.CANCELLED
        assert len(ledger_a.get_pending_proposals()) == 0

        # B: Step limit termination
        orch_b = NativeOrchestrator()
        orch_b.max_steps = 1
        executed_b = []

        async def mock_exec_b(name, **kwargs):
            executed_b.append(name)
            return {"status": "success"}

        async def mock_stream_b(payload):
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "cb1", "type": "function", "function": {"name": "execute_command", "arguments": json.dumps({"command": "echo 1"})}},
                            {"index": 1, "id": "cb2", "type": "function", "function": {"name": "execute_command", "arguments": json.dumps({"command": "echo 2"})}},
                        ]
                    }
                }]
            }

        ledger_b = TurnExecutionLedger()
        payload_b = {"messages": [{"role": "user", "content": "Run"}]}
        async for _ in orch_b.generate_stream(payload_b, stream_provider=mock_stream_b, tool_executor=mock_exec_b, ledger=ledger_b, max_depth=5):
            pass

        assert len(executed_b) == 1
        assert ledger_b._records["cb2"].state == ActionState.CANCELLED
        assert "step_limit_reached" in str(ledger_b._records["cb2"].error)
        assert len(ledger_b.get_pending_proposals()) == 0

    @pytest.mark.asyncio
    async def test_9_malformed_or_ambiguous_response_fails_closed_no_pending_mutation(self):
        """9. A malformed or ambiguous response: no pending mutation executes (fail-closed)."""
        orch = NativeOrchestrator()
        executed_calls = []

        async def mock_executor(name, **kwargs):
            executed_calls.append(name)
            return {"status": "success"}

        async def mock_stream(payload):
            msgs = payload.get("messages", [])
            tool_msgs = [m for m in msgs if m.get("role") == "tool"]
            if not tool_msgs:
                # Turn 1: Propose two calls
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "c1", "type": "function", "function": {"name": "execute_command", "arguments": json.dumps({"command": "echo 1"})}},
                                {"index": 1, "id": "c2_mutation", "type": "function", "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": "danger.txt", "content": "bad"})}},
                            ]
                        }
                    }]
                }
            else:
                # Turn 2: Ambiguous truncated/gibberish response with no tool calls
                yield {
                    "choices": [{
                        "delta": {
                            "content": "Hmm, maybe... <corrupted token stream"
                        }
                    }]
                }

        ledger = TurnExecutionLedger()
        payload = {"messages": [{"role": "user", "content": "Run safely"}]}

        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=mock_executor,
            ledger=ledger,
        ):
            pass

        # Only c1 executed; c2_mutation was NEVER executed on ambiguous output
        assert len(executed_calls) == 1
        assert executed_calls[0] == "execute_command"

        rec2 = ledger._records["c2_mutation"]
        assert rec2.state == ActionState.CANCELLED
        assert "unconfirmed_by_model" in str(rec2.error)

    @pytest.mark.asyncio
    async def test_10_invariant_every_execution_strictly_respects_one_tool_per_turn_limit(self):
        """10. Invariant: every execution strictly respects the one-tool-per-turn limit."""
        orch = NativeOrchestrator()
        turns_execution_counts: Dict[int, int] = {}
        turn_counter = [0]

        async def mock_executor(name, **kwargs):
            t = turn_counter[0]
            turns_execution_counts[t] = turns_execution_counts.get(t, 0) + 1
            return {"status": "success", "result": f"Executed {kwargs.get('command')}"}

        async def mock_stream(payload):
            msgs = payload.get("messages", [])
            tool_msgs = [m for m in msgs if m.get("role") == "tool"]
            turn_counter[0] = len(tool_msgs)

            if not tool_msgs:
                # Turn 1: Propose 3 tools
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "t1", "type": "function", "function": {"name": "execute_command", "arguments": json.dumps({"command": "cmd1"})}},
                                {"index": 1, "id": "t2", "type": "function", "function": {"name": "execute_command", "arguments": json.dumps({"command": "cmd2"})}},
                                {"index": 2, "id": "t3", "type": "function", "function": {"name": "execute_command", "arguments": json.dumps({"command": "cmd3"})}},
                            ]
                        }
                    }]
                }
            elif len(tool_msgs) == 1:
                # Turn 2: Explicitly re-emit t2
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "t2_re", "type": "function", "function": {"name": "execute_command", "arguments": json.dumps({"command": "cmd2"})}},
                            ]
                        }
                    }]
                }
            elif len(tool_msgs) == 2:
                # Turn 3: Explicitly re-emit t3
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "t3_re", "type": "function", "function": {"name": "execute_command", "arguments": json.dumps({"command": "cmd3"})}},
                            ]
                        }
                    }]
                }
            else:
                # Turn 4: Final synthesis
                yield {"choices": [{"delta": {"content": "All 3 completed."}}]}

        ledger = TurnExecutionLedger()
        payload = {"messages": [{"role": "user", "content": "Run 3 commands"}]}

        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=mock_executor,
            ledger=ledger,
        ):
            pass

        # 3 tools executed across 3 distinct turns
        assert len(turns_execution_counts) == 3
        for turn_idx, count in turns_execution_counts.items():
            assert count == 1, f"Turn {turn_idx} executed {count} tools; expected exactly 1!"
