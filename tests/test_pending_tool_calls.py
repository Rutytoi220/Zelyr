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


class TestPendingActionContinuity:
    """Offline end-to-end suite verifying durable pending proposal persistence and resume continuity."""

    def test_pending_action_lifecycle_state_machine_and_tamper_detection(self, tmp_path: Path):
        from zelyr.engine import (
            CentralExecutionLedger,
            CheckpointManager,
            CorruptedCheckpointError,
            CorruptedProposalError,
            ExecutionStatus,
            InvalidProposalTransitionError,
            PendingActionProposal,
            ProposalStatus,
        )

        # 1. Explicit transition enforcement & duplicate-dispatch rejection
        prop = PendingActionProposal(
            proposal_id="prop_1",
            execution_id="exec_1",
            original_call_id="c1",
            tool_id="execute_command",
            normalized_arguments={"command": "echo hi"},
        )
        assert prop.can_transition_to(ProposalStatus.VALIDATED)
        prop.transition(ProposalStatus.VALIDATED)
        assert prop.status == ProposalStatus.VALIDATED.value

        prop.transition(ProposalStatus.DISPATCHED)
        assert prop.dispatch_attempted is True
        assert not prop.can_transition_to(ProposalStatus.DISPATCHED)
        with pytest.raises(InvalidProposalTransitionError):
            prop.transition(ProposalStatus.DISPATCHED)
        prop.transition(ProposalStatus.COMPLETED)
        assert prop.has_result is True
        with pytest.raises(InvalidProposalTransitionError):
            prop.transition(ProposalStatus.PENDING)

        for term_st in (ProposalStatus.CANCELLED, ProposalStatus.EXPIRED):
            tp = PendingActionProposal(
                proposal_id=f"prop_{term_st.value}",
                execution_id="exec_1",
                original_call_id=f"c_{term_st.value}",
                tool_id="execute_command",
            )
            tp.transition(term_st, blocked_reason="stopped")
            with pytest.raises(InvalidProposalTransitionError):
                tp.transition(ProposalStatus.DISPATCHED)

        # FAILED requires DISPATCHED first; direct PENDING -> FAILED is invalid
        fp = PendingActionProposal(
            proposal_id="prop_failed",
            execution_id="exec_1",
            original_call_id="c_failed",
            tool_id="execute_command",
        )
        with pytest.raises(InvalidProposalTransitionError):
            fp.transition(ProposalStatus.FAILED)
        fp.transition(ProposalStatus.DISPATCHED)
        fp.transition(ProposalStatus.FAILED, blocked_reason="command_error")
        with pytest.raises(InvalidProposalTransitionError):
            fp.transition(ProposalStatus.DISPATCHED)

        blk = PendingActionProposal(
            proposal_id="prop_blk",
            execution_id="exec_1",
            original_call_id="c_blk",
            tool_id="execute_command",
        )
        blk.transition(ProposalStatus.BLOCKED, blocked_reason="waiting")
        with pytest.raises(InvalidProposalTransitionError):
            blk.transition(ProposalStatus.DISPATCHED)

        # 2. Corrupt proposal payload fails closed
        with pytest.raises(CorruptedProposalError):
            PendingActionProposal.from_dict({"proposal_id": "bad", "original_call_id": "c", "tool_id": "x"})
        with pytest.raises(CorruptedProposalError):
            PendingActionProposal.from_dict(
                {
                    "proposal_id": "bad2",
                    "execution_id": "exec_1",
                    "original_call_id": "c",
                    "tool_id": "x",
                    "status": "completed",
                    "normalized_arguments": "not_a_dict",
                }
            )

        # 3. Durable SQLite round-trip in existing execution_ledger.db + tamper failure
        import sqlite3

        db_path = tmp_path / "ledger.db"
        ledger = CentralExecutionLedger(db_path=db_path, raise_on_persistence_error=True)
        try:
            rec = ledger.create_execution("Persist pending queue", status=ExecutionStatus.INTERRUPTED)
            cp = CheckpointManager.capture_checkpoint(
                ledger,
                rec.execution_id,
                status=ExecutionStatus.INTERRUPTED,
                pending_actions=[
                    {
                        "proposal_id": "prop_a",
                        "execution_id": rec.execution_id,
                        "original_call_id": "ca",
                        "tool_id": "execute_command",
                        "normalized_arguments": {"command": "echo A", "token": "sk-secret-value-123"},
                        "proposal_order": 0,
                        "status": "pending",
                    }
                ],
            )
            assert cp is not None
            assert len(cp.pending_actions) == 1
            assert cp.pending_actions[0]["normalized_arguments"].get("token") != "sk-secret-value-123"
        finally:
            ledger.close()

        # Tamper pending_actions_json in SQLite without recomputing digest -> must fail closed
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "UPDATE task_checkpoints SET pending_actions_json = '[]' WHERE execution_id = ?",
                (rec.execution_id,),
            )
        ledger2 = CentralExecutionLedger(db_path=db_path, raise_on_persistence_error=True)
        try:
            with pytest.raises(CorruptedCheckpointError):
                ledger2.get_latest_checkpoint(rec.execution_id, verify_integrity=True)
        finally:
            ledger2.close()

    def test_interrupted_provider_error_preserves_queue_and_resumes_in_order(self, tmp_path: Path):
        import asyncio
        import httpx
        from zelyr.engine import CentralExecutionLedger, CheckpointManager, ProposalStatus

        db_path = tmp_path / "resume_queue.db"
        ledger = CentralExecutionLedger(db_path=db_path, raise_on_persistence_error=True)
        executed: List[str] = []

        async def _run():
            orch1 = NativeOrchestrator(max_steps=5, central_ledger=ledger)
            turn1 = {"n": 0}

            async def prov_interrupt(payload):
                turn1["n"] += 1
                if turn1["n"] == 1:
                    yield {
                        "choices": [{
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "id": "c1", "type": "function", "function": {"name": "execute_command", "arguments": json.dumps({"command": "step1"})}},
                                    {"index": 1, "id": "c2", "type": "function", "function": {"name": "execute_command", "arguments": json.dumps({"command": "step2"})}},
                                    {"index": 2, "id": "c3", "type": "function", "function": {"name": "execute_command", "arguments": json.dumps({"command": "step3"})}},
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }]
                    }
                else:
                    raise httpx.ConnectError("Connection lost to provider")

            async def exec_ok(name, **kwargs):
                executed.append(kwargs.get("command"))
                return {"status": "success", "output": f"ran {kwargs.get('command')}"}

            async for _ in orch1.generate_stream(
                {"messages": [{"role": "user", "content": "Run step1, then step2, then step3"}]},
                stream_provider=prov_interrupt,
                tool_executor=exec_ok,
            ):
                pass

            assert executed == ["step1"]
            exec_id = orch1.current_ledger._execution_id

            # Open fresh ledger instance (simulating process restart)
            ledger.close()
            restarted_ledger = CentralExecutionLedger(db_path=db_path, raise_on_persistence_error=True)
            try:
                assess = CheckpointManager.assess_resume(restarted_ledger, exec_id)
                assert assess.can_resume is True
                by_cid = {p.original_call_id: p for p in assess.pending_actions}
                assert by_cid["c1"].status == ProposalStatus.COMPLETED.value
                assert by_cid["c1"].dispatch_attempted is True
                assert by_cid["c2"].status == ProposalStatus.VALIDATED.value
                assert by_cid["c3"].status == ProposalStatus.VALIDATED.value
                assert by_cid["c2"].proposal_order < by_cid["c3"].proposal_order

                # Resume execution and verify one-tool-per-step execution continues c2 -> c3 in order without repeating c1
                orch2 = NativeOrchestrator(max_steps=6, central_ledger=restarted_ledger)

                async def prov_resume(payload):
                    msgs = payload.get("messages", [])
                    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
                    if len(tool_msgs) == 0:
                        yield {
                            "choices": [{
                                "delta": {
                                    "tool_calls": [
                                        {"index": 0, "id": "c2_resumed", "type": "function", "function": {"name": "execute_command", "arguments": json.dumps({"command": "step2"})}},
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }]
                        }
                    elif len(tool_msgs) == 1:
                        yield {
                            "choices": [{
                                "delta": {
                                    "tool_calls": [
                                        {"index": 0, "id": "c3_resumed", "type": "function", "function": {"name": "execute_command", "arguments": json.dumps({"command": "step3"})}},
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }]
                        }
                    else:
                        yield {"choices": [{"delta": {"content": "Finished remaining steps."}}]}

                resumed_assess, _ = await CheckpointManager.resume_execution(
                    orch2,
                    restarted_ledger,
                    exec_id,
                    stream_provider=prov_resume,
                    tool_executor=exec_ok,
                )
                assert len(resumed_assess.pending_actions) >= 2
                assert executed == ["step1", "step2", "step3"]
            finally:
                restarted_ledger.close()

        asyncio.run(_run())

    def test_dependency_ordering_failure_blocking_and_runtime_enforcement(self, tmp_path: Path):
        import asyncio
        from zelyr.engine import CentralExecutionLedger, CheckpointManager, ExecutionStatus, ProposalStatus

        db_path = tmp_path / "deps.db"
        ledger = CentralExecutionLedger(db_path=db_path, raise_on_persistence_error=True)

        async def _run():
            # A: Dep waiting on pending prerequisite, and runtime blocking out-of-order dispatch
            rec_a = ledger.create_execution("Dep chain task", status=ExecutionStatus.INTERRUPTED)
            CheckpointManager.capture_checkpoint(
                ledger,
                rec_a.execution_id,
                status=ExecutionStatus.INTERRUPTED,
                pending_actions=[
                    {
                        "proposal_id": "prop_prereq",
                        "execution_id": rec_a.execution_id,
                        "original_call_id": "c_pre",
                        "tool_id": "execute_command",
                        "normalized_arguments": {"command": "echo first"},
                        "proposal_order": 0,
                        "status": "pending",
                    },
                    {
                        "proposal_id": "prop_child",
                        "execution_id": rec_a.execution_id,
                        "original_call_id": "c_child",
                        "tool_id": "execute_command",
                        "normalized_arguments": {"command": "echo second"},
                        "proposal_order": 1,
                        "depends_on": ["prop_prereq"],
                        "status": "pending",
                    },
                ],
            )
            assess_a = CheckpointManager.assess_resume(ledger, rec_a.execution_id)
            by_pid_a = {p.proposal_id: p for p in assess_a.pending_actions}
            assert by_pid_a["prop_prereq"].status == ProposalStatus.VALIDATED.value
            assert by_pid_a["prop_child"].status == ProposalStatus.PENDING.value
            assert by_pid_a["prop_child"].dependency_status == "waiting"
            assert CheckpointManager.get_next_dispatchable_proposal(assess_a).proposal_id == "prop_prereq"

            # Attempt out-of-order child dispatch inside orchestrator before prerequisite completes -> rejected
            orch = NativeOrchestrator(max_steps=4, central_ledger=ledger)
            executed_ooo = []

            async def exec_track(name, **kwargs):
                executed_ooo.append(kwargs.get("command"))
                return {"status": "success"}

            turn_ooo = {"n": 0}

            async def prov_ooo(payload):
                turn_ooo["n"] += 1
                if turn_ooo["n"] == 1:
                    yield {
                        "choices": [{
                            "delta": {
                                "tool_calls": [{
                                    "index": 0,
                                    "id": "try_child_early",
                                    "proposal_id": "prop_child",
                                    "depends_on": ["prop_prereq"],
                                    "type": "function",
                                    "function": {"name": "execute_command", "arguments": json.dumps({"command": "echo second"})},
                                }]
                            }
                        }]
                    }
                else:
                    yield {"choices": [{"delta": {"content": "Will run prerequisite first."}}]}

            turn_ledger = TurnExecutionLedger()
            turn_ledger.record_proposed("c_pre", "execute_command", {"command": "echo first"}, proposal_id="prop_prereq")
            async for _ in orch.generate_stream(
                {"messages": [{"role": "user", "content": "Run"}]},
                stream_provider=prov_ooo,
                tool_executor=exec_track,
                ledger=turn_ledger,
            ):
                pass
            assert executed_ooo == []
            assert "dependency_not_satisfied" in str(turn_ledger._records["try_child_early"].rejection_reason)

            # B: Failed prerequisite permanently blocks dependent child proposal
            rec_b = ledger.create_execution("Failed dep chain", status=ExecutionStatus.FAILED)
            ledger.record_event(rec_b.execution_id, "tool_selected", tool_id="execute_command", correlation_id="c_fail", status="proposed", metadata={"arguments": {"command": "bad"}})
            ledger.record_event(rec_b.execution_id, "tool_execution_start", tool_id="execute_command", correlation_id="c_fail", status="started")
            ledger.record_event(rec_b.execution_id, "tool_execution_result", tool_id="execute_command", correlation_id="c_fail", status="failed", metadata={"error": "boom", "exit_code": 1})
            CheckpointManager.capture_checkpoint(
                ledger,
                rec_b.execution_id,
                status=ExecutionStatus.FAILED,
                pending_actions=[
                    {
                        "proposal_id": "prop_fail_dep",
                        "execution_id": rec_b.execution_id,
                        "original_call_id": "c_fail",
                        "tool_id": "execute_command",
                        "normalized_arguments": {"command": "bad"},
                        "proposal_order": 0,
                        "status": "pending",
                    },
                    {
                        "proposal_id": "prop_blocked_child",
                        "execution_id": rec_b.execution_id,
                        "original_call_id": "c_blk_child",
                        "tool_id": "execute_command",
                        "normalized_arguments": {"command": "echo after_bad"},
                        "proposal_order": 1,
                        "depends_on": ["prop_fail_dep"],
                        "status": "pending",
                    },
                ],
            )
            assess_b = CheckpointManager.assess_resume(ledger, rec_b.execution_id)
            by_pid_b = {p.proposal_id: p for p in assess_b.pending_actions}
            assert by_pid_b["prop_fail_dep"].status == ProposalStatus.FAILED.value
            assert by_pid_b["prop_blocked_child"].status == ProposalStatus.BLOCKED.value
            assert by_pid_b["prop_blocked_child"].dependency_status == "failed"

        try:
            asyncio.run(_run())
        finally:
            ledger.close()
    def test_security_ttl_revocation_invalid_args_and_in_flight_ambiguity(self, tmp_path: Path):
        import time as _time
        from zelyr.engine import CentralExecutionLedger, CheckpointManager, ExecutionStatus, ProposalStatus
        from zelyr.tool_registry import ToolRegistry

        db_path = tmp_path / "sec_ttl.db"
        ledger = CentralExecutionLedger(db_path=db_path, raise_on_persistence_error=True)

        class _RevokedRegistry(ToolRegistry):
            def __init__(self, allow_ext: bool = True):
                super().__init__()
                self._allow_ext = allow_ext

            def get_external_tool_entry(self, tool_name: str):
                if tool_name == "mcp_ext_tool" and self._allow_ext:
                    return type("Ext", (), {"source_id": "srv1", "source_type": "mcp", "authorized": True, "external_tool_name": "mcp_ext_tool"})()
                if tool_name == "mcp_ext_tool":
                    return type("Ext", (), {"source_id": "srv1", "source_type": "mcp", "authorized": False, "external_tool_name": "mcp_ext_tool"})()
                return None

        try:
            # 1. Expired TTL + invalid arguments + in-flight ambiguous op never dispatched
            rec = ledger.create_execution("Mixed queue guards", status=ExecutionStatus.INTERRUPTED)
            ledger.record_event(
                rec.execution_id,
                "tool_selected",
                tool_id="execute_command",
                correlation_id="c_inflight",
                status="proposed",
                metadata={"arguments": {"command": "rm -rf temp_build"}},
            )
            ledger.record_event(
                rec.execution_id,
                "tool_execution_start",
                tool_id="execute_command",
                correlation_id="c_inflight",
                status="started",
            )
            CheckpointManager.capture_checkpoint(
                ledger,
                rec.execution_id,
                status=ExecutionStatus.INTERRUPTED,
                pending_actions=[
                    {
                        "proposal_id": "p_inflight",
                        "execution_id": rec.execution_id,
                        "original_call_id": "c_inflight",
                        "tool_id": "execute_command",
                        "normalized_arguments": {"command": "rm -rf temp_build"},
                        "proposal_order": 0,
                        "status": "pending",
                    },
                    {
                        "proposal_id": "p_expired",
                        "execution_id": rec.execution_id,
                        "original_call_id": "c_exp",
                        "tool_id": "execute_command",
                        "normalized_arguments": {"command": "echo late"},
                        "proposal_order": 1,
                        "expires_at": _time.time() - 60.0,
                        "status": "pending",
                    },
                    {
                        "proposal_id": "p_bad_args",
                        "execution_id": rec.execution_id,
                        "original_call_id": "c_bad",
                        "tool_id": "execute_command",
                        "normalized_arguments": {"caller_ring_grant": 0},  # invalid schema + stripped auth grant!
                        "proposal_order": 2,
                        "status": "pending",
                    },
                    {
                        "proposal_id": "p_good",
                        "execution_id": rec.execution_id,
                        "original_call_id": "c_good",
                        "tool_id": "execute_command",
                        "normalized_arguments": {"command": "echo safe"},
                        "proposal_order": 3,
                        "status": "pending",
                    },
                ],
            )
            schemas = [
                {
                    "type": "function",
                    "function": {
                        "name": "execute_command",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"],
                        },
                    },
                }
            ]
            assess = CheckpointManager.assess_resume(
                ledger,
                rec.execution_id,
                tool_registry=ToolRegistry(),
                active_schemas=schemas,
                caller_ring=0,
            )
            by_pid = {p.proposal_id: p for p in assess.pending_actions}
            # In-flight ledger reconciliation overrides stale checkpoint status to DISPATCHED
            assert by_pid["p_inflight"].status == ProposalStatus.DISPATCHED.value
            assert by_pid["p_inflight"].dispatch_attempted is True
            # Expired proposal marked EXPIRED
            assert by_pid["p_expired"].status == ProposalStatus.EXPIRED.value
            # Invalid args marked BLOCKED and stripped checkpoint grant key
            assert by_pid["p_bad_args"].status == ProposalStatus.BLOCKED.value
            assert by_pid["p_bad_args"].authorization_status == "invalid_arguments"
            assert "caller_ring_grant" not in by_pid["p_bad_args"].normalized_arguments
            # Valid proposal remains VALIDATED and only dispatchable item
            assert by_pid["p_good"].status == ProposalStatus.VALIDATED.value
            assert by_pid["p_good"].authorization_status == "authorized"
            payload = CheckpointManager.prepare_resumed_payload(assess, tools=schemas)
            assert [p["proposal_id"] for p in payload.get("_pending_proposals", [])] == ["p_good"]
            assert "p_inflight" in payload.get("_guarded_pending_proposals", {})
            assert "p_expired" in payload.get("_guarded_pending_proposals", {})

            # 2. Revoked external tool blocks resume cleanly
            rec_ext = ledger.create_execution("External pending", status=ExecutionStatus.INTERRUPTED)
            CheckpointManager.capture_checkpoint(
                ledger,
                rec_ext.execution_id,
                status=ExecutionStatus.INTERRUPTED,
                pending_actions=[
                    {
                        "proposal_id": "p_ext",
                        "execution_id": rec_ext.execution_id,
                        "original_call_id": "c_ext",
                        "tool_id": "mcp_ext_tool",
                        "normalized_arguments": {},
                        "external_source": {"is_external": True, "source_id": "srv1", "source_type": "mcp"},
                        "status": "pending",
                    }
                ],
            )
            assess_revoked = CheckpointManager.assess_resume(
                ledger,
                rec_ext.execution_id,
                tool_registry=_RevokedRegistry(allow_ext=False),
            )
            assert assess_revoked.can_resume is False
            assert assess_revoked.pending_actions[0].status == ProposalStatus.BLOCKED.value
            assert assess_revoked.pending_actions[0].authorization_status == "denied"
        finally:
            ledger.close()

    def test_legacy_checkpoint_migration_and_cli_visibility(self, tmp_path: Path, capsys, monkeypatch):
        import sqlite3
        import sys
        from zelyr.cli.main import main
        from zelyr.engine import CentralExecutionLedger, CheckpointManager, ProposalStatus

        db_path = tmp_path / "legacy_cp.db"
        # Simulate v1 checkpoint table created BEFORE pending_actions_json column existed
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(
                """
                CREATE TABLE executions (
                    execution_id TEXT PRIMARY KEY,
                    parent_execution_id TEXT,
                    task_prompt TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'initialized',
                    verification_status TEXT NOT NULL DEFAULT 'not_verified',
                    source_component TEXT NOT NULL DEFAULT 'runtime',
                    caller_ring INTEGER,
                    task_local_scoped INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    completed_at REAL,
                    final_summary TEXT,
                    error TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    redacted INTEGER NOT NULL DEFAULT 0,
                    truncated INTEGER NOT NULL DEFAULT 0,
                    next_seq INTEGER NOT NULL DEFAULT 1,
                    latest_checkpoint_seq INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE execution_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    execution_id TEXT NOT NULL,
                    sequence_number INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    tool_id TEXT,
                    source_id TEXT,
                    source_type TEXT,
                    is_external INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'recorded',
                    correlation_id TEXT,
                    caller_ring INTEGER,
                    task_local_scoped INTEGER NOT NULL DEFAULT 0,
                    redacted INTEGER NOT NULL DEFAULT 0,
                    truncated INTEGER NOT NULL DEFAULT 0,
                    idempotency_key TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE task_checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    execution_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    checkpoint_version INTEGER NOT NULL DEFAULT 1,
                    sequence_number INTEGER NOT NULL,
                    last_event_id TEXT,
                    last_event_seq INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    verification_status TEXT NOT NULL,
                    task_prompt TEXT NOT NULL DEFAULT '',
                    planning_json TEXT NOT NULL DEFAULT '{}',
                    environment_json TEXT NOT NULL DEFAULT '{}',
                    external_tools_json TEXT NOT NULL DEFAULT '[]',
                    verified_targets_json TEXT NOT NULL DEFAULT '[]',
                    recovery_summary_json TEXT NOT NULL DEFAULT '{}',
                    checkpoint_digest TEXT NOT NULL,
                    redacted INTEGER NOT NULL DEFAULT 0,
                    truncated INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL
                );
                """
            )
            plan_legacy = {"pending_proposals": [{"call_id": "leg_c2", "tool_name": "execute_command", "arguments": {"command": "echo leg"}}]}
            cp_digest = CentralExecutionLedger._compute_checkpoint_digest({
                "checkpoint_version": 1,
                "checkpoint_id": "cp_legacy_1",
                "execution_id": "exec_leg_cp",
                "task_id": "exec_leg_cp",
                "sequence_number": 1,
                "last_event_id": None,
                "last_event_seq": 0,
                "status": "interrupted",
                "verification_status": "interrupted",
                "task_prompt": "Legacy queued task",
                "planning_state": plan_legacy,
                "environment_snapshot": {},
                "external_tools_used": [],
                "verified_targets": [],
                "recovery_summary": {},
            })
            conn.execute(
                "INSERT INTO executions (execution_id, task_prompt, status, verification_status, created_at, updated_at, next_seq, latest_checkpoint_seq) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("exec_leg_cp", "Legacy queued task", "interrupted", "interrupted", 10.0, 10.0, 1, 1),
            )
            conn.execute(
                """
                INSERT INTO task_checkpoints (
                    checkpoint_id, execution_id, task_id, checkpoint_version, sequence_number,
                    last_event_id, last_event_seq, status, verification_status, task_prompt,
                    planning_json, environment_json, external_tools_json, verified_targets_json,
                    recovery_summary_json, checkpoint_digest, redacted, truncated, created_at
                ) VALUES (?, ?, ?, 1, 1, NULL, 0, 'interrupted', 'interrupted', 'Legacy queued task', ?, '{}', '[]', '[]', '{}', ?, 0, 0, 10.0)
                """,
                ("cp_legacy_1", "exec_leg_cp", "exec_leg_cp", json.dumps(plan_legacy, sort_keys=True, separators=(",", ":")), cp_digest),
            )

        ledger = CentralExecutionLedger(db_path=db_path, raise_on_persistence_error=True)
        try:
            assess = CheckpointManager.assess_resume(ledger, "exec_leg_cp")
            assert assess.can_resume is True
            assert len(assess.pending_actions) == 1
            assert assess.pending_actions[0].original_call_id == "leg_c2"
            assert assess.pending_actions[0].status == ProposalStatus.VALIDATED.value
        finally:
            ledger.close()

        # Verify CLI JSON & human inspection include pending_proposal_count and pending_actions details
        monkeypatch.setattr(sys, "argv", ["zelyr", "executions", "--resumable", "--db-path", str(db_path), "--json"])
        main()
        list_json = json.loads(capsys.readouterr().out)
        assert list_json["resumable_tasks"][0]["pending_proposal_count"] == 1
        assert list_json["resumable_tasks"][0]["pending_actions"][0]["original_call_id"] == "leg_c2"

        monkeypatch.setattr(sys, "argv", ["zelyr", "execution", "exec_leg_cp", "--checkpoint", "--db-path", str(db_path)])
        main()
        insp_text = capsys.readouterr().out
        assert "Pending Proposals:" in insp_text
        assert "prop_leg_c2" in insp_text

        monkeypatch.setattr(sys, "argv", ["zelyr", "checkpoint", "abandon", "exec_leg_cp", "--db-path", str(db_path), "--json"])
        main()
        capsys.readouterr()

        ledger3 = CentralExecutionLedger(db_path=db_path, raise_on_persistence_error=True)
        try:
            after_abandon = CheckpointManager.assess_resume(ledger3, "exec_leg_cp")
            assert after_abandon.can_resume is False
            assert after_abandon.pending_actions[0].status == ProposalStatus.CANCELLED.value
            assert after_abandon.pending_actions[0].blocked_reason
        finally:
            ledger3.close()

    def test_verified_dep_resume_cancellation_env_drift_and_persistence_errors(self, tmp_path: Path, monkeypatch):
        import asyncio
        import sqlite3
        from zelyr.engine import (
            CentralExecutionLedger,
            CheckpointManager,
            CorruptedCheckpointError,
            ExecutionStatus,
            LedgerPersistenceError,
            ProposalStatus,
        )

        db_path = tmp_path / "edge_cases.db"
        ledger = CentralExecutionLedger(db_path=db_path, raise_on_persistence_error=True)

        async def _run():
            # 1. Dependency already verified before interruption -> child becomes VALIDATED without repeating parent
            rec_dep = ledger.create_execution(
                "Create dep.txt then echo done",
                status=ExecutionStatus.INTERRUPTED,
            )
            ledger.record_event(
                rec_dep.execution_id,
                "tool_selected",
                tool_id="execute_command",
                correlation_id="c_done",
                status="proposed",
                metadata={"arguments": {"command": "touch dep.txt"}, "proposal_id": "prop_done"},
            )
            ledger.record_event(rec_dep.execution_id, "tool_execution_start", tool_id="execute_command", correlation_id="c_done", status="started")
            ledger.record_event(rec_dep.execution_id, "tool_execution_result", tool_id="execute_command", correlation_id="c_done", status="succeeded", metadata={"verified": True, "exit_code": 0})
            CheckpointManager.capture_checkpoint(
                ledger,
                rec_dep.execution_id,
                status=ExecutionStatus.INTERRUPTED,
                pending_actions=[
                    {
                        "proposal_id": "prop_done",
                        "execution_id": rec_dep.execution_id,
                        "original_call_id": "c_done",
                        "tool_id": "execute_command",
                        "normalized_arguments": {"command": "touch dep.txt"},
                        "proposal_order": 0,
                        "status": "pending",
                    },
                    {
                        "proposal_id": "prop_after_done",
                        "execution_id": rec_dep.execution_id,
                        "original_call_id": "c_after",
                        "tool_id": "execute_command",
                        "normalized_arguments": {"command": "echo done"},
                        "proposal_order": 1,
                        "depends_on": ["prop_done"],
                        "status": "pending",
                    },
                ],
            )
            assess_dep = CheckpointManager.assess_resume(ledger, rec_dep.execution_id)
            by_pid = {p.proposal_id: p for p in assess_dep.pending_actions}
            assert by_pid["prop_done"].status == ProposalStatus.COMPLETED.value
            assert by_pid["prop_after_done"].status == ProposalStatus.VALIDATED.value
            assert by_pid["prop_after_done"].dependency_status == "satisfied"
            assert CheckpointManager.get_next_dispatchable_proposal(assess_dep).proposal_id == "prop_after_done"

            # 2. User cancellation transitions queued proposals to CANCELLED with blocked_reason & does not auto-dispatch on --confirm
            orch_can = NativeOrchestrator(max_steps=4, central_ledger=ledger)

            async def prov_can(payload):
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "cc1", "type": "function", "function": {"name": "execute_command", "arguments": json.dumps({"command": "first"})}},
                                {"index": 1, "id": "cc2", "type": "function", "function": {"name": "execute_command", "arguments": json.dumps({"command": "second"})}},
                            ]
                        }
                    }]
                }

            async def exec_cancel(name, **kwargs):
                orch_can.cancel(reason="user_cancelled")
                return {"status": "success"}

            async for _ in orch_can.generate_stream(
                {"messages": [{"role": "user", "content": "Run two steps"}]},
                stream_provider=prov_can,
                tool_executor=exec_cancel,
            ):
                pass
            can_exec_id = orch_can.current_ledger._execution_id
            assess_can = CheckpointManager.assess_resume(ledger, can_exec_id, explicit_resume=True)
            assert assess_can.can_resume is True
            can_by_cid = {p.original_call_id: p for p in assess_can.pending_actions}
            assert can_by_cid["cc2"].status == ProposalStatus.CANCELLED.value
            assert can_by_cid["cc2"].blocked_reason
            payload_can = CheckpointManager.prepare_resumed_payload(assess_can)
            assert payload_can.get("_pending_proposals") in (None, [])

            # 3. Environment capability loss blocks affected queued pending proposal
            rec_ui = ledger.create_execution("Click button", status=ExecutionStatus.INTERRUPTED)
            CheckpointManager.capture_checkpoint(
                ledger,
                rec_ui.execution_id,
                status=ExecutionStatus.INTERRUPTED,
                environment_snapshot={"os": "Linux", "desktop_available": True},
                pending_actions=[
                    {
                        "proposal_id": "prop_ui",
                        "execution_id": rec_ui.execution_id,
                        "original_call_id": "c_ui",
                        "tool_id": "interact_with_ui",
                        "normalized_arguments": {"target": "Save"},
                        "proposal_order": 0,
                        "status": "pending",
                    }
                ],
            )
            assess_ui = CheckpointManager.assess_resume(
                ledger,
                rec_ui.execution_id,
                current_environment={"os": "Linux", "desktop_available": False},
            )
            assert assess_ui.pending_actions[0].status == ProposalStatus.BLOCKED.value
            assert "unavailable" in str(assess_ui.pending_actions[0].blocked_reason).lower()

            # 4. Corrupted pending_actions_json fails closed on retrieval
            rec_bad = ledger.create_execution("Bad json prop", status=ExecutionStatus.INTERRUPTED)
            CheckpointManager.capture_checkpoint(ledger, rec_bad.execution_id, status=ExecutionStatus.INTERRUPTED)
            with sqlite3.connect(str(db_path)) as conn:
                conn.execute(
                    "UPDATE task_checkpoints SET pending_actions_json = '{not-json' WHERE execution_id = ?",
                    (rec_bad.execution_id,),
                )
            with pytest.raises(CorruptedCheckpointError):
                ledger.get_latest_checkpoint(rec_bad.execution_id, verify_integrity=True)
            assess_bad = CheckpointManager.assess_resume(ledger, rec_bad.execution_id)
            assert assess_bad.can_resume is False
            assert any("corrupted checkpoint" in r.lower() for r in assess_bad.blocking_reasons)

            # 5. Persistence failure is explicitly raised when raise_on_persistence_error=True
            rec_pf = ledger.create_execution("Fail save", status=ExecutionStatus.INTERRUPTED)

            def _boom(*_a, **_kw):
                raise RuntimeError("disk full")

            monkeypatch.setattr(ledger, "_ensure_connection", _boom)
            with pytest.raises((LedgerPersistenceError, RuntimeError)):
                CheckpointManager.capture_checkpoint(
                    ledger,
                    rec_pf.execution_id,
                    status=ExecutionStatus.INTERRUPTED,
                    strict=True,
                )

        try:
            asyncio.run(_run())
        finally:
            ledger.close()
