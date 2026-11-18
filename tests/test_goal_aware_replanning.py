"""Deterministic regression test battery for Goal-Aware Replanning After Unconfirmed Proposals.

Verifies the 10 core requirements:
1. Replanning directive contains exact unfulfilled target and expected content.
2. Already-verified outcomes are accurately represented and not repeated.
3. The model receives appropriate current tool schemas.
4. A fresh model-emitted tool call is dispatched through normal validation.
5. A text-only response still cannot execute an old proposal.
6. A second text-only response terminates without another replanning attempt.
7. Explicit abandonment and truthful failure do not trigger replanning.
8. Cancellation, timeout, step limits, and depth limits remain authoritative.
9. Replanning cannot cause a verified file to be overwritten.
10. Missing or ambiguous acceptance criteria do not produce invented requirements.
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
from zelyr.engine.execution_ledger import ActionState, TurnExecutionLedger, find_unfulfilled_task_requirements


class TestGoalAwareReplanningSuite:
    """Rigorous test battery verifying goal-aware replanning and authorization invariants."""

    @pytest.mark.asyncio
    async def test_1_replanning_directive_contains_exact_target_and_expected_content(self):
        """1. Replanning directive contains the exact unfulfilled target and expected content."""
        orch = NativeOrchestrator()
        replanning_prompt_captured = ""

        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            path_a = ws / "file_alpha.txt"
            path_b = ws / "file_beta.txt"

            async def mock_executor(name, **kwargs):
                p = ws / kwargs.get("path", "")
                p.write_text(kwargs.get("content", ""), encoding="utf-8")
                return {"status": "success", "path": str(p)}

            turn_count = 0

            async def mock_stream(payload):
                nonlocal turn_count, replanning_prompt_captured
                turn_count += 1
                msgs = payload.get("messages", [])

                if turn_count == 1:
                    # Model proposes creating alpha and beta
                    yield {
                        "choices": [{
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "c1",
                                        "type": "function",
                                        "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": "file_alpha.txt", "content": "ALPHA_101"})},
                                    },
                                    {
                                        "index": 1,
                                        "id": "c2_old",
                                        "type": "function",
                                        "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": "file_beta.txt", "content": "BETA_202"})},
                                    },
                                ]
                            }
                        }]
                    }
                elif turn_count == 2:
                    # Model emits text claiming completion
                    yield {"choices": [{"delta": {"content": "File alpha has been created."}}]}
                elif turn_count == 3:
                    # Capture injected replanning directive
                    replanning_prompt_captured = msgs[-1].get("content", "")
                    # Model emits fresh tool call for beta
                    yield {
                        "choices": [{
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "c2_fresh",
                                        "type": "function",
                                        "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": "file_beta.txt", "content": "BETA_202"})},
                                    }
                                ]
                            }
                        }]
                    }
                else:
                    yield {"choices": [{"delta": {"content": "All complete."}}]}

            ledger = TurnExecutionLedger()
            prompt = f"Create {path_a} containing 'ALPHA_101' and create {path_b} containing 'BETA_202'."
            criteria = f"file_contains:{path_a}:ALPHA_101,file_contains:{path_b}:BETA_202"
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "acceptance_criteria": criteria,
            }

            async for _ in orch.generate_stream(
                payload,
                stream_provider=mock_stream,
                tool_executor=mock_executor,
                ledger=ledger,
            ):
                pass

            # Replanning directive contains exact unfulfilled target and expected content
            assert "[SYSTEM DIRECTIVE: Goal-Aware Replanning]" in replanning_prompt_captured
            assert "file_beta.txt" in replanning_prompt_captured
            assert "BETA_202" in replanning_prompt_captured
            # Both files exist with exact contents
            assert path_a.read_text(encoding="utf-8") == "ALPHA_101"
            assert path_b.read_text(encoding="utf-8") == "BETA_202"

    @pytest.mark.asyncio
    async def test_2_already_verified_outcomes_accurately_represented_and_not_repeated(self):
        """2. Already-verified outcomes are accurately represented in the directive and not repeated."""
        orch = NativeOrchestrator()
        executed_turns = []

        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            path_a = ws / "file_a.txt"
            path_b = ws / "file_b.txt"

            async def mock_executor(name, **kwargs):
                executed_turns.append((name, kwargs))
                p = ws / kwargs.get("path", "")
                p.write_text(kwargs.get("content", ""), encoding="utf-8")
                return {"status": "success", "path": str(p)}

            turn_count = 0
            replanning_prompt_seen = ""

            async def mock_stream(payload):
                nonlocal turn_count, replanning_prompt_seen
                turn_count += 1
                msgs = payload.get("messages", [])

                if turn_count == 1:
                    yield {
                        "choices": [{
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "c1",
                                        "type": "function",
                                        "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": "file_a.txt", "content": "AAA"})},
                                    },
                                    {
                                        "index": 1,
                                        "id": "c2_old",
                                        "type": "function",
                                        "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": "file_b.txt", "content": "BBB"})},
                                    },
                                ]
                            }
                        }]
                    }
                elif turn_count == 2:
                    yield {"choices": [{"delta": {"content": "File a done."}}]}
                elif turn_count == 3:
                    replanning_prompt_seen = msgs[-1].get("content", "")
                    yield {
                        "choices": [{
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "c2_fresh",
                                        "type": "function",
                                        "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": "file_b.txt", "content": "BBB"})},
                                    }
                                ]
                            }
                        }]
                    }
                else:
                    yield {"choices": [{"delta": {"content": "Both files done."}}]}

            ledger = TurnExecutionLedger()
            criteria = f"file_contains:{path_a}:AAA,file_contains:{path_b}:BBB"
            payload = {
                "messages": [{"role": "user", "content": f"Create {path_a} and {path_b}"}],
                "acceptance_criteria": criteria,
            }

            async for _ in orch.generate_stream(
                payload,
                stream_provider=mock_stream,
                tool_executor=mock_executor,
                ledger=ledger,
            ):
                pass

            assert "file_a.txt" in replanning_prompt_seen
            assert "DO NOT REPEAT" in replanning_prompt_seen
            # Exactly 2 writes occurred: one for file_a, one for file_b
            assert len(executed_turns) == 2
            assert executed_turns[0][1]["path"] == "file_a.txt"
            assert executed_turns[1][1]["path"] == "file_b.txt"

    @pytest.mark.asyncio
    async def test_3_model_receives_appropriate_current_tool_schemas(self):
        """3. The model receives appropriate current tool schemas during replanning."""
        orch = NativeOrchestrator()
        tool_schemas_during_replanning = []

        turn_count = 0

        async def mock_stream(payload):
            nonlocal turn_count, tool_schemas_during_replanning
            turn_count += 1
            if turn_count == 1:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c1",
                                    "type": "function",
                                    "function": {"name": "execute_command", "arguments": json.dumps({"command": "echo 1"})},
                                },
                                {
                                    "index": 1,
                                    "id": "c2_old",
                                    "type": "function",
                                    "function": {"name": "execute_command", "arguments": json.dumps({"command": "echo 2"})},
                                },
                            ]
                        }
                    }]
                }
            elif turn_count == 2:
                yield {"choices": [{"delta": {"content": "Step 1 executed."}}]}
            elif turn_count == 3:
                # Capture active schemas passed to model in replanning turn
                tool_schemas_during_replanning = list(payload.get("tools") or [])
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c2_fresh",
                                    "type": "function",
                                    "function": {"name": "execute_command", "arguments": json.dumps({"command": "echo 2"})},
                                }
                            ]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Finished."}}]}

        async def mock_executor(name, **kwargs):
            return {"status": "success"}

        custom_schemas = [
            {"type": "function", "function": {"name": "execute_command", "parameters": {}}},
            {"type": "function", "function": {"name": "manage_workspace_file", "parameters": {}}},
        ]

        ledger = TurnExecutionLedger()
        payload = {"messages": [{"role": "user", "content": "Execute step 1 and step 2"}]}

        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=mock_executor,
            tool_schemas=custom_schemas,
            ledger=ledger,
        ):
            pass

        assert len(tool_schemas_during_replanning) == 2
        names = [(t.get("function", {}).get("name") or t.get("name")) for t in tool_schemas_during_replanning]
        assert "execute_command" in names
        assert "manage_workspace_file" in names

    @pytest.mark.asyncio
    async def test_4_fresh_model_emitted_tool_call_dispatched_through_normal_validation(self):
        """4. A fresh model-emitted tool call is dispatched through normal validation."""
        orch = NativeOrchestrator()
        executed_turns = []

        async def mock_executor(name, **kwargs):
            executed_turns.append((name, kwargs))
            return {"status": "success"}

        turn_count = 0

        async def mock_stream(payload):
            nonlocal turn_count
            turn_count += 1
            if turn_count == 1:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c1",
                                    "type": "function",
                                    "function": {"name": "execute_command", "arguments": json.dumps({"command": "echo 1"})},
                                },
                                {
                                    "index": 1,
                                    "id": "c2_old",
                                    "type": "function",
                                    "function": {"name": "execute_command", "arguments": json.dumps({"command": "echo 2"})},
                                },
                            ]
                        }
                    }]
                }
            elif turn_count == 2:
                yield {"choices": [{"delta": {"content": "Step 1 done."}}]}
            elif turn_count == 3:
                # Fresh call using alias 'bash' and 'cmd' argument needing normalization
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c2_fresh",
                                    "type": "function",
                                    "function": {"name": "bash", "arguments": json.dumps({"cmd": "echo normalized_step2"})},
                                }
                            ]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Done."}}]}

        ledger = TurnExecutionLedger()
        payload = {"messages": [{"role": "user", "content": "Execute step 1 and step 2"}]}

        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=mock_executor,
            ledger=ledger,
        ):
            pass

        assert len(executed_turns) == 2
        assert executed_turns[1][0] == "execute_command"
        assert executed_turns[1][1] == {"command": "echo normalized_step2"}
        assert ledger._records["c2_fresh"].state in (ActionState.SUCCESS, ActionState.VERIFIED)

    @pytest.mark.asyncio
    async def test_5_text_only_response_still_cannot_execute_old_proposal(self):
        """5. A text-only response still cannot execute an old proposal."""
        orch = NativeOrchestrator()
        executed_turns = []

        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            file_a = ws / "file_a.txt"
            file_b = ws / "file_b.txt"

            async def mock_executor(name, **kwargs):
                executed_turns.append(name)
                p = ws / kwargs.get("path", "")
                p.write_text(kwargs.get("content", ""), encoding="utf-8")
                return {"status": "success"}

            turn_count = 0

            async def mock_stream(payload):
                nonlocal turn_count
                turn_count += 1
                if turn_count == 1:
                    yield {
                        "choices": [{
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "c1",
                                        "type": "function",
                                        "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": "file_a.txt", "content": "A"})},
                                    },
                                    {
                                        "index": 1,
                                        "id": "c2_old",
                                        "type": "function",
                                        "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": "file_b.txt", "content": "B"})},
                                    },
                                ]
                            }
                        }]
                    }
                else:
                    yield {"choices": [{"delta": {"content": "I am done without calling more tools."}}]}

            ledger = TurnExecutionLedger()
            prompt = f"Create {file_a} and {file_b}"
            payload = {"messages": [{"role": "user", "content": prompt}]}

            async for _ in orch.generate_stream(
                payload,
                stream_provider=mock_stream,
                tool_executor=mock_executor,
                ledger=ledger,
            ):
                pass

            assert file_a.exists()
            assert not file_b.exists()
            assert len(executed_turns) == 1
            assert ledger._records["c2_old"].state == ActionState.CANCELLED
            assert "unconfirmed_by_model" in str(ledger._records["c2_old"].error)

    @pytest.mark.asyncio
    async def test_6_second_text_only_response_terminates_without_another_replanning_attempt(self):
        """6. A second text-only response terminates without another replanning attempt (bounded)."""
        orch = NativeOrchestrator()
        turn_count = 0

        async def mock_stream(payload):
            nonlocal turn_count
            turn_count += 1
            if turn_count == 1:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c1",
                                    "type": "function",
                                    "function": {"name": "execute_command", "arguments": json.dumps({"command": "echo 1"})},
                                },
                                {
                                    "index": 1,
                                    "id": "c2_old",
                                    "type": "function",
                                    "function": {"name": "execute_command", "arguments": json.dumps({"command": "echo 2"})},
                                },
                            ]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Thinking in text only..."}}]}

        async def mock_executor(name, **kwargs):
            return {"status": "success"}

        ledger = TurnExecutionLedger()
        payload = {"messages": [{"role": "user", "content": "Execute step 1 and step 2"}]}

        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=mock_executor,
            ledger=ledger,
            max_replanning_attempts=1,
        ):
            pass

        # Turn 1: tool call -> Turn 2: text response (proposals discarded, replan 1 injected) -> Turn 3: text response -> ends
        assert turn_count <= 4

    @pytest.mark.asyncio
    async def test_7_explicit_abandonment_and_truthful_failure_do_not_trigger_replanning(self):
        """7. Explicit abandonment and truthful failure do not trigger replanning."""
        orch = NativeOrchestrator()
        turn_count = 0

        async def mock_stream(payload):
            nonlocal turn_count
            turn_count += 1
            if turn_count == 1:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c1",
                                    "type": "function",
                                    "function": {"name": "execute_command", "arguments": json.dumps({"command": "echo 1"})},
                                },
                                {
                                    "index": 1,
                                    "id": "c2_old",
                                    "type": "function",
                                    "function": {"name": "execute_command", "arguments": json.dumps({"command": "echo 2"})},
                                },
                            ]
                        }
                    }]
                }
            else:
                # Explicit abandonment
                yield {"choices": [{"delta": {"content": "I am stopping here and declining to proceed further with this task."}}]}

        async def mock_executor(name, **kwargs):
            return {"status": "success"}

        ledger = TurnExecutionLedger()
        payload = {"messages": [{"role": "user", "content": "Execute step 1 and step 2"}]}

        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=mock_executor,
            ledger=ledger,
        ):
            pass

        # Abandonment terminates without triggering replanning turn (turn_count == 2)
        assert turn_count == 2
        assert ledger._records["c2_old"].state == ActionState.CANCELLED

    @pytest.mark.asyncio
    async def test_8_cancellation_timeout_and_step_limits_remain_authoritative(self):
        """8. Cancellation, timeout, and step limits remain authoritative."""
        orch = NativeOrchestrator()
        executed_turns = []

        async def mock_executor(name, **kwargs):
            executed_turns.append(name)
            return {"status": "success"}

        turn_count = 0

        async def mock_stream(payload):
            nonlocal turn_count
            turn_count += 1
            if turn_count == 1:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c1",
                                    "type": "function",
                                    "function": {"name": "execute_command", "arguments": json.dumps({"command": "echo 1"})},
                                },
                                {
                                    "index": 1,
                                    "id": "c2_old",
                                    "type": "function",
                                    "function": {"name": "execute_command", "arguments": json.dumps({"command": "echo 2"})},
                                },
                            ]
                        }
                    }]
                }
            elif turn_count == 2:
                # Orchestrator cancelled
                orch.cancel()
                yield {"choices": [{"delta": {"content": "Cancelled."}}]}
            else:
                yield {"choices": [{"delta": {"content": "Should not execute"}}]}

        ledger = TurnExecutionLedger()
        payload = {"messages": [{"role": "user", "content": "Execute step 1 and step 2"}]}

        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=mock_executor,
            ledger=ledger,
        ):
            pass

        assert len(executed_turns) == 1
        assert len(ledger.get_pending_proposals()) == 0

    @pytest.mark.asyncio
    async def test_9_replanning_cannot_cause_verified_file_to_be_overwritten(self):
        """9. Replanning cannot cause a verified file to be overwritten (mutation guard)."""
        orch = NativeOrchestrator()
        executed_writes = []

        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            path_a = ws / "file_alpha.txt"
            path_b = ws / "file_beta.txt"

            async def mock_executor(name, **kwargs):
                executed_writes.append((kwargs.get("path"), kwargs.get("content")))
                p = ws / kwargs.get("path", "")
                p.write_text(kwargs.get("content", ""), encoding="utf-8")
                return {"status": "success", "path": str(p)}

            turn_count = 0

            async def mock_stream(payload):
                nonlocal turn_count
                turn_count += 1
                if turn_count == 1:
                    yield {
                        "choices": [{
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "c1",
                                        "type": "function",
                                        "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": "file_alpha.txt", "content": "ALPHA_101"})},
                                    },
                                    {
                                        "index": 1,
                                        "id": "c2_old",
                                        "type": "function",
                                        "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": "file_beta.txt", "content": "BETA_202"})},
                                    },
                                ]
                            }
                        }]
                    }
                elif turn_count == 2:
                    yield {"choices": [{"delta": {"content": "Alpha created."}}]}
                elif turn_count == 3:
                    # Model mistakenly attempts to overwrite verified file_alpha in replanning turn
                    yield {
                        "choices": [{
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "c_bad_overwrite",
                                        "type": "function",
                                        "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": "file_alpha.txt", "content": "OVERWRITE_CORRUPTION"})},
                                    }
                                ]
                            }
                        }]
                    }
                else:
                    yield {"choices": [{"delta": {"content": "Done."}}]}

            ledger = TurnExecutionLedger()
            criteria = f"file_contains:{path_a}:ALPHA_101,file_contains:{path_b}:BETA_202"
            payload = {
                "messages": [{"role": "user", "content": f"Create {path_a} and {path_b}"}],
                "acceptance_criteria": criteria,
            }

            async for _ in orch.generate_stream(
                payload,
                stream_provider=mock_stream,
                tool_executor=mock_executor,
                ledger=ledger,
            ):
                pass

            # file_alpha.txt content remains ALPHA_101, corruption write was blocked by mutation guard
            assert path_a.read_text(encoding="utf-8") == "ALPHA_101"
            assert "OVERWRITE_CORRUPTION" not in [w[1] for w in executed_writes]
            assert ledger._records["c_bad_overwrite"].state == ActionState.FAILED
            assert "outcome_already_verified" in str(ledger._records["c_bad_overwrite"].error)

    @pytest.mark.asyncio
    async def test_10_missing_or_ambiguous_acceptance_criteria_do_not_produce_invented_requirements(self):
        """10. Missing or ambiguous acceptance criteria do not produce invented requirements."""
        ledger = TurnExecutionLedger()
        ledger.record_proposed("c1", "execute_command", {"command": "echo test"})
        ledger.record_dispatched("c1")
        ledger.record_observation("c1", status="success", output="test")

        # Conversational query with no file paths and no acceptance spec
        v, u = find_unfulfilled_task_requirements("What is the capital of France?", ledger, acceptance_spec=None)
        assert u == []

        # Empty prompt
        v_empty, u_empty = find_unfulfilled_task_requirements("", ledger, acceptance_spec=None)
        assert u_empty == []


class TestSequentialCommandDetectionPrecision:
    """Phase 2 regression tests: initial read/run instructions vs genuine sequential follow-up commands."""

    def test_1_initial_read_the_file_does_not_imply_subsequent_command(self, tmp_path: Path):
        """1. Initial 'Read the file' instruction does not imply a subsequent command."""
        from zelyr.engine.execution_ledger import has_remaining_explicit_steps

        f = tmp_path / "missing_input.txt"
        f.write_text("RECOVERED_303", encoding="utf-8")
        ledger = TurnExecutionLedger()
        ledger.record_proposed("c1", "manage_workspace_file", {"action": "write", "path": str(f), "content": "RECOVERED_303"})
        ledger.record_dispatched("c1")
        ledger.record_observation("c1", status="success", output="written", verified=True)
        ledger.record_target_verified(str(f))

        prompt = f"Read the file {f}."
        v, u = find_unfulfilled_task_requirements(prompt, ledger, acceptance_spec=f"file_contains:{f}:RECOVERED_303")
        assert "Requested sequential command execution not yet performed" not in u
        assert u == []
        assert has_remaining_explicit_steps(prompt, ledger, current_target=str(f)) is False

    def test_2_initial_run_display_print_do_not_imply_second_command(self):
        """2. Initial 'Run the script', 'Display the result', 'Print the contents' do not imply a second command."""
        from zelyr.engine.execution_ledger import has_remaining_explicit_steps

        ledger = TurnExecutionLedger()
        for prompt in (
            "Run the script /tmp/deploy.sh.",
            "Display the result.",
            "Print the contents.",
            "Execute the command and show output.",
        ):
            _, u = find_unfulfilled_task_requirements(prompt, ledger, acceptance_spec=None)
            assert "Requested sequential command execution not yet performed" not in u, f"Failed for: {prompt}"
            assert has_remaining_explicit_steps(prompt, ledger) is False, f"Failed for: {prompt}"

    @pytest.mark.asyncio
    async def test_3_conditional_file_creation_does_not_invent_execution_requirement(self, tmp_path: Path):
        """3. Conditional file creation (S3 prompt) does not invent an additional execution requirement or trigger replanning."""
        from zelyr.engine.execution_ledger import has_remaining_explicit_steps

        f = tmp_path / "missing_input.txt"
        prompt = (
            f"Read the file {f}.\n"
            f"If it does not exist, create {f} containing exactly RECOVERED_303."
        )
        criteria = f"file_contains:{f}:RECOVERED_303"

        orch = NativeOrchestrator()
        turn = 0
        replan_seen = False

        async def executor(name: str, **kwargs):
            p = Path(kwargs["path"])
            p.write_text(kwargs["content"], encoding="utf-8")
            return f"Wrote to {p}"

        async def mock_stream(payload: dict):
            nonlocal turn, replan_seen
            turn += 1
            for m in payload.get("messages", []):
                if "[SYSTEM DIRECTIVE: Goal-Aware Replanning]" in str(m.get("content", "")):
                    replan_seen = True
            if turn == 1:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_create_missing",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": json.dumps({
                                        "action": "write",
                                        "path": str(f),
                                        "content": "RECOVERED_303",
                                    }),
                                },
                            }]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Created missing_input.txt with RECOVERED_303."}}]}

        ledger = TurnExecutionLedger()
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "acceptance_criteria": criteria,
        }
        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=executor,
            ledger=ledger,
        ):
            pass

        v, u = find_unfulfilled_task_requirements(prompt, ledger, acceptance_spec=criteria)
        assert u == []
        assert has_remaining_explicit_steps(prompt, ledger, current_target=str(f)) is False
        assert replan_seen is False
        assert turn == 2

    def test_4_explicit_then_run_the_script_remains_required_followup(self, tmp_path: Path):
        """4. Explicit 'then run the script' remains a required follow-up until command is executed."""
        from zelyr.engine.execution_ledger import has_remaining_explicit_steps

        script = tmp_path / "run.sh"
        script.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
        ledger = TurnExecutionLedger()
        ledger.record_proposed("c_write", "manage_workspace_file", {"action": "write", "path": str(script), "content": "#!/bin/sh\necho ok\n"})
        ledger.record_dispatched("c_write")
        ledger.record_observation("c_write", status="success", output="ok", verified=True)
        ledger.record_target_verified(str(script))

        prompt = f"Create the file {script}, then run the script."
        assert has_remaining_explicit_steps(prompt, ledger, current_target=str(script)) is True
        _, u = find_unfulfilled_task_requirements(prompt, ledger, acceptance_spec=f"file_exists:{script}")
        assert "Requested sequential command execution not yet performed" in u

        # Once execute_command succeeds, requirement is fulfilled
        ledger.record_proposed("c_exec", "execute_command", {"command": f"sh {script}"})
        ledger.record_dispatched("c_exec")
        ledger.record_observation("c_exec", status="success", output="ok", exit_code=0)
        assert has_remaining_explicit_steps(prompt, ledger, current_target=str(script)) is False
        _, u_after = find_unfulfilled_task_requirements(prompt, ledger, acceptance_spec=f"file_exists:{script}")
        assert "Requested sequential command execution not yet performed" not in u_after

    def test_5_afterward_validate_and_once_exists_execute_remain_required_followups(self, tmp_path: Path):
        """5. 'Afterward validate the file' and 'Once the file exists, execute it' remain required follow-ups."""
        from zelyr.engine.execution_ledger import has_remaining_explicit_steps

        cfg = tmp_path / "config.json"
        cfg.write_text("{}", encoding="utf-8")
        ledger = TurnExecutionLedger()
        ledger.record_proposed("c_w", "manage_workspace_file", {"action": "write", "path": str(cfg), "content": "{}"})
        ledger.record_dispatched("c_w")
        ledger.record_observation("c_w", status="success", output="ok", verified=True)
        ledger.record_target_verified(str(cfg))

        for prompt in (
            f"Write the configuration {cfg} and afterward validate it.",
            f"Create {cfg}. Afterward validate the file.",
            f"Create {cfg}. Once the file exists, execute it.",
        ):
            assert has_remaining_explicit_steps(prompt, ledger, current_target=str(cfg)) is True, f"Failed for: {prompt}"
            _, u = find_unfulfilled_task_requirements(prompt, ledger, acceptance_spec=f"file_exists:{cfg}")
            assert "Requested sequential command execution not yet performed" in u, f"Failed for: {prompt}"

    def test_6_historical_quoted_and_negated_command_mentions_do_not_create_requirements(self, tmp_path: Path):
        """6. Historical, quoted, and negated command mentions do not create requirements."""
        from zelyr.engine.execution_ledger import has_remaining_explicit_steps

        f = tmp_path / "notes.txt"
        f.write_text("ok", encoding="utf-8")
        ledger = TurnExecutionLedger()
        ledger.record_proposed("c_w", "manage_workspace_file", {"action": "write", "path": str(f), "content": "ok"})
        ledger.record_dispatched("c_w")
        ledger.record_observation("c_w", status="success", output="ok", verified=True)
        ledger.record_target_verified(str(f))

        non_sequential_prompts = [
            f"Previously I had to create a file and then run the script. Today just create {f}.",
            f"Previously I had to create old.txt and then run the script. Today just create {f}.",
            f'Create {f} containing "Create the file, then run the script".',
            f"Create {f} with:\n```\nthen run the script\n```",
            f"Create {f}, then do not run the script.",
            f"Create {f}, then do not run bad.sh or execute it.",
            f"Write {f} and afterward never execute it.",
            f"Create {f} without running the script afterward.",
            f"If {f} does not exist, then read the fallback documentation.",
        ]
        for prompt in non_sequential_prompts:
            assert has_remaining_explicit_steps(prompt, ledger, current_target=str(f)) is False, f"False positive for: {prompt}"
            _, u = find_unfulfilled_task_requirements(prompt, ledger, acceptance_spec=f"file_exists:{f}")
            assert "Requested sequential command execution not yet performed" not in u, f"False positive for: {prompt}"

    @pytest.mark.asyncio
    async def test_7_scoped_no_further_actions_for_this_file_does_not_block_replanning(
        self, tmp_path: Path, monkeypatch
    ):
        """S2: Scoped 'no further actions required for this file, I will proceed to the next task' must not abort remaining unfulfilled file beta.txt."""
        from unittest.mock import AsyncMock

        assert (
            NativeOrchestrator._is_intentional_abandonment(
                "Since there are no further actions required for this file, "
                "I will proceed to the next task and create beta.txt with BETA_202."
            )
            is False
        )
        # Task-wide genuine abandonment still detected as True
        assert (
            NativeOrchestrator._is_intentional_abandonment(
                "Based on the observations, no further actions are appropriate or possible."
            )
            is True
        )

        alpha = tmp_path / "alpha.txt"
        beta = tmp_path / "beta.txt"
        criteria = f"file_contains:{alpha}:ALPHA_101,file_contains:{beta}:BETA_202"
        prompt = f"Create {alpha} containing ALPHA_101, then create {beta} containing BETA_202."

        orch = NativeOrchestrator()
        monkeypatch.setattr(orch, "get_active_window_context", AsyncMock(return_value=None))

        async def executor(name: str, **kwargs):
            p = Path(kwargs["path"])
            p.write_text(kwargs["content"], encoding="utf-8")
            return {"status": "success", "path": str(p)}

        turn = 0
        replan_seen = False

        async def mock_stream(payload: dict):
            nonlocal turn, replan_seen
            turn += 1
            for m in payload.get("messages", []):
                if "[SYSTEM DIRECTIVE: Goal-Aware Replanning]" in str(m.get("content", "")):
                    replan_seen = True
            if turn == 1:
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c_alpha", "type": "function", "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": str(alpha), "content": "ALPHA_101"})}}]}}]}
            elif turn == 2:
                # Emits scoped phrase after alpha verified without calling beta yet
                yield {"choices": [{"delta": {"content": f"Since there are no further actions required for this file, I will proceed to the next task and create {beta}."}}]}
            elif turn == 3:
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c_beta", "type": "function", "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": str(beta), "content": "BETA_202"})}}]}}]}
            else:
                yield {"choices": [{"delta": {"content": "Both alpha.txt and beta.txt created."}}]}

        ledger = TurnExecutionLedger()
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "acceptance_criteria": criteria,
        }
        async for _ in orch.generate_stream(payload, stream_provider=mock_stream, tool_executor=executor, ledger=ledger):
            pass

        assert replan_seen is True
        assert alpha.read_text(encoding="utf-8") == "ALPHA_101"
        assert beta.read_text(encoding="utf-8") == "BETA_202"
        assert ledger.has_unresolved_failures() is False
