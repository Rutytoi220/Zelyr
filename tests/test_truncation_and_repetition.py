"""Hermetic regression tests for model response truncation, repeated tool execution, and verification semantics."""

import asyncio
import json
from pathlib import Path
import tempfile
from typing import Any, AsyncGenerator, Dict, List
import pytest

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.cli.main import _execute_headless_run
from zelyr.engine.execution_ledger import ActionState, TurnExecutionLedger
from zelyr.engine.verifier import OutcomeVerifier, VerificationResult, verify_file_exists, verify_outcome


class MockStreamProvider:
    """Configurable mock stream provider for deterministic streaming tests."""

    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks

    async def __call__(self, payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        for c in self.chunks:
            yield c


@pytest.mark.asyncio
class TestResponseTruncationHandling:
    """Test response truncation handling and finish_reason preservation."""

    async def test_truncation_during_reasoning_yields_error_and_fails(self):
        """When generation reaches token length limit during reasoning with no content, report error."""
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "Let me think about how to solve this..."}, "finish_reason": None}]},
            {"choices": [{"delta": {"reasoning_content": " still thinking..."}, "finish_reason": "length"}]},
        ]
        provider = MockStreamProvider(chunks)
        orch = NativeOrchestrator(stream_provider=provider)

        events = []
        async for chunk in orch.generate_stream({"messages": [{"role": "user", "content": "test"}]}):
            events.append(chunk)

        # Confirm last_finish_reason is preserved
        assert orch.last_finish_reason == "length"
        # Confirm truncation error message was yielded
        joined_content = "".join(
            e.get("choices", [{}])[0].get("delta", {}).get("content", "")
            for e in events
        )
        assert "⚠️ [Model Error] Model generation truncated: token limit reached" in joined_content

    async def test_headless_run_exits_nonzero_on_empty_reasoning_truncation(self):
        """CLI headless run must return nonzero exit code (1) when response is truncated during thinking."""
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "Deep thought..."}, "finish_reason": "length"}]},
        ]
        provider = MockStreamProvider(chunks)
        orch = NativeOrchestrator(stream_provider=provider)

        exit_code = await _execute_headless_run(
            prompt="Analyze something complex",
            orchestrator=orch,
        )
        assert exit_code == 1

    async def test_headless_run_preserves_partial_output_on_length_exhaustion(self, capsys):
        """CLI headless run preserves useful partial output but returns nonzero exit code (1) on length truncation."""
        chunks = [
            {"choices": [{"delta": {"content": "Here is the partial solution: step 1 is done"}, "finish_reason": "length"}]},
        ]
        provider = MockStreamProvider(chunks)
        orch = NativeOrchestrator(stream_provider=provider)

        exit_code = await _execute_headless_run(
            prompt="Tell me a long story",
            orchestrator=orch,
        )
        assert exit_code == 1
        captured = capsys.readouterr()
        # Partial output is preserved on stdout
        assert "Here is the partial solution: step 1 is done" in captured.out
        # Warning is emitted to stderr
        assert "⚠️ [Model Truncated]" in captured.err

    async def test_tool_call_emitted_before_truncation_is_not_discarded(self):
        """If tool calls are produced, length exhaustion does not discard the actionable tool call."""
        chunks = [
            {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_trunc_tool",
                            "type": "function",
                            "function": {
                                "name": "manage_workspace_file",
                                "arguments": '{"action": "read", "path": "/tmp/test.txt"}',
                            },
                        }]
                    },
                    "finish_reason": "length",
                }]
            }
        ]
        provider = MockStreamProvider(chunks)
        executed_calls = []

        async def mock_executor(tool_name: str, args: Dict[str, Any]):
            executed_calls.append((tool_name, args))
            return "File content verified"

        orch = NativeOrchestrator(
            stream_provider=provider,
            tool_executor=mock_executor,
            max_steps=1,
        )

        async for _ in orch.generate_stream({"messages": [{"role": "user", "content": "Read file"}]}):
            pass

        assert len(executed_calls) == 1
        assert executed_calls[0][0] == "manage_workspace_file"


@pytest.mark.asyncio
class TestRepeatedSuccessfulActionSuppression:
    """Test prevention of unnecessary repeated tool execution."""

    async def test_repeated_identical_action_after_success_is_blocked(self):
        """An identical tool call immediately following success is rejected as redundant."""
        # Turn 1: tool call
        # Turn 2: identical tool call
        # Turn 3: final answer
        turn_counter = 0

        async def alternating_provider(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
            nonlocal turn_counter
            turn_counter += 1
            if turn_counter == 1:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_write_1",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": '{"action": "write", "path": "/tmp/demo.txt", "content": "alpha"}',
                                },
                            }]
                        }
                    }]
                }
            elif turn_counter == 2:
                # Model attempts identical tool call again
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_write_2",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": '{"action": "write", "path": "/tmp/demo.txt", "content": "alpha"}',
                                },
                            }]
                        }
                    }]
                }
            else:
                yield {
                    "choices": [{
                        "delta": {"content": "File /tmp/demo.txt has been written successfully with alpha."}
                    }]
                }

        executed_tools = []

        async def mock_executor(name: str, args: Dict[str, Any]):
            executed_tools.append((name, args))
            return "File created"

        ledger = TurnExecutionLedger()
        orch = NativeOrchestrator(
            stream_provider=alternating_provider,
            tool_executor=mock_executor,
            enable_cycle_detection=True,
        )

        payload = {
            "messages": [{"role": "user", "content": "Create /tmp/demo.txt with alpha"}],
        }

        events = []
        async for chunk in orch.generate_stream(payload, ledger=ledger):
            events.append(chunk)

        # Host tool was executed strictly ONCE
        assert len(executed_tools) == 1
        # Ledger recorded first as SUCCESS and has no unresolved failures
        assert not ledger.has_unresolved_failures()
        # Confirm model received rejection message informing that action already succeeded
        tool_messages = [m for m in payload["messages"] if m.get("role") == "tool"]
        assert len(tool_messages) >= 2
        rejected_msg = tool_messages[1]["content"]
        assert "redundant_action" in rejected_msg
        assert "already succeeded" in rejected_msg

    async def test_explicit_repeat_directive_allows_repetition(self):
        """When user prompt explicitly includes repeat directives, redundant action rejection is bypassed."""
        turn_counter = 0

        async def alternating_provider(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
            nonlocal turn_counter
            turn_counter += 1
            if turn_counter in (1, 2):
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": f"call_repeat_{turn_counter}",
                                "type": "function",
                                "function": {
                                    "name": "execute_command",
                                    "arguments": '{"command": "echo hello"}',
                                },
                            }]
                        }
                    }]
                }
            else:
                yield {
                    "choices": [{
                        "delta": {"content": "Executed command twice."}
                    }]
                }

        executed = []

        async def mock_executor(name: str, args: Dict[str, Any]):
            executed.append((name, args))
            return "hello"

        ledger = TurnExecutionLedger()
        orch = NativeOrchestrator(
            stream_provider=alternating_provider,
            tool_executor=mock_executor,
            enable_cycle_detection=True,
        )

        payload = {
            "messages": [{"role": "user", "content": "Please repeat the echo command twice"}],
        }

        async for _ in orch.generate_stream(payload, ledger=ledger):
            pass

        # Since user explicitly asked to "repeat ... twice", both executions were allowed
        assert len(executed) == 2

    async def test_negated_repeat_directive_does_not_authorize_repetition(self):
        """Negated instructions such as 'do not repeat' or 'never repeat' must not authorize duplicate execution."""
        turn_counter = 0

        async def alternating_provider(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
            nonlocal turn_counter
            turn_counter += 1
            if turn_counter in (1, 2):
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": f"call_neg_{turn_counter}",
                                "type": "function",
                                "function": {
                                    "name": "execute_command",
                                    "arguments": '{"command": "date"}',
                                },
                            }]
                        }
                    }]
                }
            else:
                yield {
                    "choices": [{
                        "delta": {"content": "Finished."}
                    }]
                }

        executed = []

        async def mock_executor(name: str, args: Dict[str, Any]):
            executed.append((name, args))
            return "Mon Nov 9"

        ledger = TurnExecutionLedger()
        orch = NativeOrchestrator(
            stream_provider=alternating_provider,
            tool_executor=mock_executor,
            enable_cycle_detection=True,
        )

        payload = {
            "messages": [{"role": "user", "content": "Execute date once. Do not repeat this command or rerun it."}],
        }

        async for _ in orch.generate_stream(payload, ledger=ledger):
            pass

        # Because instruction was negated ("Do not repeat..."), the duplicate execution is blocked
        assert len(executed) == 1

    async def test_success_distinct_from_verified_state(self):
        """Execution ledger strictly preserves the distinction between SUCCESS and VERIFIED."""
        ledger = TurnExecutionLedger()
        ledger.record_proposed("call_1", "manage_workspace_file", {"action": "write", "path": "test.txt"})
        ledger.record_dispatched("call_1")
        # Exit code 0 without independent verification
        ledger.record_observation("call_1", status="success", output="File written", exit_code=0, verified=False)

        rec = ledger._records["call_1"]
        assert rec.state == ActionState.SUCCESS
        assert rec.verified is False
        assert rec not in ledger.verified_actions()
        assert rec in ledger.unverified_actions()

        # Now record an outcome verification
        ledger.record_proposed("call_2", "verify_outcome", {"check": "file_exists", "target": "test.txt"})
        ledger.record_dispatched("call_2")
        ledger.record_observation("call_2", status="success", output="[VERIFIED] file_exists: PASS", verified=True)

        rec_v = ledger._records["call_2"]
        assert rec_v.state == ActionState.VERIFIED
        assert rec_v.verified is True
        assert rec_v in ledger.verified_actions()


class TestOutcomeVerifierSpecificationSemantics:
    """Test strict verification specifications in OutcomeVerifier."""

    def test_file_exists_rejects_expected_content_parameter(self):
        """file_exists must fail with clear error when expected content is provided."""
        res = OutcomeVerifier.verify(
            check="file_exists",
            target="/tmp/some_file.txt",
            expected="some expected content",
        )
        assert res.success is False
        assert "Incompatible verification specification" in str(res.error)
        assert "file_contains" in str(res.error)

    def test_direct_verify_file_exists_rejects_expected_content(self):
        """verify_file_exists called directly rejects expected argument."""
        res = verify_file_exists("/tmp/some_file.txt", expected="content")
        assert res.success is False
        assert "Incompatible verification specification" in str(res.error)

    def test_file_exists_succeeds_without_expected(self):
        """file_exists succeeds when file exists and expected is None."""
        with tempfile.NamedTemporaryFile() as tmp:
            res = OutcomeVerifier.verify(check="file_exists", target=tmp.name)
            assert res.success is True
            assert res.observed.get("exists") is True

    def test_file_contains_requires_expected_pattern(self):
        """file_contains fails when expected pattern is omitted."""
        with tempfile.NamedTemporaryFile() as tmp:
            res = OutcomeVerifier.verify(check="file_contains", target=tmp.name, expected=None)
            assert res.success is False
            assert "Incomplete verification specification" in str(res.error)

    def test_file_contains_verifies_content_accurately(self):
        """file_contains correctly checks file contents."""
        with tempfile.NamedTemporaryFile("w+", delete=False) as tmp:
            tmp.write("Zelyr 1.4 Model Independence")
            tmp_path = tmp.name

        try:
            res_pass = OutcomeVerifier.verify(check="file_contains", target=tmp_path, expected="Model Independence")
            assert res_pass.success is True

            res_fail = OutcomeVerifier.verify(check="file_contains", target=tmp_path, expected="Nonexistent Text")
            assert res_fail.success is False
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_cli_verify_incompatible_file_exists_returns_exit_code_2(self):
        """CLI with --verify file_exists:path:expected must fail verification and exit with code 2."""
        with tempfile.NamedTemporaryFile("w+", delete=False) as tmp:
            tmp.write("Hello")
            tmp_path = tmp.name

        chunks = [{"choices": [{"delta": {"content": "File ready"}, "finish_reason": "stop"}]}]
        provider = MockStreamProvider(chunks)
        orch = NativeOrchestrator(stream_provider=provider)

        try:
            exit_code = await _execute_headless_run(
                prompt="Check file",
                verify_spec=f"file_exists:{tmp_path}:Hello",
                orchestrator=orch,
            )
            assert exit_code == 2
        finally:
            Path(tmp_path).unlink(missing_ok=True)


@pytest.mark.asyncio
class TestAcceptanceAwareMutationControl:
    """Hermetic tests for preventing regression of verified outcomes."""

    async def test_verified_file_blocks_destructive_mutation_with_changed_arguments(self, tmp_path):
        """When an outcome satisfies explicit acceptance criteria, subsequent mutations to that target are blocked."""
        target_file = str(tmp_path / "test_regress.txt")
        turn_counter = 0

        async def provider(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
            nonlocal turn_counter
            turn_counter += 1
            if turn_counter == 1:
                # Step 1: Write correct content
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_write_1",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": json.dumps({"action": "write", "path": target_file, "content": "correct token 4412"}),
                                },
                            }]
                        }
                    }]
                }
            elif turn_counter == 2:
                # Step 2: Model attempts to mutate with changed arguments
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_write_2",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": json.dumps({"action": "write", "path": target_file, "content": "destructive overwrite"}),
                                },
                            }]
                        }
                    }]
                }
            else:
                yield {
                    "choices": [{"delta": {"content": "Task is completed and verified."}}]
                }

        executed_writes = []

        async def mock_executor(name: str, args: Dict[str, Any]):
            if name == "manage_workspace_file" and args.get("action") == "write":
                executed_writes.append(args.get("content"))
                Path(args["path"]).write_text(args.get("content", ""))
                return f"Successfully wrote {len(args.get('content', ''))} bytes to {args['path']}."
            return "ok"

        ledger = TurnExecutionLedger()
        orch = NativeOrchestrator(
            stream_provider=provider,
            tool_executor=mock_executor,
            enable_cycle_detection=True,
        )

        prompt = f"Write a file named {target_file} containing exactly: correct token 4412. Then verify it."
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "acceptance_criteria": f"file_contains:{target_file}:correct token 4412",
        }

        async for _ in orch.generate_stream(payload, ledger=ledger):
            pass

        # Execution executor was called strictly once for the correct write
        assert len(executed_writes) == 1
        assert executed_writes[0] == "correct token 4412"

        # Content on disk was preserved and never overwritten
        assert Path(target_file).read_text() == "correct token 4412"

        # The second attempt was rejected with outcome_already_verified
        tool_messages = [m for m in payload["messages"] if m.get("role") == "tool"]
        assert len(tool_messages) >= 2
        rejected_content = tool_messages[1]["content"]
        assert "outcome_already_verified" in rejected_content

        # Ledger does not have unresolved failures
        assert not ledger.has_unresolved_failures()

    async def test_verified_file_blocks_identical_repeated_mutation(self, tmp_path):
        """Identical repeated writes to an already verified resource are suppressed."""
        target_file = str(tmp_path / "test_dup.txt")
        turn_counter = 0

        async def provider(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
            nonlocal turn_counter
            turn_counter += 1
            if turn_counter in (1, 2):
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": f"call_dup_{turn_counter}",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": json.dumps({"action": "write", "path": target_file, "content": "hello"}),
                                },
                            }]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Done."}}]}

        executed_writes = []

        async def mock_executor(name: str, args: Dict[str, Any]):
            executed_writes.append(args)
            Path(args["path"]).write_text(args.get("content", ""))
            return "ok"

        ledger = TurnExecutionLedger()
        orch = NativeOrchestrator(
            stream_provider=provider,
            tool_executor=mock_executor,
            enable_cycle_detection=True,
        )

        payload = {
            "messages": [{"role": "user", "content": f"Write {target_file} with hello"}],
            "acceptance_criteria": f"file_contains:{target_file}:hello",
        }

        async for _ in orch.generate_stream(payload, ledger=ledger):
            pass

        assert len(executed_writes) == 1

    async def test_multi_file_task_allows_distinct_files(self, tmp_path):
        """Legitimate multi-file tasks are not blocked after the first file is created."""
        file1 = str(tmp_path / "file1.txt")
        file2 = str(tmp_path / "file2.txt")
        turn_counter = 0

        async def provider(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
            nonlocal turn_counter
            turn_counter += 1
            if turn_counter == 1:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_f1",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": json.dumps({"action": "write", "path": file1, "content": "part 1"}),
                                },
                            }]
                        }
                    }]
                }
            elif turn_counter == 2:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_f2",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": json.dumps({"action": "write", "path": file2, "content": "part 2"}),
                                },
                            }]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Both files written."}}]}

        created_files = []

        async def mock_executor(name: str, args: Dict[str, Any]):
            p = args["path"]
            Path(p).write_text(args.get("content", ""))
            created_files.append(p)
            return "ok"

        ledger = TurnExecutionLedger()
        orch = NativeOrchestrator(
            stream_provider=provider,
            tool_executor=mock_executor,
            enable_cycle_detection=True,
        )

        prompt = f"Create file {file1} with part 1 and file {file2} with part 2"
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "acceptance_criteria": f"file_contains:{file1}:part 1",
        }

        async for _ in orch.generate_stream(payload, ledger=ledger):
            pass

        assert len(created_files) == 2
        assert file1 in created_files
        assert file2 in created_files
        assert Path(file1).read_text() == "part 1"
        assert Path(file2).read_text() == "part 2"

    async def test_explicit_repetition_allows_repeated_mutations(self, tmp_path):
        """When user prompt explicitly requests repetition, mutations to the same target are permitted."""
        target_file = str(tmp_path / "append.log")
        turn_counter = 0

        async def provider(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
            nonlocal turn_counter
            turn_counter += 1
            if turn_counter in (1, 2):
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": f"call_app_{turn_counter}",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": json.dumps({"action": "append", "path": target_file, "content": "entry\n"}),
                                },
                            }]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Appended twice."}}]}

        appends = []

        async def mock_executor(name: str, args: Dict[str, Any]):
            appends.append(args)
            with open(args["path"], "a") as f:
                f.write(args.get("content", ""))
            return "ok"

        ledger = TurnExecutionLedger()
        orch = NativeOrchestrator(
            stream_provider=provider,
            tool_executor=mock_executor,
            enable_cycle_detection=True,
        )

        prompt = f"Append entry to {target_file} twice."
        payload = {
            "messages": [{"role": "user", "content": prompt}],
        }

        async for _ in orch.generate_stream(payload, ledger=ledger):
            pass

        assert len(appends) == 2
        assert Path(target_file).read_text() == "entry\nentry\n"

    async def test_failed_action_allows_recovery_mutation(self, tmp_path):
        """When an initial mutation fails, the model is permitted to retry and recover."""
        target_file = str(tmp_path / "recover.txt")
        turn_counter = 0

        async def provider(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
            nonlocal turn_counter
            turn_counter += 1
            if turn_counter == 1:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_fail_1",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": json.dumps({"action": "write", "path": target_file, "content": "initial"}),
                                },
                            }]
                        }
                    }]
                }
            elif turn_counter == 2:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_retry_2",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": json.dumps({"action": "write", "path": target_file, "content": "recovered"}),
                                },
                            }]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Recovered successfully."}}]}

        attempts = []

        async def mock_executor(name: str, args: Dict[str, Any]):
            attempts.append(args.get("content"))
            if len(attempts) == 1:
                raise IOError("Simulated disk error")
            Path(args["path"]).write_text(args.get("content", ""))
            return "ok"

        ledger = TurnExecutionLedger()
        orch = NativeOrchestrator(
            stream_provider=provider,
            tool_executor=mock_executor,
            enable_cycle_detection=True,
        )

        prompt = f"Write {target_file} with recovered"
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "acceptance_criteria": f"file_contains:{target_file}:recovered",
        }

        async for _ in orch.generate_stream(payload, ledger=ledger):
            pass

        assert len(attempts) == 2
        assert Path(target_file).read_text() == "recovered"
        assert not ledger.has_unresolved_failures()

    async def test_unfulfilled_acceptance_criteria_allows_correction(self, tmp_path):
        """When initial action succeeds but does NOT satisfy acceptance criteria, correction is permitted."""
        target_file = str(tmp_path / "criteria.txt")
        turn_counter = 0

        async def provider(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
            nonlocal turn_counter
            turn_counter += 1
            if turn_counter == 1:
                # Writes incorrect content
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_wrong",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": json.dumps({"action": "write", "path": target_file, "content": "wrong token"}),
                                },
                            }]
                        }
                    }]
                }
            elif turn_counter == 2:
                # Corrects content
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_right",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": json.dumps({"action": "write", "path": target_file, "content": "correct token"}),
                                },
                            }]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "File corrected."}}]}

        writes = []

        async def mock_executor(name: str, args: Dict[str, Any]):
            writes.append(args.get("content"))
            Path(args["path"]).write_text(args.get("content", ""))
            return f"Wrote {args.get('content')}"

        ledger = TurnExecutionLedger()
        orch = NativeOrchestrator(
            stream_provider=provider,
            tool_executor=mock_executor,
            enable_cycle_detection=True,
        )

        prompt = f"Write {target_file} containing correct token"
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "acceptance_criteria": f"file_contains:{target_file}:correct token",
        }

        async for _ in orch.generate_stream(payload, ledger=ledger):
            pass

        assert len(writes) == 2
        assert Path(target_file).read_text() == "correct token"

    async def test_verified_action_with_remaining_explicit_steps_allows_subsequent_step(self, tmp_path):
        """When acceptance criteria is verified but subsequent explicit commands remain, they are executed."""
        target_script = str(tmp_path / "script.sh")
        turn_counter = 0

        async def provider(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
            nonlocal turn_counter
            turn_counter += 1
            if turn_counter == 1:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_write_sh",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": json.dumps({"action": "write", "path": target_script, "content": "echo executed"}),
                                },
                            }]
                        }
                    }]
                }
            elif turn_counter == 2:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_run_cmd",
                                "type": "function",
                                "function": {
                                    "name": "execute_command",
                                    "arguments": json.dumps({"command": f"bash {target_script}"}),
                                },
                            }]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Script created and executed."}}]}

        executed_actions = []

        async def mock_executor(name: str, args: Dict[str, Any]):
            executed_actions.append(name)
            if name == "manage_workspace_file":
                Path(args["path"]).write_text(args.get("content", ""))
            return "executed"

        ledger = TurnExecutionLedger()
        orch = NativeOrchestrator(
            stream_provider=provider,
            tool_executor=mock_executor,
            enable_cycle_detection=True,
        )

        prompt = f"Write script {target_script} and then run it"
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "acceptance_criteria": f"file_exists:{target_script}",
        }

        async for _ in orch.generate_stream(payload, ledger=ledger):
            pass

        assert "manage_workspace_file" in executed_actions
        assert "execute_command" in executed_actions

    async def test_cli_headless_run_returns_exit_code_2_on_failed_verification(self, tmp_path):
        """When outcome verification fails in CLI headless run, it returns exit code 2."""
        target_file = str(tmp_path / "unmatched.txt")
        Path(target_file).write_text("actual text")

        chunks = [{"choices": [{"delta": {"content": "I have created the file."}}]}]
        provider = MockStreamProvider(chunks)
        orch = NativeOrchestrator(stream_provider=provider)

        exit_code = await _execute_headless_run(
            prompt="Write file",
            verify_spec=f"file_contains:{target_file}:expected text not present",
            orchestrator=orch,
        )
        assert exit_code == 2
