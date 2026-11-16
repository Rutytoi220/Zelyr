"""Adversarial audit tests for Acceptance-Aware Mutation Control, Target Identity, and Step Detection."""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List
import pytest

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.cli.main import _execute_headless_run
from zelyr.engine.execution_ledger import (
    ActionState,
    TurnExecutionLedger,
    canonicalize_target,
    extract_file_targets_from_text,
    has_remaining_explicit_steps,
    parse_acceptance_criteria,
)
from zelyr.engine.verifier import OutcomeVerifier


class MockStreamProvider:
    def __init__(self, turns_or_chunks: List[Any]):
        self.turns_or_chunks = turns_or_chunks
        self.turn_index = 0

    async def __call__(self, payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        if self.turn_index < len(self.turns_or_chunks):
            chunk_or_list = self.turns_or_chunks[self.turn_index]
            self.turn_index += 1
            if isinstance(chunk_or_list, list):
                for c in chunk_or_list:
                    yield c
            else:
                yield chunk_or_list
        else:
            yield {"choices": [{"delta": {"content": "Finished all work."}}]}


class TestAdversarialNaturalLanguageStepDetector:
    """Audit Section 2: 12 challenge cases for natural-language step detection & repetition."""

    def test_case_6_prompt_containing_again_in_unrelated_explanation(self):
        """Case 6: Unrelated 'again' (e.g. 'Time and time again...') must NOT trigger is_explicit_repeat."""
        prompt = "Time and time again users report corrupted files. Please write /tmp/test.txt with healthy data."
        is_repeat = NativeOrchestrator._is_explicit_repeat_requested(prompt)
        assert is_repeat is False, "Unrelated conversational 'again' must not authorize repetition!"

    def test_case_7_negated_repetition(self):
        """Case 7: 'Do not repeat this operation' must NOT trigger is_explicit_repeat."""
        prompt = "Write /tmp/test.txt with abc. Do not repeat this operation or retry."
        is_repeat = NativeOrchestrator._is_explicit_repeat_requested(prompt)
        assert is_repeat is False, "Negated repetition directive must not authorize repetition!"

    def test_sequential_operation_explicitly_requested(self):
        """Sequential operation explicitly requested ('Create the file, then append another line') -> authorized."""
        prompt = "Create the file, then append another line"
        ledger = TurnExecutionLedger()
        assert NativeOrchestrator._is_sequential_step_requested(prompt, "/tmp/notes.txt", "append", ledger) is True

    def test_sequential_operation_incidental_mention_rejected(self):
        """Sequential operation incidental mention ('The word append appears in this unrelated explanation') -> rejected."""
        prompt = "Create a file. The word append appears in this unrelated explanation"
        ledger = TurnExecutionLedger()
        assert NativeOrchestrator._is_sequential_step_requested(prompt, "/tmp/notes.txt", "append", ledger) is False

    def test_explicitly_prohibited_subsequent_mutation_rejected(self):
        """Explicitly prohibited subsequent mutation ('Create a file, but do not modify it afterward') -> rejected."""
        prompt = "Create a file, but do not modify it afterward"
        ledger = TurnExecutionLedger()
        assert NativeOrchestrator._is_sequential_step_requested(prompt, "/tmp/notes.txt", "modify", ledger) is False

    def test_sequential_coordination_conjunction_authorized(self):
        """Sequential coordination conjunction ('Write the file and subsequently update it') -> authorized."""
        prompt = "Write the file and subsequently update it"
        ledger = TurnExecutionLedger()
        assert NativeOrchestrator._is_sequential_step_requested(prompt, "/tmp/notes.txt", "update", ledger) is True

    def test_extract_file_targets_no_phantom_paths_or_trailing_dots(self):
        """extract_file_targets_from_text must not extract phantom cwd paths or trailing punctuation."""
        prompt = "Write /tmp/data/test.txt. Then verify it."
        targets = extract_file_targets_from_text(prompt)
        assert len(targets) == 1
        assert targets[0] == str(Path("/tmp/data/test.txt").resolve())

    @pytest.mark.asyncio
    async def test_case_1_create_two_distinct_files(self, tmp_path):
        """Case 1: Create two distinct files with different contents."""
        f1 = str(tmp_path / "f1.txt")
        f2 = str(tmp_path / "f2.txt")
        turn_counter = 0

        async def provider(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
            nonlocal turn_counter
            turn_counter += 1
            if turn_counter == 1:
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "type": "function", "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": f1, "content": "alpha"})}}]}}]}
            elif turn_counter == 2:
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c2", "type": "function", "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": f2, "content": "beta"})}}]}}]}
            else:
                yield {"choices": [{"delta": {"content": "Both files created."}}]}

        created = []
        async def mock_executor(name: str, args: Dict[str, Any]):
            Path(args["path"]).write_text(args["content"])
            created.append(args["path"])
            return "ok"

        orch = NativeOrchestrator(stream_provider=provider, tool_executor=mock_executor, enable_cycle_detection=True)
        payload = {
            "messages": [{"role": "user", "content": f"Create {f1} with alpha and {f2} with beta"}],
            "acceptance_criteria": [f"file_contains:{f1}:alpha", f"file_contains:{f2}:beta"],
        }
        async for _ in orch.generate_stream(payload):
            pass

        assert len(created) == 2
        assert Path(f1).read_text() == "alpha"
        assert Path(f2).read_text() == "beta"

    @pytest.mark.asyncio
    async def test_case_2_create_file_then_execute_command_reading_it(self, tmp_path):
        """Case 2: Create a file, then execute a command that reads it (e.g. cat)."""
        target_file = str(tmp_path / "hello.txt")
        turn_counter = 0

        async def provider(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
            nonlocal turn_counter
            turn_counter += 1
            if turn_counter == 1:
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": target_file, "content": "world"})}}]}}]}
            elif turn_counter == 2:
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_2", "type": "function", "function": {"name": "execute_command", "arguments": json.dumps({"command": f"cat {target_file}"})}}]}}]}
            else:
                yield {"choices": [{"delta": {"content": "File created and read successfully."}}]}

        executed = []
        async def mock_executor(name: str, args: Dict[str, Any]):
            executed.append(name)
            if name == "manage_workspace_file":
                Path(args["path"]).write_text(args.get("content", ""))
                return "File written"
            elif name == "execute_command":
                return "world"
            return "ok"

        orch = NativeOrchestrator(stream_provider=provider, tool_executor=mock_executor, enable_cycle_detection=True)
        prompt = f"Create a file {target_file} containing 'world', then read it using cat."
        payload = {"messages": [{"role": "user", "content": prompt}], "acceptance_criteria": f"file_contains:{target_file}:world"}

        async for _ in orch.generate_stream(payload):
            pass

        assert "manage_workspace_file" in executed
        assert "execute_command" in executed, "Subsequent command (cat) must not be suppressed!"

    @pytest.mark.asyncio
    async def test_case_3_create_file_then_modify_as_requested_second_step(self, tmp_path):
        """Case 3: Create a file, then modify it as a separately requested second step."""
        target_file = str(tmp_path / "notes.txt")
        turn_counter = 0

        async def provider(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
            nonlocal turn_counter
            turn_counter += 1
            if turn_counter == 1:
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_write", "type": "function", "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": target_file, "content": "step 1"})}}]}}]}
            elif turn_counter == 2:
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_append", "type": "function", "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "append", "path": target_file, "content": " and step 2"})}}]}}]}
            else:
                yield {"choices": [{"delta": {"content": "Both steps completed."}}]}

        executed = []
        async def mock_executor(name: str, args: Dict[str, Any]):
            action = args.get("action")
            executed.append(action)
            p = Path(args["path"])
            if action == "write":
                p.write_text(args.get("content", ""))
            elif action == "append":
                with open(p, "a") as f:
                    f.write(args.get("content", ""))
            return "ok"

        orch = NativeOrchestrator(stream_provider=provider, tool_executor=mock_executor, enable_cycle_detection=True)
        prompt = f"Create {target_file} containing 'step 1'. Then append ' and step 2' to {target_file}."
        payload = {"messages": [{"role": "user", "content": prompt}], "acceptance_criteria": f"file_contains:{target_file}:step 1"}

        async for _ in orch.generate_stream(payload):
            pass

        assert executed == ["write", "append"], "Explicitly requested second-step append must not be blocked!"
        assert Path(target_file).read_text() == "step 1 and step 2"

    @pytest.mark.asyncio
    async def test_case_4_write_same_content_twice_when_requested(self, tmp_path):
        """Case 4: Write same content twice when explicitly requested."""
        target_file = str(tmp_path / "log.txt")
        turn_counter = 0

        async def provider(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
            nonlocal turn_counter
            turn_counter += 1
            if turn_counter in (1, 2):
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": f"c_{turn_counter}", "type": "function", "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": target_file, "content": "ping"})}}]}}]}
            else:
                yield {"choices": [{"delta": {"content": "Written twice."}}]}

        writes = []
        async def mock_executor(name: str, args: Dict[str, Any]):
            writes.append(args)
            return "ok"

        orch = NativeOrchestrator(stream_provider=provider, tool_executor=mock_executor, enable_cycle_detection=True)
        payload = {"messages": [{"role": "user", "content": f"Write ping to {target_file}. Repeat this write twice."}]}

        async for _ in orch.generate_stream(payload):
            pass

        assert len(writes) == 2, "Explicitly requested duplicate write must be permitted!"

    @pytest.mark.asyncio
    async def test_case_5_write_same_content_twice_unrequested_is_suppressed(self, tmp_path):
        """Case 5: Write same content twice when repetition was NOT requested is blocked."""
        target_file = str(tmp_path / "unreq.txt")
        turn_counter = 0

        async def provider(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
            nonlocal turn_counter
            turn_counter += 1
            if turn_counter in (1, 2):
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": f"c_{turn_counter}", "type": "function", "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": target_file, "content": "hello"})}}]}}]}
            else:
                yield {"choices": [{"delta": {"content": "Done."}}]}

        writes = []
        async def mock_executor(name: str, args: Dict[str, Any]):
            writes.append(args)
            Path(args["path"]).write_text(args.get("content", ""))
            return "ok"

        orch = NativeOrchestrator(stream_provider=provider, tool_executor=mock_executor, enable_cycle_detection=True)
        payload = {"messages": [{"role": "user", "content": f"Write hello to {target_file}"}], "acceptance_criteria": f"file_contains:{target_file}:hello"}

        async for _ in orch.generate_stream(payload):
            pass

        assert len(writes) == 1, "Unrequested duplicate write must be suppressed!"

    @pytest.mark.asyncio
    async def test_case_8_failed_write_followed_by_correction(self, tmp_path):
        """Case 8: A failed write followed by a correction must be allowed."""
        target_file = str(tmp_path / "fix.txt")
        turn_counter = 0

        async def provider(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
            nonlocal turn_counter
            turn_counter += 1
            if turn_counter == 1:
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "type": "function", "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": target_file, "content": "initial"})}}]}}]}
            elif turn_counter == 2:
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c2", "type": "function", "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": target_file, "content": "fixed"})}}]}}]}
            else:
                yield {"choices": [{"delta": {"content": "Fixed."}}]}

        attempts = []
        async def mock_executor(name: str, args: Dict[str, Any]):
            attempts.append(args.get("content"))
            if len(attempts) == 1:
                raise IOError("Write failed")
            Path(args["path"]).write_text(args["content"])
            return "ok"

        ledger = TurnExecutionLedger()
        orch = NativeOrchestrator(stream_provider=provider, tool_executor=mock_executor, enable_cycle_detection=True)
        payload = {"messages": [{"role": "user", "content": f"Write fixed to {target_file}"}], "acceptance_criteria": f"file_contains:{target_file}:fixed"}

        async for _ in orch.generate_stream(payload, ledger=ledger):
            pass

        assert len(attempts) == 2
        assert Path(target_file).read_text() == "fixed"
        assert not ledger.has_unresolved_failures()

    @pytest.mark.asyncio
    async def test_case_9_write_followed_by_unrelated_operation(self, tmp_path):
        """Case 9: A successful write followed by an unrelated required operation (e.g. restart daemon)."""
        target_file = str(tmp_path / "status.txt")
        turn_counter = 0

        async def provider(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
            nonlocal turn_counter
            turn_counter += 1
            if turn_counter == 1:
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_w", "type": "function", "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": target_file, "content": "active"})}}]}}]}
            elif turn_counter == 2:
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_cmd", "type": "function", "function": {"name": "execute_command", "arguments": json.dumps({"command": "systemctl restart my-daemon"})}}]}}]}
            else:
                yield {"choices": [{"delta": {"content": "File written and daemon restarted."}}]}

        executed_tools = []
        async def mock_executor(name: str, args: Dict[str, Any]):
            executed_tools.append(name)
            if name == "manage_workspace_file":
                Path(args["path"]).write_text(args.get("content", ""))
            return "ok"

        orch = NativeOrchestrator(stream_provider=provider, tool_executor=mock_executor, enable_cycle_detection=True)
        prompt = f"Write {target_file} with 'active'. Then restart the daemon."
        payload = {"messages": [{"role": "user", "content": prompt}], "acceptance_criteria": f"file_contains:{target_file}:active"}

        async for _ in orch.generate_stream(payload):
            pass

        assert "manage_workspace_file" in executed_tools
        assert "execute_command" in executed_tools, "Unrelated required operation (restart daemon) must be executed!"

    @pytest.mark.asyncio
    async def test_case_10_unnecessary_mutation_after_verified_success_is_blocked(self, tmp_path):
        """Case 10: Successful write followed by unnecessary mutation with changed arguments is blocked."""
        target_file = str(tmp_path / "stable.txt")
        turn_counter = 0

        async def provider(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
            nonlocal turn_counter
            turn_counter += 1
            if turn_counter == 1:
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "type": "function", "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": target_file, "content": "token 4412"})}}]}}]}
            elif turn_counter == 2:
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c2", "type": "function", "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": target_file, "content": "altered content"})}}]}}]}
            else:
                yield {"choices": [{"delta": {"content": "Done."}}]}

        executed = []
        async def mock_executor(name: str, args: Dict[str, Any]):
            executed.append(args.get("content"))
            Path(args["path"]).write_text(args.get("content", ""))
            return "ok"

        orch = NativeOrchestrator(stream_provider=provider, tool_executor=mock_executor, enable_cycle_detection=True)
        payload = {"messages": [{"role": "user", "content": f"Write {target_file} with token 4412"}], "acceptance_criteria": f"file_contains:{target_file}:token 4412"}

        async for _ in orch.generate_stream(payload):
            pass

        assert len(executed) == 1
        assert Path(target_file).read_text() == "token 4412"

    @pytest.mark.asyncio
    async def test_case_11_multiple_acceptance_criteria_partial_satisfaction(self, tmp_path):
        """Case 11: Multiple criteria where only some are satisfied does not prematurely stop work."""
        f1 = str(tmp_path / "crit1.txt")
        f2 = str(tmp_path / "crit2.txt")
        turn_counter = 0

        async def provider(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
            nonlocal turn_counter
            turn_counter += 1
            if turn_counter == 1:
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "type": "function", "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": f1, "content": "val1"})}}]}}]}
            elif turn_counter == 2:
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c2", "type": "function", "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": f2, "content": "val2"})}}]}}]}
            else:
                yield {"choices": [{"delta": {"content": "Both ready."}}]}

        created = []
        async def mock_executor(name: str, args: Dict[str, Any]):
            Path(args["path"]).write_text(args["content"])
            created.append(args["path"])
            return "ok"

        orch = NativeOrchestrator(stream_provider=provider, tool_executor=mock_executor, enable_cycle_detection=True)
        payload = {
            "messages": [{"role": "user", "content": f"Create {f1} with val1 and {f2} with val2"}],
            "acceptance_criteria": [f"file_contains:{f1}:val1", f"file_contains:{f2}:val2"],
        }

        async for _ in orch.generate_stream(payload):
            pass

        assert len(created) == 2
        assert Path(f1).read_text() == "val1"
        assert Path(f2).read_text() == "val2"

    @pytest.mark.asyncio
    async def test_case_12_verified_target_protected_while_work_continues_on_another(self, tmp_path):
        """Case 12: Verified target 1 must remain protected from mutation while work continues on target 2."""
        f1 = str(tmp_path / "protected_target.txt")
        f2 = str(tmp_path / "active_target.txt")
        turn_counter = 0

        async def provider(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
            nonlocal turn_counter
            turn_counter += 1
            if turn_counter == 1:
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "type": "function", "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": f1, "content": "good1"})}}]}}]}
            elif turn_counter == 2:
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c2_bad", "type": "function", "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": f1, "content": "corrupted1"})}}]}}]}
            elif turn_counter == 3:
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c3_good", "type": "function", "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": f2, "content": "good2"})}}]}}]}
            else:
                yield {"choices": [{"delta": {"content": "Done."}}]}

        writes = []
        async def mock_executor(name: str, args: Dict[str, Any]):
            writes.append((args["path"], args["content"]))
            Path(args["path"]).write_text(args["content"])
            return "ok"

        orch = NativeOrchestrator(stream_provider=provider, tool_executor=mock_executor, enable_cycle_detection=True)
        payload = {
            "messages": [{"role": "user", "content": f"Create {f1} with good1 and {f2} with good2"}],
            "acceptance_criteria": [f"file_contains:{f1}:good1", f"file_contains:{f2}:good2"],
        }

        async for _ in orch.generate_stream(payload):
            pass

        assert Path(f1).read_text() == "good1"
        assert Path(f2).read_text() == "good2"


@pytest.mark.asyncio
class TestAdversarialTargetIdentityAndVerificationScope:
    """Audit Section 3: Precision of target identity and verification scope."""

    async def test_verified_file_protected_against_destructive_execute_command(self, tmp_path):
        """A verified file must be protected if the model attempts to overwrite or delete it via execute_command."""
        target_file = str(tmp_path / "protected.txt")
        turn_counter = 0

        async def provider(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
            nonlocal turn_counter
            turn_counter += 1
            if turn_counter == 1:
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_w", "type": "function", "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": target_file, "content": "good content"})}}]}}]}
            elif turn_counter == 2:
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_destroy", "type": "function", "function": {"name": "execute_command", "arguments": json.dumps({"command": f"echo 'corrupted' > {target_file}"})}}]}}]}
            else:
                yield {"choices": [{"delta": {"content": "Done."}}]}

        executed_cmds = []
        async def mock_executor(name: str, args: Dict[str, Any]):
            if name == "manage_workspace_file":
                Path(args["path"]).write_text(args.get("content", ""))
                return "Wrote good content"
            elif name == "execute_command":
                executed_cmds.append(args.get("command"))
                return "executed"
            return "ok"

        orch = NativeOrchestrator(stream_provider=provider, tool_executor=mock_executor, enable_cycle_detection=True)
        payload = {"messages": [{"role": "user", "content": f"Write {target_file} with good content"}], "acceptance_criteria": f"file_contains:{target_file}:good content"}

        async for _ in orch.generate_stream(payload):
            pass

        assert len(executed_cmds) == 0, f"Destructive shell command executed on verified target: {executed_cmds}"
        assert Path(target_file).read_text() == "good content"

    async def test_verified_file_permits_safe_read_commands(self, tmp_path):
        """A verified file MUST permit read-only shell commands (cat, grep, ls, head)."""
        target_file = str(tmp_path / "protected_read.txt")
        turn_counter = 0

        async def provider(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
            nonlocal turn_counter
            turn_counter += 1
            if turn_counter == 1:
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_w", "type": "function", "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": target_file, "content": "read me"})}}]}}]}
            elif turn_counter == 2:
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_cat", "type": "function", "function": {"name": "execute_command", "arguments": json.dumps({"command": f"cat {target_file}"})}}]}}]}
            else:
                yield {"choices": [{"delta": {"content": "Done."}}]}

        executed_cmds = []
        async def mock_executor(name: str, args: Dict[str, Any]):
            if name == "manage_workspace_file":
                Path(args["path"]).write_text(args.get("content", ""))
                return "Wrote file"
            elif name == "execute_command":
                executed_cmds.append(args.get("command"))
                return "read me"
            return "ok"

        orch = NativeOrchestrator(stream_provider=provider, tool_executor=mock_executor, enable_cycle_detection=True)
        payload = {"messages": [{"role": "user", "content": f"Write {target_file} with read me, and cat it."}], "acceptance_criteria": f"file_contains:{target_file}:read me"}

        async for _ in orch.generate_stream(payload):
            pass

        assert len(executed_cmds) == 1, "Read-only execute_command (cat) must be allowed on verified file!"

    async def test_verified_file_protected_against_rename_or_deletion(self, tmp_path):
        """A verified file must be protected from unrequested deletion or rename."""
        target_file = str(tmp_path / "no_delete.txt")
        turn_counter = 0

        async def provider(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
            nonlocal turn_counter
            turn_counter += 1
            if turn_counter == 1:
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "type": "function", "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": target_file, "content": "keep me"})}}]}}]}
            elif turn_counter == 2:
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c2_del", "type": "function", "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "delete", "path": target_file})}}]}}]}
            else:
                yield {"choices": [{"delta": {"content": "Done."}}]}

        actions = []
        async def mock_executor(name: str, args: Dict[str, Any]):
            actions.append(args.get("action"))
            p = Path(args["path"])
            if args.get("action") == "write":
                p.write_text(args.get("content", ""))
            elif args.get("action") == "delete":
                p.unlink(missing_ok=True)
            return "ok"

        orch = NativeOrchestrator(stream_provider=provider, tool_executor=mock_executor, enable_cycle_detection=True)
        payload = {"messages": [{"role": "user", "content": f"Create {target_file} with keep me"}], "acceptance_criteria": f"file_contains:{target_file}:keep me"}

        async for _ in orch.generate_stream(payload):
            pass

        assert actions == ["write"], "Unrequested deletion of verified file must be blocked!"
        assert Path(target_file).exists()

    async def test_equivalent_paths_with_dots_are_matched_correctly(self, tmp_path):
        """Relative paths and paths with . and .. must canonicalize and match the verified target."""
        target_file = str(tmp_path / "subdir" / "target.txt")
        Path(tmp_path / "subdir").mkdir(parents=True, exist_ok=True)
        dot_path = str(tmp_path / "subdir" / ".." / "subdir" / "target.txt")
        turn_counter = 0

        async def provider(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
            nonlocal turn_counter
            turn_counter += 1
            if turn_counter == 1:
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": target_file, "content": "initial"})}}]}}]}
            elif turn_counter == 2:
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_2", "type": "function", "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": dot_path, "content": "overwrite"})}}]}}]}
            else:
                yield {"choices": [{"delta": {"content": "Done."}}]}

        writes = []
        async def mock_executor(name: str, args: Dict[str, Any]):
            writes.append((args.get("path"), args.get("content")))
            Path(args["path"]).write_text(args.get("content", ""))
            return "ok"

        orch = NativeOrchestrator(stream_provider=provider, tool_executor=mock_executor, enable_cycle_detection=True)
        payload = {"messages": [{"role": "user", "content": f"Write {target_file} with initial"}], "acceptance_criteria": f"file_contains:{target_file}:initial"}

        async for _ in orch.generate_stream(payload):
            pass

        assert len(writes) == 1, "Attempted write via dot_path must be recognized as the verified file!"
        assert Path(target_file).read_text() == "initial"

    async def test_relative_and_absolute_path_equivalence(self, tmp_path):
        """A target specified as relative must protect the same file when addressed as absolute."""
        rel_name = "test_rel.txt"
        target_rel = rel_name
        target_abs = str((Path.cwd() / rel_name).resolve())
        turn_counter = 0

        async def provider(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
            nonlocal turn_counter
            turn_counter += 1
            if turn_counter == 1:
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "type": "function", "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": target_rel, "content": "data"})}}]}}]}
            elif turn_counter == 2:
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c2", "type": "function", "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": target_abs, "content": "bad"})}}]}}]}
            else:
                yield {"choices": [{"delta": {"content": "Done."}}]}

        writes = []
        async def mock_executor(name: str, args: Dict[str, Any]):
            writes.append(args)
            return "ok"

        orch = NativeOrchestrator(stream_provider=provider, tool_executor=mock_executor, enable_cycle_detection=True)
        payload = {"messages": [{"role": "user", "content": f"Write {target_rel} with data"}], "acceptance_criteria": f"file_contains:{target_rel}:data"}

        original_verify = OutcomeVerifier.verify
        def mock_verify(check, target, expected=None, **kwargs):
            if len(writes) > 0 and canonicalize_target(target) == canonicalize_target(target_rel):
                from zelyr.engine.verifier import VerificationResult
                return VerificationResult(success=True, check=check, target=target, observed="data")
            return original_verify(check, target, expected, **kwargs)

        OutcomeVerifier.verify = mock_verify
        try:
            async for _ in orch.generate_stream(payload):
                pass
        finally:
            OutcomeVerifier.verify = original_verify

        assert len(writes) == 1, "Absolute path write must be recognized as equivalent to relative verified target!"

    async def test_non_filesystem_verification_target_does_not_pollute_filesystem_paths(self):
        """Non-filesystem verification (port_listening, process_running) must not create bogus filesystem targets."""
        parsed = parse_acceptance_criteria("process_running:nginx")
        assert len(parsed) == 1
        assert parsed[0]["check"] == "process_running"
        assert parsed[0]["target"] == "nginx"

        ledger = TurnExecutionLedger()
        ledger.record_target_verified("nginx", check_type="process_running")
        verified = ledger.get_verified_targets()
        assert not any(v.startswith("/") and v.endswith("/nginx") for v in verified), \
            f"Process name 'nginx' was incorrectly recorded as filesystem path: {verified}"
        assert "process:nginx" in verified
