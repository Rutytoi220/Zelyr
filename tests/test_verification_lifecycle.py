"""End-to-end verification lifecycle integrity, CLI/ledger consistency, and semantic validator tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List
from unittest.mock import AsyncMock

import pytest

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.cli.main import _execute_headless_run
from zelyr.engine.diagnostics import (
    diagnose_execution_lifecycle,
    format_execution_lifecycle_report,
)
from zelyr.engine.execution_ledger import (
    ActionState,
    CentralExecutionLedger,
    ExecutionStatus,
    LedgerEventType,
    TurnExecutionLedger,
    VerificationStatus,
    compute_policy_compliance,
    resolve_root_execution_outcome,
)
from zelyr.eval.types import ValidationResult
from zelyr.eval.validators import (
    assert_response_semantic_facts,
    validate_response_semantic_facts,
)


def _open_isolated_ledger(db_path: Path) -> CentralExecutionLedger:
    return CentralExecutionLedger(db_path=db_path, raise_on_persistence_error=True)


def _make_orch_with_db(
    db_path: Path,
    stream_provider: Any,
    tool_executor: Any,
    *,
    max_steps: int = 5,
) -> tuple[CentralExecutionLedger, NativeOrchestrator]:
    central = _open_isolated_ledger(db_path)
    orch = NativeOrchestrator(
        central_ledger=central,
        stream_provider=stream_provider,
        tool_executor=tool_executor,
        max_steps=max_steps,
    )
    orch.workspace_root = db_path.parent
    return central, orch


@pytest.mark.asyncio
async def test_verified_success_persists_across_reopen_and_single_terminal_event(tmp_path: Path) -> None:
    db_path = tmp_path / "ledger.db"
    target = tmp_path / "hello.txt"

    async def stream_fn(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        tools = [t["function"]["name"] for t in payload.get("tools", []) if isinstance(t, dict) and "function" in t]
        # After write succeeds and satisfies --verify, tools may be disabled or model summarizes
        has_tool_result = any(m.get("role") == "tool" for m in payload.get("messages", []))
        if not has_tool_result and "manage_workspace_file" in tools:
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_write_hello",
                            "type": "function",
                            "function": {
                                "name": "manage_workspace_file",
                                "arguments": json.dumps({"action": "write", "path": str(target), "content": "Hello from Zelyr!\n"}),
                            },
                        }]
                    },
                    "finish_reason": "tool_calls",
                }]
            }
        else:
            yield {
                "choices": [{
                    "delta": {"content": "Created hello.txt with the requested text."},
                    "finish_reason": "stop",
                }]
            }

    async def tool_exec(name: str, **kwargs: Any) -> str:
        Path(kwargs["path"]).write_text(kwargs["content"], encoding="utf-8")
        return json.dumps({"status": "success", "path": kwargs["path"]})

    central, orch = _make_orch_with_db(db_path, stream_fn, tool_exec)
    try:
        rc = await _execute_headless_run(
            f"Create {target} containing Hello from Zelyr!",
            orchestrator=orch,
            verify_spec=f"file_contains:{target}:Hello from Zelyr!",
            output_json=True,
        )
        assert rc == 0
        exec_id = orch.current_ledger._execution_id
        assert exec_id is not None
    finally:
        central.close()

    # Reopen from fresh CentralExecutionLedger instance to verify persistence
    reopened = _open_isolated_ledger(db_path)
    try:
        rec = reopened.get_execution(exec_id)
        assert rec is not None
        assert rec.status == ExecutionStatus.COMPLETED.value
        assert rec.verification_status == VerificationStatus.VERIFIED.value

        events = reopened.get_events(exec_id)
        term_events = [e for e in events if e.event_type == LedgerEventType.TASK_TERMINATED.value]
        ver_events = [e for e in events if e.event_type == LedgerEventType.VERIFICATION_RECORDED.value]
        assert len(term_events) == 1
        assert len(ver_events) >= 1
        assert any(e.status == VerificationStatus.VERIFIED.value for e in ver_events)

        diag = diagnose_execution_lifecycle(reopened, exec_id)
        assert diag.found
        assert diag.request["status"] == ExecutionStatus.COMPLETED.value
        assert diag.request["verification_status"] == VerificationStatus.VERIFIED.value
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_executed_without_verification_stays_completed_unverified_and_external_boundary_preserved(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "ledger.db"
    target = tmp_path / "note.txt"

    async def stream_fn(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        has_tool = any(m.get("role") == "tool" for m in payload.get("messages", []))
        if not has_tool:
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_write_note",
                            "type": "function",
                            "function": {
                                "name": "manage_workspace_file",
                                "arguments": json.dumps({"action": "write", "path": str(target), "content": "Unverified work"}),
                            },
                        }]
                    },
                    "finish_reason": "tool_calls",
                }]
            }
        else:
            yield {
                "choices": [{
                    "delta": {"content": "Wrote note.txt."},
                    "finish_reason": "stop",
                }]
            }

    async def tool_exec(name: str, **kwargs: Any) -> str:
        Path(kwargs["path"]).write_text(kwargs["content"], encoding="utf-8")
        return json.dumps({"status": "success", "path": kwargs["path"]})

    central, orch = _make_orch_with_db(db_path, stream_fn, tool_exec)
    try:
        rc = await _execute_headless_run(
            f"Write Unverified work to {target}",
            orchestrator=orch,
            verify_spec=None,
            output_json=True,
        )
        assert rc == 0
        exec_id = orch.current_ledger._execution_id
        assert exec_id is not None

        # Simulate external harness validation passing on disk WITHOUT fabricating internal ledger verification
        assert target.read_text(encoding="utf-8") == "Unverified work"

        rec = central.get_execution(exec_id)
        assert rec is not None
        assert rec.status == ExecutionStatus.COMPLETED_UNVERIFIED.value
        assert rec.verification_status == VerificationStatus.NOT_VERIFIED.value

        events = central.get_events(exec_id)
        assert not any(e.event_type == LedgerEventType.VERIFICATION_RECORDED.value for e in events)
        assert len([e for e in events if e.event_type == LedgerEventType.TASK_TERMINATED.value]) == 1
    finally:
        central.close()


@pytest.mark.asyncio
async def test_failed_required_verification_marks_failed_and_nonzero_exit(tmp_path: Path) -> None:
    db_path = tmp_path / "ledger.db"
    target = tmp_path / "bad.txt"

    async def stream_fn(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        has_tool = any(m.get("role") == "tool" for m in payload.get("messages", []))
        if not has_tool:
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_write_wrong",
                            "type": "function",
                            "function": {
                                "name": "manage_workspace_file",
                                "arguments": json.dumps({"action": "write", "path": str(target), "content": "wrong content"}),
                            },
                        }]
                    },
                    "finish_reason": "tool_calls",
                }]
            }
        else:
            yield {
                "choices": [{
                    "delta": {"content": "Attempted writing bad.txt, verification did not match."},
                    "finish_reason": "stop",
                }]
            }

    async def tool_exec(name: str, **kwargs: Any) -> str:
        Path(kwargs["path"]).write_text(kwargs["content"], encoding="utf-8")
        return json.dumps({"status": "success", "path": kwargs["path"]})

    central, orch = _make_orch_with_db(db_path, stream_fn, tool_exec)
    try:
        rc = await _execute_headless_run(
            f"Write EXPECTED_TOKEN to {target}",
            orchestrator=orch,
            verify_spec=f"file_contains:{target}:EXPECTED_TOKEN",
            output_json=False,
        )
        assert rc == 2
        assert target.read_text(encoding="utf-8") == "wrong content"
        exec_id = orch.current_ledger._execution_id
        assert exec_id is not None

        rec = central.get_execution(exec_id)
        assert rec is not None
        assert rec.status == ExecutionStatus.FAILED.value
        assert rec.verification_status == VerificationStatus.VERIFICATION_FAILED.value
        events = central.get_events(exec_id)
        assert len([e for e in events if e.event_type == LedgerEventType.TASK_TERMINATED.value]) == 1
    finally:
        central.close()


@pytest.mark.asyncio
async def test_partial_acceptance_criteria_and_recovery_after_verification_failure(tmp_path: Path) -> None:
    # 1. Partial acceptance criteria: a.txt verified, missing.txt absent -> root MUST fail verification
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "missing_b.txt"
    file_a.write_text("ok", encoding="utf-8")

    partial_ledger = TurnExecutionLedger()
    partial_ledger.record_proposed("c1", "manage_workspace_file", {"action": "write", "path": str(file_a)})
    partial_ledger.record_dispatched("c1")
    partial_ledger.record_observation(
        "c1",
        status="success",
        output="ok",
        verified=True,
        verified_targets=[str(file_a)],
    )
    partial_res = resolve_root_execution_outcome(
        prompt="Create a.txt and missing_b.txt",
        ledger=partial_ledger,
        final_text="Created a.txt",
        acceptance_spec=f"file_exists:{file_a}, file_exists:{file_b}",
    )
    assert partial_res.execution_status == ExecutionStatus.FAILED
    assert partial_res.verification_status == VerificationStatus.VERIFICATION_FAILED
    assert partial_res.exit_code == 2

    # 2. Corrective recovery after an earlier failed draft verification -> root MUST succeed as COMPLETED/VERIFIED
    target = tmp_path / "recovered.txt"
    target.write_text("FIXED", encoding="utf-8")
    rec_ledger = TurnExecutionLedger()
    rec_ledger.record_proposed("c_draft", "manage_workspace_file", {"action": "write", "path": str(target)})
    rec_ledger.record_dispatched("c_draft")
    rec_ledger.record_observation(
        "c_draft",
        status="success",
        output="wrote draft",
        verified=False,
        verification_failed=True,
        verification_detail="Missing FIXED",
        failed_targets=[str(target)],
    )
    rec_ledger.record_proposed("c_fix", "manage_workspace_file", {"action": "write", "path": str(target)})
    rec_ledger.record_dispatched("c_fix")
    rec_ledger.record_observation(
        "c_fix",
        status="success",
        output="wrote fix",
        verified=True,
        verified_targets=[str(target)],
    )
    recovered_res = resolve_root_execution_outcome(
        prompt=f"Write FIXED to {target}",
        ledger=rec_ledger,
        final_text="Updated recovered.txt with FIXED.",
        acceptance_spec=f"file_contains:{target}:FIXED",
    )
    assert recovered_res.execution_status == ExecutionStatus.COMPLETED
    assert recovered_res.verification_status == VerificationStatus.VERIFIED
    assert recovered_res.exit_code == 0


@pytest.mark.asyncio
async def test_cli_ledger_consistency_boundaries(tmp_path: Path) -> None:
    # 1. Step limit exceeded -> rc=2 and SQLite finalized as FAILED with interruption_reason='step_limit_exceeded'
    db_lim = tmp_path / "lim.db"

    async def loop_stream(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        yield {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_loop",
                        "type": "function",
                        "function": {"name": "list_tree", "arguments": '{"path": "."}'},
                    }]
                },
                "finish_reason": "tool_calls",
            }]
        }

    central_lim, orch_lim = _make_orch_with_db(
        db_lim,
        loop_stream,
        AsyncMock(return_value=json.dumps({"success": True, "entries": []})),
        max_steps=1,
    )
    try:
        rc_lim = await _execute_headless_run("Inspect project", orchestrator=orch_lim, output_json=True)
        assert rc_lim == 2
        eid_lim = orch_lim.current_ledger._execution_id
        assert eid_lim is not None
        rec_lim = central_lim.get_execution(eid_lim)
        assert rec_lim is not None
        assert rec_lim.status == ExecutionStatus.FAILED.value
        assert rec_lim.metadata.get("interruption_reason") == "step_limit_exceeded"
        assert len([e for e in central_lim.get_events(eid_lim) if e.event_type == LedgerEventType.TASK_TERMINATED.value]) == 1
    finally:
        central_lim.close()

    # 2. Truncated output -> rc=1 and SQLite finalized as FAILED/INTERRUPTED (never completed_unverified)
    db_trunc = tmp_path / "trunc.db"

    async def trunc_stream(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        yield {
            "choices": [{
                "delta": {"reasoning_content": "thinking forever..."},
                "finish_reason": "length",
            }]
        }

    central_trunc, orch_trunc = _make_orch_with_db(
        db_trunc,
        trunc_stream,
        AsyncMock(return_value="{}"),
    )
    try:
        rc_trunc = await _execute_headless_run("Summarize repo", orchestrator=orch_trunc)
        assert rc_trunc == 1
        eid_trunc = orch_trunc.current_ledger._execution_id
        assert eid_trunc is not None
        rec_trunc = central_trunc.get_execution(eid_trunc)
        assert rec_trunc is not None
        assert rec_trunc.status in (ExecutionStatus.FAILED.value, ExecutionStatus.INTERRUPTED.value)
        assert rec_trunc.verification_status == VerificationStatus.INTERRUPTED.value
        assert len([e for e in central_trunc.get_events(eid_trunc) if e.event_type == LedgerEventType.TASK_TERMINATED.value]) == 1
    finally:
        central_trunc.close()

    # 3. Invalid --verify spec -> rc=2 and SQLite finalized FAILED/VERIFICATION_FAILED
    db_inv = tmp_path / "invalid_verify.db"

    async def ok_text_stream(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        yield {"choices": [{"delta": {"content": "Done."}, "finish_reason": "stop"}]}

    central_inv, orch_inv = _make_orch_with_db(db_inv, ok_text_stream, AsyncMock(return_value="{}"))
    try:
        rc_inv = await _execute_headless_run("Say hi", orchestrator=orch_inv, verify_spec="")
        assert rc_inv == 2
        eid_inv = orch_inv.current_ledger._execution_id
        assert eid_inv is not None
        rec_inv = central_inv.get_execution(eid_inv)
        assert rec_inv is not None
        assert rec_inv.status == ExecutionStatus.FAILED.value
        assert rec_inv.verification_status == VerificationStatus.VERIFICATION_FAILED.value
    finally:
        central_inv.close()

    # 4. Uncaught exception inside orchestrator -> rc=1 and SQLite finalized INTERRUPTED
    db_exc = tmp_path / "exc.db"

    async def boom_stream(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        if False:
            yield {}
        raise RuntimeError("synthetic transport crash")

    central_exc, orch_exc = _make_orch_with_db(db_exc, boom_stream, AsyncMock(return_value="{}"))
    try:
        rc_exc = await _execute_headless_run("Trigger crash", orchestrator=orch_exc)
        assert rc_exc == 1
        eid_exc = orch_exc.current_ledger._execution_id
        assert eid_exc is not None
        rec_exc = central_exc.get_execution(eid_exc)
        assert rec_exc is not None
        assert rec_exc.status == ExecutionStatus.INTERRUPTED.value
        assert rec_exc.verification_status == VerificationStatus.INTERRUPTED.value
    finally:
        central_exc.close()

    # 5. --verify passes alongside an unresolved action failure -> rc=2, FAILED / VERIFIED
    db_mix = tmp_path / "mix.db"
    ok_file = tmp_path / "already_ok.txt"
    ok_file.write_text("READY", encoding="utf-8")

    async def mix_stream(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        has_tool = any(m.get("role") == "tool" for m in payload.get("messages", []))
        if not has_tool:
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_fail_cmd",
                            "type": "function",
                            "function": {"name": "execute_command", "arguments": json.dumps({"command": "false"})},
                        }]
                    },
                    "finish_reason": "tool_calls",
                }]
            }
        else:
            yield {"choices": [{"delta": {"content": "Command failed while file exists."}, "finish_reason": "stop"}]}

    central_mix, orch_mix = _make_orch_with_db(
        db_mix,
        mix_stream,
        AsyncMock(return_value=json.dumps({"status": "error", "exit_code": 1, "output": "failed"})),
    )
    try:
        rc_mix = await _execute_headless_run(
            f"Run command and keep {ok_file}",
            orchestrator=orch_mix,
            verify_spec=f"file_contains:{ok_file}:READY",
        )
        assert rc_mix == 2
        eid_mix = orch_mix.current_ledger._execution_id
        assert eid_mix is not None
        rec_mix = central_mix.get_execution(eid_mix)
        assert rec_mix is not None
        assert rec_mix.status == ExecutionStatus.FAILED.value
        assert rec_mix.verification_status == VerificationStatus.VERIFIED.value
        assert rec_mix.metadata.get("interruption_reason") == "unresolved_failure"
    finally:
        central_mix.close()


@pytest.mark.asyncio
async def test_cli_cancellation_finalizes_ledger_before_propagating(tmp_path: Path) -> None:
    db_path = tmp_path / "cancel.db"
    started = asyncio.Event()

    async def cancel_stream(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        started.set()
        await asyncio.sleep(10)
        yield {"choices": [{"delta": {"content": "never"}, "finish_reason": "stop"}]}

    central, orch = _make_orch_with_db(db_path, cancel_stream, AsyncMock(return_value="{}"))
    try:
        task = asyncio.create_task(
            _execute_headless_run("Long task", orchestrator=orch)
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        eid = orch.current_ledger._execution_id
        assert eid is not None
        rec = central.get_execution(eid)
        assert rec is not None
        assert rec.status == ExecutionStatus.CANCELLED.value
        assert rec.verification_status == VerificationStatus.INTERRUPTED.value
        assert len([e for e in central.get_events(eid) if e.event_type == LedgerEventType.TASK_TERMINATED.value]) == 1
    finally:
        central.close()


def test_finalize_execution_is_idempotent_per_epoch(tmp_path: Path) -> None:
    central = _open_isolated_ledger(tmp_path / "idem.db")
    try:
        rec = central.create_execution("Do work", source_component="test")
        first = central.finalize_execution(
            rec.execution_id,
            status=ExecutionStatus.COMPLETED_UNVERIFIED,
            verification_status=VerificationStatus.NOT_VERIFIED,
            final_summary="first finish",
        )
        assert first is not None
        # Repeat call with identical or conflicting status inside the same epoch must be a safe no-op
        second = central.finalize_execution(
            rec.execution_id,
            status=ExecutionStatus.FAILED,
            verification_status=VerificationStatus.VERIFICATION_FAILED,
            final_summary="duplicate overwrite",
            error="should be ignored",
        )
        assert second is not None
        assert second.status == ExecutionStatus.COMPLETED_UNVERIFIED.value
        assert second.completed_at == first.completed_at
        events = central.get_events(rec.execution_id)
        assert len([e for e in events if e.event_type == LedgerEventType.TASK_TERMINATED.value]) == 1

        # Explicitly resuming starts a new epoch allowing one subsequent terminal event
        central.record_event(rec.execution_id, LedgerEventType.TASK_RESUMED, status="resumed")
        third = central.finalize_execution(
            rec.execution_id,
            status=ExecutionStatus.COMPLETED,
            verification_status=VerificationStatus.VERIFIED,
            final_summary="resumed finish",
        )
        assert third is not None
        assert third.status == ExecutionStatus.COMPLETED.value
        assert third.verification_status == VerificationStatus.VERIFIED.value
        events_after = central.get_events(rec.execution_id)
        assert len([e for e in events_after if e.event_type == LedgerEventType.TASK_TERMINATED.value]) == 2
    finally:
        central.close()


def test_parent_child_execution_isolation_and_diagnostics(tmp_path: Path) -> None:
    central = _open_isolated_ledger(tmp_path / "family.db")
    try:
        parent = central.create_execution("Parent task", source_component="test")
        sibling = central.create_execution("Sibling root task", source_component="test")
        child1 = central.create_execution(
            "Child 1 verify step",
            parent_execution_id=parent.execution_id,
            source_component="subagent",
        )
        child2 = central.create_execution(
            "Child 2 failing step",
            parent_execution_id=parent.execution_id,
            source_component="subagent",
        )

        central.record_event(sibling.execution_id, LedgerEventType.TOOL_SELECTED, tool_id="sibling_tool", status="selected")
        central.finalize_execution(
            child1.execution_id,
            status=ExecutionStatus.COMPLETED,
            verification_status=VerificationStatus.VERIFIED,
        )
        central.finalize_execution(
            child2.execution_id,
            status=ExecutionStatus.FAILED,
            verification_status=VerificationStatus.VERIFICATION_FAILED,
            error="Sub-check failed",
        )
        central.finalize_execution(
            parent.execution_id,
            status=ExecutionStatus.COMPLETED_UNVERIFIED,
            verification_status=VerificationStatus.NOT_VERIFIED,
        )

        children = central.list_child_executions(parent.execution_id)
        assert [c.execution_id for c in children] == [child1.execution_id, child2.execution_id]
        assert central.list_child_executions(sibling.execution_id) == []

        diag = diagnose_execution_lifecycle(central, parent.execution_id)
        assert len(diag.child_executions) == 2
        assert diag.verification["child_execution_count"] == 2
        assert "sibling_tool" not in diag.tools["unique_tools"]
        rendered = format_execution_lifecycle_report(diag)
        assert child1.execution_id in rendered
        assert child2.execution_id in rendered
        assert "status=verified" not in rendered  # child verification renders explicitly as verification=verified
        assert "verification=verified" in rendered
        assert "verification=verification_failed" in rendered
    finally:
        central.close()


@pytest.mark.asyncio
async def test_readonly_diagnostic_and_prohibition_policy_with_list_tree_alias(
    tmp_path: Path,
    capsys: Any,
) -> None:
    assert NativeOrchestrator.normalize_tool_name("list_tree") == "manage_workspace_file"
    assert NativeOrchestrator.normalize_tool_name("list_dir") == "manage_workspace_file"
    assert NativeOrchestrator.normalize_tool_arguments(
        "manage_workspace_file", {"path": "."}, original_name="list_tree"
    ) == {
        "path": ".",
        "action": "list_tree",
    }
    assert NativeOrchestrator.normalize_tool_arguments(
        "manage_workspace_file", {"path": "."}, original_name="list_dir"
    ) == {
        "path": ".",
        "action": "list_tree",
    }

    target_app = tmp_path / "app.py"
    target_app.write_text("def div(a, b):\n    return a / b\n", encoding="utf-8")
    db_path = tmp_path / "prohib.db"
    turn = 0

    async def prohib_stream(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        nonlocal turn
        turn += 1
        if turn == 1:
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_list",
                            "type": "function",
                            "function": {"name": "list_tree", "arguments": json.dumps({"path": str(tmp_path)})},
                        }]
                    },
                    "finish_reason": "tool_calls",
                }]
            }
        elif turn == 2:
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_pt",
                            "type": "function",
                            "function": {"name": "execute_command", "arguments": json.dumps({"command": "pytest -q"})},
                        }]
                    },
                    "finish_reason": "tool_calls",
                }]
            }
        elif turn == 3:
            # Model attempts prohibited edit -> blocked cleanly by prohibition guard
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_forbidden_edit",
                            "type": "function",
                            "function": {
                                "name": "manage_workspace_file",
                                "arguments": json.dumps({"action": "write", "path": str(target_app), "content": "mutated"}),
                            },
                        }]
                    },
                    "finish_reason": "tool_calls",
                }]
            }
        else:
            yield {
                "choices": [{
                    "delta": {
                        "content": "Failure is caused by ZeroDivisionError in app.py when b == 0; add a zero check before dividing."
                    },
                    "finish_reason": "stop",
                }]
            }

    async def mock_exec(name: str, **kwargs: Any) -> str:
        if name == "manage_workspace_file" and kwargs.get("action") == "list_tree":
            return json.dumps({"status": "success", "entries": ["app.py"]})
        if name == "execute_command":
            return json.dumps({"status": "error", "exit_code": 1, "output": "ZeroDivisionError: division by zero"})
        raise AssertionError(f"Unexpected mutating execution: {name} {kwargs}")

    central, orch = _make_orch_with_db(db_path, prohib_stream, mock_exec, max_steps=6)
    try:
        rc = await _execute_headless_run(
            "Inspect this directory, run pytest -q, and explain what caused the failure without modifying any files.",
            orchestrator=orch,
            output_json=True,
        )
        assert rc == 0
        out_raw = capsys.readouterr().out.strip().splitlines()[-1]
        data = json.loads(out_raw)
        assert data.get("diagnostic_test_failed") is True
        assert data["policy_compliance"]["has_prohibitions"] is True
        assert data["policy_compliance"]["compliant"] is True
        assert data["policy_compliance"]["prohibited_attempts_blocked"] == 1
        assert data["policy_compliance"]["prohibited_violations"] == 0
        assert data["policy_compliance"]["status"] == "compliant"

        eid = orch.current_ledger._execution_id
        assert eid is not None
        rec = central.get_execution(eid)
        assert rec is not None
        assert rec.status == ExecutionStatus.COMPLETED_UNVERIFIED.value
        assert rec.verification_status == VerificationStatus.NOT_VERIFIED.value
        assert rec.metadata.get("diagnostic_test_failed") is True
        assert rec.metadata["policy_compliance"]["status"] == "compliant"

        diag = diagnose_execution_lifecycle(central, eid)
        rendered = format_execution_lifecycle_report(diag)
        assert "Prohibitions:  status=compliant blocked=1 violations=0" in rendered
    finally:
        central.close()

    # Negative check on compute_policy_compliance + resolve_root_execution_outcome: mutation under prohibition -> violated
    bad_ledger = TurnExecutionLedger()
    bad_ledger.record_proposed("c_mut", "manage_workspace_file", {"action": "write", "path": "app.py"})
    bad_ledger.record_dispatched("c_mut")
    bad_ledger.record_observation("c_mut", status="success", output="mutated")
    bad_comp = compute_policy_compliance("Explain why app.py fails without modifying any files.", bad_ledger)
    assert bad_comp["status"] == "violated"
    assert bad_comp["compliant"] is False
    assert bad_comp["prohibited_violations"] == 1
    viol_res = resolve_root_execution_outcome(
        prompt="Explain why app.py fails without modifying any files.",
        ledger=bad_ledger,
        final_text="Done",
    )
    assert viol_res.execution_status == ExecutionStatus.FAILED
    assert viol_res.interruption_reason == "policy_violation"
    assert viol_res.exit_code == 2


def test_task2_semantic_response_validator_equivalence_and_rejections() -> None:
    # Accepts pyproject script entrypoint notation (e.g., greeter.cli:main) as well as file paths
    ok_toml, msg_toml = assert_response_semantic_facts(
        "This is a Python package whose entrypoint is `greeter.cli:main` (script `greet`), tested with pytest."
    )
    assert ok_toml, msg_toml

    ok_file, _ = assert_response_semantic_facts(
        "Built in Python. Main entry point is src/greeter/cli.py and test suite uses pytest."
    )
    assert ok_file

    # Rejects bare verb "greet" without CLI/entrypoint attribution or missing core facts
    fail_bare, reason_bare = assert_response_semantic_facts(
        "You can greet friends using Python and pytest."
    )
    assert not fail_bare and "entrypoint" in reason_bare

    fail_missing, reason_missing = assert_response_semantic_facts(
        "This is a Java CLI compiled with Maven."
    )
    assert not fail_missing
    assert "language" in reason_missing and "entrypoint" in reason_missing and "test_framework" in reason_missing

    # Works via ValidatorFn interface
    val_fn = validate_response_semantic_facts()
    ctx = type("Ctx", (), {"response_text": "Python CLI (`greeter.cli:main`) running under pytest."})()
    res: ValidationResult = val_fn(ctx)
    assert res.success is True
