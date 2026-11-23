"""Deterministic offline test suite for Zelyr Graph-Aware Parallel Execution."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Dict, List

import pytest

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.engine import (
    ActionState,
    CentralExecutionLedger,
    CheckpointManager,
    CorruptedTaskGraphError,
    ExecutionStatus,
    LedgerEventType,
    PendingActionProposal,
    ResumabilityStatus,
    TaskGraph,
    TaskGraphConflictError,
    TaskGraphStatus,
    TaskGraphValidationError,
    TaskNode,
    TaskNodeStatus,
    TurnExecutionLedger,
    VerificationStatus,
    get_active_execution_context,
)
from zelyr.tool_registry import TaskLocalToolScope, ToolRegistry, bind_task_local_executor
from zelyr.tools.core import BaseTool, ToolResult


def _build_schemas(*names: str, read_only_names: set[str] | None = None) -> List[Dict[str, Any]]:
    ro_set = read_only_names or set()
    schemas: List[Dict[str, Any]] = []
    for name in names:
        fn_def: Dict[str, Any] = {
            "name": name,
            "description": f"Test schema for {name}",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "path": {"type": "string"},
                    "file_path": {"type": "string"},
                    "target": {"type": "string"},
                    "content": {"type": "string"},
                    "command": {"type": "string"},
                    "query": {"type": "string"},
                    "payload": {"type": "string"},
                },
                "additionalProperties": True,
            },
        }
        if name in ro_set:
            fn_def["read_only"] = True
        schemas.append({"type": "function", "function": fn_def})
    return schemas


def _bind_scope(fn: Callable[..., Any], scope: TaskLocalToolScope) -> Callable[..., Any]:
    return bind_task_local_executor(
        fn,
        tool_names=scope.tool_names,
        tool_schemas=scope.tool_schemas,
        tools_dir=scope.tools_dir,
        tool_rings=scope.tool_rings,
    )


def _make_proposal(
    exec_id: str,
    pid: str,
    cid: str,
    tool_id: str,
    args: Dict[str, Any],
    *,
    order: int = 0,
    depends_on: List[str] | None = None,
    external_source: Dict[str, Any] | None = None,
) -> PendingActionProposal:
    return PendingActionProposal(
        proposal_id=pid,
        execution_id=exec_id,
        original_call_id=cid,
        tool_id=tool_id,
        raw_name=tool_id,
        normalized_arguments=dict(args),
        proposal_order=order,
        depends_on=list(depends_on or []),
        external_source=external_source,
        status="pending",
    )


class TestBoundedParallelSchedulingAndOverlap:
    """Verify backward-compatible default=1, true async overlap up to max_concurrency, and DAG ordering."""

    @pytest.mark.asyncio
    async def test_1_default_concurrency_is_1_and_runs_sequentially(self, tmp_path: Path):
        exec_id = "exec_default_seq"
        ledger_db = tmp_path / "ledger.db"
        central = CentralExecutionLedger(db_path=ledger_db)

        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        nodes = [
            TaskNode(node_id="n1", proposal_ids=["p1"], order_index=0),
            TaskNode(node_id="n2", proposal_ids=["p2"], order_index=1),
        ]
        graph = TaskGraph(graph_id="tg_seq", execution_id=exec_id, objective="Sequential default", nodes=nodes)
        assert graph.max_concurrency == 1

        props = [
            _make_proposal(exec_id, "p1", "c1", "manage_workspace_file", {"action": "write", "path": str(f1), "content": "one"}, order=0),
            _make_proposal(exec_id, "p2", "c2", "manage_workspace_file", {"action": "write", "path": str(f2), "content": "two"}, order=1),
        ]

        active_now = 0
        max_active = 0
        order_log: List[str] = []

        async def _exec(tool_name: str, args: dict) -> dict:
            nonlocal active_now, max_active
            active_now += 1
            max_active = max(max_active, active_now)
            order_log.append(Path(args["path"]).name)
            await asyncio.sleep(0.01)
            Path(args["path"]).write_text(args["content"], encoding="utf-8")
            active_now -= 1
            return {"status": "success", "output": "ok"}

        scope = TaskLocalToolScope(
            tool_names=["manage_workspace_file"],
            tool_schemas=_build_schemas("manage_workspace_file"),
        )
        orch = NativeOrchestrator(central_ledger=central, task_local_scope=scope, tool_executor=_bind_scope(_exec, scope))
        assert orch.max_concurrency == 1

        res = await orch.execute_task_graph(
            graph,
            props,
            acceptance_criteria=f"file_exists:{f1}; file_exists:{f2}",
        )
        assert max_active == 1
        assert order_log == ["a.txt", "b.txt"]
        assert res["graph_status"] == TaskGraphStatus.COMPLETED_VERIFIED.value
        assert graph.status == TaskGraphStatus.COMPLETED_VERIFIED.value
        central.close()

    @pytest.mark.asyncio
    async def test_2_diamond_dag_executes_independent_branches_in_parallel_and_waits_for_both_verified(self, tmp_path: Path):
        exec_id = "exec_diamond"
        central = CentralExecutionLedger(db_path=tmp_path / "ledger.db")
        f_a = tmp_path / "a.txt"
        f_b = tmp_path / "b.txt"
        f_c = tmp_path / "c.txt"
        f_d = tmp_path / "d.txt"

        graph = TaskGraph(
            graph_id="tg_diamond",
            execution_id=exec_id,
            objective="Diamond build",
            max_concurrency=2,
            nodes=[
                TaskNode(node_id="A", proposal_ids=["pA"], order_index=0),
                TaskNode(node_id="B", proposal_ids=["pB"], dependency_ids=["A"], order_index=1),
                TaskNode(node_id="C", proposal_ids=["pC"], dependency_ids=["A"], order_index=2),
                TaskNode(node_id="D", proposal_ids=["pD"], dependency_ids=["B", "C"], order_index=3),
            ],
        )
        props = [
            _make_proposal(exec_id, "pA", "cA", "manage_workspace_file", {"action": "write", "path": str(f_a), "content": "A"}, order=0),
            _make_proposal(exec_id, "pB", "cB", "manage_workspace_file", {"action": "write", "path": str(f_b), "content": "B"}, order=1, depends_on=["pA"]),
            _make_proposal(exec_id, "pC", "cC", "manage_workspace_file", {"action": "write", "path": str(f_c), "content": "C"}, order=2, depends_on=["pA"]),
            _make_proposal(exec_id, "pD", "cD", "manage_workspace_file", {"action": "write", "path": str(f_d), "content": "D"}, order=3, depends_on=["pB", "pC"]),
        ]

        entered_bc = 0
        both_in_bc = asyncio.Event()
        active_now = 0
        max_active = 0
        saw_d_with_verified_bc = False
        events: List[str] = []

        async def _exec(tool_name: str, args: dict) -> dict:
            nonlocal entered_bc, active_now, max_active, saw_d_with_verified_bc
            p = Path(args["path"])
            events.append(f"start:{p.stem}")
            active_now += 1
            max_active = max(max_active, active_now)
            if p.stem in ("b", "c"):
                entered_bc += 1
                if entered_bc == 2:
                    both_in_bc.set()
                await asyncio.wait_for(both_in_bc.wait(), timeout=2.0)
            if p.stem == "d":
                saw_d_with_verified_bc = (
                    graph.get_node("B").execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value
                    and graph.get_node("C").execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value
                    and f_b.exists()
                    and f_c.exists()
                )
            p.write_text(args["content"], encoding="utf-8")
            events.append(f"end:{p.stem}")
            active_now -= 1
            return {"status": "success", "output": f"wrote {p.name}"}

        scope = TaskLocalToolScope(
            tool_names=["manage_workspace_file"],
            tool_schemas=_build_schemas("manage_workspace_file"),
        )
        orch = NativeOrchestrator(central_ledger=central, task_local_scope=scope, tool_executor=_bind_scope(_exec, scope))
        res = await orch.execute_task_graph(
            graph,
            props,
            acceptance_criteria=f"file_exists:{f_a}; file_exists:{f_b}; file_exists:{f_c}; file_exists:{f_d}",
        )
        assert max_active == 2, f"Expected B and C to overlap concurrently, events={events}"
        assert saw_d_with_verified_bc is True
        assert res["graph_status"] == TaskGraphStatus.COMPLETED_VERIFIED.value
        assert graph.active_reservations == {}
        assert all(n.reserved_by is None for n in graph.nodes)
        central.close()

    @pytest.mark.asyncio
    async def test_3_max_concurrency_never_exceeded_and_invalid_values_rejected(self, tmp_path: Path):
        exec_id = "exec_bound"
        central = CentralExecutionLedger(db_path=tmp_path / "ledger.db")

        with pytest.raises(TaskGraphValidationError):
            TaskGraph(graph_id="bad_zero", execution_id=exec_id, nodes=[], max_concurrency=0)
        with pytest.raises(TaskGraphValidationError):
            TaskGraph(graph_id="bad_bool", execution_id=exec_id, nodes=[], max_concurrency=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            NativeOrchestrator(max_concurrency=0)

        nodes = [TaskNode(node_id=f"n{i}", proposal_ids=[f"p{i}"], order_index=i) for i in range(5)]
        files = [tmp_path / f"f{i}.txt" for i in range(5)]
        props = [
            _make_proposal(exec_id, f"p{i}", f"c{i}", "manage_workspace_file", {"action": "write", "path": str(files[i]), "content": str(i)}, order=i)
            for i in range(5)
        ]
        graph = TaskGraph(graph_id="tg_bound", execution_id=exec_id, nodes=nodes, max_concurrency=2)

        active_now = 0
        max_observed = 0
        wave_first_two = asyncio.Event()
        started_cnt = 0

        async def _exec(tool_name: str, args: dict) -> dict:
            nonlocal active_now, max_observed, started_cnt
            active_now += 1
            started_cnt += 1
            max_observed = max(max_observed, active_now)
            if started_cnt == 2:
                wave_first_two.set()
            if started_cnt <= 2:
                await asyncio.wait_for(wave_first_two.wait(), timeout=2.0)
            await asyncio.sleep(0.01)
            Path(args["path"]).write_text(args["content"], encoding="utf-8")
            active_now -= 1
            return {"status": "success", "output": "ok"}

        scope = TaskLocalToolScope(
            tool_names=["manage_workspace_file"],
            tool_schemas=_build_schemas("manage_workspace_file"),
        )
        orch = NativeOrchestrator(central_ledger=central, task_local_scope=scope, tool_executor=_bind_scope(_exec, scope))
        res = await orch.execute_task_graph(
            graph,
            props,
            max_concurrency=2,
            acceptance_criteria="; ".join(f"file_exists:{f}" for f in files),
        )
        assert max_observed == 2
        assert res["graph_status"] == TaskGraphStatus.COMPLETED_VERIFIED.value
        assert all(f.exists() for f in files)
        central.close()


class TestConflictDetectionAndReservations:
    """Verify conflict detection, shared exclusive resources, atomic locking, and deadlock prevention."""

    @pytest.mark.asyncio
    async def test_4_conflicting_file_writes_and_dir_containment_serialize_while_concurrent_reads_overlap(self, tmp_path: Path):
        shared_dir = tmp_path / "workspace"
        shared_dir.mkdir()
        target_a = shared_dir / "shared.txt"
        nested_a = shared_dir / "." / "shared.txt"

        # Case A: Two independent READY nodes writing aliased same path must serialize under max_concurrency=2
        exec_w = "exec_write_conflict"
        central = CentralExecutionLedger(db_path=tmp_path / "w.db")
        g_write = TaskGraph(
            graph_id="tg_w",
            execution_id=exec_w,
            max_concurrency=2,
            nodes=[
                TaskNode(node_id="w1", proposal_ids=["pw1"], order_index=0),
                TaskNode(node_id="w2", proposal_ids=["pw2"], order_index=1),
                TaskNode(node_id="w_parent", proposal_ids=["pw3"], order_index=2),
            ],
        )
        props_w = [
            _make_proposal(exec_w, "pw1", "cw1", "manage_workspace_file", {"action": "write", "path": str(target_a), "content": "one"}, order=0),
            _make_proposal(exec_w, "pw2", "cw2", "manage_workspace_file", {"action": "overwrite", "path": str(nested_a), "content": "two"}, order=1),
            _make_proposal(exec_w, "pw3", "cw3", "manage_workspace_file", {"action": "write", "path": str(shared_dir), "content": "dir"}, order=2),
        ]
        active_w = 0
        max_w = 0

        async def _exec_w(tool_name: str, args: dict) -> dict:
            nonlocal active_w, max_w
            active_w += 1
            max_w = max(max_w, active_w)
            await asyncio.sleep(0.02)
            active_w -= 1
            return {"status": "success", "output": "ok", "verified": True}

        scope_w = TaskLocalToolScope(tool_names=["manage_workspace_file"], tool_schemas=_build_schemas("manage_workspace_file"))
        orch_w = NativeOrchestrator(central_ledger=central, task_local_scope=scope_w, tool_executor=_bind_scope(_exec_w, scope_w))
        await orch_w.execute_task_graph(g_write, props_w)
        assert max_w == 1

        # Case B: Concurrent READ operations on the same file are safe and DO run concurrently
        target_a.write_text("readable", encoding="utf-8")
        exec_r = "exec_read_overlap"
        g_read = TaskGraph(
            graph_id="tg_r",
            execution_id=exec_r,
            max_concurrency=2,
            nodes=[
                TaskNode(node_id="r1", proposal_ids=["pr1"], order_index=0),
                TaskNode(node_id="r2", proposal_ids=["pr2"], order_index=1),
            ],
        )
        props_r = [
            _make_proposal(exec_r, "pr1", "cr1", "manage_workspace_file", {"action": "read", "path": str(target_a)}, order=0),
            _make_proposal(exec_r, "pr2", "cr2", "manage_workspace_file", {"action": "read", "path": str(target_a)}, order=1),
        ]
        active_r = 0
        max_r = 0
        both_reads = asyncio.Event()

        async def _exec_r(tool_name: str, args: dict) -> dict:
            nonlocal active_r, max_r
            active_r += 1
            max_r = max(max_r, active_r)
            if active_r == 2:
                both_reads.set()
            await asyncio.wait_for(both_reads.wait(), timeout=2.0)
            content = Path(args["path"]).read_text(encoding="utf-8")
            active_r -= 1
            return {"status": "success", "output": content, "verified": True}

        orch_r = NativeOrchestrator(central_ledger=central, task_local_scope=scope_w, tool_executor=_bind_scope(_exec_r, scope_w))
        await orch_r.execute_task_graph(g_read, props_r)
        assert max_r == 2
        central.close()

    @pytest.mark.asyncio
    async def test_5_exclusive_gui_browser_shell_and_duplicate_proposals_serialize(self, tmp_path: Path):
        central = CentralExecutionLedger(db_path=tmp_path / "excl.db")

        for label, t1, a1, t2, a2 in [
            ("gui", "interact_with_ui", {"action": "click", "target": "btn1"}, "gui_actuate", {"action": "press", "target": "btn2"}),
            ("browser", "interact_with_browser", {"action": "navigate", "url": "https://a.example"}, "playwright_web", {"action": "click", "target": "#go"}),
            ("shell", "execute_command", {"command": "echo one"}, "bash", {"command": "echo two"}),
            ("dup_write", "manage_workspace_file", {"action": "write", "path": str(tmp_path / "dup.txt"), "content": "same"},
             "manage_workspace_file", {"action": "write", "path": str(tmp_path / "dup.txt"), "content": "same"}),
        ]:
            exec_id = f"exec_{label}"
            g = TaskGraph(
                graph_id=f"tg_{label}",
                execution_id=exec_id,
                max_concurrency=4,
                nodes=[
                    TaskNode(node_id="n1", proposal_ids=["p1"], order_index=0),
                    TaskNode(node_id="n2", proposal_ids=["p2"], order_index=1),
                ],
            )
            props = [
                _make_proposal(exec_id, "p1", "c1", t1, a1, order=0),
                _make_proposal(exec_id, "p2", "c2", t2, a2, order=1),
            ]
            active_cnt = 0
            max_cnt = 0

            async def _exec(tool_name: str, args: dict) -> dict:
                nonlocal active_cnt, max_cnt
                active_cnt += 1
                max_cnt = max(max_cnt, active_cnt)
                await asyncio.sleep(0.015)
                active_cnt -= 1
                return {"status": "success", "output": "done", "verified": True}

            scope = TaskLocalToolScope(tool_names=[t1, t2], tool_schemas=_build_schemas(t1, t2))
            orch = NativeOrchestrator(central_ledger=central, task_local_scope=scope, tool_executor=_bind_scope(_exec, scope))
            await orch.execute_task_graph(g, props)
            assert max_cnt == 1, f"Category '{label}' overlapped concurrently (max={max_cnt})"
        central.close()

    def test_6_atomic_reservation_ownership_non_owner_rejection_and_sorted_deadlock_free_locks(self):
        graph = TaskGraph(
            graph_id="tg_res",
            execution_id="exec_res",
            max_concurrency=3,
            nodes=[
                TaskNode(node_id="n1", proposal_ids=["p1"], conflict_resources=["res:b", "res:a"], order_index=0),
                TaskNode(node_id="n2", proposal_ids=["p2"], conflict_resources=["res:a", "res:b"], order_index=1),
                TaskNode(node_id="n3", proposal_ids=["p3"], conflict_resources=["res:c"], order_index=2),
            ],
        )
        graph.recompute_schedule()

        # Concurrent contention to reserve n1 across two threads: only one succeeds
        results: List[str] = []
        errors: List[Exception] = []

        def _try_reserve(owner: str) -> None:
            try:
                rid = graph.reserve_node("n1", owner_id=owner)
                results.append(f"{owner}:{rid}")
            except Exception as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(_try_reserve, "worker_alpha")
            f2 = pool.submit(_try_reserve, "worker_beta")
            f1.result()
            f2.result()

        assert len(results) == 1
        assert len(errors) == 1
        assert isinstance(errors[0], TaskGraphConflictError)

        winner_owner, winner_rid = results[0].split(":", 1)
        # Canonical sorted lock ordering prevents AB/BA deadlock
        assert graph.get_node("n1").conflict_resources == ["res:a", "res:b"]

        # Overlapping resource on n2 cannot acquire any partial lock
        with pytest.raises(TaskGraphConflictError):
            graph.reserve_node("n2", owner_id="worker_gamma")
        assert graph.get_node("n2").reserved_by is None

        # Unrelated n3 reserves cleanly
        rid3 = graph.reserve_node("n3", owner_id="worker_delta")
        assert graph.get_node("n3").reserved_by == "worker_delta"

        # Non-owner or bad reservation_id cannot release n1's reservation
        with pytest.raises(TaskGraphValidationError):
            graph.release_reservation("n1", owner_id="intruder", reservation_id=winner_rid)
        with pytest.raises(TaskGraphValidationError):
            graph.release_reservation("n1", owner_id=winner_owner, reservation_id="wrong_rid")
        assert graph.get_node("n1").reserved_by == winner_owner

        # Rightful owner releases cleanly, unblocking n2
        graph.release_reservation("n1", owner_id=winner_owner, reservation_id=winner_rid, final_status=TaskNodeStatus.COMPLETED_VERIFIED.value)
        graph.release_reservation("n3", owner_id="worker_delta", reservation_id=rid3, final_status=TaskNodeStatus.COMPLETED_VERIFIED.value)
        rid2 = graph.reserve_node("n2", owner_id="worker_gamma")
        graph.release_reservation("n2", owner_id="worker_gamma", reservation_id=rid2, final_status=TaskNodeStatus.COMPLETED_VERIFIED.value)
        assert graph.active_reservations == {}


class TestFailureTimeoutAuthIsolationAndCancellation:
    """Verify isolated failure handling, timeouts, auth revocation, context isolation, and cancellation."""

    @pytest.mark.asyncio
    async def test_7_node_failure_unverified_and_timeout_do_not_corrupt_unrelated_siblings(self, tmp_path: Path):
        exec_id = "exec_isolated_fails"
        central = CentralExecutionLedger(db_path=tmp_path / "iso.db")
        f_ok = tmp_path / "ok.txt"
        f_unver = tmp_path / "unver_missing.txt"
        f_fail = tmp_path / "fail.txt"
        f_slow = tmp_path / "slow.txt"

        graph = TaskGraph(
            graph_id="tg_iso",
            execution_id=exec_id,
            max_concurrency=4,
            nodes=[
                TaskNode(node_id="n_ok", proposal_ids=["p_ok"], order_index=0),
                TaskNode(node_id="n_fail", proposal_ids=["p_fail"], order_index=1),
                TaskNode(node_id="n_slow", proposal_ids=["p_slow"], order_index=2),
                TaskNode(node_id="n_unver", proposal_ids=["p_unver"], order_index=3),
                TaskNode(node_id="n_dep_ok", proposal_ids=["p_dep_ok"], dependency_ids=["n_ok"], order_index=4),
                TaskNode(node_id="n_dep_fail", proposal_ids=["p_dep_fail"], dependency_ids=["n_fail"], order_index=5),
                TaskNode(node_id="n_dep_unver", proposal_ids=["p_dep_unver"], dependency_ids=["n_unver"], requires_verified_dependencies=True, order_index=6),
            ],
        )
        f_dep_ok = tmp_path / "dep_ok.txt"
        props = [
            _make_proposal(exec_id, "p_ok", "c_ok", "manage_workspace_file", {"action": "write", "path": str(f_ok), "content": "ok"}, order=0),
            _make_proposal(exec_id, "p_fail", "c_fail", "manage_workspace_file", {"action": "write", "path": str(f_fail), "content": "fail"}, order=1),
            _make_proposal(exec_id, "p_slow", "c_slow", "manage_workspace_file", {"action": "write", "path": str(f_slow), "content": "slow"}, order=2),
            _make_proposal(exec_id, "p_unver", "c_unver", "manage_workspace_file", {"action": "write", "path": str(f_unver), "content": "no_disk_write"}, order=3),
            _make_proposal(exec_id, "p_dep_ok", "c_dep_ok", "manage_workspace_file", {"action": "write", "path": str(f_dep_ok), "content": "dep_ok"}, order=4, depends_on=["p_ok"]),
            _make_proposal(exec_id, "p_dep_fail", "c_dep_fail", "manage_workspace_file", {"action": "write", "path": str(tmp_path / "x1.txt"), "content": "x1"}, order=5, depends_on=["p_fail"]),
            _make_proposal(exec_id, "p_dep_unver", "c_dep_unver", "manage_workspace_file", {"action": "write", "path": str(tmp_path / "x2.txt"), "content": "x2"}, order=6, depends_on=["p_unver"]),
        ]

        async def _exec(tool_name: str, args: dict) -> dict:
            p = Path(args["path"])
            if p == f_fail:
                raise RuntimeError("disk fault on fail branch")
            if p == f_slow:
                await asyncio.sleep(5.0)
                return {"status": "success", "output": "never"}
            if p == f_unver:
                return {"status": "success", "output": "claimed write"}
            p.write_text(args["content"], encoding="utf-8")
            return {"status": "success", "output": f"wrote {p.name}"}

        scope = TaskLocalToolScope(tool_names=["manage_workspace_file"], tool_schemas=_build_schemas("manage_workspace_file"))
        orch = NativeOrchestrator(central_ledger=central, task_local_scope=scope, tool_executor=_bind_scope(_exec, scope))

        res = await orch.execute_task_graph(
            graph,
            props,
            acceptance_criteria=f"file_exists:{f_ok}; file_exists:{f_dep_ok}; file_exists:{f_unver}",
            node_timeouts={"n_slow": 0.05},
        )
        assert graph.get_node("n_ok").execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value
        assert graph.get_node("n_dep_ok").execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value
        assert f_dep_ok.exists()
        assert graph.get_node("n_fail").execution_status == TaskNodeStatus.FAILED.value
        assert graph.get_node("n_slow").execution_status == TaskNodeStatus.FAILED.value
        assert "timed out" in str((graph.get_node("n_slow").failure_info or {}).get("error", "")).lower()
        assert graph.get_node("n_unver").execution_status == TaskNodeStatus.BLOCKED.value
        assert graph.get_node("n_unver").contradictory is True
        assert graph.get_node("n_dep_fail").execution_status == TaskNodeStatus.BLOCKED.value
        assert graph.get_node("n_dep_unver").execution_status == TaskNodeStatus.BLOCKED.value

        # Also verify plain COMPLETED_UNVERIFIED (without contradiction) blocks dependent requiring verified dep
        exec_uv = "exec_plain_unverified"
        g_uv = TaskGraph(
            graph_id="tg_uv",
            execution_id=exec_uv,
            max_concurrency=2,
            nodes=[
                TaskNode(node_id="u1", proposal_ids=["pu1"], order_index=0),
                TaskNode(node_id="u_dep", proposal_ids=["pu_dep"], dependency_ids=["u1"], requires_verified_dependencies=True, order_index=1),
                TaskNode(node_id="u_indep", proposal_ids=["pu_indep"], order_index=2),
            ],
        )
        f_uv_indep = tmp_path / "uv_indep.txt"
        props_uv = [
            _make_proposal(exec_uv, "pu1", "cu1", "manage_workspace_file", {"action": "write", "path": str(tmp_path / "uv1.txt"), "content": "uv"}, order=0),
            _make_proposal(exec_uv, "pu_dep", "cu_dep", "manage_workspace_file", {"action": "write", "path": str(tmp_path / "uv_dep.txt"), "content": "dep"}, order=1, depends_on=["pu1"]),
            _make_proposal(exec_uv, "pu_indep", "cu_indep", "manage_workspace_file", {"action": "write", "path": str(f_uv_indep), "content": "ok"}, order=2),
        ]
        await orch.execute_task_graph(g_uv, props_uv, execution_id=exec_uv, acceptance_criteria=f"file_exists:{f_uv_indep}")
        assert g_uv.get_node("u1").execution_status == TaskNodeStatus.COMPLETED_UNVERIFIED.value
        assert g_uv.get_node("u_dep").execution_status == TaskNodeStatus.BLOCKED.value
        assert "verification" in str(g_uv.get_node("u_dep").blocked_reason or "").lower()
        assert g_uv.get_node("u_indep").execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value
        assert not (tmp_path / "uv_dep.txt").exists()
        assert not (tmp_path / "x1.txt").exists()
        assert not (tmp_path / "x2.txt").exists()
        assert graph.active_reservations == {}
        assert res["turn_ledger"].get_structured_observation("c_ok").evidence_level == "CONFIRMED"
        assert res["turn_ledger"].get_structured_observation("c_slow").execution_status == "timed_out"
        central.close()

    @pytest.mark.asyncio
    async def test_8_per_node_auth_external_availability_and_contextvar_isolation(self, tmp_path: Path):
        exec_id = "exec_auth_ctx"
        central = CentralExecutionLedger(db_path=tmp_path / "auth.db")
        reg = ToolRegistry()

        seen_corr: Dict[str, str] = {}
        both_safe = asyncio.Event()

        class _SafeWriteTool(BaseTool):
            tool_id = "safe_target_tool"
            name = "safe_target_tool"
            description = "Writes target payload safely."
            required_ring = 2

            async def execute(self, target: str, payload: str = "") -> ToolResult:
                ctx = get_active_execution_context()
                assert ctx is not None and ctx.execution_id == exec_id
                seen_corr[target] = str(ctx.correlation_id)
                if len(seen_corr) == 2:
                    both_safe.set()
                await asyncio.wait_for(both_safe.wait(), timeout=2.0)
                Path(target).write_text(f"{ctx.correlation_id}:{payload}", encoding="utf-8")
                return ToolResult(success=True, output=ctx.correlation_id)

        class _PrivilegedTool(BaseTool):
            tool_id = "ring0_admin_tool"
            name = "ring0_admin_tool"
            description = "Privileged ring 0 tool."
            required_ring = 0

            async def execute(self, target: str) -> ToolResult:
                raise AssertionError("Denied ring 0 tool should never execute!")

        class _RevokedExternalTool(BaseTool):
            tool_id = "ext_removed_tool"
            name = "ext_removed_tool"
            description = "External tool revoked mid-flight."
            required_ring = 2
            is_external = True

            async def execute(self, target: str) -> ToolResult:
                raise AssertionError("Revoked external tool should never execute!")

        reg.register(_SafeWriteTool())
        reg.register(_PrivilegedTool())
        reg.register(_RevokedExternalTool())

        t1 = tmp_path / "t1.txt"
        t2 = tmp_path / "t2.txt"
        t3 = tmp_path / "t3.txt"
        t4 = tmp_path / "t4.txt"

        graph = TaskGraph(
            graph_id="tg_auth",
            execution_id=exec_id,
            max_concurrency=4,
            nodes=[
                TaskNode(node_id="n1", proposal_ids=["p1"], order_index=0),
                TaskNode(node_id="n2", proposal_ids=["p2"], order_index=1),
                TaskNode(node_id="n_denied", proposal_ids=["p_denied"], order_index=2),
                TaskNode(
                    node_id="n_ext_unavail",
                    proposal_ids=["p_ext"],
                    external_source={"is_external": True, "source_id": "mcp:gone", "source_type": "mcp"},
                    order_index=3,
                ),
            ],
        )
        props = [
            _make_proposal(exec_id, "p1", "call_iso_1", "safe_target_tool", {"target": str(t1), "payload": "one"}, order=0),
            _make_proposal(exec_id, "p2", "call_iso_2", "safe_target_tool", {"target": str(t2), "payload": "two"}, order=1),
            _make_proposal(exec_id, "p_denied", "call_iso_denied", "ring0_admin_tool", {"target": str(t3)}, order=2),
            _make_proposal(
                exec_id,
                "p_ext",
                "call_iso_ext",
                "ext_removed_tool",
                {"target": str(t4)},
                order=3,
                external_source={"is_external": True, "source_id": "mcp:gone", "source_type": "mcp"},
            ),
        ]

        schemas = _build_schemas("safe_target_tool", "ring0_admin_tool", "ext_removed_tool")
        # Execute directly through ToolRegistry.execute_async (tool_executor=None)
        orch = NativeOrchestrator(central_ledger=central, tool_registry=reg, caller_ring=2, tool_schemas=schemas)
        res = await orch.execute_task_graph(
            graph,
            props,
            acceptance_criteria=f"file_exists:{t1}; file_exists:{t2}",
        )
        assert seen_corr[str(t1)] == "call_iso_1"
        assert seen_corr[str(t2)] == "call_iso_2"
        assert t1.read_text(encoding="utf-8") == "call_iso_1:one"
        assert t2.read_text(encoding="utf-8") == "call_iso_2:two"
        assert graph.get_node("n_denied").execution_status == TaskNodeStatus.FAILED.value
        assert graph.get_node("n_ext_unavail").execution_status == TaskNodeStatus.FAILED.value
        assert not t3.exists() and not t4.exists()

        ev_c1 = [e for e in central.get_events(exec_id) if e.correlation_id == "call_iso_1"]
        ev_c2 = [e for e in central.get_events(exec_id) if e.correlation_id == "call_iso_2"]
        ev_den = [e for e in central.get_events(exec_id) if e.correlation_id == "call_iso_denied"]
        assert any(e.event_type == LedgerEventType.TOOL_EXECUTION_START.value for e in ev_c1)
        assert any(e.event_type == LedgerEventType.VERIFICATION_RECORDED.value for e in ev_c2)
        assert any(e.event_type == LedgerEventType.AUTHORIZATION_DECISION.value and e.status == "denied" for e in ev_den)
        assert res["turn_ledger"].get_record("call_iso_denied").state in (ActionState.FAILED, ActionState.REJECTED)
        central.close()

    @pytest.mark.asyncio
    async def test_9_cancellation_drains_all_workers_and_releases_every_lock(self, tmp_path: Path):
        exec_id = "exec_cancel"
        central = CentralExecutionLedger(db_path=tmp_path / "cancel.db")

        graph = TaskGraph(
            graph_id="tg_cancel",
            execution_id=exec_id,
            max_concurrency=2,
            nodes=[
                TaskNode(node_id="c1", proposal_ids=["pc1"], order_index=0),
                TaskNode(node_id="c2", proposal_ids=["pc2"], order_index=1),
                TaskNode(node_id="c3", proposal_ids=["pc3"], dependency_ids=["c1", "c2"], order_index=2),
            ],
        )
        props = [
            _make_proposal(exec_id, "pc1", "cc1", "manage_workspace_file", {"action": "write", "path": str(tmp_path / "c1.txt"), "content": "1"}, order=0),
            _make_proposal(exec_id, "pc2", "cc2", "manage_workspace_file", {"action": "write", "path": str(tmp_path / "c2.txt"), "content": "2"}, order=1),
            _make_proposal(exec_id, "pc3", "cc3", "manage_workspace_file", {"action": "write", "path": str(tmp_path / "c3.txt"), "content": "3"}, order=2, depends_on=["pc1", "pc2"]),
        ]

        started_both = asyncio.Event()
        started_workers = 0
        cancelled_workers = 0

        async def _slow_exec(tool_name: str, args: dict) -> dict:
            nonlocal started_workers, cancelled_workers
            started_workers += 1
            if started_workers == 2:
                started_both.set()
            try:
                await asyncio.sleep(10.0)
                return {"status": "success", "output": "done"}
            except asyncio.CancelledError:
                cancelled_workers += 1
                raise

        scope = TaskLocalToolScope(tool_names=["manage_workspace_file"], tool_schemas=_build_schemas("manage_workspace_file"))
        orch = NativeOrchestrator(central_ledger=central, task_local_scope=scope, tool_executor=_bind_scope(_slow_exec, scope))

        runner = asyncio.create_task(orch.execute_task_graph(graph, props))
        await asyncio.wait_for(started_both.wait(), timeout=2.0)

        # Mid-flight checkpoint reflects both active reservations while running
        mid_cp = central.get_latest_checkpoint(exec_id, verify_integrity=True)
        assert mid_cp is not None
        mid_tg = mid_cp["task_graph"]
        assert mid_tg["max_concurrency"] == 2
        assert len(mid_tg["active_reservations"]) == 2

        runner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await runner

        assert cancelled_workers == 2
        assert graph.active_reservations == {}
        assert all(n.reserved_by is None for n in graph.nodes)
        assert graph.get_node("c1").execution_status == TaskNodeStatus.CANCELLED.value
        assert graph.get_node("c2").execution_status == TaskNodeStatus.CANCELLED.value
        assert graph.get_node("c3").execution_status == TaskNodeStatus.BLOCKED.value
        rec = central.get_execution(exec_id)
        assert rec is not None and rec.status == ExecutionStatus.CANCELLED.value
        central.close()


class TestPersistenceRestartRecoveryAndCLI:
    """Verify checkpoint integrity, crash recovery reconciliation, and CLI --graph visibility."""

    @pytest.mark.asyncio
    async def test_10_restart_reconciles_inflight_vs_stale_unstarted_reservations_and_skips_verified(self, tmp_path: Path):
        db_path = tmp_path / "restart.db"
        central = CentralExecutionLedger(db_path=db_path)
        exec_id = "exec_crash_parallel"
        f_done = tmp_path / "done.txt"
        f_done.write_text("already_done", encoding="utf-8")
        f_inflight = tmp_path / "inflight.txt"
        f_stale = tmp_path / "stale.txt"
        f_ready = tmp_path / "ready.txt"

        central.create_execution(
            task_prompt="Parallel recovery task",
            execution_id=exec_id,
            status=ExecutionStatus.RUNNING,
        )
        turn = TurnExecutionLedger(execution_id=exec_id, central_ledger=central)
        turn.record_proposed("cd", "manage_workspace_file", {"action": "write", "path": str(f_done), "content": "done"}, proposal_id="pd")
        turn.record_validated("cd")
        turn.record_authorized("cd")
        turn.record_dispatched("cd")
        turn.record_observation("cd", status="success", output="ok", verified=True)
        turn.record_target_verified(str(f_done), check_type="file_exists")

        # In-flight call started without result
        turn.record_proposed("ci", "manage_workspace_file", {"action": "write", "path": str(f_inflight), "content": "inflight"}, proposal_id="pi")
        turn.record_validated("ci")
        turn.record_authorized("ci")
        turn.record_dispatched("ci")

        # Pre-dispatch stale reservation (never dispatched!) and disjoint ready call
        turn.record_proposed("cs", "manage_workspace_file", {"action": "write", "path": str(f_stale), "content": "stale"}, proposal_id="ps")
        turn.record_validated("cs")
        turn.record_proposed("cr", "manage_workspace_file", {"action": "write", "path": str(f_ready), "content": "ready"}, proposal_id="pr")
        turn.record_validated("cr")
        turn.record_proposed("cdep", "manage_workspace_file", {"action": "write", "path": str(tmp_path / "dep.txt"), "content": "dep"}, proposal_id="pdep", depends_on=["pi"])

        tg = TaskGraph(
            graph_id="tg_crash",
            execution_id=exec_id,
            objective="Crash reconciliation",
            max_concurrency=3,
            nodes=[
                TaskNode(node_id="n_done", proposal_ids=["pd"], execution_status=TaskNodeStatus.COMPLETED_VERIFIED.value, verification_status=VerificationStatus.VERIFIED.value, order_index=0),
                TaskNode(
                    node_id="n_inflight",
                    proposal_ids=["pi"],
                    execution_status=TaskNodeStatus.RUNNING.value,
                    verification_status=VerificationStatus.INTERRUPTED.value,
                    reserved_by="dead_worker_1",
                    reservation_id="res_dead_1",
                    conflict_resources=[f"target:{f_inflight}"],
                    order_index=1,
                ),
                TaskNode(
                    node_id="n_stale_unstarted",
                    proposal_ids=["ps"],
                    execution_status=TaskNodeStatus.RUNNING.value,
                    reserved_by="dead_worker_2",
                    reservation_id="res_dead_2",
                    conflict_resources=[f"target:{f_stale}"],
                    order_index=2,
                ),
                TaskNode(node_id="n_ready", proposal_ids=["pr"], execution_status=TaskNodeStatus.READY.value, order_index=3),
                TaskNode(node_id="n_dep_inflight", proposal_ids=["pdep"], dependency_ids=["n_inflight"], order_index=4),
            ],
            active_reservations={
                "n_inflight": {"node_id": "n_inflight", "owner_id": "dead_worker_1", "reservation_id": "res_dead_1", "resources": [f"target:{f_inflight}"], "write_targets": [str(f_inflight)]},
                "n_stale_unstarted": {"node_id": "n_stale_unstarted", "owner_id": "dead_worker_2", "reservation_id": "res_dead_2", "resources": [f"target:{f_stale}"], "write_targets": [str(f_stale)]},
            },
        )
        cp = CheckpointManager.capture_checkpoint(
            central,
            exec_id,
            turn_ledger=turn,
            status=ExecutionStatus.INTERRUPTED,
            planning_state={"max_concurrency": 3, "acceptance_criteria": f"file_exists:{f_done}; file_exists:{f_ready}"},
            task_graph=tg,
        )
        assert cp is not None and cp.task_graph["max_concurrency"] == 3

        # Tampering with persisted active_reservations or max_concurrency fails integrity verification
        bad_tg_dict = dict(cp.task_graph)
        bad_tg_dict["max_concurrency"] = 99
        with pytest.raises(CorruptedTaskGraphError):
            TaskGraph.from_dict(bad_tg_dict, verify_integrity=True)

        # Simulate fresh process opening same DB
        central.close()
        reopened = CentralExecutionLedger(db_path=db_path)
        scope = TaskLocalToolScope(tool_names=["manage_workspace_file"], tool_schemas=_build_schemas("manage_workspace_file"))
        assessment = CheckpointManager.assess_resume(reopened, exec_id, task_local_scope=scope)

        assert assessment.can_resume is True
        assert assessment.resumability_status == ResumabilityStatus.NEEDS_VERIFICATION.value
        rg = assessment.task_graph
        assert rg is not None
        assert rg.max_concurrency == 3
        assert rg.get_node("n_done").execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value
        assert rg.get_node("n_inflight").execution_status == TaskNodeStatus.RUNNING.value
        # Unstarted dead reservation released automatically so n_stale_unstarted is READY again
        assert rg.get_node("n_stale_unstarted").reserved_by is None
        assert rg.get_node("n_stale_unstarted").execution_status == TaskNodeStatus.READY.value
        assert "n_stale_unstarted" not in rg.active_reservations
        # Dependent on ambiguous in-flight stays blocked until prerequisite verified
        assert rg.get_node("n_dep_inflight").execution_status == TaskNodeStatus.BLOCKED.value

        # Verify CLI --graph and --graph --json visibility BEFORE resuming (shows running owner & ready nodes)
        cli_text = subprocess.run(
            [sys.executable, "-m", "zelyr.cli.main", "execution", exec_id, "--graph", "--db-path", str(db_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert cli_text.returncode == 0, cli_text.stderr
        assert "Max Concurrency:     3" in cli_text.stdout
        assert "Running Nodes:" in cli_text.stdout
        assert "Ready Nodes:" in cli_text.stdout
        assert "Active Reservations (1):" in cli_text.stdout
        assert "owner=dead_worker_1" in cli_text.stdout

        cli_json = subprocess.run(
            [sys.executable, "-m", "zelyr.cli.main", "execution", exec_id, "--graph", "--json", "--db-path", str(db_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert cli_json.returncode == 0, cli_json.stderr
        parsed_cli = json.loads(cli_json.stdout)
        assert parsed_cli["task_graph"]["max_concurrency"] == 3
        assert "n_inflight" in parsed_cli["task_graph"]["active_reservations"]

        # Resume parallel execution: n_done skipped, n_inflight not blindly replayed, n_ready + n_stale execute
        executed_calls: List[str] = []

        async def _resume_exec(tool_name: str, args: dict) -> dict:
            p = Path(args["path"])
            executed_calls.append(p.name)
            p.write_text(args["content"], encoding="utf-8")
            return {"status": "success", "output": f"resumed {p.name}", "verified": True}

        scoped_resume_exec = _bind_scope(_resume_exec, scope)
        orch = NativeOrchestrator(central_ledger=reopened, task_local_scope=scope, tool_executor=scoped_resume_exec)
        _, chunks = await CheckpointManager.resume_execution(
            orch,
            reopened,
            exec_id,
            explicit_resume=True,
            task_local_scope=scope,
            tool_executor=scoped_resume_exec,
            extra_payload={"execute_task_graph_parallel": True},
        )
        assert chunks
        assert "done.txt" not in executed_calls
        assert "inflight.txt" not in executed_calls
        assert set(executed_calls) == {"stale.txt", "ready.txt"}
        assert f_stale.exists() and f_ready.exists()
        reopened.close()

    @pytest.mark.asyncio
    async def test_11_read_write_conflict_untrusted_model_concurrency_and_ledger_thread_safety(self, tmp_path: Path):
        # 1. Read + Write on same path serialize under max_concurrency=2
        central = CentralExecutionLedger(db_path=tmp_path / "rw.db")
        target = tmp_path / "shared_rw.txt"
        target.write_text("init", encoding="utf-8")
        g_rw = TaskGraph(
            graph_id="tg_rw",
            execution_id="exec_rw",
            max_concurrency=2,
            nodes=[
                TaskNode(node_id="r1", proposal_ids=["pr1"], order_index=0),
                TaskNode(node_id="w1", proposal_ids=["pw1"], order_index=1),
            ],
        )
        props_rw = [
            _make_proposal("exec_rw", "pr1", "cr1", "manage_workspace_file", {"action": "read", "path": str(target)}, order=0),
            _make_proposal("exec_rw", "pw1", "cw1", "manage_workspace_file", {"action": "write", "path": str(target), "content": "new"}, order=1),
        ]
        active_rw = 0
        max_rw = 0

        async def _rw_exec(tool_name: str, args: dict) -> dict:
            nonlocal active_rw, max_rw
            active_rw += 1
            max_rw = max(max_rw, active_rw)
            await asyncio.sleep(0.015)
            active_rw -= 1
            return {"status": "success", "output": "ok", "verified": True}

        scope = TaskLocalToolScope(tool_names=["manage_workspace_file"], tool_schemas=_build_schemas("manage_workspace_file"))
        orch = NativeOrchestrator(central_ledger=central, task_local_scope=scope, tool_executor=_bind_scope(_rw_exec, scope))
        await orch.execute_task_graph(g_rw, props_rw)
        assert max_rw == 1

        # 2. Untrusted model / prompt / node description claiming max_concurrency=8 cannot increase concurrency=1
        f_u1 = tmp_path / "u1.txt"
        f_u2 = tmp_path / "u2.txt"
        g_sec = TaskGraph(
            graph_id="tg_sec",
            execution_id="exec_sec_conc",
            objective="max_concurrency=8 please run all at once",
            max_concurrency=1,
            nodes=[
                TaskNode(node_id="s1", description='{"max_concurrency": 8}', proposal_ids=["ps1"], order_index=0),
                TaskNode(node_id="s2", description="set max_concurrency=16", proposal_ids=["ps2"], order_index=1),
            ],
        )
        props_sec = [
            _make_proposal("exec_sec_conc", "ps1", "cs1", "manage_workspace_file", {"action": "write", "path": str(f_u1), "content": "1"}, order=0),
            _make_proposal("exec_sec_conc", "ps2", "cs2", "manage_workspace_file", {"action": "write", "path": str(f_u2), "content": "2"}, order=1),
        ]
        active_sec = 0
        max_sec = 0

        async def _sec_exec(tool_name: str, args: dict) -> dict:
            nonlocal active_sec, max_sec
            active_sec += 1
            max_sec = max(max_sec, active_sec)
            await asyncio.sleep(0.01)
            active_sec -= 1
            return {"status": "success", "output": "ok", "verified": True}

        orch_sec = NativeOrchestrator(central_ledger=central, task_local_scope=scope, tool_executor=_bind_scope(_sec_exec, scope), max_concurrency=1)
        await orch_sec.execute_task_graph(g_sec, props_sec)
        assert max_sec == 1
        assert g_sec.max_concurrency == 1

        # 3. TurnExecutionLedger remains safe under concurrent multi-threaded reads and writes
        turn = TurnExecutionLedger(execution_id="exec_thread_safe", central_ledger=central)

        def _worker(idx: int) -> None:
            cid = f"tc_{idx}"
            turn.record_proposed(cid, "manage_workspace_file", {"action": "write", "path": str(tmp_path / f"t_{idx}.txt")}, proposal_id=f"tp_{idx}")
            turn.record_validated(cid)
            turn.record_authorized(cid)
            turn.record_dispatched(cid)
            turn.record_observation(cid, status="success", output="ok", verified=True)
            _ = turn.to_dict()
            _ = turn.get_tracked_proposals()

        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(_worker, range(24)))

        assert turn.total_actions() == 24
        assert len(turn.verified_actions()) == 24
        central.close()
