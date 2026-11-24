"""Deterministic offline test suite for Verification-Driven Graph Recovery."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List

import pytest

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.config import ZelyrConfig
from zelyr.engine import (
    CentralExecutionLedger,
    CheckpointManager,
    CorruptedCheckpointError,
    CorruptedRecoveryStateError,
    CorruptedTaskGraphError,
    ExecutionStatus,
    LedgerEventType,
    ObservationFramingEngine,
    PendingActionProposal,
    RecoveryDecision,
    RecoveryEngine,
    RecoveryError,
    RecoveryPolicy,
    ResumabilityStatus,
    TaskCheckpoint,
    TaskGraph,
    TaskGraphConflictError,
    TaskGraphStatus,
    TaskGraphValidationError,
    TaskNode,
    TaskNodeStatus,
    TurnExecutionLedger,
    UnsupportedCheckpointVersionError,
    UnsupportedRecoveryVersionError,
    UnsupportedTaskGraphVersionError,
    VerificationStatus,
    compute_task_graph_digest,
)
from zelyr.planning import PlanValidationError
from zelyr.tool_registry import TaskLocalToolScope, ToolRegistry, bind_task_local_executor
from zelyr.tools.core import BaseTool, ToolResult


def _build_schemas(
    *names: str,
    read_only_names: set[str] | None = None,
    idempotent_names: set[str] | None = None,
) -> List[Dict[str, Any]]:
    ro = read_only_names or set()
    idem = idempotent_names or set()
    out: List[Dict[str, Any]] = []
    for nm in names:
        fn: Dict[str, Any] = {
            "name": nm,
            "description": f"Test schema for {nm}",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "target": {"type": "string"},
                    "mode": {"type": "string"},
                },
                "additionalProperties": True,
            },
        }
        if nm in ro:
            fn["read_only"] = True
        if nm in idem:
            fn["idempotent"] = True
        out.append({"type": "function", "function": fn})
    return out


def _bind_scope(fn: Callable[..., Any], scope: TaskLocalToolScope) -> Callable[..., Any]:
    return bind_task_local_executor(
        fn,
        tool_names=scope.tool_names,
        tool_schemas=scope.tool_schemas,
        tools_dir=scope.tools_dir,
        tool_rings=scope.tool_rings,
    )


def _make_prop(
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
    )


class TestRecoveryDecisionsAndPolicies:
    @pytest.mark.asyncio
    async def test_1_safe_retry_after_failure_before_execution(self, tmp_path: Path):
        exec_id = "exec_safe_pre_retry"
        central = CentralExecutionLedger(db_path=tmp_path / "safe_pre.db")
        f_cfg = tmp_path / "config.json"
        f_dep = tmp_path / "app.log"

        graph = TaskGraph(
            graph_id="tg_safe_pre",
            execution_id=exec_id,
            max_concurrency=2,
            nodes=[
                TaskNode(
                    node_id="n_cfg",
                    proposal_ids=["p_cfg"],
                    verification_requirement=f"file_exists:{f_cfg}",
                    order_index=0,
                ),
                TaskNode(
                    node_id="n_dep",
                    proposal_ids=["p_dep"],
                    dependency_ids=["n_cfg"],
                    requires_verified_dependencies=True,
                    verification_requirement=f"file_exists:{f_dep}",
                    order_index=1,
                ),
            ],
        )
        props = [
            _make_prop(exec_id, "p_cfg", "c_cfg", "manage_workspace_file", {"action": "write", "path": str(f_cfg), "content": "{}"}),
            _make_prop(exec_id, "p_dep", "c_dep", "manage_workspace_file", {"action": "write", "path": str(f_dep), "content": "ready"}, order=1, depends_on=["p_cfg"]),
        ]

        pre_calls = {"n": 0}
        exec_calls: List[str] = []

        async def _exec(name: str, args: dict) -> dict:
            exec_calls.append(Path(args["path"]).name)
            Path(args["path"]).write_text(args["content"], encoding="utf-8")
            return {"status": "success", "output": "ok"}

        def _pre_dispatch(name: str, args: dict) -> dict | None:
            if Path(args["path"]).name == "config.json" and pre_calls["n"] == 0:
                pre_calls["n"] += 1
                return {"status": "error", "error": "transient transport warmup failed", "error_type": "TRANSIENT_FAILURE"}
            return None

        _exec.pre_dispatch = _pre_dispatch  # type: ignore[attr-defined]
        scope = TaskLocalToolScope(tool_names=["manage_workspace_file"], tool_schemas=_build_schemas("manage_workspace_file"))
        orch = NativeOrchestrator(
            central_ledger=central,
            task_local_scope=scope,
            tool_executor=_bind_scope(_exec, scope),
        )
        # Preserve pre_dispatch attribute on bound executor wrapper
        orch.tool_executor.pre_dispatch = _pre_dispatch  # type: ignore[attr-defined]

        cfg = ZelyrConfig(graph_recovery_max_attempts=2, graph_recovery_backoff_base_seconds=0.0, graph_recovery_deadline_seconds=30.0)
        res = await orch.execute_task_graph(
            graph,
            props,
            recovery_policy=RecoveryPolicy.from_config(cfg),
        )
        assert res["graph_status"] == TaskGraphStatus.COMPLETED_VERIFIED.value
        assert graph.get_node("n_cfg").execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value
        assert graph.get_node("n_dep").execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value
        assert f_cfg.exists() and f_dep.exists()
        # config.json only executed once after failing safely before dispatch!
        assert exec_calls == ["config.json", "app.log"]
        rs = graph.get_node("n_cfg").recovery_state
        assert rs is not None and rs["attempt_count"] == 1
        assert rs["recovery_decision"] == RecoveryDecision.RETRY_SAFE.value
        rec_evs = [e for e in central.get_events(exec_id) if e.event_type == LedgerEventType.RECOVERY_ATTEMPT.value]
        assert len(rec_evs) >= 1
        assert rec_evs[0].metadata["recovery_decision"] == RecoveryDecision.RETRY_SAFE.value
        assert rec_evs[0].metadata["attempt"] == 1
        central.close()

    @pytest.mark.asyncio
    async def test_2_retry_after_successful_state_reconciliation(self, tmp_path: Path):
        exec_id = "exec_reconcile_retry"
        central = CentralExecutionLedger(db_path=tmp_path / "reconcile.db")
        turn = TurnExecutionLedger(execution_id=exec_id, central_ledger=central, source_component="test")
        target_f = tmp_path / "service_out.txt"
        reg = ToolRegistry()

        attempts = {"n": 0}

        class _IdempotentProvisionTool(BaseTool):
            tool_id = "idempotent_write"
            name = "idempotent_write"
            description = "Idempotent non-destructive provisioner"
            required_ring = 2
            idempotent = True

            async def execute(self, path: str, content: str = "") -> ToolResult:
                attempts["n"] += 1
                if attempts["n"] == 1:
                    return ToolResult(success=False, error="temporary lock busy", error_type="TRANSIENT_FAILURE")
                Path(path).write_text(content, encoding="utf-8")
                return ToolResult(success=True, output="written")

        reg.register(_IdempotentProvisionTool())
        schemas = _build_schemas("idempotent_write", idempotent_names={"idempotent_write"})
        graph = TaskGraph(
            graph_id="tg_reconcile",
            execution_id=exec_id,
            nodes=[
                TaskNode(
                    node_id="n_prov",
                    proposal_ids=["p_prov"],
                    verification_requirement=f"file_exists:{target_f}",
                    order_index=0,
                )
            ],
        )
        prop = _make_prop(exec_id, "p_prov", "c_prov", "idempotent_write", {"path": str(target_f), "content": "ok"})
        orch = NativeOrchestrator(central_ledger=central, tool_registry=reg, caller_ring=2, tool_schemas=schemas)

        res = await orch.execute_task_graph(graph, [prop], turn_ledger=turn)
        assert res["graph_status"] == TaskGraphStatus.COMPLETED_VERIFIED.value
        assert attempts["n"] == 2
        assert target_f.read_text(encoding="utf-8") == "ok"
        assert graph.get_node("n_prov").recovery_state["recovery_decision"] == RecoveryDecision.RETRY_AFTER_REVALIDATION.value

        # Also test non-idempotent mutator failing after dispatch requiring explicit state reconciliation
        exec2 = "exec_explicit_reconcile"
        central2 = CentralExecutionLedger(db_path=tmp_path / "reconcile2.db")
        central2.create_execution("Explicit reconcile test", execution_id=exec2, status=ExecutionStatus.FAILED)
        turn2 = TurnExecutionLedger(execution_id=exec2, central_ledger=central2, source_component="test")
        f2 = tmp_path / "explicit_target.txt"
        g2 = TaskGraph(
            graph_id="tg_exp_rec",
            execution_id=exec2,
            nodes=[
                TaskNode(
                    node_id="n_w",
                    proposal_ids=["p_w"],
                    execution_status=TaskNodeStatus.FAILED.value,
                    verification_requirement=f"file_exists:{f2}",
                    failure_info={"error": "partial write aborted", "may_have_side_effect": True},
                    recoverable=True,
                )
            ],
        )
        turn2.record_proposed("c_w", "manage_workspace_file", {"action": "write", "path": str(f2), "content": "x"})
        turn2.record_validated("c_w")
        turn2.record_authorized("c_w")
        turn2.record_dispatched("c_w")
        obs = ObservationFramingEngine.frame_structured(
            "manage_workspace_file",
            {"status": "error", "error": "interrupted after dispatch"},
            dispatched=True,
            execution_id=exec2,
            correlation_id="c_w",
        )
        turn2.record_observation("c_w", status="error", output="", error="interrupted after dispatch", structured_observation=obs)
        props2 = {"p_w": _make_prop(exec2, "p_w", "c_w", "manage_workspace_file", {"action": "write", "path": str(f2), "content": "x"})}

        # Without reconciliation, automatic retry is BLOCKED (REQUIRE_CONFIRMATION)
        a_blocked = RecoveryEngine.assess_node_recovery(g2, "n_w", central_ledger=central2, turn_ledger=turn2, proposals=props2)
        assert a_blocked.recovery_decision == RecoveryDecision.REQUIRE_CONFIRMATION.value
        assert a_blocked.reconciliation_required is True

        # After state reconciliation confirms target file does not exist & operation is non-destructive, RETRY_AFTER_REVALIDATION allowed
        a_recon = RecoveryEngine.assess_node_recovery(
            g2,
            "n_w",
            central_ledger=central2,
            turn_ledger=turn2,
            proposals=props2,
            reconcile_state=True,
        )
        assert a_recon.recovery_decision == RecoveryDecision.RETRY_AFTER_REVALIDATION.value
        assert a_recon.reconciliation_required is False
        assert a_recon.revalidation_passed is True
        central.close()
        central2.close()

    @pytest.mark.asyncio
    async def test_3_already_completed_operation_is_not_repeated(self, tmp_path: Path):
        exec_id = "exec_already_done"
        central = CentralExecutionLedger(db_path=tmp_path / "already.db")
        f_done = tmp_path / "already_written.txt"
        f_down = tmp_path / "downstream.txt"

        calls: List[str] = []

        async def _exec(name: str, args: dict) -> dict:
            p = Path(args["path"])
            calls.append(p.name)
            if p == f_done:
                # Simulate side effect succeeded on host right before network/tool error returned!
                p.write_text("VALID_ARTIFACT", encoding="utf-8")
                return {"status": "error", "error": "socket dropped after flush"}
            p.write_text(args["content"], encoding="utf-8")
            return {"status": "success", "output": "ok"}

        graph = TaskGraph(
            graph_id="tg_already",
            execution_id=exec_id,
            nodes=[
                TaskNode(
                    node_id="n_art",
                    proposal_ids=["p_art"],
                    verification_requirement=f"file_contains:{f_done}:VALID_ARTIFACT",
                    order_index=0,
                ),
                TaskNode(
                    node_id="n_next",
                    proposal_ids=["p_next"],
                    dependency_ids=["n_art"],
                    requires_verified_dependencies=True,
                    verification_requirement=f"file_exists:{f_down}",
                    order_index=1,
                ),
            ],
        )
        props = [
            _make_prop(exec_id, "p_art", "c_art", "manage_workspace_file", {"action": "write", "path": str(f_done), "content": "VALID_ARTIFACT"}),
            _make_prop(exec_id, "p_next", "c_next", "manage_workspace_file", {"action": "write", "path": str(f_down), "content": "next"}, order=1, depends_on=["p_art"]),
        ]
        scope = TaskLocalToolScope(tool_names=["manage_workspace_file"], tool_schemas=_build_schemas("manage_workspace_file"))
        orch = NativeOrchestrator(central_ledger=central, task_local_scope=scope, tool_executor=_bind_scope(_exec, scope))

        res = await orch.execute_task_graph(graph, props)
        assert res["graph_status"] == TaskGraphStatus.COMPLETED_VERIFIED.value
        assert graph.get_node("n_art").execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value
        assert graph.get_node("n_art").recovery_state["recovery_decision"] == RecoveryDecision.ALREADY_COMPLETED.value
        assert graph.get_node("n_next").execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value
        # n_art was NEVER repeated!
        assert calls == ["already_written.txt", "downstream.txt"]
        assert f_down.exists()
        central.close()

    @pytest.mark.asyncio
    async def test_4_and_10_ambiguous_side_effects_block_retry_and_isolate_dependents(self, tmp_path: Path):
        exec_id = "exec_ambiguous"
        central = CentralExecutionLedger(db_path=tmp_path / "ambig.db")
        f_mut = tmp_path / "mutator.txt"
        f_dep = tmp_path / "blocked_dep.txt"
        f_indep = tmp_path / "indep.txt"
        exec_counts: Dict[str, int] = {"mut": 0, "dep": 0, "indep": 0}

        async def _exec(name: str, args: dict) -> dict:
            p = Path(args["path"])
            if p == f_mut:
                exec_counts["mut"] += 1
                raise RuntimeError("mid-write disk I/O error")
            if p == f_dep:
                exec_counts["dep"] += 1
                return {"status": "success", "output": "should never run"}
            exec_counts["indep"] += 1
            p.write_text(args["content"], encoding="utf-8")
            return {"status": "success", "output": "ok"}

        graph = TaskGraph(
            graph_id="tg_ambig",
            execution_id=exec_id,
            max_concurrency=2,
            nodes=[
                TaskNode(node_id="n_mut", proposal_ids=["p_mut"], verification_requirement=f"file_exists:{f_mut}", order_index=0),
                TaskNode(node_id="n_dep", proposal_ids=["p_dep"], dependency_ids=["n_mut"], requires_verified_dependencies=True, order_index=1),
                TaskNode(node_id="n_indep", proposal_ids=["p_indep"], verification_requirement=f"file_exists:{f_indep}", order_index=2),
            ],
        )
        props = [
            _make_prop(exec_id, "p_mut", "c_mut", "manage_workspace_file", {"action": "write", "path": str(f_mut), "content": "a"}),
            _make_prop(exec_id, "p_dep", "c_dep", "manage_workspace_file", {"action": "write", "path": str(f_dep), "content": "b"}, order=1, depends_on=["p_mut"]),
            _make_prop(exec_id, "p_indep", "c_indep", "manage_workspace_file", {"action": "write", "path": str(f_indep), "content": "ok"}, order=2),
        ]
        scope = TaskLocalToolScope(tool_names=["manage_workspace_file"], tool_schemas=_build_schemas("manage_workspace_file"))
        orch = NativeOrchestrator(central_ledger=central, task_local_scope=scope, tool_executor=_bind_scope(_exec, scope))
        res = await orch.execute_task_graph(graph, props)

        # Ambiguous failure NEVER auto-retries!
        assert exec_counts["mut"] == 1
        assert exec_counts["dep"] == 0
        assert exec_counts["indep"] == 1
        assert f_indep.exists() and not f_dep.exists()
        assert graph.get_node("n_mut").execution_status == TaskNodeStatus.FAILED.value
        assert graph.get_node("n_mut").recovery_state["recovery_decision"] == RecoveryDecision.REQUIRE_CONFIRMATION.value
        assert graph.get_node("n_mut").recovery_state["reconciliation_required"] is True
        assert graph.get_node("n_mut").recovery_state["may_have_side_effect"] is True
        assert graph.get_node("n_dep").execution_status == TaskNodeStatus.BLOCKED.value
        assert graph.get_node("n_indep").execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value
        assert res["recovery_overview"]["blocked_dependents"][0]["node_id"] == "n_dep"
        central.close()

    @pytest.mark.asyncio
    async def test_5_contradictory_evidence_requires_reconciliation_and_blocks_until_reconciled(self, tmp_path: Path):
        exec_id = "exec_contradict"
        central = CentralExecutionLedger(db_path=tmp_path / "contra.db")
        f_c = tmp_path / "contra.txt"
        f_down = tmp_path / "after_contra.txt"

        async def _exec(name: str, args: dict) -> dict:
            p = Path(args["path"])
            if p == f_c:
                # Lies about succeeding without writing file!
                return {"status": "success", "output": "created file cleanly"}
            p.write_text(args["content"], encoding="utf-8")
            return {"status": "success", "output": "ok"}

        graph = TaskGraph(
            graph_id="tg_contra",
            execution_id=exec_id,
            nodes=[
                TaskNode(node_id="n_c", proposal_ids=["p_c"], verification_requirement=f"file_exists:{f_c}", order_index=0),
                TaskNode(node_id="n_d", proposal_ids=["p_d"], dependency_ids=["n_c"], requires_verified_dependencies=True, order_index=1),
            ],
        )
        props = {
            "p_c": _make_prop(exec_id, "p_c", "c_c", "manage_workspace_file", {"action": "write", "path": str(f_c), "content": "hi"}),
            "p_d": _make_prop(exec_id, "p_d", "c_d", "manage_workspace_file", {"action": "write", "path": str(f_down), "content": "ok"}, order=1, depends_on=["p_c"]),
        }
        scope = TaskLocalToolScope(tool_names=["manage_workspace_file"], tool_schemas=_build_schemas("manage_workspace_file"))
        orch = NativeOrchestrator(central_ledger=central, task_local_scope=scope, tool_executor=_bind_scope(_exec, scope))
        await orch.execute_task_graph(graph, props)

        # Prose claim rejected; contradictory node stays BLOCKED and n_d stays BLOCKED
        n_c = graph.get_node("n_c")
        assert n_c.contradictory is True
        assert n_c.execution_status == TaskNodeStatus.BLOCKED.value
        assert n_c.recovery_state["recovery_decision"] == RecoveryDecision.REQUIRE_CONFIRMATION.value
        assert n_c.recovery_state["reconciliation_required"] is True
        assert graph.get_node("n_d").execution_status == TaskNodeStatus.BLOCKED.value
        assert not f_down.exists()

        # Without reconcile_state=True, assessment still refuses blind promotion/retry
        assess_unrec = orch.assess_node_recovery(graph, "n_c", proposals=props, reconcile_state=False)
        assert assess_unrec.recovery_decision == RecoveryDecision.REQUIRE_CONFIRMATION.value
        assert assess_unrec.reconciliation_required is True

        # Now operator fixes/verifies host state and runs explicit reconciliation -> unlocks downstream!
        f_c.write_text("hi", encoding="utf-8")
        rec_out = orch.recover_graph_node(graph, "n_c", proposals=props, reconcile_state=True)
        assert rec_out["decision"] == RecoveryDecision.ALREADY_COMPLETED.value
        assert graph.get_node("n_c").execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value
        assert graph.get_node("n_c").contradictory is False
        assert graph.get_node("n_d").execution_status == TaskNodeStatus.READY.value
        central.close()

    @pytest.mark.asyncio
    async def test_6_and_7_max_attempts_and_retry_deadline_enforced(self, tmp_path: Path):
        exec_id = "exec_bounds"
        central = CentralExecutionLedger(db_path=tmp_path / "bounds.db")
        f_t = tmp_path / "never_ready.txt"
        pre_count = {"n": 0}

        def _always_pre_fail(name: str, args: dict) -> dict:
            pre_count["n"] += 1
            return {"status": "error", "error": "still offline", "error_type": "TRANSIENT_FAILURE"}

        async def _never_called(name: str, args: dict) -> dict:
            raise AssertionError("pre_dispatch failed, body should never run")

        _never_called.pre_dispatch = _always_pre_fail  # type: ignore[attr-defined]
        scope = TaskLocalToolScope(tool_names=["manage_workspace_file"], tool_schemas=_build_schemas("manage_workspace_file"))
        orch = NativeOrchestrator(central_ledger=central, task_local_scope=scope, tool_executor=_bind_scope(_never_called, scope))
        orch.tool_executor.pre_dispatch = _always_pre_fail  # type: ignore[attr-defined]

        graph = TaskGraph(graph_id="tg_bounds", execution_id=exec_id, nodes=[TaskNode(node_id="n1", proposal_ids=["p1"])])
        props = [_make_prop(exec_id, "p1", "c1", "manage_workspace_file", {"action": "write", "path": str(f_t), "content": "x"})]

        policy = RecoveryPolicy(max_attempts=2, backoff_base_seconds=0.01, backoff_max_seconds=0.02, deadline_seconds=30.0)
        res = await orch.execute_task_graph(graph, props, recovery_policy=policy)

        # 1 initial attempt + 2 retries = 3 invocations total, then stops BLOCKED!
        assert pre_count["n"] == 3
        n1 = graph.get_node("n1")
        assert n1.execution_status == TaskNodeStatus.FAILED.value
        assert n1.recovery_state["attempt_count"] == 2
        assert n1.recovery_state["recovery_decision"] == RecoveryDecision.BLOCKED.value
        assert n1.recovery_state["recovery_status"] == "exhausted"
        assert res["graph_status"] == TaskGraphStatus.FAILED.value

        # Test retry deadline exceeded blocks immediately even when attempts remain
        exec_dl = "exec_deadline"
        g_dl = TaskGraph(
            graph_id="tg_dl",
            execution_id=exec_dl,
            nodes=[
                TaskNode(
                    node_id="nd",
                    proposal_ids=["pd"],
                    execution_status=TaskNodeStatus.FAILED.value,
                    failure_info={"error": "pre-dispatch hiccup", "error_type": "TRANSIENT_FAILURE", "may_have_side_effect": False},
                    recovery_state={
                        "recovery_version": 1,
                        "attempt_count": 0,
                        "max_attempts": 5,
                        "retry_deadline": 100.0,
                        "recovery_history": [],
                    },
                )
            ],
        )
        p_dl = {"pd": _make_prop(exec_dl, "pd", "cd", "manage_workspace_file", {"action": "read", "path": str(f_t)})}
        assess_dl = RecoveryEngine.assess_node_recovery(
            g_dl,
            "nd",
            proposals=p_dl,
            policy=RecoveryPolicy(max_attempts=5),
            now_ts=105.0,
        )
        assert assess_dl.recovery_decision == RecoveryDecision.BLOCKED.value
        assert "deadline" in assess_dl.reason.lower()

        # Invalid unbounded policy rejected cleanly
        with pytest.raises(RecoveryError):
            RecoveryPolicy(max_attempts=999)
        central.close()

    @pytest.mark.asyncio
    async def test_8_authorization_and_external_provenance_revalidated_before_redispatch(self, tmp_path: Path):
        from zelyr.tools.external import MCPExternalToolSource, build_external_tool_id

        exec_id = "exec_auth_rec"
        central = CentralExecutionLedger(db_path=tmp_path / "auth_rec.db")
        reg = ToolRegistry()

        class _MockMCP:
            url = "https://mcp.local/sse"

            async def list_tools(self) -> List[Dict[str, Any]]:
                return [
                    {
                        "name": "ro_probe",
                        "description": "Read-only external probe",
                        "read_only": True,
                        "inputSchema": {
                            "type": "object",
                            "properties": {"target": {"type": "string"}},
                            "required": ["target"],
                            "additionalProperties": False,
                        },
                    }
                ]

            async def call_tool(self, name: str, arguments: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
                return {"success": True, "output": "ok"}

        src = MCPExternalToolSource("srv", _MockMCP())
        reg.register_external_source(src)
        await reg.discover_external_tools("srv")
        ext_tool_id = build_external_tool_id("srv", "ro_probe")
        assert reg.authorize_external_tool(ext_tool_id) is True

        prop = _make_prop(
            exec_id,
            "p_ext",
            "c_ext",
            ext_tool_id,
            {"target": "status"},
            external_source={"is_external": True, "source_id": "srv", "source_type": "mcp_sse"},
        )
        graph = TaskGraph(
            graph_id="tg_auth_rec",
            execution_id=exec_id,
            nodes=[
                TaskNode(
                    node_id="n_ext",
                    proposal_ids=["p_ext"],
                    execution_status=TaskNodeStatus.FAILED.value,
                    failure_info={"error": "pre-dispatch unavailable", "error_type": "FAILED_BEFORE_EXECUTION", "may_have_side_effect": False},
                    external_source={"is_external": True, "source_id": "srv", "source_type": "mcp_sse"},
                )
            ],
        )

        # Allowed when ring=1 (external required_ring=1) and external entry authorized
        a_ok = RecoveryEngine.assess_node_recovery(
            graph,
            "n_ext",
            central_ledger=central,
            proposals={"p_ext": prop},
            tool_registry=reg,
            caller_ring=1,
        )
        assert a_ok.recovery_decision == RecoveryDecision.RETRY_SAFE.value
        assert a_ok.revalidation_passed is True

        # Lower privilege ring=3 -> denied BEFORE redispatch!
        a_ring_denied = RecoveryEngine.assess_node_recovery(
            graph,
            "n_ext",
            central_ledger=central,
            proposals={"p_ext": prop},
            tool_registry=reg,
            caller_ring=3,
        )
        assert a_ring_denied.recovery_decision == RecoveryDecision.BLOCKED.value
        assert a_ring_denied.revalidation_passed is False

        # Revoke external authorization -> recovery BLOCKED prior to redispatch!
        reg.revoke_external_tool(ext_tool_id)
        a_denied = RecoveryEngine.assess_node_recovery(
            graph,
            "n_ext",
            central_ledger=central,
            proposals={"p_ext": prop},
            tool_registry=reg,
            caller_ring=1,
        )
        assert a_denied.recovery_decision == RecoveryDecision.BLOCKED.value
        assert a_denied.revalidation_passed is False
        central.close()


class TestGraphMutationParallelContinuityAndCLI:
    @pytest.mark.asyncio
    async def test_9_and_11_replan_corrective_alternative_preserve_verified_and_unlock_dependents(self, tmp_path: Path):
        exec_id = "exec_replan_rec"
        central = CentralExecutionLedger(db_path=tmp_path / "replan_rec.db")
        f_cfg = tmp_path / "cfg.json"
        f_alt = tmp_path / "logger_v2.py"
        f_fin = tmp_path / "main.py"

        f_cfg.write_text('{"ok": true}', encoding="utf-8")
        graph = TaskGraph(
            graph_id="tg_replan_rec",
            execution_id=exec_id,
            nodes=[
                TaskNode(
                    node_id="n_cfg",
                    proposal_ids=["p_cfg"],
                    execution_status=TaskNodeStatus.COMPLETED_VERIFIED.value,
                    verification_status=VerificationStatus.VERIFIED.value,
                    verification_requirement=f"file_exists:{f_cfg}",
                    order_index=0,
                ),
                TaskNode(
                    node_id="n_log",
                    proposal_ids=["p_log"],
                    dependency_ids=["n_cfg"],
                    execution_status=TaskNodeStatus.FAILED.value,
                    failure_info={"error": "deprecated logger template", "may_have_side_effect": True},
                    order_index=1,
                ),
                TaskNode(
                    node_id="n_main",
                    proposal_ids=["p_main"],
                    dependency_ids=["n_log"],
                    requires_verified_dependencies=True,
                    verification_requirement=f"file_exists:{f_fin}",
                    order_index=2,
                ),
            ],
        )
        props: Dict[str, Any] = {
            "p_cfg": _make_prop(exec_id, "p_cfg", "c_cfg", "manage_workspace_file", {"action": "write", "path": str(f_cfg), "content": "{}"}),
            "p_log": _make_prop(exec_id, "p_log", "c_log", "manage_workspace_file", {"action": "write", "path": str(tmp_path / "old.py"), "content": "x"}),
            "p_main": _make_prop(exec_id, "p_main", "c_main", "manage_workspace_file", {"action": "write", "path": str(f_fin), "content": "import logger"}, order=2, depends_on=["p_log"]),
        }
        props["p_cfg"].status = "completed"
        graph.recompute_schedule(known_proposals=props)
        assert graph.get_node("n_main").execution_status == TaskNodeStatus.BLOCKED.value

        central.create_execution("Build app with logger recovery", execution_id=exec_id, status=ExecutionStatus.RUNNING)
        scope = TaskLocalToolScope(tool_names=["manage_workspace_file"], tool_schemas=_build_schemas("manage_workspace_file"))
        ran: List[str] = []

        async def _exec(name: str, args: dict) -> dict:
            p = Path(args["path"])
            ran.append(p.name)
            p.write_text(args["content"], encoding="utf-8")
            return {"status": "success", "output": "ok"}

        orch = NativeOrchestrator(central_ledger=central, task_local_scope=scope, tool_executor=_bind_scope(_exec, scope))

        # Cannot silently drop completed verified prerequisite in replan_task_graph!
        with pytest.raises(PlanValidationError):
            orch.replan_graph(
                graph,
                {
                    "plan_version": 1,
                    "nodes": [
                        {"node_id": "n_cfg", "description": "Cfg", "depends_on": []},
                        {"node_id": "n_log", "description": "Logger without cfg", "depends_on": []},
                        {"node_id": "n_main", "description": "Main", "depends_on": ["n_log"]},
                    ],
                },
                known_proposals=props,
            )

        # Add corrective node first (preserves n_cfg dependency cleanly!), then replace with alternative
        f_corr = tmp_path / "dirs_ready.txt"
        orch.add_corrective_graph_node(
            graph,
            "n_log",
            corrective_node_id="n_prep_dir",
            description="Create clean output marker before retry",
            tool_id="manage_workspace_file",
            arguments={"action": "write", "path": str(f_corr), "content": "dir_ok"},
            verification_requirement=f"file_exists:{f_corr}",
            proposals=props,
        )
        assert "n_cfg" in graph.get_node("n_prep_dir").dependency_ids
        assert "n_cfg" in graph.get_node("n_log").dependency_ids
        assert "n_prep_dir" in graph.get_node("n_log").dependency_ids
        assert graph.get_node("n_cfg").execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value

        # Replace failed n_log with alternative n_log_v2; preserves original + corrective deps cleanly
        orch.replace_failed_graph_node(
            graph,
            "n_log",
            replacement_node_id="n_log_v2",
            description="Write logger_v2.py",
            tool_id="manage_workspace_file",
            arguments={"action": "write", "path": str(f_alt), "content": "LOG=2"},
            verification_requirement=f"file_exists:{f_alt}",
            proposals=props,
        )
        assert graph.get_node("n_log").execution_status == TaskNodeStatus.CANCELLED.value
        assert "n_cfg" in graph.get_node("n_log_v2").dependency_ids
        assert "n_prep_dir" in graph.get_node("n_log_v2").dependency_ids
        assert "n_log_v2" in graph.get_node("n_main").dependency_ids

        # Execute recovered subgraph -> n_cfg skipped (already verified!), n_prep_dir -> n_log_v2 -> n_main run!
        res = await orch.execute_task_graph(graph, props)
        assert res["graph_status"] == TaskGraphStatus.COMPLETED_VERIFIED.value
        assert ran == ["dirs_ready.txt", "logger_v2.py", "main.py"]
        assert f_corr.exists() and f_alt.exists() and f_fin.exists()
        central.close()

    @pytest.mark.asyncio
    async def test_12_parallel_workers_cannot_initiate_duplicate_recovery(self, tmp_path: Path):
        exec_id = "exec_parallel_rec_lock"
        central = CentralExecutionLedger(db_path=tmp_path / "par_lock.db")
        f_a = tmp_path / "a.txt"
        graph = TaskGraph(
            graph_id="tg_par_lock",
            execution_id=exec_id,
            max_concurrency=2,
            nodes=[
                TaskNode(
                    node_id="n_a",
                    proposal_ids=["p_a"],
                    execution_status=TaskNodeStatus.FAILED.value,
                    failure_info={"error": "pre-dispatch error", "error_type": "FAILED_BEFORE_EXECUTION", "may_have_side_effect": False},
                )
            ],
        )
        props = {"p_a": _make_prop(exec_id, "p_a", "c_a", "manage_workspace_file", {"action": "read", "path": str(f_a)})}

        # Worker 1 claims exclusive recovery ownership
        owner1 = graph.begin_node_recovery("n_a", "worker_1")
        assert owner1 == "worker_1"
        # Competing worker 2 attempting simultaneous recovery on n_a is rejected atomically!
        with pytest.raises(TaskGraphConflictError, match="active recovery owner"):
            RecoveryEngine.apply_recovery_decision(
                graph,
                RecoveryEngine.assess_node_recovery(graph, "n_a", central_ledger=central, proposals=props),
                prop_by_ref=props,
                central_ledger=central,
                owner_id="worker_2",
            )
        # Non-owner cannot release worker 1's recovery lock
        with pytest.raises(TaskGraphConflictError):
            graph.end_node_recovery("n_a", "worker_2")
        graph.end_node_recovery("n_a", "worker_1")

        # Also cannot initiate recovery while a node has an active worker reservation
        graph.prepare_node_for_retry("n_a", known_proposals=props)
        rid = graph.reserve_node("n_a", owner_id="active_w", max_concurrency=2)
        with pytest.raises(TaskGraphConflictError, match="reserved by active worker"):
            graph.begin_node_recovery("n_a", "recovery_intruder")
        graph.release_reservation("n_a", owner_id="active_w", reservation_id=rid, final_status=TaskNodeStatus.FAILED.value)
        central.close()

    @pytest.mark.asyncio
    async def test_13_cancellation_and_timeout_preserve_recovery_state_and_locks(self, tmp_path: Path):
        exec_id = "exec_cancel_rec"
        central = CentralExecutionLedger(db_path=tmp_path / "cancel_rec.db")
        f_slow = tmp_path / "slow.txt"
        started = asyncio.Event()

        async def _slow_tool(name: str, args: dict) -> dict:
            started.set()
            await asyncio.sleep(5.0)
            return {"status": "success", "output": "done"}

        graph = TaskGraph(
            graph_id="tg_cancel",
            execution_id=exec_id,
            max_concurrency=2,
            nodes=[
                TaskNode(
                    node_id="n_slow",
                    proposal_ids=["p_slow"],
                    recovery_state={
                        "recovery_version": 1,
                        "attempt_count": 1,
                        "retry_deadline": time.time() + 60.0,
                        "recovery_history": [],
                    },
                    order_index=0,
                ),
                TaskNode(
                    node_id="n_after",
                    proposal_ids=["p_after"],
                    dependency_ids=["n_slow"],
                    requires_verified_dependencies=True,
                    order_index=1,
                ),
            ],
        )
        props = [
            _make_prop(exec_id, "p_slow", "c_slow", "manage_workspace_file", {"action": "write", "path": str(f_slow), "content": "x"}),
            _make_prop(exec_id, "p_after", "c_after", "manage_workspace_file", {"action": "write", "path": str(tmp_path / "after.txt"), "content": "y"}, order=1, depends_on=["p_slow"]),
        ]
        scope = TaskLocalToolScope(tool_names=["manage_workspace_file"], tool_schemas=_build_schemas("manage_workspace_file"))
        orch = NativeOrchestrator(central_ledger=central, task_local_scope=scope, tool_executor=_bind_scope(_slow_tool, scope))

        task = asyncio.create_task(orch.execute_task_graph(graph, props))
        await asyncio.wait_for(started.wait(), timeout=2.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        n_slow = graph.get_node("n_slow")
        assert n_slow.execution_status == TaskNodeStatus.CANCELLED.value
        assert graph.active_reservations == {}
        assert n_slow.recovery_state is not None
        assert n_slow.recovery_state["attempt_count"] == 1
        assert n_slow.recovery_state["previous_failure_category"] == "CANCELLED"
        assert n_slow.recovery_state["may_have_side_effect"] is True
        assert n_slow.recovery_state["reconciliation_required"] is True
        # Cancelled prerequisite blocks downstream dependent cleanly
        assert graph.get_node("n_after").execution_status == TaskNodeStatus.BLOCKED.value
        central.close()

    @pytest.mark.asyncio
    async def test_14_and_15_restart_continuity_corruption_legacy_and_cli_graph_view(self, tmp_path: Path):
        db_path = tmp_path / "restart_rec.db"
        central = CentralExecutionLedger(db_path=db_path)
        exec_id = "exec_restart_cont"
        f_ok = tmp_path / "done.txt"
        f_fail = tmp_path / "pre_fail.txt"
        f_dep = tmp_path / "dep_after.txt"

        pre_calls = {"n": 0}
        ran: List[str] = []

        async def _exec(name: str, args: dict) -> dict:
            p = Path(args["path"])
            ran.append(p.name)
            p.write_text(args["content"], encoding="utf-8")
            return {"status": "success", "output": "ok"}

        def _pre_fail_first_two(name: str, args: dict) -> dict | None:
            if Path(args["path"]).name == "pre_fail.txt" and pre_calls["n"] < 2:
                pre_calls["n"] += 1
                return {"status": "error", "error": "transient pre-dispatch failure", "error_type": "TRANSIENT_FAILURE"}
            return None

        graph = TaskGraph(
            graph_id="tg_res_cont",
            execution_id=exec_id,
            max_concurrency=2,
            nodes=[
                TaskNode(node_id="n_ok", proposal_ids=["p_ok"], verification_requirement=f"file_exists:{f_ok}", order_index=0),
                TaskNode(node_id="n_pf", proposal_ids=["p_pf"], dependency_ids=["n_ok"], verification_requirement=f"file_exists:{f_fail}", order_index=1),
                TaskNode(node_id="n_dp", proposal_ids=["p_dp"], dependency_ids=["n_pf"], verification_requirement=f"file_exists:{f_dep}", order_index=2),
            ],
        )
        props = [
            _make_prop(exec_id, "p_ok", "c_ok", "manage_workspace_file", {"action": "write", "path": str(f_ok), "content": "ok"}),
            _make_prop(exec_id, "p_pf", "c_pf", "manage_workspace_file", {"action": "write", "path": str(f_fail), "content": "pf"}, order=1, depends_on=["p_ok"]),
            _make_prop(exec_id, "p_dp", "c_dp", "manage_workspace_file", {"action": "write", "path": str(f_dep), "content": "dp"}, order=2, depends_on=["p_pf"]),
        ]
        scope = TaskLocalToolScope(tool_names=["manage_workspace_file"], tool_schemas=_build_schemas("manage_workspace_file"))
        orch = NativeOrchestrator(central_ledger=central, task_local_scope=scope, tool_executor=_bind_scope(_exec, scope))
        orch.tool_executor.pre_dispatch = _pre_fail_first_two  # type: ignore[attr-defined]

        # Run with max_attempts=1 first -> n_ok succeeds, n_pf exhausts 1 retry -> FAILED, n_dp -> BLOCKED!
        res1 = await orch.execute_task_graph(
            graph,
            props,
            recovery_policy=RecoveryPolicy(max_attempts=1, backoff_base_seconds=0.0, deadline_seconds=60.0),
        )
        assert res1["graph_status"] == TaskGraphStatus.FAILED.value
        assert graph.get_node("n_ok").execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value
        assert graph.get_node("n_pf").recovery_state["attempt_count"] == 1
        assert graph.get_node("n_pf").recovery_state["recovery_status"] == "exhausted"
        assert graph.get_node("n_dp").execution_status == TaskNodeStatus.BLOCKED.value

        # CLI `zelyr execution <id> --graph` & `--json` surface recovery status, attempts, next_action, and blocked dependents!
        cli_txt = subprocess.run(
            [sys.executable, "-m", "zelyr.cli.main", "execution", exec_id, "--graph", "--db-path", str(db_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert cli_txt.returncode == 0, cli_txt.stderr
        assert "Recovery Status:" in cli_txt.stdout
        assert "attempts=1/1" in cli_txt.stdout
        assert "Blocked Dependents:" in cli_txt.stdout
        assert "n_dp (blocked_by=[n_pf])" in cli_txt.stdout

        cli_json = subprocess.run(
            [sys.executable, "-m", "zelyr.cli.main", "execution", exec_id, "--graph", "--json", "--db-path", str(db_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert cli_json.returncode == 0, cli_json.stderr
        parsed_cli = json.loads(cli_json.stdout)
        assert parsed_cli["recovery_overview"]["total_attempts"] == 1
        assert parsed_cli["recovery_overview"]["failed_nodes"][0]["node_id"] == "n_pf"
        assert parsed_cli["recovery_overview"]["blocked_dependents"][0]["node_id"] == "n_dp"

        # Restart process cleanly: attempt_count=1 is preserved and NOT reset to 0!
        central.close()
        reopened = CentralExecutionLedger(db_path=db_path)
        assess = CheckpointManager.assess_resume(reopened, exec_id, task_local_scope=scope, explicit_resume=True)
        assert assess.task_graph is not None
        assert assess.task_graph.get_node("n_ok").execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value
        assert assess.task_graph.get_node("n_pf").recovery_state["attempt_count"] == 1
        assert assess.task_graph.get_node("n_dp").execution_status == TaskNodeStatus.BLOCKED.value

        # With policy max_attempts=1 on resumed run, attempt_count=1 ALREADY meets budget so n_pf is NOT re-run!
        assess_after_resume = RecoveryEngine.assess_node_recovery(
            assess.task_graph,
            "n_pf",
            central_ledger=reopened,
            turn_ledger=assess.turn_ledger,
            proposals=assess.pending_actions,
            policy=RecoveryPolicy(max_attempts=1),
            task_local_scope=scope,
        )
        assert assess_after_resume.attempt_count == 1
        assert assess_after_resume.recovery_decision == RecoveryDecision.BLOCKED.value

        # Test ambiguous side-effect restart guard (cannot auto-retry ambiguous failure on restart!)
        exec_amb = "exec_restart_amb"
        reopened.create_execution("Ambiguous restart test", execution_id=exec_amb, status=ExecutionStatus.FAILED)
        tg_amb = TaskGraph(
            graph_id="tg_ramb",
            execution_id=exec_amb,
            nodes=[
                TaskNode(
                    node_id="na",
                    proposal_ids=["pa"],
                    execution_status=TaskNodeStatus.FAILED.value,
                    failure_info={"error": "timed out mid-write", "error_type": "TIMEOUT_ERROR", "may_have_side_effect": True},
                    recovery_state={
                        "recovery_version": 1,
                        "attempt_count": 0,
                        "recovery_decision": "require_confirmation",
                        "recovery_status": "reconciliation_required",
                        "may_have_side_effect": True,
                        "reconciliation_required": True,
                        "recovery_history": [],
                    },
                )
            ],
        )
        pa_amb = _make_prop(exec_amb, "pa", "ca", "manage_workspace_file", {"action": "write", "path": str(tmp_path / "z.txt"), "content": "z"})
        CheckpointManager.capture_checkpoint(
            reopened,
            exec_amb,
            status=ExecutionStatus.FAILED,
            pending_actions=[pa_amb],
            task_graph=tg_amb,
            emit_event=False,
        )
        assess_amb = CheckpointManager.assess_resume(reopened, exec_amb, task_local_scope=scope, explicit_resume=False)
        assert any("unreconciled side-effect state" in w for w in assess_amb.warnings)
        assert assess_amb.pending_actions[0].status == "blocked"

        # Corrupted / unsupported recovery state fails closed!
        bad_node = tg_amb.to_dict()
        bad_node["nodes"][0]["recovery_state"]["recovery_version"] = 99
        with pytest.raises((UnsupportedRecoveryVersionError, UnsupportedTaskGraphVersionError)):
            TaskGraph.from_dict(bad_node, verify_integrity=False)

        corrupt_rs = tg_amb.to_dict()
        corrupt_rs["nodes"][0]["recovery_state"]["attempt_count"] = -5
        with pytest.raises((CorruptedRecoveryStateError, CorruptedTaskGraphError)):
            TaskGraph.from_dict(corrupt_rs, verify_integrity=False)

        raw_cp = reopened.get_latest_checkpoint(exec_amb, verify_integrity=True)
        bad_cp = dict(raw_cp)
        bad_cp["recovery_summary"] = {"recovery_version": 99}
        with pytest.raises(UnsupportedCheckpointVersionError, match="recovery_version"):
            TaskCheckpoint.from_dict(bad_cp, verify_integrity=False)

        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "UPDATE task_checkpoints SET recovery_summary_json = ? WHERE execution_id = ?",
                (json.dumps({"recovery_version": 99}), exec_amb),
            )
        assess_bad_ver = CheckpointManager.assess_resume(reopened, exec_amb)
        assert assess_bad_ver.can_resume is False
        assert assess_bad_ver.resumability_status == ResumabilityStatus.BLOCKED.value
        assert any("digest" in r.lower() or "recovery_version" in r for r in assess_bad_ver.blocking_reasons)

        # Legacy graph without recovery_state computes exact digest and loads cleanly
        legacy_node = TaskNode(node_id="leg_1", description="Legacy step", proposal_ids=["p_leg"])
        legacy_graph = TaskGraph(graph_id="tg_legacy", execution_id="exec_leg", nodes=[legacy_node])
        d_legacy = legacy_graph.to_dict()
        assert "recovery_state" not in d_legacy["nodes"][0]
        assert d_legacy["graph_digest"] == compute_task_graph_digest(d_legacy)
        restored_leg = TaskGraph.from_dict(d_legacy, verify_integrity=True)
        assert restored_leg.get_node("leg_1").recovery_state is None
        reopened.close()
