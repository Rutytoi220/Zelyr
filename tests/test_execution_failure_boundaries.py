"""Cross-subsystem failure-boundary and regression tests across orchestration, graph, policy, verification, and ledger."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.config import ZelyrConfig
from zelyr.engine import (
    ActionState,
    CentralExecutionLedger,
    CheckpointManager,
    ConfirmationStore,
    ExecutionPolicyConfig,
    ExecutionPolicyEngine,
    ExecutionStatus,
    LedgerPersistenceError,
    PendingActionProposal,
    PolicyDecisionOutcome,
    RecoveryDecision,
    RecoveryEngine,
    RecoveryPolicy,
    ResumabilityStatus,
    TaskGraph,
    TaskGraphStatus,
    TaskNode,
    TaskNodeStatus,
    TurnExecutionLedger,
    VerificationStatus,
    collect_execution_metrics,
    diagnose_execution_lifecycle,
)
from zelyr.engine.execution_policy import (
    compute_canonical_argument_digest,
    extract_resource_scope,
)
from zelyr.tool_registry import TaskLocalToolScope, bind_task_local_executor
from zelyr.tools.core import ToolResult


def _build_schemas(*names: str, read_only_names: set[str] | None = None) -> List[Dict[str, Any]]:
    ro_set = read_only_names or set()
    schemas: List[Dict[str, Any]] = []
    for name in names:
        fn_def: Dict[str, Any] = {
            "name": name,
            "description": f"Boundary test schema for {name}",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "additionalProperties": True,
            },
        }
        if name in ro_set:
            fn_def["read_only"] = True
        schemas.append({"type": "function", "function": fn_def})
    return schemas


def _make_prop(
    exec_id: str,
    pid: str,
    cid: str,
    tool_id: str,
    args: Dict[str, Any],
    *,
    order: int = 0,
    depends_on: List[str] | None = None,
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
        status="pending",
    )


def _bind(fn: Any, scope: TaskLocalToolScope) -> Any:
    return bind_task_local_executor(
        fn,
        tool_names=scope.tool_names,
        tool_schemas=scope.tool_schemas,
        tool_rings=scope.tool_rings,
    )


class TestCancellationAndTimeoutBoundaries:
    """Verify deterministic cleanup, reservation release, and non-replay on cancel/timeout."""

    @pytest.mark.asyncio
    async def test_react_cancel_mid_mutating_tool_blocks_blind_replay_on_resume(self, tmp_path: Path) -> None:
        central = CentralExecutionLedger(db_path=tmp_path / "cancel_react.db")
        target_file = tmp_path / "mutated.txt"
        started_evt = asyncio.Event()
        never_finish = asyncio.Event()
        exec_count = 0

        async def _exec(tool_name: str, args: Dict[str, Any]) -> ToolResult:
            nonlocal exec_count
            exec_count += 1
            started_evt.set()
            await never_finish.wait()
            return ToolResult(success=True, output="unreachable")

        async def _stream_cancel(payload: Dict[str, Any]):
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_mut_inflight",
                                    "type": "function",
                                    "function": {
                                        "name": "manage_workspace_file",
                                        "arguments": json.dumps(
                                            {"action": "write", "path": str(target_file), "content": "v1"}
                                        ),
                                    },
                                }
                            ]
                        }
                    }
                ]
            }

        scope = TaskLocalToolScope(
            tool_names=["manage_workspace_file"],
            tool_schemas=_build_schemas("manage_workspace_file"),
            tool_rings={"manage_workspace_file": 2},
        )
        orch = NativeOrchestrator(
            central_ledger=central,
            task_local_scope=scope,
            tool_executor=_bind(_exec, scope),
        )
        orch.execution_policy_engine = ExecutionPolicyEngine(
            config=ExecutionPolicyConfig(policy_mode="autopilot")
        )
        orch.workspace_root = tmp_path

        turn_ledger = TurnExecutionLedger()

        async def _drive() -> None:
            async for _ in orch.generate_stream(
                {"messages": [{"role": "user", "content": "Write mutated.txt"}]},
                stream_provider=_stream_cancel,
                ledger=turn_ledger,
            ):
                pass

        task = asyncio.create_task(_drive())
        await started_evt.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert exec_count == 1
        exec_id = turn_ledger._execution_id
        assert exec_id is not None
        rec = central.get_execution(exec_id)
        assert rec is not None
        assert rec.status == ExecutionStatus.CANCELLED.value
        assert rec.verification_status == VerificationStatus.INTERRUPTED.value

        assessment = CheckpointManager.assess_resume(
            central,
            exec_id,
            task_local_scope=scope,
            active_schemas=scope.tool_schemas,
            explicit_resume=False,
        )
        assert assessment.resumability_status == ResumabilityStatus.NEEDS_CONFIRMATION.value
        assert any(
            op.call_id == "call_mut_inflight" and op.may_have_side_effect
            for op in (assessment.in_flight_ambiguous_ops + assessment.failed_ops)
        )

        # Even if explicitly resumed without allow_unverified_retry, ambiguous call cannot silently replay.
        replay_attempt_calls = 0

        async def _exec_no_replay(tool_name: str, args: Dict[str, Any]) -> ToolResult:
            nonlocal replay_attempt_calls
            replay_attempt_calls += 1
            return ToolResult(success=True, output="replayed!")

        replay_turns = 0

        async def _stream_replay(payload: Dict[str, Any]):
            nonlocal replay_turns
            replay_turns += 1
            if replay_turns == 1:
                yield {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_mut_inflight",
                                        "type": "function",
                                        "function": {
                                            "name": "manage_workspace_file",
                                            "arguments": json.dumps(
                                                {"action": "write", "path": str(target_file), "content": "v1"}
                                            ),
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            else:
                yield {"choices": [{"delta": {"content": "Blocked from blindly replaying."}}]}

        orch.tool_executor = _bind(_exec_no_replay, scope)
        _, resumed_chunks = await CheckpointManager.resume_execution(
            orch,
            central,
            exec_id,
            explicit_resume=True,
            allow_unverified_retry=False,
            task_local_scope=scope,
            tool_schemas=scope.tool_schemas,
            stream_provider=_stream_replay,
        )
        assert replay_attempt_calls == 0
        assert any("Blocked" in str(c) for c in resumed_chunks)
        central.close()

    @pytest.mark.asyncio
    async def test_parallel_graph_cancel_releases_reservations_preserves_verified_and_marks_ambiguity(
        self, tmp_path: Path
    ) -> None:
        exec_id = "exec_parallel_cancel"
        central = CentralExecutionLedger(db_path=tmp_path / "pg_cancel.db")
        f_a = tmp_path / "a.txt"
        f_b = tmp_path / "b.txt"
        f_c = tmp_path / "c.txt"

        graph = TaskGraph(
            graph_id="tg_cancel",
            execution_id=exec_id,
            max_concurrency=2,
            nodes=[
                TaskNode(node_id="n_a", proposal_ids=["p_a"], verification_requirement=f"file_exists:{f_a}", order_index=0),
                TaskNode(node_id="n_b", proposal_ids=["p_b"], verification_requirement=f"file_exists:{f_b}", order_index=1),
                TaskNode(
                    node_id="n_c",
                    proposal_ids=["p_c"],
                    dependency_ids=["n_b"],
                    requires_verified_dependencies=True,
                    order_index=2,
                ),
            ],
        )
        props = [
            _make_prop(exec_id, "p_a", "c_a", "manage_workspace_file", {"action": "write", "path": str(f_a), "content": "A"}, order=0),
            _make_prop(exec_id, "p_b", "c_b", "manage_workspace_file", {"action": "write", "path": str(f_b), "content": "B"}, order=1),
            _make_prop(exec_id, "p_c", "c_c", "manage_workspace_file", {"action": "write", "path": str(f_c), "content": "C"}, order=2, depends_on=["p_b"]),
        ]

        a_done = asyncio.Event()
        b_entered = asyncio.Event()
        hold_b = asyncio.Event()

        async def _exec(tool_name: str, args: Dict[str, Any]) -> ToolResult:
            p = Path(args["path"])
            if p == f_a:
                p.write_text("A", encoding="utf-8")
                a_done.set()
                return ToolResult(success=True, output="wrote A")
            if p == f_b:
                b_entered.set()
                await hold_b.wait()
                return ToolResult(success=True, output="unreachable")
            raise AssertionError(f"Dependent node C must never run: {p}")

        scope = TaskLocalToolScope(
            tool_names=["manage_workspace_file"],
            tool_schemas=_build_schemas("manage_workspace_file"),
            tool_rings={"manage_workspace_file": 2},
        )
        orch = NativeOrchestrator(central_ledger=central, task_local_scope=scope, tool_executor=_bind(_exec, scope))
        orch.execution_policy_engine = ExecutionPolicyEngine(config=ExecutionPolicyConfig(policy_mode="autopilot"))
        orch.workspace_root = tmp_path

        task = asyncio.create_task(orch.execute_task_graph(graph, props))
        await asyncio.gather(a_done.wait(), b_entered.wait())
        # Give event loop a tick so worker A finishes applying observation before cancel arrives.
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert graph.get_node("n_b").reserved_by is None
        assert graph.get_node("n_b").reservation_id is None
        assert graph.get_node("n_b").execution_status == TaskNodeStatus.CANCELLED.value
        assert graph.get_node("n_b").recovery_state["may_have_side_effect"] is True
        assert graph.get_node("n_b").recovery_state["reconciliation_required"] is True
        assert not f_c.exists()

        rec = central.get_execution(exec_id)
        assert rec is not None
        assert rec.status == ExecutionStatus.CANCELLED.value
        rebuilt, errs = CheckpointManager.reconstruct_task_graph(central, exec_id)
        assert errs == []
        assert rebuilt is not None
        assert rebuilt.get_node("n_b").reserved_by is None
        central.close()

    @pytest.mark.asyncio
    async def test_node_timeout_on_mutating_op_blocks_blind_auto_retry(self, tmp_path: Path) -> None:
        exec_id = "exec_mut_timeout"
        central = CentralExecutionLedger(db_path=tmp_path / "timeout_mut.db")
        f_slow = tmp_path / "slow.txt"
        f_after = tmp_path / "after.txt"
        exec_calls = 0

        graph = TaskGraph(
            graph_id="tg_timeout",
            execution_id=exec_id,
            max_concurrency=1,
            nodes=[
                TaskNode(node_id="n_slow", proposal_ids=["p_slow"], verification_requirement=f"file_exists:{f_slow}", order_index=0),
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
            _make_prop(exec_id, "p_slow", "c_slow", "manage_workspace_file", {"action": "write", "path": str(f_slow), "content": "slow"}, order=0),
            _make_prop(exec_id, "p_after", "c_after", "manage_workspace_file", {"action": "write", "path": str(f_after), "content": "after"}, order=1, depends_on=["p_slow"]),
        ]

        never = asyncio.Event()

        async def _slow_exec(tool_name: str, args: Dict[str, Any]) -> ToolResult:
            nonlocal exec_calls
            exec_calls += 1
            await never.wait()
            return ToolResult(success=True, output="late")

        scope = TaskLocalToolScope(
            tool_names=["manage_workspace_file"],
            tool_schemas=_build_schemas("manage_workspace_file"),
            tool_rings={"manage_workspace_file": 2},
        )
        orch = NativeOrchestrator(central_ledger=central, task_local_scope=scope, tool_executor=_bind(_slow_exec, scope))
        orch.execution_policy_engine = ExecutionPolicyEngine(config=ExecutionPolicyConfig(policy_mode="autopilot"))
        orch.workspace_root = tmp_path

        res = await orch.execute_task_graph(
            graph,
            props,
            default_node_timeout=0.05,
            recovery_policy=RecoveryPolicy.from_config(
                ZelyrConfig(graph_recovery_max_attempts=3, graph_recovery_backoff_base_seconds=0.0)
            ),
        )
        assert exec_calls == 1
        assert not f_after.exists()
        assert graph.get_node("n_after").execution_status == TaskNodeStatus.BLOCKED.value
        assert res["graph_status"] in (TaskGraphStatus.FAILED.value, TaskGraphStatus.BLOCKED.value)
        assess = RecoveryEngine.assess_node_recovery(
            graph,
            "n_slow",
            central_ledger=central,
            proposals={p.proposal_id: p for p in props},
        )
        assert assess.recovery_decision not in (
            RecoveryDecision.RETRY_SAFE.value,
            RecoveryDecision.RETRY_AFTER_REVALIDATION.value,
        )
        central.close()


class TestConfirmationAndConflictBoundaries:
    """Verify confirmation invalidation after argument/graph changes and parallel conflict serialization."""

    def test_stale_confirmation_invalidated_on_arg_mutation_replan_and_double_consume(self, tmp_path: Path) -> None:
        store = ConfirmationStore(db_path=tmp_path / "conf.db")
        engine = ExecutionPolicyEngine(
            config=ExecutionPolicyConfig(policy_mode="conservative"),
            confirmation_store=store,
        )
        args_v1 = {"action": "write", "path": str(tmp_path / "a.txt"), "content": "v1"}
        args_v2 = {"action": "write", "path": str(tmp_path / "a.txt"), "content": "v2"}
        scope_v1 = extract_resource_scope("manage_workspace_file", args_v1, workspace_root=tmp_path)
        dig_v1 = compute_canonical_argument_digest("manage_workspace_file", args_v1, scope_v1)

        tok, rec = store.issue_confirmation(
            tool_id="manage_workspace_file",
            arg_digest=dig_v1,
            resource_scope=scope_v1,
            execution_id="exec_conf_mut",
            node_id="n1",
        )

        # 1. Argument mutation is denied without consuming the v1 confirmation
        mut_dec = engine.evaluate(
            "manage_workspace_file",
            args_v2,
            normalized_arguments=args_v2,
            required_ring=2,
            is_registered=True,
            workspace_root=tmp_path,
            execution_id="exec_conf_mut",
            node_id="n1",
            confirmation_token=tok,
            confirmation_id=rec.confirmation_id,
            consume_confirmation=True,
        )
        assert mut_dec.outcome == PolicyDecisionOutcome.DENY
        assert mut_dec.reason_code == "confirmation_mismatch"

        # 2. Graph node replacement / replan binds a different node_id -> denied cleanly
        node_dec = engine.evaluate(
            "manage_workspace_file",
            args_v1,
            normalized_arguments=args_v1,
            required_ring=2,
            is_registered=True,
            workspace_root=tmp_path,
            execution_id="exec_conf_mut",
            node_id="n1_alt",
            confirmation_token=tok,
            confirmation_id=rec.confirmation_id,
            consume_confirmation=True,
        )
        assert node_dec.outcome == PolicyDecisionOutcome.DENY
        assert node_dec.reason_code == "confirmation_mismatch"

        # 3. Exact matching node + args succeeds once, then replay is rejected as already consumed
        ok_dec = engine.evaluate(
            "manage_workspace_file",
            args_v1,
            normalized_arguments=args_v1,
            required_ring=2,
            is_registered=True,
            workspace_root=tmp_path,
            execution_id="exec_conf_mut",
            node_id="n1",
            confirmation_token=tok,
            confirmation_id=rec.confirmation_id,
            consume_confirmation=True,
        )
        assert ok_dec.outcome == PolicyDecisionOutcome.ALLOW

        replay_dec = engine.evaluate(
            "manage_workspace_file",
            args_v1,
            normalized_arguments=args_v1,
            required_ring=2,
            is_registered=True,
            workspace_root=tmp_path,
            execution_id="exec_conf_mut",
            node_id="n1",
            confirmation_token=tok,
            confirmation_id=rec.confirmation_id,
            consume_confirmation=True,
        )
        assert replay_dec.outcome == PolicyDecisionOutcome.DENY
        assert replay_dec.reason_code == "confirmation_replayed"

    @pytest.mark.asyncio
    async def test_duplicate_tool_call_ids_deduped_and_shared_resource_serialized(self, tmp_path: Path) -> None:
        # Part A: ReAct duplicate tool_call_id across single response and consecutive turns
        central = CentralExecutionLedger(db_path=tmp_path / "dup_cid.db")
        f1 = tmp_path / "d1.txt"
        f2 = tmp_path / "d2.txt"
        ro_execs: List[str] = []

        async def _exec_react(tool_name: str, args: Dict[str, Any]) -> ToolResult:
            p = Path(args["path"])
            ro_execs.append(p.name)
            p.write_text(str(args.get("content") or ""), encoding="utf-8")
            return ToolResult(success=True, output=p.name)

        turn_num = 0

        async def _dup_stream(payload: Dict[str, Any]):
            nonlocal turn_num
            turn_num += 1
            if turn_num == 1:
                yield {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "dup_id",
                                        "type": "function",
                                        "function": {
                                            "name": "manage_workspace_file",
                                            "arguments": json.dumps({"action": "write", "path": str(f1), "content": "1"}),
                                        },
                                    },
                                    {
                                        "index": 1,
                                        "id": "dup_id",
                                        "type": "function",
                                        "function": {
                                            "name": "manage_workspace_file",
                                            "arguments": json.dumps({"action": "write", "path": str(f2), "content": "2"}),
                                        },
                                    },
                                ]
                            }
                        }
                    ]
                }
            elif turn_num == 2:
                yield {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "dup_id",
                                        "type": "function",
                                        "function": {
                                            "name": "manage_workspace_file",
                                            "arguments": json.dumps({"action": "write", "path": str(f2), "content": "2"}),
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            else:
                yield {"choices": [{"delta": {"content": "Both writes completed cleanly."}}]}

        scope = TaskLocalToolScope(
            tool_names=["manage_workspace_file"],
            tool_schemas=_build_schemas("manage_workspace_file"),
            tool_rings={"manage_workspace_file": 2},
        )
        orch = NativeOrchestrator(central_ledger=central, task_local_scope=scope, tool_executor=_bind(_exec_react, scope))
        orch.execution_policy_engine = ExecutionPolicyEngine(config=ExecutionPolicyConfig(policy_mode="autopilot"))
        orch.workspace_root = tmp_path

        tl = TurnExecutionLedger()
        async for _ in orch.generate_stream(
            {"messages": [{"role": "user", "content": "Write d1 and d2"}]},
            stream_provider=_dup_stream,
            ledger=tl,
        ):
            pass

        assert ro_execs == ["d1.txt", "d2.txt"]
        assert "dup_id" in tl._records
        assert "dup_id_2" in tl._records
        assert "dup_id_3" in tl._records

        # Part B: Parallel TaskGraph serializes overlapping writes on shared.txt while overlapping disjoint other.txt
        shared = tmp_path / "shared.txt"
        other = tmp_path / "other.txt"
        graph = TaskGraph(
            graph_id="tg_conflict",
            execution_id="exec_conflict_serial",
            max_concurrency=3,
            nodes=[
                TaskNode(node_id="s1", proposal_ids=["ps1"], verification_requirement=f"file_exists:{shared}", order_index=0),
                TaskNode(node_id="s2", proposal_ids=["ps2"], verification_requirement=f"file_exists:{shared}", order_index=1),
                TaskNode(node_id="o1", proposal_ids=["po1"], verification_requirement=f"file_exists:{other}", order_index=2),
            ],
        )
        g_props = [
            _make_prop("exec_conflict_serial", "ps1", "cs1", "manage_workspace_file", {"action": "write", "path": str(shared), "content": "s1"}, order=0),
            _make_prop("exec_conflict_serial", "ps2", "cs2", "manage_workspace_file", {"action": "write", "path": str(shared), "content": "s2"}, order=1),
            _make_prop("exec_conflict_serial", "po1", "co1", "manage_workspace_file", {"action": "write", "path": str(other), "content": "o1"}, order=2),
        ]
        active_shared = 0
        max_shared = 0

        async def _exec_conflict(tool_name: str, args: Dict[str, Any]) -> ToolResult:
            nonlocal active_shared, max_shared
            p = Path(args["path"])
            if p == shared:
                active_shared += 1
                max_shared = max(max_shared, active_shared)
                await asyncio.sleep(0.01)
                p.write_text(str(args.get("content") or ""), encoding="utf-8")
                active_shared -= 1
            else:
                await asyncio.sleep(0.01)
                p.write_text(str(args.get("content") or ""), encoding="utf-8")
            return ToolResult(success=True, output="ok")

        orch.tool_executor = _bind(_exec_conflict, scope)
        res = await orch.execute_task_graph(graph, g_props)
        assert max_shared == 1
        assert res["graph_status"] == TaskGraphStatus.COMPLETED_VERIFIED.value
        central.close()


class TestProviderMalformedVerifierAndPersistenceBoundaries:
    """Verify mid-turn provider failure, malformed/spoofed args, contradictory verifiers, and DB errors."""

    @pytest.mark.asyncio
    async def test_mid_turn_provider_error_and_malformed_spoofed_args_boundaries(self, tmp_path: Path) -> None:
        central = CentralExecutionLedger(db_path=tmp_path / "midturn_err.db")
        readable = tmp_path / "info.txt"
        readable.write_text("alpha", encoding="utf-8")

        async def _exec(tool_name: str, args: Dict[str, Any]) -> ToolResult:
            if tool_name == "read_file":
                return ToolResult(success=True, output=Path(args["path"]).read_text(encoding="utf-8"))
            raise AssertionError(f"Unexpected tool execution: {tool_name} {args}")

        scope = TaskLocalToolScope(
            tool_names=["read_file", "manage_workspace_file"],
            tool_schemas=_build_schemas("read_file", "manage_workspace_file", read_only_names={"read_file"}),
            tool_rings={"read_file": 2, "manage_workspace_file": 2},
        )
        orch = NativeOrchestrator(central_ledger=central, task_local_scope=scope, tool_executor=_bind(_exec, scope))
        orch.execution_policy_engine = ExecutionPolicyEngine(config=ExecutionPolicyConfig(policy_mode="conservative"))
        orch.workspace_root = tmp_path

        # Scenario A: Turn 1 read_file succeeds, Turn 2 raises ConnectionError
        t = 0

        async def _midturn_fail_stream(payload: Dict[str, Any]):
            nonlocal t
            t += 1
            if t == 1:
                yield {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_ro_1",
                                        "type": "function",
                                        "function": {"name": "read_file", "arguments": json.dumps({"path": str(readable)})},
                                    }
                                ]
                            }
                        }
                    ]
                }
            else:
                raise ConnectionError("upstream TLS reset mid-turn")

        tl1 = TurnExecutionLedger()
        with pytest.raises(ConnectionError):
            async for _ in orch.generate_stream(
                {"messages": [{"role": "user", "content": "Read info.txt"}]},
                stream_provider=_midturn_fail_stream,
                ledger=tl1,
            ):
                pass

        eid1 = tl1._execution_id
        assert eid1 is not None
        diag = diagnose_execution_lifecycle(central, eid1)
        assert diag.request["status"] == ExecutionStatus.INTERRUPTED.value
        assert "read_file" in diag.tools["unique_tools"]
        assert any(c["tool_id"] == "read_file" for c in diag.tools["calls"])
        metrics = collect_execution_metrics(central, execution_id=eid1)
        assert metrics["model_requests"]["measurement_kind"] == "measured"
        assert metrics["model_requests"]["count"] >= 2

        # Scenario B: Turn 1 malformed JSON, Turn 2 spoofed policy override key -> rejected without writing
        secret_write = tmp_path / "spoof.txt"
        b_turn = 0

        async def _malformed_then_spoof_stream(payload: Dict[str, Any]):
            nonlocal b_turn
            b_turn += 1
            if b_turn == 1:
                yield {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "bad_json_call",
                                        "type": "function",
                                        "function": {"name": "read_file", "arguments": "{not valid json"},
                                    }
                                ]
                            }
                        }
                    ]
                }
            elif b_turn == 2:
                yield {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "spoof_call",
                                        "type": "function",
                                        "function": {
                                            "name": "manage_workspace_file",
                                            "arguments": json.dumps(
                                                {
                                                    "action": "write",
                                                    "path": str(secret_write),
                                                    "content": "pwn",
                                                    "caller_ring": 0,
                                                }
                                            ),
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            else:
                yield {"choices": [{"delta": {"content": "Unable to bypass authorization."}}]}

        tl2 = TurnExecutionLedger()
        async for _ in orch.generate_stream(
            {"messages": [{"role": "user", "content": "Test malformed and spoof"}]},
            stream_provider=_malformed_then_spoof_stream,
            ledger=tl2,
        ):
            pass

        assert not secret_write.exists()
        for cid_chk in ("bad_json_call", "spoof_call"):
            rec_chk = tl2.get_record(cid_chk)
            assert rec_chk is not None
            assert rec_chk.state in (ActionState.REJECTED, ActionState.FAILED)
            obs_chk = tl2.get_structured_observation(cid_chk)
            assert obs_chk is not None
            assert obs_chk.execution_status == "failed_before_execution"
            assert obs_chk.operation_attempted.get("dispatched") is False
            assert obs_chk.recovery_guidance.get("may_have_side_effect") is False
        central.close()

    @pytest.mark.asyncio
    async def test_contradictory_verifier_blocks_dependents_and_ledger_failure_boundaries(self, tmp_path: Path) -> None:
        exec_id = "exec_contradict"
        central = CentralExecutionLedger(db_path=tmp_path / "contra.db")
        f1 = tmp_path / "claimed.txt"
        f2 = tmp_path / "downstream.txt"

        graph = TaskGraph(
            graph_id="tg_contra",
            execution_id=exec_id,
            max_concurrency=1,
            nodes=[
                TaskNode(node_id="n1", proposal_ids=["p1"], verification_requirement=f"file_exists:{f1}", order_index=0),
                TaskNode(
                    node_id="n2",
                    proposal_ids=["p2"],
                    dependency_ids=["n1"],
                    requires_verified_dependencies=True,
                    order_index=1,
                ),
            ],
        )
        props = [
            _make_prop(exec_id, "p1", "c1", "manage_workspace_file", {"action": "write", "path": str(f1), "content": "claim"}, order=0),
            _make_prop(exec_id, "p2", "c2", "manage_workspace_file", {"action": "write", "path": str(f2), "content": "dep"}, order=1, depends_on=["p1"]),
        ]

        async def _lying_exec(tool_name: str, args: Dict[str, Any]) -> ToolResult:
            # Pretends success, intentionally never writes f1
            return ToolResult(success=True, output="Successfully created claimed.txt")

        scope = TaskLocalToolScope(
            tool_names=["manage_workspace_file"],
            tool_schemas=_build_schemas("manage_workspace_file"),
            tool_rings={"manage_workspace_file": 2},
        )
        orch = NativeOrchestrator(central_ledger=central, task_local_scope=scope, tool_executor=_bind(_lying_exec, scope))
        orch.execution_policy_engine = ExecutionPolicyEngine(config=ExecutionPolicyConfig(policy_mode="autopilot"))
        orch.workspace_root = tmp_path

        res = await orch.execute_task_graph(
            graph,
            props,
            verifier_fn=lambda nid, node, obs: False if nid == "n1" else True,
            enable_recovery=False,
        )
        assert not f2.exists()
        assert graph.get_node("n1").execution_status != TaskNodeStatus.COMPLETED_VERIFIED.value
        assert graph.get_node("n2").execution_status == TaskNodeStatus.BLOCKED.value
        assert res["graph_status"] != TaskGraphStatus.COMPLETED_VERIFIED.value

        # Policy engine remains fail-closed even if central_ledger persistence fails
        class _BrokenLedger:
            def record_event(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("sqlite database is locked")

        pol_engine = ExecutionPolicyEngine(config=ExecutionPolicyConfig(policy_mode="conservative"))
        pol_dec = pol_engine.evaluate(
            "manage_workspace_file",
            {"action": "write", "path": str(tmp_path / "x.txt"), "content": "hi"},
            required_ring=2,
            is_registered=True,
            workspace_root=tmp_path,
            execution_id="exec_broken_ledger",
            central_ledger=_BrokenLedger(),  # type: ignore[arg-type]
        )
        assert pol_dec.outcome == PolicyDecisionOutcome.REQUIRE_CONFIRMATION

        # Strict central_ledger persistence failure during checkpoint still attempts finalize_execution before raising
        strict_central = CentralExecutionLedger(
            db_path=tmp_path / "strict_cp_fail.db",
            raise_on_persistence_error=True,
        )
        finalize_calls: List[tuple[str, Any]] = []
        orig_save = strict_central.save_checkpoint
        orig_finalize = strict_central.finalize_execution

        def _failing_cp(*args: Any, **kwargs: Any) -> Any:
            if len(finalize_calls) == 0:
                raise LedgerPersistenceError("simulated disk full during checkpoint")
            return orig_save(*args, **kwargs)

        def _tracking_finalize(eid: str, **kwargs: Any) -> Any:
            finalize_calls.append((eid, kwargs.get("status")))
            return orig_finalize(eid, **kwargs)

        strict_central.save_checkpoint = _failing_cp  # type: ignore[method-assign]
        strict_central.finalize_execution = _tracking_finalize  # type: ignore[method-assign]

        f_strict = tmp_path / "strict.txt"
        strict_graph = TaskGraph(
            graph_id="tg_strict",
            execution_id="exec_strict_fail",
            nodes=[TaskNode(node_id="ns", proposal_ids=["ps"], order_index=0)],
        )
        strict_props = [
            _make_prop("exec_strict_fail", "ps", "cs", "manage_workspace_file", {"action": "write", "path": str(f_strict), "content": "ok"})
        ]

        async def _ok_exec(tool_name: str, args: Dict[str, Any]) -> ToolResult:
            Path(args["path"]).write_text("ok", encoding="utf-8")
            return ToolResult(success=True, output="ok")

        strict_orch = NativeOrchestrator(
            central_ledger=strict_central,
            task_local_scope=scope,
            tool_executor=_bind(_ok_exec, scope),
        )
        strict_orch.execution_policy_engine = ExecutionPolicyEngine(config=ExecutionPolicyConfig(policy_mode="autopilot"))
        strict_orch.workspace_root = tmp_path

        with pytest.raises(LedgerPersistenceError, match="simulated disk full"):
            await strict_orch.execute_task_graph(strict_graph, strict_props)

        assert len(finalize_calls) == 1
        assert finalize_calls[0][0] == "exec_strict_fail"
        central.close()
        strict_central.close()
