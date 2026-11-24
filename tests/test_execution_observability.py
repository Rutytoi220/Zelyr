"""Deterministic offline tests for Execution and Policy Observability."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, AsyncGenerator, Dict, List
from unittest.mock import patch

import pytest

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.config import ZelyrConfig
from zelyr.engine import (
    CentralExecutionLedger,
    ExecutionPolicyConfig,
    ExecutionPolicyEngine,
    ExecutionStatus,
    LedgerEventType,
    PendingActionProposal,
    PolicyDecisionOutcome,
    ProposalStatus,
    TaskGraph,
    TaskNode,
    collect_execution_metrics,
    collect_fleet_observability,
    format_observability_summary,
)
from zelyr.tool_registry import TaskLocalToolScope


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _run_cli(args: List[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_repo_root())
    env["ZELYR_TESTING"] = "1"
    env["QT_QPA_PLATFORM"] = "offscreen"
    return subprocess.run(
        [sys.executable, "-m", "zelyr.cli.main", *args],
        cwd=str(_repo_root()),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_orchestrator_react_measures_model_tool_verification_and_steps(tmp_path: Path) -> None:
    """NativeOrchestrator ReAct turn measures model requests, steps, tool & verifier durations without inventing tokens."""
    db_path = tmp_path / "obs_react.db"
    target_file = tmp_path / "artifact.txt"
    ledger = CentralExecutionLedger(db_path=db_path, raise_on_persistence_error=True)
    scope = TaskLocalToolScope(
        tool_names=["local_write"],
        tool_schemas=[
            {
                "type": "function",
                "function": {
                    "name": "local_write",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                        "required": ["path", "content"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        tool_rings={"local_write": 1},
    )

    turn_num = {"n": 0}

    async def fake_stream(payload: dict) -> AsyncGenerator[dict, None]:
        turn_num["n"] += 1
        if turn_num["n"] == 1:
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_obs_1",
                                    "function": {
                                        "name": "local_write",
                                        "arguments": json.dumps({"path": str(target_file), "content": "ok_secret_body"}),
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        else:
            yield {"choices": [{"delta": {"content": "Created artifact.txt cleanly."}, "finish_reason": "stop"}]}

    def fake_exec(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        Path(args["path"]).write_text(args["content"], encoding="utf-8")
        return {"status": "success", "output": "written", "exit_code": 0}

    fake_exec._task_local_tool_scope = scope  # type: ignore[attr-defined]

    async def _run() -> str:
        orch = NativeOrchestrator(
            stream_provider=fake_stream,
            tool_executor=fake_exec,
            tool_schemas=list(scope.tool_schemas),
            task_local_scope=scope,
            caller_ring=1,
            central_ledger=ledger,
            config=ZelyrConfig(execution_policy_mode="autopilot"),
        )
        chunks = []
        async for ch in orch.generate_stream(
            {
                "model": "offline-test",
                "messages": [{"role": "user", "content": f"Write {target_file}"}],
                "acceptance_criteria": f"file_exists:{target_file}",
            }
        ):
            chunks.append(ch)
        return orch.current_ledger._execution_id

    try:
        exec_id = asyncio.run(_run())
        assert exec_id
        metrics = collect_execution_metrics(ledger, exec_id)
        assert metrics["found"] is True
        assert metrics["model_requests"]["measurement_kind"] == "measured"
        assert metrics["model_requests"]["count"] == 2
        assert metrics["model_requests"]["total_duration_ms"] >= 0.0
        assert metrics["model_requests"]["token_usage"] is None
        assert "token_usage" in metrics["unmeasured_metrics"]

        assert metrics["orchestration_steps"]["measurement_kind"] == "measured"
        assert metrics["orchestration_steps"]["count"] == 2

        assert metrics["tool_calls"]["total_calls"] == 1
        assert metrics["tool_duration_ms"]["measurement_kind"] == "measured"
        assert metrics["tool_duration_ms"]["measured_count"] == 1
        assert metrics["verification_duration_ms"]["measurement_kind"] == "measured"
        assert metrics["verification_duration_ms"]["measured_count"] >= 1
        assert metrics["policy_decisions"]["by_outcome"]["allow"] >= 1
        assert "ok_secret_body" not in json.dumps(metrics)
    finally:
        ledger.close()


def test_confirmation_wait_duration_monotonic_sync_and_async(tmp_path: Path) -> None:
    """Policy confirmation_wait_ms uses deterministic monotonic deltas for sync & async callbacks."""
    db_path = tmp_path / "obs_confirm.db"
    ledger = CentralExecutionLedger(db_path=db_path, raise_on_persistence_error=True)
    exec_id = "exec_obs_confirm_1"
    try:
        ledger.create_execution("Interactive policy test", execution_id=exec_id)
        eng = ExecutionPolicyEngine(config=ExecutionPolicyConfig(policy_mode="strict"))

        # Controlled monotonic clock progression: cb_t0=100.0, cb_end=100.25 -> 250.0ms
        clock = {"now": 100.0}

        def fake_mono() -> float:
            return clock["now"]

        def sync_cb(_pre: Any) -> bool:
            clock["now"] += 0.25
            return True

        with patch("zelyr.engine.execution_policy.time.monotonic", side_effect=fake_mono):
            dec_sync = eng.evaluate(
                "shell",
                {"command": "echo hi"},
                execution_id=exec_id,
                central_ledger=ledger,
                confirm_callback=sync_cb,
                consume_confirmation=True,
            )
        assert dec_sync.allowed is True
        assert dec_sync.confirmation_wait_ms == 250.0

        async def async_decline_cb(_pre: Any) -> bool:
            clock["now"] += 0.125
            return False

        with patch("zelyr.engine.execution_policy.time.monotonic", side_effect=fake_mono):
            dec_async = asyncio.run(
                eng.evaluate_async(
                    "shell",
                    {"command": "echo decline"},
                    execution_id=exec_id,
                    central_ledger=ledger,
                    confirm_callback=async_decline_cb,
                )
            )
        assert dec_async.outcome == PolicyDecisionOutcome.DENY
        assert dec_async.confirmation_wait_ms == 125.0

        metrics = collect_execution_metrics(ledger, exec_id)
        assert metrics["policy_decisions"]["total"] == 2
        assert metrics["policy_decisions"]["by_outcome"]["allow"] == 1
        assert metrics["policy_decisions"]["by_outcome"]["deny"] == 1
        cw = metrics["confirmation_wait_ms"]
        assert cw["measurement_kind"] == "measured"
        assert cw["measured_count"] == 2
        assert cw["total_ms"] == 375.0
        assert cw["avg_ms"] == 187.5
        assert cw["max_ms"] == 250.0
    finally:
        ledger.close()


def test_task_graph_parallel_observability_failures_and_timeouts(tmp_path: Path) -> None:
    """TaskGraph execution records steps, tool durations, timeouts, recovery, and failures accurately."""
    db_path = tmp_path / "obs_graph.db"
    ledger = CentralExecutionLedger(db_path=db_path, raise_on_persistence_error=True)
    scope = TaskLocalToolScope(
        tool_names=["fast_op", "slow_op"],
        tool_schemas=[
            {
                "type": "function",
                "function": {
                    "name": "fast_op",
                    "parameters": {"type": "object", "properties": {"v": {"type": "string"}}, "required": ["v"], "additionalProperties": False},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "slow_op",
                    "parameters": {"type": "object", "properties": {"v": {"type": "string"}}, "required": ["v"], "additionalProperties": False},
                },
            },
        ],
        tool_rings={"fast_op": 1, "slow_op": 1},
    )

    gate = asyncio.Event()

    async def exec_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "slow_op":
            await gate.wait()  # Interrupted deterministically by node_timeout
        return {"status": "success", "output": f"ran {args['v']}", "verified": True}

    exec_tool._task_local_tool_scope = scope  # type: ignore[attr-defined]

    p1 = PendingActionProposal(
        proposal_id="p_fast",
        execution_id="exec_obs_tg",
        original_call_id="c_fast",
        tool_id="fast_op",
        normalized_arguments={"v": "ok"},
        proposal_order=0,
        status=ProposalStatus.VALIDATED.value,
    )
    p2 = PendingActionProposal(
        proposal_id="p_slow",
        execution_id="exec_obs_tg",
        original_call_id="c_slow",
        tool_id="slow_op",
        normalized_arguments={"v": "hang"},
        proposal_order=1,
        status=ProposalStatus.VALIDATED.value,
    )
    tg = TaskGraph(
        graph_id="tg_obs_1",
        execution_id="exec_obs_tg",
        objective="Parallel observability check",
        nodes=[
            TaskNode(node_id="n1", proposal_ids=["p_fast"], order_index=0),
            TaskNode(node_id="n2", proposal_ids=["p_slow"], order_index=1, timeout_seconds=0.05),
        ],
        max_concurrency=2,
    )

    try:
        orch = NativeOrchestrator(
            tool_executor=exec_tool,
            tool_schemas=list(scope.tool_schemas),
            task_local_scope=scope,
            caller_ring=1,
            central_ledger=ledger,
        )
        asyncio.run(
            orch.execute_task_graph(
                tg,
                execution_id="exec_obs_tg",
                central_ledger=ledger,
                proposals=[p1, p2],
                max_concurrency=2,
                node_timeouts={"n2": 0.05},
                policy_config=ExecutionPolicyConfig(policy_mode="autopilot"),
                enable_recovery=True,
            )
        )

        metrics = collect_execution_metrics(ledger, "exec_obs_tg")
        assert metrics["found"] is True
        assert metrics["orchestration_steps"]["measurement_kind"] == "measured"
        assert metrics["orchestration_steps"]["count"] >= 1
        assert metrics["tool_calls"]["total_calls"] == 2
        assert metrics["tool_duration_ms"]["measured_count"] == 2
        assert metrics["failures_and_timeouts"]["timed_out_calls"] == 1
        assert metrics["recovery_attempts"]["count"] >= 1
    finally:
        gate.set()
        ledger.close()


def test_unmeasured_legacy_fields_are_unavailable_and_cli_metrics_work(tmp_path: Path) -> None:
    """Legacy events without monotonic durations distinguish 'unavailable' cleanly and CLI/fleet summaries work."""
    db_path = tmp_path / "obs_legacy_cli.db"
    ledger = CentralExecutionLedger(db_path=db_path, raise_on_persistence_error=True)
    try:
        ledger.create_execution("Legacy run", execution_id="exec_legacy_obs")
        ledger.record_event(
            "exec_legacy_obs",
            LedgerEventType.TOOL_EXECUTION_START,
            tool_id="echo",
            correlation_id="legacy_c1",
            status="started",
        )
        ledger.record_event(
            "exec_legacy_obs",
            LedgerEventType.TOOL_EXECUTION_RESULT,
            tool_id="echo",
            correlation_id="legacy_c1",
            status="failed",
            metadata={"exit_code": 1},
        )
        ledger.finalize_execution("exec_legacy_obs", status=ExecutionStatus.FAILED)

        rep = collect_execution_metrics(ledger, "exec_legacy_obs")
        assert rep["model_requests"]["measurement_kind"] == "unavailable"
        assert rep["model_requests"]["count"] is None
        assert rep["tool_duration_ms"]["measurement_kind"] == "unavailable"
        assert rep["tool_duration_ms"]["total_ms"] is None
        assert rep["failures_and_timeouts"]["failed_calls"] == 1
        assert "token_usage" in rep["unmeasured_metrics"]
        assert "tool_execution_duration_ms" in rep["unmeasured_metrics"]

        fleet = collect_fleet_observability(ledger, limit=10)
        assert fleet["execution_count"] == 1
        assert "Execution & Policy Observability" in format_observability_summary(fleet)
    finally:
        ledger.close()

    r_fleet = _run_cli(["metrics", "--db-path", str(db_path), "--json"])
    assert r_fleet.returncode == 0, r_fleet.stderr
    assert json.loads(r_fleet.stdout)["execution_count"] == 1

    r_single = _run_cli(["execution", "exec_legacy_obs", "--metrics", "--db-path", str(db_path)])
    assert r_single.returncode == 0, r_single.stderr
    assert "Execution Metrics: exec_legacy_obs" in r_single.stdout

    r_missing = _run_cli(["metrics", "missing_exec_id", "--db-path", str(db_path), "--json"])
    assert r_missing.returncode != 0
