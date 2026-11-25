"""Offline unit tests for Zelyr Distributed Multi-Model Evaluation Lab."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List
from unittest.mock import patch

import httpx
import pytest

from zelyr.config import get_config
from zelyr.eval.cli import run_benchmark_cli
from zelyr.eval.distributed import (
    DistributedEvalRunner,
    DistributedTaskSpec,
    DistributedTaskStatus,
    ModelMetadata,
    TaskAssignment,
    WorkerConfig,
    WorkerSnapshot,
    build_matrix_assignments,
    compute_interval_overlaps,
    probe_worker_async,
)
from zelyr.eval.distributed_report import generate_distributed_reports
from zelyr.eval.distributed_worker import _run_worker


class TestWorkerIsolationAndPolicyContainment:
    """Verify subprocess worker isolation against parent overrides and workspace escape."""

    @pytest.mark.asyncio
    async def test_worker_blocks_parent_env_overrides_and_isolates_state(self, tmp_path: Path, monkeypatch):
        parent_ledger = tmp_path / "parent_ledger.db"
        monkeypatch.setenv("ZELYR_OLLAMA_BASE_URL", "http://198.51.100.99:9999")
        monkeypatch.setenv("AXIOM_OLLAMA_BASE_URL", "http://198.51.100.88:8888")
        monkeypatch.setenv("ZELYR_LEDGER_DB", str(parent_ledger))
        monkeypatch.setenv("ZELYR_NUM_CTX", "2048")

        ws = tmp_path / "worker_ws"
        state_dir = tmp_path / "worker_state"
        observed: Dict[str, Any] = {}

        async def _fake_headless(prompt, model_override=None, output_json=False, orchestrator=None, verify_spec=None):
            cfg = get_config()
            observed["endpoint"] = cfg.get_effective_ollama_base_url()
            observed["num_ctx"] = cfg.get_effective_num_ctx()
            observed["config_dir"] = os.environ.get("ZELYR_CONFIG_DIR")
            observed["home"] = os.environ.get("HOME")
            observed["axiom_url"] = os.environ.get("AXIOM_OLLAMA_BASE_URL")
            observed["ledger_db"] = os.environ.get("ZELYR_LEDGER_DB")
            observed["pol_cfg"] = dict(orchestrator.execution_policy_config.resource_restrictions)
            observed["ws_root"] = str(orchestrator.workspace_root)
            return 0

        with patch("zelyr.eval.distributed_worker._execute_headless_run", side_effect=_fake_headless):
            out = await _run_worker({
                "endpoint": "http://127.0.0.1:11434",
                "model": "qwen3:4b",
                "num_ctx": 8192,
                "workspace_dir": str(ws),
                "isolated_state_dir": str(state_dir),
                "prompt": "Verify isolated workspace config.",
                "policy_mode": "basic",
            })

        assert out["exit_code"] == 0
        assert observed["endpoint"] == "http://127.0.0.1:11434"
        assert observed["num_ctx"] == 8192
        assert observed["axiom_url"] is None
        assert Path(observed["config_dir"]) == (state_dir / "config").resolve()
        assert Path(observed["home"]) == (state_dir / "home").resolve()
        assert Path(observed["ledger_db"]) == (state_dir / "config" / "worker_executions.db").resolve()
        assert observed["pol_cfg"]["enforce_workspace_containment"] is True
        assert observed["pol_cfg"]["workspace_root"] == str(ws.resolve())
        assert observed["ws_root"] == str(ws.resolve())
        assert not parent_ledger.exists()

        # Parent env vars must be restored cleanly after worker exits
        assert os.environ.get("ZELYR_OLLAMA_BASE_URL") == "http://198.51.100.99:9999"
        assert os.environ.get("AXIOM_OLLAMA_BASE_URL") == "http://198.51.100.88:8888"
        assert os.environ.get("ZELYR_NUM_CTX") == "2048"

    @pytest.mark.asyncio
    async def test_worker_captures_inference_intervals_and_denies_outside_workspace_write(self, tmp_path: Path):
        ws = tmp_path / "ws"
        state_dir = tmp_path / "state"
        outside = tmp_path / "outside_secret.txt"

        async def _fake_headless(prompt, model_override=None, output_json=False, orchestrator=None, verify_spec=None):
            async def _fake_stream(_payload):
                yield {"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]}
                await asyncio.sleep(0.015)
                yield {"choices": [{"delta": {"content": "!"}, "finish_reason": "stop"}]}

            chunks = []
            async for ev in orchestrator._stream_model_events(
                {"model": model_override},
                "http://127.0.0.1:11434",
                stream_provider=_fake_stream,
            ):
                chunks.append(ev)
            assert len(chunks) == 2

            from zelyr.engine.execution_policy import evaluate_direct_tool_policy_async

            pol_dec = await evaluate_direct_tool_policy_async(
                "manage_workspace_file",
                {"action": "write", "path": str(outside), "content": "escape"},
                workspace_root=orchestrator.workspace_root,
                policy_engine=orchestrator.execution_policy_engine,
                policy_config=orchestrator.execution_policy_config,
            )
            assert not pol_dec.allowed
            assert pol_dec.outcome.value == "deny"
            assert pol_dec.execution_id == orchestrator.current_ledger._execution_id
            return 0

        with patch("zelyr.eval.distributed_worker._execute_headless_run", side_effect=_fake_headless):
            out = await _run_worker({
                "endpoint": "http://127.0.0.1:11434",
                "model": "qwen3:4b",
                "workspace_dir": str(ws),
                "isolated_state_dir": str(state_dir),
                "prompt": "Attempt outside write under containment.",
            })

        assert out["exit_code"] == 0
        assert out["worker_error"] is None
        assert not outside.exists()
        ivs = out["telemetry"]["inference_intervals"]
        assert len(ivs) == 1
        assert ivs[0]["model"] == "qwen3:4b"
        assert ivs[0]["chunk_count"] == 2
        assert ivs[0]["end_ts"] > ivs[0]["start_ts"]
        assert ivs[0]["first_chunk_latency_seconds"] is not None
        assert len(out["policy_decisions"]) >= 1
        assert out["policy_decisions"][0]["outcome"] == "deny"


class TestCompatibilityAndContextClamping:
    """Verify reused compatibility diagnostics, unknown RAM honesty, and context capping."""

    @pytest.mark.asyncio
    async def test_probe_worker_runs_compatibility_and_clamps_num_ctx(self, monkeypatch):
        async def _mock_get(_self, url: str, **_kwargs) -> httpx.Response:
            req = httpx.Request("GET", str(url))
            if str(url).endswith("/api/version"):
                return httpx.Response(200, json={"version": "0.40.16"}, request=req)
            if str(url).endswith("/api/tags"):
                return httpx.Response(
                    200,
                    json={
                        "models": [
                            {"name": "tool-small", "digest": "sha256:aaa", "size": 1000},
                            {"name": "no-tools", "digest": "sha256:bbb", "size": 500},
                        ]
                    },
                    request=req,
                )
            if str(url).endswith("/api/ps"):
                return httpx.Response(200, json={"models": []}, request=req)
            return httpx.Response(404, json={}, request=req)

        async def _mock_post(_self, url: str, **kwargs) -> httpx.Response:
            req = httpx.Request("POST", str(url))
            if str(url).endswith("/api/show"):
                body = kwargs.get("json") or {}
                name = body.get("model") or body.get("name")
                if name == "tool-small":
                    return httpx.Response(
                        200,
                        json={
                            "capabilities": ["completion", "tools"],
                            "model_info": {"qwen3.context_length": 4096},
                            "details": {"family": "qwen3", "parameter_size": "4B"},
                        },
                        request=req,
                    )
                return httpx.Response(
                    200,
                    json={
                        "capabilities": ["completion"],
                        "model_info": {"llama.context_length": 8192},
                        "details": {"family": "llama", "parameter_size": "135M"},
                    },
                    request=req,
                )
            return httpx.Response(404, json={}, request=req)

        monkeypatch.setattr(httpx.AsyncClient, "get", _mock_get)
        monkeypatch.setattr(httpx.AsyncClient, "post", _mock_post)

        worker = WorkerConfig(worker_id="w1", endpoint="http://worker-a.local:11434", num_ctx=16384)
        snap = await probe_worker_async(worker)

        assert snap.healthy is True
        assert snap.resource_info["host_ram_total_bytes"] is None
        assert "host_ram_total_bytes" in snap.resource_info["unknown_metrics"]
        assert snap.compatibility_by_model["tool-small"]["workflow_compatible"] is True
        assert snap.compatibility_by_model["tool-small"]["capabilities"]["native_tool_calling"]["status"] == "AVAILABLE"
        assert snap.compatibility_by_model["no-tools"]["capabilities"]["native_tool_calling"]["status"] == "UNAVAILABLE"

        resolved = worker.resolve_model_settings(
            "tool-small",
            model_context_length=snap.installed_models["tool-small"].context_length,
        )
        assert resolved["num_ctx"] == 4096
        assert resolved["configured_num_ctx"] == 16384
        assert resolved["context_clamped_to_model"] is True

        # Per-model think/options overrides propagate cleanly without leaking across models
        w_ovr = WorkerConfig(
            worker_id="w_ovr",
            endpoint="http://worker-a.local:11434",
            per_model_overrides={"bonsai-2:2b": {"think": False, "options": {"num_predict": 256}}},
        )
        assert w_ovr.resolve_model_settings("bonsai-2:2b")["think"] is False
        assert w_ovr.resolve_model_settings("bonsai-2:2b")["model_options"] == {"num_predict": 256}
        assert "think" not in w_ovr.resolve_model_settings("other-model")

        # Verify preflight returns UNSUPPORTED even if capabilities=[] when compatibility reports unavailable tools
        from zelyr.eval.distributed import ResourceAwareScheduler

        snap.installed_models["no-tools"].capabilities = []
        sched = ResourceAwareScheduler([worker], {"w1": snap})
        t_tool = DistributedTaskSpec(
            task_id="S1",
            category="S1_basic",
            title="S1",
            required_capabilities=["completion", "tools"],
            build_case=lambda ws: {"prompt": "hi", "evaluate": lambda ec, _l, _t: ec == 0},
        )
        st, reason = sched.preflight_assignment(TaskAssignment(worker_id="w1", model="no-tools", task=t_tool))
        assert st == DistributedTaskStatus.UNSUPPORTED
        assert "native tool calling unavailable" in (reason or "").lower()


class TestModesOverlapProofAndComparison:
    """Verify sequential, concurrent, and comparison modes plus overlap proof and parity checks."""

    @pytest.mark.asyncio
    async def test_sequential_vs_concurrent_overlap_and_comparison_parity(self, tmp_path: Path):
        w_a = WorkerConfig(
            worker_id="worker-a",
            endpoint="http://127.0.0.1:11434",
            model_allowlist=["m-a"],
            num_ctx=8192,
            max_concurrency=1,
        )
        w_b = WorkerConfig(
            worker_id="worker-b",
            endpoint="http://192.0.2.50:11434",
            model_allowlist=["m-b"],
            num_ctx=4096,
            max_concurrency=1,
        )

        async def _prober(w: WorkerConfig) -> WorkerSnapshot:
            m = w.model_allowlist[0]
            return WorkerSnapshot(
                worker_id=w.worker_id,
                endpoint=w.endpoint,
                healthy=True,
                version="0.40.16",
                installed_models={
                    m: ModelMetadata(
                        name=m,
                        digest=f"dig-{m}",
                        size_bytes=1_000_000_000,
                        capabilities=["completion", "tools"],
                        context_length=32768,
                        compatibility={"compatible": True, "workflow": "planned_graph"},
                    )
                },
                compatibility_by_model={m: {"compatible": True, "workflow": "planned_graph"}},
                resource_info={"host_ram_total_bytes": None, "unknown_metrics": ["host_ram_total"]},
            )

        async def _exec(w: WorkerConfig, m: str, task: DistributedTaskSpec, ws: Path, case: Dict[str, Any]):
            # Intentionally avoid mutating user-visible case with private keys when using custom executor
            assert "_resolved_model_settings" not in case
            t0 = time.time()
            await asyncio.sleep(0.08)
            t1 = time.time()
            return {
                "exit_code": 0,
                "cli_json_valid": True,
                "cli_json": {"response": "done"},
                "execution_ledger": {
                    "has_unresolved_failures": False,
                    "records": [{"call_id": "c1", "tool_name": "manage_workspace_file", "arguments": {}, "state": "verified", "verified": True}],
                },
                "verifier_checks": [{"check": "file_contains", "target": "alpha.txt", "success": True}],
                "telemetry": {
                    "turn_metrics": [],
                    "inference_intervals": [
                        {
                            "turn": 1,
                            "model": m,
                            "endpoint": w.endpoint,
                            "start_ts": t0,
                            "first_chunk_ts": t0 + 0.01,
                            "end_ts": t1,
                            "duration_seconds": round(t1 - t0, 4),
                            "first_chunk_latency_seconds": 0.01,
                            "chunk_count": 3,
                        }
                    ],
                },
                "ledger_transitions": [{"sequence_number": 1, "event_type": "tool_executed", "status": "verified"}],
                "policy_decisions": [{"decision_id": "p1", "outcome": "allow"}],
            }

        task_s1 = DistributedTaskSpec(
            task_id="S1",
            category="S1_basic",
            title="S1",
            build_case=lambda ws: {"prompt": f"Create {ws / 'a.txt'}", "evaluate": lambda ec, _l, _t: ec == 0},
        )

        runner_seq = DistributedEvalRunner(
            workers=[w_a, w_b],
            output_dir=tmp_path / "seq_out",
            task_executor=_exec,
            worker_prober=_prober,
            ps_fetcher=lambda _w: asyncio.sleep(0, result=[]),
            model_unloader=lambda w, _pre, _m: asyncio.sleep(0, result={"worker_id": w.worker_id, "unloaded": [], "errors": []}),
        )
        snaps = await runner_seq.discover_workers()
        assignments = build_matrix_assignments([w_a, w_b], snaps, [task_s1], mode="comparison")

        seq_out = await runner_seq.run_mode("sequential", assignments)
        assert seq_out["execution_mode"] == "sequential"
        assert seq_out["overlap_proof"]["verified_cross_machine_inference_overlap"] is False
        assert seq_out["overlap_proof"]["verified_cross_machine_task_overlap"] is False

        runner_conc = DistributedEvalRunner(
            workers=[w_a, w_b],
            output_dir=tmp_path / "cmp_out",
            task_executor=_exec,
            worker_prober=_prober,
            ps_fetcher=lambda _w: asyncio.sleep(0, result=[]),
            model_unloader=lambda w, _pre, _m: asyncio.sleep(0, result={"worker_id": w.worker_id, "unloaded": [], "errors": []}),
        )
        runner_conc.snapshots = snaps
        cmp_out = await runner_conc.run_mode("comparison", assignments)
        assert cmp_out["execution_mode"] == "comparison"
        assert cmp_out["overlap_proof"]["verified_cross_machine_inference_overlap"] is True
        assert cmp_out["overlap_proof"]["verified_cross_machine_task_overlap"] is True
        assert len(cmp_out["overlap_proof"]["cross_machine_inference_overlap_pairs"]) == 1
        assert cmp_out["comparison"]["fair_comparison"] is False
        assert cmp_out["comparison"]["parity_differences"][0]["num_ctx_values"] == [4096, 8192]

        paths = generate_distributed_reports(cmp_out, runner_conc.output_dir)
        report_json = json.loads(paths["distributed_report_json"].read_text(encoding="utf-8"))
        summary_md = paths["distributed_summary_md"].read_text(encoding="utf-8")
        assert report_json["execution_mode"] == "comparison"
        assert report_json["overlap_proof"]["verified_cross_machine_inference_overlap"] is True
        assert report_json["cross_model_comparison"]["fair_comparison"] is False
        assert "Cross-Machine Concurrency & Inference Overlap Proof" in summary_md
        assert "DIFFERENCES DETECTED" in summary_md

        with pytest.raises(ValueError, match="Unsupported execution mode"):
            await runner_conc.run_mode("invalid_mode", assignments)

    def test_overlap_helper_rejects_same_endpoint_collision(self):
        # Even with distinct worker_ids, identical endpoint URLs must NOT prove cross-machine overlap
        from zelyr.eval.distributed import DistributedTaskResult

        r1 = DistributedTaskResult(
            model="m1",
            model_digest="d1",
            worker_id="w1",
            endpoint="http://127.0.0.1:11434",
            task_id="S1",
            category="c",
            status=DistributedTaskStatus.PASS,
            start_timestamp=10.0,
            end_timestamp=11.0,
            duration_seconds=1.0,
            inference_intervals=[{"endpoint": "http://127.0.0.1:11434", "start_ts": 10.1, "end_ts": 10.9}],
        )
        r2 = DistributedTaskResult(
            model="m2",
            model_digest="d2",
            worker_id="w2",
            endpoint="http://127.0.0.1:11434",
            task_id="S1",
            category="c",
            status=DistributedTaskStatus.PASS,
            start_timestamp=10.2,
            end_timestamp=11.2,
            duration_seconds=1.0,
            inference_intervals=[{"endpoint": "http://127.0.0.1:11434", "start_ts": 10.3, "end_ts": 10.8}],
        )
        proof = compute_interval_overlaps([r1, r2])
        assert proof["verified_cross_machine_inference_overlap"] is False
        assert proof["verified_cross_machine_task_overlap"] is False


class TestCLIDistributedModesAndSuites:
    """Verify CLI flags --mode, --suite, --max-tasks, --no-unload offline."""

    def test_cli_distributed_modes_and_suites_offline(self, tmp_path: Path):
        cfg_file = tmp_path / "workers.json"
        cfg_file.write_text(
            json.dumps({
                "workers": [
                    {"worker_id": "w_local", "endpoint": "http://127.0.0.1:11434", "model_allowlist": ["m1"], "num_ctx": 8192},
                    {"worker_id": "w_remote", "endpoint": "http://192.0.2.5:11434", "model_allowlist": ["m2"], "num_ctx": 8192},
                ]
            }),
            encoding="utf-8",
        )

        out_seq = tmp_path / "out_seq"
        with patch("httpx.AsyncClient", side_effect=AssertionError("Offline mock must not touch network")):
            rc_seq = run_benchmark_cli([
                "--distributed",
                "--mock",
                "--workers-config",
                str(cfg_file),
                "--mode",
                "sequential",
                "--suite",
                "smoke",
                "--no-unload",
                "--out-dir",
                str(out_seq),
                "--quiet",
            ])
        assert rc_seq == 0
        seq_doc = json.loads((out_seq / "distributed_report.json").read_text(encoding="utf-8"))
        assert seq_doc["execution_mode"] == "sequential"
        assert len(seq_doc["results"]) == 2
        assert seq_doc["overlap_proof"]["verified_cross_machine_inference_overlap"] is False

        out_cmp = tmp_path / "out_cmp"
        with patch("httpx.AsyncClient", side_effect=AssertionError("Offline mock must not touch network")):
            rc_cmp = run_benchmark_cli([
                "--distributed",
                "--mock",
                "--workers-config",
                str(cfg_file),
                "--mode",
                "comparison",
                "--suite",
                "s1_s10",
                "--max-tasks",
                "2",
                "--out-dir",
                str(out_cmp),
                "--quiet",
            ])
        assert rc_cmp == 0
        cmp_doc = json.loads((out_cmp / "distributed_report.json").read_text(encoding="utf-8"))
        assert cmp_doc["execution_mode"] == "comparison"
        assert len(cmp_doc["results"]) == 4
        assert cmp_doc["overlap_proof"]["verified_cross_machine_inference_overlap"] is True
        assert cmp_doc["cross_model_comparison"]["fair_comparison"] is True


class TestDistributedEvaluationReliability:
    """Comprehensive regression suite for attribution, overlap evidence, comparison honesty, failures, and integrity."""

    @pytest.mark.asyncio
    async def test_worker_execution_id_lifecycle_and_fallback_continuity(self, tmp_path: Path):
        ws = tmp_path / "ws_exec_attr"
        ws.mkdir()
        target_file = ws / "out.txt"
        state_dir = tmp_path / "state_exec_attr"
        seen_stream_exec_ids: List[str] = []

        orig_init = __import__("zelyr.agents.native_orchestrator", fromlist=["NativeOrchestrator"]).NativeOrchestrator

        turn = {"n": 0}

        async def _fake_ollama_stream(_payload):
            from zelyr.engine.execution_ledger import get_active_execution_context

            ctx = get_active_execution_context()
            assert ctx is not None
            seen_stream_exec_ids.append(ctx.execution_id)
            turn["n"] += 1
            if turn["n"] == 1:
                # Fallback JSON text tool call triggers outer _execute_headless_run loop step 2!
                content = json.dumps({
                    "name": "manage_workspace_file",
                    "arguments": {"action": "write", "path": str(target_file), "content": "ALPHA_101"},
                })
                yield {"choices": [{"delta": {"content": content}, "finish_reason": "stop"}]}
            else:
                yield {"choices": [{"delta": {"content": "Completed task cleanly."}, "finish_reason": "stop"}]}

        with patch.object(orig_init, "stream_provider", None, create=True), \
             patch("zelyr.agents.native_orchestrator.NativeOrchestrator._stream_model_events", side_effect=lambda p, u, stream_provider=None: _fake_ollama_stream(p)), \
             patch("zelyr.agents.native_orchestrator.NativeOrchestrator.get_active_window", return_value=None), \
             patch("zelyr.agents.native_orchestrator.NativeOrchestrator.get_active_window_sync", return_value=None):
            out = await _run_worker({
                "worker_id": "victus",
                "task_id": "S1",
                "endpoint": "http://127.0.0.1:11434",
                "model": "qwen2.5:7b",
                "workspace_dir": str(ws),
                "isolated_state_dir": str(state_dir),
                "prompt": f"Create {target_file} containing ALPHA_101.",
                "verify_spec": f"file_contains:{target_file}:ALPHA_101",
            })

        assert out["exit_code"] == 0
        assert target_file.read_text(encoding="utf-8") == "ALPHA_101"
        exec_id = out["execution_id"]
        assert isinstance(exec_id, str) and exec_id.startswith("exec_")
        assert out["execution_ledger"]["execution_id"] == exec_id
        # Both multi-step iterations retained the exact same execution_id
        assert len(seen_stream_exec_ids) == 2
        assert all(eid == exec_id for eid in seen_stream_exec_ids)
        # Non-empty ordered ledger transitions belonging strictly to exec_id
        transitions = out["ledger_transitions"]
        assert len(transitions) >= 2
        seqs = [t["sequence_number"] for t in transitions]
        assert seqs == sorted(seqs)
        assert all(t["execution_id"] == exec_id for t in transitions)
        assert len(out["policy_decisions"]) >= 1
        assert all(p["execution_id"] == exec_id for p in out["policy_decisions"])

    @pytest.mark.asyncio
    async def test_concurrent_workers_ledger_isolation_and_contextvar_scope(self, tmp_path: Path):
        shared_db = tmp_path / "shared_worker_executions.db"
        ws_a = tmp_path / "ws_a"
        ws_b = tmp_path / "ws_b"
        state_a = tmp_path / "state_a"
        state_b = tmp_path / "state_b"

        async def _fake_headless(prompt, model_override=None, output_json=False, orchestrator=None, verify_spec=None):
            from zelyr.engine.execution_ledger import LedgerEventType, get_active_execution_context
            from zelyr.engine.execution_policy import evaluate_direct_tool_policy_async

            ctx = get_active_execution_context()
            assert ctx is not None
            await asyncio.sleep(0.02 if "ALPHA" in prompt else 0.01)
            ctx_after = get_active_execution_context()
            assert ctx_after is not None and ctx_after.execution_id == ctx.execution_id

            target = orchestrator.workspace_root / ("a.txt" if "ALPHA" in prompt else "b.txt")
            pol = await evaluate_direct_tool_policy_async(
                "manage_workspace_file",
                {"action": "write", "path": str(target), "content": prompt},
                workspace_root=orchestrator.workspace_root,
                policy_engine=orchestrator.execution_policy_engine,
                policy_config=orchestrator.execution_policy_config,
            )
            assert pol.execution_id == ctx.execution_id
            orchestrator.central_ledger.record_event(
                ctx.execution_id,
                LedgerEventType.TOOL_EXECUTION_RESULT,
                tool_id="manage_workspace_file",
                status="succeeded",
                metadata={"marker": f"worker_marker_{model_override}", "verified": True},
            )
            orchestrator.current_ledger.record_proposed("c1", "manage_workspace_file", {"path": str(target)})
            orchestrator.current_ledger.record_validated("c1")
            orchestrator.current_ledger.record_dispatched("c1")
            orchestrator.current_ledger.record_observation("c1", status="success", verified=True)
            return 0

        with patch("zelyr.eval.distributed_worker._execute_headless_run", side_effect=_fake_headless):
            out_a, out_b = await asyncio.gather(
                _run_worker({
                    "worker_id": "worker_a",
                    "task_id": "T_A",
                    "endpoint": "http://127.0.0.1:11434",
                    "model": "model_alpha",
                    "workspace_dir": str(ws_a),
                    "isolated_state_dir": str(state_a),
                    "ledger_db_path": str(shared_db),
                    "prompt": "Write ALPHA payload",
                }),
                _run_worker({
                    "worker_id": "worker_b",
                    "task_id": "T_B",
                    "endpoint": "http://192.0.2.5:11434",
                    "model": "model_beta",
                    "workspace_dir": str(ws_b),
                    "isolated_state_dir": str(state_b),
                    "ledger_db_path": str(shared_db),
                    "prompt": "Write BETA payload",
                }),
            )

        assert out_a["execution_id"] and out_b["execution_id"]
        assert out_a["execution_id"] != out_b["execution_id"]
        assert len(out_a["ledger_transitions"]) >= 2
        assert len(out_b["ledger_transitions"]) >= 2
        assert all(t["execution_id"] == out_a["execution_id"] for t in out_a["ledger_transitions"])
        assert all(t["execution_id"] == out_b["execution_id"] for t in out_b["ledger_transitions"])
        assert any("worker_marker_model_alpha" in json.dumps(t) for t in out_a["ledger_transitions"])
        assert not any("worker_marker_model_beta" in json.dumps(t) for t in out_a["ledger_transitions"])
        assert all(p["execution_id"] == out_a["execution_id"] for p in out_a["policy_decisions"])
        assert all(p["execution_id"] == out_b["execution_id"] for p in out_b["policy_decisions"])

    @pytest.mark.asyncio
    async def test_cancelled_and_failed_worker_transcripts_and_terminal_ledger(self, tmp_path: Path):
        ws_c = tmp_path / "ws_cancel"
        state_c = tmp_path / "state_cancel"
        ws_f = tmp_path / "ws_fail"
        state_f = tmp_path / "state_fail"

        async def _cancel_headless(**_kwargs):
            raise asyncio.CancelledError()

        async def _fail_headless(**_kwargs):
            return 2

        with patch("zelyr.eval.distributed_worker._execute_headless_run", side_effect=_cancel_headless):
            out_cancel = await _run_worker({
                "endpoint": "http://127.0.0.1:11434",
                "model": "m1",
                "workspace_dir": str(ws_c),
                "isolated_state_dir": str(state_c),
                "prompt": "Cancelled run",
            })
        assert out_cancel["cancelled"] is True
        assert out_cancel["execution_id"]
        assert any(t["status"] == "cancelled" for t in out_cancel["ledger_transitions"])

        with patch("zelyr.eval.distributed_worker._execute_headless_run", side_effect=_fail_headless):
            out_fail = await _run_worker({
                "endpoint": "http://127.0.0.1:11434",
                "model": "m1",
                "workspace_dir": str(ws_f),
                "isolated_state_dir": str(state_f),
                "prompt": "Failing run",
            })
        assert out_fail["exit_code"] == 2
        assert any(t["status"] == "failed" for t in out_fail["ledger_transitions"])

        # Runner persists transcripts even on CancelledError or exception
        w = WorkerConfig(worker_id="w1", endpoint="http://127.0.0.1:11434", model_allowlist=["m1"])

        async def _prober(_w):
            return WorkerSnapshot(
                worker_id="w1",
                endpoint=_w.endpoint,
                healthy=True,
                installed_models={"m1": ModelMetadata("m1", "d1", 1000, ["completion", "tools"])},
            )

        async def _exec(_w, _m, task, _ws, _case):
            if task.task_id == "T_CANCEL":
                raise asyncio.CancelledError()
            raise RuntimeError("http://user:secretpw@worker:11434 failed")

        runner = DistributedEvalRunner(
            workers=[w],
            output_dir=tmp_path / "runner_cf",
            task_executor=_exec,
            worker_prober=_prober,
            ps_fetcher=lambda _w: asyncio.sleep(0, result=[]),
            model_unloader=lambda _w, _pre, _m: asyncio.sleep(0, result={"worker_id": "w1", "unloaded": [], "errors": []}),
        )
        t_c = DistributedTaskSpec("T_CANCEL", "cat", "Cancel", lambda _ws: {"prompt": "c"})
        t_e = DistributedTaskSpec("T_ERR", "cat", "Err", lambda _ws: {"prompt": "e"})
        out = await runner.execute_assignments([TaskAssignment("w1", "m1", t_c), TaskAssignment("w1", "m1", t_e)], concurrent=False)
        by_id = {r.task_id: r for r in out["results"]}
        assert by_id["T_CANCEL"].status == DistributedTaskStatus.CANCELLED
        assert by_id["T_CANCEL"].transcript_path and Path(by_id["T_CANCEL"].transcript_path).is_file()
        assert by_id["T_ERR"].status == DistributedTaskStatus.ERROR
        assert by_id["T_ERR"].transcript_path and Path(by_id["T_ERR"].transcript_path).is_file()
        assert "secretpw" not in (by_id["T_ERR"].error_detail or "")

    def test_concurrency_interval_validation_boundaries_and_same_endpoint_rejection(self):
        from zelyr.eval.distributed import DistributedTaskResult

        def _res(wid: str, ep: str, aid: str, eid: str, s: float, e: float, ivs: List[Dict[str, Any]]):
            return DistributedTaskResult(
                model="m",
                model_digest="d",
                worker_id=wid,
                endpoint=ep,
                task_id="S1",
                category="c",
                status=DistributedTaskStatus.PASS,
                start_timestamp=s,
                end_timestamp=e,
                duration_seconds=max(0.0, e - s),
                assignment_id=aid,
                execution_id=eid,
                inference_intervals=ivs,
            )

        # 1. Exact boundary touch [100, 105] vs [105, 110] -> NO overlap observed (valid distinct endpoints)
        p_touch = compute_interval_overlaps([
            _res("w1", "http://127.0.0.1:11434", "a1", "e1", 100.0, 105.0, [{"start_ts": 100.0, "end_ts": 105.0}]),
            _res("w2", "http://192.0.2.5:11434", "a2", "e2", 105.0, 110.0, [{"start_ts": 105.0, "end_ts": 110.0}]),
        ])
        assert p_touch["verified_cross_machine_inference_overlap"] is False
        assert p_touch["no_overlap_observed"] is True
        assert p_touch["insufficient_evidence"] is False
        assert p_touch["evidence_status"] == "no_overlap_observed"

        # 2. Malformed/negative/NaN/boolean inference intervals rejected cleanly even if task times overlap
        p_bad = compute_interval_overlaps([
            _res("w1", "http://127.0.0.1:11434", "a1", "e1", 100.0, 110.0, [
                {"start_ts": 105.0, "end_ts": 101.0},  # clock anomaly
                {"start_ts": True, "end_ts": 109.0},   # boolean invalid
                {"start_ts": float("nan"), "end_ts": 109.0},
            ]),
            _res("w2", "http://192.0.2.5:11434", "a2", "e2", 100.0, 110.0, [
                {"start_ts": 102.0, "end_ts": 108.0},
            ]),
        ])
        assert p_bad["rejected_inference_intervals_count"] == 3
        assert len(p_bad["clock_anomalies"]) >= 2
        assert p_bad["verified_cross_machine_inference_overlap"] is False
        assert p_bad["verified_cross_machine_task_overlap"] is True
        assert p_bad["insufficient_evidence"] is True
        assert p_bad["evidence_status"] == "verified_cross_machine_task_overlap_only"

        # 3. Alias endpoints (localhost vs 127.0.0.1/) rejected as same endpoint
        p_alias = compute_interval_overlaps([
            _res("w1", "http://localhost:11434/", "a1", "e1", 100.0, 110.0, [{"start_ts": 101.0, "end_ts": 108.0}]),
            _res("w2", "http://127.0.0.1:11434", "a2", "e2", 102.0, 112.0, [{"start_ts": 103.0, "end_ts": 109.0}]),
        ])
        assert p_alias["verified_cross_machine_inference_overlap"] is False
        assert p_alias["rejected_same_endpoint_overlaps_count"] >= 1
        assert p_alias["insufficient_evidence"] is True

        # 4. Two inference turns within ONE task must never count as cross-machine overlap
        p_single = compute_interval_overlaps([
            _res("w1", "http://127.0.0.1:11434", "a1", "e1", 100.0, 110.0, [
                {"turn": 1, "start_ts": 101.0, "end_ts": 105.0},
                {"turn": 2, "start_ts": 102.0, "end_ts": 108.0},
            ]),
        ])
        assert p_single["verified_cross_machine_inference_overlap"] is False
        assert p_single["insufficient_evidence"] is True
        assert p_single["evidence_status"] == "insufficient_evidence"

    def test_functional_vs_configuration_vs_performance_comparability(self):
        from zelyr.eval.distributed import DistributedTaskResult, build_cross_model_comparison, compute_configuration_fingerprint

        task_s1 = DistributedTaskSpec("S1", "core", "S1", lambda ws: {"prompt": "hi"})
        w_v = WorkerConfig("victus", "http://127.0.0.1:11434", num_ctx=4096)
        w_a = WorkerConfig("alpha-node", "http://192.0.2.5:11434", num_ctx=4096, per_model_overrides={"bonsai-2:2b": {"think": False}})

        meta_q = ModelMetadata("qwen2.5:7b", "dig-q", 4_400_000_000, ["completion", "tools"], 32768, family="qwen2", parameter_size="7.6B", quantization_level="Q4_K_M")
        meta_b = ModelMetadata("bonsai-2:2b", "dig-b", 900_000_000, ["completion", "tools", "thinking"], 262144, family="qwen35", parameter_size="1.9B", quantization_level="Q2_0")
        s_q = w_v.resolve_model_settings("qwen2.5:7b", model_context_length=meta_q.context_length)
        s_b = w_a.resolve_model_settings("bonsai-2:2b", model_context_length=meta_b.context_length)
        fp_q = compute_configuration_fingerprint(model="qwen2.5:7b", model_digest="dig-q", model_settings=s_q, task=task_s1, worker=w_v, model_meta=meta_q)
        fp_b = compute_configuration_fingerprint(model="bonsai-2:2b", model_digest="dig-b", model_settings=s_b, task=task_s1, worker=w_a, model_meta=meta_b)
        assert fp_q != fp_b

        r_q = DistributedTaskResult(
            model="qwen2.5:7b", model_digest="dig-q", worker_id="victus", endpoint="http://127.0.0.1:11434",
            task_id="S1", category="core", status=DistributedTaskStatus.PASS, start_timestamp=10.0, end_timestamp=20.35, duration_seconds=10.35,
            model_settings=s_q, execution_id="exec_q", configuration_fingerprint=fp_q,
            resource_measurements={"ps_after": [{"name": "qwen2.5:7b", "size": 4_400_000_000, "size_vram": 4_400_000_000}]},
            measured_token_usage={"prompt_eval_count_max": 2050, "eval_count_sum": 120},
            estimated_token_usage={"estimated_tokens_approx": 4000, "source": "character_heuristic"},
            inference_intervals=[{"turn": 1, "start_ts": 11.0, "end_ts": 20.0, "duration_seconds": 9.0, "first_chunk_latency_seconds": 1.2}],
        )
        r_b = DistributedTaskResult(
            model="bonsai-2:2b", model_digest="dig-b", worker_id="alpha-node", endpoint="http://192.0.2.5:11434",
            task_id="S1", category="core", status=DistributedTaskStatus.PASS, start_timestamp=10.0, end_timestamp=432.13, duration_seconds=422.13,
            model_settings=s_b, execution_id="exec_b", configuration_fingerprint=fp_b,
            resource_measurements={"ps_after": [{"name": "bonsai-2:2b", "size": 900_000_000, "size_vram": 0}]},
            measured_token_usage={"prompt_eval_count_max": 2050, "eval_count_sum": 80},
            estimated_token_usage={"estimated_tokens_approx": 3900, "source": "character_heuristic"},
            inference_intervals=[{"turn": 1, "start_ts": 11.0, "end_ts": 220.0, "duration_seconds": 209.0, "first_chunk_latency_seconds": 201.0}],
        )
        cmp_doc = build_cross_model_comparison(
            [TaskAssignment("victus", "qwen2.5:7b", task_s1), TaskAssignment("alpha-node", "bonsai-2:2b", task_s1)],
            [r_q, r_b],
            snapshots={
                "victus": WorkerSnapshot("victus", w_v.endpoint, True, installed_models={"qwen2.5:7b": meta_q}),
                "alpha-node": WorkerSnapshot("alpha-node", w_a.endpoint, True, installed_models={"bonsai-2:2b": meta_b}),
            },
            workers={"victus": w_v, "alpha-node": w_a},
        )
        assert cmp_doc["functional_parity"]["equal"] is True
        assert cmp_doc["configuration_parity"]["equal"] is False
        assert cmp_doc["fair_comparison"] is False
        assert cmp_doc["performance_comparison"]["comparable"] is False
        assert cmp_doc["performance_comparison"]["ranking_allowed"] is False
        assert len(cmp_doc["comparison_limitations"]) >= 2
        assert cmp_doc["performance_comparison"]["targets"]["alpha-node::bonsai-2:2b"]["measured_ps_vram_bytes_max"] == 0
        assert cmp_doc["performance_comparison"]["targets"]["victus::qwen2.5:7b"]["measured_prompt_tokens_max"] == 2050

    def test_granular_failure_taxonomy_and_timeout_mutating_uncertainty(self, tmp_path: Path):
        from zelyr.eval.distributed import ResourceAwareScheduler, _classify_task_outcome_detail

        w_down = WorkerConfig("w_down", "http://192.0.2.99:11434", model_allowlist=["m1"])
        w_ok = WorkerConfig("w_ok", "http://127.0.0.1:11434", model_allowlist=["small"], memory_budget_bytes=1000)
        snaps = {
            "w_down": WorkerSnapshot("w_down", w_down.endpoint, healthy=False, error="Connection refused"),
            "w_ok": WorkerSnapshot(
                "w_ok",
                w_ok.endpoint,
                healthy=True,
                installed_models={
                    "small": ModelMetadata("small", "d", 500, ["completion", "tools"]),
                    "big": ModelMetadata("big", "d", 5000, ["completion", "tools"]),
                    "nontool": ModelMetadata("nontool", "d", 500, ["completion"]),
                },
            ),
        }
        sched = ResourceAwareScheduler([w_down, w_ok], snaps)
        t_tools = DistributedTaskSpec("S1", "c", "t", lambda ws: {"prompt": "p"})

        st, cat, _ = sched.preflight_assignment_detail(TaskAssignment("w_down", "m1", t_tools))
        assert st == DistributedTaskStatus.UNAVAILABLE and cat == "WORKER_UNAVAILABLE"

        st, cat, _ = sched.preflight_assignment_detail(TaskAssignment("w_ok", "missing_m", t_tools))
        assert st == DistributedTaskStatus.UNAVAILABLE and cat == "MODEL_UNAVAILABLE"

        # Disallow check triggers MODEL_UNAVAILABLE; resource budget triggers RESOURCE_LIMIT
        w_ok.model_allowlist = []
        st, cat, _ = sched.preflight_assignment_detail(TaskAssignment("w_ok", "nontool", t_tools))
        assert st == DistributedTaskStatus.UNSUPPORTED and cat == "UNSUPPORTED"

        st, cat, _ = sched.preflight_assignment_detail(TaskAssignment("w_ok", "big", t_tools))
        assert st == DistributedTaskStatus.UNAVAILABLE and cat == "RESOURCE_LIMIT"

        # Timeout after mutating action must preserve uncertainty
        target_p = tmp_path / "mutated.txt"
        case = {"prompt": f"Write {target_p}"}
        d_to = _classify_task_outcome_detail(
            TaskAssignment("w_ok", "small", t_tools),
            case,
            {
                "timed_out": True,
                "exit_code": "TIMEOUT",
                "execution_ledger": {
                    "records": [
                        {"tool_name": "manage_workspace_file", "arguments": {"action": "write", "path": str(target_p)}, "state": "dispatched"}
                    ]
                },
            },
        )
        assert d_to["status"] == DistributedTaskStatus.TIMEOUT
        assert d_to["failure_category"] == "TIMEOUT"
        assert d_to["uncertain_side_effects"] is True
        assert str(target_p) in d_to["mutated_targets_uncertain"]

        # Runtime OOM & disconnected worker classification
        d_oom = _classify_task_outcome_detail(
            TaskAssignment("w_ok", "small", t_tools),
            case,
            {"exit_code": 1, "worker_error": "RuntimeError: CUDA out of memory", "execution_ledger": {"records": []}},
        )
        assert d_oom["status"] == DistributedTaskStatus.ERROR and d_oom["failure_category"] == "RESOURCE_LIMIT"

        d_disc = _classify_task_outcome_detail(
            TaskAssignment("w_ok", "small", t_tools),
            case,
            {"exit_code": 1, "worker_error": "ConnectTimeout: peer unreachable", "execution_ledger": {"records": []}},
        )
        assert d_disc["status"] == DistributedTaskStatus.ERROR and d_disc["failure_category"] == "WORKER_UNAVAILABLE"

    def test_benchmark_integrity_report_compatibility_and_secret_redaction(self, tmp_path: Path, monkeypatch):
        from zelyr.engine.execution_ledger import sanitize_ledger_text
        from zelyr.engine.normalizer import _redact_sensitive_text
        from zelyr.eval.distributed import DistributedTaskResult

        # Prove env-secret cache immediately invalidates upon runtime env injection
        monkeypatch.setenv("ZELYR_RUNTIME_API_SECRET", "env-super-secret-value-9911")
        cleaned, redacted, _ = sanitize_ledger_text("leaked env-super-secret-value-9911 here")
        assert redacted is True and "env-super-secret-value-9911" not in (cleaned or "")
        assert "env-super-secret-value-9911" not in _redact_sensitive_text("diag env-super-secret-value-9911")

        results = [
            DistributedTaskResult(
                model="m1", model_digest="", worker_id="w1", endpoint="http://alice:pwSecret99@192.0.2.1:11434",
                task_id="S1", category="core", status=DistributedTaskStatus.PASS, start_timestamp=1.0, end_timestamp=2.0, duration_seconds=1.0,
                execution_id="exec_1", failure_category="PASS", configuration_fingerprint="abcd1234",
                verification_result=[{"check": "file_contains", "success": True}],
            ),
            DistributedTaskResult(
                model="m2", model_digest="d2", worker_id="w2", endpoint="http://192.0.2.2:11434",
                task_id="S1", category="core", status=DistributedTaskStatus.ERROR, start_timestamp=1.0, end_timestamp=1.5, duration_seconds=0.5,
                execution_id=None, failure_category="WORKER_UNAVAILABLE", error_classification="ConnectError",
                error_detail="Failed with sk-secret-token-abc12345 at http://bob:pwBob88@192.0.2.2:11434",
                configuration_fingerprint="efgh5678",
            ),
        ]
        paths = generate_distributed_reports(
            {
                "execution_mode": "comparison",
                "results": results,
                "overlap_proof": {
                    "evidence_status": "insufficient_evidence",
                    "verified_cross_machine_inference_overlap": False,
                    "verified_cross_machine_task_overlap": False,
                    "no_overlap_observed": False,
                    "insufficient_evidence": True,
                },
                "comparison": {
                    "functional_parity": {"equal": True},
                    "configuration_parity": {"equal": False},
                    "performance_comparison": {"comparable": False, "limitations": ["Single-trial run"]},
                    "comparison_limitations": ["Single-trial run"],
                },
                "snapshots": {
                    "w1": {"worker_id": "w1", "endpoint": "http://alice:pwSecret99@192.0.2.1:11434", "healthy": True, "installed_models": {}, "resource_info": {"unknown_metrics": ["host_ram_total_bytes"]}},
                },
                "cleanup": [],
            },
            tmp_path / "integrity_out",
        )
        raw_rep = paths["distributed_report_json"].read_text(encoding="utf-8")
        raw_md = paths["distributed_summary_md"].read_text(encoding="utf-8")
        raw_snaps = paths["worker_snapshots_json"].read_text(encoding="utf-8")
        for secret in ("pwSecret99", "pwBob88", "sk-secret-token-abc12345"):
            assert secret not in raw_rep
            assert secret not in raw_md
            assert secret not in raw_snaps

        doc = json.loads(raw_rep)
        assert doc["report_version"] == "1.4"
        bi = doc["benchmark_integrity"]
        assert bi["total_tasks"] == 2
        assert bi["independently_verified_passes"] == 1
        assert bi["worker_errors"] == 1
        assert bi["failure_category_counts"]["WORKER_UNAVAILABLE"] == 1
        assert bi["performance_comparable"] is False
        # Missing digest stays explicit null, never invented
        assert doc["results"][0]["integrity"]["model_digest"] is None
        assert doc["results"][0]["integrity"]["execution_id"] == "exec_1"
        assert "Benchmark Integrity Overview" in raw_md
