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

            orchestrator.central_ledger.create_execution(task_prompt=prompt, execution_id="exec_iso_test")
            pol_dec = await orchestrator.execution_policy_engine.evaluate_async(
                "manage_workspace_file",
                {"action": "write", "path": str(outside), "content": "escape"},
                normalized_arguments={"action": "write", "path": str(outside), "content": "escape"},
                workspace_root=orchestrator.workspace_root,
                execution_id="exec_iso_test",
                central_ledger=orchestrator.central_ledger,
            )
            assert not pol_dec.allowed
            assert pol_dec.outcome.value == "deny"
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
