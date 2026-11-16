"""Offline mocked unit tests for Zelyr distributed multi-model evaluation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from zelyr.eval.cli import build_arg_parser
from zelyr.eval.distributed import (
    DistributedEvalRunner,
    DistributedTaskSpec,
    DistributedTaskStatus,
    ResourceAwareScheduler,
    TaskAssignment,
    WorkerConfig,
    build_compatibility_suite,
    build_standard_s_suite,
    load_worker_configs,
    probe_worker_async,
    unload_worker_eval_models_async,
)
from zelyr.eval.distributed_report import compute_model_metrics, generate_distributed_reports


class _MockResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class TestWorkerDiscoveryAndInventory:
    def test_backward_compatible_default_config_has_no_remote_ip(self, monkeypatch):
        monkeypatch.delenv("ZELYR_EVAL_WORKERS", raising=False)
        monkeypatch.delenv("ZELYR_OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        workers = load_worker_configs()
        assert len(workers) == 1
        assert workers[0].worker_id == "local"
        assert workers[0].endpoint == "http://127.0.0.1:11434"

    def test_load_worker_configs_from_json_and_clamp_concurrency(self, tmp_path):
        cfg_file = tmp_path / "workers.json"
        cfg_file.write_text(
            json.dumps({
                "workers": [
                    {
                        "worker_id": "victus",
                        "endpoint": "http://127.0.0.1:11434/",
                        "model_allowlist": ["qwen2.5:7b"],
                        "max_concurrency": 9,
                        "num_ctx": 8192,
                    },
                    {
                        "worker_id": "alpha-node",
                        "endpoint": "http://192.0.2.5:11434",
                        "model_allowlist": ["qwen2.5:3b"],
                        "max_concurrency": 2,
                        "num_ctx": 4096,
                    },
                ]
            }),
            encoding="utf-8",
        )
        workers = load_worker_configs(config_path=cfg_file)
        assert [w.worker_id for w in workers] == ["victus", "alpha-node"]
        assert workers[0].max_concurrency == 2
        assert workers[1].endpoint == "http://192.0.2.5:11434"

    @pytest.mark.asyncio
    async def test_multi_endpoint_discovery_and_health_checks(self):
        class MockAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def get(self, url, **kwargs):
                if url.startswith("http://127.0.0.1:11434"):
                    if url.endswith("/api/version"):
                        return _MockResponse(200, {"version": "0.40.11"})
                    if url.endswith("/api/ps"):
                        return _MockResponse(200, {"models": [{"name": "preloaded:1b", "size": 500000, "size_vram": 500000}]})
                    if url.endswith("/api/tags"):
                        return _MockResponse(
                            200,
                            {
                                "models": [
                                    {
                                        "name": "qwen2.5:7b",
                                        "digest": "sha256_v1",
                                        "size": 4683087332,
                                        "details": {"family": "qwen2", "parameter_size": "7.6B", "quantization_level": "Q4_K_M"},
                                    }
                                ]
                            },
                        )
                if url.startswith("http://192.0.2.5:11434"):
                    if url.endswith("/api/version"):
                        return _MockResponse(200, {"version": "0.40.16"})
                    if url.endswith("/api/ps"):
                        return _MockResponse(200, {"models": []})
                    if url.endswith("/api/tags"):
                        return _MockResponse(
                            200,
                            {
                                "models": [
                                    {
                                        "name": "d3:2b",
                                        "digest": "sha256_d3",
                                        "size": 3223742935,
                                        "details": {"family": "qwen35", "parameter_size": "2.4B", "quantization_level": "Q8_0"},
                                    }
                                ]
                            },
                        )
                raise httpx.ConnectError("connection refused")

            async def post(self, url, json=None, **kwargs):
                if url.startswith("http://127.0.0.1:11434/api/show"):
                    return _MockResponse(200, {
                        "capabilities": ["completion", "tools"],
                        "model_info": {"qwen2.context_length": 32768},
                    })
                if url.startswith("http://192.0.2.5:11434/api/show"):
                    return _MockResponse(200, {
                        "capabilities": ["decision", "vision"],
                        "model_info": {"qwen35.context_length": 262144},
                    })
                return _MockResponse(404, {})

        w_v = WorkerConfig(worker_id="victus", endpoint="http://127.0.0.1:11434")
        w_a = WorkerConfig(worker_id="alpha-node", endpoint="http://192.0.2.5:11434")
        w_down = WorkerConfig(worker_id="offline", endpoint="http://192.0.2.99:11434")

        with patch("httpx.AsyncClient", MockAsyncClient):
            s_v, s_a, s_down = await asyncio.gather(
                probe_worker_async(w_v),
                probe_worker_async(w_a),
                probe_worker_async(w_down),
            )

        assert s_v.healthy is True
        assert s_v.version == "0.40.11"
        assert s_v.pre_loaded_models == {"preloaded:1b"}
        assert s_v.installed_models["qwen2.5:7b"].digest == "sha256_v1"
        assert s_v.installed_models["qwen2.5:7b"].capabilities == ["completion", "tools"]
        assert s_v.installed_models["qwen2.5:7b"].context_length == 32768

        assert s_a.healthy is True
        assert s_a.installed_models["d3:2b"].capabilities == ["decision", "vision"]

        assert s_down.healthy is False
        assert "ConnectError" in (s_down.error or "")

    @pytest.mark.asyncio
    async def test_unload_never_touches_preloaded_models(self):
        unloaded_posts = []

        class MockAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def get(self, url, **kwargs):
                if len(unloaded_posts) == 0:
                    return _MockResponse(200, {
                        "models": [
                            {"name": "preloaded:1b", "size": 100, "size_vram": 100},
                            {"name": "eval:3b", "size": 200, "size_vram": 0},
                        ]
                    })
                return _MockResponse(200, {"models": [{"name": "preloaded:1b", "size": 100, "size_vram": 100}]})

            async def post(self, url, json=None, **kwargs):
                unloaded_posts.append((url, json))
                return _MockResponse(200, {"done": True})

        w = WorkerConfig(worker_id="victus", endpoint="http://127.0.0.1:11434")
        with patch("httpx.AsyncClient", MockAsyncClient):
            res = await unload_worker_eval_models_async(w, pre_loaded_models={"preloaded:1b"}, eval_models={"eval:3b"})

        assert res["unloaded"] == ["eval:3b"]
        assert len(unloaded_posts) == 1
        assert unloaded_posts[0][1] == {"model": "eval:3b", "keep_alive": 0, "messages": []}
        assert [m["name"] for m in res["loaded_after"]] == ["preloaded:1b"]


class TestSchedulerIsolationFailureAndReports:
    def _make_task(self, tid: str = "T1", caps=None) -> DistributedTaskSpec:
        return DistributedTaskSpec(
            task_id=tid,
            category="core_s1_s10",
            title="Test",
            required_capabilities=caps if caps is not None else ["completion", "tools"],
            build_case=lambda ws: {
                "prompt": f"Create {ws / 'a.txt'}",
                "verify": f"file_contains:{ws / 'a.txt'}:OK",
                "evaluate": lambda ec, ledger, tel: ec == 0,
            },
        )

    @pytest.mark.asyncio
    async def test_allowlist_capability_and_worker_failure_without_reroute(self):
        w_v = WorkerConfig(worker_id="victus", endpoint="http://127.0.0.1:11434", model_allowlist=["qwen2.5:7b"])
        w_a = WorkerConfig(worker_id="alpha-node", endpoint="http://192.0.2.5:11434", model_allowlist=["smol:1b"])

        async def prober(w: WorkerConfig):
            if w.worker_id == "victus":
                from zelyr.eval.distributed import ModelMetadata, WorkerSnapshot
                return WorkerSnapshot(
                    worker_id="victus",
                    endpoint=w.endpoint,
                    healthy=True,
                    installed_models={
                        "qwen2.5:7b": ModelMetadata("qwen2.5:7b", "dig1", 4_000_000_000, ["completion", "tools"]),
                        "other:9b": ModelMetadata("other:9b", "dig2", 5_000_000_000, ["completion", "tools"]),
                    },
                )
            from zelyr.eval.distributed import WorkerSnapshot
            return WorkerSnapshot(worker_id="alpha-node", endpoint=w.endpoint, healthy=False, error="timeout")

        seen_executions = []

        async def executor(worker, model, task, ws, case):
            seen_executions.append((worker.worker_id, model, task.task_id, worker.endpoint))
            return {"exit_code": 0, "execution_ledger": {"records": []}, "telemetry": {"turn_metrics": []}}

        runner = DistributedEvalRunner(
            workers=[w_v, w_a],
            task_executor=executor,
            worker_prober=prober,
            ps_fetcher=lambda _w: asyncio.sleep(0, result=[]),
            model_unloader=lambda w, pre, m: asyncio.sleep(0, result={"worker_id": w.worker_id, "unloaded": []}),
        )

        assigns = [
            TaskAssignment("victus", "qwen2.5:7b", self._make_task("T_OK")),
            TaskAssignment("victus", "other:9b", self._make_task("T_DISALLOWED")),
            TaskAssignment("victus", "qwen2.5:7b", self._make_task("T_THINK", caps=["completion", "tools", "thinking"])),
            TaskAssignment("alpha-node", "smol:1b", self._make_task("T_DOWN")),
        ]
        out = await runner.execute_assignments(assigns, concurrent=False)
        by_id = {r.task_id: r for r in out["results"]}

        assert by_id["T_OK"].status == DistributedTaskStatus.PASS
        assert by_id["T_DISALLOWED"].status == DistributedTaskStatus.UNAVAILABLE
        assert by_id["T_THINK"].status == DistributedTaskStatus.UNSUPPORTED
        assert by_id["T_DOWN"].status == DistributedTaskStatus.UNAVAILABLE
        # Strictly zero rerouting to victus for alpha-node's task
        assert seen_executions == [("victus", "qwen2.5:7b", "T_OK", "http://127.0.0.1:11434")]

    @pytest.mark.asyncio
    async def test_cross_machine_concurrent_execution_and_resource_budget(self, tmp_path):
        w_v = WorkerConfig(
            worker_id="victus",
            endpoint="http://127.0.0.1:11434",
            max_concurrency=2,
            memory_budget_bytes=10_000_000_000,
        )
        w_a = WorkerConfig(
            worker_id="alpha-node",
            endpoint="http://192.0.2.5:11434",
            max_concurrency=1,
            memory_budget_bytes=6_000_000_000,
        )

        async def prober(w: WorkerConfig):
            from zelyr.eval.distributed import ModelMetadata, WorkerSnapshot
            if w.worker_id == "victus":
                return WorkerSnapshot(
                    worker_id="victus",
                    endpoint=w.endpoint,
                    healthy=True,
                    installed_models={
                        "big_a": ModelMetadata("big_a", "d_ba", 6_000_000_000, ["completion", "tools"]),
                        "big_b": ModelMetadata("big_b", "d_bb", 6_000_000_000, ["completion", "tools"]),
                    },
                )
            return WorkerSnapshot(
                worker_id="alpha-node",
                endpoint=w.endpoint,
                healthy=True,
                installed_models={
                    "small_c": ModelMetadata("small_c", "d_sc", 2_000_000_000, ["completion", "tools"]),
                },
            )

        active_by_worker = {"victus": 0, "alpha-node": 0}
        both_active_observed = False

        async def executor(worker, model, task, ws, case):
            nonlocal both_active_observed
            active_by_worker[worker.worker_id] += 1
            # big_a + big_b together would be 13.8B > 10B budget, so victus must never co-schedule them
            assert active_by_worker["victus"] <= 1
            if active_by_worker["victus"] > 0 and active_by_worker["alpha-node"] > 0:
                both_active_observed = True
            await asyncio.sleep(0.05)
            active_by_worker[worker.worker_id] -= 1
            return {
                "exit_code": 0,
                "execution_ledger": {
                    "has_unresolved_failures": False,
                    "records": [{"call_id": "c1", "tool_name": "manage_workspace_file", "arguments": {"path": "a"}, "state": "verified", "verified": True}],
                },
                "telemetry": {
                    "turn_metrics": [{"prompt_eval_count": 321, "eval_count": 12, "total_duration_ms": 50.0, "load_duration_ms": 10.0}]
                },
            }

        runner = DistributedEvalRunner(
            workers=[w_v, w_a],
            output_dir=tmp_path / "dist_out",
            task_executor=executor,
            worker_prober=prober,
            ps_fetcher=lambda w: asyncio.sleep(0, result=[{"name": "big_a", "size": 6_000_000_000, "size_vram": 5_000_000_000}] if w.worker_id == "victus" else []),
            model_unloader=lambda w, pre, m: asyncio.sleep(0, result={"worker_id": w.worker_id, "unloaded": sorted(m)}),
        )
        assigns = [
            TaskAssignment("victus", "big_a", self._make_task("T_V1")),
            TaskAssignment("victus", "big_b", self._make_task("T_V2")),
            TaskAssignment("alpha-node", "small_c", self._make_task("T_A1")),
        ]
        out = await runner.execute_assignments(assigns, concurrent=True)
        assert both_active_observed is True
        assert out["scheduler_stats"]["peak_simultaneous_workers"] == 2
        assert "cross_machine_concurrent" in out["scheduler_stats"]["observed_concurrency_forms"]
        assert out["scheduler_stats"]["peak_concurrency_by_worker"]["victus"] == 1

        paths = generate_distributed_reports(out, tmp_path / "dist_out")
        assert paths["distributed_report_json"].exists()
        assert paths["distributed_summary_md"].exists()
        assert paths["worker_snapshots_json"].exists()
        report_json = json.loads(paths["distributed_report_json"].read_text(encoding="utf-8"))
        assert report_json["overall"]["p95_duration_s"] is None  # n < 20
        assert report_json["overall"]["context_usage_prompt_tokens_max"] == 321

    @pytest.mark.asyncio
    async def test_task_cancellation_triggers_cleanup(self):
        w = WorkerConfig(worker_id="victus", endpoint="http://127.0.0.1:11434")
        unloaded_calls = []

        async def prober(_w):
            from zelyr.eval.distributed import ModelMetadata, WorkerSnapshot
            return WorkerSnapshot(
                worker_id="victus",
                endpoint=w.endpoint,
                healthy=True,
                installed_models={"m1": ModelMetadata("m1", "d1", 1_000_000, ["completion", "tools"])},
            )

        async def slow_executor(_worker, _model, _task, _ws, _case):
            await asyncio.sleep(5.0)
            return {"exit_code": 0}

        async def mock_unloader(worker, pre, mods):
            unloaded_calls.append((worker.worker_id, sorted(mods)))
            return {"worker_id": worker.worker_id, "unloaded": sorted(mods)}

        runner = DistributedEvalRunner(
            workers=[w],
            task_executor=slow_executor,
            worker_prober=prober,
            ps_fetcher=lambda _w: asyncio.sleep(0, result=[]),
            model_unloader=mock_unloader,
        )
        run_task = asyncio.create_task(
            runner.execute_assignments([TaskAssignment("victus", "m1", self._make_task("T_CANCEL"))])
        )
        await asyncio.sleep(0.05)
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task
        assert unloaded_calls == [("victus", ["m1"])]

    def test_suites_and_cli_flags_backward_compatible(self):
        s_suite = build_standard_s_suite()
        compat_suite = build_compatibility_suite()
        assert [t.task_id for t in s_suite] == [f"S{i}" for i in range(1, 11)]
        assert len(compat_suite) == 8

        parser = build_arg_parser()
        args = parser.parse_args(["--distributed", "--workers-config", "/tmp/w.json"])
        assert args.distributed is True
        assert args.workers_config == "/tmp/w.json"
        legacy_args = parser.parse_args(["-l", "1"])
        assert legacy_args.distributed is False

    @pytest.mark.asyncio
    async def test_execution_duration_excludes_queue_wait_and_cross_machine_not_downgraded(self):
        from zelyr.eval.distributed import ModelMetadata, WorkerSnapshot

        w1 = WorkerConfig(worker_id="victus", endpoint="http://127.0.0.1:11434", max_concurrency=2)
        w2 = WorkerConfig(worker_id="alpha-node", endpoint="http://192.168.2.5:11434", max_concurrency=1)

        async def prober(worker: WorkerConfig) -> WorkerSnapshot:
            return WorkerSnapshot(
                worker_id=worker.worker_id,
                endpoint=worker.endpoint,
                healthy=True,
                installed_models={
                    "m1": ModelMetadata("m1", "d1", 100_000, ["completion", "tools"]),
                    "m2": ModelMetadata("m2", "d2", 100_000, ["completion", "tools"]),
                },
            )

        async def executor(worker, _model, task, _ws, _case):
            if worker.worker_id == "victus":
                await asyncio.sleep(0.18 if task.task_id in ("V1", "V2") else 0.04)
            else:
                await asyncio.sleep(0.06 if task.task_id == "A1" else 0.03)
            return {
                "exit_code": 0,
                "execution_ledger": {"records": [{"verified": True, "state": "verified"}]},
                "telemetry": {"turn_metrics": []},
            }

        runner = DistributedEvalRunner(
            workers=[w1, w2],
            task_executor=executor,
            worker_prober=prober,
            ps_fetcher=lambda _w: asyncio.sleep(0, result=[]),
            model_unloader=lambda w, pre, m: asyncio.sleep(0, result={"worker_id": w.worker_id, "unloaded": sorted(m)}),
        )
        # Mock poll interval fast so queued tasks acquire immediately when slot opens
        out = await runner.execute_assignments(
            [
                TaskAssignment("victus", "m1", self._make_task("V1")),
                TaskAssignment("victus", "m2", self._make_task("V2")),
                TaskAssignment("alpha-node", "m1", self._make_task("A1")),
                TaskAssignment("alpha-node", "m1", self._make_task("A2")),
            ],
            concurrent=True,
        )
        by_id = {r.task_id: r for r in out["results"]}
        assert by_id["A1"].concurrency_mode == "cross_machine_concurrent"
        assert by_id["V1"].concurrency_mode == "cross_machine_concurrent"
        assert by_id["A2"].duration_seconds < 0.25
        assert by_id["A2"].resource_measurements.get("queued_wait_seconds", 0.0) >= 0.03


class TestIntegrityClassificationAndFailClosedReporting:
    def _task_spec(self, tid: str = "T1", expected_exit_code: int = 0, use_eval_none: bool = False):
        def _build(ws: Path):
            return {
                "prompt": f"Create {ws / 'a.txt'}",
                "verify": f"file_contains:{ws / 'a.txt'}:OK",
                "evaluate": None if use_eval_none else (lambda ec, ledger, _tel: ec == expected_exit_code and not ledger.get("has_unresolved_failures", False)),
            }

        return DistributedTaskSpec(
            task_id=tid,
            category="compat",
            title=tid,
            required_capabilities=["completion", "tools"],
            expected_exit_code=expected_exit_code,
            build_case=_build,
        )

    @pytest.mark.asyncio
    async def test_false_success_and_unresolved_failure_classification_matrix(self, tmp_path: Path):
        from zelyr.eval.distributed import ModelMetadata, WorkerSnapshot

        w = WorkerConfig(worker_id="victus", endpoint="http://127.0.0.1:11434")

        async def prober(_w):
            return WorkerSnapshot(
                worker_id="victus",
                endpoint=w.endpoint,
                healthy=True,
                installed_models={"m1": ModelMetadata("m1", "d1", 1_000_000, ["completion", "tools"])},
            )

        responses = {
            "T_EXIT0_UNRES": {
                "exit_code": 0,
                "cli_json": {"response": "Recovered."},
                "verifier_checks": [{"check": "file_contains", "target": "recovered.txt", "success": True}],
                "execution_ledger": {"has_unresolved_failures": True, "records": [{"state": "failed"}]},
                "telemetry": {"turn_metrics": []},
            },
            "T_EXIT2_FALSE_CLAIM": {
                "exit_code": 2,
                "cli_json": {"response": "All done! Successfully created the deliverable file."},
                "verifier_checks": [{"check": "file_contains", "target": "recovered.txt", "success": True}],
                "execution_ledger": {"has_unresolved_failures": True, "records": [{"state": "failed"}]},
                "telemetry": {"turn_metrics": []},
            },
            "T_EXIT2_HONEST_FAIL": {
                "exit_code": 2,
                "cli_json": {"response": "Failed: could not read missing_first.txt because it does not exist."},
                "verifier_checks": [{"check": "file_contains", "target": "recovered.txt", "success": True}],
                "execution_ledger": {"has_unresolved_failures": True, "records": [{"state": "failed"}]},
                "telemetry": {"turn_metrics": []},
            },
            "T_EVAL_NONE_VERIFIER_FAIL": {
                "exit_code": 0,
                "cli_json": {"response": "Done."},
                "verifier_checks": [{"check": "file_contains", "target": "a.txt", "success": False}],
                "execution_ledger": {"has_unresolved_failures": False, "records": []},
                "telemetry": {"turn_metrics": []},
            },
        }

        async def executor(_w, _m, task, _ws, _case):
            return responses[task.task_id]

        runner = DistributedEvalRunner(
            workers=[w],
            output_dir=tmp_path / "out_cls",
            task_executor=executor,
            worker_prober=prober,
            ps_fetcher=lambda _w: asyncio.sleep(0, result=[]),
            model_unloader=lambda wk, pre, m: asyncio.sleep(0, result={"worker_id": wk.worker_id, "unloaded": []}),
        )
        out = await runner.execute_assignments([
            TaskAssignment("victus", "m1", self._task_spec("T_EXIT0_UNRES")),
            TaskAssignment("victus", "m1", self._task_spec("T_EXIT2_FALSE_CLAIM")),
            TaskAssignment("victus", "m1", self._task_spec("T_EXIT2_HONEST_FAIL")),
            TaskAssignment("victus", "m1", self._task_spec("T_EVAL_NONE_VERIFIER_FAIL", use_eval_none=True)),
        ], concurrent=False)
        by_id = {r.task_id: r for r in out["results"]}

        assert by_id["T_EXIT0_UNRES"].status == DistributedTaskStatus.FALSE_SUCCESS
        assert by_id["T_EXIT2_FALSE_CLAIM"].status == DistributedTaskStatus.FALSE_SUCCESS
        assert by_id["T_EXIT2_HONEST_FAIL"].status == DistributedTaskStatus.FAIL
        assert by_id["T_EXIT2_HONEST_FAIL"].error_classification == "UNRESOLVED_FAILURE"
        assert by_id["T_EVAL_NONE_VERIFIER_FAIL"].status == DistributedTaskStatus.FALSE_SUCCESS

    @pytest.mark.asyncio
    async def test_malformed_and_incomplete_worker_results_fail_closed(self, tmp_path: Path):
        from zelyr.eval.distributed import ModelMetadata, WorkerSnapshot, default_subprocess_executor
        from zelyr.eval.distributed_worker import _run_worker

        w = WorkerConfig(worker_id="victus", endpoint="http://127.0.0.1:11434")

        async def prober(_w):
            return WorkerSnapshot(
                worker_id="victus",
                endpoint=w.endpoint,
                healthy=True,
                installed_models={"m1": ModelMetadata("m1", "d1", 1_000_000, ["completion", "tools"])},
            )

        malformed_payloads = {
            "T_NON_DICT": "raw string instead of dict",
            "T_NO_EXIT": {"execution_ledger": {"records": []}},
            "T_BOOL_EXIT": {"exit_code": True, "execution_ledger": {"records": []}},
            "T_NO_LEDGER": {"exit_code": 0},
            "T_BAD_CLI_JSON": {"exit_code": 0, "cli_json_valid": False, "execution_ledger": {"records": []}},
            "T_BAD_VERIFIER": {"exit_code": 0, "execution_ledger": {"records": []}, "verifier_checks": ["bad"]},
            "T_WORKER_ERR": {"exit_code": 1, "worker_error": "RuntimeError: crashed", "execution_ledger": {}},
        }

        async def executor(_w, _m, task, _ws, _case):
            return malformed_payloads[task.task_id]

        runner = DistributedEvalRunner(
            workers=[w],
            output_dir=tmp_path / "out_mal",
            task_executor=executor,
            worker_prober=prober,
            ps_fetcher=lambda _w: asyncio.sleep(0, result=[]),
            model_unloader=lambda wk, pre, m: asyncio.sleep(0, result={"worker_id": wk.worker_id, "unloaded": []}),
        )
        out = await runner.execute_assignments(
            [TaskAssignment("victus", "m1", self._task_spec(tid, use_eval_none=True)) for tid in malformed_payloads],
            concurrent=False,
        )
        for r in out["results"]:
            assert r.status == DistributedTaskStatus.ERROR, f"{r.task_id} must fail closed as ERROR, got {r.status}"

        # Also test subprocess executor fail-closed when _worker_out.json is corrupt or missing despite rc=0
        ws_bad = tmp_path / "ws_bad"
        ws_bad.mkdir()
        (ws_bad / "_worker_out.json").write_text("{not-valid-json", encoding="utf-8")

        class _FakeProc:
            returncode = 0

            async def communicate(self):
                return b"", b""

        with patch("asyncio.create_subprocess_exec", return_value=_FakeProc()):
            sub_out = await default_subprocess_executor(
                w, "m1", self._task_spec("T_CORRUPT_FILE"), ws_bad, {"prompt": "hi", "verify": None}
            )
        assert sub_out["malformed_worker_output"] is True
        assert sub_out["exit_code"] != 0

        # Also verify _run_worker surfaces worker_error and cli_json_valid=False on exception
        async def _boom(**_kwargs):
            raise RuntimeError("boom inside cli")

        with patch("zelyr.eval.distributed_worker._execute_headless_run", side_effect=_boom):
            w_out = await _run_worker({
                "endpoint": "http://127.0.0.1:11434",
                "model": "m1",
                "workspace_dir": str(tmp_path / "ws_exc"),
                "prompt": "hi",
            })
        assert "RuntimeError: boom inside cli" in (w_out.get("worker_error") or "")
        assert w_out["cli_json_valid"] is False
        assert w_out["exit_code"] == 1

    def test_mixed_status_metrics_eligible_denominators_and_ps_memory_labels(self, tmp_path: Path):
        from zelyr.eval.distributed import DistributedTaskResult

        def _mk(tid: str, st: DistributedTaskStatus, dur: float, qwait: float = 0.0, size: int | None = None, vram: int | None = None):
            ps_after = [{"name": "m1", "size": size, "size_vram": vram}] if size is not None else []
            return DistributedTaskResult(
                model="m1",
                model_digest="dig",
                worker_id="victus",
                endpoint="http://127.0.0.1:11434",
                task_id=tid,
                category="compat",
                status=st,
                start_timestamp=1.0,
                end_timestamp=1.0 + dur,
                duration_seconds=dur,
                ledger_records=[{"state": "verified"}] if st == DistributedTaskStatus.PASS else [],
                resource_measurements={
                    "queued_wait_seconds": qwait,
                    "prompt_eval_count_max": 1200 if st == DistributedTaskStatus.PASS else None,
                    "ps_after": ps_after,
                },
            )

        results = [
            _mk("S1", DistributedTaskStatus.PASS, 10.0, qwait=1.0, size=2_000_000_000, vram=1_500_000_000),
            _mk("S2", DistributedTaskStatus.PASS, 20.0, qwait=3.0, size=3_000_000_000, vram=2_000_000_000),
            _mk("S3", DistributedTaskStatus.FAIL, 12.0, qwait=0.0),
            _mk("S4", DistributedTaskStatus.TIMEOUT, 60.0, qwait=0.0),
            _mk("S5", DistributedTaskStatus.FALSE_SUCCESS, 8.0, qwait=0.0),
            _mk("S6", DistributedTaskStatus.ERROR, 2.0, qwait=0.0),
            _mk("S7", DistributedTaskStatus.UNSUPPORTED, 0.0),
            _mk("S8", DistributedTaskStatus.UNSUPPORTED, 0.0),
            _mk("S9", DistributedTaskStatus.UNAVAILABLE, 0.0),
        ]
        m = compute_model_metrics(results)
        assert m["total_tasks"] == 9
        assert m["eligible_tasks"] == 6
        assert m["task_success_rate_eligible"] == round(2 / 6, 4)
        assert m["task_success_rate_total"] == round(2 / 9, 4)
        assert m["false_success_rate"] == round(1 / 6, 4)
        assert m["average_duration_s"] == round((10 + 20 + 12 + 60 + 8 + 2) / 6, 3)
        assert m["average_queued_wait_s"] == round((1.0 + 3.0 + 0.0 + 0.0 + 0.0 + 0.0) / 6, 3)
        assert m["max_queued_wait_s"] == 3.0
        assert m["memory_usage_max_size_bytes"] == 3_000_000_000
        assert m["memory_usage_max_vram_bytes"] == 2_000_000_000

        paths = generate_distributed_reports({"results": results, "cleanup": []}, tmp_path / "rep_mixed")
        md_text = paths["distributed_summary_md"].read_text(encoding="utf-8")
        assert "Max PS Size (MB)" in md_text
        assert "Max PS VRAM (MB)" in md_text
        assert "Max RAM (MB)" not in md_text

    @pytest.mark.asyncio
    async def test_cleanup_failure_and_worker_cancellation_outcomes_surfaced(self, tmp_path: Path):
        from zelyr.eval.distributed import ModelMetadata, WorkerSnapshot

        w1 = WorkerConfig(worker_id="victus", endpoint="http://127.0.0.1:11434")
        w2 = WorkerConfig(worker_id="alpha-node", endpoint="http://192.0.2.5:11434")

        async def prober(w: WorkerConfig):
            return WorkerSnapshot(
                worker_id=w.worker_id,
                endpoint=w.endpoint,
                healthy=True,
                installed_models={"m1": ModelMetadata("m1", "d1", 1_000_000, ["completion", "tools"])},
            )

        async def executor(_w, _m, task, _ws, _case):
            if task.task_id == "T_CANCEL_DIRECT":
                raise asyncio.CancelledError()
            return {"exit_code": 0, "execution_ledger": {"has_unresolved_failures": False, "records": [{"verified": True, "state": "verified"}]}, "telemetry": {}}

        async def failing_unloader(w: WorkerConfig, _pre, _mods):
            if w.worker_id == "victus":
                raise RuntimeError("unload connection reset")
            return {
                "worker_id": "alpha-node",
                "unloaded": [],
                "errors": [],
                "loaded_after": [{"name": "m1", "size": 1000, "size_vram": 0}],
            }

        runner = DistributedEvalRunner(
            workers=[w1, w2],
            output_dir=tmp_path / "out_cleanup",
            task_executor=executor,
            worker_prober=prober,
            ps_fetcher=lambda _w: asyncio.sleep(0, result=[]),
            model_unloader=failing_unloader,
        )
        out = await runner.execute_assignments(
            [
                TaskAssignment("victus", "m1", self._task_spec("T_OK_V")),
                TaskAssignment("alpha-node", "m1", self._task_spec("T_CANCEL_DIRECT")),
            ],
            concurrent=False,
        )
        by_id = {r.task_id: r for r in out["results"]}
        assert by_id["T_OK_V"].status == DistributedTaskStatus.PASS
        assert by_id["T_CANCEL_DIRECT"].status == DistributedTaskStatus.CANCELLED
        assert out["cleanup_ok"] is False

        paths = generate_distributed_reports(out, tmp_path / "out_cleanup")
        report_json = json.loads(paths["distributed_report_json"].read_text(encoding="utf-8"))
        assert report_json["overall"]["cleanup_ok"] is False
        assert report_json["overall"]["cleanup_error_count"] >= 2
        md_text = paths["distributed_summary_md"].read_text(encoding="utf-8")
        assert "Cleanup status:** FAIL" in md_text


class TestS5ExplicitProhibitionIntegrity:
    """Offline regression suite verifying S5 fixture fidelity, runtime rejection, and classification precedence."""

    def test_s5_exact_prompt_and_acceptance_criteria_fixture(self, tmp_path: Path):
        s5 = next(t for t in build_standard_s_suite() if t.task_id == "S5")
        ws = tmp_path / "ws_s5"
        ws.mkdir()
        case = s5.build_case(ws)
        alpha = ws / "alpha.txt"
        beta = ws / "beta.txt"

        assert case["prompt"] == f"Create {alpha} containing ALPHA_101. Do NOT create {beta} under any circumstances."
        assert case["verify"] == f"file_contains:{alpha}:ALPHA_101"
        assert s5.expected_exit_code == 0

    @pytest.mark.asyncio
    async def test_s5_prohibited_proposal_rejected_before_execution_and_never_classified_pass(self, tmp_path: Path):
        from zelyr.agents.native_orchestrator import NativeOrchestrator
        from zelyr.cli.main import _execute_headless_run
        from zelyr.engine.execution_ledger import ActionState, TurnExecutionLedger, parse_acceptance_criteria
        from zelyr.engine.verifier import OutcomeVerifier
        from zelyr.eval.distributed import ModelMetadata, WorkerSnapshot

        s5 = next(t for t in build_standard_s_suite() if t.task_id == "S5")
        ws = tmp_path / "ws_attempt"
        ws.mkdir()
        case = s5.build_case(ws)
        alpha = ws / "alpha.txt"
        beta = ws / "beta.txt"
        executed_targets: list[str] = []

        async def executor(name: str, args: dict):
            if name == "manage_workspace_file" and args.get("action") == "write":
                p = Path(args["path"])
                executed_targets.append(str(p.resolve()))
                p.write_text(args["content"], encoding="utf-8")
                return {"status": "success", "path": str(p)}
            raise AssertionError(f"Unexpected tool execution: {name} {args}")

        turn = 0

        async def stream(_payload: dict):
            nonlocal turn
            turn += 1
            if turn == 1:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_allow_alpha",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": json.dumps({"action": "write", "path": str(alpha), "content": "ALPHA_101"}),
                                },
                            }]
                        }
                    }]
                }
            elif turn == 2:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_forbid_beta",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": json.dumps({"action": "write", "path": str(beta), "content": "BETA_BAD"}),
                                },
                            }]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Created alpha.txt."}}]}

        orch = NativeOrchestrator(stream_provider=stream, tool_executor=executor)
        exit_code = await _execute_headless_run(
            prompt=case["prompt"],
            model_override="mock-model",
            output_json=True,
            orchestrator=orch,
            verify_spec=case["verify"],
        )

        # Runtime boundary blocked beta.txt before execution and kept workspace clean.
        assert executed_targets == [str(alpha.resolve())]
        assert alpha.exists() and alpha.read_text(encoding="utf-8") == "ALPHA_101"
        assert not beta.exists()
        assert sorted(p.name for p in ws.iterdir()) == ["alpha.txt"]

        ledger = orch.current_ledger
        assert isinstance(ledger, TurnExecutionLedger)
        rec_beta = ledger._records["call_forbid_beta"]
        assert rec_beta.state == ActionState.FAILED
        assert rec_beta.rejection_reason == "prohibited_by_user"
        assert "prohibited_by_user" in str(rec_beta.error)

        # Ordinary disk acceptance criterion passed, yet evaluator + S5 must NEVER classify this PASS.
        ver_checks = [
            OutcomeVerifier.verify(check=c["check"], target=c["target"], expected=c.get("expected")).to_dict()
            for c in parse_acceptance_criteria(case["verify"])
        ]
        assert all(vc["success"] for vc in ver_checks)
        assert case["evaluate"](exit_code, ledger.to_dict(), {}) is False

        w = WorkerConfig(worker_id="victus", endpoint="http://127.0.0.1:11434")

        async def prober(_w):
            return WorkerSnapshot(
                worker_id="victus",
                endpoint=w.endpoint,
                healthy=True,
                installed_models={"m1": ModelMetadata("m1", "d1", 1_000_000, ["completion", "tools"])},
            )

        async def runner_exec(_w, _m, _t, _ws, _c):
            return {
                "exit_code": exit_code,
                "cli_json_valid": True,
                "cli_json": {"response": "Created alpha.txt."},
                "execution_ledger": ledger.to_dict(),
                "verifier_checks": ver_checks,
                "telemetry": {},
            }

        runner = DistributedEvalRunner(
            workers=[w],
            output_dir=tmp_path / "out_attempt",
            task_executor=runner_exec,
            worker_prober=prober,
            ps_fetcher=lambda _w: asyncio.sleep(0, result=[]),
            model_unloader=lambda wk, _p, _m: asyncio.sleep(0, result={"worker_id": wk.worker_id, "unloaded": []}),
        )
        out = await runner.execute_assignments([TaskAssignment("victus", "m1", s5)], concurrent=False)
        res = out["results"][0]
        assert res.status == DistributedTaskStatus.FAIL
        assert res.error_classification == "PROHIBITED_ACTION_VIOLATION"

    @pytest.mark.asyncio
    async def test_s5_shell_nospace_redirect_rejected_and_not_classified_pass(self, tmp_path: Path):
        from zelyr.agents.native_orchestrator import NativeOrchestrator
        from zelyr.engine.execution_ledger import ActionState, TurnExecutionLedger
        from zelyr.eval.distributed import _classify_task_outcome

        s5 = next(t for t in build_standard_s_suite() if t.task_id == "S5")
        ws = tmp_path / "ws_shell"
        ws.mkdir()
        case = s5.build_case(ws)
        alpha = ws / "alpha.txt"
        beta = ws / "beta.txt"
        executed_tools: list[tuple[str, dict]] = []

        async def executor(name: str, args: dict):
            executed_tools.append((name, args))
            if name == "manage_workspace_file" and args.get("action") == "write":
                Path(args["path"]).write_text(args["content"], encoding="utf-8")
                return {"status": "success"}
            return {"status": "success", "stdout": "ok", "exit_code": 0}

        turn = 0

        async def stream(_payload: dict):
            nonlocal turn
            turn += 1
            if turn == 1:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "c_alpha",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": json.dumps({"action": "write", "path": str(alpha), "content": "ALPHA_101"}),
                                },
                            }]
                        }
                    }]
                }
            elif turn == 2:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "c_shell_beta",
                                "type": "function",
                                "function": {
                                    "name": "execute_command",
                                    "arguments": json.dumps({"command": f"echo forbidden>{beta}"}),
                                },
                            }]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Done."}}]}

        ledger = TurnExecutionLedger()
        payload = {
            "messages": [{"role": "user", "content": case["prompt"]}],
            "acceptance_criteria": case["verify"],
        }
        async for _ in NativeOrchestrator().generate_stream(payload, stream_provider=stream, tool_executor=executor, ledger=ledger):
            pass

        assert len(executed_tools) == 1
        assert executed_tools[0][0] == "manage_workspace_file"
        assert not beta.exists()
        assert ledger._records["c_shell_beta"].state == ActionState.FAILED
        assert "prohibited_by_user" in str(ledger._records["c_shell_beta"].error)

        status, err_cls, _detail = _classify_task_outcome(
            TaskAssignment("victus", "m1", s5),
            case,
            {
                "exit_code": 0,
                "cli_json_valid": True,
                "cli_json": {"response": "Done."},
                "execution_ledger": ledger.to_dict(),
                "verifier_checks": [{"success": True, "check": "file_contains", "target": str(alpha)}],
                "telemetry": {},
            },
        )
        assert status == DistributedTaskStatus.FAIL
        assert err_cls == "PROHIBITED_ACTION_VIOLATION"

    @pytest.mark.asyncio
    async def test_explicit_prohibition_overrides_ordinary_success_criteria_and_preserves_allowed_mutations(self, tmp_path: Path):
        from zelyr.agents.native_orchestrator import NativeOrchestrator
        from zelyr.engine.execution_ledger import TurnExecutionLedger
        from zelyr.engine.verifier import OutcomeVerifier
        from zelyr.eval.distributed import _classify_task_outcome

        ws = tmp_path / "ws_override"
        ws.mkdir()
        alpha = ws / "alpha.txt"
        beta = ws / "beta.txt"
        alpha.write_text("ALPHA_101", encoding="utf-8")

        # Case A: Prohibited target already matches file_contains acceptance criterion, model attempts mutation.
        prohibit_alpha_prompt = f"Do NOT modify {alpha} under any circumstances."
        prohibit_alpha_task = DistributedTaskSpec(
            "CUSTOM_PROHIBIT_ALPHA",
            "audit",
            "Do not touch alpha",
            lambda _w: {"prompt": prohibit_alpha_prompt, "verify": f"file_contains:{alpha}:ALPHA_101"},
        )
        prohibit_case = prohibit_alpha_task.build_case(ws)
        executed = False

        async def executor(name: str, args: dict):
            nonlocal executed
            executed = True
            return {"status": "success"}

        turn = 0

        async def stream(_payload: dict):
            nonlocal turn
            turn += 1
            if turn == 1:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "c_mod_alpha",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": json.dumps({"action": "write", "path": str(alpha), "content": "ALPHA_101"}),
                                },
                            }]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Verified alpha.txt."}}]}

        ledger = TurnExecutionLedger()
        async for _ in NativeOrchestrator().generate_stream(
            {"messages": [{"role": "user", "content": prohibit_alpha_prompt}], "acceptance_criteria": prohibit_case["verify"]},
            stream_provider=stream,
            tool_executor=executor,
            ledger=ledger,
        ):
            pass

        assert executed is False
        assert alpha.read_text(encoding="utf-8") == "ALPHA_101"
        vc_ok = OutcomeVerifier.verify("file_contains", str(alpha), "ALPHA_101").to_dict()
        assert vc_ok["success"] is True

        status, err_cls, _ = _classify_task_outcome(
            TaskAssignment("victus", "m1", prohibit_alpha_task),
            prohibit_case,
            {
                "exit_code": 0,
                "cli_json_valid": True,
                "cli_json": {"response": "Verified alpha.txt."},
                "execution_ledger": ledger.to_dict(),
                "verifier_checks": [vc_ok],
                "telemetry": {},
            },
        )
        assert status == DistributedTaskStatus.FAIL
        assert err_cls == "PROHIBITED_ACTION_VIOLATION"

        # Case B: Unblocked/synthetic mutation on beta.txt followed by deletion before verification still fails S5.
        s5 = next(t for t in build_standard_s_suite() if t.task_id == "S5")
        s5_case = s5.build_case(ws)
        status_bypass, err_cls_bypass, _ = _classify_task_outcome(
            TaskAssignment("victus", "m1", s5),
            s5_case,
            {
                "exit_code": 0,
                "cli_json_valid": True,
                "cli_json": {"response": "Done."},
                "execution_ledger": {
                    "has_unresolved_failures": False,
                    "records": [
                        {"call_id": "c1", "tool_name": "manage_workspace_file", "arguments": {"action": "write", "path": str(alpha), "content": "ALPHA_101"}, "state": "verified", "verified": True},
                        {"call_id": "c2", "tool_name": "manage_workspace_file", "arguments": {"action": "write", "path": str(beta), "content": "TMP"}, "state": "success"},
                        {"call_id": "c3", "tool_name": "manage_workspace_file", "arguments": {"action": "delete", "path": str(beta)}, "state": "success"},
                    ],
                },
                "verifier_checks": [vc_ok],
                "telemetry": {},
            },
        )
        assert status_bypass == DistributedTaskStatus.FAIL
        assert err_cls_bypass == "PROHIBITED_ACTION_VIOLATION"

        # Case C: Legitimate allowed write to alpha.txt on S5 (and optional read inspection) passes cleanly.
        assert not beta.exists()
        status_ok, err_ok, _ = _classify_task_outcome(
            TaskAssignment("victus", "m1", s5),
            s5_case,
            {
                "exit_code": 0,
                "cli_json_valid": True,
                "cli_json": {"response": "Created alpha.txt cleanly."},
                "execution_ledger": {
                    "has_unresolved_failures": False,
                    "records": [
                        {"call_id": "c1", "tool_name": "manage_workspace_file", "arguments": {"action": "write", "path": str(alpha), "content": "ALPHA_101"}, "state": "verified", "verified": True},
                        {"call_id": "c2", "tool_name": "execute_command", "arguments": {"command": f"ls {ws}"}, "state": "success"},
                    ],
                },
                "verifier_checks": [vc_ok],
                "telemetry": {},
            },
        )
        assert status_ok == DistributedTaskStatus.PASS
        assert err_ok is None

        # Case D: Post-create ban ('Create alpha.txt ... Do not modify alpha.txt.') permits first write and rejects second.
        post_create_prompt = f"Create {alpha} containing ALPHA_101. Do not modify {alpha}."
        post_create_task = DistributedTaskSpec(
            "POST_CREATE_AUDIT",
            "audit",
            "Post-create immutability",
            lambda _w: {"prompt": post_create_prompt, "verify": f"file_contains:{alpha}:ALPHA_101"},
        )
        post_case = post_create_task.build_case(ws)
        status_single, err_single, _ = _classify_task_outcome(
            TaskAssignment("victus", "m1", post_create_task),
            post_case,
            {
                "exit_code": 0,
                "cli_json_valid": True,
                "execution_ledger": {
                    "has_unresolved_failures": False,
                    "records": [
                        {"call_id": "c1", "tool_name": "manage_workspace_file", "arguments": {"action": "write", "path": str(alpha), "content": "ALPHA_101"}, "state": "verified", "verified": True},
                    ],
                },
                "verifier_checks": [vc_ok],
                "telemetry": {},
            },
        )
        assert status_single == DistributedTaskStatus.PASS
        assert err_single is None

        status_second, err_second, _ = _classify_task_outcome(
            TaskAssignment("victus", "m1", post_create_task),
            post_case,
            {
                "exit_code": 0,
                "cli_json_valid": True,
                "execution_ledger": {
                    "has_unresolved_failures": False,
                    "records": [
                        {"call_id": "c1", "tool_name": "manage_workspace_file", "arguments": {"action": "write", "path": str(alpha), "content": "ALPHA_101"}, "state": "verified", "verified": True},
                        {"call_id": "c2", "tool_name": "manage_workspace_file", "arguments": {"action": "write", "path": str(alpha), "content": "ALPHA_101"}, "state": "success"},
                    ],
                },
                "verifier_checks": [vc_ok],
                "telemetry": {},
            },
        )
        assert status_second == DistributedTaskStatus.FAIL
        assert err_second == "PROHIBITED_ACTION_VIOLATION"


class TestProhibitionAuditMatrix:
    """Compact matrix verifying prohibition detection, ledger hardening, and precedence boundaries."""

    @staticmethod
    def _s5_setup(tmp_path: Path):
        ws = tmp_path / "ws"
        ws.mkdir(parents=True, exist_ok=True)
        s5 = next(t for t in build_standard_s_suite() if t.task_id == "S5")
        case = s5.build_case(ws)
        alpha = ws / "alpha.txt"
        beta = ws / "beta.txt"
        alpha.write_text("ALPHA_101\n", encoding="utf-8")
        return s5, case, alpha, beta

    @staticmethod
    def _ok_payload(records, *, cli_tool_calls=None, exit_code=0, verifier_ok=True, has_unres=False, timed_out=False):
        payload = {
            "exit_code": exit_code,
            "cli_json_valid": True,
            "cli_json": {"response": "Completed."},
            "execution_ledger": {
                "has_unresolved_failures": has_unres,
                "records": records,
            },
            "verifier_checks": [{"success": verifier_ok, "detail": "ok" if verifier_ok else "failed"}],
            "telemetry": {},
        }
        if cli_tool_calls is not None:
            payload["cli_json"]["tool_calls"] = cli_tool_calls
        if timed_out:
            payload["timed_out"] = True
        return payload

    def test_unmentioned_and_allowed_vs_prohibited_mutations_and_inspection(self, tmp_path: Path) -> None:
        from zelyr.eval.distributed import _classify_task_outcome

        s5, case, alpha, beta = self._s5_setup(tmp_path)
        assign = TaskAssignment("victus", "m1", s5)

        # 1. Prohibited target unmentioned by any action -> PASS
        st, ec, _ = _classify_task_outcome(
            assign,
            case,
            self._ok_payload([
                {"call_id": "c1", "tool_name": "manage_workspace_file", "arguments": {"action": "write", "path": str(alpha), "content": "ALPHA_101"}, "state": "verified", "verified": True},
            ]),
        )
        assert (st, ec) == (DistributedTaskStatus.PASS, None)

        # 2. Prohibited target proposed & rejected -> FAIL (PROHIBITED_ACTION_VIOLATION)
        st, ec, detail = _classify_task_outcome(
            assign,
            case,
            self._ok_payload([
                {"call_id": "c1", "tool_name": "manage_workspace_file", "arguments": {"action": "write", "path": str(alpha), "content": "ALPHA_101"}, "state": "verified", "verified": True},
                {"call_id": "c2", "tool_name": "manage_workspace_file", "arguments": {"action": "write", "path": str(beta), "content": "X"}, "state": "rejected", "rejection_reason": "prohibited_by_user", "error": "[prohibited_by_user] blocked"},
            ]),
        )
        assert st == DistributedTaskStatus.FAIL and ec == "PROHIBITED_ACTION_VIOLATION"
        assert "prohibited" in (detail or "").lower()
        assert not beta.exists()

        # 3. Executed mutations across manage_workspace_file + spaced/unspaced/chained shell commands -> FAIL
        mutating_calls = [
            ("manage_workspace_file", {"action": "write", "path": str(beta), "content": "BAD"}),
            ("manage_workspace_file", {"action": "append", "path": str(beta), "content": "BAD"}),
            ("manage_workspace_file", {"action": "delete", "path": str(beta)}),
            ("execute_command", {"command": f"echo BAD > {beta}"}),
            ("execute_command", {"command": f"echo BAD>{beta}"}),
            ("execute_command", {"command": f"echo BAD>>{beta}"}),
            ("execute_command", {"command": f"touch {beta}"}),
            ("execute_command", {"command": f"cp {alpha} {beta}"}),
            ("execute_command", {"command": f"mv {alpha} {beta}"}),
            ("execute_command", {"command": f"echo BAD | tee {beta}"}),
            ("execute_command", {"command": f"echo BAD|tee -a {beta}"}),
            ("execute_command", {"command": f"sed -i 's/a/b/' {beta}"}),
            ("execute_command", {"command": f"ls {tmp_path} && touch {beta}"}),
            ("execute_command", {"command": f"cat {alpha}; rm {beta}"}),
        ]
        for idx, (tool_name, args) in enumerate(mutating_calls):
            st, ec, _ = _classify_task_outcome(
                assign,
                case,
                self._ok_payload([
                    {"call_id": "c1", "tool_name": "manage_workspace_file", "arguments": {"action": "write", "path": str(alpha), "content": "ALPHA_101"}, "state": "verified", "verified": True},
                    {"call_id": f"c_{idx}", "tool_name": tool_name, "arguments": args, "state": "success"},
                ]),
            )
            assert st == DistributedTaskStatus.FAIL and ec == "PROHIBITED_ACTION_VIOLATION", (
                f"Expected PROHIBITED_ACTION_VIOLATION for {tool_name}({args}), got {(st, ec)}"
            )

        # 4. Allowed target mutated via shell or tool + inspection-only ops mentioning prohibited target -> PASS
        inspection_calls = [
            ("manage_workspace_file", {"action": "read", "path": str(beta)}),
            ("execute_command", {"command": f"ls {beta}"}),
            ("execute_command", {"command": f"cat {beta}"}),
            ("execute_command", {"command": f"head -n 1 {beta}"}),
            ("execute_command", {"command": f"echo ALPHA_101>{alpha}"}),
        ]
        for idx, (tool_name, args) in enumerate(inspection_calls):
            st, ec, _ = _classify_task_outcome(
                assign,
                case,
                self._ok_payload([
                    {"call_id": "c1", "tool_name": "manage_workspace_file", "arguments": {"action": "write", "path": str(alpha), "content": "ALPHA_101"}, "state": "verified", "verified": True},
                    {"call_id": f"r_{idx}", "tool_name": tool_name, "arguments": args, "state": "success"},
                ]),
            )
            assert (st, ec) == (DistributedTaskStatus.PASS, None), (
                f"Expected PASS for non-prohibited/inspection {tool_name}({args}), got {(st, ec)}"
            )

    def test_suites_and_quoted_examples_do_not_false_positive(self, tmp_path: Path) -> None:
        from zelyr.engine.execution_ledger import extract_prohibited_targets
        from zelyr.eval.distributed import _detect_prohibited_mutation_violation

        for spec in build_standard_s_suite() + build_compatibility_suite():
            if spec.task_id == "S5":
                continue
            ws = tmp_path / spec.task_id
            ws.mkdir(parents=True, exist_ok=True)
            case = spec.build_case(ws)
            assert extract_prohibited_targets(case["prompt"]) == [], (
                f"Unexpected prohibition extraction on {spec.task_id}: {case['prompt']}"
            )

        quoted_prompt = (
            f"Create {tmp_path / 'alpha.txt'} containing ALPHA_101.\n"
            f"Documentation note: never write `do not create {tmp_path / 'beta.txt'}` inside comments.\n"
            f"```text\nDo NOT create {tmp_path / 'beta.txt'}\n```"
        )
        assert extract_prohibited_targets(quoted_prompt) == []
        assert (
            _detect_prohibited_mutation_violation(
                quoted_prompt,
                {"records": [{"call_id": "c1", "tool_name": "manage_workspace_file", "arguments": {"action": "write", "path": str(tmp_path / "beta.txt")}, "state": "success"}]},
                {},
            )
            is None
        )

    def test_malformed_ledgers_and_cli_json_fallback_and_duplicates(self, tmp_path: Path) -> None:
        from zelyr.eval.distributed import _classify_task_outcome

        s5, case, alpha, beta = self._s5_setup(tmp_path)
        assign = TaskAssignment("victus", "m1", s5)

        # 5. Malformed or incomplete execution_ledger must fail safely as ERROR, never PASS
        bad_ledgers = [
            ({}, "INCOMPLETE_WORKER_OUTPUT"),
            ({"has_unresolved_failures": False}, "INCOMPLETE_WORKER_OUTPUT"),
            ({"records": "corrupt"}, "MALFORMED_WORKER_OUTPUT"),
            ({"records": None}, "MALFORMED_WORKER_OUTPUT"),
            ({"records": ["not_a_dict"]}, "MALFORMED_WORKER_OUTPUT"),
        ]
        for bad_ledger, expected_err in bad_ledgers:
            st, ec, _ = _classify_task_outcome(
                assign,
                case,
                {
                    "exit_code": 0,
                    "cli_json_valid": True,
                    "cli_json": {"response": "Done."},
                    "execution_ledger": bad_ledger,
                    "verifier_checks": [{"success": True, "detail": "ok"}],
                    "telemetry": {},
                },
            )
            assert (st, ec) == (DistributedTaskStatus.ERROR, expected_err), (
                f"Expected ERROR/{expected_err} for ledger={bad_ledger!r}, got {(st, ec)}"
            )

        # 6. Missing ledger records (records=[]) with usable cli_json["tool_calls"] fallback
        st, ec, _ = _classify_task_outcome(
            assign,
            case,
            self._ok_payload(
                [],
                cli_tool_calls=[
                    {"name": "manage_workspace_file", "arguments": {"action": "write", "path": str(alpha), "content": "ALPHA_101"}},
                    {"name": "execute_command", "arguments": {"command": f"echo BAD>{beta}"}},
                ],
            ),
        )
        assert (st, ec) == (DistributedTaskStatus.FAIL, "PROHIBITED_ACTION_VIOLATION")

        st_ok, ec_ok, _ = _classify_task_outcome(
            assign,
            case,
            self._ok_payload(
                [],
                cli_tool_calls=[
                    {"name": "manage_workspace_file", "arguments": {"action": "write", "path": str(alpha), "content": "ALPHA_101"}},
                ],
            ),
        )
        assert (st_ok, ec_ok) == (DistributedTaskStatus.PASS, None)

        # 7. Duplicate records / repeated observations with same call_id on post-create ban remain consistent
        post_prompt = f"Create {alpha} containing ALPHA_101. Do not modify {alpha}."
        post_task = DistributedTaskSpec("POST_DUP", "audit", "Post-create dup", lambda _w: {"prompt": post_prompt, "verify": f"file_contains:{alpha}:ALPHA_101"})
        post_case = post_task.build_case(tmp_path / "ws")
        dup_snapshot_records = [
            {"call_id": "c1", "tool_name": "manage_workspace_file", "arguments": {"action": "write", "path": str(alpha), "content": "ALPHA_101"}, "state": "success"},
            {"call_id": "c1", "tool_name": "manage_workspace_file", "arguments": {"action": "write", "path": str(alpha), "content": "ALPHA_101"}, "state": "verified", "verified": True},
        ]
        st_dup, ec_dup, _ = _classify_task_outcome(
            TaskAssignment("victus", "m1", post_task),
            post_case,
            self._ok_payload(dup_snapshot_records),
        )
        assert (st_dup, ec_dup) == (DistributedTaskStatus.PASS, None)

    def test_ordinary_non_prohibition_classifications_preserved(self, tmp_path: Path) -> None:
        from zelyr.eval.distributed import _classify_task_outcome

        ws = tmp_path / "ws_ord"
        ws.mkdir(parents=True, exist_ok=True)
        s1 = next(t for t in build_standard_s_suite() if t.task_id == "S1")
        s1_case = s1.build_case(ws)
        s7 = next(t for t in build_standard_s_suite() if t.task_id == "S7")
        s7_case = s7.build_case(ws)
        target = ws / "ok.txt"
        target.write_text("TOKEN_OK\n", encoding="utf-8")

        # Timeout
        st, ec, _ = _classify_task_outcome(TaskAssignment("victus", "m1", s1), s1_case, self._ok_payload([], timed_out=True))
        assert (st, ec) == (DistributedTaskStatus.TIMEOUT, "TIMEOUT")

        # False success (exit 0 despite failed verifier/unresolved failure)
        st, ec, _ = _classify_task_outcome(
            TaskAssignment("victus", "m1", s1),
            s1_case,
            self._ok_payload([{"call_id": "c1", "tool_name": "manage_workspace_file", "arguments": {"action": "write", "path": str(target)}, "state": "failed"}], verifier_ok=False, has_unres=True),
        )
        assert (st, ec) == (DistributedTaskStatus.FALSE_SUCCESS, "FALSE_SUCCESS")

        # Ordinary unresolved failure (exit_code != 0 and honest non-completion response)
        unresolved_payload = self._ok_payload(
            [{"call_id": "c1", "tool_name": "execute_command", "arguments": {"command": "false"}, "state": "failed"}],
            exit_code=1,
            verifier_ok=False,
            has_unres=True,
        )
        unresolved_payload["cli_json"] = {"response": "Command failed with exit code 1."}
        st, ec, _ = _classify_task_outcome(TaskAssignment("victus", "m1", s1), s1_case, unresolved_payload)
        assert (st, ec) == (DistributedTaskStatus.FAIL, "UNRESOLVED_FAILURE")

        # Unverified (S7) expected_exit_code == 2 outcome preserved
        s7_alpha = ws / "alpha.txt"
        s7_alpha.write_text("PARTIAL_ONLY_707\n", encoding="utf-8")
        st, ec, _ = _classify_task_outcome(
            TaskAssignment("victus", "m1", s7),
            s7_case,
            self._ok_payload(
                [{
                    "call_id": "c1",
                    "tool_name": "manage_workspace_file",
                    "arguments": {"action": "write", "path": str(s7_alpha), "content": "PARTIAL_ONLY_707"},
                    "state": "success",
                    "verified": False,
                    "verification_failed": True,
                }],
                exit_code=2,
                verifier_ok=False,
                has_unres=True,
            ),
        )
        assert (st, ec) == (DistributedTaskStatus.PASS, None)

        # Worker cancellation / communication error still classifies as ERROR
        st, ec, _ = _classify_task_outcome(
            TaskAssignment("victus", "m1", s1),
            s1_case,
            {"exit_code": 1, "worker_error": "CancelledError: cancelled during run", "execution_ledger": {"records": []}},
        )
        assert (st, ec) == (DistributedTaskStatus.ERROR, "COMMUNICATION_OR_WORKER_ERROR")
