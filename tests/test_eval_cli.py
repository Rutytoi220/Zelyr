"""Test suite for Phase 3: Local-Model Adapter, Telemetry & Benchmark CLI."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from zelyr.eval.cli import get_all_available_tasks, run_benchmark_cli
from zelyr.eval.harness import EvalHarness, LiveOllamaStreamAdapter, check_ollama_status_async
from zelyr.eval.metrics import EvalStatus, TaskResult, export_json_report, format_summary_table
from zelyr.eval.task import EvalContext, EvalTask
from zelyr.eval.tasks.deterministic_p1 import build_task_tool_dispatch
from zelyr.eval.tasks.model_tasks_p3 import (
    build_task_live_metric_formatting,
    build_task_live_workspace_diagnostic,
    check_ollama_status,
    get_model_tasks_p3,
)


class TestLiveAdapterAndTelemetry(unittest.IsolatedAsyncioTestCase):
    """Test LiveOllamaStreamAdapter token accounting and latency profiling."""

    async def test_live_adapter_stream_and_token_extraction(self) -> None:
        """Verify LiveOllamaStreamAdapter decodes stream lines, accumulates measured_tokens, and records latencies."""
        adapter = LiveOllamaStreamAdapter(base_url="http://mock-ollama:11434", model="qwen3:8b")

        mock_lines = [
            json.dumps({"message": {"content": "Thinking about the task..."}}),
            json.dumps({"message": {"content": " Here is the solution."}}),
            json.dumps({
                "done": True,
                "prompt_eval_count": 42,
                "eval_count": 18,
            }),
        ]

        class MockStreamResponse:
            status_code = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def aiter_lines(self):
                for line in mock_lines:
                    yield line

        class MockAsyncClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def stream(self, *args, **kwargs):
                return MockStreamResponse()

        with patch("httpx.AsyncClient", return_value=MockAsyncClient()):
            chunks = []
            async for chunk in adapter.stream_chat({"messages": [{"role": "user", "content": "hi"}]}):
                chunks.append(chunk)

            self.assertEqual(len(chunks), 2)
            self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], "Thinking about the task...")
            self.assertEqual(chunks[1]["choices"][0]["delta"]["content"], " Here is the solution.")

            # Strict measured_tokens verification: 42 + 18 = 60
            self.assertEqual(adapter.measured_tokens, 60)
            self.assertEqual(len(adapter.model_latencies), 1)
            self.assertGreater(adapter.model_latencies[0], 0.0)

    async def test_live_adapter_http_error_handling(self) -> None:
        """Verify LiveOllamaStreamAdapter cleanly yields error dict when HTTP status != 200."""
        adapter = LiveOllamaStreamAdapter(base_url="http://mock-ollama:11434", model="qwen3:8b")

        class MockErrorStreamResponse:
            status_code = 500

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def aread(self):
                return b"Internal Server Error"

        class MockAsyncClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def stream(self, *args, **kwargs):
                return MockErrorStreamResponse()

        with patch("httpx.AsyncClient", return_value=MockAsyncClient()):
            chunks = []
            async for chunk in adapter.stream_chat({"messages": [{"role": "user", "content": "hi"}]}):
                chunks.append(chunk)

            self.assertEqual(len(chunks), 1)
            self.assertIn("_api_error", chunks[0])
            self.assertIn("500", chunks[0]["_api_error"])

    async def test_harness_live_model_flow_with_mocked_ollama(self) -> None:
        """Verify full live model orchestration pipeline with token measurement and tool execution."""
        task = build_task_live_workspace_diagnostic()
        harness = EvalHarness()

        turn_idx = 0

        async def mock_stream_chat(self, payload):
            nonlocal turn_idx
            turn_idx += 1
            self.measured_tokens += 125
            self.model_latencies.append(0.05)
            if turn_idx == 1:
                # Yield tool call
                yield {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "id": "call_write_1",
                                        "type": "function",
                                        "function": {
                                            "name": "write_file",
                                            "arguments": json.dumps({
                                                "path": "diagnosis.json",
                                                "content": json.dumps({
                                                    "component": "core_worker",
                                                    "error_cause": "OutOfMemoryError: killed thread",
                                                }),
                                            }),
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            else:
                yield {
                    "choices": [
                        {
                            "delta": {
                                "content": "Diagnosis report created successfully in diagnosis.json."
                            }
                        }
                    ]
                }

        with patch("zelyr.eval.harness.check_ollama_status_async", return_value=(True, "Ollama ready")), \
             patch.object(LiveOllamaStreamAdapter, "stream_chat", mock_stream_chat):

            res = await harness.run_task(task, use_live_model=True)

            self.assertEqual(res.status, EvalStatus.PASS)
            self.assertEqual(res.step_count, 2)
            self.assertEqual(res.tool_calls, ["write_file"])
            self.assertEqual(res.measured_tokens, 250)
            self.assertIn("latency_breakdown", res.artifacts)
            breakdown = res.artifacts["latency_breakdown"]
            self.assertIn("tool_latencies", breakdown)
            self.assertGreater(len(breakdown["tool_latencies"]), 0)



class TestModelTasksAndOfflineInvariants(unittest.IsolatedAsyncioTestCase):
    """Test Level 2 model tasks, status probes, and offline fallback."""

    def test_model_tasks_definitions(self) -> None:
        """Verify model task structure, tools, and metadata."""
        tasks = get_model_tasks_p3(model="qwen3:8b", base_url="http://127.0.0.1:11434")
        self.assertEqual(len(tasks), 3)

        task_ids = [t.task_id for t in tasks]
        self.assertIn("task_live_workspace_diagnostic", task_ids)
        self.assertIn("task_live_metric_formatting", task_ids)
        self.assertIn("task_live_web_research_synthesis", task_ids)

        for t in tasks:
            self.assertEqual(t.level, 2)
            self.assertTrue(t.metadata.get("use_live_model"))
            self.assertTrue(set(t.allowed_tools).issubset({"read_file", "write_file", "web_search", "fetch_web_page"}))
            self.assertIn("write_file", t.custom_tools)
            self.assertIsNotNone(t.tool_schemas)

    async def test_workspace_diagnostic_validator(self) -> None:
        """Verify diagnostic task validator checks component and error correctly."""
        task = build_task_live_workspace_diagnostic()
        with tempfile.TemporaryDirectory() as tmp_dir:
            ws = Path(tmp_dir)
            ctx = EvalContext(task=task, workspace_dir=ws, db_path=ws / "test.db")

            # Missing file -> False
            ok, msg = await task.validator_fn(ctx)
            self.assertFalse(ok)

            # Invalid JSON -> False
            (ws / "diagnosis.json").write_text("invalid json", encoding="utf-8")
            ok, msg = await task.validator_fn(ctx)
            self.assertFalse(ok)

            # Wrong component -> False
            (ws / "diagnosis.json").write_text(json.dumps({"component": "web_server", "error_cause": "OOM"}), encoding="utf-8")
            ok, msg = await task.validator_fn(ctx)
            self.assertFalse(ok)

            # Valid report -> True
            (ws / "diagnosis.json").write_text(
                json.dumps({"component": "core_worker", "error_cause": "OutOfMemoryError: killed thread"}),
                encoding="utf-8",
            )
            ok, msg = await task.validator_fn(ctx)
            self.assertTrue(ok)

    async def test_metric_formatting_validator(self) -> None:
        """Verify metric formatting task validator checks numerical values correctly."""
        task = build_task_live_metric_formatting()
        with tempfile.TemporaryDirectory() as tmp_dir:
            ws = Path(tmp_dir)
            ctx = EvalContext(task=task, workspace_dir=ws, db_path=ws / "test.db")

            # Missing file -> False
            ok, msg = await task.validator_fn(ctx)
            self.assertFalse(ok)

            # Valid summary -> True (avg 200, 1 error)
            (ws / "summary.json").write_text(
                json.dumps({"avg_latency": 200.0, "total_errors": 1}),
                encoding="utf-8",
            )
            ok, msg = await task.validator_fn(ctx)
            self.assertTrue(ok)

            # Inaccurate average -> False
            (ws / "summary.json").write_text(
                json.dumps({"avg_latency": 150.0, "total_errors": 1}),
                encoding="utf-8",
            )
            ok, msg = await task.validator_fn(ctx)
            self.assertFalse(ok)

    async def test_offline_unverified_fallback(self) -> None:
        """Verify that running a live task against an unreachable daemon returns UNVERIFIED gracefully."""
        harness = EvalHarness()
        task = build_task_live_workspace_diagnostic(base_url="http://127.0.0.1:9999")

        # Must return UNVERIFIED without unhandled exceptions
        res = await harness.run_task(task, use_live_model=True)
        self.assertEqual(res.status, EvalStatus.UNVERIFIED)
        self.assertIn("Ollama unavailable", res.message)
        self.assertIsNone(res.measured_tokens)
        self.assertEqual(res.step_count, 0)

    def test_check_ollama_status_offline(self) -> None:
        """Verify synchronous probe returns False when unreachable."""
        ok, reason = check_ollama_status(url="http://127.0.0.1:9999", timeout=0.1)
        self.assertFalse(ok)
        self.assertIn("Could not connect", reason)


class TestTelemetryAndLatencyBreakdown(unittest.IsolatedAsyncioTestCase):
    """Test telemetry breakdown between model latency and tool latency."""

    async def test_mock_stream_measured_vs_estimated_invariants(self) -> None:
        """Verify mock streams strictly maintain measured_tokens=None, estimated_tokens>0, and latency breakdown."""
        harness = EvalHarness()
        task = build_task_tool_dispatch()

        res = await harness.run_task(task)
        self.assertEqual(res.status, EvalStatus.PASS)
        # Invariant: mock streams MUST NOT have measured_tokens
        self.assertIsNone(res.measured_tokens)
        self.assertGreater(res.estimated_tokens, 0)

        # Telemetry artifact verification
        breakdown = res.artifacts.get("latency_breakdown")
        self.assertIsNotNone(breakdown)
        self.assertIn("model_latencies", breakdown)
        self.assertIn("tool_latencies", breakdown)
        self.assertIn("total_model_latency", breakdown)
        self.assertIn("total_tool_latency", breakdown)
        self.assertGreaterEqual(len(breakdown["tool_latencies"]), 1)


class TestBenchmarkCLI(unittest.TestCase):
    """Test CLI argument parsing, list mode, execution filtering, and exit codes."""

    def test_list_tasks(self) -> None:
        """Verify --list-tasks returns 0."""
        code = run_benchmark_cli(["--list-tasks"])
        self.assertEqual(code, 0)

    def test_unknown_level_returns_code_2(self) -> None:
        """Verify invalid --level returns error code 2."""
        code = run_benchmark_cli(["--level", "99"])
        self.assertEqual(code, 2)

    def test_unknown_task_returns_code_1(self) -> None:
        """Verify non-existent task returns error code 1."""
        code = run_benchmark_cli(["--task", "non_existent_task_12345"])
        self.assertEqual(code, 1)

    def test_run_single_task_and_export_json(self) -> None:
        """Verify running a single deterministic task and exporting JSON report."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "bench_report.json"
            code = run_benchmark_cli([
                "--task", "task_tool_dispatch",
                "--json", str(json_path),
                "--quiet",
            ])
            self.assertEqual(code, 0)
            self.assertTrue(json_path.is_file())

            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(data["summary"]["total"], 1)
            self.assertEqual(data["summary"]["passed"], 1)
            result_item = data["results"][0]
            self.assertEqual(result_item["task_id"], "task_tool_dispatch")
            self.assertIsNone(result_item["measured_tokens"])
            self.assertGreater(result_item["estimated_tokens"], 0)

    def test_level_2_offline_exit_code_zero(self) -> None:
        """Verify running Level 2 task when offline produces UNVERIFIED and does NOT fail CI."""
        code = run_benchmark_cli([
            "--task", "task_live_workspace_diagnostic",
            "--base-url", "http://127.0.0.1:9999",
            "--quiet",
        ])
        # UNVERIFIED must exit with 0 to prevent breaking CI
        self.assertEqual(code, 0)

    def test_live_timeout_cli_argument(self) -> None:
        """Verify --live-timeout parses float and defaults to 60.0."""
        from zelyr.eval.cli import build_arg_parser
        parser = build_arg_parser()
        args_default = parser.parse_args([])
        self.assertEqual(args_default.live_timeout, 60.0)

        args_custom = parser.parse_args(["--live-timeout", "120.5"])
        self.assertEqual(args_custom.live_timeout, 120.5)


class TestConfigurableLiveTimeouts(unittest.IsolatedAsyncioTestCase):
    """Test configurable live timeouts vs fast offline deterministic timeouts."""

    async def test_offline_task_retains_timeout_seconds_when_live_timeout_provided(self) -> None:
        """Verify that offline tasks retain task.timeout_seconds even if live_timeout is provided."""
        harness = EvalHarness()
        captured_timeout = None

        async def mock_wait_for(coro, timeout):
            nonlocal captured_timeout
            captured_timeout = timeout
            try:
                coro.close()
            except Exception:
                pass
            raise asyncio.TimeoutError()

        task = EvalTask(
            task_id="test_offline_timeout_fast",
            description="Offline fast timeout test",
            level=1,
            timeout_seconds=0.2,
        )

        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            res = await harness.run_task(task, use_live_model=False, live_timeout=60.0)
            self.assertEqual(res.status, EvalStatus.TIMEOUT)
            self.assertEqual(captured_timeout, 0.2)
            self.assertIn("timed out after 0.2s", res.message)

    async def test_live_task_applies_live_timeout_override(self) -> None:
        """Verify that live tasks apply live_timeout (overriding task base timeout)."""
        harness = EvalHarness()

        async def moderately_slow_stream(self, payload: dict):
            await asyncio.sleep(0.05)
            yield {"choices": [{"delta": {"content": "completed successfully"}}]}

        task = EvalTask(
            task_id="test_live_timeout_override",
            description="Live timeout override test",
            level=2,
            timeout_seconds=0.01,  # Short base timeout that would fail if not overridden
            validator_fn=lambda ctx: (True, "Live task succeeded"),
        )

        with patch("zelyr.eval.harness.check_ollama_status_async", return_value=(True, "Ready")):
            with patch.object(LiveOllamaStreamAdapter, "stream_chat", moderately_slow_stream):
                # When live_timeout is 0.5s, 0.05s execution completes within timeout
                res = await harness.run_task(task, use_live_model=True, live_timeout=0.5)
                self.assertEqual(res.status, EvalStatus.PASS)
                self.assertIn("Live task succeeded", res.message)

    async def test_live_task_default_timeout_is_60s(self) -> None:
        """Verify that live task defaults to 60.0s when live_timeout is None."""
        harness = EvalHarness()
        captured_timeout = None

        async def mock_wait_for(coro, timeout):
            nonlocal captured_timeout
            captured_timeout = timeout
            try:
                coro.close()
            except Exception:
                pass
            raise asyncio.TimeoutError()

        task = EvalTask(
            task_id="test_live_timeout_default",
            description="Live timeout default test",
            level=2,
            timeout_seconds=10.0,
        )

        with patch("zelyr.eval.harness.check_ollama_status_async", return_value=(True, "Ready")):
            with patch("asyncio.wait_for", side_effect=mock_wait_for):
                res = await harness.run_task(task, use_live_model=True, live_timeout=None)
                self.assertEqual(res.status, EvalStatus.TIMEOUT)
                self.assertEqual(captured_timeout, 60.0)
                self.assertIn("timed out after 60.0s", res.message)


if __name__ == "__main__":
    unittest.main()
