"""Tests for Level 4 Benchmark Evaluation Tasks & Subsystem Invariants."""

from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import time
import unittest

from zelyr.eval.harness import EvalHarness
from zelyr.eval.metrics import EvalStatus
from zelyr.eval.scorecard import generate_benchmark_scorecard, render_markdown_scorecard
from zelyr.eval.task import EvalContext
from zelyr.eval.tasks.model_tasks_p4 import (
    build_task_autonomous_runtime_bug_patching,
    build_task_complex_data_pipeline_etl,
    build_task_corrupted_state_recovery_and_rollback,
    build_task_dynamic_tool_sandbox_containment,
    build_task_multi_agent_blackboard_coordination,
    build_task_proactive_memory_decay_audit,
    get_model_tasks_p4,
)
from zelyr.eval.types import EvalLevel


class TestEvalLevel4Suite(unittest.TestCase):
    """Test suite for Level 4 benchmark tasks and failure containment."""

    def setUp(self) -> None:
        self.harness = EvalHarness()

    def test_level_4_task_discovery(self) -> None:
        tasks = get_model_tasks_p4()
        self.assertEqual(len(tasks), 6)
        task_ids = [t.task_id for t in tasks]
        self.assertIn("task_autonomous_runtime_bug_patching", task_ids)
        self.assertIn("task_multi_agent_blackboard_coordination", task_ids)
        self.assertIn("task_corrupted_state_recovery_and_rollback", task_ids)
        self.assertIn("task_complex_data_pipeline_etl", task_ids)
        self.assertIn("task_dynamic_tool_sandbox_containment", task_ids)
        self.assertIn("task_proactive_memory_decay_audit", task_ids)

        for t in tasks:
            self.assertEqual(t.level, 4)
            self.assertEqual(t.metadata.get("benchmark_level"), 4)

    def test_eval_level_enum(self) -> None:
        self.assertEqual(EvalLevel.LEVEL_1.value, "level_1")
        self.assertEqual(EvalLevel.LEVEL_2.value, "level_2")
        self.assertEqual(EvalLevel.LEVEL_3.value, "level_3")
        self.assertEqual(EvalLevel.LEVEL_4.value, "level_4")
        self.assertEqual(EvalLevel.ALL.value, "all")

    def test_execution_of_all_level_4_tasks(self) -> None:
        tasks = get_model_tasks_p4()

        async def _run_all():
            results = []
            for t in tasks:
                res = await self.harness.run_task(t)
                results.append(res)
            return results

        t0 = time.perf_counter()
        results = asyncio.run(_run_all())
        elapsed = time.perf_counter() - t0

        self.assertLess(elapsed, 3.0, f"Level 4 benchmark suite took too long: {elapsed:.2f}s")
        for res in results:
            self.assertEqual(
                res.status,
                EvalStatus.PASS,
                f"Task {res.task_id} did not pass: {res.message}",
            )

    def test_negative_validator_service_patching(self) -> None:
        task = build_task_autonomous_runtime_bug_patching()
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            # Setup broken service
            task.setup_fn(ws, None)
            ctx = EvalContext(task=task, workspace_dir=ws, db_path=ws / "mem.db")

            # Validator before repair should fail
            passed, msg = task.validator_fn(ctx)
            self.assertFalse(passed)
            self.assertIn("test_service.py failed", msg)

    def test_negative_validator_orphan_files_on_rollback(self) -> None:
        task = build_task_corrupted_state_recovery_and_rollback()
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            task.setup_fn(ws, None)
            ctx = EvalContext(task=task, workspace_dir=ws, db_path=ws / "mem.db")

            # Intentionally leave an orphan .tmp file
            orphan = ws / "tx_failed_99.tmp"
            orphan.write_text("corrupted staged data", encoding="utf-8")

            passed, msg = task.validator_fn(ctx)
            self.assertFalse(passed)
            self.assertIn("orphan temporary transaction files", msg)

    def test_negative_validator_etl_anomaly_count(self) -> None:
        task = build_task_complex_data_pipeline_etl()
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            task.setup_fn(ws, None)
            ctx = EvalContext(task=task, workspace_dir=ws, db_path=ws / "mem.db")

            # Validator fails if output.sqlite is missing
            passed, msg = task.validator_fn(ctx)
            self.assertFalse(passed)
            self.assertIn("output.sqlite was not generated", msg)

    def test_scorecard_generation_level_4(self) -> None:
        tasks = get_model_tasks_p4()

        async def _run_all():
            results = []
            for t in tasks:
                res = await self.harness.run_task(t)
                results.append(res)
            return results

        results = asyncio.run(_run_all())
        scorecard = generate_benchmark_scorecard(results, suite_name="Level 4 Test Suite")

        self.assertEqual(scorecard.total_tasks, 6)
        self.assertEqual(scorecard.passed_count, 6)
        self.assertEqual(scorecard.pass_rate, 1.0)
        self.assertIn(4, scorecard.level_breakdown)
        self.assertEqual(scorecard.level_breakdown[4]["passed"], 6)

        md = render_markdown_scorecard(scorecard)
        self.assertIn("Level 4 Multi-Turn Self-Repair & System Resilience", md)
        self.assertIn("100.0%", md)
