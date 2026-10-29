"""Tests for Level 5 Benchmark Evaluation Tasks & Subsystem Invariants."""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
import tempfile
import time
import unittest

from zelyr.eval.cli import get_all_available_tasks, run_benchmark_cli
from zelyr.eval.harness import EvalHarness
from zelyr.eval.metrics import EvalStatus
from zelyr.eval.scorecard import generate_benchmark_scorecard, render_markdown_scorecard
from zelyr.eval.task import EvalContext
from zelyr.eval.tasks.model_tasks_p5 import (
    build_task_adaptive_context_budget_compaction,
    build_task_autonomous_routine_lifecycle,
    build_task_autonomous_tool_crafting_and_hotload,
    build_task_hierarchical_subagent_delegation,
    build_task_mcp_server_gateway_federation,
    build_task_multi_turn_session_continuity,
    get_model_tasks_p5,
)
from zelyr.eval.types import EvalLevel


class TestEvalLevel5Suite(unittest.TestCase):
    """Test suite for Level 5 benchmark tasks, containment, and metrics."""

    def setUp(self) -> None:
        self.harness = EvalHarness()

    def test_level_5_task_discovery(self) -> None:
        tasks = get_model_tasks_p5()
        self.assertEqual(len(tasks), 6)
        task_ids = [t.task_id for t in tasks]
        self.assertIn("task_mcp_server_gateway_federation", task_ids)
        self.assertIn("task_autonomous_routine_lifecycle", task_ids)
        self.assertIn("task_adaptive_context_budget_compaction", task_ids)
        self.assertIn("task_hierarchical_subagent_delegation", task_ids)
        self.assertIn("task_autonomous_tool_crafting_and_hotload", task_ids)
        self.assertIn("task_multi_turn_session_continuity", task_ids)

        for t in tasks:
            self.assertEqual(t.level, 5)
            self.assertEqual(t.metadata.get("benchmark_level"), 5)

        # Global task enumeration across all levels
        all_tasks = get_all_available_tasks()
        self.assertEqual(len(all_tasks), 30)
        l5_tasks = [t for t in all_tasks if t.level == 5]
        self.assertEqual(len(l5_tasks), 6)

    def test_eval_level_enum(self) -> None:
        self.assertEqual(EvalLevel.LEVEL_1.value, "level_1")
        self.assertEqual(EvalLevel.LEVEL_2.value, "level_2")
        self.assertEqual(EvalLevel.LEVEL_3.value, "level_3")
        self.assertEqual(EvalLevel.LEVEL_4.value, "level_4")
        self.assertEqual(EvalLevel.LEVEL_5.value, "level_5")
        self.assertEqual(EvalLevel.ALL.value, "all")

    def test_execution_of_all_level_5_tasks(self) -> None:
        tasks = get_model_tasks_p5()

        async def _run_all():
            results = []
            for t in tasks:
                res = await self.harness.run_task(t)
                results.append(res)
            return results

        t0 = time.perf_counter()
        results = asyncio.run(_run_all())
        elapsed = time.perf_counter() - t0

        self.assertLess(elapsed, 1.5, f"Level 5 benchmark suite exceeded latency boundary: {elapsed:.2f}s")
        for res in results:
            self.assertEqual(
                res.status,
                EvalStatus.PASS,
                f"Task {res.task_id} did not pass: {res.message}",
            )

    def test_negative_validator_mcp_handshake_missing(self) -> None:
        task = build_task_mcp_server_gateway_federation()
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            task.setup_fn(ws, None)
            ctx = EvalContext(task=task, workspace_dir=ws, db_path=ws / "mem.db")

            passed, msg = task.validator_fn(ctx)
            self.assertFalse(passed)
            self.assertIn("mcp_handshake.json was not created", msg)

    def test_negative_validator_routine_lifecycle_missing_report(self) -> None:
        task = build_task_autonomous_routine_lifecycle()
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            task.setup_fn(ws, None)
            ctx = EvalContext(task=task, workspace_dir=ws, db_path=ws / "mem.db")

            val_res = task.validator_fn(ctx)
            if inspect.isawaitable(val_res):
                passed, msg = asyncio.run(val_res)
            else:
                passed, msg = val_res

            self.assertFalse(passed)
            self.assertIn("routine_lifecycle_report.json was not created", msg)

    def test_negative_validator_context_compaction_missing_output(self) -> None:
        task = build_task_adaptive_context_budget_compaction()
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            task.setup_fn(ws, None)
            ctx = EvalContext(task=task, workspace_dir=ws, db_path=ws / "mem.db")

            passed, msg = task.validator_fn(ctx)
            self.assertFalse(passed)
            self.assertIn("compacted_session.json was not generated", msg)

    def test_negative_validator_hierarchical_subagent_missing_blackboard(self) -> None:
        task = build_task_hierarchical_subagent_delegation()
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            task.setup_fn(ws, None)
            ctx = EvalContext(task=task, workspace_dir=ws, db_path=ws / "mem.db")

            passed, msg = task.validator_fn(ctx)
            self.assertFalse(passed)
            self.assertIn("blackboard_state.json was not created", msg)

    def test_negative_validator_tool_crafting_missing_output(self) -> None:
        task = build_task_autonomous_tool_crafting_and_hotload()
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            task.setup_fn(ws, None)
            ctx = EvalContext(task=task, workspace_dir=ws, db_path=ws / "mem.db")

            passed, msg = task.validator_fn(ctx)
            self.assertFalse(passed)
            self.assertIn("crafted_tool_output.json was not created", msg)

    def test_negative_validator_session_continuity_missing_audit(self) -> None:
        task = build_task_multi_turn_session_continuity()
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            task.setup_fn(ws, None)
            ctx = EvalContext(task=task, workspace_dir=ws, db_path=ws / "mem.db")

            passed, msg = task.validator_fn(ctx)
            self.assertFalse(passed)
            self.assertIn("session_continuity_audit.json was not created", msg)

    def test_scorecard_generation_level_5(self) -> None:
        tasks = get_model_tasks_p5()

        async def _run_all():
            results = []
            for t in tasks:
                res = await self.harness.run_task(t)
                results.append(res)
            return results

        results = asyncio.run(_run_all())
        scorecard = generate_benchmark_scorecard(results, suite_name="Level 5 Test Suite")

        self.assertEqual(scorecard.total_tasks, 6)
        self.assertEqual(scorecard.passed_count, 6)
        self.assertEqual(scorecard.pass_rate, 1.0)
        self.assertIn(5, scorecard.level_breakdown)
        self.assertEqual(scorecard.level_breakdown[5]["passed"], 6)

        md = render_markdown_scorecard(scorecard)
        self.assertIn("Autonomous Federation, Tool Synthesis & Multi-Agent Swarm Workflows", md)
        self.assertIn("100.0%", md)

    def test_cli_level_5_runner(self) -> None:
        # Test CLI execution for level 5
        exit_code = run_benchmark_cli(["--level", "5", "--quiet"])
        self.assertEqual(exit_code, 0)

        # Test single task invocation
        exit_code_single = run_benchmark_cli(["--task", "task_mcp_server_gateway_federation", "--quiet"])
        self.assertEqual(exit_code_single, 0)
