"""Comprehensive test battery for Level 3: Autonomous Self-Repair & Tool-Crafting Evaluations."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from zelyr.eval.cli import get_all_available_tasks
from zelyr.eval.harness import EvalHarness
from zelyr.eval.metrics import EvalStatus
from zelyr.eval.tasks.self_repair_l3 import (
    build_task_self_repair_bounded_retries,
    build_task_self_repair_invalid_rejection,
    build_task_self_repair_security_bypass,
    build_task_self_repair_syntax_failure,
    build_task_self_repair_unrecoverable_failure,
    get_self_repair_l3_tasks,
)
from zelyr.eval.tasks.synthetic_workflows_p2 import build_task_tool_crafting_self_repair


class TestLevel3SelfRepairEvaluations(unittest.IsolatedAsyncioTestCase):
    """Deterministic evaluation battery for Zelyr Level 3 autonomous self-repair and tool crafting."""

    async def asyncSetUp(self) -> None:
        self.harness = EvalHarness()

    async def test_scenario_1_syntax_error_detection_and_repair(self) -> None:
        """Scenario 1 & 3: Model detects syntax error, repairs AST, validates, and completes CSV sum."""
        task = build_task_self_repair_syntax_failure()
        result = await self.harness.run_task(task)

        self.assertEqual(result.status, EvalStatus.PASS)
        self.assertTrue(result.is_success)
        self.assertIn("craft_tool", result.tool_calls)
        self.assertIn("csv_aggregator", result.tool_calls)
        self.assertEqual(result.tool_calls, ["craft_tool", "craft_tool", "csv_aggregator"])
        self.assertIn("Autonomous syntax error recovery verified", result.message)

    async def test_scenario_2_runtime_exception_recovery(self) -> None:
        """Scenario 2 & 3: Model detects runtime IndexError, updates tool via ToolManager to v2 regex, and succeeds."""
        task = build_task_tool_crafting_self_repair()
        result = await self.harness.run_task(task)

        self.assertEqual(result.status, EvalStatus.PASS)
        self.assertTrue(result.is_success)
        self.assertIn("craft_tool", result.tool_calls)
        self.assertIn("manage_tools", result.tool_calls)
        self.assertIn("parse_metrics", result.tool_calls)
        self.assertGreaterEqual(result.step_count, 4)
        self.assertIn("Tool crafting and autonomous self-repair loop verified", result.message)

    async def test_scenario_4_invalid_repair_rejection(self) -> None:
        """Scenario 4: Rejects tools violating AST contracts (sync execute) or attempting Ring 0 privilege escalation."""
        task = build_task_self_repair_invalid_rejection()
        result = await self.harness.run_task(task)

        self.assertEqual(result.status, EvalStatus.PASS)
        self.assertTrue(result.is_success)
        self.assertEqual(result.tool_calls, ["craft_tool", "craft_tool"])
        self.assertIn("Security and contract enforcement verified", result.message)

    async def test_scenario_5_security_bypass_rejection(self) -> None:
        """Scenario 5: Rejects protected tool overwrites (manage_tools) and path traversal (../../evil_escape)."""
        task = build_task_self_repair_security_bypass()
        result = await self.harness.run_task(task)

        self.assertEqual(result.status, EvalStatus.PASS)
        self.assertTrue(result.is_success)
        self.assertEqual(result.tool_calls, ["manage_tools", "craft_tool"])
        self.assertIn("Protected tool protection and path traversal prevention verified", result.message)

    async def test_scenario_6_repeated_failure_bounded_retries(self) -> None:
        """Scenario 6: Repeating tool failures halt safely at max_steps boundary without infinite looping."""
        task = build_task_self_repair_bounded_retries()
        result = await self.harness.run_task(task)

        self.assertEqual(result.status, EvalStatus.STEP_LIMIT_EXCEEDED)
        self.assertFalse(result.is_success)
        self.assertEqual(result.step_count, 3)
        self.assertEqual(len(result.tool_calls), 3)
        self.assertIn("Reached maximum autonomous ReAct steps (3)", result.message)

    async def test_scenario_7_unrecoverable_failure_truthful_acknowledgment(self) -> None:
        """Scenario 7: Model truthfully acknowledges unrecoverable hardware fault without false success claims."""
        task = build_task_self_repair_unrecoverable_failure()
        result = await self.harness.run_task(task)

        self.assertEqual(result.status, EvalStatus.PASS)
        self.assertTrue(result.is_success)
        self.assertEqual(result.tool_calls, ["hardware_probe"])
        self.assertIn("Unrecoverable tool failure truthfully acknowledged without false success", result.message)

    def test_level_3_battery_discovery_and_metadata(self) -> None:
        """Verify discovery helpers and metadata conform to Level 3 specifications."""
        l3_tasks = get_self_repair_l3_tasks(include_existing_runtime_repair=True)
        self.assertEqual(len(l3_tasks), 6)

        task_ids = {t.task_id for t in l3_tasks}
        self.assertEqual(
            task_ids,
            {
                "task_self_repair_syntax_failure",
                "task_tool_crafting_self_repair",
                "task_self_repair_invalid_rejection",
                "task_self_repair_security_bypass",
                "task_self_repair_bounded_retries",
                "task_self_repair_unrecoverable_failure",
            },
        )
        for t in l3_tasks:
            self.assertEqual(t.level, 3)
            self.assertIsNotNone(t.allowed_tools)
            self.assertIsNotNone(t.tool_schemas)

    def test_cli_discovers_all_level_3_tasks(self) -> None:
        """Verify CLI get_all_available_tasks registers all 6 Level 3 tasks."""
        all_tasks = get_all_available_tasks()
        l3_tasks = [t for t in all_tasks if t.level == 3]
        self.assertEqual(len(l3_tasks), 6)

    async def test_workspace_isolation_and_cleanup_invariants(self) -> None:
        """Verify ephemeral workspaces and temporary SQLite databases are wiped post-execution."""
        task = build_task_self_repair_syntax_failure()
        result = await self.harness.run_task(task)

        self.assertEqual(result.status, EvalStatus.PASS)
        ws_str = result.artifacts.get("workspace_dir")
        db_str = result.artifacts.get("db_path")

        self.assertIsNotNone(ws_str)
        self.assertIsNotNone(db_str)
        self.assertFalse(Path(ws_str).exists())
        self.assertFalse(Path(db_str).exists())
