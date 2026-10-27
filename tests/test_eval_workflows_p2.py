"""Unit and integration test suite for Phase 2 Synthetic Workflows and Self-Repair."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from zelyr.eval.harness import EvalHarness
from zelyr.eval.metrics import EvalStatus
from zelyr.eval.tasks.synthetic_workflows_p2 import (
    build_task_multisession_memory_persistence,
    build_task_synthetic_service_triage,
    build_task_tool_crafting_self_repair,
    build_task_unrecoverable_actuator_containment,
    get_synthetic_workflows_p2_tasks,
)
from zelyr.eval.validators import (
    assert_json_field,
    assert_tool_updated,
    validate_json_field,
    validate_tool_updated,
)


class TestEvalWorkflowsP2(unittest.IsolatedAsyncioTestCase):
    """Test suite covering Level 2/3 synthetic workflows, self-repair loops, and isolation."""

    async def asyncSetUp(self) -> None:
        self.harness = EvalHarness()

    async def test_workflow_service_triage_passes(self) -> None:
        """Verify multi-step diagnostic triage isolates failure and outputs valid JSON artifact."""
        task = build_task_synthetic_service_triage()
        result = await self.harness.run_task(task)

        self.assertEqual(result.status, EvalStatus.PASS)
        self.assertTrue(result.is_success)
        self.assertIn("mock_inspect_services", result.tool_calls)
        self.assertIn("mock_read_logs", result.tool_calls)
        self.assertIn("write_file", result.tool_calls)
        self.assertEqual(
            result.tool_calls,
            ["mock_inspect_services", "mock_read_logs", "write_file"],
        )
        self.assertGreater(result.step_count, 0)
        self.assertGreater(result.estimated_tokens, 0)
        self.assertIsNone(result.measured_tokens)
        self.assertIn("Diagnostic triage verified", result.message)

    async def test_workflow_tool_crafting_self_repair_passes(self) -> None:
        """Verify dynamic tool crafting, runtime exception recovery, and self-repair via ToolManager."""
        task = build_task_tool_crafting_self_repair()
        result = await self.harness.run_task(task)

        self.assertEqual(result.status, EvalStatus.PASS)
        self.assertTrue(result.is_success)
        self.assertIn("craft_tool", result.tool_calls)
        self.assertIn("manage_tools", result.tool_calls)
        self.assertIn("parse_metrics", result.tool_calls)
        # Verify multi-turn progression
        self.assertGreaterEqual(result.step_count, 4)
        self.assertIn("Tool crafting and autonomous self-repair loop verified", result.message)

    async def test_workflow_multisession_memory_persists_and_hydrates(self) -> None:
        """Verify facts stored in session 1 are hydrated into session 2 prompts without GPU/network."""
        task = build_task_multisession_memory_persistence()
        result = await self.harness.run_task(task)

        self.assertEqual(result.status, EvalStatus.PASS)
        self.assertTrue(result.is_success)
        self.assertIn("remember_fact", result.tool_calls)
        self.assertIn("Multi-session persistence verified", result.message)

    async def test_workflow_unrecoverable_failure_categorized_cleanly(self) -> None:
        """Verify exhausted recovery options halt gracefully and classify as FAIL without false success."""
        task = build_task_unrecoverable_actuator_containment()
        result = await self.harness.run_task(task)

        self.assertIn(result.status, [EvalStatus.FAIL, EvalStatus.TOOL_ERROR])
        self.assertNotEqual(result.status, EvalStatus.FALSE_SUCCESS)
        self.assertFalse(result.is_success)
        self.assertIn(result.failure_category, ["VALIDATION_FAILED", "TOOL_ERROR"])
        self.assertIn("failing_actuator", result.tool_calls)
        self.assertIn("Hardware failure encountered", result.message)

    async def test_workflow_workspace_cleanup_invariants(self) -> None:
        """Verify ephemeral workspaces and test databases are completely torn down post-execution."""
        task = build_task_synthetic_service_triage()
        result = await self.harness.run_task(task)

        self.assertEqual(result.status, EvalStatus.PASS)

        workspace_str = result.artifacts.get("workspace_dir")
        self.assertIsNotNone(workspace_str)
        workspace_path = Path(workspace_str)

        db_str = result.artifacts.get("db_path")
        self.assertIsNotNone(db_str)
        db_path = Path(db_str)

        # Invariant: Neither temporary directory nor SQLite file remains on disk
        self.assertFalse(workspace_path.exists())
        self.assertFalse(db_path.exists())

    def test_battery_discovery_helper(self) -> None:
        """Verify get_synthetic_workflows_p2_tasks discovers all 4 tasks."""
        tasks = get_synthetic_workflows_p2_tasks()
        self.assertEqual(len(tasks), 4)
        task_ids = {t.task_id for t in tasks}
        self.assertEqual(
            task_ids,
            {
                "task_synthetic_service_triage",
                "task_tool_crafting_self_repair",
                "task_multisession_memory_persistence",
                "task_unrecoverable_actuator_containment",
            },
        )

    def test_assert_json_field_unit(self) -> None:
        """Unit tests for assert_json_field validator helper."""
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            json_file = ws / "test.json"
            json_file.write_text(json.dumps({"a": {"b": 42}, "items": ["x", "y"]}), encoding="utf-8")

            # Match nested key
            ok, msg = assert_json_field(ws, "test.json", "a.b", 42)
            self.assertTrue(ok)

            # Match list index
            ok, msg = assert_json_field(ws, "test.json", "items.1", "y")
            self.assertTrue(ok)

            # Mismatched value
            ok, msg = assert_json_field(ws, "test.json", "a.b", 99)
            self.assertFalse(ok)
            self.assertIn("expected 99", msg)

            # Missing key
            ok, msg = assert_json_field(ws, "test.json", "a.c", 42)
            self.assertFalse(ok)
            self.assertIn("not found", msg)

            # Missing file
            ok, msg = assert_json_field(ws, "nonexistent.json", "a.b", 42)
            self.assertFalse(ok)

    def test_assert_tool_updated_unit(self) -> None:
        """Unit tests for assert_tool_updated validator helper."""
        with tempfile.TemporaryDirectory() as td:
            tools_dir = Path(td)
            tool_file = tools_dir / "my_tool.py"
            tool_file.write_text("# marker: V2_TEST_MARKER\nprint('hello')\n", encoding="utf-8")

            # Match marker
            ok, msg = assert_tool_updated(tools_dir, "my_tool", "V2_TEST_MARKER")
            self.assertTrue(ok)

            # Mismatched marker
            ok, msg = assert_tool_updated(tools_dir, "my_tool", "V3_MISSING")
            self.assertFalse(ok)

            # Missing tool
            ok, msg = assert_tool_updated(tools_dir, "missing_tool", "V2_TEST_MARKER")
            self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
