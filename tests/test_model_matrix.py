"""Tests for Model Matrix Evaluation Subsystem (TASK 9).

Verifies multi-archetype evaluation harness, metric aggregation, offline containment,
and CLI matrix integration.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from zelyr.eval.cli import run_benchmark_cli
from zelyr.eval.matrix import (
    ArchetypeMetrics,
    MatrixTaskOutcome,
    ModelMatrixEvaluator,
    format_matrix_table,
    render_matrix_scorecard_markdown,
)


class TestModelMatrixEvaluator(unittest.IsolatedAsyncioTestCase):
    """Hermetic unit and integration tests for ModelMatrixEvaluator."""

    async def test_mock_matrix_evaluation_speed_and_determinism(self):
        """Verify all 3 archetypes evaluate deterministically in < 3.0 seconds."""
        evaluator = ModelMatrixEvaluator()
        t0 = time.perf_counter()
        results = await evaluator.evaluate_mock_matrix()
        duration = time.perf_counter() - t0

        self.assertLess(duration, 3.0, f"Mock matrix evaluation took too long: {duration:.2f}s")
        self.assertEqual(results.get("mode"), "mock")
        archetypes = results.get("archetypes", {})
        self.assertEqual(len(archetypes), 3)

        for arch_name in ModelMatrixEvaluator.ARCHETYPES:
            self.assertIn(arch_name, archetypes)
            arch_data = archetypes[arch_name]
            self.assertEqual(arch_data["tasks_total"], 5)
            self.assertEqual(arch_data["tasks_passed"], 5)
            self.assertEqual(len(arch_data["task_outcomes"]), 5)

        summary = results.get("summary", {})
        self.assertTrue(summary.get("all_passed"))
        self.assertEqual(summary.get("total_evaluations"), 15)

    async def test_mock_matrix_metrics_contracts(self):
        """Assert metrics contracts across archetypes:
        - false_success_blocked_rate == 1.0 (100% false success claims suppressed)
        - false_refusal_rate == 0.0 (0% false cloud refusals leaked)
        - compliance_rate >= 0.90 for compliant and degraded archetypes
        - autonomous_recovery_rate >= 0.90 across all archetypes.
        """
        evaluator = ModelMatrixEvaluator()
        results = await evaluator.evaluate_mock_matrix()
        archetypes = results.get("archetypes", {})

        for name, data in archetypes.items():
            self.assertEqual(
                data["false_success_blocked_rate"],
                1.0,
                f"{name} failed to achieve 100% false-success suppression",
            )
            self.assertEqual(
                data["false_refusal_rate"],
                0.0,
                f"{name} leaked false cloud refusal",
            )
            self.assertGreaterEqual(
                data["autonomous_recovery_rate"],
                0.90,
                f"{name} autonomous recovery rate below 90%",
            )

        # Archetype A and B must achieve >= 90% compliance
        arch_a = archetypes["Archetype A (Compliant Function Caller)"]
        arch_b = archetypes["Archetype B (Degraded 3B Markdown Caller)"]
        self.assertGreaterEqual(arch_a["compliance_rate"], 0.90)
        self.assertGreaterEqual(arch_b["compliance_rate"], 0.90)

    async def test_live_matrix_offline_containment(self):
        """Verify evaluate_live_matrix gracefully handles unreachable Ollama endpoints."""
        evaluator = ModelMatrixEvaluator()
        # Query an unreachable port
        unreachable_url = "http://127.0.0.1:59999"
        result = await evaluator.evaluate_live_matrix(base_url=unreachable_url)

        self.assertFalse(result.get("live_eval"))
        self.assertIn("Ollama daemon not reachable", result.get("reason", ""))
        self.assertEqual(result.get("models"), {})

    def test_format_matrix_table_renders_archetypes_and_metrics(self):
        """Verify format_matrix_table produces a table containing archetype names and metrics."""
        sample_data = {
            "archetypes": {
                "Archetype A (Compliant Function Caller)": {
                    "tasks_passed": 5,
                    "tasks_total": 5,
                    "compliance_rate_pct": 100.0,
                    "false_success_blocked_pct": 100.0,
                    "false_refusal_pct": 0.0,
                    "autonomous_recovery_pct": 100.0,
                    "duration_seconds": 0.35,
                },
                "Archetype B (Degraded 3B Markdown Caller)": {
                    "tasks_passed": 5,
                    "tasks_total": 5,
                    "compliance_rate_pct": 100.0,
                    "false_success_blocked_pct": 100.0,
                    "false_refusal_pct": 0.0,
                    "autonomous_recovery_pct": 100.0,
                    "duration_seconds": 0.32,
                },
            }
        }
        table_output = format_matrix_table(sample_data)
        self.assertIn("Archetype A", table_output)
        self.assertIn("Archetype B", table_output)
        self.assertIn("Compliance", table_output)
        self.assertIn("False", table_output)
        self.assertIn("Success", table_output)

    def test_render_matrix_scorecard_markdown(self):
        """Verify render_matrix_scorecard_markdown generates valid GitHub-flavored Markdown table."""
        sample_data = {
            "archetypes": {
                "Archetype A (Compliant Function Caller)": {
                    "tasks_passed": 5,
                    "tasks_total": 5,
                    "compliance_rate_pct": 100.0,
                    "false_success_blocked_pct": 100.0,
                    "false_refusal_pct": 0.0,
                    "autonomous_recovery_pct": 100.0,
                    "duration_seconds": 0.35,
                }
            }
        }
        md = render_matrix_scorecard_markdown(sample_data)
        self.assertIn("## Model Independence Empirical Matrix Scorecard", md)
        self.assertIn("| Model / Archetype | Tasks Passed |", md)
        self.assertIn("| **Archetype A (Compliant Function Caller)** | 5/5 | 100.0% |", md)


class TestCLIMatrixRunner(unittest.TestCase):
    """Test CLI commands and flags for matrix evaluation."""

    def test_cli_matrix_mock_execution(self):
        """Verify 'zelyr eval --matrix --mock' runs cleanly and exits 0."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_out = os.path.join(tmp_dir, "matrix_report.json")
            sc_out = os.path.join(tmp_dir, "matrix_scorecard.md")

            exit_code = run_benchmark_cli([
                "--matrix",
                "--mock",
                "--json", json_out,
                "--scorecard", sc_out,
                "--quiet",
            ])
            self.assertEqual(exit_code, 0)

            # Assert JSON file was written with expected keys
            self.assertTrue(os.path.exists(json_out))
            with open(json_out, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data.get("mode"), "mock")
            self.assertEqual(len(data.get("archetypes", {})), 3)

            # Assert Markdown scorecard was written
            self.assertTrue(os.path.exists(sc_out))
            with open(sc_out, "r", encoding="utf-8") as f:
                sc_text = f.read()
            self.assertIn("Model Independence Empirical Matrix Scorecard", sc_text)


if __name__ == "__main__":
    unittest.main()
