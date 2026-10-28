"""Tests for Zelyr Autonomous Benchmark Scorecard and Telemetry Engine."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from zelyr.eval.cli import run_benchmark_cli
from zelyr.eval.metrics import EvalStatus, TaskResult
from zelyr.eval.scorecard import (
    BenchmarkScorecard,
    export_scorecard,
    generate_benchmark_scorecard,
    render_markdown_scorecard,
)
from zelyr.eval.types import EvaluationResult, ResultStatus


def test_generate_benchmark_scorecard_metrics():
    """Verify scorecard metric computation, level partitioning, and failure breakdown."""
    results = [
        # Level 1 Passed
        TaskResult(
            task_id="p1_exec_read",
            status=EvalStatus.PASS,
            message="File read successfully",
            step_count=2,
            tool_calls=["read_file"],
            duration_seconds=1.0,
            estimated_tokens=150,
        ),
        # Level 1 Passed
        EvaluationResult(
            task_id="l1_state_check",
            status=ResultStatus.PASS,
            benchmark_level=1,
            step_count=1,
            tool_call_count=1,
            duration_seconds=0.5,
        ),
        # Level 2 Passed
        TaskResult(
            task_id="p2_synthetic_triage",
            status=EvalStatus.PASS,
            message="Triage report written",
            step_count=4,
            tool_calls=["file_search", "read_file", "write_file"],
            duration_seconds=3.0,
            estimated_tokens=500,
        ),
        # Level 3 Passed
        TaskResult(
            task_id="self_repair_syntax_test",
            status=EvalStatus.PASS,
            message="Dynamic tool repaired",
            step_count=3,
            tool_calls=["install_tool", "run_test"],
            duration_seconds=2.5,
            estimated_tokens=350,
        ),
        # Level 1 Timeout
        TaskResult(
            task_id="p1_hanging_cmd",
            status=EvalStatus.TIMEOUT,
            message="Process exceeded 5s",
            step_count=1,
            tool_calls=["shell"],
            duration_seconds=5.0,
            failure_category="TIMEOUT",
        ),
        # Level 2 Step Limit
        TaskResult(
            task_id="p2_looping_workflow",
            status=EvalStatus.STEP_LIMIT_EXCEEDED,
            message="Reached 10 steps",
            step_count=10,
            tool_calls=["shell"] * 10,
            duration_seconds=4.0,
            failure_category="STEP_LIMIT_EXCEEDED",
        ),
        # Level 3 Tool Error
        TaskResult(
            task_id="l3_broken_plugin",
            status=EvalStatus.TOOL_ERROR,
            message="SyntaxError in plugin",
            step_count=2,
            tool_calls=["install_tool"],
            duration_seconds=1.5,
            failure_category="TOOL_ERROR",
        ),
    ]

    scorecard = generate_benchmark_scorecard(results, suite_name="Test CI Fleet Suite")

    assert scorecard.suite_name == "Test CI Fleet Suite"
    assert scorecard.total_tasks == 7
    assert scorecard.passed_count == 4
    assert scorecard.timeout_count == 1
    assert scorecard.step_limit_count == 1
    assert scorecard.tool_error_count == 1
    assert abs(scorecard.pass_rate - (4 / 7)) < 1e-4

    # Latency metrics
    assert scorecard.total_duration_seconds == 17.5
    assert scorecard.mean_duration_seconds == 2.5
    assert scorecard.median_duration_seconds == 2.5

    # Tools & efficiency
    # Tools: 1 + 1 + 3 + 2 + 1 + 10 + 1 = 19 tool calls across 7 tasks
    assert scorecard.total_tool_calls == 19
    assert abs(scorecard.tool_efficiency_ratio - (19 / 7)) < 1e-4

    # Failure breakdown
    assert scorecard.failure_breakdown == {
        "TIMEOUT": 1,
        "STEP_LIMIT_EXCEEDED": 1,
        "TOOL_ERROR": 1,
    }

    # Level breakdown
    assert set(scorecard.level_breakdown.keys()) == {1, 2, 3}
    l1 = scorecard.level_breakdown[1]
    assert l1["total"] == 3
    assert l1["passed"] == 2
    assert l1["failed"] == 1

    l2 = scorecard.level_breakdown[2]
    assert l2["total"] == 2
    assert l2["passed"] == 1
    assert l2["failed"] == 1

    l3 = scorecard.level_breakdown[3]
    assert l3["total"] == 2
    assert l3["passed"] == 1
    assert l3["failed"] == 1


def test_render_markdown_scorecard_formatting():
    """Verify Markdown rendering produces compliant tables, badges, and ledger entries."""
    scorecard = BenchmarkScorecard(
        suite_name="Regression Sprint Matrix",
        total_tasks=3,
        passed_count=3,
        failed_count=0,
        timeout_count=0,
        tool_error_count=0,
        step_limit_count=0,
        unverified_count=0,
        pass_rate=1.0,
        total_duration_seconds=6.0,
        mean_duration_seconds=2.0,
        median_duration_seconds=2.0,
        mean_step_count=2.0,
        total_tool_calls=6,
        tool_efficiency_ratio=2.0,
        failure_breakdown={},
        level_breakdown={
            1: {"total": 2, "passed": 2, "failed": 0, "pass_rate_pct": 100.0, "mean_duration_seconds": 1.5, "total_tool_calls": 3},
            2: {"total": 1, "passed": 1, "failed": 0, "pass_rate_pct": 100.0, "mean_duration_seconds": 3.0, "total_tool_calls": 3},
        },
        tasks=[
            {"task_id": "p1_alpha", "level": 1, "status": "PASS", "step_count": 2, "tool_call_count": 2, "duration_seconds": 1.2, "message": "Success alpha"},
            {"task_id": "p1_beta", "level": 1, "status": "PASS", "step_count": 1, "tool_call_count": 1, "duration_seconds": 1.8, "message": "Success beta"},
            {"task_id": "p2_gamma", "level": 2, "status": "PASS", "step_count": 3, "tool_call_count": 3, "duration_seconds": 3.0, "message": "Success gamma"},
        ],
    )

    md = render_markdown_scorecard(scorecard)

    assert "# 📊 Zelyr Autonomous Benchmark Scorecard" in md
    assert "Overall Status:** `PASSING`" in md
    assert "Pass Rate:** `100.0%` (3/3)" in md
    assert "## 🎯 Executive KPI Summary" in md
    assert "## 🏗️ Multi-Tier Performance Matrix" in md
    assert "## 📋 Detailed Task Execution Ledger" in md
    assert "`p1_alpha`" in md
    assert "`p2_gamma`" in md
    assert "No failures recorded across benchmark tasks." in md


def test_export_scorecard_to_disk(tmp_path: Path):
    """Verify export_scorecard writes Markdown and JSON files accurately."""
    scorecard = generate_benchmark_scorecard([
        TaskResult(
            task_id="p1_sample",
            status=EvalStatus.PASS,
            message="Verified",
            step_count=1,
            tool_calls=["sample_tool"],
            duration_seconds=0.8,
        )
    ])

    md_file = tmp_path / "artifacts" / "scorecard.md"
    json_file = tmp_path / "artifacts" / "scorecard.json"

    export_scorecard(scorecard, md_path=md_file, json_path=json_file)

    assert md_file.exists()
    assert json_file.exists()

    md_content = md_file.read_text(encoding="utf-8")
    assert "# 📊 Zelyr Autonomous Benchmark Scorecard" in md_content

    json_data = json.loads(json_file.read_text(encoding="utf-8"))
    assert json_data["summary"]["total_tasks"] == 1
    assert json_data["summary"]["passed"] == 1
    assert json_data["summary"]["pass_rate_pct"] == 100.0


def test_eval_cli_scorecard_flag(tmp_path: Path):
    """Verify running eval CLI with --scorecard produces benchmark_scorecard.md and .json."""
    sc_md = tmp_path / "custom_scorecard.md"

    # Run Level 1 single deterministic task
    exit_code = run_benchmark_cli([
        "-l", "1",
        "-t", "task_tool_dispatch",
        "--scorecard", str(sc_md),
        "-q",
    ])

    assert exit_code == 0
    assert sc_md.exists()

    # Matching JSON scorecard also generated
    sc_json = sc_md.with_suffix(".json")
    assert sc_json.exists()

    md_text = sc_md.read_text(encoding="utf-8")
    assert "# 📊 Zelyr Autonomous Benchmark Scorecard" in md_text
    assert "`task_tool_dispatch`" in md_text

    json_obj = json.loads(sc_json.read_text(encoding="utf-8"))
    assert json_obj["summary"]["passed"] == 1


def test_main_cli_eval_subcommand_dispatch():
    """Verify top-level `zelyr eval --list-tasks` command dispatches cleanly."""
    import sys
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "zelyr.cli.main", "eval", "--list-tasks"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0
    assert "Available Zelyr Evaluation Benchmark Tasks:" in result.stdout
    assert "Level 1" in result.stdout
    assert "Level 2" in result.stdout
    assert "Level 3" in result.stdout
