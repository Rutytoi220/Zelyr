"""Zelyr Autonomous Benchmark Scorecard & Telemetry Engine.

Generates consolidated evaluation scorecards across Levels 1, 2, and 3,
computing pass rates, latency distributions, tool efficiency ratios,
failure taxonomy, and rendering publication-grade GitHub-flavored Markdown.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from zelyr.eval.metrics import EvalStatus, TaskResult
from zelyr.eval.types import EvaluationResult, ResultStatus


@dataclass
class BenchmarkScorecard:
    """Consolidated metric scorecard and telemetry report."""

    suite_name: str
    total_tasks: int
    passed_count: int
    failed_count: int
    timeout_count: int
    tool_error_count: int
    step_limit_count: int
    unverified_count: int
    pass_rate: float
    total_duration_seconds: float
    mean_duration_seconds: float
    median_duration_seconds: float
    mean_step_count: float
    total_tool_calls: int
    tool_efficiency_ratio: float
    failure_breakdown: Dict[str, int] = field(default_factory=dict)
    level_breakdown: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    tasks: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert scorecard to JSON-serializable dictionary."""
        return {
            "suite_name": self.suite_name,
            "summary": {
                "total_tasks": self.total_tasks,
                "passed": self.passed_count,
                "failed": self.failed_count,
                "timeout": self.timeout_count,
                "tool_error": self.tool_error_count,
                "step_limit_exceeded": self.step_limit_count,
                "unverified": self.unverified_count,
                "pass_rate_pct": round(self.pass_rate * 100, 2),
                "total_duration_seconds": round(self.total_duration_seconds, 3),
                "mean_duration_seconds": round(self.mean_duration_seconds, 3),
                "median_duration_seconds": round(self.median_duration_seconds, 3),
                "mean_step_count": round(self.mean_step_count, 2),
                "total_tool_calls": self.total_tool_calls,
                "tool_efficiency_ratio": round(self.tool_efficiency_ratio, 2),
            },
            "failure_breakdown": dict(self.failure_breakdown),
            "level_breakdown": {
                str(k): v for k, v in self.level_breakdown.items()
            },
            "tasks": list(self.tasks),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize scorecard to formatted JSON."""
        return json.dumps(self.to_dict(), indent=indent)


def _infer_task_level(task_id: str, metadata: Optional[Dict[str, Any]] = None) -> int:
    """Infer task level (1, 2, or 3) from metadata or task identifier."""
    if metadata and "benchmark_level" in metadata:
        return int(metadata["benchmark_level"])
    if metadata and "level" in metadata:
        return int(metadata["level"])

    tid = task_id.lower()
    if tid.startswith("p1_") or tid.startswith("l1_") or "level1" in tid:
        return 1
    if tid.startswith("p2_") or tid.startswith("l2_") or "synthetic" in tid:
        return 2
    if tid.startswith("p3_") or tid.startswith("l3_") or "self_repair" in tid or "level3" in tid:
        return 3
    LEVEL_4_TASK_IDS = {
        "task_autonomous_runtime_bug_patching",
        "task_multi_agent_blackboard_coordination",
        "task_corrupted_state_recovery_and_rollback",
        "task_complex_data_pipeline_etl",
        "task_dynamic_tool_sandbox_containment",
        "task_proactive_memory_decay_audit",
    }
    if tid in LEVEL_4_TASK_IDS or tid.startswith("p4_") or tid.startswith("l4_") or "level4" in tid or "level_4" in tid:
        return 4
    return 1


def _extract_task_dict(result: Union[TaskResult, EvaluationResult, Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize any task result representation into a unified dictionary."""
    if isinstance(result, TaskResult):
        status_val = result.status.value if isinstance(result.status, EvalStatus) else str(result.status)
        tool_count = len(result.tool_calls) if isinstance(result.tool_calls, list) else 0
        level = _infer_task_level(result.task_id, result.artifacts)
        exp_status = (
            result.artifacts.get("expected_status")
            if isinstance(result.artifacts, dict)
            else None
        )
        is_passed = (status_val in ("PASS", "PASSED")) or (
            exp_status is not None and status_val == str(exp_status)
        )
        fail_cat = None if is_passed or status_val == "UNVERIFIED" else (result.failure_category or status_val)
        return {
            "task_id": result.task_id,
            "status": status_val,
            "level": level,
            "is_passed": is_passed,
            "expected_status": exp_status,
            "step_count": result.step_count,
            "tool_call_count": tool_count,
            "duration_seconds": result.duration_seconds,
            "tokens": result.measured_tokens if result.measured_tokens is not None else result.estimated_tokens,
            "message": result.message or "",
            "failure_category": fail_cat,
        }

    if isinstance(result, EvaluationResult):
        status_val = result.status.value if isinstance(result.status, ResultStatus) else str(result.status)
        level = result.benchmark_level or _infer_task_level(result.task_id, result.metadata)
        exp_status = (
            result.metadata.get("expected_status")
            if isinstance(result.metadata, dict)
            else None
        )
        is_passed = (status_val in ("PASS", "PASSED")) or (
            exp_status is not None and status_val == str(exp_status)
        )
        msg = result.validator_message or result.error_message or ""
        fail_cat = None if is_passed or status_val == "UNVERIFIED" else (result.failure_category or status_val)
        return {
            "task_id": result.task_id,
            "status": status_val,
            "level": level,
            "is_passed": is_passed,
            "expected_status": exp_status,
            "step_count": result.step_count,
            "tool_call_count": result.tool_call_count,
            "duration_seconds": result.duration_seconds,
            "tokens": result.measured_tokens or result.estimated_tokens,
            "message": msg,
            "failure_category": fail_cat,
        }

    if isinstance(result, dict):
        status_val = str(result.get("status", "UNVERIFIED")).upper()
        tid = str(result.get("task_id", "unknown_task"))
        meta = result.get("metadata") or result.get("artifacts") or {}
        level = int(result.get("level", _infer_task_level(tid, meta)))
        tools = result.get("tool_calls") or result.get("tool_sequence") or []
        tool_count = int(result.get("tool_call_count", len(tools) if isinstance(tools, list) else 0))
        exp_status = result.get("expected_status") or meta.get("expected_status")
        is_passed = (
            result.get("is_passed")
            if result.get("is_passed") is not None
            else ((status_val in ("PASS", "PASSED")) or (exp_status is not None and status_val == str(exp_status)))
        )
        fail_cat = None if is_passed or status_val == "UNVERIFIED" else (result.get("failure_category") or status_val)
        return {
            "task_id": tid,
            "status": status_val,
            "level": level,
            "is_passed": is_passed,
            "expected_status": exp_status,
            "step_count": int(result.get("step_count", 0)),
            "tool_call_count": tool_count,
            "duration_seconds": float(result.get("duration_seconds", 0.0)),
            "tokens": result.get("measured_tokens") or result.get("estimated_tokens") or result.get("tokens", 0),
            "message": str(result.get("message") or result.get("validator_message") or result.get("error_message") or ""),
            "failure_category": fail_cat,
        }

    raise TypeError(f"Unsupported task result type: {type(result)}")


def generate_benchmark_scorecard(
    results: Union[List[TaskResult], List[EvaluationResult], List[Dict[str, Any]]],
    suite_name: str = "Zelyr Evaluation Benchmark Suite",
) -> BenchmarkScorecard:
    """Generate consolidated scorecard from a collection of evaluation outcomes."""
    normalized_tasks: List[Dict[str, Any]] = [_extract_task_dict(r) for r in results]
    total = len(normalized_tasks)

    if total == 0:
        return BenchmarkScorecard(
            suite_name=suite_name,
            total_tasks=0,
            passed_count=0,
            failed_count=0,
            timeout_count=0,
            tool_error_count=0,
            step_limit_count=0,
            unverified_count=0,
            pass_rate=0.0,
            total_duration_seconds=0.0,
            mean_duration_seconds=0.0,
            median_duration_seconds=0.0,
            mean_step_count=0.0,
            total_tool_calls=0,
            tool_efficiency_ratio=0.0,
            failure_breakdown={},
            level_breakdown={},
            tasks=[],
        )

    passed_count = sum(1 for t in normalized_tasks if t.get("is_passed", False))
    failed_count = sum(1 for t in normalized_tasks if not t.get("is_passed", False) and t["status"] != "UNVERIFIED")
    timeout_count = sum(1 for t in normalized_tasks if t["status"] == "TIMEOUT" and not t.get("is_passed", False))
    tool_error_count = sum(1 for t in normalized_tasks if t["status"] == "TOOL_ERROR" and not t.get("is_passed", False))
    step_limit_count = sum(1 for t in normalized_tasks if t["status"] == "STEP_LIMIT_EXCEEDED" and not t.get("is_passed", False))
    unverified_count = sum(1 for t in normalized_tasks if t["status"] == "UNVERIFIED")

    evaluated_tasks = [t for t in normalized_tasks if t["status"] != "UNVERIFIED"]
    pass_rate = (passed_count / len(evaluated_tasks)) if evaluated_tasks else 1.0

    durations = [t["duration_seconds"] for t in normalized_tasks]
    total_duration = sum(durations)
    mean_duration = statistics.mean(durations) if durations else 0.0
    median_duration = statistics.median(durations) if durations else 0.0

    steps = [t["step_count"] for t in normalized_tasks]
    mean_steps = statistics.mean(steps) if steps else 0.0

    total_tools = sum(t["tool_call_count"] for t in normalized_tasks)
    tool_efficiency = (total_tools / total) if total > 0 else 0.0

    # Failure breakdown
    failure_breakdown: Dict[str, int] = {}
    for t in normalized_tasks:
        if not t.get("is_passed", False) and t.get("failure_category"):
            cat = t["failure_category"]
            failure_breakdown[cat] = failure_breakdown.get(cat, 0) + 1

    # Level breakdown
    level_breakdown: Dict[int, Dict[str, Any]] = {}
    for lvl in sorted({t["level"] for t in normalized_tasks}):
        lvl_tasks = [t for t in normalized_tasks if t["level"] == lvl]
        lvl_passed = sum(1 for t in lvl_tasks if t.get("is_passed", False))
        lvl_total = len(lvl_tasks)
        lvl_eval = [t for t in lvl_tasks if t["status"] != "UNVERIFIED"]
        lvl_durations = [t["duration_seconds"] for t in lvl_tasks]
        level_breakdown[lvl] = {
            "total": lvl_total,
            "passed": lvl_passed,
            "failed": len(lvl_eval) - lvl_passed,
            "unverified": lvl_total - len(lvl_eval),
            "pass_rate_pct": round((lvl_passed / len(lvl_eval) * 100), 1) if lvl_eval else 100.0,
            "mean_duration_seconds": round(statistics.mean(lvl_durations), 3) if lvl_durations else 0.0,
            "total_tool_calls": sum(t["tool_call_count"] for t in lvl_tasks),
        }

    return BenchmarkScorecard(
        suite_name=suite_name,
        total_tasks=total,
        passed_count=passed_count,
        failed_count=failed_count,
        timeout_count=timeout_count,
        tool_error_count=tool_error_count,
        step_limit_count=step_limit_count,
        unverified_count=unverified_count,
        pass_rate=pass_rate,
        total_duration_seconds=total_duration,
        mean_duration_seconds=mean_duration,
        median_duration_seconds=median_duration,
        mean_step_count=mean_steps,
        total_tool_calls=total_tools,
        tool_efficiency_ratio=tool_efficiency,
        failure_breakdown=failure_breakdown,
        level_breakdown=level_breakdown,
        tasks=normalized_tasks,
    )


def render_markdown_scorecard(scorecard: Union[BenchmarkScorecard, Dict[str, Any]]) -> str:
    """Render publication-grade GitHub-flavored Markdown benchmark scorecard."""
    if isinstance(scorecard, dict):
        sc = scorecard
        summary = sc.get("summary", {})
        total = summary.get("total_tasks", 0)
        passed = summary.get("passed", 0)
        pass_rate = summary.get("pass_rate_pct", 0.0)
        suite_name = sc.get("suite_name", "Zelyr Evaluation Benchmark Suite")
        dur_total = summary.get("total_duration_seconds", 0.0)
        mean_dur = summary.get("mean_duration_seconds", 0.0)
        med_dur = summary.get("median_duration_seconds", 0.0)
        tools = summary.get("total_tool_calls", 0)
        efficiency = summary.get("tool_efficiency_ratio", 0.0)
        failures = sc.get("failure_breakdown", {})
        levels = sc.get("level_breakdown", {})
        tasks = sc.get("tasks", [])
    else:
        suite_name = scorecard.suite_name
        total = scorecard.total_tasks
        passed = scorecard.passed_count
        pass_rate = scorecard.pass_rate * 100
        dur_total = scorecard.total_duration_seconds
        mean_dur = scorecard.mean_duration_seconds
        med_dur = scorecard.median_duration_seconds
        tools = scorecard.total_tool_calls
        efficiency = scorecard.tool_efficiency_ratio
        failures = scorecard.failure_breakdown
        levels = scorecard.level_breakdown
        tasks = scorecard.tasks
        unverified_count = scorecard.unverified_count

    badge_status = "PASSING" if (total > 0 and passed == (total - unverified_count)) else ("ACCEPTABLE" if pass_rate >= 80.0 else "FAILING")
    unverified_text = f" | **Unverified (Offline):** `{unverified_count}`" if unverified_count else ""

    lines = [
        f"# 📊 Zelyr Autonomous Benchmark Scorecard — `{suite_name}`",
        "",
        f"> **Overall Status:** `{badge_status}` | **Pass Rate:** `{pass_rate:.1f}%` ({passed}/{total - unverified_count}){unverified_text} | **Duration:** `{dur_total:.2f}s`",
        "",
        "## 🎯 Executive KPI Summary",
        "",
        "| Key Metric | Value | Reference / Notes |",
        "| :--- | :--- | :--- |",
        f"| **Total Tasks Evaluated** | `{total}` | Offline deterministic & live suites |",
        f"| **Overall Pass Rate** | `{pass_rate:.1f}%` | `{passed}` passed / `{total - passed}` failed |",
        f"| **Total Execution Duration** | `{dur_total:.2f}s` | Cumulative runtime |",
        f"| **Mean Latency per Task** | `{mean_dur:.2f}s` | Arithmetic mean |",
        f"| **Median Latency per Task** | `{med_dur:.2f}s` | 50th percentile |",
        f"| **Total Tool Invocations** | `{tools}` | Total tool executions |",
        f"| **Tool Efficiency Ratio** | `{efficiency:.2f}` | Tool invocations per task |",
        "",
        "## 🏗️ Multi-Tier Performance Matrix",
        "",
        "| Benchmark Level | Focus Area | Tasks | Passed | Failed | Pass Rate | Mean Latency | Tool Calls |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    tier_descriptions = {
        1: "Deterministic Single-Turn Tools",
        2: "Synthetic Multi-Step Workflows",
        3: "Autonomous Dynamic Self-Repair",
        4: "Level 4 Multi-Turn Self-Repair & System Resilience",
    }

    for lvl in sorted(int(k) for k in levels.keys()):
        stats = levels[lvl] if isinstance(levels, dict) and lvl in levels else levels.get(str(lvl), {})
        desc = tier_descriptions.get(lvl, f"Level {lvl} Evaluation")
        lvl_total = stats.get("total", 0)
        lvl_passed = stats.get("passed", 0)
        lvl_failed = stats.get("failed", 0)
        lvl_pct = stats.get("pass_rate_pct", 0.0)
        lvl_mean = stats.get("mean_duration_seconds", 0.0)
        lvl_tools = stats.get("total_tool_calls", 0)
        lines.append(
            f"| **L{lvl}** | {desc} | {lvl_total} | {lvl_passed} | {lvl_failed} | "
            f"`{lvl_pct:.1f}%` | {lvl_mean:.2f}s | {lvl_tools} |"
        )

    lines.extend([
        "",
        "## ⚠️ Failure Categorization & Taxonomy",
        "",
    ])

    if not failures:
        lines.append("`No failures recorded across benchmark tasks.`")
    else:
        lines.extend([
            "| Failure Category / Status | Occurrences | Impact Level | Description |",
            "| :--- | :---: | :---: | :--- |",
        ])
        for cat, count in sorted(failures.items(), key=lambda x: x[1], reverse=True):
            impact = "HIGH" if cat in ("TOOL_ERROR", "TIMEOUT") else "MEDIUM"
            lines.append(f"| `{cat}` | {count} | `{impact}` | Task execution encountered {cat.lower()} |")

    lines.extend([
        "",
        "## 📋 Detailed Task Execution Ledger",
        "",
        "| Task ID | Level | Status | Steps | Tools | Duration | Verification Notes |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ])

    for t in tasks:
        msg = str(t.get("message") or "")
        if len(msg) > 40:
            msg = msg[:37] + "..."
        msg_clean = msg.replace("|", "/")
        st = t["status"]
        if t.get("is_passed"):
            st_disp = "PASS" if st in ("PASS", "PASSED") else f"PASS ({st})"
        else:
            st_disp = st
        lines.append(
            f"| `{t['task_id']}` | L{t['level']} | `{st_disp}` | {t['step_count']} | "
            f"{t['tool_call_count']} | {t['duration_seconds']:.2f}s | {msg_clean} |"
        )

    lines.append("")
    return "\n".join(lines)


def export_scorecard(
    scorecard: Union[BenchmarkScorecard, Dict[str, Any]],
    md_path: Optional[Union[str, Path]] = None,
    json_path: Optional[Union[str, Path]] = None,
) -> None:
    """Export benchmark scorecard to Markdown and/or JSON files."""
    if md_path:
        path_md = Path(md_path).resolve()
        path_md.parent.mkdir(parents=True, exist_ok=True)
        md_text = render_markdown_scorecard(scorecard)
        path_md.write_text(md_text, encoding="utf-8")

    if json_path:
        path_json = Path(json_path).resolve()
        path_json.parent.mkdir(parents=True, exist_ok=True)
        json_text = scorecard.to_json() if isinstance(scorecard, BenchmarkScorecard) else json.dumps(scorecard, indent=2)
        path_json.write_text(json_text, encoding="utf-8")
