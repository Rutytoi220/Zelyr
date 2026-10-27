"""Evaluation suite reporting and serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from zelyr.eval.types import EvaluationResult, ResultStatus


@dataclass
class BenchmarkReport:
    """Consolidated report across all benchmark evaluation tasks."""
    suite_name: str
    results: List[EvaluationResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0

    @property
    def total_tasks(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.status == ResultStatus.PASS)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if r.status == ResultStatus.FAIL)

    @property
    def step_limit_count(self) -> int:
        return sum(1 for r in self.results if r.status == ResultStatus.STEP_LIMIT_EXCEEDED)

    @property
    def timeout_count(self) -> int:
        return sum(1 for r in self.results if r.status == ResultStatus.TIMEOUT)

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if r.status == ResultStatus.ERROR)

    @property
    def unverified_count(self) -> int:
        return sum(1 for r in self.results if r.status == ResultStatus.UNVERIFIED)

    @property
    def pass_rate(self) -> float:
        return (self.passed_count / self.total_tasks) if self.total_tasks > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert benchmark report to a JSON-serializable dictionary."""
        return {
            "suite_name": self.suite_name,
            "total_tasks": self.total_tasks,
            "passed": self.passed_count,
            "failed": self.failed_count,
            "step_limit_exceeded": self.step_limit_count,
            "timeout": self.timeout_count,
            "error": self.error_count,
            "unverified": self.unverified_count,
            "pass_rate": round(self.pass_rate, 4),
            "total_duration_seconds": round(self.total_duration_seconds, 4),
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize benchmark report to JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    def summary_table(self) -> str:
        """Generate human-readable summary table."""
        lines = [
            f"=== Benchmark Report: {self.suite_name} ===",
            f"Total: {self.total_tasks} | Pass: {self.passed_count} | Fail: {self.failed_count} | "
            f"Step Limit: {self.step_limit_count} | Timeout: {self.timeout_count} | Error: {self.error_count}",
            f"Pass Rate: {self.pass_rate * 100:.1f}% | Duration: {self.total_duration_seconds:.2f}s",
            "-" * 80,
            f"{'Task ID':<30} {'Status':<16} {'Steps':<6} {'Tools':<6} {'Duration':<10} {'Message'}",
            "-" * 80,
        ]
        for r in self.results:
            msg = r.validator_message or r.error_message or ""
            if len(msg) > 30:
                msg = msg[:27] + "..."
            lines.append(
                f"{r.task_id:<30} {r.status.value:<16} {r.step_count:<6} {r.tool_call_count:<6} "
                f"{r.duration_seconds:<9.2f}s {msg}"
            )
        lines.append("-" * 80)
        return "\n".join(lines)
