"""Evaluation metrics, statuses, and reporting."""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


class EvalStatus(str, enum.Enum):
    """Categorical evaluation task statuses."""
    PASS = "PASS"
    PASSED = "PASS"
    FAIL = "FAIL"
    FAILED = "FAIL"
    FALSE_SUCCESS = "FALSE_SUCCESS"
    STEP_LIMIT_EXCEEDED = "STEP_LIMIT_EXCEEDED"
    TIMEOUT = "TIMEOUT"
    TOOL_ERROR = "TOOL_ERROR"
    UNVERIFIED = "UNVERIFIED"


@dataclass
class TaskResult:
    """Standardized metric record for a completed evaluation task."""
    task_id: str
    status: EvalStatus
    message: str = ""
    step_count: int = 0
    tool_calls: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    measured_tokens: Optional[int] = None  # Strictly None for mock streams without GPU counters
    estimated_tokens: int = 0              # Calculated via NativeOrchestrator.count_tokens
    failure_category: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status in (EvalStatus.PASS, EvalStatus.PASSED)

    @property
    def score(self) -> float:
        return 1.0 if self.is_success else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert task result to a JSON-serializable dictionary."""
        return {
            "task_id": self.task_id,
            "status": self.status.value if isinstance(self.status, EvalStatus) else str(self.status),
            "message": self.message,
            "step_count": self.step_count,
            "tool_calls": list(self.tool_calls),
            "duration_seconds": round(self.duration_seconds, 4),
            "measured_tokens": self.measured_tokens,
            "estimated_tokens": self.estimated_tokens,
            "failure_category": self.failure_category,
            "artifacts": dict(self.artifacts),
        }


def format_summary_table(results: List[TaskResult]) -> str:
    """Format a list of TaskResult records into a clean ASCII table."""
    total = len(results)
    passed = sum(1 for r in results if r.status == EvalStatus.PASS)
    false_success = sum(1 for r in results if r.status == EvalStatus.FALSE_SUCCESS)
    failed = sum(1 for r in results if r.status in (EvalStatus.FAIL, EvalStatus.TOOL_ERROR))
    step_limits = sum(1 for r in results if r.status == EvalStatus.STEP_LIMIT_EXCEEDED)
    timeouts = sum(1 for r in results if r.status == EvalStatus.TIMEOUT)

    lines = [
        "=" * 90,
        "Zelyr DETERMINISTIC EVALUATION BENCHMARK REPORT",
        "=" * 90,
        f"Total: {total} | Pass: {passed} | False Success: {false_success} | Fail: {failed} | "
        f"Step Limit: {step_limits} | Timeout: {timeouts}",
        "-" * 90,
        f"{'Task ID':<30} {'Status':<20} {'Steps':<6} {'Tokens (Est)':<14} {'Duration':<10}",
        "-" * 90,
    ]

    for r in results:
        status_str = r.status.value if isinstance(r.status, EvalStatus) else str(r.status)
        tok_disp = f"{r.measured_tokens} (m)" if r.measured_tokens is not None else str(r.estimated_tokens)
        lines.append(
            f"{r.task_id:<30} {status_str:<20} {r.step_count:<6} {tok_disp:<14} {r.duration_seconds:<9.3f}s"
        )

    lines.append("=" * 90)
    return "\n".join(lines)


def export_json_report(results: List[TaskResult], path: Path) -> None:
    """Write an evaluation report to a JSON file on disk."""
    report_data = {
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r.status == EvalStatus.PASS),
            "false_success": sum(1 for r in results if r.status == EvalStatus.FALSE_SUCCESS),
            "failed": sum(1 for r in results if r.status in (EvalStatus.FAIL, EvalStatus.TOOL_ERROR)),
            "step_limit_exceeded": sum(1 for r in results if r.status == EvalStatus.STEP_LIMIT_EXCEEDED),
            "timeout": sum(1 for r in results if r.status == EvalStatus.TIMEOUT),
        },
        "results": [r.to_dict() for r in results],
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
