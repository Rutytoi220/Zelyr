"""Evaluation types, status classifications, and result models for Zelyr evaluation."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ResultStatus(str, enum.Enum):
    """Evaluation task outcome classifications."""
    PASS = "PASS"
    FAIL = "FAIL"
    TIMEOUT = "TIMEOUT"
    STEP_LIMIT_EXCEEDED = "STEP_LIMIT_EXCEEDED"
    UNVERIFIED = "UNVERIFIED"
    ERROR = "ERROR"


class EvalLevel(str, enum.Enum):
    """Benchmark evaluation hierarchy tiers."""
    LEVEL_1 = "level_1"
    LEVEL_2 = "level_2"
    LEVEL_3 = "level_3"
    LEVEL_4 = "level_4"
    ALL = "all"


@dataclass
class ValidationResult:
    """Outcome of an independent validator inspecting observable system state."""
    success: bool
    message: str = ""
    details: Optional[Dict[str, Any]] = None

    def __bool__(self) -> bool:
        return self.success


@dataclass
class EvaluationResult:
    """Structured evaluation task execution and validation result."""
    task_id: str
    status: ResultStatus
    benchmark_level: int = 1
    step_count: int = 0
    model_requests_count: int = 0
    tool_call_count: int = 0
    tool_sequence: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    failure_category: Optional[str] = None
    validator_message: str = ""
    error_message: Optional[str] = None
    measured_tokens: Optional[Dict[str, int]] = None
    estimated_tokens: Optional[Dict[str, int]] = None
    cleanup_success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status == ResultStatus.PASS

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a JSON-serializable dictionary."""
        return {
            "task_id": self.task_id,
            "status": self.status.value if isinstance(self.status, ResultStatus) else str(self.status),
            "benchmark_level": self.benchmark_level,
            "step_count": self.step_count,
            "model_requests_count": self.model_requests_count,
            "tool_call_count": self.tool_call_count,
            "tool_sequence": list(self.tool_sequence),
            "duration_seconds": round(self.duration_seconds, 4),
            "failure_category": self.failure_category,
            "validator_message": self.validator_message,
            "error_message": self.error_message,
            "measured_tokens": self.measured_tokens,
            "estimated_tokens": self.estimated_tokens,
            "cleanup_success": self.cleanup_success,
            "metadata": dict(self.metadata),
        }
