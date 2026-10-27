"""Evaluation execution context capturing state, traces, and environment boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EvaluationContext:
    """Carries complete observable state and execution trace for an evaluation run."""
    task_id: str
    workspace_dir: Path
    db_path: Path
    memory_store: Optional[Any] = None

    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    tool_sequence: List[str] = field(default_factory=list)
    orchestration_messages: List[Dict[str, Any]] = field(default_factory=list)
    final_response: str = ""
    exception: Optional[Exception] = None
    step_count: int = 0
    model_requests_count: int = 0
    step_limit_hit: bool = False
    start_time: float = 0.0
    end_time: float = 0.0
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def resolve_workspace_path(self, relative_path: str | Path) -> Path:
        """Resolve a relative path inside the isolated workspace."""
        path = (self.workspace_dir / relative_path).resolve()
        # Security invariant: stay inside workspace_dir
        if not str(path).startswith(str(self.workspace_dir.resolve())):
            raise ValueError(f"Path traversal detected: {relative_path} escapes workspace {self.workspace_dir}")
        return path

    def record_tool_invocation(self, name: str, args: Dict[str, Any], result: Any) -> None:
        """Record an executed tool invocation and its observed output."""
        self.tool_sequence.append(name)
        self.tool_calls.append({"name": name, "arguments": args})
        self.tool_results.append({"name": name, "result": result})

    def record_model_request(self) -> None:
        """Increment model requests count."""
        self.model_requests_count += 1
