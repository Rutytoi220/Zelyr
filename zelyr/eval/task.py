"""Evaluation task and execution context models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

from zelyr.db.memory import MemoryStore


@dataclass
class EvalTask:
    """Specification of a deterministic evaluation task."""
    task_id: str
    description: str
    level: int = 1
    allowed_tools: Optional[List[str]] = None
    max_steps: int = 5
    timeout_seconds: float = 10.0
    setup_fn: Optional[Callable[[Path, MemoryStore], Union[Awaitable[None], None]]] = None
    validator_fn: Optional[Callable[[EvalContext], Union[Awaitable[Tuple[bool, str]], Tuple[bool, str]]]] = None
    cleanup_fn: Optional[Callable[[Path], Union[Awaitable[None], None]]] = None
    user_prompt: str = ""
    mock_responses: Optional[List[Dict[str, Any]]] = None
    mock_stream_fn: Optional[Callable[[dict], Any]] = None
    custom_tools: Optional[Dict[str, Callable[..., Any]]] = None
    tool_schemas: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Backward-compatibility aliases for L1 evaluation suite
    fake_tools: Dict[str, Callable[..., Any]] = field(default_factory=dict)
    mock_turns: List[Any] = field(default_factory=list)
    timeout: float = 10.0
    setup_hook: Optional[Callable[[Any], Any]] = None
    cleanup_hook: Optional[Callable[[Any], Any]] = None
    validator: Optional[Any] = None
    expected_status: Any = None
    benchmark_level: int = 1

    def __post_init__(self) -> None:
        if self.custom_tools is None and self.fake_tools:
            self.custom_tools = dict(self.fake_tools)
        elif self.custom_tools and not self.fake_tools:
            self.fake_tools = dict(self.custom_tools)

        if self.timeout != 10.0 and self.timeout_seconds == 10.0:
            self.timeout_seconds = self.timeout
        elif self.timeout_seconds != 10.0 and self.timeout == 10.0:
            self.timeout = self.timeout_seconds

        if self.benchmark_level != 1 and self.level == 1:
            self.level = self.benchmark_level
        elif self.level != 1 and self.benchmark_level == 1:
            self.benchmark_level = self.level


@dataclass
class EvalContext:
    """Live observable state and execution trace for an active evaluation task."""
    task: EvalTask
    workspace_dir: Path
    db_path: Path
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    orchestrator_messages: List[Dict[str, Any]] = field(default_factory=list)
    final_response: str = ""
    step_count: int = 0
    execution_error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    memory_store: Optional[MemoryStore] = None
    step_limit_hit: bool = False
    duration_seconds: float = 0.0
    tools_dir: Optional[Path] = None

    @property
    def tools_directory(self) -> Path:
        if self.tools_dir is not None:
            return self.tools_dir
        td = self.workspace_dir / "tools.d"
        td.mkdir(parents=True, exist_ok=True)
        return td

    @property
    def task_id(self) -> str:
        return self.task.task_id

    @property
    def tool_sequence(self) -> List[str]:
        return [c.get("name") or c.get("tool", "") for c in self.tool_calls]

    def resolve_path(self, rel_path: str | Path) -> Path:
        """Resolve a relative path safely inside the workspace."""
        ws_resolved = self.workspace_dir.resolve()
        path = (self.workspace_dir / rel_path).resolve()
        if not path.is_relative_to(ws_resolved):
            raise ValueError(f"Path traversal detected: {rel_path} escapes workspace {self.workspace_dir}")
        return path

    def resolve_workspace_path(self, rel_path: str | Path) -> Path:
        return self.resolve_path(rel_path)

    def record_tool_invocation(self, name: str, args: Dict[str, Any], result: Any) -> None:
        self.tool_calls.append({"name": name, "arguments": args, "result": result})


# Compatibility aliases
EvaluationTask = EvalTask
EvaluationContext = EvalContext
