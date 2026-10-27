"""Deterministic, serializable execution plans.

Plans are intentionally independent of agents and tools.  This keeps task
coordination inspectable and makes an execution engine responsible for deciding
when and how to invoke a concrete capability.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping
import uuid

class PlanError(ValueError):
    """Raised when a plan violates dependency or lifecycle invariants."""

class StepStatus(str, Enum):
    """Lifecycle states for a plan step."""
    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    SKIPPED = 'skipped'
_TERMINAL_STATUSES = {StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED}

@dataclass
class PlanStep:
    """A single unit of work with explicit prerequisites."""
    id: str
    description: str
    depends_on: tuple[str, ...] = ()
    requires_confirmation: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: str | None = None

    def __post_init__(self) -> None:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        if not self.id.strip():
            raise PlanError('Plan step ID cannot be empty')
        if not self.description.strip():
            raise PlanError(f"Plan step '{self.id}' must have a description")
        if self.status == StepStatus.COMPLETED and self.error is not None:
            raise PlanError(f"Completed plan step '{self.id}' cannot have an error")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the step using JSON-compatible values."""
        return {'id': self.id, 'description': self.description, 'depends_on': list(self.depends_on), 'requires_confirmation': self.requires_confirmation, 'metadata': self.metadata, 'status': self.status.value, 'result': self.result, 'error': self.error}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PlanStep:
        """Restore a serialized plan step."""
        return cls(id=str(data['id']), description=str(data['description']), depends_on=tuple((str(step_id) for step_id in data.get('depends_on', ()))), requires_confirmation=bool(data.get('requires_confirmation', False)), metadata=dict(data.get('metadata', {})), status=StepStatus(data.get('status', StepStatus.PENDING.value)), result=data.get('result'), error=data.get('error'))

@dataclass
class ExecutionPlan:
    """A dependency-aware plan that can be persisted and resumed."""
    objective: str
    steps: list[PlanStep]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        if not self.objective.strip():
            raise PlanError('Plan objective cannot be empty')
        if not self.steps:
            raise PlanError('Execution plans require at least one step')
        self._validate_dependencies()

    def ready_steps(self, *, confirmed: bool=False) -> list[PlanStep]:
        """Return pending steps whose dependencies have completed.

        Confirmation-gated steps remain unavailable until the caller explicitly
        supplies ``confirmed=True``. This prevents accidental execution of
        potentially destructive work.
        """
        return [step for step in self.steps if step.status == StepStatus.PENDING and (confirmed or not step.requires_confirmation) and all((self._step_by_id(dependency).status == StepStatus.COMPLETED for dependency in step.depends_on))]

    def start_step(self, step_id: str, *, confirmed: bool=False) -> PlanStep:
        """Mark a ready step as running and return it."""
        step = self._step_by_id(step_id)
        if step not in self.ready_steps(confirmed=confirmed):
            raise PlanError(f"Plan step '{step_id}' is not ready to start")
        step.status = StepStatus.RUNNING
        return step

    def complete_step(self, step_id: str, result: Any=None) -> PlanStep:
        """Record successful completion for a running step."""
        step = self._step_by_id(step_id)
        if step.status != StepStatus.RUNNING:
            raise PlanError(f"Plan step '{step_id}' is not running")
        step.status = StepStatus.COMPLETED
        step.result = result
        step.error = None
        return step

    def fail_step(self, step_id: str, error: str) -> PlanStep:
        """Record failure for a running step."""
        step = self._step_by_id(step_id)
        if step.status != StepStatus.RUNNING:
            raise PlanError(f"Plan step '{step_id}' is not running")
        if not error.strip():
            raise PlanError('Failed plan steps require an error message')
        step.status = StepStatus.FAILED
        step.error = error
        return step

    @property
    def is_complete(self) -> bool:
        """Whether every step has reached a terminal state."""
        return all((step.status in _TERMINAL_STATUSES for step in self.steps))

    @property
    def succeeded(self) -> bool:
        """Whether all steps completed successfully."""
        return self.is_complete and all((step.status == StepStatus.COMPLETED for step in self.steps))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full plan using JSON-compatible values."""
        return {'id': self.id, 'objective': self.objective, 'metadata': self.metadata, 'steps': [step.to_dict() for step in self.steps]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExecutionPlan:
        """Restore a serialized execution plan."""
        return cls(id=str(data['id']), objective=str(data['objective']), metadata=dict(data.get('metadata', {})), steps=[PlanStep.from_dict(step) for step in data['steps']])

    def _step_by_id(self, step_id: str) -> PlanStep:
        """Auto-generated docstring.

Args:
    step_id: Argument.

Returns:
    Return value.
"""
        for step in self.steps:
            if step.id == step_id:
                return step
        raise PlanError(f"Unknown plan step '{step_id}'")

    def _validate_dependencies(self) -> None:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        step_ids = [step.id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise PlanError('Plan step IDs must be unique')
        known_ids = set(step_ids)
        for step in self.steps:
            missing = set(step.depends_on) - known_ids
            if missing:
                raise PlanError(f"Plan step '{step.id}' has unknown dependencies: {sorted(missing)}")
            if step.id in step.depends_on:
                raise PlanError(f"Plan step '{step.id}' cannot depend on itself")
        visited: set[str] = set()
        active: set[str] = set()

        def visit(step_id: str) -> None:
            """Auto-generated docstring.

Args:
    step_id: Argument.

Returns:
    Return value.
"""
            if step_id in active:
                raise PlanError('Plan dependencies cannot contain a cycle')
            if step_id in visited:
                return
            active.add(step_id)
            for dependency in self._step_by_id(step_id).depends_on:
                visit(dependency)
            active.remove(step_id)
            visited.add(step_id)
        for step_id in step_ids:
            visit(step_id)

class TaskPlanner:
    """Factory for creating validated execution plans from explicit steps."""

    def create_plan(self, objective: str, steps: Iterable[PlanStep], *, metadata: Mapping[str, Any] | None=None) -> ExecutionPlan:
        """Create a plan without inferring actions or capabilities."""
        return ExecutionPlan(objective=objective, steps=list(steps), metadata=dict(metadata or {}))
