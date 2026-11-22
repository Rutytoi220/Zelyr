"""Zelyr engine module for orchestration and lifecycle management."""

from zelyr.engine.core import CoreEngine
from zelyr.engine.engine import Engine
from zelyr.engine.budget_manager import ContextBudgetEngine, AdaptiveContextBudgeter, ContextBudgetLimitError
from zelyr.engine.environment import RuntimeEnvironmentManifest
from zelyr.engine.execution_ledger import (
    ActionRecord,
    ActionState,
    InvalidStateTransitionError,
    TurnExecutionLedger,
    detect_consultant_deferral,
    is_unresolvable_terminal_blocker,
    has_remaining_explicit_steps,
    parse_acceptance_criteria,
    canonicalize_target,
    targets_match,
    extract_prohibited_targets,
)
from zelyr.engine.normalizer import (
    APPROVED_CANONICAL_ALIASES,
    RESERVED_AUTHORIZATION_KEYS,
    SafeArgumentNormalizer,
    strict_json_loads,
    validate_arguments,
)
from zelyr.engine.observation_framing import ObservationFramingEngine
from zelyr.engine.schema_compact import CompactSchemaGenerator

__all__ = [
    "CoreEngine",
    "Engine",
    "ContextBudgetEngine",
    "ContextBudgetLimitError",
    "AdaptiveContextBudgeter",
    "RuntimeEnvironmentManifest",
    "ActionRecord",
    "ActionState",
    "InvalidStateTransitionError",
    "TurnExecutionLedger",
    "APPROVED_CANONICAL_ALIASES",
    "RESERVED_AUTHORIZATION_KEYS",
    "SafeArgumentNormalizer",
    "strict_json_loads",
    "validate_arguments",
    "ObservationFramingEngine",
    "CompactSchemaGenerator",
    "detect_consultant_deferral",
    "is_unresolvable_terminal_blocker",
    "has_remaining_explicit_steps",
    "parse_acceptance_criteria",
    "canonicalize_target",
    "targets_match",
    "extract_prohibited_targets",
]
