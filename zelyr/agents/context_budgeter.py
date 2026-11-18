"""Context budgeting and compaction facade for Zelyr agents."""

from zelyr.engine.budget_manager import (
    AdaptiveContextBudgeter,
    BudgetEngine,
    ContextBudgetEngine,
    MODEL_CONTEXT_CATALOG,
    _group_into_atomic_turns,
    _sanitize_wire_invariants,
    count_message_tokens,
    estimate_messages_tokens,
    estimate_tokens,
    get_model_context_limit,
    group_into_atomic_turns,
    is_tool_result_message,
    sanitize_wire_invariants,
)

__all__ = [
    "ContextBudgetEngine",
    "AdaptiveContextBudgeter",
    "BudgetEngine",
    "MODEL_CONTEXT_CATALOG",
    "get_model_context_limit",
    "estimate_tokens",
    "count_message_tokens",
    "estimate_messages_tokens",
    "is_tool_result_message",
    "group_into_atomic_turns",
    "_group_into_atomic_turns",
    "sanitize_wire_invariants",
    "_sanitize_wire_invariants",
]
