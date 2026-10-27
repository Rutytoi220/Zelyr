"""Zelyr Core Module - Event-driven orchestration engine."""

from zelyr.core.engine import Engine
from zelyr.core.events import Event, EventBus
from zelyr.core.registry import Registry
from zelyr.core.context import ExecutionContext
from zelyr.core.async_bridge import run_sync, shutdown_bridge
from zelyr.core.transaction import WorkspaceTransactionManager, StagingCapExceeded

__all__ = [
    "Engine",
    "EventBus",
    "Event",
    "Registry",
    "ExecutionContext",
    "run_sync",
    "shutdown_bridge",
    "WorkspaceTransactionManager",
    "StagingCapExceeded",
]
