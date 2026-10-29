"""Zelyr Core Module - Event-driven orchestration engine."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zelyr.core.engine import Engine
    from zelyr.core.events import Event, EventBus
    from zelyr.core.registry import Registry
    from zelyr.core.context import ExecutionContext
    from zelyr.core.async_bridge import run_sync, shutdown_bridge
    from zelyr.core.transaction import WorkspaceTransactionManager, StagingCapExceeded

_LAZY_EXPORTS = {
    "Engine": ("zelyr.core.engine", "Engine"),
    "EventBus": ("zelyr.core.events", "EventBus"),
    "Event": ("zelyr.core.events", "Event"),
    "Registry": ("zelyr.core.registry", "Registry"),
    "ExecutionContext": ("zelyr.core.context", "ExecutionContext"),
    "run_sync": ("zelyr.core.async_bridge", "run_sync"),
    "shutdown_bridge": ("zelyr.core.async_bridge", "shutdown_bridge"),
    "WorkspaceTransactionManager": ("zelyr.core.transaction", "WorkspaceTransactionManager"),
    "StagingCapExceeded": ("zelyr.core.transaction", "StagingCapExceeded"),
    "SandboxedPluginRunner": ("zelyr.core.plugin_sandbox", "SandboxedPluginRunner"),
}

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
    "SandboxedPluginRunner",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        mod_name, attr_name = _LAZY_EXPORTS[name]
        import importlib

        mod = importlib.import_module(mod_name)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))
