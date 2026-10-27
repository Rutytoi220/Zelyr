"""Synchronous base agent for legacy/compatibility agents."""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class SimpleBaseAgent:
    """Sync base agent with bus/registry mapped to _event_bus/_tool_registry.

    This is the compatibility base for agents that were written against the
    original sync API (``EchoAgent``, ``OrchestratorAgent``).  New agents
    should inherit from :class:`zelyr.agents.base.BaseAgent` (async) instead.

    Constructor parameters ``bus`` and ``registry`` are stored as
    ``_event_bus`` and ``_tool_registry`` respectively, matching the
    naming convention of :class:`BaseAgent`.  The legacy attribute names
    (``self.bus``, ``self.registry``) are kept as aliases so existing
    code that accesses them continues to work.
    """

    def __init__(self, name: str, registry=None, bus=None, memory=None):
        """Initialize base agent.

        Args:
            name: Agent name
            registry: Component registry (optional)
            bus: Event bus for publishing events (optional)
            memory: Optional memory store
        """
        self.name = name
        self._tool_registry = registry
        self.registry = registry  # backward compat alias
        self._event_bus = bus
        self.bus = bus  # backward compat alias
        self.memory = memory
        self._execution_count = 0

    def run(self, task: str) -> Any:
        """Execute a task. Must be overridden by subclasses."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement run()")

    def get_info(self) -> Dict[str, Any]:
        """Return agent metadata for introspection (e.g. the CLI's ``agents`` command).

        Subclasses may override ``description`` (property or attribute) and
        ``_state`` for more specific reporting; both fall back to sensible
        defaults here so every ``SimpleBaseAgent`` subclass is introspectable
        without requiring an explicit override.
        """
        return {
            "name": self.name,
            "description": getattr(self, "description", f"{self.__class__.__name__} agent"),
            "state": getattr(self, "_state", "idle"),
            "execution_count": self._execution_count,
        }

    def _emit(self, event: str, data: Dict[str, Any]) -> None:
        """Emit an event via the event bus (if available and callable)."""
        bus = self._event_bus or self.bus
        if bus and hasattr(bus, 'publish_sync'):
            try:
                bus.publish_sync(event, data)
            except Exception:
                pass  # Silently ignore if publish fails

    def _log(self, msg: str, steps: List[str]) -> None:
        """Add a step to the log."""
        steps.append(msg)
