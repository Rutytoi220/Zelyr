"""Synchronous compatibility adapter for Zelyr agents."""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class SyncAgentAdapter:
    """Compatibility adapter providing sync interface for agents.
    
    This class wraps synchronous agent execution and provides
    event emission and logging utilities. All future agents needing
    sync execution should inherit from this class.
    
    For async-first agents, use zelyr.agents.base.BaseAgent instead.
    """

    def __init__(self, name: str, registry=None, bus=None, memory=None):
        """Initialize adapter.
        
        Args:
            name: Agent name
            registry: Component registry (optional)
            bus: Event bus for publishing events (optional)
            memory: Optional memory store
        """
        self.name = name
        self.registry = registry
        self.bus = bus
        self.memory = memory
        self._execution_count = 0

    def run(self, task: str):
        """Execute a task. Must be overridden by subclasses.
        
        Args:
            task: Task description
            
        Returns:
            AgentResult with execution outcome
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement run()")

    def get_info(self) -> Dict[str, Any]:
        """Return agent metadata for introspection.
        
        Subclasses may override ``description`` (property or attribute) and
        ``_state`` for more specific reporting; both fall back to sensible
        defaults here so every adapter subclass is introspectable
        without requiring an explicit override.
        """
        return {
            "name": self.name,
            "description": getattr(self, "description", f"{self.__class__.__name__} agent"),
            "state": getattr(self, "_state", "idle"),
            "execution_count": self._execution_count,
        }

    def _emit(self, event: str, data: Dict[str, Any]) -> None:
        """Emit an event via the event bus (if available and callable).
        
        Args:
            event: Event name/type
            data: Event payload
        """
        if self.bus and hasattr(self.bus, 'publish_sync'):
            try:
                self.bus.publish_sync(event, data)
            except Exception:
                pass  # Silently ignore if publish fails

    def _log(self, msg: str, steps: List[str]) -> None:
        """Add a step to the log.
        
        Args:
            msg: Step description
            steps: List to append to
        """
        steps.append(msg)
