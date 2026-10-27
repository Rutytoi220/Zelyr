"""Registry system for dynamic component management."""
from typing import Any, Dict, Type, Optional
from abc import ABC, abstractmethod
import threading

class Registrable(ABC):
    """Base class for registrable components."""

    @classmethod
    @abstractmethod
    def get_identifier(cls) -> str:
        """Return unique identifier for this component."""
        pass

class Registry:
    """Central registry for tools, agents, and plugins."""

    def __init__(self):
        """Auto-generated docstring.


Returns:
    Return value.
"""
        self._tools: Dict[str, Any] = {}
        self._agents: Dict[str, Any] = {}
        self._plugins: Dict[str, Any] = {}
        self._handlers: Dict[str, Any] = {}
        self._lock = threading.RLock()

    def register_tool(self, tool_id: str, tool_instance: Any) -> None:
        """Register a tool."""
        with self._lock:
            self._tools[tool_id] = tool_instance

    def unregister_tool(self, tool_id: str) -> None:
        """Unregister a tool."""
        with self._lock:
            if tool_id in self._tools:
                del self._tools[tool_id]

    def get_tool(self, tool_id: str) -> Optional[Any]:
        """Get a tool by ID."""
        with self._lock:
            return self._tools.get(tool_id)

    def has_tool(self, tool_id: str) -> bool:
        """Check if a tool is registered."""
        with self._lock:
            return tool_id in self._tools

    def list_tools(self) -> Dict[str, Any]:
        """List all registered tools."""
        with self._lock:
            return self._tools.copy()

    def register_agent(self, agent_id: str, agent_instance: Any) -> None:
        """Register an agent."""
        with self._lock:
            self._agents[agent_id] = agent_instance

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent."""
        with self._lock:
            if agent_id in self._agents:
                del self._agents[agent_id]

    def get_agent(self, agent_id: str) -> Optional[Any]:
        """Get an agent by ID."""
        with self._lock:
            return self._agents.get(agent_id)

    def list_agents(self) -> Dict[str, Any]:
        """List all registered agents."""
        with self._lock:
            return self._agents.copy()

    def register_plugin(self, plugin_id: str, plugin_instance: Any) -> None:
        """Register a plugin."""
        with self._lock:
            self._plugins[plugin_id] = plugin_instance

    def unregister_plugin(self, plugin_id: str) -> None:
        """Unregister a plugin."""
        with self._lock:
            if plugin_id in self._plugins:
                del self._plugins[plugin_id]

    def get_plugin(self, plugin_id: str) -> Optional[Any]:
        """Get a plugin by ID."""
        with self._lock:
            return self._plugins.get(plugin_id)

    def list_plugins(self) -> Dict[str, Any]:
        """List all registered plugins."""
        with self._lock:
            return self._plugins.copy()

    def register_handler(self, handler_id: str, handler: Any) -> None:
        """Register a custom handler."""
        with self._lock:
            self._handlers[handler_id] = handler

    def get_handler(self, handler_id: str) -> Optional[Any]:
        """Get a handler by ID."""
        with self._lock:
            return self._handlers.get(handler_id)

    def list_handlers(self) -> Dict[str, Any]:
        """List all registered handlers."""
        with self._lock:
            return self._handlers.copy()

    def clear_all(self) -> None:
        """Clear all registries (dangerous - use with care)."""
        with self._lock:
            self._tools.clear()
            self._agents.clear()
            self._plugins.clear()
            self._handlers.clear()
