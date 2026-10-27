"""ZelyrPlugin base class & lifecycle hook system.

Every third-party plugin MUST inherit from ``ZelyrPlugin`` and implement the
lifecycle hooks it cares about.  The rest default to no-ops.

The ``HookResult`` return type lets middleware hooks (``before_tool_execute``,
``after_tool_execute``) signal to the runtime whether execution should proceed
or be aborted.
"""
from __future__ import annotations
import logging
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)

@dataclass
class PluginToolDefinition:
    """Declares a single callable tool offered by a plugin."""
    name: str
    description: str
    handler: Any
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    'Each entry: {"name": str, "type": str, "description": str, "required": bool}'

@dataclass
class HookResult:
    """Return value from middleware lifecycle hooks.

    Use :meth:`continue_execution` or :meth:`abort` factory methods for
    clarity at call sites.
    """
    proceed: bool
    reason: Optional[str] = None
    modified_args: Optional[Dict[str, Any]] = None

    @classmethod
    def continue_execution(cls, modified_args: Optional[Dict[str, Any]]=None) -> 'HookResult':
        """Signal that execution should continue unchanged (or with modified args)."""
        return cls(proceed=True, modified_args=modified_args)  # pragma: no cover

    @classmethod
    def abort(cls, reason: str) -> 'HookResult':
        """Signal that the tool call should be cancelled."""
        return cls(proceed=False, reason=reason)  # pragma: no cover

class ZelyrPlugin(ABC):
    """Base class for all Zelyr v2 community plugins.

    Subclasses override only the lifecycle hooks they need.  The runtime
    guarantees these hooks are called in the following order:

    1. ``on_load(context)``
    2. For each tool call: ``before_tool_execute`` → tool → ``after_tool_execute``
    3. For each EventBus event: ``on_event(event)``
    4. ``on_shutdown()``
    """

    def __init__(self) -> None:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        self._tools: List[PluginToolDefinition] = []  # pragma: no cover
        self._plugin_id: str = ''  # pragma: no cover

    def register_tool(self, name: str, description: str, handler: Any, parameters: Optional[List[Dict[str, Any]]]=None) -> None:
        """Register a tool callable from inside ``on_load``."""
        self._tools.append(PluginToolDefinition(name=name, description=description, handler=handler, parameters=parameters or []))  # pragma: no cover
        logger.debug("[%s] registered tool '%s'", self._plugin_id, name)  # pragma: no cover

    @property
    def tools(self) -> List[PluginToolDefinition]:
        """Return the list of tools declared by this plugin."""
        return list(self._tools)  # pragma: no cover

    def on_load(self, context: Any) -> None:
        """Called once when the plugin is first instantiated.

        Use this to validate API keys, register tools, and set up state.
        Raise any exception to prevent loading.
        """

    def on_event(self, event: Any) -> None:
        """Called for every EventBus publication matching ``*``."""

    def before_tool_execute(self, tool_name: str, args: Dict[str, Any]) -> HookResult:
        """Middleware hook invoked before any tool (from this plugin) executes.

        Return ``HookResult.abort(...)`` to cancel the call.
        """
        return HookResult.continue_execution()  # pragma: no cover

    def after_tool_execute(self, tool_name: str, result: Any) -> HookResult:
        """Middleware hook invoked after a tool (from this plugin) executes."""
        return HookResult.continue_execution()  # pragma: no cover

    def on_shutdown(self) -> None:
        """Cleanup hook — close network connections, flush state, etc."""

# Legacy AXIOM API compatibility aliases
AxiomPlugin = ZelyrPlugin
