"""Bridge between legacy brain.action_registry actions and modern BaseTool interface.

Provides ``LegacyActionTool``, a thin adapter that wraps a single
callable from ``brain.action_registry`` so it can be registered alongside
modern tools and invoked through the same ``ToolRegistry`` / LLM
function-calling pipeline.

``create_legacy_tools()`` is a convenience factory that reads all
registered legacy actions and returns a list of ``LegacyActionTool``
instances ready for registration.  If the legacy ``brain`` package is
not importable (e.g. in a pure-zelyr environment), the factory returns
an empty list instead of raising.
"""
import logging
from typing import Any, Dict, List
from zelyr.tools import BaseTool, ToolParameter, ToolResult
logger = logging.getLogger(__name__)

class LegacyActionTool(BaseTool):
    """Wraps a single legacy action from brain.action_registry as a BaseTool."""

    def __init__(self, action_name: str, action_handler) -> None:
        """Auto-generated docstring.

Args:
    action_name: Argument.
    action_handler: Argument.

Returns:
    Return value.
"""
        super().__init__(tool_id=f'legacy_{action_name}', name=action_name, description=f'Legacy action: {action_name}')
        self._action_name = action_name
        self._handler = action_handler
        self.add_parameter(ToolParameter(name='params', type='string', description=f'Parameter string for {action_name}', required=False, default=''))

    def execute(self, params: Any='', **kwargs: Any) -> ToolResult:  # type: ignore[override]
        """Auto-generated docstring.

Args:
    params: Argument.

Returns:
    Return value.
"""
        try:
            if isinstance(params, dict):
                params = params.get('params', '')
            if not isinstance(params, str):
                params = str(params)
            success, message = self._handler(params)
            if success is None:
                return ToolResult(success=False, error='Action not handled by legacy registry')
            return ToolResult(success=bool(success), output=message)
        except Exception as e:
            return ToolResult(success=False, error=f'Legacy action failed: {e}')

def create_legacy_tools() -> List[LegacyActionTool]:
    """Create LegacyActionTool instances for all registered legacy actions.

    Returns an empty list if brain.action_registry is not importable.
    """
    try:
        from brain.action_registry import get_action_registry
        registry = get_action_registry()
        tools = []
        for action_name, handler in registry.actions.items():
            tools.append(LegacyActionTool(action_name, handler))
        return tools
    except ImportError:
        logger.warning('Legacy brain.action_registry not available')
        return []
