import logging
from abc import ABC
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)

class ToolResult:
    """Represents the result of a tool execution."""

    def __init__(
        self,
        success: bool,
        output: Any = None,
        error: Optional[str] = None,
        metadata: Optional[Dict] = None,
        remedy_hint: Optional[str] = None,
        allowed_actions: Optional[List[str]] = None,
    ):
        """Represents the result of a tool execution."""
        self.success = success
        self.output = output
        self.error = error
        self.metadata = metadata or {}
        self.remedy_hint = remedy_hint if remedy_hint is not None else self.metadata.get("remedy_hint")
        self.allowed_actions = allowed_actions if allowed_actions is not None else self.metadata.get("allowed_actions", [])
        if self.remedy_hint:
            self.metadata["remedy_hint"] = self.remedy_hint
        if self.allowed_actions:
            self.metadata["allowed_actions"] = self.allowed_actions

    def __repr__(self):
        """Return human-readable representation."""
        if self.success:
            return f"ToolResult(success={self.success}, output={self.output!r})"
        if self.remedy_hint:
            return f"ToolResult(success={self.success}, error={self.error!r}, remedy_hint={self.remedy_hint!r})"
        return f"ToolResult(success={self.success}, error={self.error!r})"

    def to_dict(self, tool: str = "", arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return the strict Zelyr tool-result envelope."""
        result = {"output": self.output, "error": self.error, "metadata": self.metadata}
        if self.remedy_hint:
            result["remedy_hint"] = self.remedy_hint
        if self.allowed_actions:
            result["allowed_actions"] = self.allowed_actions

        envelope = {
            "tool": tool,
            "arguments": arguments or {},
            "result": result,
            "success": bool(self.success),
        }
        if not self.success and self.error:
            envelope["error"] = self.error
        if self.remedy_hint:
            envelope["remedy_hint"] = self.remedy_hint
        if self.allowed_actions:
            envelope["allowed_actions"] = self.allowed_actions
        return envelope

class ToolParameter:
    """Parameter definition for a tool."""

    def __init__(self, name: str, type: str, description: str, required: bool=True, default: Any=None):
        """Auto-generated docstring.

Args:
    name: Argument.
    type: Argument.
    description: Argument.
    required: Argument.
    default: Argument.

Returns:
    Return value.
"""
        self.name = name
        self.type = type
        self.description = description
        self.required = required
        self.default = default

class ToolCapability:
    """Formal capability metadata for dynamic tier routing and schema pruning."""

    def __init__(
        self,
        tier: Optional[int] = 1,
        is_core: bool = False,
        requires_bridge: bool = False,
        requires_window: bool = False,
        token_cost: int = 150,
    ):
        self.tier = tier
        self.is_core = is_core
        self.requires_bridge = requires_bridge
        self.requires_window = requires_window
        self.token_cost = token_cost

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier,
            "is_core": self.is_core,
            "requires_bridge": self.requires_bridge,
            "requires_window": self.requires_window,
            "token_cost": self.token_cost,
        }


class BaseTool(ABC):
    """Base class for all tools."""

    def __init__(
        self,
        tool_id: str | None = None,
        name: str | None = None,
        description: str | None = None,
        tier: int = 1,
        is_core: bool = False,
        requires_bridge: bool = False,
        requires_window: bool = False,
        token_cost: int = 150,
    ):
        self._tool_id = tool_id
        self._name = name
        self._description = description
        self.tier = tier
        self.is_core = is_core
        self.requires_bridge = requires_bridge
        self.requires_window = requires_window
        self.token_cost = token_cost
        self.capability = ToolCapability(
            tier=tier,
            is_core=is_core,
            requires_bridge=requires_bridge,
            requires_window=requires_window,
            token_cost=token_cost,
        )
        self.parameters: List[ToolParameter] = []
        self._execution_count = 0

    @property
    def tool_id(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return self._tool_id or ""

    @property
    def name(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return self._name or ""

    @property
    def description(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return self._description or ""

    @property
    def schema(self) -> Dict[str, Any]:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        if hasattr(self, 'parameters') and self.parameters:
            properties = {}
            required = []
            for p in self.parameters:
                properties[p.name] = {'type': p.type, 'description': p.description}
                if p.required:
                    required.append(p.name)
            return {'type': 'object', 'properties': properties, 'required': required}
        return {'type': 'object', 'properties': {}}

    def add_parameter(self, param: ToolParameter) -> None:
        """Auto-generated docstring.

Args:
    param: Argument.

Returns:
    Return value.
"""
        self.parameters.append(param)

    def get_info(self) -> Dict[str, Any]:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return {'tool_id': self.tool_id, 'name': self.name, 'description': self.description, 'parameters': [{'name': p.name, 'type': p.type, 'description': p.description, 'required': p.required, 'default': p.default} for p in self.parameters], 'execution_count': self._execution_count}

    def validate_parameters(self, **kwargs) -> bool:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        if not hasattr(self, 'parameters') or not self.parameters:
            return True
        required_params = {p.name for p in self.parameters if p.required}
        provided_params = set(kwargs.keys())
        if not required_params.issubset(provided_params):
            missing = required_params - provided_params
            logger.error(f'Missing required parameters: {missing}')
            return False
        return True

    def __call__(self, *args, **kwargs) -> ToolResult:
        """Invoke the tool, adapting kwargs to whichever calling convention
        ``execute`` uses (a single ``params``/``arguments`` dict, or explicit
        keyword arguments), and bridging async ``execute`` implementations
        onto a synchronous return value.

        The single-dict-parameter adaptation is determined purely from the
        ``execute`` signature and applies whether ``execute`` is sync or
        async, so callers do not need to know a tool's implementation style.
        """
        self._execution_count += 1
        if hasattr(self, 'parameters') and self.parameters:
            if not self.validate_parameters(**kwargs):
                return ToolResult(success=False, output=None, error='Invalid parameters')
        import inspect
        import asyncio
        sig = inspect.signature(self.execute)
        params_list = list(sig.parameters.values())
        single_dict_param = len(params_list) == 1 and (params_list[0].annotation == Dict[str, Any] or params_list[0].name in ('params', 'arguments', 'kwargs'))
        if single_dict_param:
            execute_args = args if args else (kwargs,)
            execute_kwargs = {}
        else:
            execute_args = args
            execute_kwargs = kwargs
        result = self.execute(*execute_args, **execute_kwargs)
        if not asyncio.iscoroutine(result):
            return result
        from zelyr.core.async_bridge import run_sync
        return run_sync(result)

    def execute(self, *args, **kwargs) -> ToolResult:  # type: ignore[override]
        """Execute the tool implementation."""
        raise NotImplementedError(f'{self.__class__.__name__} must implement execute()')

