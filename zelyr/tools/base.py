from typing import Any, Dict
from zelyr.tools.core import BaseTool, ToolResult

class ZelyrPlugin(BaseTool):
    """Base class for user-defined dynamic plugins."""
    plugin_name: str = ""
    plugin_description: str = ""
    plugin_parameters: Dict[str, Any] = {}
    requires_approval: bool = False
    
    @property
    def tool_id(self) -> str:
        return self.plugin_name
        
    @property
    def name(self) -> str:
        return self.plugin_name
        
    @property
    def description(self) -> str:
        return self.plugin_description
        
    @property
    def schema(self) -> Dict[str, Any]:
        params = self.plugin_parameters
        if not params or not isinstance(params, dict) or params.get("type") in ("", None) or params.get("properties") is None:
            return {"type": "object", "properties": {}}
        return params
        
    async def execute(self, **kwargs) -> ToolResult:
        raise NotImplementedError("Plugins must implement execute(**kwargs)")

def zelyr_tool(name: str, description: str, parameters: dict, requires_approval: bool = False):
    """Decorator to convert a function into a Zelyr plugin tool."""
    def decorator(func):
        func.__zelyr_tool__ = True
        func.__axiom_tool__ = True  # legacy plugin marker
        func.__tool_name__ = name
        func.__tool_description__ = description
        func.__tool_parameters__ = parameters
        func.__tool_requires_approval__ = requires_approval
        return func
    return decorator

class DecoratorTool(BaseTool):
    """Internal wrapper for functions decorated with @zelyr_tool."""
    def __init__(self, func, name: str, description: str, parameters: dict, requires_approval: bool = False):
        super().__init__()
        self._func = func
        self._tool_name = name
        self._tool_description = description
        self._tool_parameters = parameters
        self.requires_approval = requires_approval
        
    @property
    def tool_id(self) -> str:
        return self._tool_name
        
    @property
    def name(self) -> str:
        return self._tool_name
        
    @property
    def description(self) -> str:
        return self._tool_description
        
    @property
    def schema(self) -> dict:
        params = self._tool_parameters
        if not params or not isinstance(params, dict) or params.get("type") in ("", None) or params.get("properties") is None:
            return {"type": "object", "properties": {}}
        return params
        
    async def execute(self, params: dict) -> ToolResult:
        import inspect
        if inspect.iscoroutinefunction(self._func):
            result = await self._func(**params)
        else:
            result = self._func(**params)
            
        if isinstance(result, ToolResult):
            return result
        return ToolResult(success=True, output=result)

# Legacy AXIOM API compatibility aliases
AxiomPlugin = ZelyrPlugin
axiom_tool = zelyr_tool
