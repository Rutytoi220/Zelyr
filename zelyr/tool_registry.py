"""Type-safe registry and invocation surface for Zelyr tools.

``zelyr.tools`` defines two calling conventions for tool implementations:

- The dict-parameter family (``EchoTool``, ``ShellTool``, ``FileReadTool``,
  ``FileWriteTool``, ``SystemInfoTool``, ``FileTool``) whose ``execute``
  methods are ``async def execute(self, params: Dict[str, Any])``.  # type: ignore[override]
- The legacy keyword-argument family (``ShellCommandTool``, ``ReadFileTool``,
  ``WriteFileTool``, ``PythonExecTool``) whose ``execute`` methods are
  synchronous and accept explicit keyword parameters.

Callers that invoke ``tool.execute(...)`` directly must know which calling
convention a given tool uses. ``ToolRegistry`` removes that burden: it
validates registrations against the real :class:`zelyr.tools.BaseTool`,
tracks tools by their ``tool_id``, generates OpenAI-compatible
function-calling schemas for LLM tool-calling loops, and exposes a single
``execute()`` entry point that works for both tool families by delegating to
``BaseTool.__call__`` (which already performs the signature-based dispatch
and sync/async bridging).

``ToolRegistry`` acts as a specialized View/Adapter over
``zelyr.core.registry.Registry``.  When a ``core_registry`` is injected,
all storage operations delegate to it so that tools registered through
the Engine's registry are immediately visible through this adapter, and
vice-versa.  When no ``core_registry`` is provided, an isolated instance
is created for backwards compatibility.
"""
from __future__ import annotations
from zelyr.compat import config_dir
import threading
from typing import Any, Dict, List, Optional
from zelyr.tools import BaseTool, ToolResult

class ToolRegistryError(Exception):
    """Raised when a tool registry operation is invalid."""

class ToolRegistry:
    """Thread-safe registry and invocation surface for Zelyr tools.

    When *core_registry* is provided, ``ToolRegistry`` delegates all
    storage to it and adds schema generation and safe execution on top.
    When omitted, an isolated ``core.registry.Registry`` is created so
    that existing ``ToolRegistry()`` call-sites continue to work.
    """

    def __init__(self, core_registry=None) -> None:
        """Auto-generated docstring.

Args:
    core_registry: Argument.

Returns:
    Return value.
"""
        from zelyr.core.registry import Registry as CoreRegistry
        self._core_registry = core_registry or CoreRegistry()
        self._lock = threading.RLock()

    def register(self, tool: BaseTool) -> None:
        """Register ``tool`` using its own ``tool_id`` as the registry key."""
        if not isinstance(tool, BaseTool):
            raise ToolRegistryError(f'Component must be an instance of BaseTool, got {type(tool).__name__}')
        tool_id = tool.tool_id
        if not tool_id:
            raise ToolRegistryError('Tool must define a non-empty tool_id')
        self.register_tool(tool_id, tool)

    def register_tool(self, tool_id: str, tool: BaseTool) -> None:
        """Register ``tool`` under an explicit ``tool_id``.

        Mirrors :meth:`zelyr.core.registry.Registry.register_tool` so this
        registry can act as a drop-in replacement for existing tool-storage
        callers such as the CLI.
        """
        if not tool_id:
            raise ToolRegistryError('tool_id cannot be empty')
        if not isinstance(tool, BaseTool):
            raise ToolRegistryError(f'Component must be an instance of BaseTool, got {type(tool).__name__}')
        with self._lock:
            if self._core_registry.has_tool(tool_id):
                raise ToolRegistryError(f"Tool '{tool_id}' is already registered")
            self._core_registry.register_tool(tool_id, tool)

    def unregister_tool(self, tool_id: str) -> bool:
        """Remove a registered tool. Returns False if it was not registered."""
        with self._lock:
            if not self._core_registry.has_tool(tool_id):
                return False
            self._core_registry.unregister_tool(tool_id)
            return True

    def get_tool(self, tool_id: str) -> Optional[BaseTool]:
        """Return the tool registered under ``tool_id``, or ``None``."""
        tool = self._core_registry.get_tool(tool_id)
        if tool is None:
            aliases = {
                "safe_file_search": "file_search",
                "file_opener": "file_read",
                "read_file": "file_read",
                "write_file": "file_write",
                "shell_command": "shell",
                "run_command": "shell",
            }
            if tool_id in aliases:
                tool = self._core_registry.get_tool(aliases[tool_id])
        return tool

    def list_tools(self) -> Dict[str, BaseTool]:
        """Return a shallow copy of all registered tools keyed by ``tool_id``."""
        tools = self._core_registry.list_tools()
        import json
        from pathlib import Path
        config_path = config_dir() / "settings.json"
        
        disabled = []
        try:
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    disabled = data.get('disabled_plugins', [])
        except Exception:
            pass
            
        if disabled:
            return {k: v for k, v in tools.items() if k not in disabled}
        return tools

    def __contains__(self, tool_id: str) -> bool:
        """Auto-generated docstring.

Args:
    tool_id: Argument.

Returns:
    Return value.
"""
        return self._core_registry.has_tool(tool_id)

    def __len__(self) -> int:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return len(self._core_registry.list_tools())

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Return OpenAI-compatible function-calling schemas for every tool."""
        schemas: List[Dict[str, Any]] = []
        for tool_id, tool in self.list_tools().items():
            if not hasattr(tool, 'description') or not hasattr(tool, 'schema'):
                continue
            schemas.append({'type': 'function', 'function': {'name': tool_id, 'description': tool.description or tool_id, 'parameters': tool.schema or {'type': 'object', 'properties': {}}}})
        return schemas

    def _pre_validate_path(self, tool_id: str, arguments: Dict[str, Any]) -> Optional[ToolResult]:
        """Intercept slang or hallucinatory file paths before execution."""
        from zelyr.tools import ToolResult
        import os
        
        path_arg = None
        if 'file_path' in arguments:
            path_arg = arguments['file_path']
        elif 'path' in arguments:
            path_arg = arguments['path']
        elif 'params' in arguments and isinstance(arguments['params'], dict):
            path_arg = arguments['params'].get('file_path') or arguments['params'].get('path')
            
        if path_arg and isinstance(path_arg, str) and tool_id in ('file_opener', 'file_read', 'read_file'):
            try:
                from zelyr.tools import resolve_safe_path
                from pathlib import Path
                
                # Basic slang/placeholder trap and file existence check using smart resolver
                resolved_p = resolve_safe_path(path_arg, Path.cwd())
                if not resolved_p.exists():
                    import json
                    error_payload = {
                        "status": "FATAL_ERROR",
                        "error_type": "PRE_EXECUTION_GUARD",
                        "message": f"[System Guard]: Execution blocked. The path '{path_arg}' (resolved to '{resolved_p}') DOES NOT EXIST.",
                        "action_required": "Run 'file_search' or 'ls' first to get valid absolute paths. Stop guessing filenames."
                    }
                    return ToolResult(success=False, error=json.dumps(error_payload))
            except Exception:
                pass
        return None

    def execute(self, tool_id: str, **arguments: Any) -> ToolResult:  # type: ignore[override]
        """Execute a registered tool safely, regardless of its calling convention."""
        tool = self.get_tool(tool_id)
        if tool is None:
            return ToolResult(success=False, error=f'Tool not found: {tool_id}')
        
        if getattr(tool, 'requires_approval', False):
            try:
                from zelyr.services.governor import GovernorService
                governor = GovernorService.instance()
                if governor and governor.is_strict_mode():
                    return ToolResult(success=False, error="This tool requires approval, but was invoked synchronously. Governor gates are only supported in async mode.")
            except Exception:
                pass
        val_error = self._pre_validate_path(tool.tool_id, arguments)
        if val_error:
            return val_error
            
        try:
            return tool(**arguments)
        except Exception as exc:
            import traceback
            import logging
            tb = traceback.format_exc()
            if "zelyr/dynamic_plugins" in tb or ".config/zelyr/plugins" in tb or ".config/axiom/plugins" in tb:
                logging.getLogger("zelyr.plugin_crash").error(tb)
            from zelyr.utils.optional_deps import OptionalDependencyError
            if isinstance(exc, (ImportError, OptionalDependencyError)):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Tool requires optional dependency: {exc}",
                )
            return ToolResult(success=False, error=str(exc))

    async def execute_async(self, tool_id: str, **arguments: Any) -> ToolResult:
        """Async version of execute.

        Directly invokes the tool's execute method and awaits it if it is
        asynchronous, bypassing the sync-bridging logic in __call__.
        """
        import asyncio
        import inspect
        from zelyr.tools import ToolResult
        tool = self.get_tool(tool_id)
        if tool is None:
            return ToolResult(success=False, error=f'Tool not found: {tool_id}')
            
        if getattr(tool, 'requires_approval', False):
            try:
                from zelyr.services.governor import GovernorService
                governor = GovernorService.instance()
                if governor and governor.is_strict_mode():
                    approved = await governor.request_approval(tool.name, arguments)
                    if not approved:
                        return ToolResult(success=False, error="User denied execution via The Governor.")
            except Exception as e:
                return ToolResult(success=False, error=f"Governor failed: {e}")
        val_error = self._pre_validate_path(tool.tool_id, arguments)
        if val_error:
            return val_error
            
        try:
            sig = inspect.signature(tool.execute)
            params_list = list(sig.parameters.values())
            single_dict_param = len(params_list) == 1 and (params_list[0].annotation == Dict[str, Any] or params_list[0].name in ('params', 'arguments', 'kwargs'))
            if single_dict_param:
                result = tool.execute(arguments)
            else:
                result = tool.execute(**arguments)
            if asyncio.iscoroutine(result):
                return await result
            return result
        except Exception as exc:
            import traceback
            import logging
            tb = traceback.format_exc()
            if "zelyr/dynamic_plugins" in tb or ".config/zelyr/plugins" in tb or ".config/axiom/plugins" in tb:
                logging.getLogger("zelyr.plugin_crash").error(tb)
            from zelyr.utils.optional_deps import OptionalDependencyError
            if isinstance(exc, (ImportError, OptionalDependencyError)):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Tool requires optional dependency: {exc}",
                )
            return ToolResult(success=False, error=str(exc))
