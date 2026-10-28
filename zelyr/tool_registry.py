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
from zelyr.tools.core import BaseTool, ToolResult

class BuiltinToolDef:
    """Lightweight metadata descriptor for a built-in tool."""

    def __init__(
        self,
        tool_id: str,
        description: str,
        module_path: str,
        class_name: str,
        constructor_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.tool_id = tool_id
        self.name = tool_id
        self.description = description
        self.module_path = module_path
        self.class_name = class_name
        self.constructor_kwargs = constructor_kwargs or {}
        self._instance: Optional[BaseTool] = None

    def instantiate(self) -> BaseTool:
        if self._instance is None:
            import importlib
            mod = importlib.import_module(self.module_path)
            cls = getattr(mod, self.class_name)
            self._instance = cls(**self.constructor_kwargs)
        return self._instance


BUILTIN_TOOLS: Dict[str, BuiltinToolDef] = {
    "app_launcher": BuiltinToolDef(
        tool_id="app_launcher",
        description="Launch common default apps (e.g. browser, calculator, terminal, file_explorer)",
        module_path="zelyr.tools.os_assist",
        class_name="AppLauncherTool",
    ),
    "clipboard_read": BuiltinToolDef(
        tool_id="clipboard_read",
        description="Read the current text content of the system clipboard.",
        module_path="zelyr.tools.clipboard_tools",
        class_name="ClipboardReadTool",
    ),
    "clipboard_write": BuiltinToolDef(
        tool_id="clipboard_write",
        description="Write text content to the system clipboard.",
        module_path="zelyr.tools.clipboard_tools",
        class_name="ClipboardWriteTool",
    ),
    "echo": BuiltinToolDef(
        tool_id="echo",
        description="Echoes the input back to verify tool calls are working.",
        module_path="zelyr.tools",
        class_name="EchoTool",
    ),
    "file_read": BuiltinToolDef(
        tool_id="file_read",
        description="Reads files and extracts text. Supports plain text (.txt, .py, .json, .md) and document extraction (.pdf).",
        module_path="zelyr.tools",
        class_name="FileReadTool",
        constructor_kwargs={"base_dir": "."},
    ),
    "file_search": BuiltinToolDef(
        tool_id="file_search",
        description="Find files safely in standard user directories (Desktop, Documents, etc.)",
        module_path="zelyr.tools.os_assist",
        class_name="SafeFileSearchTool",
    ),
    "file_write": BuiltinToolDef(
        tool_id="file_write",
        description="Write, append, or delete files safely.",
        module_path="zelyr.tools",
        class_name="FileWriteTool",
        constructor_kwargs={"base_dir": "."},
    ),
    "shell": BuiltinToolDef(
        tool_id="shell",
        description="Execute Linux shell commands. Do NOT run interactive commands.",
        module_path="zelyr.tools",
        class_name="ShellTool",
    ),
    "system_info": BuiltinToolDef(
        tool_id="system_info",
        description="Retrieve CPU, Memory, Disk, and OS information.",
        module_path="zelyr.tools",
        class_name="SystemInfoTool",
    ),
    "task_scheduler": BuiltinToolDef(
        tool_id="task_scheduler",
        description="Schedule a task to run at a specific future time. Supports natural language like 'in 1 hour', 'in 30 minutes', or a cron expression.",
        module_path="zelyr.tools.task_scheduler",
        class_name="TaskSchedulerTool",
    ),
}

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
            resolved_id = aliases.get(tool_id, tool_id)
            tool = self._core_registry.get_tool(resolved_id)
            if tool is None:
                if resolved_id in BUILTIN_TOOLS:
                    tool = self._load_builtin_tool(resolved_id)
                elif tool_id in BUILTIN_TOOLS:
                    tool = self._load_builtin_tool(tool_id)
        return tool

    def get_builtin_tools(self) -> Dict[str, BuiltinToolDef]:
        """Return metadata descriptors for all standard built-in tools."""
        return dict(BUILTIN_TOOLS)

    def _load_builtin_tool(self, tool_id: str) -> Optional[BaseTool]:
        defn = BUILTIN_TOOLS.get(tool_id)
        if defn is None:
            return None
        with self._lock:
            if not self._core_registry.has_tool(tool_id):
                instance = defn.instantiate()
                self._core_registry.register_tool(tool_id, instance)
                return instance
            return self._core_registry.get_tool(tool_id)

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
        from zelyr.tools.core import ToolResult
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
        from zelyr.tools.core import ToolResult
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
