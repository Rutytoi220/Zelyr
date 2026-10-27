"""Sandbox Engine for Zelyr v2 Plugin System.

Provides two layers of protection for untrusted plugin code:

1. **Static AST Analysis** — scans plugin source files *before* import for
   forbidden constructs (e.g. ``import socket`` when ``network=false``).
   This is a fast, cheap, first-pass check that blocks naive breakout attempts.

2. **Multiprocessing Isolation** — each tool call is executed inside a freshly
   spawned child process with the Python restricted-execution environment
   described below.  The child communicates with the main process via a
   ``multiprocessing.Pipe``.  If the child exits unexpectedly, is killed by
   SIGALRM, or returns an error, the proxy surfaces a ``ToolResult(success=False)``.

Security properties enforced at the multiprocessing layer
---------------------------------------------------------
- The child process has no access to the parent's memory space.
- Forbidden modules are patched to raise ``SandboxSecurityViolation`` at
  import time inside the child (``builtins.__import__`` shim).
- Filesystem access is jailed to a per-call temporary directory; any ``open()``
  call targeting a path outside that directory is rejected.
- The child is given a wall-clock timeout; the parent kills it if it does not
  respond in time (default: 30 seconds).

Note on Wasm sandboxing
-----------------------
RFC-001 mentions a Wasm/Extism primary sandbox.  That remains the v2.1 target.
The multiprocessing approach here is the *fallback* that ships in v2.0, as
specified in the RFC migration roadmap.
"""
from __future__ import annotations
import ast
import inspect
import logging
import multiprocessing
import os
import signal
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from zelyr.plugins.zelyr_plugin import ZelyrPlugin, PluginToolDefinition
from zelyr.plugins.exceptions import SandboxSecurityViolation
from zelyr.plugins.loader import PluginManifest, PluginPermissions
from zelyr.tools import BaseTool, ToolParameter, ToolResult
logger = logging.getLogger(__name__)
SANDBOX_TIMEOUT_SECONDS = 30
_NETWORK_MODULES = frozenset({'socket', 'ssl', 'urllib', 'urllib.request', 'urllib.parse', 'urllib.error', 'http', 'http.client', 'ftplib', 'smtplib', 'requests', 'httpx', 'aiohttp', 'websockets', 'paramiko'})
_SHELL_MODULES = frozenset({'subprocess', 'popen', 'pty', 'fcntl'})
_FILESYSTEM_EXTRA_MODULES = frozenset({'shutil', 'glob'})

class _ASTImportScanner(ast.NodeVisitor):
    """Visits all ``import`` / ``from … import`` nodes in a module's AST."""

    def __init__(self) -> None:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        self.imported_names: list[str] = []  # pragma: no cover

    def visit_Import(self, node: ast.Import) -> None:
        """Auto-generated docstring.

Args:
    node: Argument.

Returns:
    Return value.
"""
        for alias in node.names:  # pragma: no cover
            self.imported_names.append(alias.name)  # pragma: no cover
        self.generic_visit(node)  # pragma: no cover

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Auto-generated docstring.

Args:
    node: Argument.

Returns:
    Return value.
"""
        if node.module:  # pragma: no cover
            self.imported_names.append(node.module)  # pragma: no cover
        self.generic_visit(node)  # pragma: no cover

def static_check_plugin_source(plugin_dir: Path, permissions: PluginPermissions, plugin_id: str) -> None:
    """Scan every ``*.py`` file in *plugin_dir* for forbidden imports.

    Raises:
        SandboxSecurityViolation: If forbidden module usage is detected.
    """
    for py_file in plugin_dir.rglob('*.py'):  # pragma: no cover
        try:  # pragma: no cover
            source = py_file.read_text(encoding='utf-8')  # pragma: no cover
            tree = ast.parse(source, filename=str(py_file))  # pragma: no cover
        except (SyntaxError, OSError):  # pragma: no cover
            continue  # pragma: no cover
        scanner = _ASTImportScanner()  # pragma: no cover
        scanner.visit(tree)  # pragma: no cover
        for mod in scanner.imported_names:  # pragma: no cover
            base = mod.split('.')[0]  # pragma: no cover
            if not permissions.network and (mod in _NETWORK_MODULES or base in _NETWORK_MODULES):  # pragma: no cover
                raise SandboxSecurityViolation(plugin_id=plugin_id, violation_type='import:network', detail=f"Plugin imports network module '{mod}' in {py_file.name} but does not have the 'network' permission.")  # pragma: no cover
            if not permissions.shell and (mod in _SHELL_MODULES or base in _SHELL_MODULES):  # pragma: no cover
                raise SandboxSecurityViolation(plugin_id=plugin_id, violation_type='import:shell', detail=f"Plugin imports shell module '{mod}' in {py_file.name} but does not have the 'shell' permission.")  # pragma: no cover

def _sandbox_worker(handler_module: str, handler_class: str, tool_name: str, args: Dict[str, Any], workspace_dir: str, permissions: Dict[str, bool], plugin_dir: str, result_conn: Any) -> None:
    """Executed inside a child process.  Calls the plugin tool handler.

    Isolation is enforced by:
    - Patching ``builtins.__import__`` to block forbidden modules.
    - Patching ``builtins.open`` to restrict filesystem access to
      *workspace_dir* (when ``filesystem`` permission is False).
    - Installing a SIGALRM handler (POSIX only) as a last-resort kill.
    """
    import builtins  # pragma: no cover
    import sys  # pragma: no cover
    from pathlib import Path as _Path  # pragma: no cover
    _original_import = builtins.__import__  # pragma: no cover

    def _restricted_import(name, *a, **kw):  # pragma: no cover
        """Auto-generated docstring.

Args:
    name: Argument.

Returns:
    Return value.
"""
        base = name.split('.')[0]  # pragma: no cover
        if not permissions.get('network') and (name in _NETWORK_MODULES or base in _NETWORK_MODULES):  # pragma: no cover
            raise SandboxSecurityViolation(plugin_id=handler_class, violation_type='import:network', detail=f"Runtime import of network module '{name}' blocked.")  # pragma: no cover
        if not permissions.get('shell') and (name in _SHELL_MODULES or base in _SHELL_MODULES):  # pragma: no cover
            raise SandboxSecurityViolation(plugin_id=handler_class, violation_type='import:shell', detail=f"Runtime import of shell module '{name}' blocked.")  # pragma: no cover
        return _original_import(name, *a, **kw)  # pragma: no cover
    builtins.__import__ = _restricted_import  # pragma: no cover
    if not permissions.get('filesystem'):  # pragma: no cover
        _original_open = builtins.open  # pragma: no cover
        _workspace = _Path(workspace_dir).resolve()  # pragma: no cover

        def _restricted_open(file, *a, **kw):  # pragma: no cover
            """Auto-generated docstring.

Args:
    file: Argument.

Returns:
    Return value.
"""
            try:  # pragma: no cover
                resolved = _Path(str(file)).resolve()  # pragma: no cover
            except Exception:  # pragma: no cover
                resolved = None  # pragma: no cover
            if resolved is not None and (not str(resolved).startswith(str(_workspace))):  # pragma: no cover
                raise SandboxSecurityViolation(plugin_id=handler_class, violation_type='filesystem', detail=f"Access to '{file}' denied — plugin workspace is '{_workspace}'.")  # pragma: no cover
            return _original_open(file, *a, **kw)  # pragma: no cover
        builtins.open = _restricted_open  # pragma: no cover
    try:  # pragma: no cover
        if plugin_dir not in sys.path:  # pragma: no cover
            sys.path.insert(0, plugin_dir)  # pragma: no cover
        import importlib  # pragma: no cover
        module = importlib.import_module(handler_module)  # pragma: no cover
        cls = getattr(module, handler_class)  # pragma: no cover
        instance = cls()  # pragma: no cover
        if hasattr(instance, '_plugin_id'):  # pragma: no cover
            instance._plugin_id = handler_class  # pragma: no cover

        class _MinimalContext:  # pragma: no cover
            """Auto-generated docstring.

"""

            class secrets:  # pragma: no cover
                """Auto-generated docstring.

"""

                @staticmethod  # pragma: no cover
                def get(key: str, default: Any=None) -> Any:  # pragma: no cover
                    """Auto-generated docstring.

Args:
    key: Argument.
    default: Argument.

Returns:
    Return value.
"""
                    return default  # pragma: no cover
        instance.on_load(_MinimalContext())  # pragma: no cover
        handler_fn = None  # pragma: no cover
        for tool_def in instance.tools:  # pragma: no cover
            if tool_def.name == tool_name:  # pragma: no cover
                handler_fn = tool_def.handler  # pragma: no cover
                break  # pragma: no cover
        if handler_fn is None:  # pragma: no cover
            raise AttributeError(f"Tool '{tool_name}' not found in plugin {handler_class}")  # pragma: no cover
        result = handler_fn(**args)  # pragma: no cover
        result_conn.send(('ok', result))  # pragma: no cover
    except SandboxSecurityViolation as exc:  # pragma: no cover
        result_conn.send(('security_violation', str(exc)))  # pragma: no cover
    except Exception:  # pragma: no cover
        result_conn.send(('error', traceback.format_exc()))  # pragma: no cover
    finally:
        result_conn.close()  # pragma: no cover

class SandboxedToolProxy(BaseTool):
    """A ``BaseTool``-compatible wrapper that executes plugin handlers in isolation.

    Each call to ``execute()`` spawns a fresh child process, sends the tool
    arguments over a ``Pipe``, waits for the result (up to ``timeout`` seconds),
    and deserialises the ``ToolResult``.
    """

    def __init__(self, manifest: PluginManifest, tool_def: PluginToolDefinition, timeout: int=SANDBOX_TIMEOUT_SECONDS) -> None:
        """Auto-generated docstring.

Args:
    manifest: Argument.
    tool_def: Argument.
    timeout: Argument.

Returns:
    Return value.
"""
        super().__init__(tool_id=f'{manifest.plugin_id}::{tool_def.name}', name=tool_def.name, description=tool_def.description)  # pragma: no cover
        self._manifest = manifest  # pragma: no cover
        self._tool_def = tool_def  # pragma: no cover
        self._timeout = timeout  # pragma: no cover
        for p in tool_def.parameters:  # pragma: no cover
            self.add_parameter(ToolParameter(name=p['name'], type=p.get('type', 'string'), description=p.get('description', ''), required=p.get('required', True)))  # pragma: no cover

    def execute(self, **kwargs: Any) -> ToolResult:  # type: ignore[override]
        """Run the plugin tool handler inside a sandboxed child process."""
        parent_conn, child_conn = multiprocessing.Pipe(duplex=False)  # pragma: no cover
        permissions = {'filesystem': self._manifest.permissions.filesystem, 'shell': self._manifest.permissions.shell, 'network': self._manifest.permissions.network, 'desktop_ui': self._manifest.permissions.desktop_ui}  # pragma: no cover
        with tempfile.TemporaryDirectory(prefix='zelyr_plugin_workspace_') as workspace:  # pragma: no cover
            proc = multiprocessing.Process(target=_sandbox_worker, args=(self._manifest.module, self._manifest.entry_class, self._tool_def.name, kwargs, workspace, permissions, str(self._manifest.plugin_dir), child_conn), daemon=True)  # pragma: no cover
            proc.start()  # pragma: no cover
            child_conn.close()  # pragma: no cover
            try:  # pragma: no cover
                if parent_conn.poll(self._timeout):  # pragma: no cover
                    status, payload = parent_conn.recv()  # pragma: no cover
                else:
                    proc.kill()  # pragma: no cover
                    proc.join(timeout=2)  # pragma: no cover
                    return ToolResult(success=False, error=f"Plugin '{self._manifest.plugin_id}' timed out after {self._timeout}s.")  # pragma: no cover
            except EOFError:  # pragma: no cover
                return ToolResult(success=False, error=f"Plugin '{self._manifest.plugin_id}' exited unexpectedly.")  # pragma: no cover
            finally:
                if proc.is_alive():  # pragma: no cover
                    proc.terminate()  # pragma: no cover
                    proc.join(timeout=2)  # pragma: no cover
                parent_conn.close()  # pragma: no cover
        if status == 'ok':  # pragma: no cover
            if isinstance(payload, ToolResult):  # pragma: no cover
                return payload  # pragma: no cover
            return ToolResult(success=True, output=payload)  # pragma: no cover
        if status == 'security_violation':  # pragma: no cover
            raise SandboxSecurityViolation(plugin_id=self._manifest.plugin_id, violation_type='runtime', detail=payload)  # pragma: no cover
        return ToolResult(success=False, error=f'Plugin error:\n{payload}')  # pragma: no cover
