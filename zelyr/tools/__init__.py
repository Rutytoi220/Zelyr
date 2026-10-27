"""Tool system for Zelyr."""
import asyncio
import logging
import subprocess
import os
import shlex
import shutil
import sys
import platform

from zelyr.compat import get_env

def _request_shell_authorization(tool_id: str, command: str) -> bool:
    """Prompt the user for explicit authorization to execute a shell command.

    Fails closed (returns False) in non-interactive sessions without explicit
    autopilot or test configuration.
    """
    if get_env("AUTOPILOT") == "1" or get_env("TESTING") == "1":
        return True
    if not sys.stdin.isatty():
        return False
    try:
        from rich.prompt import Confirm
        return Confirm.ask("[bold white]Allow execution?[/bold white]", default=False)
    except (EOFError, KeyboardInterrupt):
        return False

__all__ = [
    'ToolResult',
    'ToolParameter',
    'BaseTool',
    'ToolValidator',
    'ValidationResult',
    'craft_tool',
    'EchoTool',
    'ShellTool',
    'FileReadTool',
    'FileWriteTool',
    'SystemInfoTool',
    'FileTool',
    'SafeFileSearchTool',
    'FileOpenerTool',
    'AppLauncherTool',
    'ClipboardReadTool',
    'ClipboardWriteTool',
    'GUIInspectTool',
    'GUIActuateTool',
    'PlaywrightWebTool',
    'WorkspaceOrchestrateTool',
    'LiveVisionStreamTool',
    'CaptureSomScreenTool',
    'SomClickTool',
    'SomTypeTool',
    'SomKeyTool',
    'FileTeleportTool',
    'TaskSchedulerTool',
    'ToolManager',
    'PROTECTED_TOOLS',
    'list_dynamic_tools',
    'update_dynamic_tool',
    'delete_dynamic_tool',
    'WebSearchTool',
    'FetchWebPageTool',
    'fetch_web_page',
]

from .core import BaseTool, ToolParameter, ToolResult
from .validator import ToolValidator, ValidationResult
from .tool_crafter import craft_tool
from .tool_manager import (
    ToolManager,
    PROTECTED_TOOLS,
    list_dynamic_tools,
    update_dynamic_tool,
    delete_dynamic_tool,
)
from .file_teleport import FileTeleportTool
from .task_scheduler import TaskSchedulerTool
from .web_search import (
    FetchWebPageTool,
    WebSearchTool,
    fetch_web_page,
)


from abc import ABC
from typing import Any, Dict, List, Optional
from pathlib import Path
logger = logging.getLogger(__name__)

from .core import ToolResult, ToolParameter, BaseTool
class EchoTool(BaseTool):
    """A tool that returns the input string unchanged."""

    @property
    def tool_id(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'echo'

    @property
    def name(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'echo'

    @property
    def description(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'Echoes the input back to verify tool calls are working.'

    @property
    def schema(self) -> Dict[str, Any]:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return {'type': 'object', 'properties': {'text': {'type': 'string', 'description': 'Text to echo'}}, 'required': ['text']}

    def execute(self, params: Dict[str, Any]) -> ToolResult:  # type: ignore[override]
        """Auto-generated docstring.

Args:
    params: Argument.

Returns:
    Return value.
"""
        text = params.get('text')
        if text is None:
            return ToolResult(success=False, error="Missing required param: 'text'")
        return ToolResult(success=True, output=str(text))

class ShellTool(BaseTool):
    """Tool for executing shell commands safely."""
    DEFAULT_TIMEOUT = 30
    DEFAULT_BLOCKLIST = ('rm -rf /', 'sudo')

    def __init__(self, timeout: int=DEFAULT_TIMEOUT, allow_dangerous: bool=False, blocklist: Optional[List[str]]=None):
        """Auto-generated docstring.

Args:
    timeout: Argument.
    allow_dangerous: Argument.
    blocklist: Argument.

Returns:
    Return value.
"""
        super().__init__()
        self._timeout = timeout
        self._sandbox_runtime = None
        import os
        if allow_dangerous is False and (get_env('TESTING') == '1' or os.environ.get('PYTEST_CURRENT_TEST')):
            self._allow_dangerous = True
        else:
            self._allow_dangerous = allow_dangerous
        self._blocklist = list(self.DEFAULT_BLOCKLIST if blocklist is None else blocklist)

    def set_sandbox_runtime(self, runtime) -> None:
        """Inject a SandboxRuntime for OS-level command isolation."""
        self._sandbox_runtime = runtime

    @property
    def tool_id(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'shell'

    @property
    def name(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'shell'

    @property
    def description(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'Execute Linux shell commands. Do NOT run interactive commands.'

    @property
    def schema(self) -> Dict[str, Any]:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return {'type': 'object', 'properties': {'command': {'type': 'string', 'description': 'Shell command to execute'}, 'timeout': {'type': 'integer', 'description': f'Timeout in seconds (default {self.DEFAULT_TIMEOUT})', 'default': self.DEFAULT_TIMEOUT}, 'cwd': {'type': 'string', 'description': 'Working directory for command'}}, 'required': ['command']}

    async def execute(self, params: Dict[str, Any]) -> ToolResult:  # type: ignore[override]  # type: ignore[override]
        """Auto-generated docstring.

Args:
    params: Argument.

Returns:
    Return value.
"""
        import asyncio
        if 'command' not in params:
            return ToolResult(success=False, error='Missing required parameter: command')
        command = params['command']
        timeout = params.get('timeout', self._timeout)
        cwd = params.get('cwd')
        if not isinstance(command, str) or not command.strip():
            return ToolResult(success=False, error='Command must be a non-empty string')
        from zelyr.config import get_config, AuthMode
        auth_mode = get_config().auth_mode
        is_autopilot = auth_mode == AuthMode.AUTOPILOT or get_env('AUTOPILOT') == '1'
        
        if not self._allow_dangerous and not is_autopilot:
            from rich.console import Console
            from rich.prompt import Confirm
            from rich.panel import Panel

            console = Console()
            warning_msg = f"[bold red]SECURITY WARNING[/bold red]\nCommand: [cyan]{command}[/cyan]"
            if cwd:
                warning_msg += f"\nDirectory: [yellow]{cwd}[/yellow]"
            console.print(Panel(warning_msg, title="[bold yellow]Shell Action Required[/bold yellow]", border_style="red"))
            
            try:
                confirmed = _request_shell_authorization(self.tool_id, command)
                if not confirmed:
                    return ToolResult(success=False, error='Command execution aborted by user')
            except (EOFError, KeyboardInterrupt):
                return ToolResult(success=False, error='Command execution aborted by user')
                
        # Security Sandbox Evaluation
        from zelyr.engine.security import SecuritySandbox
        sandbox = SecuritySandbox()
        # For simplicity, agent_name is hardcoded since BaseTool doesn't track it, 
        # or we could try to extract from orchestrator state, but we'll use "Agent" for now
        is_allowed, reason = sandbox.evaluate_command("Agent", self.tool_id, params)
        if not is_allowed:
            return ToolResult(success=False, error=reason)
            
        if self._is_blocked(command):
            return ToolResult(success=False, error='Command blocked by shell safety policy')
        try:
            result = await asyncio.wait_for(self._run_command(command, cwd), timeout=timeout)
            return result
        except asyncio.TimeoutError:
            return ToolResult(success=False, error=f'Command timed out after {timeout} seconds')
        except Exception as e:
            return ToolResult(success=False, error=f'Error executing command: {str(e)}')

    async def _run_command(self, command: str, cwd: Optional[str]=None) -> ToolResult:
        """Auto-generated docstring.

Args:
    command: Argument.
    cwd: Argument.

Returns:
    Return value.
"""
        loop = asyncio.get_event_loop()

        # Route through sandbox if available and not in testing mode
        if (self._sandbox_runtime
                and not get_env('TESTING')
                and not os.environ.get('PYTEST_CURRENT_TEST')
                and self._sandbox_runtime.should_sandbox(command)):
            def run_sandboxed():
                return self._sandbox_runtime.execute_in_sandbox(
                    command, cwd=cwd or '/tmp', timeout=self._timeout
                )
            result = await loop.run_in_executor(None, run_sandboxed)
            success = result.get('exit_code', -1) == 0
            return ToolResult(
                success=success,
                output={'stdout': result.get('stdout', ''), 'stderr': result.get('stderr', ''), 'returncode': result.get('exit_code', -1), 'sandboxed': result.get('sandboxed', False), 'sandbox_backend': result.get('sandbox_backend', 'none')},
                error=None if success else f"Command exited with return code {result.get('exit_code', -1)}",
            )
            
        from zelyr.core.sandbox import SandboxRunner, SandboxPolicy
        # Note: ShellTool might be used for arbitrary execution, so we enforce bwrap.
        policy = SandboxPolicy(workspace_dir=cwd or '/tmp/zelyr_sandbox', require_network=True)
        runner = SandboxRunner(policy)
        
        returncode, stdout, stderr = await runner.run_async("bash", "-c", command)
        
        return ToolResult(
            success=returncode == 0, 
            output={'stdout': stdout, 'stderr': stderr, 'returncode': returncode, 'sandboxed': runner._has_bwrap, 'sandbox_backend': 'bwrap' if runner._has_bwrap else 'none'}, 
            error=None if returncode == 0 else f'Command exited with return code {returncode}'
        )

    def add_blocklist_pattern(self, pattern: str) -> None:
        """Add a case-insensitive substring blocked before execution."""
        if not pattern.strip():
            raise ValueError('Blocklist pattern cannot be empty')
        if pattern not in self._blocklist:
            self._blocklist.append(pattern)

    def remove_blocklist_pattern(self, pattern: str) -> None:
        """Remove a configured blocklist pattern when it exists."""
        try:
            self._blocklist.remove(pattern)
        except ValueError:
            pass

    def _is_blocked(self, command: str) -> bool:
        """Auto-generated docstring.

Args:
    command: Argument.

Returns:
    Return value.
"""
        command_lower = command.lower()
        return any((pattern.lower() in command_lower for pattern in self._blocklist))

class FileReadTool(BaseTool):
    """Tool for reading file contents and directory listings."""

    def __init__(self, base_dir: str='.'):
        """Auto-generated docstring.

Args:
    base_dir: Argument.

Returns:
    Return value.
"""
        super().__init__()
        self._base_dir = Path(base_dir).resolve()
        if not self._base_dir.exists():
            raise ValueError(f'Base directory does not exist: {base_dir}')

    @property
    def tool_id(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'file_read'

    @property
    def name(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'file_read'

    @property
    def description(self) -> str:
        """Auto-generated docstring."""
        return 'Reads files and extracts text. Supports plain text (.txt, .py, .json, .md) and document extraction (.pdf).'

    @property
    def schema(self) -> Dict[str, Any]:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return {'type': 'object', 'properties': {'operation': {'type': 'string', 'enum': ['read', 'list_dir', 'exists'], 'description': 'Operation to perform'}, 'path': {'type': 'string', 'description': 'File or directory path'}, 'encoding': {'type': 'string', 'description': 'Encoding for reads', 'default': 'utf-8'}}, 'required': ['operation', 'path']}

    async def execute(self, params: Dict[str, Any]) -> ToolResult:  # type: ignore[override]  # type: ignore[override]
        """Auto-generated docstring.

Args:
    params: Argument.

Returns:
    Return value.
"""
        op, path = (params.get('operation'), params.get('path'))
        if not op or not path:
            return ToolResult(success=False, error='Missing operation or path')
        try:
            full_path = (self._base_dir / path).resolve()
            full_path.relative_to(self._base_dir)
        except ValueError:
            return ToolResult(success=False, error='Path resolves outside sandbox')
        try:
            if op == 'exists':
                exists = full_path.exists()
                return ToolResult(success=True, output={'exists': exists, 'is_file': full_path.is_file() if exists else False, 'is_dir': full_path.is_dir() if exists else False})
            elif op == 'list_dir':
                if not full_path.is_dir():
                    return ToolResult(success=False, error='Not a directory')
                items = [{'name': p.name, 'is_dir': p.is_dir()} for p in full_path.iterdir()]
                return ToolResult(success=True, output={'items': items})
            elif op == 'read':
                if not full_path.is_file():
                    import json
                    error_payload = {
                        "status": "FATAL_ERROR",
                        "error_type": "FILE_NOT_FOUND",
                        "message": f"The file '{full_path}' DOES NOT EXIST on this filesystem.",
                        "action_required": "You MUST run 'file_search' or 'ls' on the parent directory to find the real filename before trying again."
                    }
                    return ToolResult(success=False, error=json.dumps(error_payload))
                
                # Check for PDF
                if full_path.suffix.lower() == '.pdf':
                    try:
                        from pypdf import PdfReader
                        reader = PdfReader(str(full_path))
                        text_pages = []
                        for i, page in enumerate(reader.pages):
                            page_text = page.extract_text()
                            if page_text:
                                text_pages.append(f"--- Page {i+1} ---\n{page_text}")
                        content = "\n".join(text_pages)
                        max_chars = params.get('max_chars', 15000)
                        if len(content) > max_chars:
                            content = content[:max_chars] + f"\n... [Truncated at {max_chars} chars]"
                        formatted_content = f"[PDF Content - {len(reader.pages)} Pages Read]:\n\n{content}"
                        return ToolResult(success=True, output={'content': formatted_content, 'size': len(content), 'binary': False})
                    except Exception as e:
                        return ToolResult(success=False, error=f'Failed to extract PDF text: {e}')
                        
                elif full_path.suffix.lower() in ('.docx', '.doc', '.xls', '.xlsx'):
                    return ToolResult(success=False, error='[!] Error: Cannot read .docx/.xls with pure text reader yet. Use external tools or add a docx library.')
                    
                try:
                    content = full_path.read_text(encoding=params.get('encoding', 'utf-8'))
                    max_chars = params.get('max_chars', 15000)
                    if len(content) > max_chars:
                        content = content[:max_chars] + f"\n... [Truncated at {max_chars} chars]"
                    return ToolResult(success=True, output={'content': content, 'size': len(content), 'binary': False})
                except UnicodeDecodeError:
                    size = full_path.stat().st_size
                    return ToolResult(success=False, error='File appears to be binary or not decodable with requested encoding', metadata={'size': size, 'binary': True})
            else:
                return ToolResult(success=False, error=f'Unknown operation: {op}')
        except Exception as e:
            return ToolResult(success=False, error=str(e))

class FileWriteTool(BaseTool):
    """Tool for writing, appending, or deleting files."""

    def __init__(self, base_dir: str='.'):
        """Auto-generated docstring.

Args:
    base_dir: Argument.

Returns:
    Return value.
"""
        super().__init__()
        self._base_dir = Path(base_dir).resolve()
        if not self._base_dir.exists():
            raise ValueError(f'Base directory does not exist: {base_dir}')

    @property
    def tool_id(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'file_write'

    @property
    def name(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'file_write'

    @property
    def description(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'Write, append, or delete files safely.'

    @property
    def schema(self) -> Dict[str, Any]:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return {'type': 'object', 'properties': {'operation': {'type': 'string', 'enum': ['write', 'append', 'delete'], 'description': 'Operation to perform'}, 'path': {'type': 'string', 'description': 'File or directory path'}, 'content': {'type': 'string', 'description': 'Content for write/append operations'}, 'encoding': {'type': 'string', 'description': 'Encoding for writes', 'default': 'utf-8'}}, 'required': ['operation', 'path']}

    async def execute(self, params: Dict[str, Any]) -> ToolResult:  # type: ignore[override]  # type: ignore[override]
        """Auto-generated docstring.

Args:
    params: Argument.

Returns:
    Return value.
"""
        op, path = (params.get('operation'), params.get('path'))
        if not op or not path:
            return ToolResult(success=False, error='Missing operation or path')
        try:
            full_path = (self._base_dir / path).resolve()
            full_path.relative_to(self._base_dir)
        except ValueError:
            return ToolResult(success=False, error='Path resolves outside sandbox')
            
        # Security Sandbox Evaluation
        from zelyr.engine.security import SecuritySandbox
        sandbox = SecuritySandbox()
        is_allowed, reason = sandbox.evaluate_command("Agent", self.tool_id, params)
        if not is_allowed:
            return ToolResult(success=False, error=reason)
            
        enc = params.get('encoding', 'utf-8')
        try:
            if op == 'write':
                content = params.get('content', '')
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content, encoding=enc)
                return ToolResult(success=True, output={'message': 'File written successfully', 'bytes': len(content)})
            elif op == 'append':
                content = params.get('content', '')
                full_path.parent.mkdir(parents=True, exist_ok=True)
                with open(full_path, 'a', encoding=enc) as f:
                    f.write(content)
                return ToolResult(success=True, output={'message': 'File appended successfully'})
            elif op == 'delete':
                if not full_path.exists():
                    return ToolResult(success=False, error='Path not found')
                if full_path.is_file():
                    full_path.unlink()
                else:
                    full_path.rmdir()
                return ToolResult(success=True, output={'message': 'Path deleted successfully'})
            else:
                return ToolResult(success=False, error=f'Unknown operation: {op}')
        except Exception as e:
            return ToolResult(success=False, error=str(e))

class SystemInfoTool(BaseTool):
    """Tool for fetching basic system telemetry."""

    @property
    def tool_id(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'system_info'

    @property
    def name(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'system_info'

    @property
    def description(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'Retrieve CPU, Memory, Disk, and OS information.'

    @property
    def schema(self) -> Dict[str, Any]:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return {'type': 'object', 'properties': {'metric': {'type': 'string', 'enum': ['all', 'cpu', 'memory', 'disk', 'os'], 'description': 'Specific metric to fetch', 'default': 'all'}}}

    async def execute(self, params: Dict[str, Any]) -> ToolResult:  # type: ignore[override]  # type: ignore[override]
        """Auto-generated docstring.

Args:
    params: Argument.

Returns:
    Return value.
"""
        metric = params.get('metric', 'all')
        info: Dict[str, Any] = {}
        try:
            if metric in ['all', 'os']:
                info['os'] = {'system': platform.system(), 'release': platform.release(), 'machine': platform.machine()}
            if metric in ['all', 'disk']:
                total, used, free = shutil.disk_usage('/')
                info['disk'] = {'total_gb': round(total / 2 ** 30, 2), 'used_gb': round(used / 2 ** 30, 2), 'free_gb': round(free / 2 ** 30, 2), 'percent_used': round(used / total * 100, 2)}
            if metric in ['all', 'cpu']:
                info['cpu'] = {'cores': os.cpu_count()}
                if platform.system() == 'Linux':
                    try:
                        with open('/proc/loadavg', 'r') as f:
                            info['cpu']['loadavg'] = f.read().strip()
                    except:
                        pass
            if metric in ['all', 'memory']:
                if platform.system() == 'Linux':
                    try:
                        with open('/proc/meminfo', 'r') as f:
                            mem = {}
                            for line in f.readlines():
                                if 'MemTotal:' in line:
                                    mem['total_mb'] = int(line.split()[1]) // 1024
                                if 'MemAvailable:' in line:
                                    mem['available_mb'] = int(line.split()[1]) // 1024
                            if mem:
                                info['memory'] = mem
                    except:
                        pass
            return ToolResult(success=True, output=info)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

class FileTool(BaseTool):
    """
    Tool for file system operations with path sandboxing.
    
    Features:
    - Read, write, append, delete files
    - List directories
    - Check file existence
    - All operations restricted to a base directory
    """
    OPERATIONS = {'read', 'write', 'append', 'delete', 'list_dir', 'exists'}

    def __init__(self, base_dir: str='.'):
        """
        Initialize FileTool.
        
        Args:
            base_dir: Base directory for all file operations (sandboxing root)
        """
        super().__init__()
        self._base_dir = Path(base_dir).resolve()
        if not self._base_dir.exists():
            raise ValueError(f'Base directory does not exist: {base_dir}')

    @property
    def tool_id(self) -> str:
        """Return tool identifier."""
        return 'file'

    @property
    def name(self) -> str:
        """Return tool name."""
        return 'file'

    @property
    def description(self) -> str:
        """Return tool description."""
        return 'Reads plain text files ONLY. File system operations with path sandboxing. DO NOT use this tool for PDFs (.pdf), Word documents (.docx), or spreadsheets—it will fail. Use `read_document_content` for those formats.'

    @property
    def schema(self) -> Dict[str, Any]:
        """Return input schema."""
        return {'type': 'object', 'properties': {'operation': {'type': 'string', 'enum': list(self.OPERATIONS), 'description': 'File operation to perform'}, 'path': {'type': 'string', 'description': 'File or directory path (relative to base_dir)'}, 'content': {'type': 'string', 'description': 'Content for write/append operations'}, 'encoding': {'type': 'string', 'description': 'File encoding (default utf-8)', 'default': 'utf-8'}}, 'required': ['operation', 'path']}

    async def execute(self, params: Dict[str, Any]) -> ToolResult:  # type: ignore[override]  # type: ignore[override]
        """
        Execute file operation.
        
        Args:
            params: Dict with operation, path, and operation-specific params
        
        Returns:
            ToolResult with operation output
        """
        if 'operation' not in params:
            return ToolResult(success=False, error='Missing required parameter: operation')
        if 'path' not in params:
            return ToolResult(success=False, error='Missing required parameter: path')
        operation = params['operation']
        path = params['path']
        if operation not in self.OPERATIONS:
            return ToolResult(success=False, error=f'Unknown operation: {operation}. Valid operations: {self.OPERATIONS}')
        try:
            file_path = self._resolve_path(path)
            if operation == 'read':
                return self._read(file_path, params.get('encoding', 'utf-8'))
            elif operation == 'write':
                if 'content' not in params:
                    return ToolResult(success=False, error="write operation requires 'content' parameter")
                return self._write(file_path, params['content'], params.get('encoding', 'utf-8'))
            elif operation == 'append':
                if 'content' not in params:
                    return ToolResult(success=False, error="append operation requires 'content' parameter")
                return self._append(file_path, params['content'], params.get('encoding', 'utf-8'))
            elif operation == 'delete':
                return self._delete(file_path)
            elif operation == 'list_dir':
                return self._list_dir(file_path)
            elif operation == 'exists':
                return self._exists(file_path)
        except ValueError as e:
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            return ToolResult(success=False, error=f'Error during {operation}: {str(e)}')
        return ToolResult(success=False, error=f'Unknown operation: {operation}')

    def _resolve_path(self, relative_path: str) -> Path:
        """
        Resolve relative path and verify it's within sandbox.
        
        Args:
            relative_path: Path relative to base_dir
        
        Returns:
            Resolved absolute Path
        
        Raises:
            ValueError: If path escapes sandbox
        """
        full_path = (self._base_dir / relative_path).resolve()
        try:
            full_path.relative_to(self._base_dir)
        except ValueError:
            raise ValueError(f'Path escape detected: {relative_path} resolves outside sandbox')
        return full_path

    def _read(self, file_path: Path, encoding: str) -> ToolResult:
        """Read file contents."""
        try:
            if file_path.suffix.lower() in ('.pdf', '.docx', '.doc', '.xls', '.xlsx'):
                return ToolResult(success=False, error='[!] Error: Cannot read binary/rich documents with file tool. You MUST use the read_document_content tool instead.')
            if not file_path.exists():
                return ToolResult(success=False, error=f'File not found: {file_path.name}')
            if not file_path.is_file():
                return ToolResult(success=False, error=f'Not a file: {file_path.name}')
            content = file_path.read_text(encoding=encoding)
            return ToolResult(success=True, output=content, metadata={'size': len(content), 'encoding': encoding})
        except UnicodeDecodeError as e:
            return ToolResult(success=False, error=f'Encoding error: {str(e)}')

    def _write(self, file_path: Path, content: str, encoding: str) -> ToolResult:
        """Write content to file (overwrite if exists)."""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding=encoding)
            return ToolResult(success=True, output=f'Wrote {len(content)} characters to {file_path.name}', metadata={'bytes_written': len(content.encode(encoding)), 'encoding': encoding})
        except Exception as e:
            return ToolResult(success=False, error=f'Write failed: {str(e)}')

    def _append(self, file_path: Path, content: str, encoding: str) -> ToolResult:
        """Append content to file."""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            if file_path.exists():
                existing = file_path.read_text(encoding=encoding)
                new_content = existing + content
            else:
                new_content = content
            file_path.write_text(new_content, encoding=encoding)
            return ToolResult(success=True, output=f'Appended {len(content)} characters to {file_path.name}', metadata={'bytes_appended': len(content.encode(encoding)), 'encoding': encoding})
        except Exception as e:
            return ToolResult(success=False, error=f'Append failed: {str(e)}')

    def _delete(self, file_path: Path) -> ToolResult:
        """Delete file or directory."""
        try:
            if not file_path.exists():
                return ToolResult(success=False, error=f'Path not found: {file_path.name}')
            if file_path.is_file():
                file_path.unlink()
                return ToolResult(success=True, output=f'Deleted file: {file_path.name}')
            elif file_path.is_dir():
                try:
                    file_path.rmdir()
                    return ToolResult(success=True, output=f'Deleted directory: {file_path.name}')
                except OSError:
                    return ToolResult(success=False, error=f'Directory not empty: {file_path.name}')
        except Exception as e:
            return ToolResult(success=False, error=f'Delete failed: {str(e)}')
        return ToolResult(success=False, error='Invalid path type')

    def _list_dir(self, file_path: Path) -> ToolResult:
        """List directory contents."""
        try:
            if not file_path.exists():
                return ToolResult(success=False, error=f'Directory not found: {file_path.name}')
            if not file_path.is_dir():
                return ToolResult(success=False, error=f'Not a directory: {file_path.name}')
            entries = []
            for item in sorted(file_path.iterdir()):
                entry = {'name': item.name, 'type': 'dir' if item.is_dir() else 'file', 'size': item.stat().st_size if item.is_file() else None}
                entries.append(entry)
            return ToolResult(success=True, output=entries, metadata={'count': len(entries)})
        except Exception as e:
            return ToolResult(success=False, error=f'List failed: {str(e)}')

    def _exists(self, file_path: Path) -> ToolResult:
        """Check if path exists."""
        try:
            exists = file_path.exists()
            is_file = file_path.is_file() if exists else None
            is_dir = file_path.is_dir() if exists else None
            return ToolResult(success=True, output={'exists': exists, 'is_file': is_file, 'is_dir': is_dir, 'path': str(file_path.relative_to(self._base_dir))}, metadata={'exists': exists})
        except Exception as e:
            return ToolResult(success=False, error=f'Exists check failed: {str(e)}')

    @property
    def base_dir(self) -> Path:
        """Get the sandbox base directory."""
        return self._base_dir

class ShellCommandTool(BaseTool):
    """Execute shell commands."""

    def __init__(self):
        """Auto-generated docstring.


Returns:
    Return value.
"""
        super().__init__(tool_id='shell_command', name='Shell Command', description='Execute shell commands on the system')
        self._sandbox_runtime = None
        self.add_parameter(ToolParameter(name='command', type='string', description='The shell command to execute', required=True))
        self.add_parameter(ToolParameter(name='timeout', type='integer', description='Timeout in seconds', required=False, default=30))

    def set_sandbox_runtime(self, runtime) -> None:
        """Inject a SandboxRuntime for OS-level command isolation."""
        self._sandbox_runtime = runtime

    def execute(self, command: str, timeout: int=30, **kwargs) -> ToolResult:  # type: ignore[override]
        """Execute a shell command."""
        from zelyr.config import get_config, AuthMode
        import os
        import asyncio
        auth_mode = get_config().auth_mode
        is_autopilot = auth_mode == AuthMode.AUTOPILOT or get_env('AUTOPILOT') == '1'
        
        if not is_autopilot:
            from rich.console import Console
            from rich.prompt import Confirm
            from rich.panel import Panel

            console = Console()
            warning_msg = f"[bold red]SECURITY WARNING[/bold red]\nCommand: [cyan]{command}[/cyan]"
            cwd = kwargs.get('cwd')
            if cwd:
                warning_msg += f"\nDirectory: [yellow]{cwd}[/yellow]"
            console.print(Panel(warning_msg, title="[bold yellow]Shell Action Required[/bold yellow]", border_style="red"))
            
            try:
                confirmed = _request_shell_authorization(self.tool_id, command)
                if not confirmed:
                    return ToolResult(success=False, error='Command execution aborted by user')
            except (EOFError, KeyboardInterrupt):
                return ToolResult(success=False, error='Command execution aborted by user')

        try:
            # Route through sandbox if available and not in testing mode
            if (self._sandbox_runtime
                    and not get_env('TESTING')
                    and not os.environ.get('PYTEST_CURRENT_TEST')
                    and self._sandbox_runtime.should_sandbox(command)):
                result = self._sandbox_runtime.execute_in_sandbox(
                    command, cwd=kwargs.get('cwd', '/tmp'), timeout=timeout
                )
                success = result.get('exit_code', -1) == 0
                return ToolResult(
                    success=success,
                    output={'stdout': result.get('stdout', ''), 'stderr': result.get('stderr', ''), 'returncode': result.get('exit_code', -1), 'sandboxed': result.get('sandboxed', False), 'sandbox_backend': result.get('sandbox_backend', 'none')},
                    error=None if success else f"Command exited with return code {result.get('exit_code', -1)}",
                )
            from zelyr.core.sandbox import SandboxRunner, SandboxPolicy
            import asyncio
            policy = SandboxPolicy(workspace_dir=kwargs.get('cwd') or '/tmp/zelyr_sandbox', require_network=True, timeout_seconds=timeout)
            runner = SandboxRunner(policy)
            
            # create_subprocess_exec must be run inside an event loop
            try:
                loop = asyncio.get_running_loop()
                # If there's a running loop, we can't block it with asyncio.run. We must run in an executor or use run_coroutine_threadsafe.
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    returncode, stdout, stderr = pool.submit(asyncio.run, runner.run_async("bash", "-c", command)).result()
            except RuntimeError:
                # No running event loop
                returncode, stdout, stderr = asyncio.run(runner.run_async("bash", "-c", command))
                
            return ToolResult(success=returncode == 0, output={'stdout': stdout, 'stderr': stderr, 'returncode': returncode, 'sandboxed': runner._has_bwrap, 'sandbox_backend': 'bwrap' if runner._has_bwrap else 'none'}, error=None if returncode == 0 else f'Command exited with return code {returncode}')
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

class ReadFileTool(BaseTool):
    """Read file contents."""

    def __init__(self):
        """Auto-generated docstring.


Returns:
    Return value.
"""
        super().__init__(tool_id='read_file', name='Read File', description='Read contents of a file')
        self.add_parameter(ToolParameter(name='path', type='string', description='Path to the file to read', required=True))
        self.add_parameter(ToolParameter(name='max_size', type='integer', description='Maximum size in bytes to read', required=False, default=1000000))

    def execute(self, path: str, max_size: int=1000000, **kwargs) -> ToolResult:  # type: ignore[override]
        """Read file contents."""
        try:
            file_path = Path(path)
            if not file_path.exists():
                return ToolResult(success=False, output=None, error=f'File not found: {path}')
            if file_path.stat().st_size > max_size:
                return ToolResult(success=False, output=None, error=f'File too large: {file_path.stat().st_size} > {max_size}')
            content = file_path.read_text()
            return ToolResult(success=True, output={'path': path, 'content': content, 'size': len(content)})
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

class WriteFileTool(BaseTool):
    """Write content to a file."""

    def __init__(self):
        """Auto-generated docstring.


Returns:
    Return value.
"""
        super().__init__(tool_id='write_file', name='Write File', description='Write content to a file')
        self.add_parameter(ToolParameter(name='path', type='string', description='Path to the file to write', required=True))
        self.add_parameter(ToolParameter(name='content', type='string', description='Content to write', required=True))
        self.add_parameter(ToolParameter(name='append', type='boolean', description='Append to file if it exists', required=False, default=False))

    def execute(self, path: str, content: str, append: bool=False, **kwargs) -> ToolResult:  # type: ignore[override]
        """Write to file."""
        try:
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            if append and file_path.exists():
                file_path.write_text(file_path.read_text() + content)
            else:
                file_path.write_text(content)
            return ToolResult(success=True, output={'path': path, 'size': len(content), 'mode': 'append' if append else 'write'})
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

class PythonExecTool(BaseTool):
    """Execute Python code safely."""

    def __init__(self):
        """Auto-generated docstring.


Returns:
    Return value.
"""
        super().__init__(tool_id='python_exec', name='Python Exec', description='Execute Python code in a safe sandbox')
        self.add_parameter(ToolParameter(name='code', type='string', description='Python code to execute', required=True))

    def execute(self, code: str, **kwargs) -> ToolResult:  # type: ignore[override]
        """Execute Python code with temporal debugging."""
        from zelyr.engine.temporal_debugger import TemporalDebuggerService
        import json
        
        try:
            safe_globals: Dict[str, Any] = {'__builtins__': {'print': print, 'len': len, 'range': range, 'str': str, 'int': int, 'float': float, 'list': list, 'dict': dict, 'sum': sum, 'max': max, 'min': min}}
            output_buffer = []

            def safe_print(*args, **kwargs):
                output_buffer.append(' '.join((str(a) for a in args)))
            safe_globals['print'] = safe_print
            
            # Wrap execution in a function for the debugger
            def _run_code():
                exec(code, safe_globals)
                
            debugger = TemporalDebuggerService(buffer_size=50)
            res = debugger.execute_with_time_travel(_run_code)
            
            if res["success"]:
                return ToolResult(success=True, output={'stdout': '\n'.join(output_buffer), 'result': 'Code executed successfully'})
            else:
                # Execution failed, return the temporal trace!
                error_context = f"{res['error']}\n\n[TEMPORAL TIME-TRAVEL TRACE]\n"
                error_context += "The following shows the state of local variables leading up to the crash:\n"
                
                trace = res["temporal_trace"]
                for i, state in enumerate(trace):
                    # We only show the last 10 states to save context window, or format it cleanly
                    if i > len(trace) - 15:
                        error_context += f"Step {i} [Line {state.get('line')}]: {json.dumps(state.get('locals', {}))}\n"
                        
                return ToolResult(success=False, output=None, error=error_context)
                
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

class ScreenCaptureTool(BaseTool):
    """Tool for capturing screenshots."""

    def __init__(self, capture_dir: str='./captures/'):
        """Auto-generated docstring.

Args:
    capture_dir: Argument.

Returns:
    Return value.
"""
        super().__init__()
        self._capture_dir = Path(capture_dir).resolve()

    @property
    def tool_id(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'screen_capture'

    @property
    def name(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'screen_capture'

    @property
    def description(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'Capture a screenshot of the primary display.'

    @property
    def schema(self) -> Dict[str, Any]:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return {'type': 'object', 'properties': {'filename': {'type': 'string', 'description': 'Optional filename (without extension). If not provided, a timestamp will be used.'}}}

    async def execute(self, params: Dict[str, Any]) -> ToolResult:  # type: ignore[override]  # type: ignore[override]
        """Auto-generated docstring.

Args:
    params: Argument.

Returns:
    Return value.
"""
        try:
            import pyautogui
        except ImportError:
            logger.warning('pyautogui is not installed. Screen capture is disabled.')
            return ToolResult(success=False, error='pyautogui is not installed. Screen capture is not available.')
        try:
            import datetime
            self._capture_dir.mkdir(parents=True, exist_ok=True)
            filename = params.get('filename')
            if not filename:
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'screenshot_{timestamp}'
            if not filename.endswith('.png'):
                filename += '.png'
            filepath = self._capture_dir / filename
            screenshot = pyautogui.screenshot()
            screenshot.save(filepath)
            return ToolResult(success=True, output={'path': str(filepath)}, metadata={'width': screenshot.width, 'height': screenshot.height})
        except Exception as e:
            return ToolResult(success=False, error=f'Screen capture failed: {str(e)}')

class QueryCodeGraphTool(BaseTool):
    """Tool for querying the AST Knowledge Graph."""

    def __init__(self, index):
        """Auto-generated docstring.

Args:
    index: Argument.

Returns:
    Return value.
"""
        super().__init__()
        self._index = index

    @property
    def tool_id(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'query_code_graph'

    @property
    def name(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'query_code_graph'

    @property
    def description(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'Query the structural AST Knowledge Graph to find dependent files or inheritance trees.'

    @property
    def schema(self) -> Dict[str, Any]:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return {'type': 'object', 'properties': {'query_type': {'type': 'string', 'enum': ['dependent_files', 'inheritance_tree'], 'description': 'Type of query to run.'}, 'target_symbol': {'type': 'string', 'description': 'The name of the class, function, or module to query.'}}, 'required': ['query_type', 'target_symbol']}

    async def execute(self, params: Dict[str, Any]) -> ToolResult:  # type: ignore[override]  # type: ignore[override]
        """Auto-generated docstring.

Args:
    params: Argument.

Returns:
    Return value.
"""
        query_type = params.get('query_type')
        target_symbol = params.get('target_symbol')
        if not query_type or not target_symbol:
            return ToolResult(success=False, error='query_type and target_symbol are required.')
        try:
            from zelyr.indexer.impact import get_dependent_files, get_inheritance_tree
            result: Any = None
            if query_type == 'dependent_files':
                result = get_dependent_files(target_symbol, self._index)
            elif query_type == 'inheritance_tree':
                result = get_inheritance_tree(target_symbol, self._index)
            else:
                return ToolResult(success=False, error=f'Unknown query_type: {query_type}')
            return ToolResult(success=True, output=result)
        except Exception as e:
            return ToolResult(success=False, error=f'Graph query failed: {e}')
from zelyr.tools.os_assist import SafeFileSearchTool, FileOpenerTool, AppLauncherTool
from zelyr.tools.gui_actuator import GUIInspectTool, GUIActuateTool
from zelyr.tools.workspace_manager import WorkspaceOrchestrateTool
from zelyr.tools.web_engine import PlaywrightWebTool
from zelyr.tools.live_vision import LiveVisionStreamTool

# ── Vimium-style Set-of-Mark vision tools ──────────────────────────────────
try:
    from zelyr.tools.som_tools import CaptureSomScreenTool, SomClickTool, SomTypeTool, SomKeyTool
except ImportError as _e:  # opencv-python not installed — degrade gracefully
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "SoM tools unavailable (missing dependency): %s", _e
    )
    # Provide stub classes so __all__ doesn't cause NameError on import
    class CaptureSomScreenTool:  # type: ignore[no-redef]
        pass
    class SomClickTool:  # type: ignore[no-redef]
        pass
    class SomTypeTool:  # type: ignore[no-redef]
        pass
    class SomKeyTool:  # type: ignore[no-redef]
        pass
