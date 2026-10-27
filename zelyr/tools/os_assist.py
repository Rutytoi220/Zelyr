"""Assistive OS tools for non-technical user workflows."""
import os
import platform
import subprocess
import mimetypes
from pathlib import Path
import io
import base64
from typing import Any, Dict, List
from zelyr.tools.core import BaseTool, ToolResult, ToolParameter

class SafeFileSearchTool(BaseTool):
    """Searches for files strictly within safe user directories."""
    SAFE_DIRS = ['Desktop', 'Documents', 'Downloads', 'Pictures']

    @property
    def tool_id(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'safe_file_search'

    @property
    def name(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'SafeFileSearchTool'

    @property
    def description(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'Find files safely in standard user directories (Desktop, Documents, etc.)'

    @property
    def schema(self) -> Dict[str, Any]:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return {'type': 'object', 'properties': {'query': {'type': 'string', 'description': 'Glob pattern to search (e.g., *tax*.pdf)'}, 'search_dir': {'type': 'string', 'description': "Specific safe dir (e.g., Documents) or 'user_default'", 'default': 'user_default'}}, 'required': ['query']}

    async def execute(self, params: Dict[str, Any]) -> ToolResult:  # type: ignore[override]  # type: ignore[override]
        """Auto-generated docstring.

Args:
    params: Argument.

Returns:
    Return value.
"""
        query = params.get('query')
        if not query:
            return ToolResult(success=False, error='Missing query')
        if '..' in query or query.startswith('/'):
            return ToolResult(success=False, error='Path traversal payloads are strictly blocked.')
        search_dir = params.get('search_dir', 'user_default')
        home = Path.home()
        dirs_to_search = []
        if search_dir == 'user_default':
            dirs_to_search = [home / d for d in self.SAFE_DIRS if (home / d).exists()]
        elif search_dir in self.SAFE_DIRS:
            target = home / search_dir
            if target.exists():
                dirs_to_search = [target]
            else:
                return ToolResult(success=False, error=f'Directory {search_dir} does not exist.')
        else:
            return ToolResult(success=False, error=f"Directory '{search_dir}' is not a permitted safe directory.")
        results: List[Dict[str, Any]] = []
        for d in dirs_to_search:
            try:
                for path in d.rglob(query):
                    if path.is_file():
                        stat = path.stat()
                        results.append({'name': path.name, 'path': str(path.absolute()), 'size': stat.st_size, 'mtime': stat.st_mtime})
                        if len(results) >= 50:
                            break
            except Exception as e:
                pass
            if len(results) >= 50:
                break
        results.sort(key=lambda x: float(str(x['mtime'])), reverse=True)
        return ToolResult(success=True, output=results)

class FileOpenerTool(BaseTool):
    """Safely opens a file with the default OS application and explains the file format."""

    @property
    def tool_id(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'file_opener'

    @property
    def name(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'FileOpenerTool'

    @property
    def description(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'Safely opens a file using the default desktop application.'

    @property
    def schema(self) -> Dict[str, Any]:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return {'type': 'object', 'properties': {'file_path': {'type': 'string', 'description': 'Absolute path to the file'}}, 'required': ['file_path']}

    async def execute(self, params: Dict[str, Any]) -> ToolResult:  # type: ignore[override]  # type: ignore[override]
        """Auto-generated docstring.

Args:
    params: Argument.

Returns:
    Return value.
"""
        file_path = params.get('file_path')
        if not file_path:
            return ToolResult(success=False, error='Missing file_path')
            
        from zelyr.tools import resolve_safe_path
        path = resolve_safe_path(file_path, Path.cwd())
        if not path.exists() or not path.is_file():
            import json
            error_payload = {
                "status": "FATAL_ERROR",
                "error_type": "FILE_NOT_FOUND",
                "message": f"The file '{path}' DOES NOT EXIST on this filesystem.",
                "action_required": "You MUST run 'file_search' or 'ls' on the parent directory to find the real filename before trying again."
            }
            return ToolResult(success=False, error=json.dumps(error_payload))
        try:
            if platform.system() == 'Windows':
                startfile = getattr(os, 'startfile')
                startfile(str(path))
                open_cmd = 'os.startfile'
            elif platform.system() == 'Darwin':
                subprocess.Popen(['open', str(path)], start_new_session=True)
                open_cmd = 'open'
            else:
                subprocess.Popen(['xdg-open', str(path)], start_new_session=True)
                open_cmd = 'xdg-open'
        except Exception as e:
            return ToolResult(success=False, error=f'Failed to open application: {e}')
        mtype, _ = mimetypes.guess_type(str(path))
        ext = path.suffix.lower()
        description = 'an unknown file format'
        if mtype:
            if mtype.startswith('image/'):
                description = 'a picture/image'
            elif mtype.startswith('video/'):
                description = 'a video'
            elif mtype.startswith('audio/'):
                description = 'an audio file'
            elif mtype == 'application/pdf':
                description = 'a PDF document'
            elif mtype == 'text/plain':
                description = 'a simple text document'
            elif mtype == 'text/html':
                description = 'a webpage document'
            elif 'zip' in mtype or 'archive' in mtype:
                description = 'a compressed archive'
        elif ext in ['.docx', '.doc']:
            description = 'a Microsoft Word document'
        elif ext in ['.xlsx', '.xls']:
            description = 'a spreadsheet'
        elif ext in ['.pptx', '.ppt']:
            description = 'a presentation'
        return ToolResult(success=True, output={'message': f'[✓] Successfully spawned {open_cmd} for {path}. This is {description}.'})

class AppLauncherTool(BaseTool):
    """Maps safe natural language concepts to default OS executables."""
    SAFE_APPS = {'browser': {'Windows': 'start https://', 'Darwin': "open -a 'Google Chrome'", 'Linux': 'xdg-open http://'}, 'calculator': {'Windows': 'calc.exe', 'Darwin': 'open -a Calculator', 'Linux': 'gnome-calculator'}, 'terminal': {'Windows': 'cmd.exe', 'Darwin': 'open -a Terminal', 'Linux': 'gnome-terminal'}, 'file_explorer': {'Windows': 'explorer.exe', 'Darwin': 'open .', 'Linux': 'xdg-open .'}}

    @property
    def tool_id(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'app_launcher'

    @property
    def name(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'AppLauncherTool'

    @property
    def description(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'Launch common default apps (e.g. browser, calculator, terminal, file_explorer)'

    @property
    def schema(self) -> Dict[str, Any]:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return {'type': 'object', 'properties': {'app_name': {'type': 'string', 'description': 'Name of the app concept (e.g. browser)'}}, 'required': ['app_name']}

    async def execute(self, params: Dict[str, Any]) -> ToolResult:  # type: ignore[override]  # type: ignore[override]
        """Auto-generated docstring.

Args:
    params: Argument.

Returns:
    Return value.
"""
        app_name = params.get('app_name', '').lower()
        if app_name not in self.SAFE_APPS:
            return ToolResult(success=False, error=f'Unrecognized safe app concept: {app_name}. Valid apps: {list(self.SAFE_APPS.keys())}')
        sys_type = platform.system()
        cmd = self.SAFE_APPS[app_name].get(sys_type)
        if not cmd:
            return ToolResult(success=False, error=f'{app_name} is not configured for {sys_type}')
        try:
            subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return ToolResult(success=True, output={'message': f'Successfully launched {app_name}'})
        except Exception as e:
            return ToolResult(success=False, error=f'Failed to launch app: {e}')

class CaptureScreenContextTool(BaseTool):
    """Securely captures the desktop screen into memory for diagnostic context."""

    @property
    def tool_id(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'capture_screen_context'

    @property
    def name(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'CaptureScreenContextTool'

    @property
    def description(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'Captures a screenshot of the current desktop to diagnose visual issues or errors.'

    @property
    def schema(self) -> Dict[str, Any]:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return {'type': 'object', 'properties': {}, 'required': []}

    async def execute(self, params: Dict[str, Any]) -> ToolResult:  # type: ignore[override]  # type: ignore[override]
        """Auto-generated docstring.

Args:
    params: Argument.

Returns:
    Return value.
"""
        try:
            import pyautogui
            from PIL import Image
            img = pyautogui.screenshot()
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=75)
            buffer.seek(0)
            img_b64 = base64.b64encode(buffer.read()).decode('utf-8')
            return ToolResult(success=True, output={'message': 'Screen context successfully captured into memory buffer.', 'image_data': f'data:image/jpeg;base64,{img_b64[:20]}...[TRUNCATED_FOR_LOGS]', 'resolution': img.size, 'status': 'secure_in_memory_only'})
        except (Exception, SystemExit) as e:
            return ToolResult(success=False, error=f'Failed to capture screen context: {e}')
