import shutil
import subprocess
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# To prevent circular imports, we will import BaseTool and ToolResult locally
# or assume they are injected / we can just use typing tricks.
# Actually, we can import them from zelyr.tools at runtime or just import from zelyr.tools!
# Since we moved the import to the bottom of __init__.py, `from zelyr.tools.core import BaseTool, ToolResult` will work!

from zelyr.tools.core import BaseTool, ToolResult

class ClipboardReadTool(BaseTool):
    """Tool for reading text from the system clipboard."""
    
    @property
    def tool_id(self) -> str:
        return 'clipboard_read'

    @property
    def name(self) -> str:
        return 'clipboard_read'

    @property
    def description(self) -> str:
        return 'Read the current text content of the system clipboard.'

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            'type': 'object',
            'properties': {},
            'required': []
        }

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        try:
            if shutil.which("wl-paste"):
                proc = subprocess.run(["wl-paste", "--no-newline"], capture_output=True, text=True, check=True)
                return ToolResult(success=True, output=proc.stdout)
            elif shutil.which("xclip"):
                proc = subprocess.run(["xclip", "-selection", "clipboard", "-o"], capture_output=True, text=True, check=True)
                return ToolResult(success=True, output=proc.stdout)
            elif shutil.which("xsel"):
                proc = subprocess.run(["xsel", "--clipboard", "--output"], capture_output=True, text=True, check=True)
                return ToolResult(success=True, output=proc.stdout)
            else:
                return ToolResult(success=False, error="No clipboard utility found (wl-paste, xclip, xsel).")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

class ClipboardWriteTool(BaseTool):
    """Tool for writing text to the system clipboard."""
    
    @property
    def tool_id(self) -> str:
        return 'clipboard_write'

    @property
    def name(self) -> str:
        return 'clipboard_write'

    @property
    def description(self) -> str:
        return 'Write text content to the system clipboard.'

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            'type': 'object',
            'properties': {
                'text': {
                    'type': 'string',
                    'description': 'Text to copy to clipboard'
                }
            },
            'required': ['text']
        }

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        text = params.get('text', '')
        try:
            if shutil.which("wl-copy"):
                subprocess.run(["wl-copy"], input=text, text=True, check=True)
                return ToolResult(success=True, output="Text copied to clipboard.")
            elif shutil.which("xclip"):
                subprocess.run(["xclip", "-selection", "clipboard", "-i"], input=text, text=True, check=True)
                return ToolResult(success=True, output="Text copied to clipboard.")
            elif shutil.which("xsel"):
                subprocess.run(["xsel", "--clipboard", "--input"], input=text, text=True, check=True)
                return ToolResult(success=True, output="Text copied to clipboard.")
            else:
                return ToolResult(success=False, error="No clipboard utility found (wl-copy, xclip, xsel).")
        except Exception as e:
            return ToolResult(success=False, error=str(e))
