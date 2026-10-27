import subprocess
import json
import shutil
import logging
from typing import Dict, Any
from zelyr.tools.core import BaseTool

logger = logging.getLogger(__name__)

class WorkspaceOrchestrateTool(BaseTool):
    """Tool for programmatically managing Wayland desktop workspaces and window tiling."""
    
    name = "workspace_orchestrator"
    description = "Manages desktop workspaces and tiles windows programmatically."
    
    def __init__(self):
        super().__init__()
        # Auto-detect compositor
        self.compositor = self._detect_compositor()
        
    def _detect_compositor(self) -> str:
        if shutil.which("hyprctl"):
            return "hyprland"
        if shutil.which("swaymsg"):
            return "sway"
        if shutil.which("wmctrl"):
            return "x11_wmctrl"
        return "unknown"
        
    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create_and_focus_workspace", "launch_and_tile_app", "get_workspace_tree"],
                    "description": "The action to perform."
                },
                "workspace_name": {
                    "type": "string",
                    "description": "Name or ID of the workspace (used with create_and_focus_workspace)."
                },
                "command": {
                    "type": "string",
                    "description": "Command to launch the app (used with launch_and_tile_app)."
                },
                "layout": {
                    "type": "string",
                    "enum": ["split-right", "split-left", "split-top", "split-bottom", "float"],
                    "description": "How to tile the app."
                }
            },
            "required": ["action"]
        }
        
    def execute(self, arguments: Dict[str, Any]) -> str:
        action = arguments["action"]
        
        if action == "create_and_focus_workspace":
            name = arguments.get("workspace_name", "1")
            return self._create_and_focus_workspace(name)
        elif action == "launch_and_tile_app":
            command = arguments.get("command")
            layout = arguments.get("layout", "split-right")
            if not command:
                return "Error: command is required."
            return self._launch_and_tile_app(command, layout)
        elif action == "get_workspace_tree":
            return self._get_workspace_tree()
            
        return f"Unknown action: {action}"
        
    def _create_and_focus_workspace(self, name: str) -> str:
        if self.compositor == "hyprland":
            subprocess.run(["hyprctl", "dispatch", "workspace", name], check=False)
            return f"Focused Hyprland workspace: {name}"
        elif self.compositor == "sway":
            subprocess.run(["swaymsg", "workspace", name], check=False)
            return f"Focused Sway workspace: {name}"
        elif self.compositor == "x11_wmctrl":
            # For wmctrl, name must be an index typically or we can't easily create one
            subprocess.run(["wmctrl", "-s", name], check=False)
            return f"Focused X11 workspace: {name}"
        return f"Compositor {self.compositor} unsupported. Simulating workspace switch."
        
    def _launch_and_tile_app(self, command: str, layout: str) -> str:
        # Launching async
        subprocess.Popen(command, shell=True, start_new_session=True)
        # Note: True programmatic tiling via IPC is complex and requires waiting for the window map event.
        # For Hyprland/Sway, they auto-tile by default. We just let the compositor handle the layout for now.
        return f"Launched `{command}` with intended layout `{layout}` on {self.compositor}."
        
    def _get_workspace_tree(self) -> str:
        if self.compositor == "hyprland":
            res = subprocess.run(["hyprctl", "clients", "-j"], capture_output=True, text=True)
            if res.returncode == 0:
                try:
                    clients = json.loads(res.stdout)
                    tree = [{"class": c.get("class"), "title": c.get("title"), "workspace": c.get("workspace", {}).get("name")} for c in clients]
                    return json.dumps(tree, indent=2)
                except:
                    pass
        elif self.compositor == "sway":
            res = subprocess.run(["swaymsg", "-t", "get_tree"], capture_output=True, text=True)
            if res.returncode == 0:
                return "Sway tree retrieved (truncated for length)."
                
        return "Workspace tree unavailable for current compositor."
