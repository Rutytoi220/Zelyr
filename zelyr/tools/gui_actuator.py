import logging
import subprocess
import shutil
from typing import Dict, Any, List
from zelyr.tools.core import BaseTool, ToolResult, ToolParameter

logger = logging.getLogger(__name__)

class GUIInspectTool(BaseTool):
    """Uses pyatspi to inspect the desktop GUI accessibility tree."""

    def __init__(self):
        super().__init__(
            tool_id="gui_inspect",
            name="gui_inspect",
            description="Inspects the Linux desktop GUI using AT-SPI to find active windows, buttons, labels, and their coordinates."
        )
        self.add_parameter(ToolParameter(
            name="app_name",
            type="string",
            description="Optional application name to filter the inspection (e.g. 'firefox', 'gnome-terminal'). If omitted, lists top-level apps.",
            required=False,
            default=""
        ))

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        app_name = params.get("app_name", "").lower()
        
        try:
            import pyatspi
        except ImportError:
            return ToolResult(
                success=False, 
                error="pyatspi is not installed. Please install it (e.g., python3-pyatspi) to use GUI inspection."
            )
            
        try:
            desktop = pyatspi.Registry.getDesktop(0)
            if not desktop:
                return ToolResult(success=False, error="Could not connect to AT-SPI desktop registry.")
                
            results = []
            
            for app in desktop:
                if not app:
                    continue
                name = app.name.lower()
                
                if app_name and app_name not in name:
                    continue
                    
                app_info = {"app_name": app.name, "windows": []}
                for window in app:
                    if not window:
                        continue
                    try:
                        extents = window.queryComponent().getExtents(pyatspi.DESKTOP_COORDS)
                        win_info = {
                            "name": window.name,
                            "role": window.getRoleName(),
                            "rect": [extents.x, extents.y, extents.width, extents.height],
                            "children": []
                        }
                        
                        # Simplified recursive scrape (just 1 level deep for demo)
                        for child in window:
                            if child:
                                try:
                                    child_extents = child.queryComponent().getExtents(pyatspi.DESKTOP_COORDS)
                                    win_info["children"].append({
                                        "name": child.name,
                                        "role": child.getRoleName(),
                                        "rect": [child_extents.x, child_extents.y, child_extents.width, child_extents.height]
                                    })
                                except Exception:
                                    pass
                                    
                        app_info["windows"].append(win_info)
                    except Exception:
                        pass
                
                results.append(app_info)
                
            import json
            return ToolResult(success=True, output=json.dumps(results, indent=2))
            
        except Exception as e:
            return ToolResult(success=False, error=f"AT-SPI Inspection failed: {str(e)}")

class GUIActuateTool(BaseTool):
    """Uses ydotool (or xdotool) to actuate the desktop (click, type, keys)."""

    def __init__(self):
        super().__init__(
            tool_id="gui_actuate",
            name="gui_actuate",
            description="Executes synthetic GUI events (mouse clicks, typing, hotkeys) using ydotool or xdotool."
        )
        self.add_parameter(ToolParameter(
            name="action",
            type="string",
            description="The action to perform: 'click', 'type', or 'key'."
        ))
        self.add_parameter(ToolParameter(
            name="x",
            type="integer",
            description="X coordinate for click action.",
            required=False
        ))
        self.add_parameter(ToolParameter(
            name="y",
            type="integer",
            description="Y coordinate for click action.",
            required=False
        ))
        self.add_parameter(ToolParameter(
            name="text",
            type="string",
            description="Text to type or key combo (e.g. 'ctrl+c').",
            required=False
        ))

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        action = params.get("action", "")
        
        ydotool = shutil.which("ydotool")
        xdotool = shutil.which("xdotool")
        
        tool_bin = ydotool or xdotool
        if not tool_bin:
            return ToolResult(success=False, error="Neither ydotool nor xdotool found on system.")
            
        cmd = []
        if action == "click":
            x = params.get("x")
            y = params.get("y")
            if x is None or y is None:
                return ToolResult(success=False, error="click requires x and y coordinates.")
            if tool_bin == ydotool:
                cmd = [tool_bin, "mousemove", "--absolute", str(x), str(y), "click", "1"]
            else:
                cmd = [tool_bin, "mousemove", str(x), str(y), "click", "1"]
                
        elif action == "type":
            text = params.get("text")
            if not text:
                return ToolResult(success=False, error="type requires 'text' parameter.")
            cmd = [tool_bin, "type", text]
            
        elif action == "key":
            text = params.get("text")
            if not text:
                return ToolResult(success=False, error="key requires 'text' parameter (e.g. 'ctrl+c').")
            cmd = [tool_bin, "key", text]
            
        else:
            return ToolResult(success=False, error=f"Unknown action: {action}")
            
        try:
            logger.info(f"Executing GUI Actuator: {' '.join(cmd)}")
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return ToolResult(success=True, output=f"Executed {action} successfully.\n{res.stdout}")
        except subprocess.CalledProcessError as e:
            return ToolResult(success=False, error=f"Actuation failed: {e.stderr}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))
