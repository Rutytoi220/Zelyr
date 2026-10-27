"""Deep Wayland Compositor Hooks.

Interfaces directly with the Wayland Compositor (targeting Hyprland IPC)
to draw UI elements (e.g. glowing rectangles, floating annotations) over
other windows without requiring focus. Enables VisionAgent to physically
"point" at the screen.
"""
import logging
import subprocess
import json
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class WaylandOverlayInjector:
    """Injects visual elements into the Wayland compositor."""
    
    def __init__(self, compositor: str = "hyprland"):
        self.compositor = compositor
        
    def _is_hyprland(self) -> bool:
        """Check if we are running under Hyprland."""
        return self.compositor == "hyprland"
        
    def highlight_window(self, window_title: str) -> bool:
        """Draws a visual highlight (e.g., changing border color) via Hyprland IPC."""
        if not self._is_hyprland():
            logger.warning("WaylandOverlayInjector: Highlighting only supported on Hyprland currently.")
            return False
            
        try:
            logger.info(f"WaylandOverlayInjector: Attempting to highlight window matching '{window_title}'...")
            
            # Fetch clients
            result = subprocess.run(["hyprctl", "-j", "clients"], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("WaylandOverlayInjector: Failed to fetch hyprctl clients.")
                return False
                
            clients = json.loads(result.stdout)
            target_address = None
            
            for client in clients:
                if window_title.lower() in client.get("title", "").lower() or window_title.lower() in client.get("class", "").lower():
                    target_address = client.get("address")
                    break
                    
            if not target_address:
                logger.warning(f"WaylandOverlayInjector: No window found matching '{window_title}'")
                return False
                
            # Dispatch border color change to highlight
            # 'rgb(ff0000)' for glowing red
            dispatch_cmd = ["hyprctl", "dispatch", "setprop", f"address:{target_address}", "bordercolor", "rgb(ff0000)"]
            subprocess.run(dispatch_cmd, capture_output=True)
            
            logger.info(f"WaylandOverlayInjector: Successfully highlighted window {target_address}")
            return True
            
        except FileNotFoundError:
            logger.error("WaylandOverlayInjector: 'hyprctl' command not found. Are you on Hyprland?")
            return False
        except Exception as e:
            logger.error(f"WaylandOverlayInjector: Error during highlight - {e}")
            return False
            
    def draw_floating_annotation(self, text: str, x: int, y: int):
        """
        Draws text at a specific coordinate.
        In a full implementation, this uses GTK Layer Shell or wlr-layer-shell.
        For now, we mock the output.
        """
        logger.info(f"WaylandOverlayInjector: [MOCK LAYER-SHELL] Drawing annotation '{text}' at ({x}, {y})")
        return True
