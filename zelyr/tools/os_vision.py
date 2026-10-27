import os
import sys
import shutil
import subprocess
from pathlib import Path
from zelyr.sdk.plugin import tool

@tool(
    name="capture_screen",
    description="Captures the entire screen and returns the absolute path to the saved screenshot image."
)
def capture_screen() -> str:
    """Takes a screenshot of the entire desktop and saves it to a temporary file."""
    output_path = "/tmp/zelyr_vision.png"
    
    # Ensure the directory exists (for non-Unix platforms where /tmp might not exist, use a safe default)
    if sys.platform == "win32":
        output_path = os.path.join(os.environ.get("TEMP", "C:\\Temp"), "zelyr_vision.png")
    
    is_linux = sys.platform.startswith("linux")
    is_wayland = is_linux and os.environ.get("WAYLAND_DISPLAY") is not None
    
    if is_wayland:
        # Wayland specific screen capture
        if not shutil.which("grim"):
            return "Error: 'grim' is required on Linux Wayland for screen capture. Please install it (e.g. 'sudo apt install grim' or 'sudo pacman -S grim')."
            
        try:
            result = subprocess.run(["grim", output_path], capture_output=True, text=True, check=True)
            return output_path
        except subprocess.CalledProcessError as e:
            return f"Error executing grim: {e.stderr}"
    else:
        # Windows, macOS, or Linux X11 fallback
        try:
            import pyautogui
            screenshot = pyautogui.screenshot()
            screenshot.save(output_path)
            return output_path
        except ImportError:
            # Fallback to Pillow if pyautogui is not installed
            try:
                from PIL import ImageGrab
                screenshot = ImageGrab.grab()
                screenshot.save(output_path)
                return output_path
            except ImportError:
                return "Error: 'pyautogui' or 'Pillow' is required for screen capture on this platform. Please install them."
        except Exception as e:
            return f"Error capturing screen: {str(e)}"
