import os
import sys
import logging
from zelyr.sdk.plugin import tool
from zelyr.tools.input_controller import WaylandInputController

logger = logging.getLogger(__name__)

_controller = None

def _get_ctrl() -> WaylandInputController:
    global _controller
    if _controller is None:
        _controller = WaylandInputController()
    return _controller

def _is_linux() -> bool:
    """Check if the current OS is Linux."""
    return sys.platform.startswith("linux")

@tool(
    name="move_mouse",
    description="Moves the mouse cursor to the specified absolute (x, y) coordinates on the screen."
)
def move_mouse(x: int, y: int) -> str:
    """Moves the mouse cursor to absolute (x, y)."""
    if _is_linux():
        try:
            _get_ctrl().mouse_move(x, y)
            return f"Mouse moved to ({x}, {y}) via ydotool."
        except Exception as e:
            return f"Error executing ydotool mouse_move: {e}"
    else:
        try:
            import pyautogui
            pyautogui.moveTo(x, y)
            return f"Mouse moved to ({x}, {y}) via pyautogui."
        except ImportError:
            return "Error: 'pyautogui' is required on this platform for mouse control. Please install it."
        except Exception as e:
            return f"Error moving mouse: {str(e)}"

@tool(
    name="click_mouse",
    description="Clicks the mouse. button can be 'left', 'right', or 'middle'."
)
def click_mouse(button: str = 'left') -> str:
    """Clicks the mouse."""
    if _is_linux():
        try:
            _get_ctrl().mouse_click(button)
            return f"Clicked {button} mouse button via ydotool."
        except Exception as e:
            return f"Error executing ydotool click: {e}"
    else:
        try:
            import pyautogui
            pyautogui.click(button=button)
            return f"Clicked {button} mouse button via pyautogui."
        except ImportError:
            return "Error: 'pyautogui' is required on this platform for mouse control. Please install it."
        except Exception as e:
            return f"Error clicking mouse: {str(e)}"

@tool(
    name="type_text",
    description="Types the given string of text on the keyboard as if it were typed manually."
)
def type_text(text: str) -> str:
    """Types text on the keyboard."""
    if _is_linux():
        try:
            _get_ctrl().keyboard_type(text)
            return "Text typed successfully via ydotool."
        except Exception as e:
            return f"Error executing ydotool type: {e}"
    else:
        try:
            import pyautogui
            pyautogui.write(text)
            return "Text typed successfully via pyautogui."
        except ImportError:
            return "Error: 'pyautogui' is required on this platform for keyboard control. Please install it."
        except Exception as e:
            return f"Error typing text: {str(e)}"

@tool(
    name="press_key",
    description="Presses a single key or key combination (e.g., 'enter', 'tab', 'super', 'ctrl+c')."
)
def press_key(key: str) -> str:
    """Presses a key on the keyboard."""
    if _is_linux():
        try:
            _get_ctrl().keyboard_press(key)
            return f"Pressed key '{key}' via ydotool."
        except Exception as e:
            return f"Error executing ydotool key press: {e}"
    else:
        try:
            import pyautogui
            pyautogui.press(key)
            return f"Pressed key '{key}' via pyautogui."
        except ImportError:
            return "Error: 'pyautogui' is required on this platform. Please install it."
        except Exception as e:
            return f"Error pressing key: {str(e)}"
