"""Desktop control actions: screenshot, click, type, move mouse."""

import os
import base64
import time
from typing import Tuple, Optional
from PIL import ImageGrab
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Primary Wayland Controller (ydotool) ──────────────────────────────────
WAYLAND_CONTROLLER = None
try:
    from zelyr.tools.input_controller import WaylandInputController
    WAYLAND_CONTROLLER = WaylandInputController()
    logger.info("actions/desktop: WaylandInputController (ydotool) initialized.")
except Exception as e:
    logger.debug("actions/desktop: WaylandInputController not initialized: %s", e)

try:
    import pyautogui
    pyautogui.FAILSAFE = True  # Move mouse to corner to abort
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    logger.warning("pyautogui not installed, using WaylandInputController fallback")


def _get_wayland_ctrl() -> Optional[WaylandInputController]:
    global WAYLAND_CONTROLLER
    if WAYLAND_CONTROLLER is None:
        try:
            from zelyr.tools.input_controller import WaylandInputController
            WAYLAND_CONTROLLER = WaylandInputController()
        except Exception:
            pass
    return WAYLAND_CONTROLLER


def click_mouse(x: int = None, y: int = None, button: str = 'left', clicks: int = 1) -> Tuple[bool, str]:
    """Click the mouse at specified coordinates."""
    ctrl = _get_wayland_ctrl()
    if ctrl:
        try:
            if x is not None and y is not None:
                ctrl.mouse_move(int(x), int(y))
            ctrl.mouse_click(button=button, clicks=clicks)
            logger.info("Clicked with button '%s' (x=%s, y=%s) via ydotool", button, x, y)
            return True, f"Clicked at ({x}, {y})"
        except Exception as e:
            logger.exception("Failed to click mouse via ydotool")
            return False, f"Click failed: {e}"

    if not PYAUTOGUI_AVAILABLE:
        return False, "Input simulation not available (neither ydotool nor pyautogui)"

    try:
        x, y = int(x), int(y)
        pyautogui.click(x, y, clicks=clicks, button=button)
        logger.info("Clicked at (%d, %d) with button '%s'", x, y, button)
        return True, f"Clicked at ({x}, {y})"
    except Exception as e:
        logger.exception("Failed to click mouse")
        return False, f"Click failed: {e}"


def type_text(text: str, interval: float = 0.05) -> Tuple[bool, str]:
    """Type text at current cursor position."""
    ctrl = _get_wayland_ctrl()
    if ctrl:
        try:
            ctrl.keyboard_type(text)
            logger.info("Typed via ydotool: %s", text[:50])
            return True, f"Typed: {text[:50]}"
        except Exception as e:
            logger.exception("Failed to type text via ydotool")
            return False, f"Type failed: {e}"

    if not PYAUTOGUI_AVAILABLE:
        return False, "Input simulation not available (neither ydotool nor pyautogui)"

    try:
        pyautogui.typewrite(text, interval=interval)
        logger.info("Typed: %s", text[:50])
        return True, f"Typed: {text[:50]}"
    except Exception as e:
        logger.exception("Failed to type text")
        return False, f"Type failed: {e}"


def press_keys(keys: str) -> Tuple[bool, str]:
    """Press keyboard keys."""
    ctrl = _get_wayland_ctrl()
    if ctrl:
        try:
            ctrl.keyboard_press(keys)
            logger.info("Pressed keys via ydotool: %s", keys)
            return True, f"Pressed: {keys}"
        except Exception as e:
            logger.exception("Failed to press keys via ydotool")
            return False, f"Key press failed: {e}"

    if not PYAUTOGUI_AVAILABLE:
        return False, "Input simulation not available (neither ydotool nor pyautogui)"

    try:
        key_list = [k.strip().lower() for k in keys.split('+')]
        pyautogui.hotkey(*key_list)
        logger.info("Pressed keys: %s", keys)
        return True, f"Pressed: {keys}"
    except Exception as e:
        logger.exception("Failed to press keys")
        return False, f"Key press failed: {e}"


def move_mouse(x: int, y: int, duration: float = 0.5) -> Tuple[bool, str]:
    """Move mouse to specified coordinates."""
    ctrl = _get_wayland_ctrl()
    if ctrl:
        try:
            ctrl.mouse_move(int(x), int(y))
            logger.info("Moved mouse to (%d, %d) via ydotool", x, y)
            return True, f"Moved mouse to ({x}, {y})"
        except Exception as e:
            logger.exception("Failed to move mouse via ydotool")
            return False, f"Move failed: {e}"

    if not PYAUTOGUI_AVAILABLE:
        return False, "Input simulation not available (neither ydotool nor pyautogui)"

    try:
        x, y = int(x), int(y)
        pyautogui.moveTo(x, y, duration=duration)
        logger.info("Moved mouse to (%d, %d)", x, y)
        return True, f"Moved mouse to ({x}, {y})"
    except Exception as e:
        logger.exception("Failed to move mouse")
        return False, f"Move failed: {e}"


def take_screenshot(encode_base64: bool = True) -> Tuple[bool, str]:
    """Take a screenshot of the current desktop.
    
    Args:
        encode_base64: If True, return base64-encoded PNG. If False, return file path.
    
    Returns:
        (success, base64_string or file_path)
    """
    try:
        screenshot = ImageGrab.grab()
        
        if encode_base64:
            # Convert to base64 for LLM analysis
            import io
            buffer = io.BytesIO()
            screenshot.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            logger.debug("Screenshot captured and encoded to base64 (%d bytes)", len(img_base64))
            return True, img_base64
        else:
            # Save to file
            screenshot_path = '/tmp/zelyr_screenshot.png'
            screenshot.save(screenshot_path)
            logger.info("Screenshot saved to %s", screenshot_path)
            return True, screenshot_path
            
    except Exception as e:
        logger.exception("Failed to take screenshot")
        return False, f"Screenshot failed: {e}"


def click_mouse(x: int, y: int, button: str = 'left', clicks: int = 1) -> Tuple[bool, str]:
    """Click the mouse at specified coordinates.
    
    Args:
        x: X coordinate
        y: Y coordinate
        button: 'left', 'right', or 'middle'
        clicks: Number of clicks
    
    Returns:
        (success, message)
    """
    if not PYAUTOGUI_AVAILABLE:
        return False, "pyautogui not available"
    
    try:
        x, y = int(x), int(y)
        pyautogui.click(x, y, clicks=clicks, button=button)
        logger.info("Clicked at (%d, %d) with button '%s'", x, y, button)
        return True, f"Clicked at ({x}, {y})"
    except Exception as e:
        logger.exception("Failed to click mouse")
        return False, f"Click failed: {e}"


def type_text(text: str, interval: float = 0.05) -> Tuple[bool, str]:
    """Type text at current cursor position.
    
    Args:
        text: Text to type
        interval: Delay between keystrokes (seconds)
    
    Returns:
        (success, message)
    """
    if not PYAUTOGUI_AVAILABLE:
        return False, "pyautogui not available"
    
    try:
        pyautogui.typewrite(text, interval=interval)
        logger.info("Typed: %s", text[:50])
        return True, f"Typed: {text[:50]}"
    except Exception as e:
        logger.exception("Failed to type text")
        return False, f"Type failed: {e}"


def press_keys(keys: str) -> Tuple[bool, str]:
    """Press keyboard keys.
    
    Args:
        keys: Key names separated by '+' (e.g., 'ctrl+c', 'alt+tab', 'enter')
    
    Returns:
        (success, message)
    """
    if not PYAUTOGUI_AVAILABLE:
        return False, "pyautogui not available"
    
    try:
        key_list = [k.strip().lower() for k in keys.split('+')]
        pyautogui.hotkey(*key_list)
        logger.info("Pressed keys: %s", keys)
        return True, f"Pressed: {keys}"
    except Exception as e:
        logger.exception("Failed to press keys")
        return False, f"Key press failed: {e}"


def move_mouse(x: int, y: int, duration: float = 0.5) -> Tuple[bool, str]:
    """Move mouse to specified coordinates.
    
    Args:
        x: X coordinate
        y: Y coordinate
        duration: Time to move (seconds)
    
    Returns:
        (success, message)
    """
    if not PYAUTOGUI_AVAILABLE:
        return False, "pyautogui not available"
    
    try:
        x, y = int(x), int(y)
        pyautogui.moveTo(x, y, duration=duration)
        logger.info("Moved mouse to (%d, %d)", x, y)
        return True, f"Moved mouse to ({x}, {y})"
    except Exception as e:
        logger.exception("Failed to move mouse")
        return False, f"Move failed: {e}"


def get_screen_size() -> Tuple[bool, str]:
    """Get screen resolution.
    
    Returns:
        (success, "WIDTHxHEIGHT")
    """
    try:
        size = ImageGrab.grab().size
        logger.info("Screen size: %dx%d", size[0], size[1])
        return True, f"{size[0]}x{size[1]}"
    except Exception as e:
        logger.exception("Failed to get screen size")
        return False, f"Failed to get screen size: {e}"
