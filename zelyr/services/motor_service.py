import logging
import sys

logger = logging.getLogger("zelyr.automation")

# ── Primary Wayland Controller (ydotool) ──────────────────────────────────
WAYLAND_CONTROLLER = None
try:
    from zelyr.tools.input_controller import WaylandInputController
    WAYLAND_CONTROLLER = WaylandInputController()
    logger.info("Motor Cortex: WaylandInputController (ydotool) active.")
except Exception as e:
    logger.debug(f"Motor Cortex: WaylandInputController not loaded: {e}")

# ── Fallback X11 Controller (pyautogui) ───────────────────────────────────
HAS_PYAUTOGUI = False
try:
    import pyautogui
    pyautogui.FAILSAFE = True
    HAS_PYAUTOGUI = True
except BaseException as e:
    logger.debug(f"Motor Cortex: pyautogui not loaded ({e})")


class MotorService:
    """Service to handle mouse and keyboard automation safely on Wayland and X11."""

    @staticmethod
    def _get_controller():
        global WAYLAND_CONTROLLER
        if WAYLAND_CONTROLLER is None:
            try:
                from zelyr.tools.input_controller import WaylandInputController
                WAYLAND_CONTROLLER = WaylandInputController()
            except Exception:
                pass
        return WAYLAND_CONTROLLER

    @classmethod
    def mouse_move(cls, x: int, y: int, duration: float = 0.5):
        """Move the mouse to a specific coordinate."""
        ctrl = cls._get_controller()
        if ctrl:
            try:
                logger.info(f"Motor Cortex (Wayland): Moving mouse to ({x}, {y})")
                ctrl.mouse_move(x, y)
                return True, f"Mouse moved to {x}, {y}"
            except Exception as e:
                logger.error(f"Motor Cortex (Wayland): Failed to move mouse: {e}")
                return False, str(e)

        if not HAS_PYAUTOGUI:
            return False, "Motor Cortex is offline (neither ydotool nor pyautogui available)."
        try:
            logger.info(f"Motor Cortex (X11): Moving mouse to ({x}, {y})")
            pyautogui.moveTo(x, y, duration=duration)
            return True, f"Mouse moved to {x}, {y}"
        except pyautogui.FailSafeException:
            logger.warning("Motor Cortex: FailSafeException triggered during move!")
            return False, "FailSafe triggered! Action aborted by user."
        except Exception as e:
            logger.error(f"Motor Cortex: Failed to move mouse: {e}")
            return False, str(e)

    @classmethod
    def mouse_click(cls, x: int = None, y: int = None, button: str = "left"):
        """Click the mouse at a specific coordinate."""
        ctrl = cls._get_controller()
        if ctrl:
            try:
                if x is not None and y is not None:
                    ctrl.mouse_move(x, y)
                logger.info(f"Motor Cortex (Wayland): Clicking {button} at ({x}, {y})")
                ctrl.mouse_click(button)
                return True, f"Clicked {button} at {x}, {y}"
            except Exception as e:
                logger.error(f"Motor Cortex (Wayland): Failed to click mouse: {e}")
                return False, str(e)

        if not HAS_PYAUTOGUI:
            return False, "Motor Cortex is offline (neither ydotool nor pyautogui available)."
        try:
            logger.info(f"Motor Cortex (X11): Clicking {button} at ({x}, {y})")
            if x is not None and y is not None:
                pyautogui.click(x=x, y=y, button=button)
            else:
                pyautogui.click(button=button)
            return True, f"Clicked {button} at {x}, {y}"
        except pyautogui.FailSafeException:
            logger.warning("Motor Cortex: FailSafeException triggered during click!")
            return False, "FailSafe triggered! Action aborted by user."
        except Exception as e:
            logger.error(f"Motor Cortex: Failed to click mouse: {e}")
            return False, str(e)

    @classmethod
    def keyboard_type(cls, text: str, interval: float = 0.05):
        """Type text using the keyboard."""
        ctrl = cls._get_controller()
        if ctrl:
            try:
                logger.info(f"Motor Cortex (Wayland): Typing text: '{text}'")
                ctrl.keyboard_type(text)
                return True, "Typed text successfully"
            except Exception as e:
                logger.error(f"Motor Cortex (Wayland): Failed to type text: {e}")
                return False, str(e)

        if not HAS_PYAUTOGUI:
            return False, "Motor Cortex is offline (neither ydotool nor pyautogui available)."
        try:
            logger.info(f"Motor Cortex (X11): Typing text: '{text}'")
            pyautogui.write(text, interval=interval)
            return True, "Typed text successfully"
        except pyautogui.FailSafeException:
            logger.warning("Motor Cortex: FailSafeException triggered during typing!")
            return False, "FailSafe triggered! Action aborted by user."
        except Exception as e:
            logger.error(f"Motor Cortex: Failed to type text: {e}")
            return False, str(e)

    @classmethod
    def keyboard_press(cls, key: str):
        """Press a specific key."""
        ctrl = cls._get_controller()
        if ctrl:
            try:
                logger.info(f"Motor Cortex (Wayland): Pressing key: {key}")
                ctrl.keyboard_press(key)
                return True, f"Pressed key {key}"
            except Exception as e:
                logger.error(f"Motor Cortex (Wayland): Failed to press key: {e}")
                return False, str(e)

        if not HAS_PYAUTOGUI:
            return False, "Motor Cortex is offline (neither ydotool nor pyautogui available)."
        try:
            logger.info(f"Motor Cortex (X11): Pressing key: {key}")
            pyautogui.press(key)
            return True, f"Pressed key {key}"
        except pyautogui.FailSafeException:
            logger.warning("Motor Cortex: FailSafeException triggered during key press!")
            return False, "FailSafe triggered! Action aborted by user."
        except Exception as e:
            logger.error(f"Motor Cortex: Failed to press key: {e}")
            return False, str(e)
