import logging
from zelyr.services.os_controller.base import BaseOSController

logger = logging.getLogger(__name__)

class StandardController(BaseOSController):
    """OS Controller for X11, Windows, and macOS using pyautogui."""

    @property
    def can_click(self) -> bool:
        return True

    @property
    def can_type(self) -> bool:
        return True

    @property
    def can_capture(self) -> bool:
        return True

    @property
    def can_manage_windows(self) -> bool:
        return True

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> None:
        import pyautogui
        try:
            pyautogui.click(x=x, y=y, button=button, clicks=clicks)
        except Exception as e:
            logger.error("Standard click failed: %s", e)
            raise RuntimeError(f"Standard click failed: {e}")

    def type_text(self, text: str) -> None:
        import pyautogui
        try:
            pyautogui.typewrite(text)
        except Exception as e:
            logger.error("Standard type_text failed: %s", e)
            raise RuntimeError(f"Standard type_text failed: {e}")

    def press_key(self, key: str) -> None:
        import pyautogui
        try:
            pyautogui.press(key)
        except Exception as e:
            logger.error("Standard press_key failed: %s", e)
            raise RuntimeError(f"Standard press_key failed: {e}")
