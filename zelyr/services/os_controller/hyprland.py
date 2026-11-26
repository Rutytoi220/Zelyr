import logging
from typing import Optional
from zelyr.services.os_controller.base import BaseOSController

logger = logging.getLogger(__name__)


class HyprlandController(BaseOSController):
    """OS Controller for Wayland + Hyprland backed by WaylandInputController."""

    def __init__(self) -> None:
        self._ctrl: Optional[object] = None

    def _get_ctrl(self):
        if self._ctrl is None:
            from zelyr.tools.input_controller import WaylandInputController

            self._ctrl = WaylandInputController(auto_check=False)
        return self._ctrl

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

    def move_mouse(self, x: int, y: int) -> None:
        """Move pointer to desktop coordinates (x, y)."""
        try:
            self._get_ctrl().mouse_move(x, y)
        except Exception as e:
            logger.error("Hyprland move_mouse failed: %s", e)
            raise RuntimeError(f"Hyprland move_mouse failed: {e}") from e

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> None:
        """Move cursor to (x, y) and click using Wayland/uinput input controller."""
        try:
            ctrl = self._get_ctrl()
            ctrl.mouse_click(button=button, clicks=clicks, x=x, y=y)
        except Exception as e:
            logger.error("Hyprland click failed: %s", e)
            raise RuntimeError(f"Hyprland click failed: {e}") from e

    def type_text(self, text: str) -> None:
        """Type text via Wayland/uinput input controller."""
        try:
            self._get_ctrl().keyboard_type(text)
        except Exception as e:
            logger.error("Hyprland type_text failed: %s", e)
            raise RuntimeError(f"Hyprland type_text failed: {e}") from e

    def press_key(self, key: str) -> None:
        """Press key or modifier combo via Wayland/uinput input controller."""
        try:
            self._get_ctrl().keyboard_press(key)
        except Exception as e:
            logger.error("Hyprland press_key failed: %s", e)
            raise RuntimeError(f"Hyprland press_key failed: {e}") from e
