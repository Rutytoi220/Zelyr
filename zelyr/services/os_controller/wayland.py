import logging
from typing import Optional
from zelyr.services.os_controller.base import BaseOSController
from zelyr.tools.input_controller import WaylandInputController

logger = logging.getLogger(__name__)


class WaylandController(BaseOSController):
    """Universal OS Controller for Wayland environments using ydotool / uinput."""

    def __init__(self, input_controller: Optional[WaylandInputController] = None):
        try:
            self._ctrl = input_controller or WaylandInputController()
        except Exception as e:
            logger.warning("Failed to initialize WaylandInputController: %s", e)
            self._ctrl = None

    @property
    def can_click(self) -> bool:
        return self._ctrl is not None

    @property
    def can_type(self) -> bool:
        return self._ctrl is not None

    @property
    def can_capture(self) -> bool:
        return True

    @property
    def can_manage_windows(self) -> bool:
        return True

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> None:
        if not self._ctrl:
            raise RuntimeError("Wayland controller is offline (ydotoold unavailable)")
        self._ctrl.mouse_move(x, y)
        self._ctrl.mouse_click(button, clicks)

    def type_text(self, text: str) -> None:
        if not self._ctrl:
            raise RuntimeError("Wayland controller is offline (ydotoold unavailable)")
        self._ctrl.keyboard_type(text)

    def press_key(self, key: str) -> None:
        if not self._ctrl:
            raise RuntimeError("Wayland controller is offline (ydotoold unavailable)")
        self._ctrl.keyboard_press(key)
