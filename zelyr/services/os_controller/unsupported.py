import logging
from zelyr.services.os_controller.base import BaseOSController

logger = logging.getLogger(__name__)

class UnsupportedController(BaseOSController):
    """Null object controller for unsupported environments (e.g., GNOME Wayland)."""

    @property
    def can_click(self) -> bool:
        return False

    @property
    def can_type(self) -> bool:
        return False

    @property
    def can_capture(self) -> bool:
        return True  # Assuming screen capture is abstracted elsewhere (e.g. grim/dbus)

    @property
    def can_manage_windows(self) -> bool:
        return False

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> None:
        raise NotImplementedError("Mouse automation is not supported on this specific environment.")

    def type_text(self, text: str) -> None:
        raise NotImplementedError("Keyboard automation is not supported on this specific environment.")

    def press_key(self, key: str) -> None:
        raise NotImplementedError("Keyboard automation is not supported on this specific environment.")
