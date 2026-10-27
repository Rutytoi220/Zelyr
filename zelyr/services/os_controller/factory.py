import os
import shutil
import logging
from zelyr.services.os_controller.base import BaseOSController
from zelyr.services.os_controller.hyprland import HyprlandController
from zelyr.services.os_controller.wayland import WaylandController
from zelyr.services.os_controller.standard import StandardController
from zelyr.services.os_controller.unsupported import UnsupportedController

logger = logging.getLogger(__name__)


def get_os_controller() -> BaseOSController:
    """Auto-detect the OS environment and return the appropriate controller."""
    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
    current_desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").lower()

    if session_type == "wayland":
        # Check if ydotool is available for universal kernel-level input
        if shutil.which("ydotool"):
            logger.info("Detected Wayland with ydotool support. Using WaylandController.")
            return WaylandController()

        if "hyprland" in current_desktop:
            logger.info("Detected Wayland+Hyprland. Using HyprlandController.")
            return HyprlandController()

        logger.warning(
            f"Detected Wayland compositor ({current_desktop}) without ydotool. "
            "Install ydotool for universal input automation."
        )
        return UnsupportedController()
    else:
        # Fallback for X11, Windows, macOS
        logger.info("Detected standard OS environment (or X11). Using StandardController.")
        return StandardController()
