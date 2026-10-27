import logging
import os
from PySide6.QtGui import QGuiApplication

logger = logging.getLogger(__name__)

class VisionService:
    """Service to interact with the system screen visually."""

    @staticmethod
    def capture_screen(output_path: str = "/tmp/zelyr_capture.png") -> str:
        """Captures the primary screen and saves it to a file. Returns the path."""
        try:
            screen = QGuiApplication.primaryScreen()
            if screen:
                pixmap = screen.grabWindow(0)
                pixmap.save(output_path, "PNG")
                return output_path
            return ""
        except Exception as e:
            logger.error(f"Failed to capture screen: {e}")
            return ""
