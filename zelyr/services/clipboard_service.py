import logging
from PySide6.QtGui import QGuiApplication

logger = logging.getLogger(__name__)

class ClipboardService:
    """Service to interact with the system clipboard securely."""

    @staticmethod
    def get_text() -> str:
        """Retrieves text from the system clipboard."""
        try:
            clipboard = QGuiApplication.clipboard()
            if clipboard:
                return clipboard.text()
            return ""
        except Exception as e:
            logger.error(f"Failed to read clipboard: {e}")
            return ""
