"""Native Linux DBus Notifications via notify-send."""
import logging
import subprocess

logger = logging.getLogger(__name__)

class DesktopNotifier:
    """Dispatches native desktop notifications."""

    @staticmethod
    def notify(title: str, body: str, icon: str = "dialog-information") -> None:
        """
        Sends a native Linux desktop notification using notify-send.
        
        Args:
            title: The title of the notification.
            body: The body/message of the notification.
            icon: Icon name (e.g. 'dialog-warning', 'dialog-information').
        """
        try:
            subprocess.run(
                ["notify-send", "-a", "Zelyr", "-i", icon, title, body],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            logger.info(f"Notification sent: {title}")
        except FileNotFoundError:
            logger.warning("notify-send not found. Desktop notifications are disabled.")
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
