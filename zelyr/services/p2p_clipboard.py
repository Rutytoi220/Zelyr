import logging
import asyncio
import socket
from PySide6.QtGui import QClipboard
from PySide6.QtWidgets import QApplication
from zelyr.core.events import EventBus

logger = logging.getLogger(__name__)

class P2PClipboardService:
    """Synchronizes system clipboard text across the LAN mesh via encrypted IPC."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.app = QApplication.instance()
        if not self.app:
            self.app = QApplication([])
        
        self.clipboard = self.app.clipboard()
        self.hostname = socket.gethostname()
        self.is_syncing = False
        
        # Prevent echo loops
        self._last_synced_text = ""
        
        # Connect to clipboard changes
        self.clipboard.dataChanged.connect(self._on_local_copy)
        
        # Subscribe to incoming mesh syncs
        if self.event_bus:
            self.event_bus.subscribe("mesh.clipboard.received", self._on_mesh_received)

    def toggle_sync(self, state: bool):
        self.is_syncing = state
        logger.info(f"P2P Clipboard Sync toggled: {'ON' if state else 'OFF'}")

    def _on_local_copy(self):
        if not self.is_syncing:
            return
            
        text = self.clipboard.text()
        if text and text != self._last_synced_text:
            logger.debug(f"P2P Clipboard: Local copy detected, broadcasting ({len(text)} chars)")
            self._last_synced_text = text
            
            # Publish event to be picked up by the mesh client broadcaster
            if self.event_bus:
                self.event_bus.publish_sync("mesh.clipboard.broadcast", {
                    "content": text,
                    "hostname": self.hostname
                })

    def _on_mesh_received(self, event):
        """Called when a remote node broadcasts clipboard data."""
        if not self.is_syncing:
            return
            
        content = event.data.get("content", "")
        if content and content != self._last_synced_text:
            self._last_synced_text = content
            self.clipboard.setText(content)
            logger.info("P2P Clipboard: Applied remote content to local clipboard")
