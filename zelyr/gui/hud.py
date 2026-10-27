import sys
import asyncio
import qasync
import logging
import shutil
import subprocess
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
    QLineEdit, QPushButton, QTextEdit, QLabel
)
from PySide6.QtCore import Qt, QSize, Slot, QMetaObject, Q_ARG, Qt as QtCoreQt
from PySide6.QtGui import QIcon, QFont, QColor

from zelyr.client.ipc_client import ZelyrDaemonClient

logger = logging.getLogger(__name__)

class HUDWindow(QWidget):
    def __init__(self):
        super().__init__()
        # Frameless, top-level popup
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self.setFixedSize(700, 400)
        self._init_ui()
        self._client = ZelyrDaemonClient()
        self._client.on_event = self._on_daemon_event
        self._voice_engine = None
        self._is_recording = False
        
        # Center on screen
        screen = QApplication.primaryScreen().geometry()
        self.move(
            (screen.width() - self.width()) // 2,
            (screen.height() - self.height()) // 4
        )

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Main container with rounded corners and dark theme
        container = QWidget()
        container.setObjectName("container")
        container
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(15, 15, 15, 15)
        container_layout.setSpacing(10)
        
        # Header row: Icon, Input, Paste Button
        header_layout = QHBoxLayout()
        
        self.status_icon = QLabel("⚡")
        self.status_icon.setFont(QFont("Arial", 16))
        header_layout.addWidget(self.status_icon)
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask Zelyr... (Press Enter to submit)")
        
        self.input_field.returnPressed.connect(self._on_submit)
        header_layout.addWidget(self.input_field)
        
        self.dictate_btn = QPushButton("🎙️ Dictate")
        
        self.dictate_btn.clicked.connect(self._on_dictate_toggle)
        header_layout.addWidget(self.dictate_btn)

        self.crop_btn = QPushButton("✂️ Crop Vision")
        
        self.crop_btn.clicked.connect(self._on_crop_vision)
        header_layout.addWidget(self.crop_btn)
        
        self.paste_btn = QPushButton("📋 Paste Context")
        
        self.paste_btn.clicked.connect(self._on_paste_context)
        header_layout.addWidget(self.paste_btn)

        self.mesh_sync_btn = QPushButton("📋 Mesh Sync")
        self.mesh_sync_btn.setCheckable(True)
        self.mesh_sync_btn
        self.mesh_sync_btn.clicked.connect(self._on_mesh_sync_toggle)
        header_layout.addWidget(self.mesh_sync_btn)
        
        container_layout.addLayout(header_layout)
        
        # Output Area (hidden by default until submission)
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        
        self.output_area.hide()
        container_layout.addWidget(self.output_area)
        
        main_layout.addWidget(container)
        
    def showEvent(self, event):
        self.input_field.setFocus()
        super().showEvent(event)
        
        # Connect client when shown
        loop = asyncio.get_event_loop()
        asyncio.create_task(self._client.connect())

    def _on_paste_context(self):
        text = ""
        try:
            if shutil.which("wl-paste"):
                proc = subprocess.run(["wl-paste", "--no-newline"], capture_output=True, text=True)
                text = proc.stdout
            elif shutil.which("xclip"):
                proc = subprocess.run(["xclip", "-selection", "clipboard", "-o"], capture_output=True, text=True)
                text = proc.stdout
        except Exception as e:
            logger.error(f"Failed to read clipboard: {e}")
            
        if text:
            current = self.input_field.text()
            self.input_field.setText(f"[System Clipboard Context]:\n{text}\n\n[User Request]: {current}")
            self.input_field.setFocus()

    def _on_dictate_toggle(self):
        if not self._voice_engine:
            from zelyr.tools.voice_engine import VoiceDictationEngine
            self._voice_engine = VoiceDictationEngine()
            
        if not self._is_recording:
            self._is_recording = True
            self.dictate_btn.setText("🔴 Recording...")
            self.dictate_btn.setProperty("status", "danger"); self.dictate_btn.style().unpolish(self.dictate_btn); self.dictate_btn.style().polish(self.dictate_btn)
            self._voice_engine.start_recording()
        else:
            self._is_recording = False
            self.dictate_btn.setText("⏳ Processing...")
            self.dictate_btn.setProperty("status", "warning"); self.dictate_btn.style().unpolish(self.dictate_btn); self.dictate_btn.style().polish(self.dictate_btn)
            
            # Process in background
            def process():
                text = self._voice_engine.stop_recording_and_transcribe()
                # Run back in main thread
                QMetaObject.invokeMethod(
                    self, 
                    "_on_dictate_finished", 
                    QtCoreQt.QueuedConnection, 
                    Q_ARG(str, text)
                )
                
            import threading
            threading.Thread(target=process, daemon=True).start()

    @Slot(str)
    def _on_dictate_finished(self, text: str):
        self.dictate_btn.setText("🎙️ Dictate")
        
        if text:
            current = self.input_field.text()
            self.input_field.setText(f"{current} {text}".strip())
        self.input_field.setFocus()

    def _on_submit(self):
        prompt = self.input_field.text().strip()
        if not prompt:
            return
            
        self.input_field.clear()
        self.output_area.show()
        self.output_area.clear()
        self.status_icon.setText("🔄")
        
        if not self._client.is_connected:
            self.output_area.append("Error: Not connected to daemon.")
            self.status_icon.setText("❌")
            return
            
        loop = asyncio.get_event_loop()
        asyncio.create_task(self._client.submit_task(prompt))

    def _on_daemon_event(self, data: dict):
        event_type = data.get("event_type", "")
        payload = data.get("payload", {})
        
        if event_type == "llm.token":
            chunk = payload.get("chunk", "")
            # Use QTimer or signal in a real app, but qasync loop makes this thread-safe enough
            self.output_area.insertPlainText(chunk)
            self.output_area.verticalScrollBar().setValue(self.output_area.verticalScrollBar().maximum())
        elif event_type == "orchestrator.finished":
            self.status_icon.setText("✅")

    def _on_crop_vision(self):
        """Open the fullscreen transparent overlay to draw a crop region."""
        from zelyr.gui.overlay_hud import CropOverlayWindow
        self._overlay = CropOverlayWindow()
        self._overlay.crop_selected.connect(self._handle_crop)
        self._overlay.show()
        
    def _handle_crop(self, x: int, y: int, w: int, h: int):
        """Send the cropped region coordinates to VisionAgent for analysis."""
        self.input_field.setText(f"/vision --crop {x},{y},{w},{h} What is in this region?")
        self.input_field.setFocus()
        self.show()

    def _on_mesh_sync_toggle(self, checked: bool):
        # Notify the P2P clipboard service
        # In a real integration, the HUD would pass this down via EventBus
        self._client.send_message({
            "type": "event",
            "topic": "gui.clipboard.sync_toggled",
            "data": {"state": checked}
        })

def run_hud():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    hud = HUDWindow()
    hud.show()
    
    with loop:
        loop.run_forever()

if __name__ == "__main__":
    run_hud()
