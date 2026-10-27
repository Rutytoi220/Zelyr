import sys
import asyncio
import qasync
import logging
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QTextEdit, QLabel
from PySide6.QtCore import Qt, QSize, Slot, QMetaObject, Q_ARG, Qt as QtCoreQt
from PySide6.QtGui import QIcon, QFont, QColor, QPainter, QPen

logger = logging.getLogger(__name__)

class AudioVisualizerWidget(QWidget):
    """Draws a live audio waveform."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(40)
        self.waveform = []
        
    def update_waveform(self, data: list):
        self.waveform = data
        self.update()
        
    def paintEvent(self, event):
        if not self.waveform:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor("#a6e3a1"))
        pen.setWidth(2)
        painter.setPen(pen)
        
        width = self.width()
        height = self.height()
        mid = height / 2
        
        step = width / max(1, len(self.waveform))
        
        for i, val in enumerate(self.waveform):
            x = i * step
            # val is between 0 and 1
            h = val * mid
            painter.drawLine(x, mid - h, x, mid + h)

class JarvisOverlayWindow(QWidget):
    """Unified frameless, translucent overlay for Voice + Vision interaction."""
    
    def __init__(self):
        super().__init__()
        # Frameless, top-level popup, translucent
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(800, 500)
        
        # We simulate the services being attached
        self._voice_daemon = None 
        self._vision_stream = None
        
        self._init_ui()
        self._center_on_screen()
        
    def _center_on_screen(self):
        screen = QApplication.primaryScreen().geometry()
        self.move(
            (screen.width() - self.width()) // 2,
            (screen.height() - self.height()) // 4
        )

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        container = QWidget()
        container.setObjectName("container")
        container
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(20, 20, 20, 20)
        container_layout.setSpacing(15)
        
        # Top Bar: Status + Input
        top_bar = QHBoxLayout()
        self.status_icon = QLabel("🧠")
        self.status_icon.setFont(QFont("Arial", 24))
        top_bar.addWidget(self.status_icon)
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask Zelyr... or hold [Voice Command]")
        self.input_field
        self.input_field.returnPressed.connect(self._on_submit)
        top_bar.addWidget(self.input_field)
        
        container_layout.addLayout(top_bar)
        
        # Audio Visualizer
        self.visualizer = AudioVisualizerWidget()
        container_layout.addWidget(self.visualizer)
        
        # Control Buttons
        btn_layout = QHBoxLayout()
        
        self.voice_btn = QPushButton("Voice Command")
        self._style_btn(self.voice_btn, "#cba6f7")
        self.voice_btn.pressed.connect(self._start_voice)
        self.voice_btn.released.connect(self._stop_voice)
        btn_layout.addWidget(self.voice_btn)
        
        self.vision_btn = QPushButton("👁️ Select Screen Target")
        self._style_btn(self.vision_btn, "#f9e2af")
        self.vision_btn.clicked.connect(self._start_vision_select)
        btn_layout.addWidget(self.vision_btn)
        
        container_layout.addLayout(btn_layout)
        
        # Output Area
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        self.output_area
        self.output_area.hide()
        container_layout.addWidget(self.output_area)
        
        main_layout.addWidget(container)
        
    def _style_btn(self, btn: QPushButton, accent: str):
        btn
        
    def _start_voice(self):
        self.input_field.setPlaceholderText("Listening...")
        self.visualizer.update_waveform([0.2, 0.5, 0.8, 0.4, 0.9, 0.3, 0.6]) # Mock active waveform
        logger.info("Jarvis: Voice STT recording started.")
        
    def _stop_voice(self):
        self.input_field.setPlaceholderText("Ask Zelyr... or hold [Voice Command]")
        self.visualizer.update_waveform([])
        self.input_field.setText("Explain this error message highlighted in red and read the fix aloud to me.")
        logger.info("Jarvis: Voice STT recording stopped. Processing...")
        
    def _start_vision_select(self):
        logger.info("Jarvis: Activating Wayland Crop Overlay for Vision Targeting.")
        # Trigger crop overlay logic from original HUD
        
    def _on_submit(self):
        prompt = self.input_field.text().strip()
        if not prompt: return
        
        self.input_field.clear()
        self.output_area.show()
        self.output_area.setText("Thinking...\n")
        logger.info(f"Jarvis: Query Submitted: {prompt}")
        
def run_jarvis():
    app = QApplication.instance() or QApplication(sys.argv)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    jarvis = JarvisOverlayWindow()
    jarvis.show()
    
    with loop:
        loop.run_forever()

if __name__ == "__main__":
    run_jarvis()
