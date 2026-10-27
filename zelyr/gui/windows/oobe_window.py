from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, 
    QSpacerItem, QSizePolicy, QWidget, QLineEdit, QRadioButton, QButtonGroup
)
from PySide6.QtCore import Qt, Signal
from zelyr.gui.config_manager import get_ui_config_manager, UIConfig

_OOBE_QSS = """
    QDialog {
        background-color: #1e1e2e;
        color: #E4E4E7;
    }
    QLabel {
        color: #E4E4E7;
        font-family: 'Inter', sans-serif;
    }
    QLabel#titleLabel {
        font-size: 28px;
        font-weight: 800;
        color: #E4E4E7;
    }
    QLabel#subtitleLabel {
        font-size: 14px;
        color: #A1A1AA;
    }
    QPushButton.colorBtn {
        border-radius: 20px;
        min-width: 40px;
        min-height: 40px;
        max-width: 40px;
        max-height: 40px;
        border: 2px solid #313244;
    }
    QPushButton.colorBtn:checked {
        border: 3px solid #FFFFFF;
    }
    QLineEdit {
        background-color: #11111b;
        color: #E4E4E7;
        border: 1px solid #313244;
        border-radius: 6px;
        padding: 8px;
    }
    QPushButton#initBtn {
        background-color: #2ECC71;
        color: #11111B;
        padding: 10px 20px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 14px;
        border: none;
    }
    QPushButton#initBtn:hover {
        background-color: #27AE60;
    }
    QPushButton#initBtn:disabled {
        background-color: #313244;
        color: #71717A;
    }
    QRadioButton {
        color: #E4E4E7;
        font-family: 'Inter', sans-serif;
        font-size: 13px;
        spacing: 8px;
    }
    QRadioButton::indicator {
        width: 14px;
        height: 14px;
        border-radius: 7px;
        border: 2px solid #313244;
        background-color: #1e1e2e;
    }
    QRadioButton::indicator:checked {
        border: 4px solid #2ECC71;
        background-color: #1e1e2e;
    }
"""

class ColorButton(QPushButton):
    def __init__(self, color_hex: str, parent=None):
        super().__init__(parent)
        self.color_hex = color_hex
        self.setCheckable(True)
        self.setProperty("class", "colorBtn")
        self
        self.setCursor(Qt.CursorShape.PointingHandCursor)

class OOBEWindow(QDialog):
    
    initialization_complete = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Zelyr Setup")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setFixedSize(600, 420)
        self
        
        self.selected_color = "#2ECC71"  # Default Green
        self.selected_voice_mode = "push_to_talk"
        
        self._build_ui()
        
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Header
        title = QLabel("Welcome to Zelyr")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        subtitle = QLabel("Please select an accent color for your interface.")
        subtitle.setObjectName("subtitleLabel")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)
        
        layout.addSpacerItem(QSpacerItem(0, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))
        
        # Color Selection
        color_layout = QHBoxLayout()
        color_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        color_layout.setSpacing(15)
        
        self.color_btns = []
        preset_colors = ["#007ACC", "#4CAF50", "#E5A50A", "#9C27B0"]
        
        for c in preset_colors:
            btn = ColorButton(c)
            btn.clicked.connect(lambda checked, hex_code=c: self._on_color_selected(hex_code))
            color_layout.addWidget(btn)
            self.color_btns.append(btn)
            if c == self.selected_color:
                btn.setChecked(True)
                
        layout.addLayout(color_layout)
        
        # Custom Hex Input
        hex_layout = QHBoxLayout()
        hex_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hex_layout.setSpacing(16)
        hex_label = QLabel("Custom Hex:")
        self.hex_input = QLineEdit()
        self.hex_input.setMinimumHeight(32)
        self.hex_input.setPlaceholderText("#FFFFFF")
        self.hex_input.setFixedWidth(120)
        self.hex_input.textChanged.connect(self._on_hex_changed)
        
        hex_layout.addWidget(hex_label)
        hex_layout.addWidget(self.hex_input)
        layout.addLayout(hex_layout)
        
        layout.addSpacerItem(QSpacerItem(0, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))
        
        # Voice Mode Selection
        voice_title = QLabel("Voice Mode:")
        voice_title
        voice_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(voice_title)
        
        voice_layout = QHBoxLayout()
        voice_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        voice_layout.setSpacing(32)
        
        self.voice_group = QButtonGroup(self)
        
        self.btn_ptt = QRadioButton("Push-to-Talk")
        self.btn_ptt.setChecked(True)
        self.btn_ptt.setToolTip("Privacy First. Zelyr only listens when you click the microphone.")
        
        self.btn_wake = QRadioButton("🗣️ Wake Word (Hey Zelyr...)")
        self.btn_wake.setToolTip("Always Listening. Zelyr waits for the wake word in the background.")
        
        self.voice_group.addButton(self.btn_ptt)
        self.voice_group.addButton(self.btn_wake)
        
        voice_layout.addWidget(self.btn_ptt)
        voice_layout.addWidget(self.btn_wake)
        layout.addLayout(voice_layout)
        
        self.btn_ptt.toggled.connect(self._on_voice_mode_changed)
        
        layout.addSpacerItem(QSpacerItem(0, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        
        # Init Button
        self.init_btn = QPushButton("Initialize Zelyr")
        self.init_btn.setObjectName("initBtn")
        self.init_btn.setMinimumHeight(40)
        self.init_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.init_btn.clicked.connect(self._on_initialize)
        layout.addWidget(self.init_btn, alignment=Qt.AlignmentFlag.AlignCenter)

    def _on_color_selected(self, hex_code: str):
        self.selected_color = hex_code
        self.hex_input.blockSignals(True)
        self.hex_input.setText(hex_code)
        self.hex_input.blockSignals(False)
        
        for btn in self.color_btns:
            btn.setChecked(btn.color_hex == hex_code)

    def _on_hex_changed(self, text: str):
        if text.startswith("#") and len(text) == 7:
            self.selected_color = text
            for btn in self.color_btns:
                btn.setChecked(btn.color_hex == text.upper() or btn.color_hex == text.lower())

    def _on_voice_mode_changed(self):
        if self.btn_ptt.isChecked():
            self.selected_voice_mode = "push_to_talk"
        else:
            self.selected_voice_mode = "wake_word"

    def _on_initialize(self):
        self.init_btn.setEnabled(False)
        self.init_btn.setText("Saving...")
        
        manager = get_ui_config_manager()
        config = manager.load()
        config.accent_color = self.selected_color
        config.voice_mode = self.selected_voice_mode
        manager.save(config)
        
        self.initialization_complete.emit()
        self.accept()
