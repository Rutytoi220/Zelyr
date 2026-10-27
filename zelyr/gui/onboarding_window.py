from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, 
    QComboBox, QGraphicsOpacityEffect, QButtonGroup, QSpacerItem, QSizePolicy,
    QFrame
)
from PySide6.QtCore import Qt, QPropertyAnimation, QTimer, QEasingCurve, Slot
from PySide6.QtGui import QFont, QColor
from zelyr.config import get_config
from zelyr.services.profile_service import ProfileService, ProfileLevel

# ─────────────────────────────────────────────────────────────────────
# Zelyr v7.1.0 — Modernized Out-Of-Box Experience (OOBE)
# Design Language: Dark, minimal, rounded — Apple/Discord aesthetic.
# Accent: Zelyr Brand Green #2ECC71
# ─────────────────────────────────────────────────────────────────────

_GLOBAL_QSS = """
    * {
        font-family: 'Inter', 'Segoe UI', 'SF Pro Display', 'Helvetica Neue', sans-serif;
    }

    QMainWindow {
        background-color: #18181B;
        color: #E4E4E7;
    }

    QLabel {
        background: transparent;
        color: #E4E4E7;
    }

    QLabel#titleLabel {
        font-size: 56px;
        font-weight: 800;
        color: #2ECC71;
        letter-spacing: 0.15em;
    }

    QLabel#subtitleLabel {
        font-size: 16px;
        color: #A1A1AA;
        line-height: 1.6;
    }

    QLabel#sectionLabel {
        font-size: 14px;
        font-weight: 600;
        color: #D4D4D8;
    }

    QLabel#hintLabel {
        font-size: 12px;
        color: #71717A;
        padding-top: 4px;
    }

    QPushButton#profileBtn {
        background-color: #27272A;
        color: #A1A1AA;
        padding: 12px 24px;
        border-radius: 8px;
        border: 1px solid #3F3F46;
        font-size: 14px;
        font-weight: 500;
    }

    QPushButton#profileBtn:hover {
        background-color: #3F3F46;
        color: #E4E4E7;
        border: 1px solid #52525B;
    }

    QPushButton#profileBtn:checked {
        background-color: #2ECC71;
        color: #11111B;
        font-weight: 700;
        border: 1px solid #27AE60;
    }

    QComboBox#complexityDropdown {
        background-color: #27272A;
        color: #E4E4E7;
        padding: 10px 14px;
        border-radius: 8px;
        border: 1px solid #3F3F46;
        font-size: 14px;
        min-width: 180px;
    }

    QComboBox#complexityDropdown:hover {
        border: 1px solid #52525B;
    }

    QComboBox#complexityDropdown::drop-down {
        border: none;
        padding-right: 10px;
    }

    QComboBox#complexityDropdown QAbstractItemView {
        background-color: #27272A;
        color: #E4E4E7;
        selection-background-color: #2ECC71;
        selection-color: #11111B;
        border: 1px solid #3F3F46;
        border-radius: 6px;
        padding: 4px;
    }

    QPushButton#initBtn {
        background-color: #2ECC71;
        color: #11111B;
        padding: 14px 32px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 16px;
        border: none;
        min-height: 48px;
    }

    QPushButton#initBtn:hover {
        background-color: #27AE60;
    }

    QPushButton#initBtn:pressed {
        background-color: #229954;
    }

    QPushButton#initBtn:disabled {
        background-color: #3F3F46;
        color: #71717A;
    }

    QFrame#separator {
        background-color: #27272A;
        max-height: 1px;
        border: none;
    }
"""


class OnboardingWindow(QMainWindow):
    def __init__(self, bridge, completion_callback):
        super().__init__()
        self.bridge = bridge
        self.completion_callback = completion_callback
        
        # Borderless, dark-themed
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.resize(900, 620)
        self
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.setContentsMargins(80, 60, 80, 60)
        self.main_layout.setSpacing(0)
        
        # ── Top Spacer ──
        self.main_layout.addSpacerItem(
            QSpacerItem(0, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        )

        # ── Phase 1: Logo ──
        self.logo_label = QLabel("Zelyr")
        self.logo_label.setObjectName("titleLabel")
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.logo_effect = QGraphicsOpacityEffect()
        self.logo_effect.setOpacity(0)
        self.logo_label.setGraphicsEffect(self.logo_effect)
        self.main_layout.addWidget(self.logo_label)
        
        self.main_layout.addSpacerItem(
            QSpacerItem(0, 16, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        )

        # ── Phase 2: Typewriter subtitle ──
        self.dialogue_label = QLabel("")
        self.dialogue_label.setObjectName("subtitleLabel")
        self.dialogue_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dialogue_label.setWordWrap(True)
        self.dialogue_label.setMinimumHeight(50)
        
        self.dialogue_effect = QGraphicsOpacityEffect()
        self.dialogue_effect.setOpacity(1)
        self.dialogue_label.setGraphicsEffect(self.dialogue_effect)
        self.main_layout.addWidget(self.dialogue_label)
        
        self.main_layout.addSpacerItem(
            QSpacerItem(0, 36, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        )

        # ── Phase 3: Setup Pane ──
        self.setup_pane = QWidget()
        self.setup_layout = QVBoxLayout(self.setup_pane)
        self.setup_layout.setContentsMargins(0, 0, 0, 0)
        self.setup_layout.setSpacing(28)
        
        # ── Experience Level ──
        prof_section = QVBoxLayout()
        prof_section.setSpacing(10)
        
        prof_label = QLabel("Experience Level:")
        prof_label.setObjectName("sectionLabel")
        prof_section.addWidget(prof_label)
        
        prof_buttons = QHBoxLayout()
        prof_buttons.setSpacing(10)
        
        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)
        
        self.btn_std = QPushButton("Standard")
        self.btn_adv = QPushButton("Advanced")
        self.btn_dev = QPushButton("Developer")
        
        self.btn_std.setObjectName("profileBtn")
        self.btn_adv.setObjectName("profileBtn")
        self.btn_dev.setObjectName("profileBtn")
        
        self.btn_std.setCheckable(True)
        self.btn_adv.setCheckable(True)
        self.btn_dev.setCheckable(True)
        self.btn_std.setChecked(True)
        
        for b in [self.btn_std, self.btn_adv, self.btn_dev]:
            self.btn_group.addButton(b)
            prof_buttons.addWidget(b)
            
        prof_section.addLayout(prof_buttons)
        
        prof_hint = QLabel("You can change this anytime from Settings.")
        prof_hint.setObjectName("hintLabel")
        prof_section.addWidget(prof_hint)
        
        self.setup_layout.addLayout(prof_section)
        
        # ── Separator ──
        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.Shape.HLine)
        self.setup_layout.addWidget(sep)
        
        # ── AI Response Style ──
        comp_section = QVBoxLayout()
        comp_section.setSpacing(10)
        
        comp_label = QLabel("AI Response Style:")
        comp_label.setObjectName("sectionLabel")
        comp_section.addWidget(comp_label)
        
        self.comp_dropdown = QComboBox()
        self.comp_dropdown.setObjectName("complexityDropdown")
        self.comp_dropdown.addItems(["Concise", "Detailed", "Academic"])
        self.comp_dropdown.setCurrentText("Detailed")
        comp_section.addWidget(self.comp_dropdown)
        
        comp_hint = QLabel("Controls how verbose Zelyr's answers are.")
        comp_hint.setObjectName("hintLabel")
        comp_section.addWidget(comp_hint)
        
        self.setup_layout.addLayout(comp_section)
        
        # ── Spacer before CTA ──
        self.setup_layout.addSpacerItem(
            QSpacerItem(0, 12, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        )
        
        # ── Primary Call-To-Action ──
        self.init_btn = QPushButton("Let's Get Started")
        self.init_btn.setObjectName("initBtn")
        self.init_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.init_btn.clicked.connect(self._on_initialize)
        self.setup_layout.addWidget(self.init_btn)
        
        self.setup_effect = QGraphicsOpacityEffect()
        self.setup_effect.setOpacity(0)
        self.setup_pane.setGraphicsEffect(self.setup_effect)
        self.main_layout.addWidget(self.setup_pane)
        
        # ── Bottom Spacer ──
        self.main_layout.addSpacerItem(
            QSpacerItem(0, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        )

        # ── Animation State ──
        self._typewriter_text = ""
        self._typewriter_index = 0
        self._typewriter_timer = QTimer(self)
        self._typewriter_timer.timeout.connect(self._typewriter_tick)
        
        # Start Sequence
        QTimer.singleShot(500, self._start_phase_1)
        
    def _start_phase_1(self):
        """Fade in Logo."""
        self.anim1 = QPropertyAnimation(self.logo_effect, b"opacity")
        self.anim1.setDuration(2000)
        self.anim1.setStartValue(0.0)
        self.anim1.setEndValue(1.0)
        self.anim1.setEasingCurve(QEasingCurve.Type.InOutSine)
        self.anim1.finished.connect(lambda: QTimer.singleShot(1000, self._start_phase_2))
        self.anim1.start()
        
    def _start_phase_2(self):
        """Typewriter Dialogue."""
        self._typewriter_text = (
            "Welcome to Zelyr.\n"
            "To get things set up perfectly, how would you like your workspace to look?"
        )
        self._typewriter_index = 0
        self._typewriter_timer.start(35)
        
    @Slot()
    def _typewriter_tick(self):
        if self._typewriter_index < len(self._typewriter_text):
            # Pause on newline for natural breath
            if self._typewriter_text[self._typewriter_index] == "\n":
                self._typewriter_timer.stop()
                QTimer.singleShot(1200, lambda: self._typewriter_timer.start(35))
                
            self.dialogue_label.setText(self._typewriter_text[:self._typewriter_index + 1])
            self._typewriter_index += 1
        else:
            self._typewriter_timer.stop()
            QTimer.singleShot(800, self._start_phase_3)
            
    def _start_phase_3(self):
        """Fade in setup pane."""
        self.anim3 = QPropertyAnimation(self.setup_effect, b"opacity")
        self.anim3.setDuration(1200)
        self.anim3.setStartValue(0.0)
        self.anim3.setEndValue(1.0)
        self.anim3.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.anim3.start()
        
    @Slot()
    def _on_initialize(self):
        self.init_btn.setEnabled(False)
        self.init_btn.setText("Setting things up…")
        
        # Save config
        config = get_config()
        config.first_launch = False
        config.llm_complexity = self.comp_dropdown.currentText().lower()
        
        level = ProfileLevel.STANDARD
        if self.btn_adv.isChecked():
            level = ProfileLevel.ADVANCED
        elif self.btn_dev.isChecked():
            level = ProfileLevel.DEVELOPER
            
        config.ui_profile_level = level.value
        config.save()
        
        # Fade out whole window
        self.fade_out = QPropertyAnimation(self, b"windowOpacity")
        self.fade_out.setDuration(1200)
        self.fade_out.setStartValue(1.0)
        self.fade_out.setEndValue(0.0)
        self.fade_out.setEasingCurve(QEasingCurve.Type.OutQuad)
        self.fade_out.finished.connect(self._handoff)
        self.fade_out.start()
        
    def _handoff(self):
        self.close()
        self.completion_callback()
