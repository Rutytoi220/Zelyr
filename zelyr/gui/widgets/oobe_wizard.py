from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QStackedWidget, QWidget, QComboBox, QPlainTextEdit, QFrame,
    QGridLayout, QLineEdit
)
from PySide6.QtCore import Qt, Signal
from zelyr.config import get_config
from zelyr.core.persona import PERSONA_PRESETS



class ThemeCard(QFrame):
    clicked = Signal(str)

    def __init__(self, key: str, display_name: str, gradient_css: str):
        super().__init__()
        self.card_key = key
        self.setProperty("class", "Card")
        self.setProperty("selected", False)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.preview = QFrame()
        self.preview.setStyleSheet(f"border-top-left-radius: 8px; border-top-right-radius: 8px; {gradient_css}")
        self.preview.setMinimumHeight(100)
        
        text_container = QFrame()
        text_layout = QVBoxLayout(text_container)
        text_layout.setContentsMargins(12, 12, 12, 12)
        title = QLabel(display_name)
        title.setObjectName("oobe_card_title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text_layout.addWidget(title)
        
        layout.addWidget(self.preview)
        layout.addWidget(text_container)
        
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumSize(220, 160)

    def mousePressEvent(self, event):
        self.clicked.emit(self.card_key)
        super().mousePressEvent(event)

    def set_selected(self, selected: bool):
        self.setProperty("selected", selected)
        self.style().unpolish(self)
        self.style().polish(self)

class SelectionCard(QFrame):
    """Reusable clickable selection card for themes and personas."""
    clicked = Signal(str)

    def __init__(self, key: str, display_name: str, desc: str):
        super().__init__()
        self.card_key = key
        self.setProperty("class", "Card")
        self.setProperty("selected", False)

        layout = QVBoxLayout(self)

        title = QLabel(display_name)
        title.setObjectName("oobe_card_title")

        subtitle = QLabel(desc)
        subtitle.setObjectName("oobe_card_subtitle")

        layout.addWidget(title)
        layout.addWidget(subtitle)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(220, 150)

    def mousePressEvent(self, event):
        self.clicked.emit(self.card_key)
        super().mousePressEvent(event)

    def set_selected(self, selected: bool):
        self.setProperty("selected", selected)
        self.style().unpolish(self)
        self.style().polish(self)


class OobeWizardDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zelyr Setup Wizard")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setFixedSize(800, 600)
        self.setObjectName("oobe_dialog")

        self.config = get_config()

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(40, 40, 40, 40)

        self.stack = QStackedWidget()
        self.layout.addWidget(self.stack)


        self._build_welcome_page()
        self._build_theme_page()
        self._build_persona_page()
        self._build_directives_page()

        # UNIFIED TOOLBAR
        toolbar = QHBoxLayout()
        toolbar.addStretch()
        
        self.nav_btn = QPushButton("Next")
        self.nav_btn.setObjectName("oobe_primary_btn")
        self.nav_btn.setFixedWidth(200)
        self.nav_btn.clicked.connect(self._on_nav_clicked)
        
        toolbar.addWidget(self.nav_btn)
        self.layout.addLayout(toolbar)
        
        self.stack.currentChanged.connect(self._on_page_changed)



    def _on_page_changed(self, index: int):
        if index == self.stack.count() - 1:
            self.nav_btn.setText("Finish Setup & Boot Zelyr")
        else:
            self.nav_btn.setText("Next")

    def _on_nav_clicked(self):
        idx = self.stack.currentIndex()
        if idx < self.stack.count() - 1:
            self.stack.setCurrentIndex(idx + 1)
        else:
            self._on_finish()

    def _build_welcome_page(self):
        page = QWidget()
        l = QVBoxLayout(page)
        l.addStretch()

        title = QLabel("Welcome to Zelyr.")
        title.setObjectName("oobe_title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        sub = QLabel("Let's configure your sovereign desktop agent.")
        sub.setObjectName("oobe_subtitle")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)

        l.addWidget(title)
        l.addWidget(sub)
        l.addStretch()



        self.stack.addWidget(page)

    def _build_theme_page(self):
        page = QWidget()
        l = QVBoxLayout(page)

        title = QLabel("Choose Your Aesthetic")
        title.setObjectName("oobe_title")
        l.addWidget(title)
        l.addSpacing(40)

        grid = QGridLayout()
        grid.setSpacing(20)

        self.theme_cards = []
        themes = [
            ("zelyr_pro", "Zelyr Pro", "background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #161B22, stop:1 #0D1117); border-bottom: 2px solid #A78BFA;"),
            ("nordic", "Nordic Frost", "background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2E3440, stop:1 #3B4252); border-bottom: 2px solid #88C0D0;"),
            ("jarvis", "J.A.R.V.I.S", "background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0A0F0D, stop:1 #050806); border-bottom: 2px solid #00FF41;")
        ]

        row, col = 0, 0
        for t_id, t_name, t_grad in themes:
            card = ThemeCard(t_id, t_name, t_grad)
            card.clicked.connect(self._on_theme_selected)
            grid.addWidget(card, row, col)
            self.theme_cards.append(card)
            col += 1
            if col > 2:
                col = 0
                row += 1

        l.addLayout(grid)
        l.addStretch()

        self._on_theme_selected(self.config.theme or "minimalist")

        btn = QPushButton("Next")
        btn.setObjectName("oobe_primary_btn")
        btn.setFixedWidth(150)
        btn.clicked.connect(lambda: self.stack.setCurrentIndex(2))

        h_btn = QHBoxLayout()
        h_btn.addStretch()
        h_btn.addWidget(btn)
        l.addLayout(h_btn)

        self.stack.addWidget(page)

    def _on_theme_selected(self, theme_name: str):
        self.config.theme = theme_name
        for card in self.theme_cards:
            card.set_selected(card.card_key == theme_name)
            
        # LIVE RELOAD
        from zelyr.gui.styles.theme_manager import get_theme_manager
        from PySide6.QtWidgets import QApplication
        tm = get_theme_manager()
        tm.apply_theme(QApplication.instance(), theme_name)

    def _build_persona_page(self):
        page = QWidget()
        l = QVBoxLayout(page)

        title = QLabel("Persona: Identity & Cognition")
        title.setObjectName("oobe_title")
        l.addWidget(title)
        l.addSpacing(20)

        l.addWidget(QLabel("Select a Persona:"))
        persona_hl = QHBoxLayout()
        persona_hl.setSpacing(20)

        self.persona_cards = []
        persona_defs = [
            ("default", "Standard Zelyr", "Helpful, concise AI assistant"),
            ("jarvis", "JARVIS / Terminal", "Cold, analytical system operator"),
            ("minimal", "Minimalist", "Code-only, zero conversational filler")
        ]

        for p_key, p_name, p_desc in persona_defs:
            card = SelectionCard(p_key, p_name, p_desc)
            card.clicked.connect(self._on_persona_selected)
            persona_hl.addWidget(card)
            self.persona_cards.append(card)

        l.addLayout(persona_hl)
        l.addSpacing(30)

        l.addWidget(QLabel("Customize:"))
        grid = QGridLayout()
        grid.setSpacing(15)

        self.name_input = QLineEdit()
        self.role_input = QLineEdit()
        self.tone_combo = QComboBox()
        self.tone_combo.setObjectName("oobe_combo")
        self.tone_combo.addItems(["Highly Formal", "Balanced", "Casual/Snarky"])

        self.verbosity_combo = QComboBox()
        self.verbosity_combo.setObjectName("oobe_combo")
        self.verbosity_combo.addItems(["Concise", "Standard", "Explain like I'm 5"])

        self.tech_depth_combo = QComboBox()
        self.tech_depth_combo.setObjectName("oobe_combo")
        self.tech_depth_combo.addItems(["Layman", "Standard", "Developer", "Systems Architect"])

        self.formatting_combo = QComboBox()
        self.formatting_combo.setObjectName("oobe_combo")
        self.formatting_combo.addItems(["Standard Markdown", "Heavy Code Blocks", "Bullet-Point Strict"])

        grid.addWidget(QLabel("Name:"), 0, 0)
        grid.addWidget(self.name_input, 0, 1)
        grid.addWidget(QLabel("Role:"), 1, 0)
        grid.addWidget(self.role_input, 1, 1)
        grid.addWidget(QLabel("Tone:"), 2, 0)
        grid.addWidget(self.tone_combo, 2, 1)
        grid.addWidget(QLabel("Verbosity:"), 3, 0)
        grid.addWidget(self.verbosity_combo, 3, 1)
        grid.addWidget(QLabel("Technical Depth:"), 4, 0)
        grid.addWidget(self.tech_depth_combo, 4, 1)
        grid.addWidget(QLabel("Formatting:"), 5, 0)
        grid.addWidget(self.formatting_combo, 5, 1)

        l.addLayout(grid)
        l.addStretch()

        saved_key = self.config.persona_key if self.config.persona_key in PERSONA_PRESETS else "default"
        self._apply_persona_preset(saved_key)

        btn = QPushButton("Next")
        btn.setObjectName("oobe_primary_btn")
        btn.setFixedWidth(150)
        btn.clicked.connect(lambda: self.stack.setCurrentIndex(3))

        h_btn = QHBoxLayout()
        h_btn.addStretch()
        h_btn.addWidget(btn)
        l.addLayout(h_btn)

        self.stack.addWidget(page)

    def _on_persona_selected(self, persona_key: str):
        self.config.persona_key = persona_key
        self._apply_persona_preset(persona_key)
        for card in self.persona_cards:
            card.set_selected(card.card_key == persona_key)

    def _apply_persona_preset(self, persona_key: str):
        preset = PERSONA_PRESETS.get(persona_key, PERSONA_PRESETS["default"])

        tone_map = {
            "highly_formal": "Highly Formal",
            "balanced": "Balanced",
            "casual_snarky": "Casual/Snarky"
        }
        verbosity_map = {
            "explain_like_im_5": "Explain like I'm 5",
            "standard": "Standard",
            "concise": "Concise"
        }
        tech_map = {
            "layman": "Layman",
            "standard": "Standard",
            "developer": "Developer",
            "systems_architect": "Systems Architect"
        }
        format_map = {
            "standard": "Standard Markdown",
            "standard_markdown": "Standard Markdown",
            "heavy_code": "Heavy Code Blocks",
            "bullet_point": "Bullet-Point Strict"
        }

        identity = preset.get("identity", {})
        comm = preset.get("communication", {})

        self.name_input.setText(identity.get("name", "Zelyr"))
        self.role_input.setText(identity.get("role", "Desktop Agent"))
        self.tone_combo.setCurrentText(tone_map.get(comm.get("tone", "balanced"), "Balanced"))
        self.verbosity_combo.setCurrentText(verbosity_map.get(comm.get("verbosity", "standard"), "Standard"))
        self.tech_depth_combo.setCurrentText(tech_map.get(comm.get("technical_depth", "standard"), "Standard"))
        self.formatting_combo.setCurrentText(format_map.get(comm.get("formatting_preference", "standard"), "Standard Markdown"))

    def _build_directives_page(self):
        page = QWidget()
        l = QVBoxLayout(page)

        title = QLabel("Advanced Behavior & Directives")
        title.setObjectName("oobe_title")
        l.addWidget(title)

        sub = QLabel("Configure strict operational behaviors.")
        sub.setObjectName("oobe_subtitle")
        l.addWidget(sub)
        l.addSpacing(20)

        from PySide6.QtWidgets import QCheckBox, QGridLayout
        grid = QGridLayout()
        grid.setSpacing(15)

        self.initiative_combo = QComboBox()
        self.initiative_combo.setObjectName("oobe_combo")
        self.initiative_combo.addItems(["Reactive (Wait for prompt)", "Proactive (Suggest next steps)"])

        self.confirmation_combo = QComboBox()
        self.confirmation_combo.setObjectName("oobe_combo")
        self.confirmation_combo.addItems(["Auto-execute all", "Ask before destructive", "Ask before ANY terminal command"])
        self.confirmation_combo.setCurrentText("Ask before destructive")

        grid.addWidget(QLabel("Initiative:"), 0, 0)
        grid.addWidget(self.initiative_combo, 0, 1)
        grid.addWidget(QLabel("Confirmation Policy:"), 1, 0)
        grid.addWidget(self.confirmation_combo, 1, 1)

        l.addLayout(grid)
        l.addSpacing(15)

        self.cb_monologue = QCheckBox("Show Inner Monologue (<thought> block)")
        self.cb_monologue.setObjectName("oobe_check_box")
        self.cb_confidence = QCheckBox("Provide Confidence %")
        self.cb_confidence.setObjectName("oobe_check_box")
        self.cb_explain = QCheckBox("Explain dangerous commands")
        self.cb_explain.setObjectName("oobe_check_box")
        self.cb_explain.setChecked(True)
        self.cb_emojis = QCheckBox("Use emojis")
        self.cb_emojis.setObjectName("oobe_check_box")

        l.addWidget(self.cb_monologue)
        l.addWidget(self.cb_confidence)
        l.addWidget(self.cb_explain)
        l.addWidget(self.cb_emojis)
        l.addSpacing(20)

        l.addWidget(QLabel("Global System Instructions (Directives):"))
        self.directives_input = QPlainTextEdit()
        self.directives_input.setObjectName("oobe_text_edit")
        self.directives_input.setPlaceholderText("e.g., Never hallucinate. Format lists with dashes.")
        l.addWidget(self.directives_input)

        l.addStretch()

        btn = QPushButton("Boot Zelyr")
        btn.setObjectName("oobe_primary_btn")
        btn.setFixedWidth(200)
        btn.clicked.connect(self._on_finish)

        h_btn = QHBoxLayout()
        h_btn.addStretch()
        h_btn.addWidget(btn)
        l.addLayout(h_btn)

        self.stack.addWidget(page)

    def _on_finish(self):
        init_map = {
            "Reactive (Wait for prompt)": "reactive",
            "Proactive (Suggest next steps)": "proactive"
        }
        conf_map = {
            "Auto-execute all": "auto_execute",
            "Ask before destructive": "ask_before_destructive",
            "Ask before ANY terminal command": "ask_before_any"
        }

        persona = {
            "identity": {
                "name": self.name_input.text(),
                "role": self.role_input.text()
            },
            "communication": {
                "tone": self.tone_combo.currentText().lower(),
                "verbosity": self.verbosity_combo.currentText().lower(),
                "technical_depth": self.tech_depth_combo.currentText().lower(),
                "formatting_preference": self.formatting_combo.currentText().lower()
            },
            "behavior": {
                "initiative": init_map.get(self.initiative_combo.currentText(), "reactive"),
                "confirmation_policy": conf_map.get(self.confirmation_combo.currentText(), "ask_before_destructive"),
                "show_inner_monologue": self.cb_monologue.isChecked(),
                "provide_confidence_percentage": self.cb_confidence.isChecked(),
                "explain_dangerous_commands": self.cb_explain.isChecked(),
                "use_emojis": self.cb_emojis.isChecked()
            },
            "directives": [line.strip() for line in self.directives_input.toPlainText().split('\n') if line.strip()]
        }

        self.config.persona = persona
        self.config.persona_key = self.config.persona_key or "default"
        self.config.oobe_completed = True
        self.config.save()

        self.accept()
