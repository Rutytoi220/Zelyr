from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QDialogButtonBox, QCheckBox, QFrame, QLineEdit
)
from PySide6.QtCore import Qt

from zelyr.gui.styles.squircle import AnimatedSquircleButton


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(440)
        
        self.setObjectName("settings_dialog")
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(14)
        
        # Title
        title_label = QLabel("System Settings")
        title_label.setObjectName("settings_title_label")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #FFFFFF;")
        self.layout.addWidget(title_label)
        
        # Mock Model Selector
        self.model_combo = QComboBox()
        self.model_combo.setObjectName("settings_model_combo")
        self.model_combo.addItems(["qwen2.5:1.5b", "qwen3:8b", "MiMo-V2.6", "qwen3-vl:2b"])
        self.layout.addWidget(QLabel("Default LLM Model:"))
        self.layout.addWidget(self.model_combo)
        
        from zelyr.config import get_config
        config = get_config()
        
        self.plugins_checkbox = QCheckBox("Enable Third-Party Plugins")
        self.plugins_checkbox.setObjectName("settings_checkbox")
        self.plugins_checkbox.setChecked(getattr(config, 'allow_third_party_plugins', False))
        
        self.plugins_warning = QLabel("DANGER: Allowing third-party plugins executes unverified Python code on your machine.")
        self.plugins_warning.setObjectName("danger_warning_label")
        self.plugins_warning.setProperty("status", "danger")
        self.plugins_warning.style().unpolish(self.plugins_warning)
        self.plugins_warning.style().polish(self.plugins_warning)
        self.plugins_warning.setWordWrap(True)
        
        self.layout.addWidget(self.plugins_checkbox)
        self.layout.addWidget(self.plugins_warning)
        
        # Persona Selector
        self.persona_combo = QComboBox()
        self.persona_combo.setObjectName("settings_persona_combo")
        from zelyr.core.persona import PERSONA_PRESETS
        self.persona_combo.addItems(list(PERSONA_PRESETS.keys()))
        current_persona = getattr(config, 'persona_key', 'default')
        if current_persona in PERSONA_PRESETS:
            self.persona_combo.setCurrentText(current_persona)
        
        self.layout.addWidget(QLabel("Agent Persona:"))
        self.layout.addWidget(self.persona_combo)
        
        # Wake Word Toggle
        self.wake_word_checkbox = QCheckBox("🎙️ Always-On Wake Word ('hey jarvis')")
        self.wake_word_checkbox.setObjectName("settings_checkbox_wakeword")
        self.wake_word_checkbox.setChecked(getattr(config, 'wake_word_enabled', False))
        self.layout.addWidget(self.wake_word_checkbox)

        # Network Engine Mode Toggle (Local Engine vs Remote Server Engine)
        engine_box = QFrame()
        engine_box.setObjectName("settings_engine_box")
        engine_layout = QVBoxLayout(engine_box)
        engine_layout.setContentsMargins(10, 10, 10, 10)
        engine_layout.setSpacing(6)

        engine_title = QLabel("🌐 Compute Engine & Network")
        engine_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #FFFFFF;")
        engine_layout.addWidget(engine_title)

        self.engine_mode_combo = QComboBox()
        self.engine_mode_combo.setObjectName("settings_engine_mode_combo")
        self.engine_mode_combo.addItems([
            "Local Engine",
            "Remote Server Engine",
            "FastAPI Bridge (localhost:8000)",
        ])
        _mode = getattr(config, 'engine_mode', 'local')
        _mode_idx = {"local": 0, "remote": 1, "fastapi": 2}.get(_mode, 0)
        self.engine_mode_combo.setCurrentIndex(_mode_idx)
        engine_layout.addWidget(self.engine_mode_combo)

        ip_layout = QHBoxLayout()
        ip_label = QLabel("Remote Server IP:Port :")
        self.server_ip_edit = QLineEdit()
        self.server_ip_edit.setObjectName("settings_server_ip_input")
        self.server_ip_edit.setPlaceholderText("127.0.0.1:9412")
        self.server_ip_edit.setText(getattr(config, 'remote_server_ip', '127.0.0.1:9412'))
        self.server_ip_edit.setEnabled(_mode_idx == 1)
        self.engine_mode_combo.currentIndexChanged.connect(
            lambda idx: self.server_ip_edit.setEnabled(idx == 1)
        )
        ip_layout.addWidget(ip_label)
        ip_layout.addWidget(self.server_ip_edit)
        engine_layout.addLayout(ip_layout)

        self.layout.addWidget(engine_box)

        # Theme Builder Button
        self.theme_builder_btn = AnimatedSquircleButton("🎨 Open Sovereign Theme Builder", radius=10.0, n=3.2, duration_ms=180)
        self.theme_builder_btn.setObjectName("settings_theme_builder_btn")
        self.theme_builder_btn.clicked.connect(self._open_theme_builder)
        self.layout.addWidget(self.theme_builder_btn)
        
        self.layout.addStretch(1)
        
        # Button Box
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
        self.layout.addWidget(self.button_box)

    def _open_theme_builder(self):
        from zelyr.gui.widgets.theme_builder_dialog import ThemeBuilderDialog
        dlg = ThemeBuilderDialog(self)
        dlg.exec()
        
    def accept(self):
        from zelyr.config import get_config
        config = get_config()
        config.allow_third_party_plugins = self.plugins_checkbox.isChecked()
        config.wake_word_enabled = self.wake_word_checkbox.isChecked()
        
        config.persona_key = self.persona_combo.currentText()
        from zelyr.core.persona import PERSONA_PRESETS
        config.persona = PERSONA_PRESETS.get(config.persona_key, PERSONA_PRESETS["default"])

        # Network compute mode (0=local, 1=remote, 2=fastapi)
        _idx_to_mode = {0: "local", 1: "remote", 2: "fastapi"}
        config.engine_mode = _idx_to_mode.get(self.engine_mode_combo.currentIndex(), "local")
        config.remote_server_ip = self.server_ip_edit.text().strip() or "127.0.0.1:9412"

        config.save()
        super().accept()
