"""Zelyr Sovereign Theme Builder Dialog.

A graphical live theme editor and QSS engine for Zelyr.
Allows picking colors for all 16 schema tokens using QColorDialog,
dynamically injecting them into base.qss.template, applying the stylesheet live
to QApplication, and persisting custom themes to themes/custom.json.
"""

from __future__ import annotations

import json
import logging
from functools import partial
from pathlib import Path
from typing import Dict, Any, Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QColor, QPainter, QBrush, QPen, QFont
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QColorDialog, QComboBox, QScrollArea, QWidget,
    QFrame, QMessageBox, QApplication
)

from zelyr.gui.styles.squircle import SquirclePath, AnimatedSquircleButton
from zelyr.gui.styles.theme_manager import get_theme_manager, THEMES_DIR, QSS_TEMPLATE

logger = logging.getLogger(__name__)

# The 16 design schema tokens
SCHEMA_TOKENS: list[tuple[str, str, str]] = [
    ("bg_base", "Base Background", "#0D1117"),
    ("bg_surface", "Surface / Panels", "#161B22"),
    ("bg_deep", "Deep Dark Base", "#090D13"),
    ("primary", "Primary Brand", "#8B5CF6"),
    ("accent", "Accent Highlight", "#A78BFA"),
    ("accent_hover", "Accent Hover", "#7C3AED"),
    ("text_main", "Main Text", "#FAFAFA"),
    ("text_muted", "Muted / Dimmed Text", "#8B949E"),
    ("borders", "Borders & Outlines", "#30363D"),
    ("danger", "Danger / Error", "#EF4444"),
    ("success", "Success / Online", "#10B981"),
    ("warning", "Warning / Attention", "#F59E0B"),
    ("info", "Information / Badges", "#3B82F6"),
    ("card_bg", "Card Background", "#1F242C"),
    ("input_bg", "Input Field Background", "#0B0E14"),
    ("selection", "Selection Highlight", "#4C1D95"),
]

PRESETS: dict[str, dict[str, str]] = {
    "Zelyr Pro (Default)": {
        "bg_base": "#0D1117",
        "bg_surface": "#161B22",
        "bg_deep": "#090D13",
        "primary": "#8B5CF6",
        "accent": "#A78BFA",
        "accent_hover": "#7C3AED",
        "text_main": "#FAFAFA",
        "text_muted": "#8B949E",
        "borders": "#30363D",
        "danger": "#EF4444",
        "success": "#10B981",
        "warning": "#F59E0B",
        "info": "#3B82F6",
        "card_bg": "#1F242C",
        "input_bg": "#0B0E14",
        "selection": "#4C1D95",
    },
    "High-Contrast Neon": {
        "bg_base": "#000000",
        "bg_surface": "#080C10",
        "bg_deep": "#000000",
        "primary": "#00FFCC",
        "accent": "#FF007F",
        "accent_hover": "#00E6B8",
        "text_main": "#FFFFFF",
        "text_muted": "#00FFCC",
        "borders": "#00FFCC",
        "danger": "#FF1744",
        "success": "#00E676",
        "warning": "#FFEA00",
        "info": "#00E5FF",
        "card_bg": "#0D1117",
        "input_bg": "#000000",
        "selection": "#FF007F",
    },
    "Jarvis Cyberpunk": {
        "bg_base": "#000000",
        "bg_surface": "#0A0A0A",
        "bg_deep": "#050505",
        "primary": "#00FFFF",
        "accent": "#00AAAA",
        "accent_hover": "#008888",
        "text_main": "#00FFFF",
        "text_muted": "#004444",
        "borders": "#003333",
        "danger": "#FF4444",
        "success": "#00FF88",
        "warning": "#FFAA00",
        "info": "#00FFFF",
        "card_bg": "#0A0A0A",
        "input_bg": "#050505",
        "selection": "#00AAAA",
    },
    "Nordic Frost": {
        "bg_base": "#2E3440",
        "bg_surface": "#3B4252",
        "bg_deep": "#242933",
        "primary": "#88C0D0",
        "accent": "#81A1C1",
        "accent_hover": "#8FBCBB",
        "text_main": "#ECEFF4",
        "text_muted": "#D8DEE9",
        "borders": "#4C566A",
        "danger": "#BF616A",
        "success": "#A3BE8C",
        "warning": "#EBCB8B",
        "info": "#5E81AC",
        "card_bg": "#3B4252",
        "input_bg": "#2E3440",
        "selection": "#88C0D0",
    }
}


class ColorSwatchButton(QPushButton):
    """Button displaying a squircle color swatch with hex label."""

    def __init__(self, color_hex: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._color = QColor(color_hex)
        self.setText(self._color.name().upper())
        self.setFixedHeight(34)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_color(self, color: QColor | str) -> None:
        self._color = QColor(color) if isinstance(color, str) else color
        self.setText(self._color.name().upper())
        self.update()

    def get_color(self) -> QColor:
        return self._color

    def get_hex(self) -> str:
        return self._color.name().upper()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = self.rect().adjusted(1, 1, -1, -1)
        path = SquirclePath(rect, n=3.2, radius=8.0)

        # Swatch fill
        painter.fillPath(path, QBrush(self._color))

        # Border
        is_light = self._color.lightnessF() > 0.5
        border_col = QColor(0, 0, 0, 100) if is_light else QColor(255, 255, 255, 60)
        painter.setPen(QPen(border_col, 1.2))
        painter.drawPath(path)

        # Draw Hex string on top of the swatch with high contrast
        text_col = QColor("#000000") if is_light else QColor("#FFFFFF")
        painter.setPen(text_col)
        font = QFont("JetBrains Mono", 9)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._color.name().upper())


class TokenRowWidget(QFrame):
    """Row widget representing one schema token editor."""

    color_changed = Signal(str, str)  # token_key, new_hex

    def __init__(
        self,
        token_key: str,
        display_name: str,
        initial_hex: str,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.token_key = token_key
        self.setObjectName(f"token_row_{token_key}")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(10)

        # Token key and description label
        label_box = QVBoxLayout()
        label_box.setSpacing(2)

        key_label = QLabel(token_key)
        key_label.setStyleSheet("font-family: 'JetBrains Mono', monospace; font-weight: bold; font-size: 11px; color: #FFFFFF;")
        
        name_label = QLabel(display_name)
        name_label.setStyleSheet("font-size: 10px; color: #8B949E;")

        label_box.addWidget(key_label)
        label_box.addWidget(name_label)
        layout.addLayout(label_box, 1)

        # Color Swatch Button
        self.swatch = ColorSwatchButton(initial_hex)
        self.swatch.setObjectName(f"swatch_{token_key}")
        self.swatch.setFixedWidth(90)
        layout.addWidget(self.swatch)

        # Explicit Pick Button
        self.pick_btn = AnimatedSquircleButton("Pick", radius=8.0, n=3.2, duration_ms=150)
        self.pick_btn.setObjectName(f"pick_btn_{token_key}")
        self.pick_btn.setFixedSize(56, 32)
        layout.addWidget(self.pick_btn)

        # Reference to swatch which renders the hex label
        self.hex_label = self.swatch

    def _pick_color(self) -> None:
        self.pick_btn.click()

    def set_color(self, hex_val: str) -> None:
        self.swatch.set_color(hex_val)
        self.color_changed.emit(self.token_key, hex_val)

    def get_hex(self) -> str:
        return self.swatch.get_hex()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(0, 0, 0, 0)
        path = SquirclePath(rect, n=3.2, radius=10.0)
        painter.fillPath(path, QBrush(QColor(255, 255, 255, 8)))
        painter.setPen(QPen(QColor(255, 255, 255, 18), 1.0))
        painter.drawPath(path)


class ThemeBuilderDialog(QDialog):
    """The Sovereign Theme Builder — Interactive Live Theme Customizer & Compiler."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("theme_builder_dialog")
        self.setWindowTitle("🎨 Zelyr Sovereign Theme Builder")
        self.resize(780, 680)

        self._theme_manager = get_theme_manager()
        self._current_tokens: dict[str, str] = {}
        self._token_widgets: dict[str, TokenRowWidget] = {}

        self._build_ui()
        self._load_initial_values()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(14)

        # Header section
        header_card = QFrame()
        header_card.setObjectName("header_card")
        header_layout = QHBoxLayout(header_card)
        header_layout.setContentsMargins(12, 10, 12, 10)

        title_vbox = QVBoxLayout()
        title_lbl = QLabel("Sovereign Theme Builder")
        title_lbl.setStyleSheet("font-size: 18px; font-weight: bold; color: #FFFFFF;")
        sub_lbl = QLabel("Live token stylesheet compiler & high-contrast engine")
        sub_lbl.setStyleSheet("font-size: 11px; color: #8B949E;")
        title_vbox.addWidget(title_lbl)
        title_vbox.addWidget(sub_lbl)
        header_layout.addLayout(title_vbox, 1)

        # Preset Selector
        preset_box = QHBoxLayout()
        preset_lbl = QLabel("Presets:")
        preset_lbl.setStyleSheet("color: #FFFFFF; font-weight: bold;")
        self.preset_combo = QComboBox()
        self.preset_combo.setObjectName("theme_preset_combo")
        self.preset_combo.addItems(list(PRESETS.keys()))
        self.preset_combo.currentTextChanged.connect(self._on_preset_selected)
        preset_box.addWidget(preset_lbl)
        preset_box.addWidget(self.preset_combo)
        header_layout.addLayout(preset_box)

        main_layout.addWidget(header_card)

        # Scroll area for 16 schema token editors
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setObjectName("theme_tokens_scroll")
        scroll.setStyleSheet("QScrollArea { border: 1px solid #30363D; border-radius: 12px; background: #0D1117; }")

        grid_container = QWidget()
        grid_container.setStyleSheet("background: transparent;")
        grid_layout = QGridLayout(grid_container)
        grid_layout.setContentsMargins(10, 10, 10, 10)
        grid_layout.setSpacing(10)

        # Arrange all 16 tokens in 2 columns
        for idx, (token_key, display_name, default_hex) in enumerate(SCHEMA_TOKENS):
            row_idx = idx // 2
            col_idx = idx % 2

            row_widget = TokenRowWidget(token_key, display_name, default_hex)
            row_widget.color_changed.connect(self._on_color_changed)
            row_widget.pick_btn.clicked.connect(lambda *args, k=token_key: self._open_color_picker(k))
            row_widget.swatch.clicked.connect(lambda *args, k=token_key: self._open_color_picker(k))
            self._token_widgets[token_key] = row_widget
            grid_layout.addWidget(row_widget, row_idx, col_idx)

        scroll.setWidget(grid_container)
        main_layout.addWidget(scroll, 1)

        # Live Preview Banner
        self.preview_banner = QFrame()
        self.preview_banner.setFixedHeight(46)
        banner_layout = QHBoxLayout(self.preview_banner)
        banner_layout.setContentsMargins(14, 0, 14, 0)
        self.preview_lbl = QLabel("Live Preview: Active theme tokens compiled in real-time.")
        self.preview_lbl.setStyleSheet("font-size: 11px; font-weight: bold;")
        banner_layout.addWidget(self.preview_lbl)
        main_layout.addWidget(self.preview_banner)

        # Action Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        self.reset_btn = AnimatedSquircleButton("Reset Default", radius=12.0, n=3.2, duration_ms=180)
        self.reset_btn.setObjectName("reset_default_btn")
        self.reset_btn.setFixedSize(130, 38)
        self.reset_btn.clicked.connect(self._reset_default)
        btn_layout.addWidget(self.reset_btn)

        btn_layout.addStretch(1)

        self.cancel_btn = AnimatedSquircleButton("Cancel", radius=12.0, n=3.2, duration_ms=180)
        self.cancel_btn.setObjectName("cancel_theme_btn")
        self.cancel_btn.setFixedSize(100, 38)
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)

        self.apply_btn = AnimatedSquircleButton("Apply & Save Theme", radius=12.0, n=3.2, duration_ms=180)
        self.apply_btn.setFixedSize(160, 38)
        self.apply_btn.setObjectName("apply_theme_btn")
        self.apply_btn.clicked.connect(self._apply_theme)
        btn_layout.addWidget(self.apply_btn)

        main_layout.addLayout(btn_layout)

    def _load_initial_values(self) -> None:
        active_tokens = self._theme_manager.theme
        for token_key, _, default_hex in SCHEMA_TOKENS:
            current_val = active_tokens.get(token_key, default_hex)
            if not current_val.startswith("#"):
                current_val = default_hex
            self._current_tokens[token_key] = current_val
            if token_key in self._token_widgets:
                self._token_widgets[token_key].set_color(current_val)
        self._update_preview_banner()

    def _on_preset_selected(self, preset_name: str) -> None:
        if preset_name in PRESETS:
            preset = PRESETS[preset_name]
            for token_key, val in preset.items():
                if token_key in self._token_widgets:
                    self._token_widgets[token_key].set_color(val)
                    self._current_tokens[token_key] = val
            self._update_preview_banner()

    def _on_color_changed(self, token_key: str, new_hex: str) -> None:
        self._current_tokens[token_key] = new_hex
        self._update_preview_banner()

    def _update_preview_banner(self) -> None:
        primary = self._current_tokens.get("primary", "#8B5CF6")
        surface = self._current_tokens.get("bg_surface", "#161B22")
        text = self._current_tokens.get("text_main", "#FFFFFF")
        borders = self._current_tokens.get("borders", "#30363D")

        self.preview_banner.setStyleSheet(f"""
            QFrame {{
                background-color: {surface};
                border: 1px solid {borders};
                border-radius: 12px;
            }}
            QLabel {{
                color: {text};
            }}
        """)
        self.preview_lbl.setText(f"Active Palette: Primary {primary} · Surface {surface} · Text {text}")

    @Slot(str)
    def _open_color_picker(self, token_name: str, *args: Any, **kwargs: Any) -> None:
        """Open QColorDialog for a token, update hex code display, and store new color in working dictionary."""
        initial_hex = self._current_tokens.get(token_name, "#FAFAFA")
        dialog = QColorDialog(QColor(initial_hex), self)
        dialog.setWindowTitle(f"Select Color for {token_name}")
        if dialog.exec():
            color = dialog.selectedColor()
            if not color.isValid():
                color = dialog.currentColor()
            if color.isValid():
                hex_str = color.name().upper()
                self._current_tokens[token_name] = hex_str
                if token_name in self._token_widgets:
                    self._token_widgets[token_name].set_color(hex_str)
                self._update_preview_banner()

    @Slot()
    def _reset_default(self, *args: Any, **kwargs: Any) -> None:
        """Reload the default schema tokens into the UI and local working dictionary."""
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentText("Zelyr Pro (Default)")
        self.preset_combo.blockSignals(False)
        for token_key, _, default_hex in SCHEMA_TOKENS:
            self._current_tokens[token_key] = default_hex
            if token_key in self._token_widgets:
                self._token_widgets[token_key].set_color(default_hex)
        self._update_preview_banner()

    _on_reset = _reset_default

    @Slot()
    def _apply_theme(self, *args: Any, **kwargs: Any) -> None:
        """Slot to save theme dictionary to themes/custom.json, apply live, and accept dialog."""
        self.apply_theme_live(close_on_apply=True)

    def apply_theme_live(self, close_on_apply: bool = True) -> None:
        """Inject colors into base.qss.template, apply live to QApplication, and save themes/custom.json."""
        # 1. Collect all token values
        for token_key, widget in self._token_widgets.items():
            self._current_tokens[token_key] = widget.get_hex()

        # Build full token dictionary including geometry and typography tokens
        full_tokens = dict(self._current_tokens)
        full_tokens.setdefault("spacing_sm", "8px")
        full_tokens.setdefault("spacing_md", "16px")
        full_tokens.setdefault("radius_sm", "4px")
        full_tokens.setdefault("radius_md", "8px")
        full_tokens.setdefault("radius_lg", "12px")
        full_tokens.setdefault("radius_pill", "20px")
        full_tokens.setdefault("radius_circle", "20px")
        full_tokens.setdefault("font_main", "'Inter', sans-serif")
        full_tokens.setdefault("font_mono", "'JetBrains Mono', monospace")

        # 2. Inject into base.qss.template
        if QSS_TEMPLATE.exists():
            try:
                with open(QSS_TEMPLATE, "r", encoding="utf-8") as f:
                    qss_content = f.read()

                for k, v in full_tokens.items():
                    qss_content = qss_content.replace(f"@{k}@", v)

                # 3. Apply live to QApplication
                app = QApplication.instance()
                if app:
                    app.setStyleSheet(qss_content)
                    logger.info("Live stylesheet applied to QApplication successfully.")

                # Update theme manager internal state and broadcast signal
                self._theme_manager._active_theme_data = full_tokens
                self._theme_manager._active_theme_name = "custom"
                self._theme_manager.theme_changed.emit("custom")

            except Exception as e:
                logger.error(f"Failed to inject theme stylesheet: {e}")
                QMessageBox.critical(self, "Theme Engine Error", f"Failed to apply stylesheet: {e}")
                return

        # 4. Save output as themes/custom.json
        manifest_data = {
            "id": "custom",
            "name": "Custom Sovereign",
            "author": "Zelyr User",
            "version": "1.0.0",
            "tokens": full_tokens
        }

        # Save to both THEMES_DIR/custom.json and project-root themes/custom.json
        save_paths = [
            THEMES_DIR / "custom.json",
            Path("themes/custom.json"),
            Path(__file__).resolve().parent.parent.parent.parent / "themes" / "custom.json"
        ]

        for p in save_paths:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(manifest_data, f, indent=4)
                logger.info(f"Custom theme saved to {p}")
            except Exception as e:
                logger.warning(f"Failed saving custom theme to {p}: {e}")

        # Refresh registry so custom theme is permanently available
        self._theme_manager._registry.discover_themes()
        if close_on_apply:
            self.accept()


def main():
    """Standalone entrypoint for theme builder visual verification."""
    import sys
    app = QApplication.instance() or QApplication(sys.argv)
    
    tm = get_theme_manager()
    tm.apply_theme(app, "zelyr_pro")

    dialog = ThemeBuilderDialog()
    # Check if simulation argument provided
    if "--neon" in sys.argv:
        dialog.preset_combo.setCurrentText("High-Contrast Neon")
        dialog.apply_theme_live(close_on_apply=False)

    dialog.showMaximized()
    dialog.raise_()
    dialog.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
