"""Zelyr Desktop — Theme Engine.

A strict design-token system for the native PySide6 UI.
Loads JSON themes and a QSS template to generate global PySide6 stylesheets.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from zelyr.gui.styles.theme_registry import ThemeRegistry

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

logger = logging.getLogger(__name__)

THEMES_DIR = Path(__file__).parent / "themes"
QSS_TEMPLATE = Path(__file__).parent / "base.qss.template"

# Legacy theme IDs remain accepted for saved user configurations.
LEGACY_THEME_IDS = {"axiom_pro": "zelyr_pro"}

class ThemeManager(QObject):
    """Singleton holder of the active theme and QSS generator."""

    theme_changed = Signal(str)  # Emits the new theme name

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._registry = ThemeRegistry(THEMES_DIR)
        self._active_theme_name: str = ""
        self._active_theme_data: Dict[str, Any] = {}
        self._load_themes()

    def _load_themes(self):
        """Load all JSON themes securely via ThemeRegistry."""
        self._registry.discover_themes()
        # Fallback for old getter logic if needed, but we mostly use self._registry.themes now


    @property
    def theme(self) -> Dict[str, Any]:
        """Return the raw JSON dictionary of the current theme."""
        return self._active_theme_data

    @property
    def active_theme_name(self) -> str:
        return self._active_theme_name

    def apply_theme(self, app: QApplication, theme_name: str = "zelyr_pro") -> None:
        """Apply a theme globally to the QApplication.

        ``axiom_pro`` is accepted as the legacy name of the default theme.
        """
        theme_name = LEGACY_THEME_IDS.get(theme_name, theme_name)
        if theme_name not in self._registry.themes:
            logger.warning(f"Theme '{theme_name}' not found. Falling back to default if available.")
            if not self._registry.themes:
                return
            theme_name = list(self._registry.themes.keys())[0]

        self._active_theme_name = theme_name
        theme_package = self._registry.themes[theme_name]
        self._active_theme_data = theme_package.get("tokens", {})

        if not QSS_TEMPLATE.exists():
            logger.error(f"QSS template not found: {QSS_TEMPLATE}")
            return

        try:
            with open(QSS_TEMPLATE, 'r') as f:
                qss = f.read()

            # Replace tokens
            for key, value in self._active_theme_data.items():
                if isinstance(value, str):
                    qss = qss.replace(f"@{key}@", value)

            app.setStyleSheet(qss)
            logger.info(f"Theme '{theme_name}' applied successfully.")
            self.theme_changed.emit(theme_name)
        except Exception as e:
            logger.error(f"Failed to apply theme '{theme_name}': {e}")


_manager: Optional[ThemeManager] = None

def get_theme_manager() -> ThemeManager:
    global _manager
    if _manager is None:
        _manager = ThemeManager()
    return _manager
