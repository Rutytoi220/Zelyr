"""Regression check that every shipped theme resolves all of its QSS design tokens.

Originally a root-level manual script (``test_qss.py``) that executed at import
time and was therefore never collected by pytest.  Converted into a collectible,
headless test so an unresolved ``@token@`` placeholder can no longer silently
reach the rendered stylesheet.
"""

from __future__ import annotations

import os
import re

import pytest

# Qt must never try to reach a real display server during CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

TOKEN_PATTERN = re.compile(r"@[a-zA-Z0-9_]+@")

pytest.importorskip("PySide6.QtWidgets", reason="PySide6 is required for GUI theming")


def _themes_dir():
    from pathlib import Path

    from zelyr.gui.styles import theme_manager

    return Path(theme_manager.__file__).parent / "themes"


def test_theme_registry_discovers_shipped_themes():
    """The shipped theme directory must expose at least the default Zelyr theme."""
    from zelyr.gui.styles.theme_registry import ThemeRegistry

    registry = ThemeRegistry(_themes_dir())
    registry.discover_themes()
    theme_ids = registry.list_theme_ids()

    assert theme_ids, "no themes discovered in zelyr/gui/styles/themes"
    assert "zelyr_pro" in theme_ids


@pytest.mark.parametrize("theme_id", ["zelyr_pro", "jarvis", "nordic"])
def test_theme_stylesheet_has_no_unresolved_tokens(theme_id):
    """Applying a theme must not leave raw ``@token@`` placeholders in the QSS."""
    from PySide6.QtWidgets import QApplication

    from zelyr.gui.styles.theme_manager import get_theme_manager

    app = QApplication.instance() or QApplication([])

    manager = get_theme_manager()
    manager.apply_theme(app, theme_id)

    unresolved = set(TOKEN_PATTERN.findall(app.styleSheet()))
    assert not unresolved, f"theme '{theme_id}' left unresolved QSS tokens: {sorted(unresolved)}"
