"""Backward-compatible shim.

The color/geometry/typography constants that used to live in this module
(and the ``QGraphicsDropShadowEffect``-based ``apply_glow`` helper) have been
replaced by the dynamic token engine in :mod:`zelyr.gui.styles.theme_manager`
— see that module for the ``Theme`` / ``ThemeManager`` / ``ThemeRegistry``
architecture. Widgets should import ``get_theme_manager`` directly rather
than this module; it is kept only so any stray external import of
``zelyr.gui.styles.palette`` does not hard-fail.
"""
from __future__ import annotations

from zelyr.gui.styles.theme_manager import (  # noqa: F401
    ColorTokens,
    GeometryTokens,
    Theme,
    ThemeManager,
    ThemeRegistry,
    TypographyTokens,
    get_theme_manager,
)
