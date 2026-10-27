"""Zelyr Desktop — Apple Squircle & Notion Animation Engine.

Implements mathematically continuous super-ellipse (Lamé curve) paths:
    |x/a|^n + |y/b|^n = 1  (where n ≈ 3.2)
eliminating standard arc curvature discontinuities (curvature jumps from 0 to 1/r).
Also provides Notion-style smooth animations using QEasingCurve.Type.OutExpo.
"""

from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import (
    QRectF, QRect, QPointF, Qt, Property,
    QPropertyAnimation, QEasingCurve, QObject
)
from PySide6.QtGui import (
    QPainter, QPainterPath, QColor, QPen, QBrush, QFontMetrics, QIcon
)
from PySide6.QtWidgets import QPushButton, QFrame, QWidget


class SquirclePath(QPainterPath):
    """Custom QPainterPath generating Apple continuous super-ellipses.

    Mathematical formulation:
        |x / a|^n + |y / b|^n = 1, where n ≈ 3.2 (iOS / macOS continuous curvature).
    
    When radius is None:
        Generates a full super-ellipse bounding the entire rectangle.
    When radius is provided:
        Generates a continuous rounded rectangle where each of the 4 corners
        is a super-ellipse quadrant of degree n ≈ 3.2, smoothly joined by
        tangential straight edges.
    """

    def __init__(
        self,
        rect: QRectF | QRect,
        n: float = 3.2,
        radius: Optional[float] = None,
        steps: int = 128
    ) -> None:
        super().__init__()
        if isinstance(rect, QRect):
            rect = QRectF(rect)
        self._rect = rect
        self._n = n
        self._radius = radius
        self._steps = steps
        self._build_path()

    def _build_path(self) -> None:
        w = self._rect.width()
        h = self._rect.height()
        if w <= 0 or h <= 0:
            return

        x0 = self._rect.x()
        y0 = self._rect.y()
        n = self._n
        power = 2.0 / n

        # If no corner radius is specified, generate pure global super-ellipse
        if self._radius is None:
            a = w / 2.0
            b = h / 2.0
            cx = x0 + a
            cy = y0 + b
            steps = max(32, self._steps)

            for i in range(steps + 1):
                t = (2.0 * math.pi * i) / steps
                ct = math.cos(t)
                st = math.sin(t)
                sgn_ct = 1.0 if ct > 0 else (-1.0 if ct < 0 else 0.0)
                sgn_st = 1.0 if st > 0 else (-1.0 if st < 0 else 0.0)
                px = cx + a * sgn_ct * (abs(ct) ** power)
                py = cy + b * sgn_st * (abs(st) ** power)
                if i == 0:
                    self.moveTo(px, py)
                else:
                    self.lineTo(px, py)
            self.closeSubpath()
            return

        # Continuous squircle corner rounded rect
        max_r = min(w, h) / 2.0
        r = min(self._radius, max_r) if self._radius > 0 else max_r
        steps_corner = max(8, self._steps // 4)

        # 1. Top edge: left-to-right
        self.moveTo(x0 + r, y0)
        self.lineTo(x0 + w - r, y0)

        # 2. Top-Right corner: angle pi/2 down to 0
        cx_tr = x0 + w - r
        cy_tr = y0 + r
        for i in range(1, steps_corner + 1):
            t = (math.pi / 2.0) * (1.0 - i / steps_corner)
            ct = math.cos(t)
            st = math.sin(t)
            dx = (ct ** power) * r
            dy = -(st ** power) * r
            self.lineTo(cx_tr + dx, cy_tr + dy)

        # 3. Right edge
        self.lineTo(x0 + w, y0 + h - r)

        # 4. Bottom-Right corner: angle 0 to pi/2
        cx_br = x0 + w - r
        cy_br = y0 + h - r
        for i in range(1, steps_corner + 1):
            t = (math.pi / 2.0) * (i / steps_corner)
            ct = math.cos(t)
            st = math.sin(t)
            dx = (ct ** power) * r
            dy = (st ** power) * r
            self.lineTo(cx_br + dx, cy_br + dy)

        # 5. Bottom edge: right-to-left
        self.lineTo(x0 + r, y0 + h)

        # 6. Bottom-Left corner: angle pi/2 down to 0
        cx_bl = x0 + r
        cy_bl = y0 + h - r
        for i in range(1, steps_corner + 1):
            t = (math.pi / 2.0) * (1.0 - i / steps_corner)
            ct = math.cos(t)
            st = math.sin(t)
            dx = -(ct ** power) * r
            dy = (st ** power) * r
            self.lineTo(cx_bl + dx, cy_bl + dy)

        # 7. Left edge: bottom-to-top
        self.lineTo(x0, y0 + r)

        # 8. Top-Left corner: angle 0 to pi/2
        cx_tl = x0 + r
        cy_tl = y0 + r
        for i in range(1, steps_corner + 1):
            t = (math.pi / 2.0) * (i / steps_corner)
            ct = math.cos(t)
            st = math.sin(t)
            dx = -(ct ** power) * r
            dy = -(st ** power) * r
            self.lineTo(cx_tl + dx, cy_tl + dy)

        self.closeSubpath()


def create_squircle_path(
    rect: QRectF | QRect,
    n: float = 3.2,
    radius: Optional[float] = None,
    steps: int = 128
) -> SquirclePath:
    """Create a SquirclePath with super-ellipse degree n ≈ 3.2."""
    return SquirclePath(rect, n=n, radius=radius, steps=steps)


def paint_squircle(
    painter: QPainter,
    rect: QRectF | QRect,
    bg_color: QColor | str,
    border_color: Optional[QColor | str] = None,
    border_width: float = 1.0,
    radius: Optional[float] = None,
    n: float = 3.2
) -> None:
    """Utility to render an antialiased squircle on a QPainter."""
    if isinstance(rect, QRect):
        rect = QRectF(rect)
    if isinstance(bg_color, str):
        bg_color = QColor(bg_color)
    if isinstance(border_color, str):
        border_color = QColor(border_color)

    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    draw_rect = rect.adjusted(
        border_width / 2.0, border_width / 2.0,
        -border_width / 2.0, -border_width / 2.0
    )
    path = SquirclePath(draw_rect, n=n, radius=radius)

    painter.fillPath(path, QBrush(bg_color))

    if border_color and border_color.alpha() > 0 and border_width > 0:
        pen = QPen(border_color, border_width)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawPath(path)

    painter.restore()


class AnimatedSquircleButton(QPushButton):
    """A QPushButton with continuous Apple squircle geometry and Notion OutExpo hover animation."""

    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
        radius: float = 14.0,
        n: float = 3.2,
        duration_ms: int = 180
    ) -> None:
        super().__init__(text, parent)
        self._radius = radius
        self._n = n
        self._duration_ms = duration_ms
        self._hover_progress = 0.0

        self._anim = QPropertyAnimation(self, b"hoverProgress", self)
        self._anim.setDuration(self._duration_ms)
        self._anim.setEasingCurve(QEasingCurve.Type.OutExpo)

        # Style hooks
        self._custom_bg = None
        self._custom_hover_bg = None
        self._custom_border = None
        self._custom_text = None
        self._custom_hover_text = None

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

    def get_hover_progress(self) -> float:
        return self._hover_progress

    def set_hover_progress(self, val: float) -> None:
        self._hover_progress = val
        self.update()

    hoverProgress = Property(float, get_hover_progress, set_hover_progress)

    def enterEvent(self, event) -> None:
        self._anim.stop()
        self._anim.setDuration(self._duration_ms)
        self._anim.setEasingCurve(QEasingCurve.Type.OutExpo)
        self._anim.setStartValue(self._hover_progress)
        self._anim.setEndValue(1.0)
        self._anim.start()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._anim.stop()
        self._anim.setDuration(self._duration_ms)
        self._anim.setEasingCurve(QEasingCurve.Type.OutExpo)
        self._anim.setStartValue(self._hover_progress)
        self._anim.setEndValue(0.0)
        self._anim.start()
        super().leaveEvent(event)

    def set_colors(
        self,
        bg: str | QColor,
        hover_bg: str | QColor,
        border: Optional[str | QColor] = None,
        text: Optional[str | QColor] = None,
        hover_text: Optional[str | QColor] = None
    ) -> None:
        self._custom_bg = QColor(bg)
        self._custom_hover_bg = QColor(hover_bg)
        self._custom_border = QColor(border) if border else None
        self._custom_text = QColor(text) if text else None
        self._custom_hover_text = QColor(hover_text) if hover_text else None
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = SquirclePath(rect, n=self._n, radius=self._radius)

        # Base vs hover color interpolation
        if self._custom_bg and self._custom_hover_bg:
            r = int(self._custom_bg.red() + (self._custom_hover_bg.red() - self._custom_bg.red()) * self._hover_progress)
            g = int(self._custom_bg.green() + (self._custom_hover_bg.green() - self._custom_bg.green()) * self._hover_progress)
            b = int(self._custom_bg.blue() + (self._custom_hover_bg.blue() - self._custom_bg.blue()) * self._hover_progress)
            a = int(self._custom_bg.alpha() + (self._custom_hover_bg.alpha() - self._custom_bg.alpha()) * self._hover_progress)
            bg_col = QColor(r, g, b, a)
        else:
            # Default dark mode theme values
            base_alpha = 18 if not self.isChecked() else 255
            hover_alpha = int(base_alpha + (60 if not self.isChecked() else 0) * self._hover_progress)
            if self.isChecked():
                bg_col = QColor(139, 92, 246)  # Primary purple
            else:
                bg_col = QColor(255, 255, 255, hover_alpha)

        painter.fillPath(path, QBrush(bg_col))

        if self._custom_border:
            pen = QPen(self._custom_border, 1.0)
            painter.setPen(pen)
            painter.drawPath(path)

        # Draw icon and text
        painter.setFont(self.font())
        if self._custom_text and self._custom_hover_text:
            tr = int(self._custom_text.red() + (self._custom_hover_text.red() - self._custom_text.red()) * self._hover_progress)
            tg = int(self._custom_text.green() + (self._custom_hover_text.green() - self._custom_text.green()) * self._hover_progress)
            tb = int(self._custom_text.blue() + (self._custom_hover_text.blue() - self._custom_text.blue()) * self._hover_progress)
            text_col = QColor(tr, tg, tb)
        else:
            text_col = QColor(255, 255, 255) if (self.isChecked() or self._hover_progress > 0.5) else QColor(200, 205, 215)
        painter.setPen(text_col)

        opt_icon = self.icon()
        if not opt_icon.isNull():
            icon_size = self.iconSize()
            icon_rect = QRectF(
                rect.x() + 12,
                rect.y() + (rect.height() - icon_size.height()) / 2.0,
                icon_size.width(),
                icon_size.height()
            )
            opt_icon.paint(painter, icon_rect.toRect())
            text_rect = QRectF(
                rect.x() + 12 + icon_size.width() + 8,
                rect.y(),
                rect.width() - (24 + icon_size.width()),
                rect.height()
            )
        else:
            text_rect = rect

        align = Qt.AlignmentFlag.AlignCenter if not self.text().startswith("+") and not self.text().startswith("⬡") else (Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        if align & Qt.AlignmentFlag.AlignLeft:
            text_rect = text_rect.adjusted(14, 0, -14, 0)
        painter.drawText(text_rect, align, self.text())
