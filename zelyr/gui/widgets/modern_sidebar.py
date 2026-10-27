import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QTreeWidget,
    QTreeWidgetItem, QLabel, QHBoxLayout, QFrame, QButtonGroup,
    QStyledItemDelegate, QStyle, QStyleOptionViewItem,
)
from PySide6.QtCore import Qt, Signal, QSize, QRect
from PySide6.QtGui import QIcon, QColor, QPainter, QBrush
from zelyr.gui.styles.theme_manager import ThemeManager

from zelyr.gui.styles.squircle import AnimatedSquircleButton


class SquircleItemDelegate(QStyledItemDelegate):
    """Custom delegate that renders a floating, inset rounded-rect selection
    highlight instead of Qt's default full-width band.

    The QSS ``margin`` property on ``QTreeWidget::item`` shifts text content
    but does NOT clip the painted selection background — that band always
    spans the full viewport width.  This delegate intercepts the paint call
    and draws its own highlight geometry: 4 px inset on left/right and 2 px
    inset on top/bottom with an 8 px corner radius.
    """

    #: Fallback colour used for the selection background pill when no theme is active.
    BG_SURFACE_FALLBACK: str = "#161B22"

    def __init__(self, bg_surface_hex: str = BG_SURFACE_FALLBACK, parent=None):
        super().__init__(parent)
        self._bg_surface = QColor(bg_surface_hex)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_surface_color(self, hex_color: str) -> None:
        """Refresh the highlight colour when the theme changes at runtime."""
        self._bg_surface = QColor(hex_color)

    # ------------------------------------------------------------------
    # QStyledItemDelegate overrides
    # ------------------------------------------------------------------
    def paint(self, painter: QPainter, option, index) -> None:  # type: ignore[override]
        """Paint the item, replacing Qt's full-width selection band with a
        floating squircle pill inset by 4 px left/right and 2 px top/bottom."""
        is_selected = bool(option.state & QStyle.StateFlag.State_Selected)
        is_hovered  = bool(option.state & QStyle.StateFlag.State_MouseOver)

        if is_selected or is_hovered:
            # Build an inset rect — 4 px L/R, 2 px T/B.
            inset_rect = QRect(
                option.rect.x() + 4,
                option.rect.y() + 2,
                option.rect.width() - 8,
                option.rect.height() - 4,
            )

            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(self._bg_surface))
            painter.drawRoundedRect(inset_rect, 8.0, 8.0)
            painter.restore()

            # Re-draw text/icons WITHOUT the selection background flag so
            # Qt's default initStyleOption won't paint another fill on top.
            opt = QStyleOptionViewItem(option)
            opt.state &= ~QStyle.StateFlag.State_Selected
            opt.state &= ~QStyle.StateFlag.State_MouseOver
            super().paint(painter, opt, index)
        else:
            super().paint(painter, option, index)

class SegmentedControl(QFrame):
    value_changed = Signal(str)

    def __init__(self, theme_manager: ThemeManager):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setObjectName("segmented_control")
        self.theme_manager = theme_manager
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(4, 4, 4, 4)
        self.layout.setSpacing(4)

        self.buttons = []
        self.active_btn = None

        # QButtonGroup enforces exclusive single-selection natively —
        # this makes the :checked QSS pseudo-state fire without unpolish/polish.
        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)

        modes = ["Basic", "Strict", "Autopilot"]
        for i, mode in enumerate(modes):
            btn = QPushButton(mode)
            btn.setObjectName("segmented_btn")
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setProperty("status", "inactive")

            if i == 0:
                btn.setProperty("position", "first")
            elif i == len(modes) - 1:
                btn.setProperty("position", "last")
            else:
                btn.setProperty("position", "middle")

            btn.clicked.connect(lambda checked, b=btn: self._on_toggled(b))
            self._btn_group.addButton(btn, i)
            self.buttons.append(btn)
            self.layout.addWidget(btn)

        self.buttons[0].setChecked(True)
        self.buttons[0].setProperty("status", "active")
        self.active_btn = self.buttons[0]

    def _on_toggled(self, clicked_btn):
        for btn in self.buttons:
            if btn != clicked_btn:
                btn.setChecked(False)
                btn.setProperty("status", "inactive")
                btn.style().unpolish(btn)
                btn.style().polish(btn)
        clicked_btn.setChecked(True)
        clicked_btn.setProperty("status", "active")
        clicked_btn.style().unpolish(clicked_btn)
        clicked_btn.style().polish(clicked_btn)
        self.active_btn = clicked_btn
        self.value_changed.emit(clicked_btn.text().lower())

class ModernSidebar(QFrame):
    new_chat_requested = Signal()
    new_project_requested = Signal()
    new_project_chat_requested = Signal(str)
    conversation_selected = Signal(str, str)
    mode_changed = Signal(str)
    chat_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.theme_manager = getattr(parent, 'theme_manager', ThemeManager())
        
        self.setFixedWidth(280)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(12, 12, 12, 12)
        self.layout.setSpacing(16)

        self.mode_selector = SegmentedControl(self.theme_manager)
        self.mode_selector.value_changed.connect(self.mode_changed.emit)
        self.layout.addWidget(self.mode_selector)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        
        self.new_chat_btn = AnimatedSquircleButton("+ New Chat", radius=14.0, n=3.2, duration_ms=180)
        self.new_chat_btn.setObjectName("sidebar_action_btn")
        self.new_chat_btn.clicked.connect(self.new_chat_requested.emit)

        self.new_proj_btn = AnimatedSquircleButton("+ Project", radius=14.0, n=3.2, duration_ms=180)
        self.new_proj_btn.setObjectName("sidebar_action_btn")
        self.new_proj_btn.clicked.connect(self.new_project_requested.emit)

        btn_layout.addWidget(self.new_chat_btn)
        btn_layout.addWidget(self.new_proj_btn)
        self.layout.addLayout(btn_layout)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(1)
        self.tree.setHeaderHidden(True)
        self.tree.setRootIsDecorated(False)
        self.tree.setIndentation(10)
        self.tree.setUniformRowHeights(True)
        self.tree.setAnimated(True)
        self.tree.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.tree.setAttribute(Qt.WidgetAttribute.WA_MacShowFocusRect, False)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        # Expand to fill all available space, pushing hub_btn firmly to the bottom
        from PySide6.QtWidgets import QSizePolicy
        self.tree.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.layout.addWidget(self.tree, 1)  # stretch factor 1 = fill remaining space
        self.tree.itemSelectionChanged.connect(self._on_tree_selection)

        # ── Squircle selection delegate ───────────────────────────────────────
        # Pull bg_surface from the active theme so the pill colour is always in
        # sync with the current palette.  Falls back to the hardcoded dark value
        # if no theme is loaded yet.
        _bg_surface = (
            self.theme_manager.theme.get("bg_surface", SquircleItemDelegate.BG_SURFACE_FALLBACK)
            if self.theme_manager and self.theme_manager.theme
            else SquircleItemDelegate.BG_SURFACE_FALLBACK
        )
        self._convo_delegate = SquircleItemDelegate(bg_surface_hex=_bg_surface, parent=self.tree)
        self.tree.setItemDelegate(self._convo_delegate)

        # NO addStretch here — tree's stretch factor pushes hub_btn to bottom

        self.hub_btn = AnimatedSquircleButton("⬡  Zelyr Hub", radius=14.0, n=3.2, duration_ms=180)
        self.hub_btn.setObjectName("sidebar_action_btn")
        self.layout.addWidget(self.hub_btn)


        # settings_btn is kept as a hidden attribute so external callers (main_window.py)
        # can still connect its clicked signal without crashing.  It is NOT added to the
        # layout — the bottom Settings button has been removed from the UI per design spec.
        self.settings_btn = AnimatedSquircleButton("⚙  Settings", radius=14.0, n=3.2, duration_ms=180)
        self.settings_btn.setObjectName("sidebar_action_btn")
        self.settings_btn.setParent(self)   # keeps widget alive, not in layout
        self.settings_btn.hide()

        self._apply_theme()

    def _on_tree_selection(self):
        items = self.tree.selectedItems()
        if not items:
            return
        item = items[0]
        if item.parent():
            chat_id = item.data(0, Qt.ItemDataRole.UserRole)
            project_id = item.parent().data(0, Qt.ItemDataRole.UserRole)
            self.conversation_selected.emit(project_id, chat_id)

    def _show_context_menu(self, pos):
        from PySide6.QtWidgets import QMenu
        item = self.tree.itemAt(pos)
        if item and not item.parent():
            project_id = item.data(0, Qt.ItemDataRole.UserRole)
            menu = QMenu(self)
            new_chat_action = menu.addAction("+ New Chat in Project")
            action = menu.exec(self.tree.mapToGlobal(pos))
            if action == new_chat_action:
                self.new_project_chat_requested.emit(project_id)

    def populate_projects(self, projects_data: list) -> None:
        self.tree.clear()
        
        for p_data in projects_data:
            proj = p_data["project"]
            chats = p_data["chats"]
            
            proj_item = QTreeWidgetItem(self.tree)
            proj_item.setText(0, proj.get("name", "Unnamed Project"))
            proj_item.setData(0, Qt.ItemDataRole.UserRole, proj.get("id"))
            proj_item.setExpanded(True)
            proj_item.setSizeHint(0, QSize(0, 44))
            
            font = proj_item.font(0)
            font.setBold(True)
            proj_item.setFont(0, font)
            proj_item.setFlags(proj_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            
            for chat in chats:
                chat_item = QTreeWidgetItem(proj_item)
                chat_item.setText(0, chat.get("title", "New Chat"))
                chat_item.setData(0, Qt.ItemDataRole.UserRole, chat.get("id"))
                chat_item.setSizeHint(0, QSize(0, 44))

    def _apply_theme(self):
        pass
