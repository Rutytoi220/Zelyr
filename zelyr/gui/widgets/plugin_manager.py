from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QCheckBox, QPushButton, QScrollArea, QWidget, QFrame
)
from PySide6.QtCore import Qt, Signal
import logging

from zelyr.config import get_config

logger = logging.getLogger(__name__)

from zelyr.gui.styles.squircle import SquirclePath, AnimatedSquircleButton
from PySide6.QtCore import QPropertyAnimation, QEasingCurve, Property, QRectF
from PySide6.QtGui import QPainter, QBrush, QPen, QColor

class PluginCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("plugin_card")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setContentsMargins(14, 10, 14, 10)
        self._hover_progress = 0.0

        self._anim = QPropertyAnimation(self, b"hoverProgress", self)
        self._anim.setDuration(180)
        self._anim.setEasingCurve(QEasingCurve.Type.OutExpo)

    def get_hover_progress(self) -> float:
        return self._hover_progress

    def set_hover_progress(self, val: float) -> None:
        self._hover_progress = val
        self.update()

    hoverProgress = Property(float, get_hover_progress, set_hover_progress)

    def enterEvent(self, event):
        self._anim.stop()
        self._anim.setDuration(180)
        self._anim.setEasingCurve(QEasingCurve.Type.OutExpo)
        self._anim.setStartValue(self._hover_progress)
        self._anim.setEndValue(1.0)
        self._anim.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._anim.stop()
        self._anim.setDuration(180)
        self._anim.setEasingCurve(QEasingCurve.Type.OutExpo)
        self._anim.setStartValue(self._hover_progress)
        self._anim.setEndValue(0.0)
        self._anim.start()
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        if rect.width() <= 0 or rect.height() <= 0:
            return

        # Continuous super-ellipse curvature with degree n ≈ 3.2
        path = SquirclePath(rect, n=3.2, radius=16.0)

        # Dynamic hover highlight
        alpha_add = int(self._hover_progress * 15)
        bg = QColor(22 + alpha_add, 27 + alpha_add, 34 + alpha_add)
        painter.fillPath(path, QBrush(bg))

        # Border transition on hover
        border_r = int(48 + (139 - 48) * self._hover_progress)
        border_g = int(54 + (92 - 54) * self._hover_progress)
        border_b = int(61 + (246 - 61) * self._hover_progress)
        pen = QPen(QColor(border_r, border_g, border_b), 1.0)
        painter.setPen(pen)
        painter.drawPath(path)

class PluginManagerDialog(QDialog):
    """Dialog to manage loaded plugins and toggle them on/off."""
    
    plugins_updated = Signal()
    
    def __init__(self, bridge, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Zelyr Plugin Manager")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        self.config = get_config()
        self._bridge = bridge
        self._loading = True
        
        self._build_ui()
        
        # Connect to bridge to receive tools, then request them
        self._bridge.tools_received.connect(self._populate_tools)
        self._bridge.request_tools()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        
        title = QLabel("Plugin Hub")
        title.setObjectName("hub_name")
        layout.addWidget(title)
        
        desc = QLabel("Enable or disable Zelyr tools and plugins. Changes apply instantly to the active agent.")
        desc.setObjectName("hub_desc")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setObjectName("plugin_scroll")
        
        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setSpacing(10)
        
        self.loading_lbl = QLabel("Fetching plugins from daemon...")
        self.loading_lbl.setObjectName("plugin_loading")
        self.container_layout.addWidget(self.loading_lbl)
        
        self.container_layout.addStretch()
        scroll.setWidget(self.container)
        layout.addWidget(scroll)
        
        close_btn = AnimatedSquircleButton("Close", radius=10.0, n=3.2, duration_ms=180)
        close_btn.setFixedWidth(100)
        close_btn.setObjectName("plugin_close")
        close_btn.clicked.connect(self.accept)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def _populate_tools(self, tools_list: list):
        while self.container_layout.count():
            item = self.container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        self._loading = True
                
        for t in tools_list:
            tool_id = t["id"]
            desc_text = t["description"]
            is_enabled = t["enabled"]
            
            card = PluginCard()
            card_layout = QHBoxLayout(card)
            
            info_layout = QVBoxLayout()
            name_lbl = QLabel(tool_id)
            name_lbl.setObjectName("plugin_name")
            
            desc_lbl = QLabel(desc_text)
            desc_lbl.setObjectName("plugin_desc")
            desc_lbl.setWordWrap(True)
            
            info_layout.addWidget(name_lbl)
            info_layout.addWidget(desc_lbl)
            card_layout.addLayout(info_layout)
            
            toggle = QCheckBox("Enabled" if is_enabled else "Disabled")
            toggle.setChecked(is_enabled)
            toggle.setProperty("tool_id", tool_id)
            toggle.setObjectName("plugin_toggle")
            toggle.toggled.connect(self._on_plugin_toggled)
            
            card_layout.addWidget(toggle)
            self.container_layout.addWidget(card)
            
        self.container_layout.addStretch()
        self._loading = False

    def _on_plugin_toggled(self, checked: bool):
        if self._loading:
            return
            
        sender = self.sender()
        if not sender:
            return
            
        tool_id = sender.property("tool_id")
        
        sender.setText("Enabled" if checked else "Disabled")
        
        self._bridge.toggle_tool(tool_id, checked)
        
        logger.info(f"Plugin {tool_id} IPC toggle requested (enabled={checked}).")
        self.plugins_updated.emit()
