"""System Hub Dialog.

A centralized, clean grid UI hosting all secondary modules that were
pruned from the crowded top toolbar in v6.0 LTS.
"""
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QGridLayout,
    QPushButton,
    QLabel,
    QFrame
)
from PySide6.QtCore import Qt, QSize
import logging

logger = logging.getLogger(__name__)

class SystemHubDialog(QDialog):
    """Centralized control center for Zelyr modules."""

    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.setWindowTitle("Zelyr System Hub")
        self.setMinimumSize(600, 400)
        pass
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("<h2>System Hub</h2>")
        header.setObjectName("hub_name")
        layout.addWidget(header)
        
        # Profile Switcher
        from zelyr.services.profile_service import ProfileService, ProfileLevel
        from PySide6.QtWidgets import QHBoxLayout, QButtonGroup
        ps = ProfileService.instance()
        current_profile = ps.get_profile()

        profile_layout = QHBoxLayout()
        profile_layout.setContentsMargins(0, 0, 0, 15)
        profile_label = QLabel("UI Profile:")
        profile_label.setObjectName("hub_author")
        profile_layout.addWidget(profile_label)

        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)

        btn_std = QPushButton("Standard")
        btn_adv = QPushButton("Advanced")
        btn_dev = QPushButton("Developer")
        
        btn_std.setCheckable(True)
        btn_adv.setCheckable(True)
        btn_dev.setCheckable(True)

        if current_profile == ProfileLevel.STANDARD:
            btn_std.setChecked(True)
        elif current_profile == ProfileLevel.ADVANCED:
            btn_adv.setChecked(True)
        elif current_profile == ProfileLevel.DEVELOPER:
            btn_dev.setChecked(True)

        for b, lvl in [(btn_std, ProfileLevel.STANDARD), (btn_adv, ProfileLevel.ADVANCED), (btn_dev, ProfileLevel.DEVELOPER)]:
            b.setObjectName("plugin_card")
            b.clicked.connect(lambda checked=False, l=lvl: ps.set_profile(l))
            self.btn_group.addButton(b)
            profile_layout.addWidget(b)

        profile_layout.addStretch()
        layout.addLayout(profile_layout)
        
        # Grid Layout
        grid = QGridLayout()
        grid.setSpacing(15)
        
        # Define the buttons to map to the main_window's methods
        buttons = [
            ("⏱️ Automation", "#f9e2af", self.main_window._open_scheduler_dialog if hasattr(self.main_window, '_open_scheduler_dialog') else None),
            ("💡 IoT / Physical", "#f9e2af", self.main_window._open_iot_dialog if hasattr(self.main_window, '_open_iot_dialog') else None),
            ("🛡️ eBPF Firewall", "#f38ba8", self.main_window._open_firewall_dialog if hasattr(self.main_window, '_open_firewall_dialog') else None),
            ("🛡️ Security Log", "#fab387", self.main_window._open_audit_dialog if hasattr(self.main_window, '_open_audit_dialog') else None),
            ("❤️ System Health", "#f38ba8", self.main_window._open_health_radar if hasattr(self.main_window, '_open_health_radar') else None),
            ("🕸️ Entity Graph", "#cba6f7", self.main_window._open_graph_dialog if hasattr(self.main_window, '_open_graph_dialog') else None),
            ("⏱️ Recall Timeline", "#89b4fa", self.main_window._open_recall_dialog if hasattr(self.main_window, '_open_recall_dialog') else None),
            ("🛡️ Cyber Audit", "#fab387", self.main_window._open_security_dialog if hasattr(self.main_window, '_open_security_dialog') else None),
            ("🧩 Skill Library", "#a6e3a1", self.main_window._open_skill_dialog if hasattr(self.main_window, '_open_skill_dialog') else None),
            ("📊 Telemetry Trace", "#89dceb", self.main_window._open_telemetry_dialog if hasattr(self.main_window, '_open_telemetry_dialog') else None),
            ("Settings", "#cdd6f4", self.main_window._open_settings_dialog if hasattr(self.main_window, '_open_settings_dialog') else None),
        ]
        
        row, col = 0, 0
        for text, color, callback in buttons:
            btn = QPushButton(text)
            btn.setObjectName("plugin_toggle")
            if callback:
                btn.clicked.connect(self._wrap_callback(callback))
            else:
                btn.setEnabled(False)
                btn.setToolTip("Module not found or disabled.")
                
            grid.addWidget(btn, row, col)
            
            col += 1
            if col > 1: # 2 columns
                col = 0
                row += 1
                
        layout.addLayout(grid)
        layout.addStretch()
        
    def _wrap_callback(self, callback):
        """Wraps the callback to close the dialog before executing."""
        def wrapped():
            self.accept()
            callback()
        return wrapped
