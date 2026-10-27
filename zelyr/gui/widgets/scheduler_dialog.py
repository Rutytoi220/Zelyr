"""Automation Dialog UI.

A minimalist interface to toggle core autonomous background
triggers built in v5.0+ (REM Sleep, Power Governor, Watchdog, Interceptor).
"""
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFrame, QWidget
)
import logging

logger = logging.getLogger(__name__)

class SchedulerDialog(QDialog):
    """Minimalist UI for managing autonomous triggers."""

    def __init__(self, scheduler_service=None, parent=None, event_bus=None):
        super().__init__(parent)
        self.scheduler_service = scheduler_service
        self.event_bus = event_bus
        self.setWindowTitle("⏱️ Zelyr Automation Triggers")
        self.setMinimumSize(500, 400)
        pass

        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("<h2>Autonomous Background Triggers</h2>")
        header.setObjectName("scheduler_task_label")
        layout.addWidget(header)
        
        desc = QLabel("Easily toggle Zelyr's core background subsystems without complex cron rules.")
        desc.setObjectName("scheduler_empty")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Trigger List
        self._add_toggle_row(layout, "🌌 Nightly REM Sleep", "Compacts Relational Index memory at 03:00 AM.", "rem_sleep", True)
        self._add_toggle_row(layout, "🔋 Power Governor", "Dynamically throttles AI models when on battery power.", "power_gov", True)
        self._add_toggle_row(layout, "🔌 Hardware Interceptor", "Zero-trust sandbox for incoming USB/BLE mounts.", "hw_intercept", True)
        self._add_toggle_row(layout, "📂 Directory Watchdog", "Pre-computes responses based on file system events.", "watchdog", False)
        
        layout.addStretch()

    def _add_toggle_row(self, parent_layout, title: str, description: str, trigger_id: str, default_state: bool):
        row = QFrame()
        row.setObjectName("scheduler_task_row")
        row_layout = QHBoxLayout(row)
        
        text_layout = QVBoxLayout()
        t_label = QLabel(f"<b>{title}</b>")
        t_label.setObjectName("scheduler_task_label")
        
        d_label = QLabel(description)
        d_label.setObjectName("scheduler_cron_label")
        
        text_layout.addWidget(t_label)
        text_layout.addWidget(d_label)
        row_layout.addLayout(text_layout)
        
        row_layout.addStretch()
        
        btn = QPushButton("ON" if default_state else "OFF")
        self._style_toggle_btn(btn, default_state)
        btn.setProperty("trigger_id", trigger_id)
        btn.setProperty("state", default_state)
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(lambda _, b=btn: self._on_toggle(b))
        
        row_layout.addWidget(btn)
        parent_layout.addWidget(row)

    def _style_toggle_btn(self, btn: QPushButton, state: bool):
        if state:
            btn.setObjectName("scheduler_toggle")
        else:
            btn.setObjectName("scheduler_delete")
        btn.style().unpolish(btn)
        btn.style().polish(btn)

    @Slot()
    def _on_toggle(self, btn: QPushButton):
        current_state = btn.property("state")
        new_state = not current_state
        trigger_id = btn.property("trigger_id")
        
        btn.setProperty("state", new_state)
        btn.setText("ON" if new_state else "OFF")
        self._style_toggle_btn(btn, new_state)
        
        logger.info(f"Automation Dialog: Toggled {trigger_id} to {new_state}")
        
        if self.event_bus:
            self.event_bus.publish_sync(f"system.toggle.{trigger_id}", {"state": new_state})
