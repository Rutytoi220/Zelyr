"""Hardware I/O Matrix Dashboard.

PySide6 dialogue displaying live battery/power state telemetry
and a scrolling ledger of intercepted USB/Bluetooth block devices.
"""
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGroupBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)
from PySide6.QtCore import Qt, QTimer
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class HardwareMatrixDialog(QDialog):
    """Visualizes Hardware Power States and Intercepted Peripherals."""

    def __init__(self, parent=None, event_bus=None):
        super().__init__(parent)
        self.setWindowTitle("Hardware I/O & Power Matrix")
        self.setMinimumSize(700, 500)
        self.event_bus = event_bus
        self._init_ui()
        
        if self.event_bus:
            self.event_bus.subscribe("hardware.usb.intercept", self._on_usb_intercept)
            self.event_bus.subscribe("hardware.usb.released", self._on_usb_released)
            self.event_bus.subscribe("power.state.critical", self._on_power_state)
            self.event_bus.subscribe("power.state.normal", self._on_power_state)

    def _init_ui(self):
        layout = QVBoxLayout(self)

        header = QLabel("<h2>Hardware Telemetry & Peripheral Security</h2>")
        header.setObjectName("update_header")
        layout.addWidget(header)

        power_group = QGroupBox("Cognitive Power Governor")
        power_layout = QVBoxLayout()
        
        self.power_label = QLabel("Status: MAX PERFORMANCE (AC Power)")
        self.power_label.setObjectName("hw_power_label")
        self.power_label.setProperty("status", "success")
        self.power_label.style().unpolish(self.power_label)
        self.power_label.style().polish(self.power_label)
        power_layout.addWidget(self.power_label)
        
        self.model_label = QLabel("Active Base Model: llama3:8b (8.0B Parameters)")
        self.model_label.setObjectName("hw_model_label")
        power_layout.addWidget(self.model_label)
        
        btn_layout = QHBoxLayout()
        self.btn_sim_eco = QPushButton("Simulate Unplug (<30%)")
        self.btn_sim_eco.clicked.connect(self._simulate_eco_mode)
        self.btn_sim_ac = QPushButton("Simulate Plug In")
        self.btn_sim_ac.clicked.connect(self._simulate_ac_mode)
        btn_layout.addWidget(self.btn_sim_eco)
        btn_layout.addWidget(self.btn_sim_ac)
        
        power_layout.addLayout(btn_layout)
        power_group.setLayout(power_layout)
        layout.addWidget(power_group)

        usb_group = QGroupBox("USB/BLE Zero-Trust Interceptor")
        usb_layout = QVBoxLayout()
        
        self.usb_table = QTableWidget(0, 4)
        self.usb_table.setHorizontalHeaderLabels(["Timestamp", "Node", "Type", "Sandboxed Status"])
        self.usb_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        usb_layout.addWidget(self.usb_table)
        
        self.btn_sim_usb = QPushButton("Simulate USB Insertion (/dev/sdc1)")
        self.btn_sim_usb.clicked.connect(self._simulate_usb)
        usb_layout.addWidget(self.btn_sim_usb)
        
        usb_group.setLayout(usb_layout)
        layout.addWidget(usb_group)

    def _on_usb_intercept(self, event):
        data = event.data
        self._add_usb_row(data.get("node"), "USB Mass Storage", "SANDBOXED (Scanning)", "danger")
        
    def _on_usb_released(self, event):
        data = event.data
        for row in range(self.usb_table.rowCount()):
            if self.usb_table.item(row, 1).text() == data.get("node"):
                item = self.usb_table.item(row, 3)
                item.setText("RELEASED (Safe)")
                item.setProperty("status", "success")
                item.style().unpolish(item)
                item.style().polish(item)
                break

    def _on_power_state(self, event):
        data = event.data
        if event.name == "power.state.critical":
            self.power_label.setText(f"Status: ECO MODE ({data.get('percent')}%)")
            self.power_label.setProperty("status", "danger")
            self.power_label.style().unpolish(self.power_label)
            self.power_label.style().polish(self.power_label)
            self.model_label.setText(f"Active Base Model: {data.get('target_model')} (Sub-2B Parameters)")
        else:
            self.power_label.setText(f"Status: MAX PERFORMANCE ({data.get('percent')}%)")
            self.power_label.setProperty("status", "success")
            self.power_label.style().unpolish(self.power_label)
            self.power_label.style().polish(self.power_label)
            self.model_label.setText(f"Active Base Model: {data.get('target_model')} (8.0B Parameters)")

    def _add_usb_row(self, node, type_str, status, status_property):
        row = self.usb_table.rowCount()
        self.usb_table.insertRow(row)
        
        items = [
            QTableWidgetItem(datetime.now().strftime("%H:%M:%S")),
            QTableWidgetItem(str(node)),
            QTableWidgetItem(type_str),
            QTableWidgetItem(status)
        ]
        
        for col, item in enumerate(items):
            item.setForeground(Qt.GlobalColor.white)
            if col == 3:
                item.setProperty("status", status_property)
                item.style().unpolish(item)
                item.style().polish(item)
            self.usb_table.setItem(row, col, item)
            
        self.usb_table.scrollToBottom()

    def _simulate_eco_mode(self):
        class MockEvent:
            name = "power.state.critical"
            data = {"percent": 15.0, "target_model": "llama3.2:1b"}
        self._on_power_state(MockEvent())
        
    def _simulate_ac_mode(self):
        class MockEvent:
            name = "power.state.normal"
            data = {"percent": 100.0, "target_model": "llama3:8b"}
        self._on_power_state(MockEvent())

    def _simulate_usb(self):
        import random
        node = f"/dev/sd{chr(random.randint(99, 102))}1"
        self._add_usb_row(node, "USB Flash Drive", "SANDBOXED (Scanning)", "danger")
        QTimer.singleShot(2000, lambda: self._on_usb_released(type('Mock', (object,), {'data': {'node': node}})))
