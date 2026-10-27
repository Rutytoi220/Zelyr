"""eBPF Firewall & Swarm Control Dashboard.

PySide6 dialogue displaying live intercepted network connections from the eBPF module,
and active ephemeral Docker Swarm nodes.
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

logger = logging.getLogger(__name__)

class FirewallControlDialog(QDialog):
    """Dashboard for eBPF Firewall and Swarm Replication."""

    def __init__(self, parent=None, event_bus=None):
        super().__init__(parent)
        self.setWindowTitle("🛡️ eBPF Firewall & Swarm Hub")
        self.setMinimumSize(700, 500)
        self.event_bus = event_bus
        self._init_ui()
        
        # Subscribe to firewall events if event bus is available
        if self.event_bus:
            self.event_bus.subscribe("network.intercept.info", self._on_intercept_info)
            self.event_bus.subscribe("network.intercept.critical", self._on_intercept_critical)
            
        # For mock UI demonstration
        self._mock_timer = QTimer(self)
        self._mock_timer.timeout.connect(self._add_mock_connection)
        self._mock_timer.start(3000)

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("<h2>Kernel Firewall & Distributed Swarm</h2>")
        header
        layout.addWidget(header)

        # Firewall Group
        firewall_group = QGroupBox("eBPF Intercept Matrix")
        firewall_group
        fw_layout = QVBoxLayout()
        
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["PID", "Process", "Source IP", "Dest IP", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table
        fw_layout.addWidget(self.table)
        
        firewall_group.setLayout(fw_layout)
        layout.addWidget(firewall_group)

        # Swarm Group
        swarm_group = QGroupBox("Active Swarm Nodes (Docker)")
        swarm_group
        swarm_layout = QVBoxLayout()
        
        self.swarm_label = QLabel("No active ephemeral nodes.")
        self.swarm_label
        swarm_layout.addWidget(self.swarm_label)
        
        btn_layout = QHBoxLayout()
        self.btn_spawn = QPushButton("Simulate Swarm Job")
        self.btn_spawn.clicked.connect(self._simulate_swarm)
        btn_layout.addWidget(self.btn_spawn)
        
        swarm_layout.addLayout(btn_layout)
        swarm_group.setLayout(swarm_layout)
        layout.addWidget(swarm_group)

    def _on_intercept_info(self, event):
        """Handle standard intercept."""
        data = event.data
        self._add_table_row(data.get("pid"), data.get("comm"), data.get("saddr"), data.get("daddr", "") + f":{data.get('dport')}", "MONITORED", "#a6adc8")
        
    def _on_intercept_critical(self, event):
        """Handle critical intercept."""
        data = event.data
        self._add_table_row(data.get("pid"), data.get("comm"), data.get("saddr"), data.get("daddr", "") + f":{data.get('dport')}", "BLOCKED", "#f38ba8")

    def _add_table_row(self, pid, comm, saddr, daddr, status, color):
        row = self.table.rowCount()
        self.table.insertRow(row)
        
        items = [
            QTableWidgetItem(str(pid)),
            QTableWidgetItem(str(comm)),
            QTableWidgetItem(str(saddr)),
            QTableWidgetItem(str(daddr)),
            QTableWidgetItem(status)
        ]
        
        for col, item in enumerate(items):
            item.setForeground(Qt.GlobalColor.white)
            if col == 4:
                item
            self.table.setItem(row, col, item)
            
        # Keep table small
        if self.table.rowCount() > 100:
            self.table.removeRow(0)
            
        self.table.scrollToBottom()

    def _simulate_swarm(self):
        """Mock triggering a swarm job for UI testing."""
        self.swarm_label.setText("3 Active Nodes: [zelyr-worker-a1b2c3] [zelyr-worker-d4e5f6] [zelyr-worker-g7h8i9]")
        self.swarm_label
        QTimer.singleShot(4000, lambda: self.swarm_label.setText("No active ephemeral nodes."))
        QTimer.singleShot(4000, lambda: self.swarm_label)

    def _add_mock_connection(self):
        """Add mock data for visual testing."""
        import random
        pid = random.randint(1000, 9999)
        daddr = f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}:443"
        if random.random() > 0.8:
            self._add_table_row(pid, "curl", "10.0.0.5", "198.51.100.42:80", "BLOCKED", "#f38ba8")
        else:
            self._add_table_row(pid, "python3", "10.0.0.5", daddr, "MONITORED", "#a6adc8")
