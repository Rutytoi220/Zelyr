import math
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, 
    QFrame, QGridLayout
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont

class StatusPill(QLabel):
    def __init__(self, label: str):
        super().__init__(label)
        self.setAlignment(Qt.AlignCenter)
        self.set_status("offline")

    def set_status(self, status: str):
        """status in ['online', 'offline', 'warning']"""
        self.setProperty("status", status)
        self.style().unpolish(self)
        self.style().polish(self)


class TelemetryBar(QWidget):
    def __init__(self, title: str):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        
        # Header
        header_layout = QHBoxLayout()
        self.title_lbl = QLabel(title)
        self.title_lbl.setObjectName("telemetry_title")
        self.val_lbl = QLabel("0%")
        self.val_lbl.setObjectName("telemetry_val")
        
        header_layout.addWidget(self.title_lbl)
        header_layout.addStretch()
        header_layout.addWidget(self.val_lbl)
        layout.addLayout(header_layout)
        
        # Bar
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setObjectName("telemetry_bar")
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        
        layout.addWidget(self.bar)

    def set_value(self, percent: float, label_text: str = ""):
        self.bar.setValue(int(percent))
        if label_text:
            self.val_lbl.setText(label_text)
        else:
            self.val_lbl.setText(f"{percent:.1f}%")
        
        # Color transition logic based on %
        if percent > 90:
            status = "danger"
        elif percent > 75:
            status = "warning"
        else:
            status = "normal"
            
        self.bar.setProperty("status", status)
        self.bar.style().unpolish(self.bar)
        self.bar.style().polish(self.bar)

class HealthRadarWidget(QFrame):
    """Real-Time Telemetry HUD & Health Radar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("healthRadar")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # --- HEADER ---
        header = QLabel("📡 Health Radar")
        header
        layout.addWidget(header)
        
        # --- LATENCY / LIFECYCLE ---
        info_layout = QHBoxLayout()
        
        self.lifecycle_lbl = QLabel("State: UNKNOWN")
        self.lifecycle_lbl
        
        self.latency_lbl = QLabel("Latency: -- ms")
        self.latency_lbl
        
        self.tasks_lbl = QLabel("Tasks: 0")
        self.tasks_lbl
        
        info_layout.addWidget(self.lifecycle_lbl)
        info_layout.addStretch()
        info_layout.addWidget(self.tasks_lbl)
        info_layout.addStretch()
        info_layout.addWidget(self.latency_lbl)
        layout.addLayout(info_layout)
        
        # --- RESOURCE BARS ---
        self.cpu_bar = TelemetryBar("CPU")
        self.ram_bar = TelemetryBar("RAM")
        self.vram_bar = TelemetryBar("VRAM")
        
        layout.addWidget(self.cpu_bar)
        layout.addWidget(self.ram_bar)
        layout.addWidget(self.vram_bar)
        
        # --- WORKER PILLS ---
        workers_label = QLabel("Active Workers")
        workers_label
        layout.addWidget(workers_label)
        
        grid = QGridLayout()
        grid.setSpacing(6)
        
        self.pill_watchdog = StatusPill("Watchdog")
        self.pill_indexer = StatusPill("Indexer")
        self.pill_pruner = StatusPill("Pruner")
        self.pill_mcp = StatusPill("MCP (0)")
        
        grid.addWidget(self.pill_watchdog, 0, 0)
        grid.addWidget(self.pill_indexer, 0, 1)
        grid.addWidget(self.pill_pruner, 1, 0)
        grid.addWidget(self.pill_mcp, 1, 1)
        
        layout.addLayout(grid)

    @Slot(dict)
    def update_telemetry(self, data: dict):
        # Update bars
        cpu = data.get("cpu_percent", 0.0)
        self.cpu_bar.set_value(cpu, f"{cpu:.1f}%")
        
        ram_pct = data.get("ram_percent", 0.0)
        ram_mb = data.get("ram_mb", 0.0)
        self.ram_bar.set_value(ram_pct, f"{ram_mb:.0f} MB")
        
        vram_pct = data.get("vram_percent", 0.0)
        vram_mb = data.get("vram_mb", 0.0)
        gpu_name = data.get("gpu_name", "Integrated / CPU-Only")
        
        if "Integrated" in gpu_name:
            self.vram_bar.set_value(0, "Shared")
            self.vram_bar.title_lbl.setText("GPU (Integrated)")
        else:
            self.vram_bar.set_value(vram_pct, f"{vram_mb:.0f} MB")
            self.vram_bar.title_lbl.setText(f"VRAM ({gpu_name.split()[0]})")
            
        # Update text info
        latency = data.get("loop_latency_ms", 0.0)
        self.latency_lbl.setText(f"Latency: {latency:.0f} ms")
        self.tasks_lbl.setText(f"Tasks: {data.get('active_tasks', 0)}")
        
        # MCP Connection tracking
        mcp_count = data.get("mcp_servers_configured", 0)
        self.pill_mcp.setText(f"MCP ({mcp_count})")
        if mcp_count > 0:
            self.pill_mcp.set_status("online")
        else:
            self.pill_mcp.set_status("offline")
            
    @Slot(dict)
    def update_lifecycle(self, data: dict):
        new_state = data.get("new_state", "UNKNOWN")
        self.lifecycle_lbl.setText(f"State: {new_state}")
        
    @Slot(dict)
    def update_service_status(self, data: dict):
        service = data.get("service")
        state = data.get("state")
        
        status_map = {
            "READY": "online",
            "DEGRADED": "warning",
            "BOOTING": "offline",
            "OFFLINE": "offline"
        }
        status_color = status_map.get(state, "offline")
        
        if service == "dir_watchdog":
            self.pill_watchdog.set_status(status_color)
        elif service == "indexer":
            self.pill_indexer.set_status(status_color)
        elif service == "pruner":
            self.pill_pruner.set_status(status_color)
