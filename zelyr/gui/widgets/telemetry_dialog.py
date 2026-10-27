"""Zelyr Desktop v6.1 — Telemetry Trace Dialog."""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, 
    QTableWidgetItem, QHeaderView, QPushButton, QAbstractItemView
)

class TelemetryDialog(QDialog):
    """Displays a live scrolling table of structured OpenTelemetry execution traces."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📊 Telemetry Trace Observer")
        self.setMinimumSize(800, 500)
        self

        layout = QVBoxLayout(self)
        
        # Header
        header_layout = QHBoxLayout()
        title = QLabel("Structured Trace Stream")
        header_layout.addWidget(title)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_data)
        header_layout.addStretch()
        header_layout.addWidget(refresh_btn)
        layout.addLayout(header_layout)
        
        # Table
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Timestamp", "Correlation ID", "Operation", "Duration (ms)"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        
        layout.addWidget(self.table)
        
        # Auto-refresh timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_data)
        self._timer.start(2000)
        
        # Load initial mock data
        self._refresh_data()
        
    def _refresh_data(self):
        """Fetch trace logs and update table. (Mocked for now)"""
        # In a real implementation, this would read from the SQLite events table or a structlog sink.
        import time
        import random
        
        # Generate some mock recent traces
        if self.table.rowCount() < 15:
            row = self.table.rowCount()
            self.table.insertRow(row)
            
            ts = time.strftime("%H:%M:%S")
            corr_id = f"txn-{random.randint(1000, 9999)}"
            ops = ["ExecuteTool", "SandboxEntry", "LLMInference", "ContextRetrieve"]
            op = random.choice(ops)
            dur = str(random.randint(10, 1500))
            
            self.table.setItem(row, 0, QTableWidgetItem(ts))
            self.table.setItem(row, 1, QTableWidgetItem(corr_id))
            self.table.setItem(row, 2, QTableWidgetItem(op))
            self.table.setItem(row, 3, QTableWidgetItem(dur))
            
            self.table.scrollToBottom()
