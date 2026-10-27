import sys
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, 
    QHeaderView, QPushButton, QHBoxLayout, QLabel,
    QAbstractItemView
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from zelyr.engine.audit_ledger import AuditLedger

class AuditDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Security Sandbox Audit Ledger")
        self.setMinimumSize(800, 500)
        
        self.ledger = AuditLedger()
        
        self._init_ui()
        self._load_data()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header_layout = QHBoxLayout()
        title = QLabel("🛡️ Security Sandbox Executions")
        title
        header_layout.addWidget(title)
        
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self._load_data)
        header_layout.addWidget(refresh_btn, alignment=Qt.AlignRight)
        layout.addLayout(header_layout)
        
        # Table
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([
            "Timestamp", "Agent", "Tool", "Risk Level", "Status", "Arguments"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table
        layout.addWidget(self.table)
        
    def _load_data(self):
        self.table.setRowCount(0)
        logs = self.ledger.get_recent_logs(200)
        
        for row, log in enumerate(logs):
            self.table.insertRow(row)
            
            # Timestamp
            self.table.setItem(row, 0, QTableWidgetItem(log.get("timestamp", "")))
            
            # Agent
            self.table.setItem(row, 1, QTableWidgetItem(log.get("agent_name", "")))
            
            # Tool
            self.table.setItem(row, 2, QTableWidgetItem(log.get("tool_name", "")))
            
            # Risk Level
            risk = log.get("risk_level", "")
            risk_item = QTableWidgetItem(risk)
            if risk == "HIGH":
                risk_item.setForeground(QColor("#f38ba8"))  # Red
            else:
                risk_item.setForeground(QColor("#a6e3a1"))  # Green
            self.table.setItem(row, 3, risk_item)
            
            # Status
            status = log.get("status", "")
            status_item = QTableWidgetItem(status)
            if status == "BLOCKED":
                status_item.setForeground(QColor("#f38ba8"))  # Red
            elif status == "ALLOWED":
                status_item.setForeground(QColor("#a6e3a1"))  # Green
            self.table.setItem(row, 4, status_item)
            
            # Arguments
            args_str = str(log.get("arguments", ""))
            # Truncate if too long
            if len(args_str) > 100:
                args_str = args_str[:100] + "..."
            self.table.setItem(row, 5, QTableWidgetItem(args_str))
