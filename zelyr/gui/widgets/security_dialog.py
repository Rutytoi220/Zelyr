from PySide6.QtWidgets import QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView, QLabel, QHBoxLayout, QPushButton, QMessageBox
from PySide6.QtCore import Qt, QThread, Signal
from zelyr.engine.cyber_auditor import SecurityAuditorAgent
import asyncio

class AuditWorker(QThread):
    finished = Signal(dict)
    
    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        agent = SecurityAuditorAgent(event_bus=None, tool_registry=None, llm_client=None)
        res = loop.run_until_complete(agent.run_audit())
        loop.close()
        self.finished.emit(res)

class SecurityDashboardDialog(QDialog):
    """UI for triggering and reviewing proactive cybersecurity audits."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cyber Security Dashboard")
        self.setMinimumSize(800, 500)
        self._init_ui()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        self.score_label = QLabel("Posture Score: --")
        self.score_label.setObjectName("security_score")
        self.score_label.setProperty("status", "success")
        self.score_label.style().unpolish(self.score_label)
        self.score_label.style().polish(self.score_label)
        self.score_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.score_label)
        
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Severity", "Rule", "Detail", "Action"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)
        
        toolbar = QHBoxLayout()
        self.btn_audit = QPushButton("Run Security Audit")
        self.btn_audit.clicked.connect(self._run_audit)
        toolbar.addWidget(self.btn_audit)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        
        self._run_audit()
        
    def _run_audit(self):
        self.btn_audit.setEnabled(False)
        self.btn_audit.setText("Running Audit...")
        self.table.setRowCount(0)
        self.score_label.setText("Posture Score: Calculating...")
        self.score_label.setProperty("status", "warning")
        self.score_label.style().unpolish(self.score_label)
        self.score_label.style().polish(self.score_label)
        
        self.worker = AuditWorker()
        self.worker.finished.connect(self._on_audit_complete)
        self.worker.start()
        
    def _on_audit_complete(self, result: dict):
        self.btn_audit.setEnabled(True)
        self.btn_audit.setText("Run Security Audit")
        
        score = result.get("posture_score", 0)
        findings = result.get("findings", [])
        
        if score >= 90:
            status = "success"
        elif score >= 60:
            status = "warning"
        else:
            status = "danger"
            
        self.score_label.setText(f"Posture Score: {score}% ({result.get('status', '')})")
        self.score_label.setProperty("status", status)
        self.score_label.style().unpolish(self.score_label)
        self.score_label.style().polish(self.score_label)
        
        for i, finding in enumerate(findings):
            self.table.insertRow(i)
            self.table.setItem(i, 0, QTableWidgetItem(finding["level"]))
            self.table.setItem(i, 1, QTableWidgetItem(finding["rule"]))
            self.table.setItem(i, 2, QTableWidgetItem(finding["detail"]))
            
            heal_btn = QPushButton("Heal Vulnerability")
            heal_btn.setObjectName("security_heal")
            heal_btn.clicked.connect(lambda checked, r=finding["rule"]: self._trigger_heal(r))
            self.table.setCellWidget(i, 3, heal_btn)

    def _trigger_heal(self, rule: str):
        QMessageBox.information(self, "HealerAgent Triggered", f"Dispatched task to HealerAgent to remediate: {rule}")
