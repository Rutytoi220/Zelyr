from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton, QGroupBox, QListWidget
from PySide6.QtCore import Qt, QTimer
import psutil

class GovernorDialog(QDialog):
    """UI for managing the hardware VRAM governor and background process priorities."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hardware Governor")
        self.setMinimumSize(500, 400)
        self._init_ui()
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_metrics)
        self.timer.start(1000)
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        metrics_group = QGroupBox("System Resources")
        metrics_layout = QVBoxLayout()
        
        self.cpu_label = QLabel("CPU Usage:")
        self.cpu_bar = QProgressBar()
        metrics_layout.addWidget(self.cpu_label)
        metrics_layout.addWidget(self.cpu_bar)
        
        self.vram_label = QLabel("VRAM Usage (Proxy):")
        self.vram_bar = QProgressBar()
        metrics_layout.addWidget(self.vram_label)
        metrics_layout.addWidget(self.vram_bar)
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        proc_group = QGroupBox("Throttled Background Workers")
        proc_layout = QVBoxLayout()
        self.proc_list = QListWidget()
        proc_layout.addWidget(self.proc_list)
        proc_group.setLayout(proc_layout)
        layout.addWidget(proc_group)
        
        mesh_group = QGroupBox("Active P2P Mesh Nodes (Encrypted)")
        mesh_layout = QVBoxLayout()
        self.mesh_list = QListWidget()
        self.mesh_list.addItem("Node: Arch-Workstation (Authenticated)")
        mesh_layout.addWidget(self.mesh_list)
        mesh_group.setLayout(mesh_layout)
        layout.addWidget(mesh_group)
        
    def _update_metrics(self):
        cpu = psutil.cpu_percent()
        self.cpu_bar.setValue(int(cpu))
        self.cpu_bar.setFormat(f"{cpu}%")
        
        vram = 50.0
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0)
                total = torch.cuda.get_device_properties(0).total_memory
                vram = (allocated / total) * 100.0
        except ImportError:
            pass
            
        self.vram_bar.setValue(int(vram))
        self.vram_bar.setFormat(f"{vram:.1f}%")
        
        self.proc_list.clear()
        for proc in psutil.process_iter(['pid', 'name', 'nice']):
            if proc.info['nice'] and proc.info['nice'] > 0:
                if 'ollama' in proc.info['name'].lower() or 'python' in proc.info['name'].lower():
                    self.proc_list.addItem(f"PID: {proc.info['pid']} | {proc.info['name']} (Nice: {proc.info['nice']})")

import json
from PySide6.QtWidgets import QTextEdit

class ExecutionGateDialog(QDialog):
    def __init__(self, tool_name: str, arguments: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Zelyr Governor - Execution Gate")
        self.setModal(True)
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        
        warning = QLabel("Zelyr is attempting to run a high-risk tool:")
        warning.setObjectName("warning")
        warning.setProperty("status", "danger")
        warning.style().unpolish(warning)
        warning.style().polish(warning)
        layout.addWidget(warning)
        
        tname = QLabel(f"Tool: {tool_name}")
        tname
        layout.addWidget(tname)
        
        layout.addWidget(QLabel("Arguments:"))
        
        args_text = QTextEdit()
        args_text.setReadOnly(True)
        args_text.setObjectName("governor_args")
        args_text.setText(json.dumps(arguments, indent=2))
        layout.addWidget(args_text)
        
        btn_layout = QHBoxLayout()
        
        self.deny_btn = QPushButton("Deny (Esc)")
        self.deny_btn.setObjectName("governor_deny")
        self.deny_btn.clicked.connect(self.reject)
        
        self.approve_btn = QPushButton("Approve (Enter)")
        self.approve_btn.setObjectName("governor_approve")
        self.approve_btn.clicked.connect(self.accept)
        
        btn_layout.addWidget(self.deny_btn)
        btn_layout.addWidget(self.approve_btn)
        layout.addLayout(btn_layout)
