from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QRadioButton, QButtonGroup, QGroupBox, QListWidget, QPushButton
from PySide6.QtCore import Qt
from zelyr.engine.container_sandbox import ContainerSandboxManager

class SandboxManagerDialog(QDialog):
    """UI for managing container isolation for sub-agents."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📦 Sandbox Orchestrator")
        self.setMinimumSize(450, 300)
        
        self.container_mgr = ContainerSandboxManager()
        self._init_ui()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        mode_group = QGroupBox("Execution Mode")
        mode_layout = QHBoxLayout()
        
        self.btn_group = QButtonGroup(self)
        
        self.radio_bwrap = QRadioButton("Strict (Bubblewrap)")
        self.radio_podman = QRadioButton("Container (Podman)")
        self.radio_host = QRadioButton("Host (Unsafe)")
        
        self.btn_group.addButton(self.radio_bwrap)
        self.btn_group.addButton(self.radio_podman)
        self.btn_group.addButton(self.radio_host)
        
        mode_layout.addWidget(self.radio_bwrap)
        mode_layout.addWidget(self.radio_podman)
        mode_layout.addWidget(self.radio_host)
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # Set current mode
        current = self.container_mgr.get_mode()
        if current == "bwrap":
            self.radio_bwrap.setChecked(True)
        elif current == "podman":
            self.radio_podman.setChecked(True)
        else:
            self.radio_host.setChecked(True)
            
        self.radio_bwrap.toggled.connect(lambda c: self._on_mode_change("bwrap") if c else None)
        self.radio_podman.toggled.connect(lambda c: self._on_mode_change("podman") if c else None)
        self.radio_host.toggled.connect(lambda c: self._on_mode_change("host") if c else None)
        
        process_group = QGroupBox("Active Sandboxed Processes")
        process_layout = QVBoxLayout()
        self.process_list = QListWidget()
        self.process_list.addItem("No active sandboxed tasks.")
        process_layout.addWidget(self.process_list)
        process_group.setLayout(process_layout)
        
        layout.addWidget(process_group)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
    def _on_mode_change(self, mode: str):
        self.container_mgr.set_mode(mode)
        # Notify parent if needed, handled via singleton/manager reference usually
