from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget, QListWidgetItem, QMessageBox
import os

class VMManagerDialog(QDialog):
    """UI for managing QEMU/KVM Micro-VMs."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🖥️ KVM Sandbox Manager")
        self.setMinimumSize(500, 300)
        self.vm_mgr = None
        self._init_ui()
        self._load_vms()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Header controls
        header = QHBoxLayout()
        header.addWidget(QLabel("Active Micro-VMs:"))
        
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.clicked.connect(self._load_vms)
        header.addWidget(self.refresh_btn)
        
        self.destroy_btn = QPushButton("🗑️ Destroy Selected")
        self.destroy_btn
        self.destroy_btn.clicked.connect(self._destroy_selected)
        header.addWidget(self.destroy_btn)
        
        layout.addLayout(header)
        
        # VM List
        self.vm_list = QListWidget()
        layout.addWidget(self.vm_list)
        
    def _load_vms(self):
        self.vm_list.clear()
        if not self.vm_mgr:
            try:
                from zelyr.engine.vm_orchestrator import MicroVMManager
                self.vm_mgr = MicroVMManager()
            except ImportError:
                self.vm_list.addItem("Failed to load MicroVMManager.")
                return
                
        if not self.vm_mgr.active_vms:
            self.vm_list.addItem("No active VMs.")
            return
            
        for vm_id, info in self.vm_mgr.active_vms.items():
            item = QListWidgetItem(f"VM: {vm_id[:8]}... | Status: {info['status']}")
            item.setData(100, vm_id) # Store ID in custom role
            self.vm_list.addItem(item)
            
    def _destroy_selected(self):
        selected = self.vm_list.selectedItems()
        if not selected or not self.vm_mgr:
            return
            
        vm_id = selected[0].data(100)
        if vm_id:
            try:
                self.vm_mgr.destroy_vm(vm_id)
                self._load_vms()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to destroy VM: {e}")
