from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, QListWidget, QListWidgetItem
from PySide6.QtCore import Qt
import subprocess
import os

class ZelyrFSDialog(QDialog):
    """UI for managing and exploring the ZelyrFS FUSE mount."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📁 ZelyrFS Explorer")
        self.setMinimumSize(600, 400)
        new_mount = os.path.expanduser("~/ZelyrFS")
        legacy_mount = os.path.expanduser("~/AxiomFS")
        self.mount_path = legacy_mount if os.path.ismount(legacy_mount) and not os.path.ismount(new_mount) else new_mount
        self._init_ui()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Header controls
        header = QHBoxLayout()
        self.status_label = QLabel(f"Status: Checking {self.mount_path}...")
        header.addWidget(self.status_label)
        
        self.mount_btn = QPushButton("Mount ZelyrFS")
        self.mount_btn.clicked.connect(self._mount_fs)
        header.addWidget(self.mount_btn)
        
        self.unmount_btn = QPushButton("Unmount")
        self.unmount_btn.clicked.connect(self._unmount_fs)
        header.addWidget(self.unmount_btn)
        
        layout.addLayout(header)
        
        # Explorer view
        explorer = QHBoxLayout()
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self._on_item_clicked)
        explorer.addWidget(self.file_list, 1)
        
        self.file_content = QTextEdit()
        self.file_content.setReadOnly(True)
        explorer.addWidget(self.file_content, 2)
        
        layout.addLayout(explorer)
        
        self._check_status()
        
    def _check_status(self):
        if os.path.ismount(self.mount_path):
            self.status_label.setText(f"Status: Mounted at {self.mount_path}")
            self.mount_btn.setEnabled(False)
            self.unmount_btn.setEnabled(True)
            self._refresh_list()
        else:
            self.status_label.setText(f"Status: Not Mounted")
            self.mount_btn.setEnabled(True)
            self.unmount_btn.setEnabled(False)
            self.file_list.clear()
            self.file_content.clear()
            
    def _mount_fs(self):
        # We start the FUSE mount in a background process
        try:
            cmd = f"python3 -c 'from zelyr.fs.zelyr_fs import mount_zelyr_fs; mount_zelyr_fs(\"{self.mount_path}\")'"
            subprocess.Popen(cmd, shell=True)
            # Give it a second to mount
            from PySide6.QtCore import QTimer
            QTimer.singleShot(1000, self._check_status)
        except Exception as e:
            self.status_label.setText(f"Error mounting: {e}")
            
    def _unmount_fs(self):
        try:
            subprocess.run(["fusermount", "-u", self.mount_path], check=False)
            self._check_status()
        except Exception as e:
            self.status_label.setText(f"Error unmounting: {e}")
            
    def _refresh_list(self):
        self.file_list.clear()
        if not os.path.ismount(self.mount_path):
            return
            
        try:
            # We mock the deep traversal for the UI
            self.file_list.addItem("/by-concept/docker")
            self.file_list.addItem("/by-concept/docker/context.md")
            self.file_list.addItem("/by-service/nginx.service")
        except:
            pass
            
    def _on_item_clicked(self, item: QListWidgetItem):
        path = os.path.join(self.mount_path, item.text().lstrip("/"))
        if path.endswith(".md"):
            try:
                with open(path, "r") as f:
                    self.file_content.setPlainText(f.read())
            except Exception as e:
                self.file_content.setPlainText(f"Error reading file: {e}")
        else:
            self.file_content.setPlainText("Directory selected. Select a file to view contents.")

# Legacy AXIOM API compatibility aliases
AxiomFSDialog = ZelyrFSDialog
