import asyncio
import json
import socket
import logging
from pathlib import Path
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QStackedWidget, QWidget, QLineEdit, QMessageBox, QApplication
)

from zelyr.config import get_config, set_config, ZelyrConfig
from zelyr.gui.styles.theme_manager import THEMES_DIR, get_theme_manager
from zelyr.network.sync_server import SyncServer, SyncClient
from zelyr.network.crypto import generate_sync_pin

logger = logging.getLogger(__name__)

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def gather_sync_payload() -> dict:
    config = get_config()
    themes = {}
    if THEMES_DIR.exists():
        for child in THEMES_DIR.iterdir():
            if child.is_file() and child.suffix == '.json':
                themes[child.name] = child.read_text(encoding='utf-8')
    return {
        "config": config.to_dict(),
        "themes": themes
    }

class HostThread(QThread):
    finished = Signal()
    error = Signal(str)

    def __init__(self, pin: str):
        super().__init__()
        self.pin = pin
        self.server = SyncServer(port=9411)
        self.loop = asyncio.new_event_loop()

    def run(self):
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self.server.start(self.pin, gather_sync_payload))
            self.loop.run_forever()
        except Exception as e:
            self.error.emit(str(e))
        
    def stop(self):
        if self.server:
            asyncio.run_coroutine_threadsafe(self.server.stop(), self.loop)
        self.loop.call_soon_threadsafe(self.loop.stop)

class ClientThread(QThread):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, target_ip: str, pin: str):
        super().__init__()
        self.target_ip = target_ip
        self.pin = pin
        self.loop = asyncio.new_event_loop()

    def run(self):
        asyncio.set_event_loop(self.loop)
        client = SyncClient(self.target_ip, port=9411)
        try:
            payload = self.loop.run_until_complete(client.sync(self.pin))
            self.finished.emit(payload)
        except Exception as e:
            self.error.emit(str(e))


class SyncDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Zero-Auth P2P Device Sync")
        self.setMinimumSize(500, 300)
        
        self.host_thread = None
        
        self.layout = QVBoxLayout(self)
        
        self.stack = QStackedWidget()
        self.layout.addWidget(self.stack)
        
        # 1. Selection View
        self.selection_view = QWidget()
        sel_layout = QVBoxLayout(self.selection_view)
        sel_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sel_layout.setSpacing(20)
        
        title = QLabel("Device Sync")
        title.setObjectName("sync_pin")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sel_layout.addWidget(title)
        
        host_btn = QPushButton("Host a Sync")
        host_btn.setObjectName("sync_host")
        host_btn.clicked.connect(self.show_host_view)
        sel_layout.addWidget(host_btn)
        
        connect_btn = QPushButton("Connect to Device")
        connect_btn.setObjectName("sync_connect")
        connect_btn.clicked.connect(self.show_client_view)
        sel_layout.addWidget(connect_btn)
        
        self.stack.addWidget(self.selection_view)
        
        # 2. Host View
        self.host_view = QWidget()
        host_layout = QVBoxLayout(self.host_view)
        host_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.ip_label = QLabel(f"Your IP: {get_local_ip()}")
        self.ip_label.setProperty("status", "muted")
        self.ip_label.style().unpolish(self.ip_label)
        self.ip_label.style().polish(self.ip_label)
        self.ip_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        host_layout.addWidget(self.ip_label)
        
        self.pin_label = QLabel("000000")
        self.pin_label.setObjectName("sync_pin")
        self.pin_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        host_layout.addWidget(self.pin_label)
        
        status_label = QLabel("Waiting for connection...")
        status_label.setObjectName("sync_status")
        status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        host_layout.addWidget(status_label)
        
        self.stack.addWidget(self.host_view)
        
        # 3. Client View
        self.client_view = QWidget()
        client_layout = QVBoxLayout(self.client_view)
        client_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.ip_input = QLineEdit()
        self.ip_input.setPlaceholderText("Target IP Address (e.g. 192.168.1.5)")
        self.ip_input.setObjectName("sync_input")
        client_layout.addWidget(self.ip_input)
        
        self.pin_input = QLineEdit()
        self.pin_input.setPlaceholderText("6-Digit PIN")
        self.pin_input.setObjectName("sync_input")
        client_layout.addWidget(self.pin_input)
        
        sync_btn = QPushButton("Sync Now")
        sync_btn.setObjectName("sync_action")
        sync_btn.clicked.connect(self.perform_sync)
        client_layout.addWidget(sync_btn)
        
        self.stack.addWidget(self.client_view)

    def show_host_view(self):
        pin = generate_sync_pin()
        self.pin_label.setText(pin)
        self.host_thread = HostThread(pin)
        self.host_thread.start()
        self.stack.setCurrentWidget(self.host_view)

    def show_client_view(self):
        self.stack.setCurrentWidget(self.client_view)

    def perform_sync(self):
        ip = self.ip_input.text().strip()
        pin = self.pin_input.text().strip()
        if not ip or not pin:
            QMessageBox.warning(self, "Validation", "Please enter both IP and PIN.")
            return
            
        self.client_thread = ClientThread(ip, pin)
        self.client_thread.finished.connect(self.on_sync_success)
        self.client_thread.error.connect(self.on_sync_error)
        self.client_thread.start()

    def on_sync_success(self, payload: dict):
        try:
            # 1. Update Config
            config_data = payload.get("config", {})
            new_config = ZelyrConfig.from_dict(config_data)
            set_config(new_config)
            new_config.save()
            
            # 2. Update Themes
            themes = payload.get("themes", {})
            if not THEMES_DIR.exists():
                THEMES_DIR.mkdir(parents=True, exist_ok=True)
                
            for theme_name, theme_content in themes.items():
                theme_path = THEMES_DIR / theme_name
                theme_path.write_text(theme_content, encoding='utf-8')
                
            # 3. Reload Theme Manager
            mgr = get_theme_manager()
            mgr._load_themes()
            
            # Apply theme if one is specified in the config
            active_theme = new_config.theme if hasattr(new_config, 'theme') else "zelyr_pro"
            mgr.apply_theme(QApplication.instance(), active_theme)
            
            QMessageBox.information(self, "Sync Complete", "Device sync was successful. Settings and themes have been applied.")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Sync Error", f"Failed to apply sync payload: {e}")

    def on_sync_error(self, err_msg: str):
        QMessageBox.critical(self, "Sync Failed", f"Could not sync with device:\n{err_msg}")

    def closeEvent(self, event):
        if self.host_thread:
            self.host_thread.stop()
            self.host_thread.wait()
        super().closeEvent(event)
