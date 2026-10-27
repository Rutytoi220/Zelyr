from zelyr.compat import config_dir
import json
import logging
import urllib.request
from pathlib import Path
from typing import List, Dict, Any

from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QTabWidget, QWidget, QScrollArea, 
    QFrame, QLabel, QPushButton, QHBoxLayout
)

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = [
    {
        "id": "sys_info",
        "name": "System Info Tool",
        "author": "zelyr-core",
        "version": "1.0.0",
        "desc": "Returns RAM/CPU usage",
        "tags": ["[System]", "[Diagnostics]"],
        "download_url": None,
        "code": "from zelyr.tools.base import BaseTool\n\nclass SysInfoTool(BaseTool):\n    def __init__(self):\n        super().__init__('sys_info', 'SysInfoTool', 'Returns RAM/CPU usage')\n    async def execute(self, **kw): return __import__('zelyr.tools', fromlist=['ToolResult']).ToolResult(success=True, output='System Nominal')"
    }
]

DEFAULT_THEMES_MANIFEST = [
    {
        "id": "nordic",
        "name": "Nordic Frost",
        "author": "Community",
        "version": "1.0",
        "desc": "A beautiful icy blue theme for late-night coding.",
        "url": "mock://nordic"
    },
    {
        "id": "malicious",
        "name": "H4ck3r R3d",
        "author": "Evil Corp",
        "version": "1.0",
        "desc": "Danger! This theme tries to do bad things.",
        "url": "mock://malicious"
    }
]

MOCK_THEME_CONTENTS = {
    "mock://nordic": json.dumps({
        "id": "nordic",
        "name": "Nordic Frost",
        "author": "Community",
        "version": "1.0",
        "tokens": {
            "bg_base": "#2e3440",
            "fg_base": "#eceff4",
            "accent": "#88c0d0"
        }
    }),
    "mock://malicious": json.dumps({
        "id": "malicious",
        "name": "H4ck3r R3d",
        "author": "Evil Corp",
        "version": "1.0",
        "tokens": {
            "bg_base": "#000000",
            "fg_base": "#ff0000",
            "accent": "url('../../../../../../etc/shadow')"
        }
    })
}

MANIFEST_URL = "https://raw.githubusercontent.com/ChienGPT/zelyr-registry/main/manifest.json"

class FetchManifestThread(QThread):
    finished = Signal(list)
    error = Signal(str)

    def run(self):
        try:
            req = urllib.request.Request(MANIFEST_URL, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=3.0) as response:
                data = json.loads(response.read().decode('utf-8'))
                if isinstance(data, list):
                    self.finished.emit(data)
                else:
                    self.finished.emit(DEFAULT_MANIFEST)
        except Exception as e:
            logger.warning(f"Failed to fetch live manifest, falling back to default: {e}")
            self.finished.emit(DEFAULT_MANIFEST)

class InstallToolThread(QThread):
    finished = Signal(str, str) # tool_id, status (success/error)
    
    def __init__(self, item: dict):
        super().__init__()
        self.item = item
        
    def run(self):
        try:
            code = self.item.get("code")
            download_url = self.item.get("download_url")
            
            if not code and download_url:
                req = urllib.request.Request(download_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=5.0) as response:
                    code = response.read().decode('utf-8')
            
            if not code:
                self.finished.emit(self.item['id'], "Error: No code or download URL provided")
                return
                
            plugins_dir = config_dir() / "plugins"
            plugins_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = plugins_dir / f"{self.item['id']}.py"
            file_path.write_text(code, encoding="utf-8")
            self.finished.emit(self.item['id'], "Success")
        except Exception as e:
            logger.error(f"Failed to download/install tool {self.item.get('id')}: {e}")
            self.finished.emit(self.item.get('id', 'unknown'), f"Error: {str(e)}")

class FetchThemesManifestThread(QThread):
    finished = Signal(list)
    error = Signal(str)

    def run(self):
        # In a real scenario, this would make an HTTP request to a themes manifest URL
        self.finished.emit(DEFAULT_THEMES_MANIFEST)

class DownloadThemeThread(QThread):
    finished = Signal(str, str) # theme_id, status (success/error)
    
    def __init__(self, item: dict):
        super().__init__()
        self.item = item
        
    def run(self):
        try:
            theme_id = self.item["id"]
            url = self.item.get("url")
            
            from zelyr.gui.styles.theme_manager import THEMES_DIR
            if not THEMES_DIR.exists():
                THEMES_DIR.mkdir(parents=True, exist_ok=True)
                
            theme_path = THEMES_DIR / f"{theme_id}.json"
            
            if url.startswith("mock://"):
                content = MOCK_THEME_CONTENTS.get(url, "{}")
            else:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=5.0) as response:
                    content = response.read().decode('utf-8')
                    
            theme_path.write_text(content, encoding='utf-8')
            
            # Validate via ThemeRegistry
            from zelyr.gui.styles.theme_registry import ThemeRegistry, ThemeValidationError
            registry = ThemeRegistry(THEMES_DIR)
            try:
                registry.validate_theme(theme_path)
                self.finished.emit(theme_id, "Success")
            except ThemeValidationError as ve:
                theme_path.unlink(missing_ok=True)
                self.finished.emit(theme_id, f"Validation Error: {str(ve)}")
            except Exception as ve:
                theme_path.unlink(missing_ok=True)
                self.finished.emit(theme_id, f"Error: {str(ve)}")

        except Exception as e:
            logger.error(f"Failed to download/install theme {self.item.get('id')}: {e}")
            self.finished.emit(self.item.get('id', 'unknown'), f"Error: {str(e)}")

class ZelyrHubDialog(QDialog):
    def closeEvent(self, event):
        """Ensure threads are gracefully terminated to prevent C++ SIGABRT on teardown."""
        if hasattr(self, 'fetch_thread') and self.fetch_thread.isRunning():
            self.fetch_thread.quit()
            self.fetch_thread.wait(100)
            
        if hasattr(self, 'fetch_themes_thread') and self.fetch_themes_thread.isRunning():
            self.fetch_themes_thread.quit()
            self.fetch_themes_thread.wait(100)
            
        for thread_list_name in ['_install_threads', '_install_theme_threads']:
            if hasattr(self, thread_list_name):
                threads = getattr(self, thread_list_name)
                for thread in threads:
                    if thread.isRunning():
                        thread.quit()
                        thread.wait(100)
        super().closeEvent(event)

    tool_installed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Zelyr Hub")
        self.setMinimumSize(650, 500)
        
        self.layout = QVBoxLayout(self)
        
        self.tabs = QTabWidget()
        self.tools_tab = QWidget()
        self.themes_tab = QWidget()
        self.mcp_tab = QWidget()
        self.swarm_tab = QWidget()
        
        self.tabs.addTab(self.tools_tab, "Tools")
        self.tabs.addTab(self.themes_tab, "Themes")
        self.tabs.addTab(self.mcp_tab, "🔌 MCP Servers")
        self.tabs.addTab(self.swarm_tab, "💻 Swarm Sync")
        
        self.layout.addWidget(self.tabs)
        
        self.tools_layout = QVBoxLayout(self.tools_tab)
        
        self.loading_label = QLabel("Loading live tools manifest...")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tools_layout.addWidget(self.loading_label)

        self.themes_layout = QVBoxLayout(self.themes_tab)
        self.themes_loading_label = QLabel("Fetching community themes...")
        self.themes_loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.themes_layout.addWidget(self.themes_loading_label)
        
        self._fetch_manifest()
        self._fetch_themes_manifest()
        self._init_mcp_tab()
        self._init_swarm_tab()

    def _init_mcp_tab(self):
        self.mcp_layout = QVBoxLayout(self.mcp_tab)
        
        # Header actions
        header_layout = QHBoxLayout()
        header_lbl = QLabel("Manage dynamic Model Context Protocol (MCP) servers.")
        header_lbl.setObjectName("hub_desc")
        header_layout.addWidget(header_lbl)
        header_layout.addStretch()
        
        self.add_mcp_btn = QPushButton("+ Add Server")
        self.add_mcp_btn.clicked.connect(self._on_add_mcp_server)
        header_layout.addWidget(self.add_mcp_btn)
        self.mcp_layout.addLayout(header_layout)
        
        # List area
        self.mcp_scroll = QScrollArea()
        self.mcp_scroll.setWidgetResizable(True)
        self.mcp_scroll_content = QWidget()
        self.mcp_list_layout = QVBoxLayout(self.mcp_scroll_content)
        self.mcp_list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.mcp_list_layout.setSpacing(8)
        
        self.mcp_scroll.setWidget(self.mcp_scroll_content)
        self.mcp_layout.addWidget(self.mcp_scroll)
        
        # Connect bridge signals
        try:
            bridge = self.parent()._bridge
            bridge.mcp_servers_updated.connect(self._on_mcp_servers_updated)
            # Ask for initial status
            bridge.send_get_mcp_status()
        except AttributeError:
            pass

    def _on_add_mcp_server(self):
        from PySide6.QtWidgets import QDialog, QFormLayout, QLineEdit, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle("Add MCP Server")
        dlg.setMinimumWidth(400)
        
        layout = QFormLayout(dlg)
        name_input = QLineEdit()
        name_input.setPlaceholderText("e.g. github")
        cmd_input = QLineEdit()
        cmd_input.setPlaceholderText("e.g. npx")
        args_input = QLineEdit()
        args_input.setPlaceholderText("e.g. -y @modelcontextprotocol/server-github")
        
        layout.addRow("Server Name:", name_input)
        layout.addRow("Command:", cmd_input)
        layout.addRow("Arguments:", args_input)
        
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)
        
        if dlg.exec() == QDialog.DialogCode.Accepted:
            name = name_input.text().strip()
            cmd = cmd_input.text().strip()
            args_raw = args_input.text().strip()
            args = []
            
            import shlex
            if args_raw:
                try:
                    args = shlex.split(args_raw)
                except ValueError:
                    args = args_raw.split()
                    
            if name and cmd:
                try:
                    self.parent()._bridge.send_add_mcp_server(name, cmd, args)
                except AttributeError:
                    pass

    def _on_mcp_servers_updated(self, payload: dict):
        # Clear existing
        while self.mcp_list_layout.count():
            child = self.mcp_list_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        servers = payload.get("connected_servers", [])
        if not servers:
            empty_lbl = QLabel("No MCP servers configured.")
            empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_lbl.setObjectName("hub_desc")
            self.mcp_list_layout.addWidget(empty_lbl)
            return
            
        for srv in servers:
            card = QFrame()
            card.setObjectName("hub_card")
            card_layout = QHBoxLayout(card)
            
            # Left: Info
            info_layout = QVBoxLayout()
            name_lbl = QLabel(srv.get("name", "Unknown"))
            name_lbl.setObjectName("hub_name")
            
            cmd = srv.get("command", "")
            args = " ".join(srv.get("args", []))
            cmd_lbl = QLabel(f"{cmd} {args}")
            cmd_lbl.setObjectName("hub_desc")
            
            status = srv.get("status", "OFFLINE")
            color = "#00cc66" if status == "ONLINE" else "#ff4444"
            status_lbl = QLabel(f"● {status} ({srv.get('tools_count', 0)} tools)")
            status_lbl.setObjectName("hub_tags")
            
            info_layout.addWidget(name_lbl)
            info_layout.addWidget(cmd_lbl)
            info_layout.addWidget(status_lbl)
            
            card_layout.addLayout(info_layout)
            card_layout.addStretch()
            
            # Right: Actions
            delete_btn = QPushButton("Remove")
            delete_btn.setProperty("status", "danger")
            delete_btn.clicked.connect(lambda _, n=srv.get("name"): self._remove_mcp_server(n))
            card_layout.addWidget(delete_btn, alignment=Qt.AlignmentFlag.AlignVCenter)
            
            self.mcp_list_layout.addWidget(card)

    def _remove_mcp_server(self, name: str):
        from PySide6.QtWidgets import QMessageBox
        reply = QMessageBox.question(self, "Confirm Remove", f"Remove MCP server '{name}'?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.parent()._bridge.send_remove_mcp_server(name)
            except AttributeError:
                pass

    def _fetch_manifest(self):
        self.fetch_thread = FetchManifestThread()
        self.fetch_thread.finished.connect(self._on_manifest_fetched)
        self.fetch_thread.start()

    def _on_manifest_fetched(self, manifest: list):
        self.loading_label.hide()
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setObjectName("hub_scroll")
        
        scroll_content = QWidget()
        scroll_content.setObjectName("hub_scroll_content")
        self.scroll_layout = QVBoxLayout(scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_layout.setSpacing(12)
        
        for item in manifest:
            card = self._create_tool_card(item)
            self.scroll_layout.addWidget(card)
            
        scroll.setWidget(scroll_content)
        self.tools_layout.addWidget(scroll)

    def _create_tool_card(self, item: dict) -> QFrame:
        card = QFrame()
        card.setObjectName("hub_card")
        
        layout = QHBoxLayout(card)
        
        text_layout = QVBoxLayout()
        text_layout.setSpacing(4)
        
        header_layout = QHBoxLayout()
        
        name_label = QLabel(item.get("name", "Unknown Tool"))
        name_label.setObjectName("hub_name")
        header_layout.addWidget(name_label)
        
        if "author" in item:
            author_label = QLabel(f"by @{item['author']}")
            author_label.setObjectName("hub_author")
            header_layout.addWidget(author_label)
            
        if "version" in item:
            version_label = QLabel(f"v{item['version']}")
            version_label.setObjectName("hub_version")
            header_layout.addWidget(version_label)
            
        header_layout.addStretch()
        text_layout.addLayout(header_layout)
        
        if "tags" in item and item["tags"]:
            tags_str = " ".join(item["tags"])
            tags_label = QLabel(tags_str)
            tags_label.setObjectName("hub_tags")
            text_layout.addWidget(tags_label)
            
        desc_label = QLabel(item.get("desc", ""))
        desc_label.setObjectName("hub_desc")
        desc_label.setWordWrap(True)
        text_layout.addWidget(desc_label)
        
        layout.addLayout(text_layout)
        layout.addStretch()
        
        install_btn = QPushButton()
        self._update_btn_state(install_btn, item["id"])
        install_btn.clicked.connect(lambda _, btn=install_btn, i=item: self._install_tool(i, btn))
        layout.addWidget(install_btn, alignment=Qt.AlignmentFlag.AlignVCenter)
        
        return card

    def _update_btn_state(self, btn: QPushButton, tool_id: str):
        plugins_dir = config_dir() / "plugins"
        file_path = plugins_dir / f"{tool_id}.py"
        
        if file_path.exists():
            btn.setText("Installed")
            btn.setProperty("status", "success")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
            btn.setEnabled(False)
        else:
            btn.setText("1-Click Install")
            btn.setProperty("status", "info")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
            btn.setEnabled(True)

    def _install_tool(self, item: dict, btn: QPushButton):
        btn.setText("Installing...")
        btn.setEnabled(False)
        
        if not hasattr(self, '_install_threads'):
            self._install_threads = []
            
        thread = InstallToolThread(item)
        self._install_threads.append(thread)
        thread.finished.connect(lambda tid, status: self._on_install_finished(tid, status, btn, thread))
        thread.start()

    def _on_install_finished(self, tool_id: str, status: str, btn: QPushButton, thread: QThread):
        if status == "Success":
            self._update_btn_state(btn, tool_id)
            self.tool_installed.emit(tool_id)
        else:
            btn.setText("Failed")
            btn.setProperty("status", "danger")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
            btn.setEnabled(False)
            
        if thread in self._install_threads:
            self._install_threads.remove(thread)

    def _fetch_themes_manifest(self):
        self.fetch_themes_thread = FetchThemesManifestThread()
        self.fetch_themes_thread.finished.connect(self._on_themes_manifest_fetched)
        self.fetch_themes_thread.start()

    def _on_themes_manifest_fetched(self, manifest: list):
        self.themes_loading_label.hide()
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setObjectName("hub_scroll")
        
        scroll_content = QWidget()
        scroll_content.setObjectName("hub_scroll_content")
        self.themes_scroll_layout = QVBoxLayout(scroll_content)
        self.themes_scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.themes_scroll_layout.setSpacing(12)
        
        for item in manifest:
            card = self._create_theme_card(item)
            self.themes_scroll_layout.addWidget(card)
            
        scroll.setWidget(scroll_content)
        self.themes_layout.addWidget(scroll)

    def _create_theme_card(self, item: dict) -> QFrame:
        card = QFrame()
        card.setObjectName("hub_card")
        
        layout = QHBoxLayout(card)
        text_layout = QVBoxLayout()
        text_layout.setSpacing(4)
        
        header_layout = QHBoxLayout()
        name_label = QLabel(item.get("name", "Unknown Theme"))
        name_label.setObjectName("hub_name")
        header_layout.addWidget(name_label)
        
        if "author" in item:
            author_label = QLabel(f"by @{item['author']}")
            author_label.setObjectName("hub_author")
            header_layout.addWidget(author_label)
            
        if "version" in item:
            version_label = QLabel(f"v{item['version']}")
            version_label.setObjectName("hub_version")
            header_layout.addWidget(version_label)
            
        header_layout.addStretch()
        text_layout.addLayout(header_layout)
        
        desc_label = QLabel(item.get("desc", ""))
        desc_label.setObjectName("hub_desc")
        desc_label.setWordWrap(True)
        text_layout.addWidget(desc_label)
        
        layout.addLayout(text_layout)
        layout.addStretch()
        
        install_btn = QPushButton()
        self._update_theme_btn_state(install_btn, item["id"])
        install_btn.clicked.connect(lambda _, btn=install_btn, i=item: self._install_theme(i, btn))
        layout.addWidget(install_btn, alignment=Qt.AlignmentFlag.AlignVCenter)
        
        return card

    def _update_theme_btn_state(self, btn: QPushButton, theme_id: str):
        from zelyr.gui.styles.theme_manager import THEMES_DIR
        file_path = THEMES_DIR / f"{theme_id}.json"
        
        if file_path.exists():
            btn.setText("Applied")
            btn.setProperty("status", "success")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
            btn.setEnabled(False)
        else:
            btn.setText("Install")
            btn.setProperty("status", "info")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
            btn.setEnabled(True)

    def _install_theme(self, item: dict, btn: QPushButton):
        btn.setText("Installing...")
        btn.setEnabled(False)
        
        if not hasattr(self, '_install_theme_threads'):
            self._install_theme_threads = []
            
        thread = DownloadThemeThread(item)
        self._install_theme_threads.append(thread)
        thread.finished.connect(lambda tid, status: self._on_theme_install_finished(tid, status, btn, thread))
        thread.start()

    def _on_theme_install_finished(self, theme_id: str, status: str, btn: QPushButton, thread: QThread):
        if status == "Success":
            self._update_theme_btn_state(btn, theme_id)
            # Apply the theme immediately
            from zelyr.gui.styles.theme_manager import get_theme_manager
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            mgr = get_theme_manager()
            mgr._load_themes()
            mgr.apply_theme(app, theme_id)
        else:
            btn.setText("Failed")
            btn.setProperty("status", "danger")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
            btn.setEnabled(False)
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Theme Installation Failed", f"Validation failed for theme '{theme_id}':\n{status}")
            
        if thread in self._install_theme_threads:
            self._install_theme_threads.remove(thread)

    def _init_swarm_tab(self):
        from PySide6.QtWidgets import QInputDialog, QMessageBox
        import httpx
        import asyncio
        from zelyr.network.p2p_sync import get_receiver_protocol, set_receiver_pin, P2PSyncProtocol
        
        layout = QVBoxLayout(self.swarm_tab)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        info_label = QLabel(
            "Synchronize your Zelyr settings, themes, and personas securely across the LAN "
            "or Tailscale mesh. Connections are end-to-end encrypted using AES-GCM and ECDH."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        self.pin_label = QLabel("Not in pairing mode.")
        self.pin_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pin_label.setObjectName("sync_pin")
        layout.addWidget(self.pin_label)
        
        btn_layout = QHBoxLayout()
        
        gen_btn = QPushButton("Generate Pairing Code (Receive)")
        gen_btn.clicked.connect(self._generate_pin)
        btn_layout.addWidget(gen_btn)
        
        link_btn = QPushButton("Link to Device (Send)")
        link_btn.clicked.connect(self._link_device)
        btn_layout.addWidget(link_btn)
        
        layout.addLayout(btn_layout)

    def _generate_pin(self):
        from zelyr.network.p2p_sync import set_receiver_pin, P2PSyncProtocol
        pin = P2PSyncProtocol.generate_pin()
        set_receiver_pin(pin)
        self.pin_label.setText(f"PIN: {pin[:3]}-{pin[3:]}\nWaiting for connection...")
        
    def _link_device(self):
        from PySide6.QtWidgets import QInputDialog, QMessageBox
        from zelyr.network.p2p_sync import P2PSyncProtocol
        import httpx
        
        ip, ok = QInputDialog.getText(self, "Link Device", "Enter Target IP (e.g., 100.x.x.x):")
        if not ok or not ip:
            return
            
        pin, ok = QInputDialog.getText(self, "Enter PIN", "Enter 6-digit PIN from target device:")
        if not ok or not pin:
            return
            
        pin = pin.replace("-", "").strip()
        
        def on_result(success, msg):
            if success:
                QMessageBox.information(self, "Success", msg)
            else:
                QMessageBox.critical(self, "Sync Failed", msg)
                
        self._sync_thread = SyncDeviceThread(ip, pin)
        self._sync_thread.finished.connect(on_result)
        self._sync_thread.start()

class SyncDeviceThread(QThread):
    finished = Signal(bool, str)
    
    def __init__(self, ip, pin):
        super().__init__()
        self.ip = ip
        self.pin = pin
        
    def run(self):
        import httpx
        from zelyr.network.p2p_sync import P2PSyncProtocol
        try:
            protocol = P2PSyncProtocol()
            pub_pem = protocol.get_public_key_pem()
            r1 = httpx.post(f"http://{self.ip}:11435/sync/pair", json={"public_key": pub_pem}, timeout=10.0)
            if r1.status_code != 200:
                raise Exception(f"Pairing rejected: {r1.text}")
                
            target_pub = r1.json()["public_key"]
            protocol.derive_shared_key(target_pub, self.pin)
            
            payload = protocol.export_state()
            r2 = httpx.post(f"http://{self.ip}:11435/sync/commit", json=payload, timeout=10.0)
            if r2.status_code != 200:
                raise Exception(f"Commit rejected: {r2.text}")
                
            self.finished.emit(True, "Synchronized successfully.")
        except Exception as e:
            self.finished.emit(False, str(e))

# Legacy AXIOM API compatibility aliases
AxiomHubDialog = ZelyrHubDialog
