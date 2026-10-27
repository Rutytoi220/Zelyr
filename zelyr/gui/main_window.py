"""Zelyr Desktop v6.0 LTS — Main Application Window.

Implements the modular layout:
  - Toolbar (model badge, auth mode selector, expert toggle)
  - Central chat viewport with streaming message bubbles
  - Collapsible right dock: System Telemetry & Expert Logs
  - Bottom control bar: multiline input + send button
  - Status bar: active model + mode badge
"""
from __future__ import annotations
import html
import logging
import os
from typing import TYPE_CHECKING
from PySide6.QtCore import Qt, QSize, Slot, QTimer, Signal
from PySide6.QtGui import QAction, QFont, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import QButtonGroup, QDockWidget, QFrame, QHBoxLayout, QLabel, QMainWindow, QPushButton, QScrollArea, QSizePolicy, QStatusBar, QTextEdit, QToolBar, QToolButton, QVBoxLayout, QWidget, QSystemTrayIcon, QMenu, QApplication, QSplitter, QFileDialog, QDialog
from zelyr.config import get_config, AuthMode
from zelyr.core.shortcuts import SHORTCUTS
from zelyr.gui.styles.theme_manager import get_theme_manager
from zelyr.gui.widgets.swarm_pill import SwarmPill
from zelyr.gui.widgets.settings_dialog import SettingsDialog
from zelyr.gui.widgets.hub_dialog import ZelyrHubDialog
from zelyr.gui.windows.project_dialog import ProjectDialog
if TYPE_CHECKING:
    from zelyr.gui.bridge import ZelyrBridge
logger = logging.getLogger(__name__)
_AUTH_LABELS = {AuthMode.BASIC: '🛡️ BASIC', AuthMode.AUTOPILOT: '⚡ AUTOPILOT', AuthMode.STRICT: '🔒 STRICT'}
_AUTH_COLORS = {AuthMode.BASIC: '#f59e0b', AuthMode.AUTOPILOT: '#10b981', AuthMode.STRICT: '#ef4444'}

class ChatInputEdit(QTextEdit):
    """Custom input edit that sends on Enter and adds a newline on Shift+Enter."""
    send_requested = Signal()

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                super().keyPressEvent(event)
            else:
                self.send_requested.emit()
                event.accept()
        else:
            super().keyPressEvent(event)

from PySide6.QtCore import QPropertyAnimation, QEasingCurve
from PySide6.QtWidgets import QComboBox, QLineEdit
import subprocess

class SettingsDrawer(QFrame):
    """Sleek dark-mode settings/swarm sidebar."""

    model_changed = Signal(str)
    connect_requested = Signal(str)   # emits the host string
    disconnect_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximumWidth(0)
        self.setMinimumWidth(0)
        self.setObjectName("surface")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # ── Title ──────────────────────────────────────────────────────── #
        title = QLabel("Settings")
        title
        layout.addWidget(title)

        # ── Model Settings ─────────────────────────────────────────────── #
        model_label = QLabel("\U0001f9e0 Model Settings")
        model_label.setProperty("status", "muted")
        layout.addWidget(model_label)

        self.model_combo = QComboBox()
        
        layout.addWidget(self.model_combo)
        self._populate_models()
        self.model_combo.currentTextChanged.connect(self.model_changed)
        
        # ── Theme Settings ─────────────────────────────────────────────── #
        theme_label = QLabel("🎨 Theme Settings")
        theme_label.setProperty("status", "muted")
        layout.addWidget(theme_label)
        
        self.theme_combo = QComboBox()
        from zelyr.gui.styles.theme_manager import get_theme_manager
        tm = get_theme_manager()
        self.theme_combo.addItems(["zelyr_pro", "jarvis"] + [t for t in tm._registry.list_theme_ids() if t not in ["zelyr_pro", "jarvis"]])
        
        current = tm.active_theme_name or "zelyr_pro"
        if current in [self.theme_combo.itemText(i) for i in range(self.theme_combo.count())]:
            self.theme_combo.setCurrentText(current)
            
        layout.addWidget(self.theme_combo)
        self.theme_combo.currentTextChanged.connect(
            lambda text: get_theme_manager().apply_theme(QApplication.instance(), text)
        )

        # ── Swarm Nodes ────────────────────────────────────────────────── #
        node_label = QLabel("\U0001f310 Swarm Nodes")
        node_label.setProperty("status", "muted")
        layout.addWidget(node_label)

        # Status indicator
        self._status_label = QLabel("● Disconnected")
        self._status_label.setProperty("status", "danger"); self._status_label
        layout.addWidget(self._status_label)

        node_layout = QHBoxLayout()
        node_layout.setContentsMargins(0, 0, 0, 0)
        node_layout.setSpacing(6)
        self.node_input = QLineEdit()
        self.node_input.setPlaceholderText("192.168.x.x:8000")
        

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setCursor(Qt.PointingHandCursor)
        self.connect_btn.setProperty("status", "")
        self.connect_btn.style().unpolish(self.connect_btn)
        self.connect_btn.style().polish(self.connect_btn)
        self.connect_btn.clicked.connect(self._on_connect_clicked)

        node_layout.addWidget(self.node_input, 1)
        node_layout.addWidget(self.connect_btn)
        layout.addLayout(node_layout)

        # Node list — shows connected node
        self._node_list_label = QLabel("")
        self._node_list_label.setProperty("status", "muted"); self._node_list_label
        self._node_list_label.setWordWrap(True)
        layout.addWidget(self._node_list_label)

        layout.addStretch()

        # ── Audio / TTS Controls ─────────────────────────────────────────── #
        audio_label = QLabel("🔉 Audio")
        audio_label.setProperty("status", "muted"); audio_label
        layout.addWidget(audio_label)

        self._tts_toggle_btn = QPushButton("🔊  Voice Responses: ON")
        self._tts_toggle_btn.setCheckable(True)
        self._tts_toggle_btn.setChecked(True)
        self._tts_toggle_btn.setCursor(Qt.PointingHandCursor)
        self._tts_toggle_btn
        self._tts_toggle_btn.toggled.connect(self._on_tts_toggled)
        layout.addWidget(self._tts_toggle_btn)

        layout.addSpacing(20)

        self._sync_btn = QPushButton("\U0001f504 Device Sync")
        self._sync_btn.setCursor(Qt.PointingHandCursor)
        self._sync_btn
        self._sync_btn.clicked.connect(self._on_sync_clicked)
        layout.addWidget(self._sync_btn)

        layout.addSpacing(10)

        self._update_btn = QPushButton("🔄 Check for Updates")
        self._update_btn.setCursor(Qt.PointingHandCursor)
        self._update_btn
        self._update_btn.clicked.connect(self._on_update_clicked)
        layout.addWidget(self._update_btn)

        # ── Animation ────────────────────────────────────────────────────── #
        self.anim = QPropertyAnimation(self, b"maximumWidth")
        self.anim.setDuration(300)
        self.anim.setEasingCurve(QEasingCurve.Type.InOutQuart)

        self._connected = False

    def _on_tts_toggled(self, checked: bool):
        label = "🔊  Voice Responses: ON" if checked else "🔇  Voice Responses: OFF"
        self._tts_toggle_btn.setText(label)
    
    @property
    def tts_enabled(self) -> bool:
        """Whether TTS voice responses are currently enabled."""
        return self._tts_toggle_btn.isChecked()

    def _on_sync_clicked(self):
        from zelyr.gui.widgets.sync_dialog import SyncDialog
        dlg = SyncDialog(self)
        dlg.exec()

    def _on_update_clicked(self):
        from zelyr.gui.widgets.update_dialog import UpdateDialog
        dlg = UpdateDialog(self)
        dlg.exec()

    # ── Private helpers ────────────────────────────────────────────────── #
    def _populate_models(self):
        """Fetch installed Ollama models and populate the dropdown."""
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItem("Auto-Select")
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().splitlines()
                for line in lines[1:]:  # skip header row
                    parts = line.split()
                    if parts:
                        self.model_combo.addItem(parts[0])
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass  # Ollama not installed / timed out — silently skip
        except Exception as e:
            logger.warning(f"SettingsDrawer: could not fetch ollama models: {e}")
        finally:
            self.model_combo.blockSignals(False)

    def _on_connect_clicked(self):
        if self._connected:
            self.disconnect_requested.emit()
        else:
            host = self.node_input.text().strip()
            if host:
                self.connect_btn.setText("Connecting…")
                self.connect_btn.setEnabled(False)
                self.connect_btn.setProperty("status", "warning")
                self.connect_btn.style().unpolish(self.connect_btn)
                self.connect_btn.style().polish(self.connect_btn)
                self.connect_requested.emit(host)

    # ── Public slots called by MainWindow ─────────────────────────────── #
    def on_swarm_connected(self):
        self._connected = True
        self._status_label.setText("● Connected")
        self._status_label.setProperty("status", "success"); self._status_label.style().unpolish(self._status_label); self._status_label.style().polish(self._status_label)
        host = self.node_input.text().strip()
        self._node_list_label.setText(f"Node: {host}")
        self.connect_btn.setText("Disconnect")
        self.connect_btn.setEnabled(True)
        self.connect_btn.setProperty("status", "active")
        self.connect_btn.style().unpolish(self.connect_btn)
        self.connect_btn.style().polish(self.connect_btn)

    def on_swarm_disconnected(self):
        self._connected = False
        self._status_label.setText("● Disconnected")
        self._status_label.setProperty("status", "danger"); self._status_label
        self._node_list_label.setText("")
        self.connect_btn.setText("Connect")
        self.connect_btn.setEnabled(True)
        self.connect_btn.setProperty("status", "")
        self.connect_btn.style().unpolish(self.connect_btn)
        self.connect_btn.style().polish(self.connect_btn)

    def on_swarm_error(self, msg: str):
        self._connected = False
        self._status_label.setText(f"● Error: {msg[:40]}")
        self._status_label.setProperty("status", "warning"); self._status_label.style().unpolish(self._status_label); self._status_label.style().polish(self._status_label)
        self.connect_btn.setText("Retry")
        self.connect_btn.setEnabled(True)
        self.connect_btn.setProperty("status", "")
        self.connect_btn.style().unpolish(self.connect_btn)
        self.connect_btn.style().polish(self.connect_btn)

    # ── Toggle animation ──────────────────────────────────────────────── #
    def toggle(self):
        if self.maximumWidth() == 0:
            self._populate_models()   # refresh models each open
            self.anim.setStartValue(0)
            self.anim.setEndValue(300)
            self.anim.start()
        else:
            self.anim.setStartValue(self.width())
            self.anim.setEndValue(0)
            self.anim.start()

class MainWindow(QMainWindow):
    """Zelyr Desktop v6.0 LTS main application window."""

    def __init__(self, bridge: 'ZelyrBridge', parent: QWidget | None=None) -> None:
        super().__init__(parent)
        self._bridge = bridge
        self._streaming_bubble = None
        self._streaming_text = ''
        self.setWindowTitle('Zelyr Pro — Sovereign AI')
        self.setMinimumSize(800, 600)
        self.resize(1000, 750)
        self._theme_manager = get_theme_manager()

        from zelyr.memory.projects import ProjectManager
        self._project_manager = ProjectManager()
        self._current_project_id = "general"
        self._current_chat_id = None
        # Ensure the default project directory exists on first run
        if not any(p["id"] == "general" for p in self._project_manager.get_projects()):
            self._project_manager.create_project(
                title="General",
                context_text="",
                project_id="general"
            )

        from zelyr.gui.widgets.modern_chat import ModernChatDisplay
        from zelyr.gui.widgets.modern_sidebar import ModernSidebar

        central_widget = QWidget()
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        
        self.sidebar = ModernSidebar(self)
        self.sidebar.new_chat_requested.connect(self._on_new_chat)
        self.sidebar.new_project_requested.connect(self._on_new_project_requested)
        self.sidebar.conversation_selected.connect(self._on_conversation_selected)
        self.sidebar.mode_changed.connect(self._on_mode_changed)
        self.sidebar.settings_btn.clicked.connect(self._show_settings)
        self.sidebar.hub_btn.clicked.connect(self._show_hub)
        self.splitter.addWidget(self.sidebar)

        self._chat_display = ModernChatDisplay(self)
        self._chat_display.input_bar.attach_btn.clicked.connect(self._on_attach_file)
        self.splitter.addWidget(self._chat_display)
        
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        self.splitter.setHandleWidth(4)
        self.splitter.setSizes([280, 800])

        
        main_layout.addWidget(self.splitter, 1)
        
        from zelyr.gui.widgets.health_radar import HealthRadarWidget
        self.health_radar = HealthRadarWidget(self)
        self.health_radar.setFixedWidth(280)
        self.health_radar.hide()
        main_layout.addWidget(self.health_radar)
        
        self.settings_drawer = SettingsDrawer(self)
        main_layout.addWidget(self.settings_drawer)
        self._chat_display.settings_btn.clicked.connect(self.settings_drawer.toggle)
        
        # Radar sidebar button has been removed per design spec (v11.2 purge).
        # The radar_btn stub is kept so any residual references don't raise AttributeError.
        self.radar_btn = QPushButton("📡 Radar")
        self.radar_btn.setCheckable(True)
        self.radar_btn.clicked.connect(self.health_radar.setVisible)
        self.radar_btn.setParent(self)
        self.radar_btn.hide()  # NOT added to sidebar layout

        # ── Swarm Client ──────────────────────────────────────────────── #
        from zelyr.gui.swarm_client import SwarmClient
        self._swarm = SwarmClient(self)
        self._swarm.connected.connect(self.settings_drawer.on_swarm_connected)
        self._swarm.connected.connect(lambda: self._chat_display.add_bubble('assistant', 'Swarm Node connected — prompts will be routed remotely.'))
        self._swarm.disconnected.connect(self.settings_drawer.on_swarm_disconnected)
        self._swarm.disconnected.connect(lambda: self._chat_display.add_bubble('assistant', '🔌 Swarm Node disconnected — falling back to local engine.'))
        self._swarm.connection_error.connect(self.settings_drawer.on_swarm_error)
        self._swarm.connection_error.connect(lambda msg: self._chat_display.add_bubble('tool', f'⚠️ Swarm connection error: {msg}'))
        self._swarm.response_complete.connect(self._on_swarm_response)
        self.settings_drawer.connect_requested.connect(self._swarm.connect_to_node)
        self.settings_drawer.disconnect_requested.connect(self._swarm.disconnect_from_node)

        # ── Model selection ────────────────────────────────────────────── #
        self.settings_drawer.model_changed.connect(self._on_model_changed)

        self._refresh_sidebar()

        self.setCentralWidget(central_widget)

        self._input = self._chat_display.input_bar.input_edit
        self._chat_display.input_bar.message_ready.connect(self._on_send)
        self._chat_display.input_bar.mic_toggled.connect(self._on_mic_toggled)
        self._connect_bridge()
        self._init_audio()
        self._init_tray()
        QTimer.singleShot(0, self._init_hotkey)
        self._register_local_shortcuts()
        # (geometry test bubbles removed — watermark shows cleanly on startup)
        
        # Init startup status
        self._startup_status = QLabel("⏳ Background services starting...")
        self._startup_status.setProperty("status", "warning"); self._startup_status
        self.statusBar().addWidget(self._startup_status)

        self._apply_theme()
        self._theme_manager.theme_changed.connect(self._apply_theme)

    def _apply_theme(self):
        t = self._theme_manager.theme
                
    def _show_settings(self):
        dialog = SettingsDialog(self)
        if dialog.exec():
            if self._bridge and hasattr(self._bridge, 'send_config_updated'):
                self._bridge.send_config_updated()

    def _show_hub(self):
        dialog = ZelyrHubDialog(self)
        dialog.tool_installed.connect(self._on_tool_installed)
        dialog.exec()

    def _on_tool_installed(self, tool_id: str):
        if self._bridge and hasattr(self._bridge, 'send_reload_plugins'):
            self._bridge.send_reload_plugins()
        self._chat_display.add_bubble('system', f'✅ Tool "{tool_id}" installed and dynamically hot-reloaded.')

    def _on_attach_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Attach File", "", "Text Files (*.txt *.py *.md *.json *.csv);;All Files (*)"
        )
        if file_name:
            try:
                with open(file_name, "r", encoding="utf-8") as f:
                    file_content = f.read()
                current_text = self._chat_display.input_bar.input_area.toPlainText()
                new_text = f"{current_text}\n\n--- FILE: {file_name} ---\n{file_content}\n"
                self._chat_display.input_bar.input_area.setPlainText(new_text)
            except Exception as e:
                logger.error("Failed to read file: %s", e)

    def _refresh_sidebar(self) -> None:
        projects = self._project_manager.get_projects()
        data = []
        for p in projects:
            chats = self._project_manager.get_conversations(p["id"])
            data.append({"project": p, "chats": chats})
        self.sidebar.populate_projects(data)

    def _on_new_project_requested(self) -> None:
        try:
            dialog = ProjectDialog(self)
            dialog.project_created.connect(self._create_project)
            dialog.exec()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Project creation stub — dialog failed: %s", exc)

    def _create_project(self, title: str, context: str, files: list) -> None:
        self._project_manager.create_project(title=title, context_text=context, attached_files=files)
        self._refresh_sidebar()

    def _on_conversation_selected(self, project_id: str, chat_id: str) -> None:
        self._current_project_id = project_id
        self._current_chat_id = chat_id
        chat_data = self._project_manager.load_conversation(project_id, chat_id)
        
        from zelyr.gui.widgets.modern_chat import ModernChatBubble
        for i in reversed(range(self._chat_display.chat_layout.count())):
            item = self._chat_display.chat_layout.itemAt(i)
            if item.widget() and isinstance(item.widget(), ModernChatBubble):
                item.widget().deleteLater()
                
        if chat_data:
            if hasattr(self._chat_display, 'watermark'):
                self._chat_display.watermark.hide()
            messages = chat_data.get("messages", [])
            for msg in messages:
                self._chat_display.add_bubble(msg.get("role", "user"), msg.get("content", ""))
        else:
            if hasattr(self._chat_display, 'watermark'):
                self._chat_display.watermark.show()
            # Watermark QLabel handles the empty-state display — no phantom bubble.

    def _on_new_chat(self) -> None:
        # Just clear the UI, don't create an empty chat file yet
        self._current_chat_id = None
        self._refresh_sidebar()
        
        from zelyr.gui.widgets.modern_chat import ModernChatBubble
        for i in reversed(range(self._chat_display.chat_layout.count())):
            item = self._chat_display.chat_layout.itemAt(i)
            if item.widget() and isinstance(item.widget(), ModernChatBubble):
                item.widget().deleteLater()
        
        if hasattr(self._chat_display, 'watermark'):
            self._chat_display.watermark.show()
        # Do NOT add_bubble here — the watermark QLabel IS the empty-state decoration.

    def _on_mode_changed(self, mode: str) -> None:
        if self._bridge and hasattr(self._bridge, 'set_auth_mode'):
            self._bridge.set_auth_mode(mode)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if hasattr(self, '_temporal_service'):
            self._temporal_service.start()
        else:
            try:
                from zelyr.services.scheduler import TemporalService
                self._temporal_service = TemporalService()
                self._temporal_service.start()
            except Exception:
                pass
        if hasattr(self, '_sys_watchdog'):
            self._sys_watchdog.start()
        else:
            try:
                from zelyr.services.sys_watchdog import SystemHealthWatchdog
                self._sys_watchdog = SystemHealthWatchdog()
                self._sys_watchdog.start()
            except Exception:
                pass

    def closeEvent(self, event) -> None:
        """Handle window close."""
        if event.spontaneous():
            event.ignore()
            self.hide()
            logger.info('Window hidden to system tray.')
        else:
            super().closeEvent(event)

    def force_quit(self) -> None:
        """Actually shut down the application."""
        if hasattr(self, '_swarm'):
            self._swarm.shutdown()
        if hasattr(self, '_temporal_service'):
            self._temporal_service.stop()
        if hasattr(self, '_sys_watchdog'):
            self._sys_watchdog.stop()
        if hasattr(self, '_wake_daemon') and self._wake_daemon:
            self._wake_daemon.stop()
        if hasattr(self, '_hotkey_service'):
            self._hotkey_service.stop()
        logger.info('Zelyr shutting down gracefully.')
        QApplication.quit()

    def _init_audio(self) -> None:
        """Initialize the AudioManager facade (TTS + STT) in a background thread.

        AudioManager may trigger WhisperModel loading which takes several
        seconds.  Running it on a daemon thread lets the UI paint and respond
        immediately while the model loads silently in the background.
        """
        import threading

        def _load():
            try:
                import socket
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(0.5)
                    if s.connect_ex(('127.0.0.1', 8000)) == 0:
                        logger.info("Daemon is running (port 8000 open). Skipping GUI local audio engine to prevent lock contention.")
                        self._audio = None
                        return
                        
                from zelyr.core.audio import AudioManager
                self._audio = AudioManager.instance()
                logger.info('AudioManager initialized (background thread)')
            except Exception as exc:
                self._audio = None
                logger.warning('AudioManager init failed (expected if daemon owns lock): %s', exc)

        threading.Thread(target=_load, daemon=True).start()

    def _init_tray(self) -> None:
        self.tray_icon = QSystemTrayIcon(self)
        icon = self.style().standardIcon(self.style().StandardPixmap.SP_ComputerIcon)
        self.tray_icon.setIcon(icon)
        tray_menu = QMenu()
        toggle_action = QAction('Show/Hide Zelyr', self)
        toggle_action.triggered.connect(lambda: self.hide() if self.isVisible() else self.show())
        tray_menu.addAction(toggle_action)
        settings_action = QAction('Settings', self)
        settings_action.triggered.connect(lambda: None)
        tray_menu.addAction(settings_action)
        quit_action = QAction('Quit Zelyr', self)
        quit_action.triggered.connect(self.force_quit)
        tray_menu.addAction(quit_action)
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

    def _init_hotkey(self) -> None:
        try:
            from zelyr.services.hotkey_service import GlobalHotkeyService
            self._hotkey_service = GlobalHotkeyService()
            self._hotkey_service.signaler.toggle_requested.connect(
                lambda: self.hide() if self.isVisible() else self.show()
            )
            self._hotkey_service.signaler.context_summoned.connect(self._on_context_summoned)
            self._hotkey_service.signaler.vision_summoned.connect(self._on_vision_summoned)
            # New registry-driven global signals
            self._hotkey_service.signaler.start_audio.connect(self._on_global_start_audio)
            self._hotkey_service.signaler.capture_screen.connect(self._on_vision_summoned)
            self._hotkey_service.start()
        except ImportError:
            logger.error('pynput not found. Global hotkey disabled.')

    def _register_local_shortcuts(self) -> None:
        """Create QShortcut bindings for every non-global entry in SHORTCUTS."""
        actions = {
            "new_chat":       lambda: self._on_new_chat(),
            "new_project":    lambda: self._on_new_project_requested(),
            "focus_input":    lambda: self._chat_display.input_bar.input_area.setFocus(),
            "toggle_sidebar": lambda: self._toggle_sidebar(),
            "clear_chat":     lambda: self._on_clear_chat(),
            "open_settings":  lambda: self.settings_drawer.toggle(),
            "prev_chat":      lambda: self._on_prev_chat(),
            "next_chat":      lambda: self._on_next_chat(),
            # search_history: no UI hook yet — skipped gracefully
        }
        for action_id, spec in SHORTCUTS.items():
            if spec.get("global"):
                continue
            key = spec.get("default")
            if not key:
                continue
            handler = actions.get(action_id)
            if handler is None:
                continue
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(handler)
            print(f"[GUI] Registered local shortcut: {action_id} -> {key}")

    def _toggle_sidebar(self) -> None:
        """Collapse or restore the sidebar via the splitter with smooth Notion OutExpo animation."""
        from PySide6.QtCore import QVariantAnimation, QEasingCurve
        sizes = self.splitter.sizes()
        total_width = sum(sizes)
        current_width = sizes[0]
        target_width = 0 if current_width > 0 else 280

        if hasattr(self, "_sidebar_anim") and self._sidebar_anim.state() == QVariantAnimation.State.Running:
            self._sidebar_anim.stop()

        self._sidebar_anim = QVariantAnimation(self)
        self._sidebar_anim.setDuration(220)
        self._sidebar_anim.setEasingCurve(QEasingCurve.Type.OutExpo)
        self._sidebar_anim.setStartValue(current_width)
        self._sidebar_anim.setEndValue(target_width)

        def _step(val):
            w = int(val)
            self.splitter.setSizes([w, max(0, total_width - w)])

        self._sidebar_anim.valueChanged.connect(_step)
        self._sidebar_anim.start()

    def _on_clear_chat(self) -> None:
        """Clear all bubbles from the chat view."""
        layout = self._chat_display.scroll_layout
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _on_prev_chat(self) -> None:
        """Select the previous chat item in the sidebar tree."""
        self._navigate_chat(direction=-1)

    def _on_next_chat(self) -> None:
        """Select the next chat item in the sidebar tree."""
        self._navigate_chat(direction=+1)

    def _navigate_chat(self, direction: int) -> None:
        """Walk the sidebar tree by `direction` (+1 next, -1 prev).

        Project headers are not selectable (ItemIsSelectable cleared in
        populate_projects), so we skip them and only land on chat leaves.
        """
        tree = self.sidebar.tree

        # Flatten all tree items into a list in display order.
        def _flatten(parent_item=None):
            items = []
            count = parent_item.childCount() if parent_item else tree.topLevelItemCount()
            for i in range(count):
                child = parent_item.child(i) if parent_item else tree.topLevelItem(i)
                items.append(child)
                items.extend(_flatten(child))
            return items

        all_items = _flatten()
        # Keep only selectable chat leaves (project headers have ItemIsSelectable cleared).
        selectable = [
            it for it in all_items
            if bool(it.flags() & Qt.ItemFlag.ItemIsSelectable)
        ]
        if not selectable:
            return

        selected = tree.selectedItems()
        current = selected[0] if selected else None

        if current and current in selectable:
            idx = selectable.index(current)
        else:
            # Nothing selected yet — jump to first or last depending on direction.
            idx = -1 if direction > 0 else 0

        new_idx = (idx + direction) % len(selectable)
        tree.setCurrentItem(selectable[new_idx])
        tree.scrollToItem(selectable[new_idx])

    def _on_global_start_audio(self) -> None:
        """Activate the mic from a global hotkey — show window then start STT."""
        self.show()
        self.activateWindow()
        self.raise_()
        # Simulate a mic toggle-on if the button exists
        if hasattr(self._chat_display, 'input_bar'):
            mic_btn = self._chat_display.input_bar.mic_btn
            if not mic_btn.isChecked():
                mic_btn.click()

    @Slot()
    def _on_context_summoned(self) -> None:
        """Fetch clipboard, inject into prompt, and summon."""
        from zelyr.services.clipboard_service import ClipboardService
        clipboard_text = ClipboardService.get_text()
        if clipboard_text:
            injection = f'\\n```\\n{clipboard_text}\\n```\\n'
            self._input.setText(injection)
            cursor = self._input.textCursor()
            cursor.setPosition(0)
            self._input.setTextCursor(cursor)
        self.show()
        self.activateWindow()
        self.raise_()
        if hasattr(self, '_input'):
            self._input.setFocus()

    @Slot()
    def _on_vision_summoned(self) -> None:
        """Capture screen, preview it, and summon."""
        from zelyr.services.vision_service import VisionService
        import os
        path = VisionService.capture_screen()
        if path and os.path.exists(path):
            self._chat_display.attach_image(path)
        self.show()
        self.activateWindow()
        self.raise_()
        if hasattr(self, '_input'):
            self._input.setFocus()

    def _check_updates_async(self):
        import asyncio
        from zelyr.updater.manager import UpdateManager

        async def _check():
            mgr = UpdateManager()
            res = await mgr.check_for_updates()
            if res.get('update_available'):
                self._status_updates.setText(f"[ Updates: {res['latest_version']} Available ]")
                self._status_updates.setProperty("status", "danger"); self._status_updates.style().unpolish(self._status_updates); self._status_updates.style().polish(self._status_updates)
                from PySide6.QtCore import QMetaObject, Q_ARG, Qt
                QMetaObject.invokeMethod(self, '_prompt_update', Qt.ConnectionType.QueuedConnection, Q_ARG(str, res['latest_version']))
            else:
                self._status_updates.setText('[ Updates: Up-to-Date ]')
                self._status_updates.setProperty("status", "success"); self._status_updates.style().unpolish(self._status_updates); self._status_updates.style().polish(self._status_updates)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(_check(), loop)
        else:
            asyncio.run(_check())

    def _connect_bridge(self) -> None:
        if not self._bridge:
            return
        
        # Core Chat Event
        if hasattr(self._bridge, 'chat_event'):
            self._bridge.chat_event.connect(lambda role, text: self._chat_display.add_bubble(role, text))
            
        # Standard Events
        self._bridge.token_received.connect(self._on_token)
        self._bridge.response_finished.connect(self._on_response_finished)
        self._bridge.telemetry_updated.connect(self._on_telemetry)
        self._bridge.error_occurred.connect(self._on_error)
        self._bridge.system_alert.connect(self._on_system_alert)
        
        # Health Radar Hooks
        if hasattr(self._bridge, 'telemetry_tick_received'):
            self._bridge.telemetry_tick_received.connect(self.health_radar.update_telemetry)
        if hasattr(self._bridge, 'lifecycle_state_changed'):
            self._bridge.lifecycle_state_changed.connect(self.health_radar.update_lifecycle)
        if hasattr(self._bridge, 'startup_service_update'):
            self._bridge.startup_service_update.connect(self.health_radar.update_service_status)
            
        self._bridge.request_gui_auth.connect(self._on_request_gui_auth)
        self._bridge.swarm_status_changed.connect(self._on_swarm_status_changed)
        self._bridge.swarm_agent_started.connect(self._on_swarm_started)
        self._bridge.swarm_agent_token.connect(self._on_swarm_token)
        self._bridge.swarm_agent_completed.connect(self._on_swarm_completed)
        self._bridge.zelyrfs_status.connect(self._on_zelyrfs_status)
        self._bridge.governor_approval_requested.connect(self._on_approval_requested)
        if hasattr(self._bridge, 'tool_status_changed'):
            self._bridge.tool_status_changed.connect(self._on_tool_status)
        pass
        self._bridge.ui_widget_generated.connect(self._on_widget_generated)
        pass
        self._bridge.startup_service_update.connect(self._on_startup_service_update)
        self._bridge.lifecycle_state_changed.connect(self._on_lifecycle_state_changed)
        self._bridge.agent_loop_status.connect(self._on_agent_loop_status)
        self._bridge.power_mode_changed.connect(self._on_power_mode_changed)

    def _on_startup_service_update(self, payload: dict) -> None:
        service = payload.get("service")
        state = payload.get("state")
        if service and state:
            if state == "READY":
                self._startup_status.setText(f"✅ {service} Ready")
                self._startup_status.setProperty("status", "success")
                self._startup_status.style().unpolish(self._startup_status)
                self._startup_status.style().polish(self._startup_status)
            elif state == "DEGRADED":
                self._startup_status.setText(f"⚠️ {service} Degraded")
                self._startup_status.setProperty("status", "danger")
                self._startup_status.style().unpolish(self._startup_status)
                self._startup_status.style().polish(self._startup_status)
        elif state == "READY":
            # The final startup.service.ready event has no service name
            self._startup_status.setText("✅ System Ready")
            self._startup_status.setProperty("status", "success")
            self._startup_status.style().unpolish(self._startup_status)
            self._startup_status.style().polish(self._startup_status)
            QTimer.singleShot(5000, lambda: self._startup_status.hide())

    def _on_lifecycle_state_changed(self, payload: dict) -> None:
        new_state = payload.get("new_state", "")
        if new_state == "READY":
            self._startup_status.setText("✅ System Ready")
            self._startup_status.setProperty("status", "success")
            self._startup_status.style().unpolish(self._startup_status)
            self._startup_status.style().polish(self._startup_status)
            QTimer.singleShot(5000, lambda: self._startup_status.hide())
        elif new_state == "DEGRADED":
            self._startup_status.setText("⚠️ System Degraded")
            self._startup_status.setProperty("status", "danger")
            self._startup_status.style().unpolish(self._startup_status)
            self._startup_status.style().polish(self._startup_status)
        elif new_state == "GUI_READY":
            self._startup_status.setText("⏳ Core initializing...")
            self._startup_status.setProperty("status", "warning")
            self._startup_status.style().unpolish(self._startup_status)
            self._startup_status.style().polish(self._startup_status)

    def _on_zelyrfs_status(self, status: str) -> None:
        self._status_zelyrfs.setText(f'ZelyrFS: {status}')

    def _on_daemon_connection_changed(self, state: str) -> None:
        if state == 'connected':
            self._status_daemon.setText('⚡ Daemon: Connected')
            self._status_daemon.setProperty("status", "success")
            self._status_daemon.style().unpolish(self._status_daemon)
            self._status_daemon.style().polish(self._status_daemon)
        elif state == 'connecting':
            self._status_daemon.setText('⏳ Daemon: Starting...')
            self._status_daemon.setProperty("status", "warning")
            self._status_daemon.style().unpolish(self._status_daemon)
            self._status_daemon.style().polish(self._status_daemon)
        else:
            self._status_daemon.setText('🔌 Daemon: Disconnected')
            self._status_daemon.setProperty("status", "danger")
            self._status_daemon.style().unpolish(self._status_daemon)
            self._status_daemon.style().polish(self._status_daemon)

    @Slot(dict)
    def _on_swarm_status_changed(self, payload: dict) -> None:
        pass

    def _on_request_gui_auth(self, tool_name: str, arguments: str, ctx: dict) -> None:
        from PySide6.QtWidgets import QMessageBox
        msg = QMessageBox(self)
        msg.setWindowTitle('[SECURITY APPROVAL REQUIRED]')
        msg.setText(f'Zelyr requests permission to execute an external action:\\n\\nTool: {tool_name}\\nCommand / Args: {arguments}')
        msg.setIcon(QMessageBox.Icon.Warning)
        allow_btn = msg.addButton('Allow Execution', QMessageBox.ButtonRole.AcceptRole)
        allow_btn.setProperty("status", "success")
        deny_btn = msg.addButton('Deny Action', QMessageBox.ButtonRole.RejectRole)
        deny_btn.setProperty("status", "danger")
        
        msg.exec()
        ctx['result']['granted'] = msg.clickedButton() == allow_btn
        ctx['event'].set()

    def _on_submit_background_task(self, prompt: str) -> None:
        self._chat_display.add_bubble('user', f'[SYSTEM/BACKGROUND] {prompt}')
        self._streaming_text = ''
        self._streaming_bubble = self._chat_display.add_bubble('assistant', '')
        self._active_swarm_pill = None
        self._bridge.submit_task(prompt)

    @Slot()
    def _on_send(self) -> None:
        text = self._input.toPlainText().strip()
        attachment_path = getattr(self._chat_display, '_current_attachment_path', None)
        
        if not text and not attachment_path:
            return
        
        self._input.clear()
        
        # Build the routing payload
        if attachment_path:
            import base64
            try:
                with open(attachment_path, 'rb') as f:
                    raw = f.read()
                b64 = base64.b64encode(raw).decode('utf-8')
                # Detect mime type from extension
                ext = attachment_path.rsplit('.', 1)[-1].lower()
                mime = 'image/png' if ext == 'png' else 'image/jpeg'
                multimodal_payload = [
                    {"type": "text", "text": text or "Describe this image."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
                ]
                route_payload = multimodal_payload  # list-style OpenAI content
                display_text = f'{text}\\n\\n*[Image: {os.path.basename(attachment_path)}]*' if text else f'*[Image: {os.path.basename(attachment_path)}]*'
            except Exception as e:
                logger.error(f'Failed to encode attachment: {e}')
                route_payload = text
                display_text = text
            finally:
                self._chat_display.clear_attachment()
        else:
            route_payload = text
            display_text = text
        
        self._chat_display.add_bubble('user', display_text.strip())
        self._streaming_text = ''
        self._streaming_bubble = self._chat_display.add_bubble('assistant', '⏳ Routing to Swarm Node…' if self._swarm.is_connected else '')
        self._active_swarm_pill = None

        # Ensure we have a conversation file created before saving
        if hasattr(self, '_project_manager'):
            if not self._current_chat_id:
                self._current_chat_id = self._project_manager.create_conversation(self._current_project_id, "New Chat")
            self._project_manager.append_message(self._current_project_id, self._current_chat_id, {
                'role': 'user',
                'content': display_text.strip()
            })
            self._refresh_sidebar()

        # Route: FastAPI Bridge, Remote Server Engine, Swarm Node, or Local Engine
        from zelyr.config import get_config
        config = get_config()
        engine_mode = getattr(config, 'engine_mode', 'local')
        if engine_mode == 'fastapi':
            self._send_fastapi_inference_task(text if not attachment_path else str(route_payload))
        elif engine_mode == 'remote':
            self._send_remote_server_task(text if not attachment_path else str(route_payload))
        elif self._swarm.is_connected:
            sent = self._swarm.send_prompt(text if not attachment_path else str(route_payload))
            if not sent:
                self._streaming_bubble.set_text('')
                self._bridge.submit_task(route_payload)
        else:
            self._bridge.submit_task(route_payload)

    def _send_remote_server_task(self, prompt: str) -> None:
        """Offload agentic execution to the centralized Zelyr Remote Server."""
        import threading
        import json
        from urllib.request import Request, urlopen
        from zelyr.config import get_config
        config = get_config()
        server_addr = getattr(config, 'remote_server_ip', '127.0.0.1:9412')
        if not server_addr.startswith("http://") and not server_addr.startswith("https://"):
            url = f"http://{server_addr}/v1/orchestrate"
        else:
            url = f"{server_addr}/v1/orchestrate"

        bubble = self._streaming_bubble

        def _worker():
            try:
                payload = json.dumps({"prompt": prompt, "session_id": getattr(self, '_current_chat_id', None)}).encode('utf-8')
                req = Request(url, data=payload, headers={'Content-Type': 'application/json'}, method='POST')
                with urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode('utf-8'))
                    answer = data.get("response", "Task completed.")
                    QTimer.singleShot(0, lambda: self._on_remote_server_response(answer, bubble))
            except Exception as e:
                err_msg = f"[Remote Server Error]: {e} — Falling back to local engine..."
                QTimer.singleShot(0, lambda: self._on_remote_server_error(prompt, bubble, err_msg))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_remote_server_response(self, response: str, bubble) -> None:
        if bubble:
            bubble.set_text(response)
        self._streaming_bubble = None
        self._streaming_text = ''
        self._chat_display._scroll_to_bottom()
        if hasattr(self, '_project_manager') and self._current_chat_id:
            self._project_manager.append_message(self._current_project_id, self._current_chat_id, {
                'role': 'assistant',
                'content': response
            })

    def _on_remote_server_error(self, prompt: str, bubble, error_msg: str) -> None:
        if bubble:
            bubble.set_text(error_msg)
        # Fallback to local engine
        self._bridge.submit_task(prompt)

    # ------------------------------------------------------------------
    # FastAPI HTTP Bridge (QThread)
    # ------------------------------------------------------------------

    def _send_fastapi_inference_task(self, prompt: str) -> None:
        """Post a chat-completion request to the local FastAPI backend.

        Uses :class:`~zelyr.gui.workers.InferenceWorker` (a ``QThread``
        subclass) so the HTTP round-trip never blocks the Qt event loop.
        A strong reference is kept in ``self._current_worker`` to prevent
        the Python GC from destroying the thread before it finishes.
        """
        from zelyr.gui.workers import InferenceWorker

        bubble = self._streaming_bubble

        worker = InferenceWorker(prompt, parent=self)
        # Keep a strong reference — without this Qt/Python GC may destroy
        # the thread object before run() has finished.
        self._current_worker = worker

        worker.response_received.connect(
            lambda text: self._on_fastapi_response(text, bubble)
        )
        worker.error_received.connect(
            lambda msg: self._on_fastapi_error(msg, bubble)
        )
        # Clean up the reference once the thread is done
        worker.finished.connect(self._on_fastapi_worker_finished)

        worker.start()

    @Slot()
    def _on_fastapi_worker_finished(self) -> None:
        """Release the worker reference after the QThread exits."""
        self._current_worker = None

    @Slot(str)
    def _on_fastapi_response(self, response: str, bubble) -> None:
        """Inject the FastAPI assistant reply into the chat bubble."""
        if bubble:
            bubble.set_text(response)
        self._streaming_bubble = None
        self._streaming_text = ''
        self._chat_display._scroll_to_bottom()
        if hasattr(self, '_project_manager') and self._current_chat_id:
            self._project_manager.append_message(
                self._current_project_id,
                self._current_chat_id,
                {'role': 'assistant', 'content': response},
            )

    @Slot(str)
    def _on_fastapi_error(self, error_msg: str, bubble) -> None:
        """Display the FastAPI error inside the assistant bubble."""
        if bubble:
            bubble.set_text(error_msg)
        self._streaming_bubble = None
        self._streaming_text = ''
        self._chat_display._scroll_to_bottom()



    @Slot(str)
    def _on_swarm_response(self, response: str) -> None:
        """Handle the final response from a remote Zelyr Swarm Node."""
        if self._streaming_bubble:
            self._streaming_bubble.set_text(response)
            self._streaming_bubble = None
        self._streaming_text = ''
        self._chat_display._scroll_to_bottom()
        # Persist to local JSON history
        if hasattr(self, '_project_manager') and self._current_chat_id:
            self._project_manager.append_message(self._current_project_id, self._current_chat_id, {
                'role': 'assistant',
                'content': response
            })
        # TTS — respects the speaker toggle in SettingsDrawer
        audio = getattr(self, '_audio', None)
        if audio and self.settings_drawer.tts_enabled and self._bridge:
            import asyncio
            asyncio.run_coroutine_threadsafe(audio.speak(response), self._bridge._loop)

    @Slot(str)
    def _on_model_changed(self, model: str) -> None:
        """Apply model selection from the Settings Drawer dropdown."""
        if model == 'Auto-Select':
            return
        try:
            from zelyr.config import get_config
            cfg = get_config()
            cfg.ollama_model = model
            if self._bridge and hasattr(self._bridge, '_engine'):
                llm = getattr(self._bridge._engine, 'ollama', None)
                if llm and hasattr(llm, 'config'):
                    llm.config.model = model
            logger.info(f'Model switched to: {model}')
            self._chat_display.add_bubble('assistant', f'🔄 Model switched to **{model}**')
        except Exception as e:
            logger.warning(f'Could not apply model change: {e}')

    @Slot(bool)
    def _on_mic_toggled(self, listening: bool) -> None:
        """Handle push-to-talk: start/stop recording and transcribe."""
        audio = getattr(self, '_audio', None)
        mic_btn = self._chat_display.input_bar.mic_btn
        if not audio or not audio.has_stt:
            # No STT available: uncheck the button visually and bail
            if mic_btn and mic_btn.isChecked():
                mic_btn.blockSignals(True)
                mic_btn.setChecked(False)
                mic_btn.blockSignals(False)
            return

        if listening:
            import asyncio
            async def _start():
                from PySide6.QtCore import QMetaObject, Qt, Q_ARG
                QMetaObject.invokeMethod(self._input, 'setPlaceholderText', Qt.ConnectionType.QueuedConnection, Q_ARG(str, 'Loading Voice Engine...'))
                QMetaObject.invokeMethod(mic_btn, 'setEnabled', Qt.ConnectionType.QueuedConnection, Q_ARG(bool, False))
                
                await audio.prepare_stt()
                
                QMetaObject.invokeMethod(mic_btn, 'setEnabled', Qt.ConnectionType.QueuedConnection, Q_ARG(bool, True))
                if not audio.has_stt:
                    mic_btn.blockSignals(True)
                    QMetaObject.invokeMethod(mic_btn, 'setChecked', Qt.ConnectionType.QueuedConnection, Q_ARG(bool, False))
                    mic_btn.blockSignals(False)
                    QMetaObject.invokeMethod(self._input, 'setPlaceholderText', Qt.ConnectionType.QueuedConnection, Q_ARG(str, 'Message...'))
                    return
                
                audio.start_listening_sync()
                QMetaObject.invokeMethod(self._input, 'setPlaceholderText', Qt.ConnectionType.QueuedConnection, Q_ARG(str, 'Listening...'))

            if self._bridge:
                asyncio.run_coroutine_threadsafe(_start(), self._bridge._loop)
        else:
            self._input.setPlaceholderText('Transcribing...')
            audio_data = audio.stop_listening()
            import asyncio

            async def _transcribe():
                text = await audio.transcribe(audio_data)
                from PySide6.QtCore import QMetaObject, Q_ARG
                QMetaObject.invokeMethod(
                    self, '_on_transcription_complete',
                    Qt.ConnectionType.QueuedConnection,
                    Q_ARG(str, text)
                )
            if self._bridge:
                asyncio.run_coroutine_threadsafe(_transcribe(), self._bridge._loop)

    @Slot(str)
    def _on_transcription_complete(self, text: str) -> None:
        self._input.setPlaceholderText('Ask Zelyr anything… (Enter to send, Shift+Enter for new line)')
        if text:
            current = self._input.toPlainText()
            self._input.setPlainText((current + ' ' + text).strip())

    @Slot()
    def _on_wake_word(self) -> None:
        self._chat_display.add_bubble('user', '[Wake Word Detected] Listening...')
        pass

    @Slot(str)
    def _on_token(self, token: str) -> None:
        if self._streaming_bubble:
            self._streaming_text += token
            # Do NOT html.escape here — _basic_markdown in the bubble already handles escaping
            self._streaming_bubble.set_text(self._streaming_text)
            self._chat_display._scroll_to_bottom()

    @Slot(str, str)
    def _on_tool_status(self, tool_id: str, status: str) -> None:
        pill_text = f"<i>[Zelyr is executing: {tool_id}...]</i><br><b>Status:</b> {status}"
        if self._streaming_bubble:
            self._streaming_text = pill_text
            self._streaming_bubble.set_text(self._streaming_text)
        else:
            self._streaming_text = pill_text
            self._streaming_bubble = self._chat_display.add_bubble('assistant', self._streaming_text)
        self._chat_display._scroll_to_bottom()

    @Slot(dict)
    def _on_agent_loop_status(self, payload: dict) -> None:
        if hasattr(self, '_chat_display'):
            action = payload.get('action', f"Zelyr is reasoning (Step {payload.get('iteration', '?')})...")
            self._chat_display.set_status(action)
            
    @Slot(bool, str)
    def _on_power_mode_changed(self, throttled: bool, msg: str) -> None:
        if hasattr(self, '_chat_display'):
            bubble = self._chat_display.add_bubble('system', msg)
            if bubble:
                # Optional: Make the bubble visually distinct
                if throttled:
                    bubble
                else:
                    bubble

    def _on_swarm_started(self, agent_name: str, task: str) -> None:
        if hasattr(self, '_swarm_hud'):
            self._chat_display.swarm_hud.add_pill(agent_name, task)

    @Slot(str, str)
    def _on_swarm_token(self, agent_name: str, chunk: str) -> None:
        if hasattr(self, '_swarm_hud'):
            self._chat_display.swarm_hud.update_pill(agent_name, chunk)

    @Slot(str, str)
    def _on_swarm_completed(self, agent_name: str, result: str) -> None:
        if hasattr(self, '_swarm_hud'):
            self._chat_display.swarm_hud.remove_pill(agent_name)

    @Slot(dict)
    def _on_telemetry(self, data: dict) -> None:
        pass

    def _on_response_finished(self, text: str) -> None:
        if self._streaming_bubble and (not self._streaming_text):
            # Do NOT html.escape — _basic_markdown already escapes safely
            self._streaming_bubble.set_text(text)
        self._streaming_bubble = None
        self._streaming_text = ''
        if hasattr(self, '_chat_display'):
            self._chat_display.clear_status()
            self._chat_display._scroll_to_bottom()
        
        # Save to JSON backend
        if hasattr(self, '_project_manager') and self._current_chat_id:
            self._project_manager.append_message(self._current_project_id, self._current_chat_id, {
                "role": "assistant",
                "content": text
            })
            self._refresh_sidebar()
            
        if hasattr(self, '_tts') and self._tts:
            import asyncio
            asyncio.run_coroutine_threadsafe(self._tts.speak(text), self._bridge._loop)

    @Slot(str)
    def _on_error(self, message: str) -> None:
        if self._streaming_bubble:
            self._streaming_bubble.deleteLater()
            self._streaming_bubble = None
            self._streaming_text = ''
        self._chat_display.add_bubble('tool', f'⚠️ {message}')
        self._chat_display._scroll_to_bottom()

    @Slot(str, str)
    def _on_system_alert(self, level: str, message: str) -> None:
        icon = "🚨" if level == "danger" else "⚠️"
        self._chat_display.add_bubble('assistant', f'{icon} **{message}**')
        self._chat_display._scroll_to_bottom()

    @Slot(bool, float)
    def _on_ollama_status_changed(self, is_online: bool, latency: float) -> None:
        if is_online:
            self._ollama_status_label.setText(f'🟢 Ollama: Online ({latency:.0f}ms)')
            self._ollama_status_label.setProperty("status", "success"); self._ollama_status_label.style().unpolish(self._ollama_status_label); self._ollama_status_label.style().polish(self._ollama_status_label)
            self._ollama_start_btn.setVisible(False)
            self._ollama_start_action.setVisible(False)
            self._bridge.refresh_models()
        else:
            self._ollama_status_label.setText('🔴 Ollama: Offline')
            self._ollama_status_label.setProperty("status", "danger"); self._ollama_status_label.style().unpolish(self._ollama_status_label); self._ollama_status_label.style().polish(self._ollama_status_label)
            self._ollama_start_btn.setVisible(True)
            self._ollama_start_action.setVisible(True)
            if getattr(self, '_first_ollama_ping', True):
                from zelyr.config import get_config
                if get_config().auto_ollama_start:
                    self._on_ollama_start_clicked()
        self._first_ollama_ping = False

    @Slot(str, dict)
    @Slot(dict)
    def _on_widget_generated(self, payload: dict) -> None:
        widget_type = payload.get('widget_type', 'unknown')
        spec = payload.get('spec', {})
        try:
            from zelyr.gui.widgets.sandbox_container import SandboxContainer
            sandbox = SandboxContainer(widget_type, spec, self)
            count = self._chat_display.chat_layout.count()
            self._chat_display.chat_layout.insertWidget(count - 1, sandbox)
            self._chat_display._scroll_to_bottom()
        except Exception as e:
            print(f'Failed to render widget: {e}')

    def _on_approval_requested(self, tool_name: str, arguments: dict) -> None:
        from zelyr.gui.widgets.governor_dialog import ExecutionGateDialog
        dlg = ExecutionGateDialog(tool_name, arguments, self)
        approved = dlg.exec_() == QDialog.Accepted
        self._bridge.send_approval_response(tool_name, approved)

    @Slot()
    def _on_settings_updated(self):
        import zelyr.gui.app as gui_app
        from PySide6.QtWidgets import QApplication
        from zelyr.config import get_config
        gui_app._load_stylesheet(QApplication.instance())
        config = get_config()
        if config.model_selection_mode == 'manual':
            self.update_model_label(f'{config.ollama_model} (Manual)')
        else:
            self.update_model_label(config.ollama_model)

    @Slot()
    def _on_ollama_start_clicked(self) -> None:
        self._ollama_status_label.setText('🟡 Starting Daemon...')
        self._ollama_status_label.setProperty("status", "warning"); self._ollama_status_label.style().unpolish(self._ollama_status_label); self._ollama_status_label.style().polish(self._ollama_status_label)
        self._ollama_start_btn.setEnabled(False)
        self._ollama_monitor.trigger_rapid_polling()
        import threading

        def _spawn():
            success = self._ollama_monitor.spawn_ollama_service()
            if not success:
                from PySide6.QtCore import QTimer
                QTimer.singleShot(0, lambda: self._ollama_start_btn.setEnabled(True))
        threading.Thread(target=_spawn, daemon=True).start()

    def _on_dock_visibility_changed(self, visible: bool) -> None:
        self._expert_btn.setChecked(visible)
        self._expert_btn.setText(f"Expert Mode: {('ON' if visible else 'OFF')}")
        
    def update_model_label(self, model: str) -> None:
        pass
