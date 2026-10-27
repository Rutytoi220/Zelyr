"""Zelyr Desktop v3.0 — Application Entry-Point.

Usage
-----
CLI shortcut:   zelyr-gui
Module run:     python3 -m zelyr.gui.app
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PySide6.QtWidgets import QApplication, QSystemTrayIcon
    from zelyr.gui.main_window import MainWindow

logger = logging.getLogger(__name__)


def _load_stylesheet(app: "QApplication") -> None:  # type: ignore[name-defined]
    from zelyr.gui.styles.theme_manager import get_theme_manager
    from zelyr.gui.config_manager import get_ui_config_manager
    
    ui_config = get_ui_config_manager().load()
    theme_name = ui_config.theme.lower()
    
    # We map 'minimalist' or 'system' or 'dark' to zelyr_pro as the default fallback for now
    if theme_name not in ["zelyr_pro", "jarvis"]:
        theme_name = "zelyr_pro"
        
    theme_manager = get_theme_manager()
    theme_manager.apply_theme(app, theme_name)
    logger.debug("Applied theme via ThemeManager: %s", theme_name)


def _build_tray(app: "QApplication", window: "MainWindow") -> "QSystemTrayIcon":  # type: ignore[name-defined]
    from PySide6.QtWidgets import QSystemTrayIcon, QMenu
    from PySide6.QtGui import QIcon, QAction
    from zelyr.config import get_config, AuthMode

    tray = QSystemTrayIcon(app)
    
    # Use real icon if exists
    assets_dir = Path(__file__).parent / "assets"
    if (assets_dir / "logo.svg").exists():
        tray.setIcon(QIcon(str(assets_dir / "logo.svg")))
    elif (assets_dir / "logo.png").exists():
        tray.setIcon(QIcon(str(assets_dir / "logo.png")))
    else:
        # Use a simple text-based fallback icon (Unicode circle in a pixmap)
        from PySide6.QtGui import QPixmap, QColor, QPainter
        pix = QPixmap(32, 32)
        pix.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pix)
        painter.setPen(QColor("#10b981"))
        painter.setFont(app.font())
        painter.drawText(pix.rect(), 0x84, "Z")  # Qt.AlignCenter
        painter.end()
        tray.setIcon(QIcon(pix))
        
    tray.setToolTip("Zelyr Desktop v3.0")

    menu = QMenu()

    show_action = QAction("Show / Hide Window", menu)
    show_action.triggered.connect(lambda: (
        window.show() if not window.isVisible() else window.hide()
    ))
    menu.addAction(show_action)

    menu.addSeparator()

    def _set_mode(mode: AuthMode) -> None:
        get_config().auth_mode = mode
        window._refresh_auth_ui()

    mode_autopilot = QAction("⚡ Mode: Autopilot", menu)
    mode_autopilot.triggered.connect(lambda: _set_mode(AuthMode.AUTOPILOT))
    menu.addAction(mode_autopilot)

    mode_basic = QAction("🛡️ Mode: Basic", menu)
    mode_basic.triggered.connect(lambda: _set_mode(AuthMode.BASIC))
    menu.addAction(mode_basic)

    mode_strict = QAction("🔒 Mode: Strict", menu)
    mode_strict.triggered.connect(lambda: _set_mode(AuthMode.STRICT))
    menu.addAction(mode_strict)

    menu.addSeparator()

    quit_action = QAction("Quit Zelyr", menu)
    quit_action.triggered.connect(app.quit)
    menu.addAction(quit_action)

    tray.setContextMenu(menu)
    tray.activated.connect(
        lambda reason: window.show() if reason == QSystemTrayIcon.ActivationReason.Trigger else None
    )
    tray.show()
    return tray


def run_gui() -> None:
    """Launch the Zelyr Desktop GUI (blocking call)."""
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
    except ImportError:
        print(
            "[Zelyr GUI] PySide6 is not installed.\n"
            "Run: pip install PySide6 qasync\n"
            "Then restart: zelyr-gui"
        )
        sys.exit(1)

    try:
        import qasync
    except ImportError:
        print(
            "[Zelyr GUI] qasync is not installed.\n"
            "Run: pip install qasync\n"
            "Then restart: zelyr-gui"
        )
        sys.exit(1)

    # --- QApplication ---
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    
    try:
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    except AttributeError:
        pass
        
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Zelyr Desktop")
    app.setApplicationVersion("3.0")
    app.setOrganizationName("Zelyr")
    
    # --- Single Instance IPC ---
    from PySide6.QtNetwork import QLocalSocket, QLocalServer
    socket = QLocalSocket()
    socket.connectToServer("zelyr_desktop_v3")
    if socket.waitForConnected(500):
        socket.write(b"WAKEUP")
        socket.waitForBytesWritten(1000)
        socket.disconnectFromServer()
        sys.exit(0)

    server = QLocalServer()
    server.removeServer("zelyr_desktop_v3")
    server.listen("zelyr_desktop_v3")
    app._ipc_server = server  # Prevent GC

    # Wayland/Desktop grouping (Qt 6.5+ automatically appends .desktop)
    app.setDesktopFileName("Zelyr")
    
    # Don't quit when closing the main window (keep tray running)
    app.setQuitOnLastWindowClosed(False)
    
    # Global App Icon
    from PySide6.QtGui import QIcon
    assets_dir = Path(__file__).parent / "assets"
    if (assets_dir / "logo.svg").exists():
        app.setWindowIcon(QIcon(str(assets_dir / "logo.svg")))
    elif (assets_dir / "logo.png").exists():
        app.setWindowIcon(QIcon(str(assets_dir / "logo.png")))
        
    _load_stylesheet(app)

    # --- Zelyr core wiring ---
    from zelyr.gui.bridge import ZelyrBridge
    from zelyr.gui.main_window import MainWindow
    from zelyr.config import get_config

    bridge = ZelyrBridge()

    # --- qasync event loop ---
    # We must set this up BEFORE the windows so async signals can fire if needed
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    bridge.set_event_loop(loop)

    config = get_config()
    window = None
    tray = None
    
    def launch_main_hud():
        nonlocal window, tray
        window = MainWindow(bridge=bridge)
        if hasattr(config, "ollama_model") and config.ollama_model:
            window.update_model_label(config.ollama_model)
        tray = _build_tray(app, window)
        window.show()
        # Emit GUI_READY lifecycle event
        from zelyr.core.lifecycle import AppLifecycleState
        try:
            loop.call_soon(
                lambda: bridge._client and None  # ensure loop is alive
            )
        except Exception:
            pass
        # Publish lifecycle GUI_READY on daemon-side if connected
        import json
        if bridge._client and hasattr(bridge._client, 'websocket'):
            try:
                asyncio.ensure_future(
                    bridge._client.websocket.send(json.dumps({
                        "type": "publish",
                        "event": {
                            "event_type": "lifecycle.state_changed",
                            "source": "gui",
                            "data": {"new_state": AppLifecycleState.GUI_READY.value}
                        }
                    }))
                )
            except Exception:
                pass

    # Check first launch via new unified config
    if not config.oobe_completed:
        from zelyr.gui.widgets.oobe_wizard import OobeWizardDialog
        wizard = OobeWizardDialog()
        wizard.exec()
        
        # Refresh config in case it was updated during the wizard
        config = get_config()
        
    # Launch main HUD after ensuring OOBE is complete
    launch_main_hud()

    def _on_ipc_connection() -> None:
        sock = server.nextPendingConnection()
        def on_ready_read():
            data = sock.readAll().data()
            if b"WAKEUP" in data:
                if window:
                    window.show()
                    window.raise_()
                    window.activateWindow()
                elif hasattr(app, "_onboarding"):
                    app._onboarding.show()
                    app._onboarding.raise_()
                    app._onboarding.activateWindow()
            sock.disconnectFromServer()
        sock.readyRead.connect(on_ready_read)
        # Handle case where data is already available
        if sock.bytesAvailable():
            on_ready_read()

    server.newConnection.connect(_on_ipc_connection)

    # Start IPC Client Connection
    bridge.initialize_client()

    with loop:
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            pass


def main() -> None:
    """Entrypoint for the ``zelyr-gui`` console script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_gui()


if __name__ == "__main__":
    main()
