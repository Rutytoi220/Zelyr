"""Disposable PySide6 desktop GUI fixture with local HTTP state inspection for real CLI verification.

Runs exclusively for automated desktop testing on an isolated Hyprland workspace
and exposes read-only/reset state via localhost HTTP so test harnesses can
independently verify typed text, button clicks, focus changes, and key events.
"""

from __future__ import annotations

import argparse
import http.server
import json
import signal
import sys
import threading
import time
from typing import Any, Dict, List

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class FixtureWindow(QWidget):
    """High-contrast disposable PySide6 desktop application for live Zelyr testing."""

    reset_requested = Signal()
    activate_requested = Signal()

    def __init__(
        self,
        title: str = "Zelyr Desktop Test Fixture",
        distraction: bool = False,
        cross_surface: bool = False,
        initial_status: str = "IDLE",
    ) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self._lock = threading.Lock()
        self._counter = 0
        self._applied_text = ""
        self._cross_surface = cross_surface
        self._initial_status = initial_status
        self._status_text = initial_status
        self._mouse_clicks = 0
        self._double_clicks = 0
        self._right_clicks = 0
        self._key_presses = 0
        self._focus_in_count = 0
        self._focus_out_count = 0
        self._event_log: List[Dict[str, Any]] = []
        self._distraction = distraction

        if distraction:
            self._init_distraction_ui()
        else:
            self._init_fixture_ui()

        self.installEventFilter(self)
        self.reset_requested.connect(self._reset_ui_slot)
        self.activate_requested.connect(self._activate_ui_slot)

    def _init_distraction_ui(self) -> None:
        self.setFixedSize(360, 180)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        lbl = QLabel("Zelyr Distraction Window\n(Aux focus-steal fixture)")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setFont(QFont("Sans", 13, QFont.Weight.Bold))
        layout.addWidget(lbl)
        self.setStyleSheet("background-color: #2d1f24; color: #f8fafc;")

    def _init_fixture_ui(self) -> None:
        self.setFixedSize(640, 420)
        self.setStyleSheet(
            """
            QWidget {
                background-color: #161b22;
                color: #f0f6fc;
                font-family: Sans;
            }
            QLabel#headerLabel {
                font-size: 20px;
                font-weight: bold;
                color: #79c0ff;
                padding: 4px;
            }
            QLabel#statusLabel {
                background-color: #0d1117;
                border: 2px solid #30363d;
                border-radius: 6px;
                padding: 12px;
                font-size: 16px;
                font-weight: bold;
                color: #ffffff;
            }
            QLineEdit#inputField {
                background-color: #0b1623;
                border: 3px solid #58a6ff;
                border-radius: 6px;
                padding: 12px 16px;
                font-size: 17px;
                color: #ffffff;
                min-height: 40px;
            }
            QLineEdit#inputField:focus {
                border: 3px solid #79c0ff;
                background-color: #111923;
            }
            QPushButton#incrementBtn {
                background-color: #238636;
                border: 2px solid #3fb950;
                border-radius: 6px;
                padding: 14px 18px;
                font-size: 16px;
                font-weight: bold;
                color: #ffffff;
                min-height: 28px;
            }
            QPushButton#incrementBtn:hover {
                background-color: #2ea043;
            }
            QPushButton#applyBtn {
                background-color: #1f6feb;
                border: 2px solid #58a6ff;
                border-radius: 6px;
                padding: 14px 18px;
                font-size: 16px;
                font-weight: bold;
                color: #ffffff;
                min-height: 28px;
            }
            QPushButton#applyBtn:hover {
                background-color: #388bfd;
            }
            QPushButton#resetBtn {
                background-color: #30363d;
                border: 1px solid #8b949e;
                border-radius: 6px;
                padding: 8px 14px;
                font-size: 13px;
                color: #e6edf3;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        header_title = (
            "Zelyr Cross-Surface Confirmation"
            if self._cross_surface
            else "Zelyr Desktop Test Fixture"
        )
        self.header_label = QLabel(header_title)
        self.header_label.setObjectName("headerLabel")
        self.header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.header_label)

        self.status_label = QLabel()
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        hint_text = (
            "Confirmation Payload Input Field:"
            if self._cross_surface
            else "Command Note Input Field:"
        )
        input_hint = QLabel(hint_text)
        input_hint.setStyleSheet("font-size: 14px; color: #c9d1d9; font-weight: bold;")
        layout.addWidget(input_hint)

        self.input_field = QLineEdit()
        self.input_field.setObjectName("inputField")
        self.input_field.setPlaceholderText(
            "Enter confirmation string and click Confirm..."
            if self._cross_surface
            else "Click here and type text..."
        )
        self.input_field.returnPressed.connect(self._on_apply)
        self.input_field.installEventFilter(self)
        layout.addWidget(self.input_field)

        self.increment_btn = QPushButton("Increment Counter")
        self.increment_btn.setObjectName("incrementBtn")
        self.increment_btn.clicked.connect(self._on_increment)
        self.increment_btn.installEventFilter(self)

        apply_label = "Confirm" if self._cross_surface else "Apply Text"
        self.apply_btn = QPushButton(apply_label)
        self.apply_btn.setObjectName("applyBtn")
        self.apply_btn.clicked.connect(self._on_apply)
        self.apply_btn.installEventFilter(self)

        self.reset_btn = QPushButton("Reset State")
        self.reset_btn.setObjectName("resetBtn")
        self.reset_btn.clicked.connect(self._reset_ui_slot)
        self.reset_btn.installEventFilter(self)

        if self._cross_surface:
            # Dedicated single full-width Confirm button for crisp cross-surface targeting
            self.increment_btn.hide()
            self.reset_btn.hide()
            layout.addWidget(self.apply_btn)
        else:
            btn_row = QHBoxLayout()
            btn_row.setSpacing(16)
            btn_row.addWidget(self.increment_btn, stretch=1)
            btn_row.addWidget(self.apply_btn, stretch=1)
            layout.addLayout(btn_row)
            layout.addWidget(self.reset_btn)

        self._refresh_status_label()

    def _refresh_status_label(self) -> None:
        if self._distraction:
            return
        applied = self._applied_text if self._applied_text else "(none)"
        self.status_label.setText(
            f"Status: {self._status_text} | Counter: {self._counter} | Applied: {applied}"
        )

    def _record_event(self, kind: str, detail: str = "") -> None:
        with self._lock:
            self._event_log.append({"ts": round(time.time(), 3), "kind": kind, "detail": detail})
            if len(self._event_log) > 40:
                self._event_log = self._event_log[-40:]

    def _on_increment(self) -> None:
        with self._lock:
            self._counter += 1
            self._status_text = f"CLICKED:{self._counter}"
        self._refresh_status_label()
        self._record_event("increment", f"counter={self._counter}")

    def _on_apply(self) -> None:
        text = self.input_field.text()
        with self._lock:
            self._applied_text = text
            self._status_text = f"CONFIRMED:{text}" if self._cross_surface else f"APPLIED:{text}"
        self._refresh_status_label()
        self._record_event("apply", text)

    def _activate_ui_slot(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    def _reset_ui_slot(self) -> None:
        with self._lock:
            self._counter = 0
            self._applied_text = ""
            self._status_text = self._initial_status
            self._mouse_clicks = 0
            self._double_clicks = 0
            self._right_clicks = 0
            self._key_presses = 0
            self._focus_in_count = 0
            self._focus_out_count = 0
            self._event_log.clear()
        if not self._distraction:
            self.input_field.clear()
            self.clearFocus()
            self._refresh_status_label()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        etype = event.type()
        if watched is self and etype == QEvent.Type.WindowActivate:
            with self._lock:
                self._focus_in_count += 1
            self._record_event("window_activate", self.windowTitle())
        elif watched is self and etype == QEvent.Type.WindowDeactivate:
            with self._lock:
                self._focus_out_count += 1
            self._record_event("window_deactivate", self.windowTitle())
        elif etype == QEvent.Type.MouseButtonPress:
            with self._lock:
                self._mouse_clicks += 1
                if hasattr(event, "button") and event.button() == Qt.MouseButton.RightButton:
                    self._right_clicks += 1
            self._record_event("mouse_press", watched.objectName() or type(watched).__name__)
        elif etype == QEvent.Type.MouseButtonDblClick:
            with self._lock:
                self._double_clicks += 1
            self._record_event("mouse_dblclick", watched.objectName() or type(watched).__name__)
        elif etype == QEvent.Type.KeyPress:
            modifier_keys = {
                Qt.Key.Key_Shift,
                Qt.Key.Key_Control,
                Qt.Key.Key_Alt,
                Qt.Key.Key_AltGr,
                Qt.Key.Key_Meta,
                Qt.Key.Key_Super_L,
                Qt.Key.Key_Super_R,
            }
            k_code = event.key() if hasattr(event, "key") else None
            if k_code not in modifier_keys:
                with self._lock:
                    self._key_presses += 1
                key_txt = event.text() if hasattr(event, "text") else ""
                self._record_event("key_press", repr(key_txt or k_code))
        return super().eventFilter(watched, event)

    def snapshot_state(self) -> Dict[str, Any]:
        with self._lock:
            base: Dict[str, Any] = {
                "window_title": self.windowTitle(),
                "is_active_window": bool(self.isActiveWindow()),
                "counter": self._counter,
                "applied_text": self._applied_text,
                "status_text": self._status_text,
                "mouse_clicks": self._mouse_clicks,
                "double_clicks": self._double_clicks,
                "right_clicks": self._right_clicks,
                "key_presses": self._key_presses,
                "focus_in_count": self._focus_in_count,
                "focus_out_count": self._focus_out_count,
                "event_log": list(self._event_log),
            }
        if not self._distraction:
            pos = self.frameGeometry()
            base["window_geometry"] = {"x": pos.x(), "y": pos.y(), "width": pos.width(), "height": pos.height()}
            base["input_text"] = self.input_field.text()
            base["input_has_focus"] = bool(self.input_field.hasFocus())
            base["widget_centers_local"] = {
                w_name: [w.mapTo(self, w.rect().center()).x(), w.mapTo(self, w.rect().center()).y()]
                for w_name, w in (
                    ("input_field", self.input_field),
                    ("increment_btn", self.increment_btn),
                    ("apply_btn", self.apply_btn),
                    ("reset_btn", self.reset_btn),
                )
            }
        return base


def _make_handler(window: FixtureWindow, app: QApplication):
    class _Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/state":
                self._send_json(200, window.snapshot_state())
            elif self.path == "/health":
                self._send_json(200, {"ok": True, "title": window.windowTitle()})
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/reset":
                window.reset_requested.emit()
                time.sleep(0.05)
                self._send_json(200, {"ok": True, "state": window.snapshot_state()})
            elif self.path == "/activate":
                window.activate_requested.emit()
                time.sleep(0.05)
                self._send_json(200, {"ok": True, "state": window.snapshot_state()})
            elif self.path == "/quit":
                self._send_json(200, {"ok": True})
                app.quit()
            else:
                self._send_json(404, {"error": "not found"})

    return _Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Disposable Zelyr Desktop PySide6 Test Fixture")
    parser.add_argument("--port", type=int, required=True, help="Localhost HTTP port for state inspection")
    parser.add_argument("--title", type=str, default="Zelyr Desktop Test Fixture", help="Window title")
    parser.add_argument("--distraction", action="store_true", help="Launch auxiliary focus-steal distraction window")
    parser.add_argument("--cross-surface", action="store_true", help="Enable Cross-Surface confirmation mode")
    parser.add_argument("--initial-status", type=str, default=None, help="Initial status label text")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    signal.signal(signal.SIGTERM, lambda *_a: app.quit())
    signal.signal(signal.SIGINT, lambda *_a: app.quit())

    init_status = args.initial_status or ("AWAITING_CONFIRMATION" if args.cross_surface else "IDLE")
    win = FixtureWindow(
        title=args.title,
        distraction=args.distraction,
        cross_surface=args.cross_surface,
        initial_status=init_status,
    )
    win.show()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), _make_handler(win, app))
    srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
    srv_thread.start()

    try:
        return app.exec()
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
