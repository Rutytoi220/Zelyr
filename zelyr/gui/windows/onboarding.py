"""Zelyr V11 — "First Contact" Out-Of-Box Experience (OOBE).

A frameless, dark-mode, 4-page onboarding wizard shown the very first time a
user launches the Zelyr desktop application (see ``zelyr.gui.app.run_gui``).

Pages
-----
1. Welcome             — brand introduction.
2. System Diagnostics  — live background scan of Ollama / Tailscale / Audio.
3. The Arsenal         — introduces the 3 core Zelyr tool pillars.
4. Model Configuration — pick default Chat + Vision models from Ollama.

All system probing (Ollama HTTP ping, Tailscale IP discovery, `ollama list`)
runs on background ``QThread`` workers so the Qt/qasync event loop is never
blocked. This replaces the previous single-page accent-color/voice-mode
picker that used to live in ``zelyr.gui.windows.oobe_window`` (that module is
now a thin backward-compatible shim re-exporting :class:`OOBEWindow` from
here).
"""
from __future__ import annotations

import logging
import socket
import subprocess
from pathlib import Path

from PySide6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    Qt,
    QThread,
    Signal,
)
from PySide6.QtGui import QGuiApplication, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from zelyr.config import get_config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
#  Stylesheet — dark, minimal, rounded. Consistent with the Zelyr OOBE
#  sub-brand palette already used elsewhere in zelyr/gui/windows.
# ─────────────────────────────────────────────────────────────────────
_OOBE_QSS = """
    QDialog {
        background-color: #1e1e2e;
        color: #E4E4E7;
    }
    QWidget {
        font-family: 'Inter', 'Segoe UI', 'SF Pro Display', sans-serif;
    }
    QLabel {
        background: transparent;
        color: #E4E4E7;
    }
    QLabel#oobeTitle {
        font-size: 40px;
        font-weight: 800;
        color: #E4E4E7;
    }
    QLabel#oobeSubtitle {
        font-size: 15px;
        color: #A1A1AA;
    }
    QLabel#oobeHeading {
        font-size: 24px;
        font-weight: 700;
        color: #E4E4E7;
    }
    QLabel#oobeHint {
        font-size: 13px;
        color: #94E2AC;
    }
    QLabel#oobeHint[warning="true"] {
        color: #F5A623;
    }
    QLabel#oobeSectionLabel {
        font-size: 13px;
        font-weight: 600;
        color: #D4D4D8;
    }

    /* Diagnostics rows */
    QFrame#diagRow {
        background-color: #181825;
        border: 1px solid #313244;
        border-radius: 10px;
    }
    QLabel#diagStatus {
        font-size: 20px;
    }
    QLabel#diagTitle {
        font-size: 14px;
        font-weight: 600;
        color: #E4E4E7;
    }
    QLabel#diagDetail {
        font-size: 12px;
        color: #A1A1AA;
    }

    /* Arsenal cards */
    QFrame#arsenalCard {
        background-color: #181825;
        border: 1px solid #313244;
        border-radius: 10px;
    }
    QLabel#arsenalIcon {
        font-size: 28px;
    }
    QLabel#arsenalTitle {
        font-size: 15px;
        font-weight: 700;
        color: #E4E4E7;
    }
    QLabel#arsenalTag {
        font-size: 11px;
        font-weight: 700;
        color: #2ECC71;
        letter-spacing: 0.08em;
    }
    QLabel#arsenalDesc {
        font-size: 12px;
        color: #A1A1AA;
    }

    /* Model configuration */
    QComboBox#oobeCombo {
        background-color: #181825;
        color: #E4E4E7;
        padding: 10px 12px;
        border-radius: 8px;
        border: 1px solid #313244;
        font-size: 14px;
    }
    QComboBox#oobeCombo:hover {
        border: 1px solid #45475A;
    }
    QComboBox#oobeCombo::drop-down {
        border: none;
    }
    QComboBox#oobeCombo QAbstractItemView {
        background-color: #181825;
        color: #E4E4E7;
        selection-background-color: #2ECC71;
        selection-color: #11111B;
        border: 1px solid #313244;
    }
    QComboBox#oobeCombo:disabled {
        color: #71717A;
    }

    /* Bottom nav bar */
    QFrame#oobeNavBar {
        background-color: #11111B;
        border-top: 1px solid #313244;
    }
    QLabel#navDot {
        font-size: 10px;
        color: #45475A;
    }
    QLabel#navDot[active="true"] {
        color: #2ECC71;
    }
    QPushButton#navBackBtn {
        background-color: transparent;
        color: #A1A1AA;
        padding: 10px 20px;
        border-radius: 8px;
        border: 1px solid #313244;
        font-size: 13px;
        font-weight: 600;
    }
    QPushButton#navBackBtn:hover {
        background-color: #181825;
        color: #E4E4E7;
    }
    QPushButton#navBackBtn:disabled {
        color: #45475A;
        border: 1px solid #262637;
    }
    QPushButton#navNextBtn {
        background-color: #2ECC71;
        color: #11111B;
        padding: 10px 28px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 13px;
        border: none;
        min-width: 90px;
    }
    QPushButton#navNextBtn:hover {
        background-color: #27AE60;
    }
    QPushButton#navNextBtn:disabled {
        background-color: #313244;
        color: #71717A;
    }
"""


# ─────────────────────────────────────────────────────────────────────
#  Background workers
# ─────────────────────────────────────────────────────────────────────
class _DiagnosticsWorker(QThread):
    """Probes Ollama / Tailscale / Audio subsystem readiness off the UI thread."""

    check_finished = Signal(str, bool, str)  # check_id, ok, detail

    def run(self) -> None:
        self._check_ollama()
        self._check_tailscale()
        self._check_audio()

    def _check_ollama(self) -> None:
        ok, detail = False, "Not running"
        try:
            from zelyr.llm.ollama_client import OllamaClient, OllamaConfig

            base_url = get_config().ollama_base_url or "http://127.0.0.1:11434"
            client = OllamaClient(OllamaConfig(base_url=base_url, timeout=2.5))
            ok = client.is_available()
            detail = "Running on " + base_url.split("//")[-1] if ok else "Not reachable"
        except Exception as e:
            logger.debug("OOBE: ollama check failed: %s", e)
            detail = "Unavailable"
        self.check_finished.emit("ollama", ok, detail)

    def _check_tailscale(self) -> None:
        ip = self._tailscale_ip_via_cli() or self._tailscale_ip_via_interfaces()
        ok = ip is not None
        detail = f"Connected — {ip}" if ip else "Not connected"
        self.check_finished.emit("tailscale", ok, detail)

    @staticmethod
    def _tailscale_ip_via_cli() -> str | None:
        try:
            result = subprocess.run(
                ["tailscale", "ip", "-4"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode == 0:
                candidate = result.stdout.strip().splitlines()[0].strip()
                if candidate.startswith("100."):
                    return candidate
        except FileNotFoundError:
            pass  # tailscale binary not on PATH
        except Exception:
            pass
        return None

    @staticmethod
    def _tailscale_ip_via_interfaces() -> str | None:
        """Fallback: scan local network interfaces for a Tailscale CGNAT address."""
        try:
            import psutil

            for addrs in psutil.net_if_addrs().values():
                for addr in addrs:
                    if addr.family == socket.AF_INET and addr.address.startswith("100."):
                        return addr.address
        except Exception:
            pass
        return None

    def _check_audio(self) -> None:
        ok, detail = False, "Unavailable"
        try:
            from zelyr.core.audio import AudioManager

            am = AudioManager.instance()
            has_tts, has_stt = am.has_tts, am.has_stt
            ok = has_tts or has_stt
            if has_tts and has_stt:
                detail = "TTS + STT ready"
            elif has_tts:
                detail = "TTS ready (STT unavailable)"
            elif has_stt:
                detail = "STT ready (TTS unavailable)"
        except Exception as e:
            logger.debug("OOBE: audio check failed: %s", e)
        self.check_finished.emit("audio", ok, detail)


class _ModelListWorker(QThread):
    """Runs the equivalent of `ollama list` (HTTP /api/tags) in the background."""

    models_ready = Signal(list)

    def run(self) -> None:
        models: list[str] = []
        try:
            from zelyr.llm.ollama_client import OllamaClient, OllamaConfig

            base_url = get_config().ollama_base_url or "http://127.0.0.1:11434"
            client = OllamaClient(OllamaConfig(base_url=base_url, timeout=5.0))
            models = client.list_models()
        except Exception as e:
            logger.debug("OOBE: model list fetch failed: %s", e)
        self.models_ready.emit(models)


_VISION_HINTS = ("llava", "vision", "minicpm-v", "bakllava", "moondream", "-vl", "vl:", "pixtral")


def _guess_vision_model(models: list[str]) -> str | None:
    """Best-effort default pick for the Vision Model dropdown."""
    for model in models:
        low = model.lower()
        if any(hint in low for hint in _VISION_HINTS):
            return model
    return None


# ─────────────────────────────────────────────────────────────────────
#  Fade-in helper mixin
# ─────────────────────────────────────────────────────────────────────
class _FadeMixin:
    """Gives a QWidget page a reusable opacity fade-in animation."""

    def _init_fade(self) -> None:
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(1.0)
        self.setGraphicsEffect(self._opacity_effect)
        self._fade_anim: QPropertyAnimation | None = None

    def fade_in(self, duration: int = 450) -> None:
        self._opacity_effect.setOpacity(0.0)
        anim = QPropertyAnimation(self._opacity_effect, b"opacity", self)
        anim.setDuration(duration)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
        self._fade_anim = anim  # keep a reference alive


# ─────────────────────────────────────────────────────────────────────
#  Page 1 — Welcome
# ─────────────────────────────────────────────────────────────────────
class _WelcomePage(QWidget, _FadeMixin):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_fade()

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(18)

        logo_path = Path(__file__).resolve().parent.parent / "assets" / "logo.png"
        if logo_path.exists():
            logo_label = QLabel()
            pixmap = QPixmap(str(logo_path)).scaledToHeight(
                96, Qt.TransformationMode.SmoothTransformation
            )
            logo_label.setPixmap(pixmap)
            logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(logo_label)

        title = QLabel("Welcome to Zelyr")
        title.setObjectName("oobeTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("The Sovereign AI Operating System.")
        subtitle.setObjectName("oobeSubtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)


# ─────────────────────────────────────────────────────────────────────
#  Page 2 — System Diagnostics
# ─────────────────────────────────────────────────────────────────────
class _DiagnosticRow(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("diagRow")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(14)

        self.status_label = QLabel("⏳")
        self.status_label.setObjectName("diagStatus")
        self.status_label.setFixedWidth(28)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("diagTitle")
        text_col.addWidget(self.title_label)

        self.detail_label = QLabel("Scanning…")
        self.detail_label.setObjectName("diagDetail")
        text_col.addWidget(self.detail_label)

        layout.addLayout(text_col)
        layout.addStretch()

    def set_result(self, ok: bool, detail: str) -> None:
        # NOTE: no nested QGraphicsOpacityEffect here — the parent page
        # already carries one for its page-level fade-in, and Qt does not
        # compose two independent opacity effects across the same widget
        # subtree cleanly (it produces stale/duplicated paint regions).
        # The live ⏳ → ✅/⚠️ transition itself already reads as "animated"
        # since it happens asynchronously as each background probe completes.
        self.status_label.setText("✅" if ok else "⚠️")
        self.detail_label.setText(detail)


class _DiagnosticsPage(QWidget, _FadeMixin):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_fade()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(60, 40, 60, 20)
        layout.setSpacing(14)

        title = QLabel("System Diagnostics")
        title.setObjectName("oobeHeading")
        layout.addWidget(title)

        subtitle = QLabel("Zelyr is scanning your machine for local AI capabilities.")
        subtitle.setObjectName("oobeSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        layout.addSpacing(8)

        self.rows: dict[str, _DiagnosticRow] = {
            "ollama": _DiagnosticRow("Ollama Daemon detected…"),
            "tailscale": _DiagnosticRow("Tailscale Network…"),
            "audio": _DiagnosticRow("Local Audio Engine…"),
        }
        for row in self.rows.values():
            layout.addWidget(row)

        layout.addStretch()

    def apply_result(self, check_id: str, ok: bool, detail: str) -> None:
        row = self.rows.get(check_id)
        if row:
            row.set_result(ok, detail)


# ─────────────────────────────────────────────────────────────────────
#  Page 3 — The Arsenal
# ─────────────────────────────────────────────────────────────────────
class _ArsenalCard(QFrame):
    def __init__(self, icon: str, title: str, tag: str, description: str, parent=None):
        super().__init__(parent)
        self.setObjectName("arsenalCard")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(16)

        icon_label = QLabel(icon)
        icon_label.setObjectName("arsenalIcon")
        icon_label.setFixedWidth(48)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon_label)

        text_col = QVBoxLayout()
        text_col.setSpacing(3)

        title_label = QLabel(title)
        title_label.setObjectName("arsenalTitle")
        text_col.addWidget(title_label)

        tag_label = QLabel(tag.upper())
        tag_label.setObjectName("arsenalTag")
        text_col.addWidget(tag_label)

        desc_label = QLabel(description)
        desc_label.setObjectName("arsenalDesc")
        desc_label.setWordWrap(True)
        text_col.addWidget(desc_label)

        layout.addLayout(text_col, 1)


class _ArsenalPage(QWidget, _FadeMixin):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_fade()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(60, 40, 60, 20)
        layout.setSpacing(14)

        title = QLabel("The Arsenal")
        title.setObjectName("oobeHeading")
        layout.addWidget(title)

        subtitle = QLabel("Three pillars power Zelyr's local-first autonomy.")
        subtitle.setObjectName("oobeSubtitle")
        layout.addWidget(subtitle)

        layout.addSpacing(8)

        cards = [
            (
                "👻",
                "Ghost in the Machine",
                "Native Automation",
                "Sees, clicks, types, and operates your desktop directly — no middleman scripts required.",
            ),
            (
                "🧠",
                "Sensory Engine",
                "Vision & Voice",
                "Real-time screen understanding, speech recognition, and natural voice replies — all local.",
            ),
            (
                "🌐",
                "The Swarm",
                "Remote Nodes",
                "Offloads heavy work to your other machines over an encrypted Tailscale mesh (100.x.x.x).",
            ),
        ]
        for icon, title_text, tag, desc in cards:
            layout.addWidget(_ArsenalCard(icon, title_text, tag, desc))

        layout.addStretch()


# ─────────────────────────────────────────────────────────────────────
#  Page 4 — Model Configuration
# ─────────────────────────────────────────────────────────────────────
class _ModelConfigPage(QWidget, _FadeMixin):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_fade()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(60, 40, 60, 20)
        layout.setSpacing(14)

        title = QLabel("Model Configuration")
        title.setObjectName("oobeHeading")
        layout.addWidget(title)

        subtitle = QLabel("Choose your default models from what's already installed in Ollama.")
        subtitle.setObjectName("oobeSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        layout.addSpacing(8)

        self.status_label = QLabel("🔎 Scanning installed models (ollama list)…")
        self.status_label.setObjectName("oobeHint")
        layout.addWidget(self.status_label)

        layout.addSpacing(6)

        chat_col = QVBoxLayout()
        chat_col.setSpacing(6)
        chat_col.addWidget(self._section_label("💬 Chat Model"))
        self.chat_combo = QComboBox()
        self.chat_combo.setObjectName("oobeCombo")
        self.chat_combo.setEnabled(False)
        chat_col.addWidget(self.chat_combo)
        layout.addLayout(chat_col)

        vision_col = QVBoxLayout()
        vision_col.setSpacing(6)
        vision_col.addWidget(self._section_label("👁️ Vision Model"))
        self.vision_combo = QComboBox()
        self.vision_combo.setObjectName("oobeCombo")
        self.vision_combo.setEnabled(False)
        vision_col.addWidget(self.vision_combo)
        layout.addLayout(vision_col)

        layout.addStretch()

    @staticmethod
    def _section_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("oobeSectionLabel")
        return label

    def _set_status(self, text: str, warning: bool) -> None:
        self.status_label.setText(text)
        self.status_label.setProperty("warning", "true" if warning else "false")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def populate_models(self, models: list[str]) -> None:
        self.chat_combo.clear()
        self.vision_combo.clear()

        if not models:
            self._set_status(
                "⚠️ No installed models found. You can still finish setup and "
                "pull a model later with `ollama pull <model>`.",
                warning=True,
            )
            self.chat_combo.addItem("No models detected")
            self.vision_combo.addItem("No models detected")
            self.chat_combo.setEnabled(False)
            self.vision_combo.setEnabled(False)
            return

        self._set_status(f"✅ Found {len(models)} installed model(s).", warning=False)
        self.chat_combo.addItems(models)
        self.vision_combo.addItems(models)
        self.chat_combo.setEnabled(True)
        self.vision_combo.setEnabled(True)

        current_chat = get_config().ollama_model
        if current_chat in models:
            self.chat_combo.setCurrentText(current_chat)

        vision_guess = _guess_vision_model(models)
        if vision_guess:
            self.vision_combo.setCurrentText(vision_guess)

    def selected_chat_model(self) -> str:
        return self.chat_combo.currentText() if self.chat_combo.isEnabled() else ""

    def selected_vision_model(self) -> str:
        return self.vision_combo.currentText() if self.vision_combo.isEnabled() else ""


# ─────────────────────────────────────────────────────────────────────
#  Main wizard
# ─────────────────────────────────────────────────────────────────────
class OOBEWindow(QDialog):
    """Zelyr V11 "First Contact" onboarding wizard.

    Emits ``initialization_complete`` once the user finishes Page 4 — the
    caller (``zelyr.gui.app.run_gui``) is expected to only construct the main
    ``MainWindow`` after this signal fires, so the main chat UI never appears
    until the wizard is done.
    """

    initialization_complete = Signal()

    _PAGE_COUNT = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to Zelyr")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setFixedSize(840, 640)
        self

        self._diag_worker: _DiagnosticsWorker | None = None
        self._model_worker: _ModelListWorker | None = None
        self._fade_out_anim: QPropertyAnimation | None = None
        self._welcome_faded = False

        self._build_ui()
        self._center_on_screen()
        self._on_page_changed(0)

        # Kick off background scans immediately — QThread.start() is
        # non-blocking, so no need to defer this via a timer (which would
        # otherwise risk firing after the dialog has already been destroyed,
        # e.g. in fast test teardown or a rapid user click-through).
        self._start_background_scans()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Fade the welcome page in exactly once, the first time the wizard
        # actually becomes visible (guaranteed to be a live widget, unlike a
        # QTimer.singleShot callback scheduled from __init__).
        if not self._welcome_faded:
            self._welcome_faded = True
            self.welcome_page.fade_in(600)

    # ------------------------------------------------------------------ #
    #  UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.stack = QStackedWidget()
        self.welcome_page = _WelcomePage()
        self.diagnostics_page = _DiagnosticsPage()
        self.arsenal_page = _ArsenalPage()
        self.model_page = _ModelConfigPage()

        for page in (self.welcome_page, self.diagnostics_page, self.arsenal_page, self.model_page):
            self.stack.addWidget(page)

        self.stack.currentChanged.connect(self._on_page_changed)
        root.addWidget(self.stack, 1)

        # ── Bottom nav bar ──
        nav_frame = QFrame()
        nav_frame.setObjectName("oobeNavBar")
        nav_layout = QHBoxLayout(nav_frame)
        nav_layout.setContentsMargins(30, 16, 30, 16)
        nav_layout.setSpacing(12)

        dots_layout = QHBoxLayout()
        dots_layout.setSpacing(8)
        self.dots = []
        for _ in range(self._PAGE_COUNT):
            dot = QLabel("●")
            dot.setObjectName("navDot")
            dots_layout.addWidget(dot)
            self.dots.append(dot)
        nav_layout.addLayout(dots_layout)

        nav_layout.addStretch()

        self.back_btn = QPushButton("Back")
        self.back_btn.setObjectName("navBackBtn")
        self.back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.back_btn.clicked.connect(self._go_back)
        nav_layout.addWidget(self.back_btn)

        self.next_btn = QPushButton("Next")
        self.next_btn.setObjectName("navNextBtn")
        self.next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.next_btn.clicked.connect(self._go_next)
        nav_layout.addWidget(self.next_btn)

        root.addWidget(nav_frame)

    def _center_on_screen(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            x = geo.x() + (geo.width() - self.width()) // 2
            y = geo.y() + (geo.height() - self.height()) // 2
            self.move(x, y)

    # ------------------------------------------------------------------ #
    #  Background scans
    # ------------------------------------------------------------------ #
    def _start_background_scans(self) -> None:
        self._diag_worker = _DiagnosticsWorker()
        self._diag_worker.check_finished.connect(self.diagnostics_page.apply_result)
        self._diag_worker.start()

        self._model_worker = _ModelListWorker()
        self._model_worker.models_ready.connect(self.model_page.populate_models)
        self._model_worker.start()

    # ------------------------------------------------------------------ #
    #  Navigation
    # ------------------------------------------------------------------ #
    def _go_next(self) -> None:
        index = self.stack.currentIndex()
        if index >= self._PAGE_COUNT - 1:
            self._on_finish()
            return
        self.stack.setCurrentIndex(index + 1)

    def _go_back(self) -> None:
        index = self.stack.currentIndex()
        if index > 0:
            self.stack.setCurrentIndex(index - 1)

    def _on_page_changed(self, index: int) -> None:
        page = self.stack.widget(index)
        if isinstance(page, _FadeMixin):
            page.fade_in()

        self.back_btn.setEnabled(index > 0)
        self.next_btn.setText("Finish" if index == self._PAGE_COUNT - 1 else "Next")

        for i, dot in enumerate(self.dots):
            dot.setProperty("active", "true" if i == index else "false")
            dot.style().unpolish(dot)
            dot.style().polish(dot)

    # ------------------------------------------------------------------ #
    #  Finish / handoff
    # ------------------------------------------------------------------ #
    def _on_finish(self) -> None:
        self.next_btn.setEnabled(False)
        self.back_btn.setEnabled(False)
        self.next_btn.setText("Launching…")

        config = get_config()
        chat_model = self.model_page.selected_chat_model()
        vision_model = self.model_page.selected_vision_model()
        if chat_model:
            config.ollama_model = chat_model
        if vision_model:
            config.vision_model = vision_model
        config.oobe_completed = True
        config.first_launch = False
        config.save()

        self._fade_out_anim = QPropertyAnimation(self, b"windowOpacity")
        self._fade_out_anim.setDuration(500)
        self._fade_out_anim.setStartValue(1.0)
        self._fade_out_anim.setEndValue(0.0)
        self._fade_out_anim.setEasingCurve(QEasingCurve.Type.OutQuad)
        self._fade_out_anim.finished.connect(self._handoff)
        self._fade_out_anim.start()

    def _handoff(self) -> None:
        self.initialization_complete.emit()
        self.accept()
        self.accept()

    # ------------------------------------------------------------------ #
    #  Safety guards
    # ------------------------------------------------------------------ #
    def keyPressEvent(self, event) -> None:
        # First-run setup cannot be skipped via Escape — avoids stranding the
        # app with no window at all (tray/MainWindow don't exist yet).
        if event.key() == Qt.Key.Key_Escape:
            event.ignore()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        for worker in (self._diag_worker, self._model_worker):
            if worker is not None and worker.isRunning():
                worker.wait(3000)
        super().closeEvent(event)
