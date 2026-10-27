from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QTextBrowser, QPushButton, QScrollArea, QFrame, QLabel, QSizePolicy
)
from PySide6.QtCore import Qt, QSize, Signal
from zelyr.gui.styles.theme_manager import ThemeManager
import markdown
from typing import Optional
import re
from zelyr.gui.jit_gui_compiler import JITSecurityException

class AutoExpandTextEdit(QTextEdit):
    return_pressed = Signal()

    def __init__(self, theme_manager: ThemeManager):
        super().__init__()
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setPlaceholderText("Ask Zelyr...")
        self.textChanged.connect(self.adjust_height)
        self.setFixedHeight(36) 

    def adjust_height(self):
        doc_height = self.document().size().height()
        new_height = max(36, min(int(doc_height) + 4, 120))
        self.setFixedHeight(new_height)


    def keyPressEvent(self, event):
        from PySide6.QtCore import Qt
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                super().keyPressEvent(event) # Shift+Enter = Newline
            else:
                self.return_pressed.emit() # Enter = Send
                event.accept()
        else:
            super().keyPressEvent(event)

class ModernInputBar(QFrame):
    message_ready = Signal(str)
    image_attached = Signal(str)
    mic_toggled = Signal(bool)

    def __init__(self, theme_manager: ThemeManager):
        super().__init__()
        self.setObjectName("input_container")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumHeight(48)
        
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(8, 4, 8, 4)
        self.layout.setSpacing(8)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        from PySide6.QtGui import QIcon
        from PySide6.QtCore import QSize
        import os
        from pathlib import Path
        
        # Resolve absolute path to assets/icons
        base_dir = Path(__file__).parent.parent.resolve()
        icons_dir = base_dir / "assets" / "icons"


        from zelyr.gui.styles.squircle import AnimatedSquircleButton

        self.attach_btn = AnimatedSquircleButton(radius=18, n=3.2, duration_ms=150)
        self.attach_btn.setIcon(QIcon(str(icons_dir / "attach.svg")))
        self.attach_btn.setIconSize(QSize(18, 18))
        self.attach_btn.setObjectName("attach_btn")
        self.attach_btn.setFixedSize(36, 36)

        self.input_area = AutoExpandTextEdit(theme_manager)
        self.input_edit = self.input_area  # ALIAS for main_window.py
        self.input_area.setFixedHeight(36)
        
        self.mic_btn = AnimatedSquircleButton(radius=18, n=3.2, duration_ms=150)
        self.mic_btn.setObjectName("mic_btn")
        self.mic_btn.setIcon(QIcon(str(icons_dir / "mic.svg")))
        self.mic_btn.setIconSize(QSize(18, 18))
        self.mic_btn.setCheckable(True)
        self.mic_btn.setFixedSize(36, 36)
        self.mic_btn.toggled.connect(self.mic_toggled.emit)
        
        self.send_btn = AnimatedSquircleButton(radius=18, n=3.2, duration_ms=150)
        self.send_btn.setObjectName("send_btn")
        self.send_btn.setIcon(QIcon(str(icons_dir / "send.svg")))
        self.send_btn.setIconSize(QSize(18, 18))
        self.send_btn.setFixedSize(36, 36)
        self.send_btn.clicked.connect(self._on_send, Qt.ConnectionType.UniqueConnection)

        
        self.input_area.return_pressed.connect(self.send_btn.click, Qt.ConnectionType.UniqueConnection)

        self.layout.addWidget(self.attach_btn)
        self.layout.addWidget(self.input_area)
        self.layout.addWidget(self.mic_btn)
        self.layout.addWidget(self.send_btn)

        self._apply_theme()

    def _apply_theme(self):
        # Delegate styles to base.qss.template
        pass
        
    def _on_send(self):
        text = self.input_area.toPlainText().strip()
        if text:
            self.message_ready.emit(text)
            self.input_area.clear()

class ModernChatBubble(QFrame):
    def __init__(self, role: str, text: str, theme_manager: ThemeManager):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setObjectName("chat_bubble")
        self.role = role
        self.setProperty("role", role)
        self._raw_text = text
        self.theme_manager = theme_manager

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(16, 12, 16, 12)
        self.layout.setSpacing(0)

        self.text_browser = QTextBrowser()
        self.text_browser.setOpenExternalLinks(True)
        self.text_browser.document().setDocumentMargin(0)
        self.text_browser.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.text_browser.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        html_content = markdown.markdown(self._raw_text, extensions=['fenced_code', 'tables'])
        self.text_browser.setHtml(html_content)
        
        self.layout.addWidget(self.text_browser)
        self._apply_theme()

    def _apply_theme(self):
        self.update()

    def paintEvent(self, event):
        from zelyr.gui.styles.squircle import SquirclePath
        from PySide6.QtGui import QPainter, QBrush, QPen, QColor
        from PySide6.QtCore import QRectF

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        if rect.width() <= 0 or rect.height() <= 0:
            return

        # Continuous super-ellipse curvature with degree n ≈ 3.2
        radius = min(rect.height() / 2.0, 18.0)
        path = SquirclePath(rect, n=3.2, radius=radius)

        tokens = getattr(self.theme_manager, 'theme', {}) if hasattr(self, 'theme_manager') and self.theme_manager else {}
        primary = QColor(tokens.get("primary", "#8B5CF6"))
        surface = QColor(tokens.get("bg_surface", "#161B22"))
        borders = QColor(tokens.get("borders", "#30363D"))

        if self.role == "user":
            painter.fillPath(path, QBrush(primary))
        else:
            painter.fillPath(path, QBrush(surface))
            pen = QPen(borders, 1.0)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawPath(path)

    def set_text(self, text: str):
        self._raw_text = text
        html_content = markdown.markdown(self._raw_text, extensions=['fenced_code', 'tables'])
        self.text_browser.setHtml(html_content)
        # Trigger a height recalculation after content changes.
        self.resizeEvent(None)

    def resizeEvent(self, event):
        if event is not None:
            super().resizeEvent(event)
        import math
        vp_width = self.text_browser.viewport().width()
        if vp_width > 0:
            # Force text reflow to the exact viewport width so Qt knows the height.
            self.text_browser.document().setTextWidth(vp_width)
        exact_h = math.ceil(self.text_browser.document().size().height())
        if exact_h > 0:
            self.text_browser.setMinimumHeight(exact_h)
            self.setMinimumHeight(exact_h + 24)  # 24px = 12px top + 12px bottom padding

class ModernChatDisplay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.theme_manager = getattr(parent, 'theme_manager', ThemeManager())
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(32, 0, 32, 24)
        self.layout.setSpacing(24)

        from zelyr.gui.widgets.swarm_hud import SwarmHUD
        self.swarm_hud = SwarmHUD()
        if hasattr(self, 'swarm_hud') and self.swarm_hud is not None:
            self.swarm_hud.hide()
            self.swarm_hud.setObjectName("swarm_hud")
        self.layout.addWidget(self.swarm_hud)
        
        top_bar = QHBoxLayout()
        top_bar.addStretch()
        from zelyr.gui.styles.squircle import AnimatedSquircleButton
        self.settings_btn = AnimatedSquircleButton("⚙  Settings", radius=14.0, n=3.2, duration_ms=180)
        self.settings_btn.setObjectName("sidebar_action_btn")
        self.settings_btn.setFixedHeight(34)
        top_bar.addWidget(self.settings_btn)
        self.layout.addLayout(top_bar)

        self.scroll_area = QScrollArea()
        self.scroll_area.verticalScrollBar().setSingleStep(15)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        self.scroll_widget = QWidget()
        self.scroll_widget.setObjectName("chat_scroll_widget")
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.chat_layout = self.scroll_layout
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.setSpacing(16)
        self.scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        # Sentinel stretch — keeps bubbles pinned to natural height.
        # add_bubble inserts before this item so bubbles never stretch to fill.
        self.scroll_layout.addStretch(1)
        
        self.scroll_area.setWidget(self.scroll_widget)
        
        self.watermark = QLabel("Zelyr v11.2", self.scroll_area.viewport())
        self.watermark.setObjectName("chat_watermark")
        self.watermark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.watermark.lower()
        
        self.layout.addWidget(self.scroll_area)

        # Image attached indicator
        self.image_indicator = QLabel("📸 Screen Captured")
        self.image_indicator.setStyleSheet("color: #A5B4FC; background: #312E81; padding: 4px 8px; border-radius: 8px; font-size: 11px;")
        self.image_indicator.hide()
        self.layout.addWidget(self.image_indicator, 0, Qt.AlignmentFlag.AlignLeft)

        self.input_bar = ModernInputBar(self.theme_manager)
        self.layout.addWidget(self.input_bar, 0, Qt.AlignmentFlag.AlignBottom)
        self.input_bar.message_ready.connect(self._handle_user_message)
        self.input_bar.attach_btn.clicked.connect(self._capture_screen)
        self._current_worker = None
        self._temp_bubble = None
        self._current_image_b64 = None

    def _capture_screen(self):
        import subprocess
        import base64
        try:
            result = subprocess.run(["grim", "-c", "-"], capture_output=True, check=True)
            self._current_image_b64 = base64.b64encode(result.stdout).decode("utf-8")
            self.image_indicator.show()
        except Exception as e:
            print(f"Screenshot failed: {e}")

    def _handle_user_message(self, text: str):
        # 1. Extract text and clear input box is handled by ModernInputBar._on_send
        # 2. Render the 'user' bubble immediately
        self.add_bubble("user", text)
        
        # 3. Render a temporary 'assistant' bubble with a loading state
        self._temp_bubble = self.add_bubble("assistant", "Thinking...")
        self._current_text = ""
        self._first_token_received = False
        
        # 4. Instantiate the InferenceWorker, connect signals and start
        from zelyr.gui.workers import InferenceWorker
        worker = InferenceWorker(text, image_b64=self._current_image_b64, parent=self)
        self._current_worker = worker
        
        worker.response_received.connect(self._on_inference_response)
        worker.error_received.connect(self._on_inference_error)
        worker.token_received.connect(self._on_token_received)
        
        worker.start()
        
        # Cleanup
        self._current_image_b64 = None
        self.image_indicator.hide()

    def _on_token_received(self, token: str):
        if self._temp_bubble:
            if not self._first_token_received:
                self._current_text = ""
                self._first_token_received = True
            self._current_text += token
            self._temp_bubble.set_text(self._current_text)

    def _on_inference_response(self, text: str):
        if self._temp_bubble:
            self._temp_bubble.set_text(text)
            self._temp_bubble = None
        self._current_worker = None

    def _on_inference_error(self, error_msg: str):
        if self._temp_bubble:
            self._temp_bubble.set_text(error_msg)
            self._temp_bubble = None
        self._current_worker = None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'watermark'):
            self.watermark.resize(self.scroll_area.viewport().size())

    def attach_image(self, filepath: str):
        pass # mock
        
    def clear_attachment(self):
        pass # mock
        
    def _scroll_to_bottom(self):
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        )

    def add_bubble(self, role: str, text: str):
        if text.strip() == "Zelyr v11.2":
            return None

        if text.strip() == "":
            text = "<i>[Executing Tool...]</i>"

        if self.watermark.isVisible():
            self.watermark.hide()

        # Parse for JIT Generative UI widget block
        jit_match = re.search(r"<gui_widget>\s*(.*?)\s*</gui_widget>", text, re.DOTALL)
        if jit_match:
            compiler = None
            code = jit_match.group(1)
            try:
                widget = compiler.compile_widget(code)
                if widget:
                    # Render the widget instead of the text
                    return self.add_dynamic_widget(widget, title="Generated Interface")
            except JITSecurityException as e:
                # Render error instead
                text = text.replace(jit_match.group(0), f"**[SECURITY BLOCKED]** JIT Compilation stopped: {str(e)}")
            except Exception as e:
                text = text.replace(jit_match.group(0), f"**[GUI ERROR]** JIT Compilation failed: {str(e)}")

        # Clean up any leftover tags just in case
        text = re.sub(r"<gui_widget>.*?</gui_widget>", "", text, flags=re.DOTALL)

        bubble = ModernChatBubble(role, text, self.theme_manager)

        from PySide6.QtWidgets import QSpacerItem
        
        wrapper = QHBoxLayout()
        if role == "user":
            wrapper.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
            wrapper.addWidget(bubble)
        else:
            wrapper.addWidget(bubble)
            wrapper.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        # Insert before the sentinel stretch so bubbles don't expand vertically.
        self.scroll_layout.insertLayout(self.scroll_layout.count() - 1, wrapper)
        self._scroll_to_bottom()
        return bubble

    def add_dynamic_widget(self, widget: QWidget, title: Optional[str] = None):
        """Embeds a compiled PySide6 widget directly into the chat flow."""
        if self.watermark.isVisible():
            self.watermark.hide()
            
        frame = QFrame()
        frame.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        frame.setObjectName("dynamic_widget_frame")
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        
        if title:
            title_lbl = QLabel(title)
            title_lbl.setObjectName("dynamic_widget_title")
            # Apply some bold styling or a specific font if we want, but objectName is enough
            font = title_lbl.font()
            font.setBold(True)
            title_lbl.setFont(font)
            layout.addWidget(title_lbl)
            
        layout.addWidget(widget)
        
        wrapper = QHBoxLayout()
        wrapper.addWidget(frame)
        wrapper.addStretch() # Align left like AI bubbles
        
        self.scroll_layout.insertLayout(self.scroll_layout.count() - 1, wrapper)
        self._scroll_to_bottom()
        return frame

    def set_status(self, action_text: str):
        if not hasattr(self, 'status_pill'):
            self.status_pill = QLabel()
            self.status_pill.setObjectName("chat_status_pill")
            self.status_pill.setProperty("status", "muted")
            self.status_pill.style().unpolish(self.status_pill)
            self.status_pill.style().polish(self.status_pill)
            self.status_pill_wrapper = QHBoxLayout()
            self.status_pill_wrapper.addWidget(self.status_pill)
            self.status_pill_wrapper.addStretch()
            # Insert before the stretch at the end
            self.scroll_layout.insertLayout(self.scroll_layout.count() - 1, self.status_pill_wrapper)
            
        self.status_pill.setText(f"<i>[{action_text}]</i>")
        self.status_pill.show()
        self._scroll_to_bottom()

    def clear_status(self):
        if hasattr(self, 'status_pill') and self.status_pill:
            self.status_pill.hide()
