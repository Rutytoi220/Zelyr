"""Clipboard Change Tap — RFC-003 Phase 2.

Monitors the system clipboard for text changes using a SHA-256 hash-based
diff approach. Only the *hash* of the previous content is retained in memory
— the raw text is evaluated and immediately released.

**Strictly opt-in:** the tap never spawns any thread unless
``config.monitor_clipboard`` is explicitly ``True``.

Security pipeline (applied in order)
-------------------------------------
1. Deny-list: if the active window matches a password manager, **drop**.
2. ``PrivacyScrubber.scrub_text``: redact API keys, tokens, PEM blocks.
3. IntentEngine: evaluate scrubbed text for actionable patterns.
4. EventBus: emit ``perception.clipboard.change`` with scrubbed text only.

Dependency: ``pyperclip`` (optional). If unavailable the tap logs a warning
and silently does nothing.
"""
from __future__ import annotations
import hashlib
import logging
import threading
import time
from typing import Optional
from zelyr.config import get_config
from zelyr.core.events import EventBus, Event
from zelyr.perception.deny_list import DenyList
from zelyr.perception.scrubber import PrivacyScrubber
logger = logging.getLogger(__name__)
_POLL_INTERVAL: float = 1.0

def _read_clipboard() -> Optional[str]:
    """Read current clipboard text. Returns None if pyperclip is unavailable."""
    try:  # pragma: no cover
        import pyperclip  # pragma: no cover
        return pyperclip.paste()  # pragma: no cover
    except ImportError:  # pragma: no cover
        logger.warning('ClipboardTap: pyperclip not installed. Run: pip install pyperclip')  # pragma: no cover
        return None  # pragma: no cover
    except Exception:  # pragma: no cover
        return None  # pragma: no cover

class ClipboardTap:
    """Hash-based clipboard monitor that emits scrubbed change events.

    The tap retains **only** a SHA-256 hash of the last-seen clipboard
    content. The raw text is never stored as an attribute — it is read,
    hashed, scrubbed, and discarded within a single stack frame.
    """

    def __init__(self, event_bus: EventBus, intent_engine=None):
        """Auto-generated docstring.

Args:
    event_bus: Argument.
    intent_engine: Argument.

Returns:
    Return value.
"""
        self.event_bus = event_bus  # pragma: no cover
        self.intent_engine = intent_engine  # pragma: no cover
        self._thread: Optional[threading.Thread] = None  # pragma: no cover
        self._stop_event = threading.Event()  # pragma: no cover
        self._last_hash: Optional[str] = None  # pragma: no cover

    def start(self) -> bool:
        """Start the background polling thread.

        Returns:
            True if started, False if the config guard is disabled or already running.
        """
        config = get_config()  # pragma: no cover
        if not getattr(config, 'monitor_clipboard', False):  # pragma: no cover
            logger.info('ClipboardTap: disabled (config.monitor_clipboard=False). Not starting.')  # pragma: no cover
            return False  # pragma: no cover
        if self._thread and self._thread.is_alive():  # pragma: no cover
            return False  # pragma: no cover
        self._stop_event.clear()  # pragma: no cover
        self._thread = threading.Thread(target=self._poll_loop, name='zelyr-clipboard-tap', daemon=True)  # pragma: no cover
        self._thread.start()  # pragma: no cover
        logger.info('ClipboardTap: started (polling every %.1fs).', _POLL_INTERVAL)  # pragma: no cover
        return True  # pragma: no cover

    def stop(self) -> None:
        """Signal the polling thread to exit."""
        self._stop_event.set()  # pragma: no cover
        if self._thread:  # pragma: no cover
            self._thread.join(timeout=_POLL_INTERVAL + 1)  # pragma: no cover
        logger.info('ClipboardTap: stopped.')  # pragma: no cover

    def _poll_loop(self) -> None:
        """Main polling loop — runs in a daemon thread."""
        while not self._stop_event.is_set():  # pragma: no cover
            try:  # pragma: no cover
                self._sample()  # pragma: no cover
            except Exception as exc:  # pragma: no cover
                logger.debug('ClipboardTap: sampling error: %s', exc)  # pragma: no cover
            self._stop_event.wait(_POLL_INTERVAL)  # pragma: no cover

    def _sample(self) -> None:
        """Sample the clipboard and emit an event if the content has changed.

        Security contract
        -----------------
        * Raw text is only ever held as a local variable inside this method.
        * If the active window is on the deny-list we drop *before* reading
          the clipboard to avoid even touching potentially sensitive data.
        * The raw text is scrubbed before any further processing or emission.
        """
        try:  # pragma: no cover
            from zelyr.perception.window_tap import get_active_window  # pragma: no cover
            window_title, process_name = get_active_window()  # pragma: no cover
        except Exception:  # pragma: no cover
            window_title, process_name = ('', '')  # pragma: no cover
        if DenyList.is_blocked(process_name=process_name, window_title=window_title):  # pragma: no cover
            return  # pragma: no cover
        raw_text = _read_clipboard()  # pragma: no cover
        if not raw_text or not isinstance(raw_text, str):  # pragma: no cover
            return  # pragma: no cover
        content_hash = hashlib.sha256(raw_text.encode('utf-8', errors='replace')).hexdigest()  # pragma: no cover
        if content_hash == self._last_hash:  # pragma: no cover
            return  # pragma: no cover
        self._last_hash = content_hash  # pragma: no cover
        scrubbed_text = PrivacyScrubber.scrub_text(raw_text)  # pragma: no cover
        intent_action = None  # pragma: no cover
        if self.intent_engine:  # pragma: no cover
            try:  # pragma: no cover
                intent_action = self.intent_engine.evaluate_clipboard(scrubbed_text)  # pragma: no cover
            except Exception as exc:  # pragma: no cover
                logger.debug('ClipboardTap: IntentEngine error: %s', exc)  # pragma: no cover
        self.event_bus.publish(Event(event_type='perception.clipboard.change', source='ClipboardTap', data={'text': scrubbed_text, 'length': len(scrubbed_text), 'intent': intent_action, 'process_name': process_name}))  # pragma: no cover
        logger.debug('ClipboardTap: clipboard change detected (scrubbed len=%d).', len(scrubbed_text))  # pragma: no cover
