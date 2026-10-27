"""Zelyr Services — Global Hotkey Listener.

Reads every action marked ``global=True`` from the central shortcut registry
(``zelyr.core.shortcuts.SHORTCUTS``) and registers them with pynput's
``GlobalHotKeys``.  Emits a corresponding Qt Signal on activation so the main
Qt thread can react safely.

Adding a new global hotkey
--------------------------
1. Add the action to ``SHORTCUTS`` in ``zelyr/core/shortcuts.py`` with
   ``"global": True`` and a pynput-format key string as ``"default"``.
2. Add a matching ``Signal()`` to ``HotkeySignaler`` below.
3. Add a line to ``_ACTION_SIGNAL_MAP`` mapping the action_id to that signal.
That's it — no further changes needed here.
"""

import logging
import threading
from typing import Dict, Optional

from PySide6.QtCore import QObject, Signal

try:
    from pynput import keyboard
    _PYNPUT_AVAILABLE = True
except ImportError:
    _PYNPUT_AVAILABLE = False

from zelyr.core.shortcuts import SHORTCUTS

logger = logging.getLogger(__name__)


class HotkeySignaler(QObject):
    """QObject that bridges pynput thread events into the Qt main thread safely.

    One Signal per global action.  Add a new Signal here whenever you add a
    new ``global=True`` entry to the shortcut registry.
    """
    toggle_requested = Signal()    # toggle_zelyr
    context_summoned = Signal()    # (legacy — kept for backwards compat)
    vision_summoned  = Signal()    # (legacy — kept for backwards compat)
    start_audio      = Signal()    # start_audio
    capture_screen   = Signal()    # capture_screen


# Maps action_id (from SHORTCUTS) → the HotkeySignaler method name to emit.
_ACTION_SIGNAL_MAP: Dict[str, str] = {
    "toggle_zelyr":   "toggle_requested",
    "start_audio":    "start_audio",
    "capture_screen": "capture_screen",
}


class GlobalHotkeyService:
    """Dynamically registers all global hotkeys from the shortcut registry."""

    def __init__(self) -> None:
        self.signaler = HotkeySignaler()
        self._listener: Optional[object] = None
        self._running = False

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the global hotkey listener in a daemon background thread."""
        if self._running:
            return
        if not _PYNPUT_AVAILABLE:
            logger.error("pynput not installed — global hotkeys disabled.")
            return

        self._running = True

        # Build the pynput hotkey map dynamically from the registry.
        hotkey_map: Dict[str, object] = {}
        
        def _normalize_pynput_hotkey(kstr: str) -> str:
            parts = kstr.lower().split("+")
            normalized = []
            for p in parts:
                p = p.strip()
                if p in ("super", "win", "windows", "meta"):
                    normalized.append("<cmd>")
                elif p in ("ctrl", "control"):
                    normalized.append("<ctrl>")
                elif p in ("alt", "option"):
                    normalized.append("<alt>")
                elif p == "shift":
                    normalized.append("<shift>")
                elif p == "space":
                    normalized.append("<space>")
                elif p == "enter":
                    normalized.append("<enter>")
                elif p == "tab":
                    normalized.append("<tab>")
                elif p == "esc" or p == "escape":
                    normalized.append("<esc>")
                else:
                    normalized.append(p)
            return "+".join(normalized)

        for action_id, spec in SHORTCUTS.items():
            if not spec.get("global"):
                continue
            key_str: Optional[str] = spec.get("default")
            if not key_str:
                continue
                
            pynput_key = _normalize_pynput_hotkey(key_str)
            
            signal_name = _ACTION_SIGNAL_MAP.get(action_id)
            if signal_name is None:
                logger.warning(
                    "[HotkeyService] No signal mapped for global action '%s'. "
                    "Add it to _ACTION_SIGNAL_MAP in hotkey_service.py.",
                    action_id,
                )
                continue

            # Capture by value in the closure.
            def _make_handler(sig_name: str, act_id: str):
                def _handler():
                    logger.info("[HotkeyService] Global hotkey fired: %s", act_id)
                    getattr(self.signaler, sig_name).emit()
                return _handler

            hotkey_map[pynput_key] = _make_handler(signal_name, action_id)
            logger.debug(
                "[HotkeyService] Registered global hotkey: %s (%s) -> %s",
                key_str, pynput_key, action_id,
            )

        if not hotkey_map:
            logger.warning("[HotkeyService] No global hotkeys to register.")
            return

        def _run_listener():
            logger.info(
                "[HotkeyService] Starting pynput listener for %d hotkeys.",
                len(hotkey_map),
            )
            try:
                # Wrap initialization to gracefully drop malformed keys without crashing the daemon loop
                try:
                    self._listener = keyboard.GlobalHotKeys(hotkey_map)
                except ValueError as ve:
                    logger.error(f"[HotkeyService] Failed to parse hotkey syntax: {ve}. Please check SHORTCUTS configuration.")
                    return
                except Exception as ex:
                    logger.error(f"[HotkeyService] Could not hook into global keyboard events: {ex}")
                    return
                    
                self._listener.start()
                self._listener.join()
            except Exception as exc:
                logger.error("[HotkeyService] Listener error: %s", exc)

        thread = threading.Thread(target=_run_listener, daemon=True)
        thread.start()

    def stop(self) -> None:
        """Stop the pynput listener cleanly."""
        self._running = False
        if self._listener:
            try:
                self._listener.stop()
            except Exception as exc:
                logger.error("[HotkeyService] Error stopping listener: %s", exc)
            self._listener = None
        logger.info("[HotkeyService] Global hotkey listener stopped.")
