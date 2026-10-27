"""Zelyr Core — Centralized Keyboard Shortcut Registry.

Every bindable action in Zelyr is defined here.  This is the single source
of truth read by both the Qt GUI layer (local shortcuts) and the background
pynput listener (global hotkeys).

Schema
------
Each entry in SHORTCUTS has the shape::

    {
        "name":    str,          # Human-readable label shown in Settings
        "default": str | None,   # Key string; None = unbound by default
        "global":  bool,         # True = pynput GlobalHotKeys (works when app hidden)
                                 # False = QShortcut (works only when window focused)
        "description": str,      # One-liner for settings UI tooltips
    }

Key String Formats
------------------
Local (Qt / QKeySequence):
    "Ctrl+N", "Ctrl+Shift+N", "Ctrl+K", …

Global (pynput GlobalHotKeys notation):
    "<ctrl>+<alt>+<space>", "<cmd>+<shift>+v", …
    Note: <cmd> = the Meta/Windows/Command key.

Adding a new action
-------------------
1. Add it here with a unique snake_case action_id.
2. If ``global=False``, handle it in ``_register_local_shortcuts`` inside
   ``zelyr/gui/main_window.py``.
3. If ``global=True``, no extra code is needed — ``hotkey_service.py``
   picks it up automatically.
"""

from __future__ import annotations

from typing import Dict, Any

# ── Registry ──────────────────────────────────────────────────────────────────

SHORTCUTS: Dict[str, Dict[str, Any]] = {
    # ── Local (Qt) ────────────────────────────────────────────────────────────
    "new_chat": {
        "name": "New Chat",
        "default": "Ctrl+N",
        "global": False,
        "description": "Start a fresh conversation in the current project.",
    },
    "new_project": {
        "name": "New Project",
        "default": "Ctrl+Shift+N",
        "global": False,
        "description": "Create a new project workspace.",
    },
    "focus_input": {
        "name": "Focus Input",
        "default": "Ctrl+K",
        "global": False,
        "description": "Jump focus to the chat input bar.",
    },
    "toggle_sidebar": {
        "name": "Toggle Sidebar",
        "default": "Ctrl+B",
        "global": False,
        "description": "Show or hide the left conversation sidebar.",
    },
    "clear_chat": {
        "name": "Clear Chat",
        "default": "Ctrl+L",
        "global": False,
        "description": "Clear all messages in the current chat view.",
    },
    "open_settings": {
        "name": "Open Settings",
        "default": None,
        "global": False,
        "description": "Open the settings drawer (unbound by default).",
    },
    "search_history": {
        "name": "Search History",
        "default": None,
        "global": False,
        "description": "Search through conversation history (unbound by default).",
    },
    "prev_chat": {
        "name": "Previous Chat",
        "default": "Alt+Up",
        "global": False,
        "description": "Select the previous chat in the sidebar.",
    },
    "next_chat": {
        "name": "Next Chat",
        "default": "Alt+Down",
        "global": False,
        "description": "Select the next chat in the sidebar.",
    },

    # ── Global (pynput — works even when Zelyr is hidden) ─────────────────────
    "toggle_zelyr": {
        "name": "Toggle Zelyr",
        "default": "<ctrl>+<alt>+<space>",
        "global": True,
        "description": "Show or hide the Zelyr window from anywhere.",
    },
    "start_audio": {
        "name": "Push-to-Talk",
        "default": "<cmd>+<shift>+v",
        "global": True,
        "description": "Activate the STT voice input from anywhere.",
    },
    "capture_screen": {
        "name": "Capture Screen",
        "default": "<cmd>+<shift>+s",
        "global": True,
        "description": "Capture the screen and open SoM vision overlay.",
    },
}
