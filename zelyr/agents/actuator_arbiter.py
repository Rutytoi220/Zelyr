"""Zelyr Actuator Arbiter.

Deterministic runtime actuator arbiter that factors in active compositor
window state (e.g. Hyprland) and browser extension bridge status before
routing tasks to automation tiers and exposing tool schemas.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Browser window class / title patterns
BROWSER_WINDOW_PATTERNS = [
    r"\bzen\b",
    r"\bzen-browser\b",
    r"\bzen_browser\b",
    r"\bfirefox\b",
    r"\bchrome\b",
    r"\bchromium\b",
    r"\bbrave\b",
    r"\bbrave-browser\b",
    r"\bvivaldi\b",
    r"\bopera\b",
    r"\bedge\b",
    r"\bmsedge\b",
    r"\blibrewolf\b",
    r"\belectron\b",
]

# Terminal and developer environment window class patterns
TERMINAL_IDE_PATTERNS = [
    r"\bkitty\b",
    r"\balacritty\b",
    r"\bfoot\b",
    r"\bwezterm\b",
    r"\bgnome-terminal\b",
    r"\bxterm\b",
    r"\bkonsole\b",
    r"\btmux\b",
    r"\bcode\b",
    r"\bcursor\b",
    r"\bneovim\b",
    r"\bnvim\b",
    r"\bemacs\b",
    r"\bvim\b",
    r"\bterminator\b",
    r"\burxvt\b",
]

# Explicit Tier 3 visual/canvas/grounding indicators
TIER_3_EXPLICIT_PATTERNS = [
    r"\b(no dom|without dom|non-accessible|canvas|opengl|vulkan|game|spaceship|pixel|coordinates|legacy binary)\b",
    r"\b(grounding|vision engine|screenshot|look at screen|on my display|on screen|visual grounding)\b",
]

# Desktop GUI automation / observation indicators (capture_screen, desktop_control, pointer/keyboard actions)
DESKTOP_GUI_PATTERNS = [
    r"\b(capture_screen|screen_capture|screenshot|screen capture|take a screenshot)\b",
    r"\b(desktop_control|move_mouse|mouse_move|mouse_click|double[- ]click|right[- ]click|left[- ]click|middle[- ]click|mouse_drag|drag and drop|mouse_scroll|mouse cursor|pointer)\b",
    r"\b(keyboard_type|type_text|keyboard_press|press_key|hotkey|key combo|press enter|press return|press tab)\b",
    r"\b(visible state|window layout|ui controls?|what appears in the .* window)\b",
    r"\b(?:focus|inspect|observe|click|type|press|drag|scroll)\b[^.\n]{0,60}\b(?:fixture|button|textbox|text box|text field|input field|checkbox|dropdown|dialog|popup|label|gui)\b",
]

# Pure OS / compositor window IPC patterns (do not require GUI pointer/keyboard/screenshot tools)
_PURE_WINDOW_IPC_PATTERNS = [
    r"\b(switch to|move to|send to)\s+(workspace|desktop)\b",
    r"\b(close|minimize|maximize|tile|float)\s+(the\s+)?window\b",
    r"\b(focus|activate)\s+(the\s+)?(terminal|editor|browser|kitty|alacritty|vesktop|window)\s*$",
    r"\b(list windows|active window|fullscreen|toggle floating|hyprctl|hyprland)\b",
]

# Tier 1 Desktop/OS indicators
TIER_1_PATTERNS = [
    r"\b(workspace|workspaces|fullscreen|floating|tile|tiling)\b",
    r"\b(focus|switch to|move to|close)\s+(window|workspace)\b",
    r"\b(active window|list windows|hyprctl|hyprland)\b",
    r"\b(process|processes|pid|kill\s+process|kill\s+pid|journal|journalctl|systemd|clipboard|wl-copy|wl-paste)\b",
    r"\b(notify|notification|notify-send|alert)\b",
    r"\b(network|gateway|ip\s+address|interfaces|tailscale|dns|ping|vpn|wifi)\b",
    r"\b(media|music|playerctl|playback|volume|track|song|mpris)\b",
    r"\b(file|files|patch|rollback|undo|backup)\b",
    r"\b(repository|repo|project|codebase|directory|folder|tests?|pytest|unittest|entrypoint|pyproject|bug|diff|git)\b",
    r"\b(command|bash|shell|terminal|exec|run command|cli|uptime|hostname|uname|kernel|distro|cpu|memory|disk|df|lsblk)\b",
]

# Tier 2 Browser / Web indicators
TIER_2_PATTERNS = [
    r"\b(browser|chrome|chromium|brave|zen|firefox|monkeytype|gemini|youtube|electron|google|bing|duckduckgo)\b",
    r"\b(dom|html|css selector|website|webpage|web page|web app|tab|url|href)\b",
    r"\b(evaluate_script|javascript|js)\b",
    r"\b(?:inspect|evaluate)\s+(?:the\s+|this\s+|current\s+)?(?:dom|element|elements|page|webpage|web\s+page|tab|tabs|selector|css|html|script)\b",
    r"\b(?:inspect|evaluate)\s+(?:in\s+)?(?:browser|chrome|chromium|firefox|zen|brave)\b",
    r"\b(scroll|scroll_page|extract_page_content|extract\s+page|page\s+content|markdown|read\s+article|pin\s+tab|duplicate\s+tab|reload\s+tab)\b",
    r"\b(open_tab|switch_tab|navigate_url|click_element|fill_element|get_page_snapshot)\b",
    r"\b(web_search|fetch_web_page|web search|search the web|search web)\b",
]

# Generic UI interaction indicators (click, type, fill, submit...)
UI_INTERACTION_PATTERNS = [
    r"\b(click|press|tap|double-click|right-click)\b",
    r"\b(type|fill|input|enter|submit|select)\b",
    r"\b(button|link|input|textbox|checkbox|dropdown|form)\b",
]


def is_browser_window(active_window: Optional[Dict[str, Any] | str]) -> bool:
    """Checks whether the given window context corresponds to a web browser."""
    if not active_window:
        return False
    if isinstance(active_window, str):
        target_str = active_window.lower()
    elif isinstance(active_window, dict):
        w_class = str(active_window.get("class") or active_window.get("initialClass") or "").lower()
        w_title = str(active_window.get("title") or active_window.get("initialTitle") or "").lower()
        target_str = f"{w_class} {w_title}"
    else:
        return False

    return any(re.search(pat, target_str) for pat in BROWSER_WINDOW_PATTERNS)


def is_terminal_or_ide_window(active_window: Optional[Dict[str, Any] | str]) -> bool:
    """Checks whether the given window context corresponds to a terminal emulator or IDE."""
    if not active_window:
        return False
    if isinstance(active_window, str):
        target_str = active_window.lower()
    elif isinstance(active_window, dict):
        w_class = str(active_window.get("class") or active_window.get("initialClass") or "").lower()
        w_title = str(active_window.get("title") or active_window.get("initialTitle") or "").lower()
        target_str = f"{w_class} {w_title}"
    else:
        return False

    return any(re.search(pat, target_str) for pat in TERMINAL_IDE_PATTERNS)


class ActuatorArbiter:
    """Deterministic runtime arbitrator between Tier 1 (IPC/OS), Tier 2 (Browser DOM),
    and Tier 3 (Vision Grounding).
    """

    TIER_TOOLS = {
        "tier1_ipc": ["manage_desktop_window", "manage_system_process", "execute_command"],
        "tier2_browser": ["interact_with_browser", "web_search", "fetch_web_page"],
        "tier3_vision": ["desktop_control", "capture_screen", "interact_with_ui"],
    }

    CROSS_SURFACE_DESKTOP_PATTERNS = [
        *DESKTOP_GUI_PATTERNS,
        *TIER_3_EXPLICIT_PATTERNS,
        r"\b(desktop\s+(?:app|application|window|fixture|ui|gui)|pyside\d?|gtk|qt)\b",
        r"\b(desktop)\b[^.\n]{0,80}\b(window|button|field|input|confirm|submit|label|status|screen)\b",
    ]

    @classmethod
    def has_cross_surface_browser_and_desktop_intent(cls, task: str) -> bool:
        """Return True when task explicitly spans both browser automation and desktop GUI surfaces."""
        task_clean = re.sub(r"\[Active Window:[^\]]*\]\s*", "", task or "")
        task_lower = task_clean.lower().strip()
        if not task_lower:
            return False
        has_browser = (
            any(re.search(p, task_lower) for p in TIER_2_PATTERNS)
            or ("http://" in task_lower)
            or ("https://" in task_lower)
        )
        if not has_browser:
            return False
        return any(re.search(p, task_lower) for p in cls.CROSS_SURFACE_DESKTOP_PATTERNS)

    @classmethod
    def arbitrate(
        cls,
        task: str,
        active_window: Optional[Dict[str, Any]] = None,
        bridge_connected: bool = False,
    ) -> str:
        """Determines the appropriate automation tier based on task description,
        active compositor window metadata, and browser extension bridge connectivity.

        Parameters
        ----------
        task : str
            The user task or query description.
        active_window : Optional[Dict[str, Any]], default None
            Metadata of the active window (e.g. from hyprctl activewindow -j).
        bridge_connected : bool, default False
            Whether the browser extension bridge has an active connected client.

        Returns
        -------
        str
            Tier identifier: 'tier1_ipc', 'tier2_browser', or 'tier3_vision'.
        """
        # Strip injected telemetry header if present so window metadata doesn't pollute task intent
        task_clean = re.sub(r"\[Active Window:[^\]]*\]\s*", "", task or "")
        task_lower = task_clean.lower().strip()
        browser_active = is_browser_window(active_window)
        has_functioning_dom_bridge = browser_active and bridge_connected

        # 1. Evaluate tier indicators cleanly before committing to Tier 1 vs Tier 3
        is_tier1 = any(re.search(p, task_lower) for p in TIER_1_PATTERNS)
        is_tier2_explicit = any(re.search(p, task_lower) for p in TIER_2_PATTERNS)
        is_tier3_explicit = any(re.search(p, task_lower) for p in TIER_3_EXPLICIT_PATTERNS)
        is_desktop_gui = any(re.search(p, task_lower) for p in DESKTOP_GUI_PATTERNS)
        is_ui_interaction = any(re.search(p, task_lower) for p in UI_INTERACTION_PATTERNS)
        is_pure_window_ipc = any(re.search(p, task_lower) for p in _PURE_WINDOW_IPC_PATTERNS)

        has_desktop_gui_intent = is_tier3_explicit or is_desktop_gui or (
            is_ui_interaction and not is_tier2_explicit and not is_tier1
        )

        # Prioritized Tier 1 OS/shell/repo/workspace operations (unless browser or desktop GUI input/observation requested)
        if is_tier1 and not is_tier2_explicit and (is_pure_window_ipc or not (is_tier3_explicit or is_desktop_gui)):
            return "tier1_ipc"

        # 2. Explicit visual/screen/desktop GUI request:
        # Must route to browser when browser + DOM bridge is active (negative gating).
        if has_desktop_gui_intent:
            if has_functioning_dom_bridge:
                # Browser tab with functioning DOM bridge: strictly route to Tier 2, never Tier 3
                return "tier2_browser"
            return "tier3_vision"

        # 3. Explicit Tier 2 browser keywords:
        if is_tier2_explicit:
            return "tier2_browser"

        # 4. Context-aware browser interaction:
        # If browser window is active AND extension bridge is connected:
        # All UI interaction tasks (click button, submit form, type text...) route to Tier 2
        if has_functioning_dom_bridge:
            if is_ui_interaction or browser_active:
                return "tier2_browser"

        # 5. Terminal / IDE window active:
        # Route general tasks and developer workflows to Tier 1
        if is_terminal_or_ide_window(active_window):
            return "tier1_ipc"

        # 6. Fallback when no active window or generic task:
        # Without explicit desktop GUI/visual grounding or active browser, route to Tier 1
        return "tier1_ipc"


def arbitrate(
    task: str,
    active_window: Optional[Dict[str, Any]] = None,
    bridge_connected: bool = False,
) -> str:
    """Convenience module-level function for ActuatorArbiter.arbitrate."""
    return ActuatorArbiter.arbitrate(task, active_window, bridge_connected)
