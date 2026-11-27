"""Zelyr Actuator Arbiter.

Deterministic runtime actuator arbiter that factors in active compositor
window state (e.g. Hyprland) and browser extension bridge status before
routing tasks to automation tiers and exposing tool schemas.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

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

# Explicit desktop target tokens preventing ordinary code/HTML UI wording from misrouting to mouse control
_EXPLICIT_DESKTOP_TARGET_PATTERNS = [
    *TIER_3_EXPLICIT_PATTERNS,
    r"\b(capture_screen|screen_capture|screenshot|screen capture|take a screenshot|desktop_control)\b",
    r"\b(mouse_move|mouse_click|double[- ]click|right[- ]click|left[- ]click|mouse_drag|mouse_scroll|keyboard_type|keyboard_press)\b",
    r"\b(desktop\s+(?:app|application|window|fixture|ui|gui)|fixture\s+window|test\s+fixture)\b",
    r"\b(?:desktop|app(?:lication)?|fixture|distraction|confirmation|popup|active|focused|target)\s+window\b",
    r"\b(?:in|on|of|from|to)\s+(?:the\s+|that\s+|this\s+|target\s+|focused\s+|active\s+)?window\b",
    r"\bwindow\s+(?:titled|title|matching)\b",
    r"\bfocus\b[^.\n]{0,40}\bwindow\b",
    r"\bfocus_lost\b",
    r"\b(?:pyside\d?|gtk|qt)\s+(?:window|fixture|gui|app|dialog|widget)\b",
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
    r"\b\w+\.(?:py|tsx?|jsx?|rs|go|java|html|css|scss|json|ya?ml|toml|md|sh|sql)\b",
    r"\b(repository|repo|project|codebase|directory|folder|tests?|pytest|unittest|entrypoint|pyproject|bug|diff|git|src|component|function|class|module|schema|endpoint)\b",
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

    DESKTOP_READ_ONLY_ACTIONS = frozenset({
        "get_screen_size",
        "inspect_windows",
        "get_active_window",
        "get_cursor_position",
        "observe",
        "screenshot",
    })
    WINDOW_READ_ONLY_ACTIONS = frozenset({
        "list",
        "list_windows",
        "clients",
        "get_active",
        "get_active_window",
        "active",
        "activewindow",
        "status",
        "monitors",
        "workspaces",
        "get_geometry",
    })
    WINDOW_CONTROL_TOOLS = frozenset({
        "manage_desktop_window",
        "managedesktopwindowtool",
        "switch_workspace",
        "focus_window",
        "desktop_window",
        "hyprland_ipc",
        "hyprland",
        "window_manager",
    })
    DESKTOP_CONTROL_ALIASES = frozenset({
        "mouse_move",
        "move_mouse",
        "mouse_click",
        "click_mouse",
        "double_click",
        "mouse_drag",
        "mouse_scroll",
        "keyboard_type",
        "type_text",
        "keyboard_press",
        "press_key",
    })
    DESKTOP_INPUT_TOOLS = frozenset({
        "desktop_control",
        "desktop_automation",
        "desktopautomationtool",
        "interact_with_ui",
        "interactwithuitool",
        "gui_actuate",
        "click_tag",
        "som_click",
        "somclicktool",
        "som_type",
        "somtypetool",
        "som_key",
        "somkeytool",
        "press_keys",
        "ydotool_click",
        "ydotool_type",
        "ydotool_key",
        "nxbt_controller",
        *DESKTOP_CONTROL_ALIASES,
    })
    _EXECUTE_ACTION_DESKTOP_MUTATIONS = frozenset({
        "click",
        "double_click",
        "right_click",
        "left_click",
        "middle_click",
        "drag",
        "scroll",
        "type",
        "press",
        "hotkey",
        *DESKTOP_CONTROL_ALIASES,
    })
    _EXECUTE_ACTION_WINDOW_MUTATIONS = frozenset({
        "focus",
        "focus_window",
        "fullscreen",
        "close",
        "close_window",
        "minimize",
        "maximize",
        "tile",
        "float",
        "move",
        "move_window",
        "workspace",
        "switch_workspace",
    })

    @staticmethod
    def extract_current_user_prompt(messages: Optional[List[Dict[str, Any]]]) -> str:
        """Return the latest real user prompt, ignoring synthetic system/vision directives and prior turns."""
        for msg in reversed(messages or ()):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, list):
                text = " ".join(part.get("text", "") for part in content if isinstance(part, dict)).strip()
            elif isinstance(content, str):
                text = content.strip()
            else:
                continue
            if not text or text.startswith(("[SYSTEM", "[Transient desktop screenshot")):
                continue
            return text
        return ""

    @classmethod
    def _normalize_guard_tool_name(cls, tool_name: Any) -> str:
        """Strip tool/namespace/legacy prefixes to resolve the canonical guard key."""
        clean = str(tool_name or "").strip().lower()
        changed = True
        while changed:
            changed = False
            for prefix in ("tool.", "tools.", "functions.", "function."):
                if clean.startswith(prefix):
                    clean = clean.removeprefix(prefix).strip()
                    changed = True
        if clean.startswith("legacy_"):
            stripped = clean.removeprefix("legacy_").strip()
            if stripped in cls.DESKTOP_INPUT_TOOLS or stripped in cls.WINDOW_CONTROL_TOOLS:
                return stripped
        return clean

    @classmethod
    def filter_tool_schemas_for_intent(
        cls,
        schemas: List[Dict[str, Any]],
        user_prompt: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Drop mutating desktop GUI schemas unless user_prompt explicitly requests desktop GUI interaction."""
        prompt_text = user_prompt if isinstance(user_prompt, str) else ""
        if cls.has_explicit_desktop_gui_intent(prompt_text):
            return list(schemas)
        filtered: List[Dict[str, Any]] = []
        for schema in schemas or ():
            if not isinstance(schema, dict):
                continue
            fn_obj = schema.get("function") if isinstance(schema.get("function"), dict) else {}
            raw_name = fn_obj.get("name") or schema.get("name") or ""
            norm_name = cls._normalize_guard_tool_name(raw_name)
            if norm_name in cls.DESKTOP_INPUT_TOOLS:
                continue
            filtered.append(schema)
        return filtered

    @classmethod
    def check_desktop_intent_guard(
        cls,
        tool_name: str,
        tool_args: Optional[Dict[str, Any]],
        *,
        user_prompt: Optional[str],
        acceptance_spec: Any = None,
    ) -> Optional[Tuple[str, str]]:
        """Block unrequested desktop pointer/keyboard input or window/workspace mutations across all dispatch paths."""
        from zelyr.tool_registry import canonicalize_tool_invocation

        canon_name, canon_args = canonicalize_tool_invocation(tool_name, tool_args)
        fn_low = cls._normalize_guard_tool_name(canon_name)
        act_low = str((canon_args or {}).get("action") or "").strip().lower()

        if fn_low == "execute_action":
            is_mut_dc = act_low in cls._EXECUTE_ACTION_DESKTOP_MUTATIONS
            is_mut_win = act_low in cls._EXECUTE_ACTION_WINDOW_MUTATIONS
        else:
            is_mut_dc = fn_low in cls.DESKTOP_INPUT_TOOLS and not (
                fn_low in ("desktop_control", "desktop_automation", "desktopautomationtool")
                and act_low in cls.DESKTOP_READ_ONLY_ACTIONS
            )
            is_mut_win = (
                fn_low in cls.WINDOW_CONTROL_TOOLS
                and act_low not in cls.WINDOW_READ_ONLY_ACTIONS
            )
        if not (is_mut_dc or is_mut_win):
            return None

        from zelyr.engine.execution_ledger import KNOWN_DESKTOP_CHECKS, parse_acceptance_criteria

        prompt_text = user_prompt if isinstance(user_prompt, str) else ""
        has_desktop_criteria = any(
            str(c.get("check") or "").strip().lower() in KNOWN_DESKTOP_CHECKS
            for c in (parse_acceptance_criteria(acceptance_spec) if acceptance_spec else ())
        )
        gui_allowed = (
            cls.has_explicit_desktop_gui_intent(prompt_text)
            or has_desktop_criteria
        )
        win_allowed = (
            gui_allowed
            or cls.has_explicit_window_control_intent(prompt_text)
        )
        if is_mut_dc and not gui_allowed:
            return (
                "unrequested_desktop_control",
                "Refused unrequested desktop pointer/keyboard action during a non-desktop task.",
            )
        if is_mut_win and not win_allowed:
            return (
                "unrequested_window_focus",
                "Refused unrequested window focus or workspace switch during a non-desktop task.",
            )
        return None

    CROSS_SURFACE_DESKTOP_PATTERNS = [
        *DESKTOP_GUI_PATTERNS[:-1],
        r"\b(?:focus|inspect|observe|click|type|press|drag|scroll)\b[^.\n]{0,60}\b(?:fixture|gui)\b",
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
        is_tier1 = any(re.search(p, task_lower) for p in TIER_1_PATTERNS)
        is_explicit_target = any(re.search(p, task_lower) for p in _EXPLICIT_DESKTOP_TARGET_PATTERNS)
        if is_tier1 and not (is_explicit_target or re.search(r"\bdesktop\b", task_lower)):
            return False
        return any(re.search(p, task_lower) for p in cls.CROSS_SURFACE_DESKTOP_PATTERNS)

    @classmethod
    def has_explicit_desktop_gui_intent(cls, task: str) -> bool:
        """Return True only when the task explicitly requests desktop GUI observation or input actuation."""
        task_clean = re.sub(r"\[Active Window:[^\]]*\]\s*", "", task or "")
        task_lower = task_clean.lower().strip()
        if not task_lower:
            return False
        if cls.has_cross_surface_browser_and_desktop_intent(task_clean):
            return True
        is_tier1 = any(re.search(p, task_lower) for p in TIER_1_PATTERNS)
        is_tier2_explicit = any(re.search(p, task_lower) for p in TIER_2_PATTERNS)
        is_explicit_target = any(re.search(p, task_lower) for p in _EXPLICIT_DESKTOP_TARGET_PATTERNS)
        is_desktop_gui = any(re.search(p, task_lower) for p in DESKTOP_GUI_PATTERNS)
        is_ui_interaction = any(re.search(p, task_lower) for p in UI_INTERACTION_PATTERNS)
        if is_tier1 and not is_explicit_target:
            return False
        if is_tier2_explicit and not is_explicit_target:
            return False
        return bool(is_explicit_target or is_desktop_gui or (is_ui_interaction and not is_tier1 and not is_tier2_explicit))

    @classmethod
    def has_explicit_window_control_intent(cls, task: str) -> bool:
        """Return True when task explicitly requests compositor window/workspace control or desktop GUI focus."""
        task_clean = re.sub(r"\[Active Window:[^\]]*\]\s*", "", task or "")
        task_lower = task_clean.lower().strip()
        if not task_lower:
            return False
        if cls.has_explicit_desktop_gui_intent(task_clean):
            return True
        return bool(
            any(re.search(p, task_lower) for p in _PURE_WINDOW_IPC_PATTERNS)
            or re.search(r"\b(focus|switch\s+to|move\s+to|close|minimize|maximize|tile|float|fullscreen)\b[^.\n]{0,40}\b(window|workspace|display|monitor|hyprland|0x[0-9a-f]+)\b", task_lower)
        )

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
        is_explicit_desktop_target = any(re.search(p, task_lower) for p in _EXPLICIT_DESKTOP_TARGET_PATTERNS)
        is_desktop_gui = any(re.search(p, task_lower) for p in DESKTOP_GUI_PATTERNS)
        is_ui_interaction = any(re.search(p, task_lower) for p in UI_INTERACTION_PATTERNS)
        is_pure_window_ipc = any(re.search(p, task_lower) for p in _PURE_WINDOW_IPC_PATTERNS)

        has_desktop_gui_intent = is_tier3_explicit or (
            is_desktop_gui and (is_explicit_desktop_target or (not is_tier1 and not is_tier2_explicit))
        ) or (
            is_ui_interaction and not is_tier2_explicit and not is_tier1
        )

        # Prioritized Tier 1 OS/shell/repo/workspace operations (unless browser or explicit desktop GUI input/observation requested)
        if is_tier1 and not is_tier2_explicit and (is_pure_window_ipc or not has_desktop_gui_intent):
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
