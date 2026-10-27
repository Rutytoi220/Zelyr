"""Hybrid "Fastest Path" Tool Router for Zelyr.

Prioritizes native Linux shell commands, system APIs, and CLI utilities
(e.g., xdg-open, systemctl, curl, wpctl) over GUI mouse clicks and keyboard typing.
Physical motor cortex actions (ydotool) are preserved strictly as the fallback
for visually locked apps or canvas-based GUI elements without CLI equivalents.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class FastPathPlan:
    """Represents an intercepted action rerouted to native CLI execution."""

    original_action: str
    command: List[str]
    description: str
    confidence: float
    category: str  # 'browser', 'app', 'file', 'service', 'audio', 'network'


class FastestPathRouter:
    """Evaluates proposed actions and determines if a faster, deterministic CLI path exists."""

    URL_PATTERN = re.compile(
        r"(https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9-]+\.(?:com|org|net|io|edu|gov|dev|app|ai)[^\s]*)",
        re.IGNORECASE,
    )

    KNOWN_APP_BINARIES = {
        "firefox": "firefox",
        "zen": "zen-browser",
        "zen browser": "zen-browser",
        "zen-browser": "zen-browser",
        "chrome": "google-chrome",
        "google chrome": "google-chrome",
        "chromium": "chromium",
        "terminal": "kitty",
        "kitty": "kitty",
        "alacritty": "alacritty",
        "nautilus": "nautilus",
        "files": "nautilus",
        "calculator": "gnome-calculator",
        "calc": "gnome-calculator",
        "text editor": "gedit",
        "gedit": "gedit",
        "code": "code",
        "vscode": "code",
    }

    @classmethod
    def can_route_to_cli(
        cls, action: str, params: Union[Dict[str, Any], str]
    ) -> Optional[FastPathPlan]:
        """Examines the action and parameters to determine if a CLI shortcut exists.

        Returns a FastPathPlan if a native command is preferred, or None if GUI automation is required.
        """
        action_lower = (action or "").strip().lower()

        # Extract text or url parameters if available
        param_str = ""
        if isinstance(params, dict):
            for k in ("url", "path", "text", "query", "target", "command", "name", "app"):
                if k in params and params[k]:
                    param_str = str(params[k]).strip()
                    break
            if not param_str:
                param_str = " ".join(f"{k}={v}" for k, v in params.items())
        elif isinstance(params, str):
            param_str = params.strip()

        combined_text = f"{action_lower} {param_str}".lower()

        # ── 1. Web URLs & Browser Navigation ──────────────────────────────
        url_match = cls.URL_PATTERN.search(param_str or action)
        if url_match or any(w in combined_text for w in ["open browser", "navigate to", "open url", "browse to"]):
            target_url = url_match.group(1) if url_match else "https://google.com"
            if not target_url.startswith(("http://", "https://")):
                target_url = "https://" + target_url

            xdg_bin = shutil.which("xdg-open")
            if xdg_bin:
                return FastPathPlan(
                    original_action=action,
                    command=["xdg-open", target_url],
                    description=f"Directly launching browser with URL '{target_url}' via xdg-open",
                    confidence=0.98,
                    category="browser",
                )

        # ── 2. Application Launching ──────────────────────────────────────
        if any(w in action_lower for w in ["open_app", "launch_app", "start_app", "open_application"]) or (
            action_lower in ["open", "launch", "run"] and not param_str.startswith("/")
        ):
            target_app = (
                params.get("name", params.get("app", param_str))
                if isinstance(params, dict)
                else param_str
            ).strip().lower()

            # Check if app is in known binaries
            bin_name = cls.KNOWN_APP_BINARIES.get(target_app, target_app)
            resolved_bin = shutil.which(bin_name)
            if resolved_bin:
                return FastPathPlan(
                    original_action=action,
                    command=[resolved_bin],
                    description=f"Directly launching executable '{resolved_bin}' via CLI",
                    confidence=0.95,
                    category="app",
                )

            # Try gtk-launch if available
            gtk_launch = shutil.which("gtk-launch")
            if gtk_launch and target_app:
                return FastPathPlan(
                    original_action=action,
                    command=["gtk-launch", target_app],
                    description=f"Directly launching application '{target_app}' via gtk-launch",
                    confidence=0.90,
                    category="app",
                )

        # ── 3. Files & Folders ────────────────────────────────────────────
        if any(w in action_lower for w in ["open_folder", "open_file", "view_file"]) or (
            action_lower == "open" and (param_str.startswith("/") or param_str.startswith("~"))
        ):
            target_path = params.get("path", param_str) if isinstance(params, dict) else param_str
            xdg_bin = shutil.which("xdg-open")
            if xdg_bin and target_path:
                return FastPathPlan(
                    original_action=action,
                    command=["xdg-open", target_path],
                    description=f"Directly opening path '{target_path}' via xdg-open",
                    confidence=0.95,
                    category="file",
                )

        # ── 4. System Services & State ────────────────────────────────────
        service_match = re.search(
            r"(?:start|stop|restart|status|reload)\s+(?:service\s+)?([a-zA-Z0-9_\-]+)",
            combined_text,
        )
        if service_match and shutil.which("systemctl"):
            verb = re.search(r"(start|stop|restart|status|reload)", combined_text).group(1)
            srv_name = service_match.group(1)
            return FastPathPlan(
                original_action=action,
                command=["systemctl", verb, srv_name],
                description=f"Managing system service '{srv_name}' with systemctl {verb}",
                confidence=0.99,
                category="service",
            )

        # ── 5. Audio / Volume Control ─────────────────────────────────────
        if any(w in combined_text for w in ["volume", "mute", "unmute", "sound"]):
            wpctl = shutil.which("wpctl")
            if wpctl:
                if "mute" in combined_text and "unmute" not in combined_text:
                    cmd = ["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "1"]
                elif "unmute" in combined_text:
                    cmd = ["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "0"]
                elif "up" in combined_text or "increase" in combined_text:
                    cmd = ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", "5%+"]
                elif "down" in combined_text or "decrease" in combined_text:
                    cmd = ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", "5%-"]
                else:
                    cmd = ["wpctl", "status"]

                return FastPathPlan(
                    original_action=action,
                    command=cmd,
                    description=f"Adjusting audio via native pipewire CLI ({' '.join(cmd)})",
                    confidence=0.95,
                    category="audio",
                )

        # ── 6. Fallback: No deterministic CLI shortcut found ─────────────
        return None

    @classmethod
    def execute_fast_path(cls, plan: FastPathPlan, timeout: float = 10.0) -> Tuple[bool, str]:
        """Executes the native CLI command specified in the plan."""
        logger.info(
            "[FASTEST PATH ROUTER] Intercepted '%s' -> Executing native CLI: %s",
            plan.original_action,
            " ".join(plan.command),
        )
        try:
            # For GUI launching tools (xdg-open, apps), run detached/non-blocking
            if plan.category in ("browser", "app", "file"):
                subprocess.Popen(
                    plan.command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                return (
                    True,
                    f"[FASTEST PATH ROUTER] Successfully dispatched native command: {' '.join(plan.command)} "
                    f"({plan.description}). Avoided brittle GUI mouse simulation.",
                )

            # For system/service commands, capture output synchronously
            res = subprocess.run(
                plan.command, capture_output=True, text=True, timeout=timeout
            )
            output = (res.stdout or res.stderr or "").strip()
            if res.returncode == 0:
                return (
                    True,
                    f"[FASTEST PATH ROUTER] Native command succeeded: {' '.join(plan.command)}\n{output}".strip(),
                )
            else:
                return (
                    False,
                    f"[FASTEST PATH ROUTER] Native command returned code {res.returncode}: {output}".strip(),
                )

        except Exception as e:
            logger.warning("[FASTEST PATH ROUTER] Execution error: %s", e)
            return False, f"Failed executing fast path command: {e}"

    @classmethod
    def evaluate_and_intercept(
        cls, tool_name: str, arguments: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Evaluates a tool call and intercepts it if a CLI shortcut exists.

        Used directly by agent loops and orchestrators to convert motor/GUI tool calls
        into instant CLI executions.
        """
        # If the tool is desktop_control or motor tools:
        if tool_name in ("desktop_control", "mouse_click", "keyboard_type", "execute_action"):
            action = arguments.get("action", tool_name)
            # Check if there is an intent to open, navigate, or manage services
            plan = cls.can_route_to_cli(action, arguments)
            if plan:
                ok, msg = cls.execute_fast_path(plan)
                return {
                    "tool": tool_name,
                    "intercepted": True,
                    "fast_path": True,
                    "command": plan.command,
                    "success": ok,
                    "output": msg,
                }

        # Check generic open/navigate tools
        plan = cls.can_route_to_cli(tool_name, arguments)
        if plan:
            ok, msg = cls.execute_fast_path(plan)
            return {
                "tool": tool_name,
                "intercepted": True,
                "fast_path": True,
                "command": plan.command,
                "success": ok,
                "output": msg,
            }

        return None
