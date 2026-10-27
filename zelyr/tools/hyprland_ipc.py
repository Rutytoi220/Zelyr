"""Hyprland Native System IPC Tool for Zelyr.

Provides deterministic Tier 1 window and workspace management via hyprctl.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from typing import Any, Dict, Optional

from zelyr.tools.core import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


class ManageDesktopWindowTool(BaseTool):
    """Deterministically manages Hyprland windows and workspaces via hyprctl."""

    def __init__(self):
        super().__init__(
            tool_id="manage_desktop_window",
            name="manage_desktop_window",
            description=(
                "Tier 1 Hyprland System IPC: Deterministically manage desktop windows, workspaces, "
                "and focus via hyprctl. Actions: 'list', 'get_active', 'focus', 'workspace', "
                "'move_to_workspace', 'toggle_floating', 'toggle_fullscreen', 'close'."
            ),
            tier=1,
            is_core=True,
            requires_bridge=False,
            requires_window=True,
            token_cost=150,
        )
        self.parameters = [
            ToolParameter(
                name="action",
                type="string",
                description="The window management action to perform: 'list', 'get_active', 'focus', 'workspace', 'move_to_workspace', 'toggle_floating', 'toggle_fullscreen', 'close'.",
                required=True,
            ),
            ToolParameter(
                name="target",
                type="string",
                description="Target window title/class, window address (e.g., '0x55a8e8690210'), or workspace ID/name.",
                required=False,
                default="",
            ),
        ]

    async def _run_hyprctl(self, command: str) -> tuple[int, str, str]:
        if not shutil.which("hyprctl"):
            return -1, "", "hyprctl executable not found on PATH."

        proc = await asyncio.create_subprocess_shell(
            f"hyprctl {command}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return proc.returncode or 0, stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        action = params.get("action", "").strip().lower()
        target = str(params.get("target", "") or "").strip()

        if not shutil.which("hyprctl"):
            return ToolResult(
                False,
                error="hyprctl binary not found. This tool requires the Hyprland Wayland compositor.",
            )

        try:
            if action in ("list", "list_windows", "clients"):
                code, out, err = await self._run_hyprctl("clients -j")
                if code != 0:
                    return ToolResult(False, error=f"hyprctl clients failed: {err}")
                try:
                    clients = json.loads(out)
                    # Filter and summarize clients for clean LLM consumption
                    summary = []
                    for c in clients:
                        summary.append({
                            "address": c.get("address"),
                            "class": c.get("class"),
                            "title": c.get("title"),
                            "workspace": c.get("workspace", {}).get("name"),
                            "at": c.get("at"),
                            "size": c.get("size"),
                            "floating": c.get("floating"),
                            "fullscreen": bool(c.get("fullscreen")),
                        })
                    return ToolResult(
                        True,
                        output={
                            "count": len(summary),
                            "windows": summary,
                        },
                    )
                except json.JSONDecodeError:
                    return ToolResult(True, output=out)

            elif action in ("get_active", "active", "activewindow"):
                code, out, err = await self._run_hyprctl("activewindow -j")
                if code != 0:
                    return ToolResult(False, error=f"hyprctl activewindow failed: {err}")
                try:
                    data = json.loads(out)
                    return ToolResult(True, output=data)
                except json.JSONDecodeError:
                    return ToolResult(True, output=out)

            elif action in ("focus", "focuswindow"):
                if not target:
                    return ToolResult(False, error="Target window class, title, or address is required for 'focus' action.")
                dispatch_arg = f"address:{target}" if target.startswith("0x") else target
                code, out, err = await self._run_hyprctl(f"dispatch focuswindow {dispatch_arg}")
                if code != 0 or "failed" in err.lower():
                    return ToolResult(False, error=f"Failed to focus window: {err or out}")
                return ToolResult(True, output={"action": "focus", "target": target, "message": f"Focused window matching '{target}'."})

            elif action in ("workspace", "switch_workspace"):
                if not target:
                    return ToolResult(False, error="Target workspace ID or name is required for 'workspace' action.")
                code, out, err = await self._run_hyprctl(f"dispatch workspace {target}")
                if code != 0 or "failed" in err.lower():
                    return ToolResult(False, error=f"Failed to switch workspace: {err or out}")
                return ToolResult(True, output={"action": "workspace", "workspace": target, "message": f"Switched to workspace '{target}'."})

            elif action in ("move_to_workspace", "movetoworkspace", "move_to_ws"):
                if not target:
                    return ToolResult(False, error="Target workspace ID or name is required for 'move_to_workspace' action.")
                code, out, err = await self._run_hyprctl(f"dispatch movetoworkspace {target}")
                if code != 0 or "failed" in err.lower():
                    return ToolResult(False, error=f"Failed to move window to workspace: {err or out}")
                return ToolResult(True, output={"action": "move_to_workspace", "target_workspace": target, "message": f"Moved window to workspace '{target}'."})

            elif action in ("toggle_floating", "togglefloating"):
                code, out, err = await self._run_hyprctl("dispatch togglefloating")
                if code != 0:
                    return ToolResult(False, error=f"Failed to toggle floating: {err}")
                return ToolResult(True, output={"action": "toggle_floating", "message": "Toggled floating state of active window."})

            elif action in ("toggle_fullscreen", "fullscreen"):
                code, out, err = await self._run_hyprctl("dispatch fullscreen 1")
                if code != 0:
                    return ToolResult(False, error=f"Failed to toggle fullscreen: {err}")
                return ToolResult(True, output={"action": "fullscreen", "message": "Toggled fullscreen state of active window."})

            elif action in ("close", "closewindow"):
                arg = f"address:{target}" if target.startswith("0x") else target
                code, out, err = await self._run_hyprctl(f"dispatch closewindow {arg}" if arg else "dispatch closewindow")
                if code != 0:
                    return ToolResult(False, error=f"Failed to close window: {err}")
                return ToolResult(True, output={"action": "close", "target": target or "active", "message": f"Closed window '{target or 'active'}'."})

            else:
                return ToolResult(
                    False,
                    error=f"Unknown action '{action}'. Supported actions: list, get_active, focus, workspace, move_to_workspace, toggle_floating, toggle_fullscreen, close.",
                )

        except Exception as exc:
            logger.exception("Error executing Hyprland IPC command")
            return ToolResult(False, error=f"Hyprland IPC execution error: {exc}")
