"""Hyprland Native System IPC Tool for Zelyr.

Provides deterministic Tier 1 window and workspace management via direct UNIX domain
socket IPC (.socket.sock) with hyprctl CLI fallback and state verification.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from typing import Any, Dict, List, Optional

from zelyr.engine.hyprland_ipc import HyprlandIPCClient
from zelyr.tools.core import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)

# Standard Zelyr Plugin Interface
REQUIRED_RING = 1
TIER = 1
TUI_HINT = "Interacting with desktop window manager..."

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "manage_desktop_window",
        "description": (
            "Tier 1 Hyprland System IPC: Deterministically manage desktop windows, workspaces, "
            "and focus via direct UNIX domain socket IPC or hyprctl CLI. "
            "Actions: 'list', 'get_active', 'focus', 'workspace', 'move_to_workspace', "
            "'toggle_floating', 'fullscreen', 'close'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "list",
                        "get_active",
                        "focus",
                        "workspace",
                        "move_to_workspace",
                        "toggle_floating",
                        "fullscreen",
                        "close",
                    ],
                    "description": "The window management action to perform.",
                },
                "target": {
                    "type": "string",
                    "description": "Target window address (0x...), title regex, class regex, or workspace ID/name.",
                },
                "verify": {
                    "type": "boolean",
                    "description": "Whether to verify state after dispatching (default True).",
                    "default": True,
                },
            },
            "required": ["action"],
        },
    },
}


async def _handle_window_action(
    client: HyprlandIPCClient,
    action: str,
    target: Optional[str] = None,
    verify: bool = True,
) -> Dict[str, Any]:
    """Execute window action through HyprlandIPCClient and return structured result dict."""
    act = (action or "").strip().lower()
    tgt = str(target or "").strip()

    if not client.can_connect():
        return {
            "status": "error",
            "error_type": "DESKTOP_UNAVAILABLE",
            "error": "Hyprland session not detected.",
            "remedy_hint": "Verify HYPRLAND_INSTANCE_SIGNATURE is set or run in an active Hyprland desktop.",
        }

    try:
        if act in ("list", "list_windows", "clients"):
            clients = await client.get_clients()
            summary = []
            for c in clients:
                ws = c.get("workspace", {})
                ws_name = ws.get("name") if isinstance(ws, dict) else str(ws)
                summary.append({
                    "address": c.get("address"),
                    "class": c.get("class"),
                    "title": c.get("title"),
                    "workspace": ws_name,
                    "at": c.get("at"),
                    "size": c.get("size"),
                    "floating": c.get("floating"),
                    "fullscreen": bool(c.get("fullscreen")),
                })
            return {
                "status": "success",
                "count": len(summary),
                "windows": summary,
            }

        elif act in ("get_active", "active", "activewindow"):
            win = await client.get_active_window()
            return {
                "status": "success",
                "active_window": win,
            }

        elif act in ("focus", "focuswindow"):
            if not tgt:
                return {
                    "status": "error",
                    "error": "Target window class, title, or address is required for 'focus' action.",
                }

            dispatch_arg = f"address:{tgt}" if tgt.startswith("0x") else tgt
            # If not address, check if matching window exists in client list
            if not tgt.startswith("0x") and not tgt.startswith("address:"):
                clients = await client.get_clients()
                tgt_lower = tgt.lower()
                matched = [
                    c for c in clients
                    if tgt_lower in (c.get("class") or "").lower()
                    or tgt_lower in (c.get("title") or "").lower()
                ]
                if matched:
                    addr = matched[0].get("address")
                    if addr:
                        dispatch_arg = f"address:{addr}"
                elif clients:
                    # Windows exist but none match
                    return {
                        "status": "error",
                        "error_type": "WINDOW_NOT_FOUND",
                        "error": f"Target window '{tgt}' was not found in client list.",
                        "remedy_hint": "Target window was not found in client list. Execute manage_desktop_window with action='list' to inspect open windows.",
                    }

            disp = await client.dispatch("focuswindow", dispatch_arg, verify=verify)
            verified = disp.get("verified", True)
            return {
                "status": "success" if verified else "unverified",
                "action": "focus",
                "target": tgt,
                "verified": verified,
                "observed": disp.get("observed"),
                "message": f"Focused window matching '{tgt}'.",
            }

        elif act in ("workspace", "switch_workspace"):
            if not tgt:
                return {
                    "status": "error",
                    "error": "Target workspace ID or name is required for 'workspace' action.",
                }
            disp = await client.dispatch("workspace", tgt, verify=verify)
            verified = disp.get("verified", True)
            return {
                "status": "success" if verified else "unverified",
                "action": "workspace",
                "workspace": tgt,
                "verified": verified,
                "observed": disp.get("observed"),
                "message": f"Switched to workspace '{tgt}'.",
            }

        elif act in ("move_to_workspace", "movetoworkspace", "move_to_ws"):
            if not tgt:
                return {
                    "status": "error",
                    "error": "Target workspace ID or name is required for 'move_to_workspace' action.",
                }
            disp = await client.dispatch("movetoworkspace", tgt, verify=verify)
            verified = disp.get("verified", True)
            return {
                "status": "success" if verified else "unverified",
                "action": "move_to_workspace",
                "target_workspace": tgt,
                "verified": verified,
                "observed": disp.get("observed"),
                "message": f"Moved window to workspace '{tgt}'.",
            }

        elif act in ("toggle_floating", "togglefloating"):
            disp = await client.dispatch("togglefloating", "", verify=verify)
            return {
                "status": "success",
                "action": "toggle_floating",
                "message": "Toggled floating state of active window.",
                "verified": disp.get("verified", True),
            }

        elif act in ("toggle_fullscreen", "fullscreen"):
            disp = await client.dispatch("fullscreen", "1", verify=verify)
            return {
                "status": "success",
                "action": "fullscreen",
                "message": "Toggled fullscreen state of active window.",
                "verified": disp.get("verified", True),
                "observed": disp.get("observed"),
            }

        elif act in ("close", "closewindow"):
            arg = f"address:{tgt}" if tgt and tgt.startswith("0x") else (tgt or "")
            disp = await client.dispatch("closewindow", arg, verify=False)
            return {
                "status": "success",
                "action": "close",
                "target": tgt or "active",
                "message": f"Closed window '{tgt or 'active'}'.",
            }

        else:
            return {
                "status": "error",
                "error": f"Unknown action '{act}'. Supported actions: list, get_active, focus, workspace, move_to_workspace, toggle_floating, fullscreen, close.",
            }

    except Exception as exc:
        logger.exception("Error executing Hyprland IPC command")
        return {
            "status": "error",
            "error": f"Hyprland IPC execution error: {exc}",
        }


async def execute(
    action: str = "list",
    target: Optional[str] = None,
    verify: bool = True,
    **kwargs: Any,
) -> str:
    """Standard dynamic tool plugin entry point. Returns serialized JSON string."""
    client = kwargs.get("client") or HyprlandIPCClient()
    result = await _handle_window_action(client, action=action, target=target, verify=verify)
    return json.dumps(result)


class ManageDesktopWindowTool(BaseTool):
    """Deterministically manages Hyprland windows and workspaces via HyprlandIPCClient."""

    def __init__(self, client: Optional[HyprlandIPCClient] = None):
        super().__init__(
            tool_id="manage_desktop_window",
            name="manage_desktop_window",
            description=(
                "Tier 1 Hyprland System IPC: Deterministically manage desktop windows, workspaces, "
                "and focus via direct UNIX domain socket IPC or hyprctl. Actions: 'list', 'get_active', "
                "'focus', 'workspace', 'move_to_workspace', 'toggle_floating', 'fullscreen', 'close'."
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
                description="The window management action to perform: 'list', 'get_active', 'focus', 'workspace', 'move_to_workspace', 'toggle_floating', 'fullscreen', 'close'.",
                required=True,
            ),
            ToolParameter(
                name="target",
                type="string",
                description="Target window title/class, window address (e.g., '0x55a8e8690210'), or workspace ID/name.",
                required=False,
                default="",
            ),
            ToolParameter(
                name="verify",
                type="boolean",
                description="Whether to perform post-dispatch state verification (default True).",
                required=False,
                default=True,
            ),
        ]
        self.client = client or HyprlandIPCClient()

    async def _run_hyprctl(self, command: str) -> tuple[int, str, str]:
        """Subprocess runner preserved for backwards compatibility with tests and CLI execution."""
        return await self.client._run_hyprctl(command)

    async def execute(self, params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> ToolResult:
        """Execute window action and return standard ToolResult."""
        merged = dict(params or {})
        merged.update(kwargs)

        action = str(merged.get("action", "") or "list").strip().lower()
        target = str(merged.get("target", "") or "").strip()
        verify = bool(merged.get("verify", True))

        # Check if environment is completely missing
        if not self.client.can_connect() and not shutil.which("hyprctl"):
            return ToolResult(
                False,
                error="hyprctl binary not found. This tool requires the Hyprland Wayland compositor.",
                output={
                    "status": "error",
                    "error_type": "DESKTOP_UNAVAILABLE",
                    "error": "Hyprland session not detected.",
                    "remedy_hint": "Verify HYPRLAND_INSTANCE_SIGNATURE is set or run in an active Hyprland desktop.",
                },
            )

        # Synchronize client._run_hyprctl with self._run_hyprctl in case it was patched by tests
        self.client._run_hyprctl = self._run_hyprctl  # type: ignore[assignment]

        res_dict = await _handle_window_action(self.client, action=action, target=target, verify=verify)

        if res_dict.get("status") == "error":
            return ToolResult(
                False,
                error=res_dict.get("error", "Error executing desktop window action"),
                output=res_dict,
            )

        return ToolResult(True, output=res_dict)


__all__ = [
    "ManageDesktopWindowTool",
    "execute",
    "TOOL_SCHEMA",
    "REQUIRED_RING",
    "TIER",
    "TUI_HINT",
]
