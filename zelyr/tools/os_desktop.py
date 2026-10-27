"""Tier 1 Linux Desktop Media & Audio Control for Zelyr.

Provides deterministic control over local media players using MPRIS D-Bus via playerctl:
1. Status and metadata query (title, artist, album, status, volume).
2. Playback state control (play-pause, next, previous).
3. Volume adjustment and query.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from typing import Any, Dict, Optional

from zelyr.tools.core import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


class ManageMediaPlaybackTool(BaseTool):
    """Deterministically controls Linux media playback and audio via playerctl / MPRIS."""

    def __init__(self):
        super().__init__(
            tool_id="manage_media_playback",
            name="manage_media_playback",
            description=(
                "Tier 1 Linux Desktop Media & Audio Control: Control media players via MPRIS/playerctl. "
                "Query playback status, track metadata (title, artist, album), toggle play/pause, "
                "skip or return tracks, and adjust volume. Actions: 'status', 'play_pause', 'next', 'previous', 'volume'."
            ),
        )
        self.parameters = [
            ToolParameter(
                name="action",
                type="string",
                description="Media action to perform: 'status', 'play_pause', 'next', 'previous', 'volume'.",
                required=True,
            ),
            ToolParameter(
                name="player",
                type="string",
                description="Target media player name (e.g. 'spotify', 'firefox', 'chromium', 'vlc'). Defaults to active player.",
                required=False,
                default="",
            ),
            ToolParameter(
                name="value",
                type="string",
                description="Volume level or adjustment (e.g. '+5%', '-5%', '50%', '0.5') when action is 'volume'.",
                required=False,
                default="",
            ),
        ]

    async def _run_playerctl(self, *args: str, player: str = "") -> tuple[int, str, str]:
        if not shutil.which("playerctl"):
            return -1, "", "playerctl binary not found on PATH."

        cmd = ["playerctl"]
        if player:
            cmd.extend(["-p", player])
        cmd.extend(args)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=3.0)
            return (
                proc.returncode or 0,
                stdout.decode("utf-8", errors="replace").strip(),
                stderr.decode("utf-8", errors="replace").strip(),
            )
        except asyncio.TimeoutError:
            return -1, "", "playerctl command timed out after 3.0s."
        except Exception as exc:
            return -1, "", str(exc)

    async def execute(self, params: Optional[Dict[str, Any]] = None, **kwargs) -> ToolResult:
        merged = dict(params or {})
        merged.update(kwargs)

        action = str(merged.get("action") or "status").strip().lower()
        player = str(merged.get("player") or "").strip()
        value = str(merged.get("value") or "").strip()

        if not shutil.which("playerctl"):
            return ToolResult(
                True,
                output={
                    "status": "No active media players detected",
                    "note": "playerctl utility is not installed.",
                },
            )

        try:
            if action in ("status", "metadata", "info"):
                fmt = '{"player": "{{playerName}}", "status": "{{status}}", "title": "{{title}}", "artist": "{{artist}}", "album": "{{album}}"}'
                code, out, err = await self._run_playerctl("metadata", "--format", fmt, player=player)
                if code != 0 or not out:
                    return ToolResult(
                        True,
                        output={"status": "No active media players detected"},
                    )

                try:
                    data = json.loads(out)
                except json.JSONDecodeError:
                    data = {"raw_metadata": out, "status": "Active"}

                # Query volume as supplementary info
                vol_code, vol_out, _ = await self._run_playerctl("volume", player=player)
                if vol_code == 0 and vol_out:
                    try:
                        vol_val = float(vol_out)
                        data["volume"] = round(vol_val, 2)
                        data["volume_percent"] = f"{int(vol_val * 100)}%"
                    except ValueError:
                        data["volume"] = vol_out

                return ToolResult(True, output=data)

            elif action in ("play_pause", "play-pause", "toggle"):
                code, out, err = await self._run_playerctl("play-pause", player=player)
                if code != 0 and "no players" in err.lower():
                    return ToolResult(True, output={"status": "No active media players detected"})
                if code != 0:
                    return ToolResult(False, error=f"playerctl play-pause failed: {err or out}")
                return ToolResult(True, output={"action": "play_pause", "message": "Toggled media playback state."})

            elif action in ("next", "skip"):
                code, out, err = await self._run_playerctl("next", player=player)
                if code != 0 and "no players" in err.lower():
                    return ToolResult(True, output={"status": "No active media players detected"})
                if code != 0:
                    return ToolResult(False, error=f"playerctl next failed: {err or out}")
                return ToolResult(True, output={"action": "next", "message": "Skipped to next media track."})

            elif action in ("previous", "prev"):
                code, out, err = await self._run_playerctl("previous", player=player)
                if code != 0 and "no players" in err.lower():
                    return ToolResult(True, output={"status": "No active media players detected"})
                if code != 0:
                    return ToolResult(False, error=f"playerctl previous failed: {err or out}")
                return ToolResult(True, output={"action": "previous", "message": "Returned to previous media track."})

            elif action == "volume":
                if not value:
                    code, out, err = await self._run_playerctl("volume", player=player)
                    if code != 0:
                        if "no players" in err.lower():
                            return ToolResult(True, output={"status": "No active media players detected"})
                        return ToolResult(False, error=f"Failed to query volume: {err or out}")
                    try:
                        vol_float = float(out)
                        return ToolResult(
                            True,
                            output={"volume": round(vol_float, 2), "volume_percent": f"{int(vol_float * 100)}%"},
                        )
                    except ValueError:
                        return ToolResult(True, output={"volume": out})

                # Format volume value for playerctl
                target_val = value.strip()
                if target_val.startswith("+"):
                    clean = target_val[1:].rstrip("%")
                    try:
                        num = float(clean)
                        if num > 1.0 or "%" in target_val:
                            num = num / 100.0
                        arg = f"{num:g}+"
                    except ValueError:
                        arg = target_val
                elif target_val.startswith("-"):
                    clean = target_val[1:].rstrip("%")
                    try:
                        num = float(clean)
                        if num > 1.0 or "%" in target_val:
                            num = num / 100.0
                        arg = f"{num:g}-"
                    except ValueError:
                        arg = target_val
                elif target_val.endswith("%"):
                    clean = target_val.rstrip("%")
                    try:
                        num = float(clean) / 100.0
                        arg = f"{num:g}"
                    except ValueError:
                        arg = target_val
                else:
                    arg = target_val

                code, out, err = await self._run_playerctl("volume", arg, player=player)
                if code != 0 and "no players" in err.lower():
                    return ToolResult(True, output={"status": "No active media players detected"})
                if code != 0:
                    return ToolResult(False, error=f"Failed to set volume: {err or out}")

                # Query new volume
                _, new_vol_out, _ = await self._run_playerctl("volume", player=player)
                return ToolResult(
                    True,
                    output={"action": "volume", "requested": value, "applied": arg, "current_volume": new_vol_out or "unknown"},
                )

            else:
                return ToolResult(
                    False,
                    error=f"Unsupported media action '{action}'. Supported: status, play_pause, next, previous, volume.",
                )

        except Exception as exc:
            logger.exception("Media playback control error")
            return ToolResult(False, error=f"Media playback error: {exc}")


class SendDesktopNotificationTool(BaseTool):
    """Deterministically dispatches desktop notifications on Linux via notify-send."""

    def __init__(self):
        super().__init__(
            tool_id="send_desktop_notification",
            name="send_desktop_notification",
            description=(
                "Tier 1 Linux Desktop Notification: Dispatch native desktop notifications via notify-send. "
                "Supports urgency levels ('low', 'normal', 'critical'), notification title, message, and app name."
            ),
        )
        self.parameters = [
            ToolParameter(
                name="title",
                type="string",
                description="Notification title string.",
                required=True,
            ),
            ToolParameter(
                name="message",
                type="string",
                description="Notification body message string.",
                required=True,
            ),
            ToolParameter(
                name="urgency",
                type="string",
                description="Urgency level: 'low', 'normal', or 'critical'. Defaults to 'normal'.",
                required=False,
                default="normal",
            ),
            ToolParameter(
                name="app_name",
                type="string",
                description="Application name displayed with the notification. Defaults to 'Zelyr'.",
                required=False,
                default="Zelyr",
            ),
        ]

    async def execute(self, params: Optional[Dict[str, Any]] = None, **kwargs) -> ToolResult:
        merged = dict(params or {})
        merged.update(kwargs)

        title = str(merged.get("title") or "").strip()
        message = str(merged.get("message") or "").strip()
        urgency = str(merged.get("urgency") or "normal").strip().lower()
        app_name = str(merged.get("app_name") or "Zelyr").strip()

        if not title:
            return ToolResult(False, error="Title is required for desktop notification.")
        if not message:
            return ToolResult(False, error="Message body is required for desktop notification.")
        if urgency not in ("low", "normal", "critical"):
            urgency = "normal"

        if not shutil.which("notify-send"):
            logger.info("notify-send binary not found on PATH. Falling back to console output.")
            return ToolResult(
                True,
                output={
                    "success": True,
                    "fallback": "console",
                    "delivered": False,
                    "title": title,
                    "message": message,
                    "urgency": urgency,
                },
            )

        cmd = ["notify-send", "-u", urgency, "-a", app_name, title, message]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=3.0)
            if proc.returncode != 0:
                err_text = stderr.decode("utf-8", errors="replace").strip()
                return ToolResult(False, error=f"notify-send failed: {err_text}")

            return ToolResult(
                True,
                output={
                    "success": True,
                    "title": title,
                    "urgency": urgency,
                    "app_name": app_name,
                    "delivered": True,
                },
            )
        except asyncio.TimeoutError:
            return ToolResult(False, error="notify-send command timed out after 3.0s.")
        except Exception as exc:
            return ToolResult(False, error=f"Failed to dispatch desktop notification: {exc}")

