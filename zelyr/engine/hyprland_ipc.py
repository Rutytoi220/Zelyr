"""Direct Asynchronous Hyprland UNIX Domain Socket IPC Actuator with Deterministic State Verification.

Provides low-latency native Wayland interaction with Hyprland via direct UNIX socket IPC
(.socket.sock), bounded buffer reads, CLI fallback via hyprctl, and deterministic post-dispatch
state verification.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class HyprlandUnavailableError(RuntimeError):
    """Raised when Hyprland IPC socket and hyprctl CLI are both unavailable."""


class HyprlandIPCClient:
    """Asynchronous client for Hyprland UNIX domain socket IPC with CLI fallback."""

    def __init__(self, socket_path: Optional[Union[str, Path]] = None, timeout: float = 2.0):
        self._custom_socket_path = Path(socket_path) if socket_path else None
        self.timeout = timeout

    def get_socket_path(self) -> Optional[Path]:
        """Resolve the active Hyprland UNIX domain command socket path.

        Follows standard Hyprland specification:
        $XDG_RUNTIME_DIR/hypr/$HYPRLAND_INSTANCE_SIGNATURE/.socket.sock
        """
        if self._custom_socket_path is not None:
            return self._custom_socket_path

        runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
        signature = os.environ.get("HYPRLAND_INSTANCE_SIGNATURE")
        if runtime_dir and signature:
            return Path(runtime_dir) / "hypr" / signature / ".socket.sock"
        return None

    def is_available(self) -> bool:
        """Check if direct Hyprland UNIX domain socket is available on filesystem."""
        sock = self.get_socket_path()
        if sock is not None:
            try:
                return sock.is_socket() or sock.exists()
            except OSError:
                return False
        return False

    def can_connect(self) -> bool:
        """Check if either direct UNIX socket or hyprctl CLI fallback is available."""
        return self.is_available() or bool(shutil.which("hyprctl"))

    async def _run_hyprctl(self, command: str) -> tuple[int, str, str]:
        """Execute hyprctl command via subprocess."""
        if not shutil.which("hyprctl"):
            return -1, "", "hyprctl executable not found on PATH."

        cmd_parts = ["hyprctl"] + command.split()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
            return (
                proc.returncode or 0,
                stdout.decode("utf-8", errors="replace"),
                stderr.decode("utf-8", errors="replace"),
            )
        except asyncio.TimeoutError:
            return -1, "", f"hyprctl command timed out after {self.timeout}s."
        except Exception as exc:
            return -1, "", str(exc)

    async def _cli_fallback(self, command: str, timeout: float = 2.0) -> str:
        """Fallback to executing hyprctl CLI when direct socket is unreachable."""
        clean = command.strip()
        if clean.startswith("j/"):
            subcmd = clean[2:]
            code, out, err = await self._run_hyprctl(f"{subcmd} -j")
        elif clean.startswith("dispatch "):
            code, out, err = await self._run_hyprctl(clean)
        else:
            code, out, err = await self._run_hyprctl(clean)

        if code != 0 and not out.strip():
            raise HyprlandUnavailableError(f"hyprctl fallback failed (code {code}): {err}")
        return out

    async def send_command(self, command: str, timeout: Optional[float] = None) -> str:
        """Send command over direct Hyprland UNIX domain socket with bounded read and CLI fallback.

        Buffer limit: 2MB up to EOF.
        """
        eff_timeout = timeout if timeout is not None else self.timeout
        sock = self.get_socket_path()

        if sock and (sock.is_socket() or sock.exists()):
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_unix_connection(str(sock)),
                    timeout=eff_timeout,
                )
                try:
                    writer.write(command.encode("utf-8"))
                    await writer.drain()

                    # Bounded buffer read up to 2MB
                    max_bytes = 2 * 1024 * 1024
                    chunks = []
                    total_bytes = 0
                    while True:
                        chunk = await asyncio.wait_for(reader.read(4096), timeout=eff_timeout)
                        if not chunk:
                            break
                        chunks.append(chunk)
                        total_bytes += len(chunk)
                        if total_bytes >= max_bytes:
                            break
                    return b"".join(chunks).decode("utf-8", errors="replace")
                finally:
                    try:
                        writer.close()
                        await writer.wait_closed()
                    except Exception:
                        pass
            except Exception as sock_exc:
                logger.debug("Direct Hyprland socket communication failed: %s; falling back to CLI", sock_exc)

        # Fallback to hyprctl if available
        if shutil.which("hyprctl"):
            return await self._cli_fallback(command, timeout=eff_timeout)

        raise HyprlandUnavailableError("Hyprland IPC socket and hyprctl CLI not available.")

    # -------------------------------------------------------------------------
    # Typed Queries
    # -------------------------------------------------------------------------

    async def get_clients(self) -> List[Dict[str, Any]]:
        """Query all open windows/clients on the compositor (j/clients)."""
        resp = await self.send_command("j/clients")
        try:
            return json.loads(resp)
        except json.JSONDecodeError as err:
            logger.debug("Failed to decode clients JSON: %s", err)
            return []

    async def get_active_window(self) -> Dict[str, Any]:
        """Query the currently focused window (j/activewindow)."""
        resp = await self.send_command("j/activewindow")
        try:
            return json.loads(resp)
        except json.JSONDecodeError as err:
            logger.debug("Failed to decode active window JSON: %s", err)
            return {}

    async def get_workspaces(self) -> List[Dict[str, Any]]:
        """Query all existing workspaces (j/workspaces)."""
        resp = await self.send_command("j/workspaces")
        try:
            return json.loads(resp)
        except json.JSONDecodeError as err:
            logger.debug("Failed to decode workspaces JSON: %s", err)
            return []

    async def get_active_workspace(self) -> Dict[str, Any]:
        """Query the currently active workspace (j/activeworkspace)."""
        resp = await self.send_command("j/activeworkspace")
        try:
            return json.loads(resp)
        except json.JSONDecodeError as err:
            logger.debug("Failed to decode active workspace JSON: %s", err)
            return {}

    async def get_monitors(self) -> List[Dict[str, Any]]:
        """Query connected compositor monitors and logical dimensions (j/monitors)."""
        resp = await self.send_command("j/monitors")
        try:
            raw = json.loads(resp)
        except json.JSONDecodeError as err:
            logger.debug("Failed to decode monitors JSON: %s", err)
            return []
        if not isinstance(raw, list):
            return []
        monitors: List[Dict[str, Any]] = []
        for mon in raw:
            if not isinstance(mon, dict):
                continue
            scale = float(mon.get("scale") or 1.0)
            if scale <= 0:
                scale = 1.0
            px_w = int(mon.get("width") or 0)
            px_h = int(mon.get("height") or 0)
            log_w = max(1, int(round(px_w / scale))) if px_w > 0 else 0
            log_h = max(1, int(round(px_h / scale))) if px_h > 0 else 0
            ws = mon.get("activeWorkspace") if isinstance(mon.get("activeWorkspace"), dict) else {}
            monitors.append({
                "id": mon.get("id"),
                "name": str(mon.get("name") or ""),
                "x": int(mon.get("x") or 0),
                "y": int(mon.get("y") or 0),
                "width": px_w,
                "height": px_h,
                "logical_width": log_w,
                "logical_height": log_h,
                "scale": scale,
                "focused": bool(mon.get("focused")),
                "active_workspace": ws.get("name") or ws.get("id"),
            })
        return monitors

    async def get_cursor_pos(self) -> Optional[tuple[int, int]]:
        """Query the current compositor cursor position (j/cursorpos)."""
        resp = await self.send_command("j/cursorpos")
        try:
            data = json.loads(resp)
        except json.JSONDecodeError:
            return None
        if isinstance(data, dict) and "x" in data and "y" in data:
            return int(round(float(data["x"]))), int(round(float(data["y"])))
        return None

    async def get_desktop_geometry(self) -> Dict[str, Any]:
        """Compute active monitor rectangles and total virtual desktop bounding box."""
        monitors = await self.get_monitors()
        if not monitors:
            return {
                "monitors": [],
                "focused_monitor": None,
                "min_x": 0,
                "min_y": 0,
                "max_x": 0,
                "max_y": 0,
                "virtual_width": 0,
                "virtual_height": 0,
            }
        min_x = min(m["x"] for m in monitors)
        min_y = min(m["y"] for m in monitors)
        max_x = max(m["x"] + m["logical_width"] for m in monitors)
        max_y = max(m["y"] + m["logical_height"] for m in monitors)
        focused = next((m for m in monitors if m.get("focused")), monitors[0])
        return {
            "monitors": monitors,
            "focused_monitor": focused,
            "min_x": min_x,
            "min_y": min_y,
            "max_x": max_x,
            "max_y": max_y,
            "virtual_width": max_x - min_x,
            "virtual_height": max_y - min_y,
        }

    @staticmethod
    def point_in_monitors(x: int, y: int, monitors: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Return the monitor dict containing logical point (x, y), or None if out of bounds."""
        for m in monitors:
            mx = int(m.get("x", 0))
            my = int(m.get("y", 0))
            mw = int(m.get("logical_width") or m.get("width") or 0)
            mh = int(m.get("logical_height") or m.get("height") or 0)
            if mx <= x < mx + mw and my <= y < my + mh:
                return m
        return None

    @staticmethod
    def _match_client(clients: List[Dict[str, Any]], target: str) -> Optional[Dict[str, Any]]:
        """Find the first Hyprland client matching address, class, or title."""
        tgt = str(target or "").strip()
        if not tgt:
            return None
        tgt_l = tgt.lower()
        if tgt_l.startswith("address:"):
            addr = tgt_l.removeprefix("address:").strip()
            for c in clients:
                if str(c.get("address") or "").lower() == addr:
                    return c
            return None
        if tgt_l.startswith("0x"):
            for c in clients:
                if str(c.get("address") or "").lower() == tgt_l:
                    return c
            return None
        for c in clients:
            c_cls = str(c.get("class") or c.get("initialClass") or "").lower()
            c_title = str(c.get("title") or c.get("initialTitle") or "").lower()
            if tgt_l in c_title or tgt_l in c_cls:
                return c
        return None

    async def get_window_by_target(self, target: str) -> Optional[Dict[str, Any]]:
        """Resolve an open window client matching address, class, or title."""
        clients = await self.get_clients()
        return self._match_client(clients, target)

    # -------------------------------------------------------------------------
    # Dispatch with Post-Execution State Verification
    # -------------------------------------------------------------------------

    @staticmethod
    def _lua_quote(val: str) -> str:
        escaped = str(val or "").replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    @classmethod
    def _build_lua_eval_dispatch(cls, command: str, args: str = "") -> Optional[str]:
        """Translate legacy Hyprland dispatch syntax to Hyprland 0.57+ Lua eval hl.dispatch(...)."""
        cmd_l = (command or "").strip().lower()
        arg_s = str(args or "").strip()
        if cmd_l == "workspace" and arg_s:
            return f"eval hl.dispatch(hl.dsp.focus({{workspace={cls._lua_quote(arg_s)}}}))"
        if cmd_l == "focuswindow" and arg_s:
            return f"eval hl.dispatch(hl.dsp.focus({{window={cls._lua_quote(arg_s)}}}))"
        if cmd_l == "movecursor":
            parts = arg_s.replace(",", " ").split()
            if len(parts) >= 2:
                try:
                    ix, iy = int(float(parts[0])), int(float(parts[1]))
                    return f"eval hl.dispatch(hl.dsp.cursor.move({{x={ix}, y={iy}}}))"
                except ValueError:
                    pass
        if cmd_l in ("movetoworkspace", "movetoworkspacesilent") and arg_s:
            silent = "true" if cmd_l == "movetoworkspacesilent" else "false"
            if "," in arg_s:
                ws_part, win_part = [p.strip() for p in arg_s.split(",", 1)]
                ws_val = ws_part if ws_part.lstrip("-").isdigit() else cls._lua_quote(ws_part)
                return f"eval hl.dispatch(hl.dsp.window.move({{workspace={ws_val}, window={cls._lua_quote(win_part)}, silent={silent}}}))"
            ws_val = arg_s if arg_s.lstrip("-").isdigit() else cls._lua_quote(arg_s)
            return f"eval hl.dispatch(hl.dsp.window.move({{workspace={ws_val}, silent={silent}}}))"
        if cmd_l == "togglefloating":
            if arg_s:
                return f"eval hl.dispatch(hl.dsp.window.float({{window={cls._lua_quote(arg_s)}, mode=\"toggle\"}}))"
            return 'eval hl.dispatch(hl.dsp.window.float({mode="toggle"}))'
        if cmd_l == "fullscreen":
            mode = "maximized" if arg_s == "1" else "fullscreen"
            return f"eval hl.dispatch(hl.dsp.window.fullscreen({{mode={cls._lua_quote(mode)}}}))"
        if cmd_l == "closewindow":
            if arg_s:
                return f"eval hl.dispatch(hl.dsp.window.close({{window={cls._lua_quote(arg_s)}}}))"
            return "eval hl.dispatch(hl.dsp.window.close({}))"
        if cmd_l in ("resizewindowpixel", "movewindowpixel") and arg_s:
            coords_part, _, win_part = arg_s.partition(",")
            tokens = coords_part.replace("exact", "").strip().split()
            if len(tokens) >= 2:
                try:
                    vx, vy = int(float(tokens[0])), int(float(tokens[1]))
                    fn_name = "resize" if cmd_l == "resizewindowpixel" else "move"
                    win_field = f"window={cls._lua_quote(win_part.strip())}, " if win_part.strip() else ""
                    return f"eval hl.dispatch(hl.dsp.window.{fn_name}({{{win_field}x={vx}, y={vy}, relative=false}}))"
                except ValueError:
                    pass
        return None

    async def dispatch(self, command: str, args: str = "", verify: bool = True) -> Dict[str, Any]:
        """Dispatch an actuator command with optional post-execution state verification."""
        cmd_str = f"dispatch {command} {args}".strip() if args else f"dispatch {command}"
        raw_resp = await self.send_command(cmd_str)
        raw_lower = str(raw_resp or "").lower()
        if (
            "note: dispatch in lua" in raw_lower
            or "hl.dispatch" in raw_lower
            or "expected near" in raw_lower
        ):
            lua_cmd = self._build_lua_eval_dispatch(command, args)
            if lua_cmd:
                raw_resp = await self.send_command(lua_cmd)

        verified = True
        observed: Any = raw_resp

        if verify:
            verified = False
            try:
                cmd_lower = command.lower().strip()
                if cmd_lower in ("workspace",):
                    target = str(args).strip()
                    ws = await self.get_active_workspace()
                    ws_id = str(ws.get("id", ""))
                    ws_name = str(ws.get("name", ""))
                    verified = (ws_id == target or ws_name == target)
                    observed = ws
                elif cmd_lower in ("focuswindow",):
                    win = await self.get_active_window()
                    target = str(args).strip()
                    if target.startswith("address:"):
                        addr = target.removeprefix("address:").strip().lower()
                        verified = (win.get("address") or "").lower() == addr
                    elif target.startswith("0x"):
                        verified = (win.get("address") or "").lower() == target.lower()
                    else:
                        target_l = target.lower()
                        win_class = (win.get("class") or "").lower()
                        win_title = (win.get("title") or "").lower()
                        verified = target_l in win_class or target_l in win_title
                    observed = win
                elif cmd_lower in ("movetoworkspace",):
                    target = str(args).strip()
                    clients = await self.get_clients()
                    active = await self.get_active_window()
                    active_addr = (active.get("address") or "").lower()
                    matching = [c for c in clients if (c.get("address") or "").lower() == active_addr]
                    if matching:
                        c = matching[0]
                        ws = c.get("workspace", {})
                        ws_id = str(ws.get("id", "")) if isinstance(ws, dict) else ""
                        ws_name = str(ws.get("name", "")) if isinstance(ws, dict) else str(ws)
                        verified = (ws_id == target or ws_name == target)
                        observed = c
                    else:
                        verified = any(
                            str(c.get("workspace", {}).get("id")) == target
                            or str(c.get("workspace", {}).get("name")) == target
                            for c in clients
                        )
                        observed = {"clients_matching_target": target}
                else:
                    verified = True
                    observed = raw_resp
            except Exception as v_err:
                logger.debug("Post-dispatch state verification query failed: %s", v_err)
                verified = False
                observed = {"raw": raw_resp, "verification_error": str(v_err)}

        return {
            "status": "success" if verified else "failed",
            "command": command,
            "args": args,
            "verified": verified,
            "observed": observed,
        }


__all__ = [
    "HyprlandUnavailableError",
    "HyprlandIPCClient",
]
