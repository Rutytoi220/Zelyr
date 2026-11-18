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

    # -------------------------------------------------------------------------
    # Dispatch with Post-Execution State Verification
    # -------------------------------------------------------------------------

    async def dispatch(self, command: str, args: str = "", verify: bool = True) -> Dict[str, Any]:
        """Dispatch an actuator command with optional post-execution state verification."""
        cmd_str = f"dispatch {command} {args}".strip() if args else f"dispatch {command}"
        raw_resp = await self.send_command(cmd_str)

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
