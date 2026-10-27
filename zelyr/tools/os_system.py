"""Tier 1 Linux & Desktop Power Tools for Zelyr.

Provides high-speed, deterministic OS control tools that run via native Linux syscalls,
standard CLI utilities, and Hyprland IPC:
1. manage_system_process: Deterministic listing, inspection, and termination with PID protection.
2. manage_system_clipboard: Wayland/X11 clipboard reading and writing.
3. query_system_journal: journalctl query tool for system diagnostics.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import signal
import socket
import time
from typing import Any, Dict, List, Optional, Tuple

from zelyr.tools.core import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)

# Critical system processes protected from termination
PROTECTED_COMM_NAMES = {
    "systemd",
    "init",
    "hyprland",
    "waybar",
    "sway",
    "xorg",
    "xwayland",
    "gnome-shell",
    "kwin",
    "dbus-daemon",
    "pipewire",
    "wireplumber",
}


class ManageSystemProcessTool(BaseTool):
    """Deterministically manages Linux processes, sockets, and signals."""

    def __init__(self):
        super().__init__(
            tool_id="manage_system_process",
            name="manage_system_process",
            description=(
                "Tier 1 Linux Process Management: Deterministically list top processes, inspect specific PID/process/port (:port), "
                "or terminate a process with signal. Actions: 'list', 'inspect', 'kill'. Protects system critical PIDs (PID 1, desktop manager, current process)."
            ),
        )
        self.parameters = [
            ToolParameter(
                name="action",
                type="string",
                description="Process action: 'list' (top resource users), 'inspect' (detailed PID, name, or :port stats), 'kill' (terminate process).",
                required=True,
            ),
            ToolParameter(
                name="target",
                type="string",
                description="Process identifier: PID (e.g. 1234), process name (e.g. 'zen'), or port number (e.g. ':41144').",
                required=False,
                default="",
            ),
            ToolParameter(
                name="signal",
                type="integer",
                description="Signal to send when action is 'kill': default 15 (SIGTERM), supports 9 (SIGKILL).",
                required=False,
                default=15,
            ),
        ]

    def _is_pid_protected(self, pid: int, comm: str = "") -> Tuple[bool, str]:
        """Check if PID is protected against termination."""
        if pid == 1:
            return True, "PID 1 (init/systemd) is protected by the operating system kernel."
        if pid == os.getpid():
            return True, f"PID {pid} is the current Zelyr runtime process."
        if pid == os.getppid():
            return True, f"PID {pid} is the parent process."

        # Check comm from /proc/<pid>/comm or /proc/<pid>/stat if not provided
        if not comm:
            try:
                with open(f"/proc/{pid}/comm", "r", encoding="utf-8", errors="replace") as f:
                    comm = f.read().strip()
            except Exception:
                pass

        comm_lower = comm.lower()
        for protected in PROTECTED_COMM_NAMES:
            if protected in comm_lower:
                return True, f"Process '{comm}' (PID {pid}) is a protected core system/desktop component."

        return False, ""

    async def _find_pid_by_port(self, port_str: str) -> Optional[int]:
        """Resolve PID listening on a specific TCP/UDP port."""
        port = port_str.lstrip(":")
        if not port.isdigit():
            return None

        # 1. Try ss -tulpn
        if shutil.which("ss"):
            try:
                proc = await asyncio.create_subprocess_exec(
                    "ss", "-tulpn",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)
                lines = stdout.decode("utf-8", errors="replace").splitlines()
                port_pattern = re.compile(rf":{port}\b")
                for line in lines:
                    if port_pattern.search(line):
                        pid_match = re.search(r"pid=(\d+)", line)
                        if pid_match:
                            return int(pid_match.group(1))
            except Exception:
                pass

        # 2. Try lsof -i :<port> -t
        if shutil.which("lsof"):
            try:
                proc = await asyncio.create_subprocess_exec(
                    "lsof", f"-i:{port}", "-t",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)
                out = stdout.decode().strip()
                if out:
                    first_pid = out.splitlines()[0].strip()
                    if first_pid.isdigit():
                        return int(first_pid)
            except Exception:
                pass

        # 3. Try fuser <port>/tcp
        if shutil.which("fuser"):
            try:
                proc = await asyncio.create_subprocess_exec(
                    "fuser", f"{port}/tcp",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)
                pids = stdout.decode().strip().split()
                if pids and pids[0].isdigit():
                    return int(pids[0])
            except Exception:
                pass

        return None

    async def _get_top_processes(self, limit: int = 15) -> List[Dict[str, Any]]:
        """Retrieve top processes sorted by CPU usage."""
        cmd = ["ps", "-eo", "pid,ppid,comm,%cpu,%mem", "--sort=-%cpu"]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3.0)
        lines = stdout.decode("utf-8", errors="replace").splitlines()
        results = []
        if len(lines) > 1:
            for line in lines[1 : limit + 1]:
                parts = line.strip().split(None, 4)
                if len(parts) >= 5:
                    try:
                        results.append({
                            "pid": int(parts[0]),
                            "ppid": int(parts[1]),
                            "comm": parts[2],
                            "cpu": float(parts[3]),
                            "mem": float(parts[4]),
                        })
                    except ValueError:
                        continue
        return results

    async def _inspect_pid(self, pid: int) -> Optional[Dict[str, Any]]:
        """Retrieve detailed metrics for a given PID."""
        if not os.path.exists(f"/proc/{pid}"):
            return None

        cmd = ["ps", "-p", str(pid), "-o", "pid,ppid,%cpu,%mem,comm,args"]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)
        lines = stdout.decode("utf-8", errors="replace").splitlines()
        if len(lines) > 1:
            parts = lines[1].strip().split(None, 5)
            if len(parts) >= 6:
                return {
                    "pid": int(parts[0]),
                    "ppid": int(parts[1]),
                    "cpu": float(parts[2]),
                    "mem": float(parts[3]),
                    "comm": parts[4],
                    "args": parts[5],
                    "status": "running",
                }
            elif len(parts) >= 5:
                return {
                    "pid": int(parts[0]),
                    "ppid": int(parts[1]),
                    "cpu": float(parts[2]),
                    "mem": float(parts[3]),
                    "comm": parts[4],
                    "args": parts[4],
                    "status": "running",
                }
        return {"pid": pid, "status": "running"}

    async def _find_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Find processes matching command name or arguments."""
        cmd = ["ps", "-eo", "pid,ppid,%cpu,%mem,comm,args", "--sort=-%cpu"]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3.0)
        lines = stdout.decode("utf-8", errors="replace").splitlines()
        matches = []
        target_lower = name.lower()
        if len(lines) > 1:
            for line in lines[1:]:
                parts = line.strip().split(None, 5)
                if len(parts) >= 6:
                    comm = parts[4]
                    args = parts[5]
                    if target_lower in comm.lower() or target_lower in args.lower():
                        try:
                            matches.append({
                                "pid": int(parts[0]),
                                "ppid": int(parts[1]),
                                "cpu": float(parts[2]),
                                "mem": float(parts[3]),
                                "comm": comm,
                                "args": args,
                            })
                            if len(matches) >= 15:
                                break
                        except ValueError:
                            continue
        return matches

    async def execute(self, params: Optional[Dict[str, Any]] = None, **kwargs) -> ToolResult:
        merged = dict(params or {})
        merged.update(kwargs)

        action = str(merged.get("action") or "list").strip().lower()
        target = str(merged.get("target") or "").strip()

        if action == "list":
            procs = await self._get_top_processes(15)
            return ToolResult(True, output={"processes": procs, "count": len(procs)})

        if action == "inspect":
            if not target:
                procs = await self._get_top_processes(15)
                return ToolResult(True, output={"processes": procs, "count": len(procs)})

            # Port lookup (e.g. ':41144' or '41144' if not an existing PID)
            if target.startswith(":") or (target.isdigit() and int(target) > 1024 and not os.path.exists(f"/proc/{target}")):
                pid = await self._find_pid_by_port(target)
                if pid:
                    info = await self._inspect_pid(pid)
                    return ToolResult(True, output={"port": target, "process": info})
                return ToolResult(False, error=f"No active process found listening on port {target}")

            # Direct PID lookup
            if target.isdigit():
                pid = int(target)
                info = await self._inspect_pid(pid)
                if info:
                    return ToolResult(True, output=info)
                return ToolResult(False, error=f"Process with PID {pid} not found.")

            # Process name lookup
            matches = await self._find_by_name(target)
            if matches:
                return ToolResult(True, output={"matches": matches, "count": len(matches)})
            return ToolResult(False, error=f"No running process matching '{target}' found.")

        if action == "kill":
            if not target:
                return ToolResult(False, error="Target PID, process name, or :port is required for 'kill' action.")

            target_pid: Optional[int] = None
            if target.startswith(":"):
                target_pid = await self._find_pid_by_port(target)
                if not target_pid:
                    return ToolResult(False, error=f"No process found holding port {target}")
            elif target.isdigit():
                target_pid = int(target)
            else:
                matches = await self._find_by_name(target)
                if not matches:
                    return ToolResult(False, error=f"No running process matching '{target}' found to kill.")
                target_pid = matches[0]["pid"]

            # Protection guardrail
            is_prot, reason = self._is_pid_protected(target_pid)
            if is_prot:
                return ToolResult(False, error=f"Termination refused: {reason}")

            # Parse signal
            sig_arg = merged.get("signal", 15)
            sig_num = signal.SIGTERM
            if str(sig_arg) in ("9", "SIGKILL", "KILL"):
                sig_num = signal.SIGKILL
            elif str(sig_arg) in ("15", "SIGTERM", "TERM"):
                sig_num = signal.SIGTERM
            elif isinstance(sig_arg, int):
                sig_num = sig_arg

            try:
                os.kill(target_pid, sig_num)
                return ToolResult(True, output={
                    "killed_pid": target_pid,
                    "signal": int(sig_num),
                    "message": f"Process {target_pid} terminated with signal {int(sig_num)}.",
                })
            except ProcessLookupError:
                return ToolResult(False, error=f"Process with PID {target_pid} does not exist.")
            except PermissionError:
                return ToolResult(False, error=f"Permission denied to kill PID {target_pid}.")
            except Exception as e:
                return ToolResult(False, error=f"Failed to terminate PID {target_pid}: {e}")

        return ToolResult(False, error=f"Unsupported action: '{action}'. Use 'list', 'inspect', or 'kill'.")


class ManageSystemClipboardTool(BaseTool):
    """Deterministically reads and writes the system clipboard on Linux."""

    def __init__(self):
        super().__init__(
            tool_id="manage_system_clipboard",
            name="manage_system_clipboard",
            description=(
                "Tier 1 Linux Clipboard Management: Read or write to the system clipboard in Wayland (wl-paste/wl-copy) "
                "with fallback to X11 (xclip/xsel). Actions: 'read', 'write'."
            ),
        )
        self.parameters = [
            ToolParameter(
                name="action",
                type="string",
                description="Clipboard action: 'read' (fetch clipboard text) or 'write' (set clipboard content).",
                required=True,
            ),
            ToolParameter(
                name="content",
                type="string",
                description="Text content to write to the clipboard when action is 'write'.",
                required=False,
                default="",
            ),
        ]

    def _is_wayland(self) -> bool:
        return bool(os.environ.get("WAYLAND_DISPLAY") or os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"))

    async def execute(self, params: Optional[Dict[str, Any]] = None, **kwargs) -> ToolResult:
        merged = dict(params or {})
        merged.update(kwargs)

        action = str(merged.get("action") or "read").strip().lower()
        content = str(merged.get("content") or "")

        is_wayland = self._is_wayland()

        if action == "read":
            # 1. Wayland wl-paste
            if is_wayland and shutil.which("wl-paste"):
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "wl-paste", "-n",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3.0)
                    if proc.returncode == 0:
                        text = stdout.decode("utf-8", errors="replace")
                        return ToolResult(True, output={"content": text, "length": len(text)})
                except Exception:
                    pass

            # 2. X11 xclip
            if shutil.which("xclip"):
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "xclip", "-selection", "clipboard", "-o",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3.0)
                    if proc.returncode == 0:
                        text = stdout.decode("utf-8", errors="replace")
                        return ToolResult(True, output={"content": text, "length": len(text)})
                except Exception:
                    pass

            # 3. X11 xsel
            if shutil.which("xsel"):
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "xsel", "--clipboard", "--output",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3.0)
                    if proc.returncode == 0:
                        text = stdout.decode("utf-8", errors="replace")
                        return ToolResult(True, output={"content": text, "length": len(text)})
                except Exception:
                    pass

            # 4. Fallback to wl-paste even if WAYLAND_DISPLAY not set
            if shutil.which("wl-paste"):
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "wl-paste", "-n",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3.0)
                    if proc.returncode == 0:
                        text = stdout.decode("utf-8", errors="replace")
                        return ToolResult(True, output={"content": text, "length": len(text)})
                except Exception:
                    pass

            return ToolResult(False, error="No supported clipboard utility found or clipboard is unavailable (wl-paste, xclip, xsel).")

        if action == "write":
            # 1. Wayland wl-copy
            if is_wayland and shutil.which("wl-copy"):
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "wl-copy",
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    await asyncio.wait_for(proc.communicate(input=content.encode("utf-8")), timeout=3.0)
                    if proc.returncode == 0:
                        return ToolResult(True, output={"content": content, "length": len(content), "message": "Clipboard updated successfully via wl-copy."})
                except Exception:
                    pass

            # 2. X11 xclip
            if shutil.which("xclip"):
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "xclip", "-selection", "clipboard",
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    await asyncio.wait_for(proc.communicate(input=content.encode("utf-8")), timeout=3.0)
                    if proc.returncode == 0:
                        return ToolResult(True, output={"content": content, "length": len(content), "message": "Clipboard updated successfully via xclip."})
                except Exception:
                    pass

            # 3. X11 xsel
            if shutil.which("xsel"):
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "xsel", "--clipboard", "--input",
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    await asyncio.wait_for(proc.communicate(input=content.encode("utf-8")), timeout=3.0)
                    if proc.returncode == 0:
                        return ToolResult(True, output={"content": content, "length": len(content), "message": "Clipboard updated successfully via xsel."})
                except Exception:
                    pass

            # 4. Fallback to wl-copy even if WAYLAND_DISPLAY not set
            if shutil.which("wl-copy"):
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "wl-copy",
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    await asyncio.wait_for(proc.communicate(input=content.encode("utf-8")), timeout=3.0)
                    if proc.returncode == 0:
                        return ToolResult(True, output={"content": content, "length": len(content), "message": "Clipboard updated successfully via wl-copy."})
                except Exception:
                    pass

            return ToolResult(False, error="No supported clipboard utility found (wl-copy, xclip, xsel).")

        return ToolResult(False, error=f"Unsupported action: '{action}'. Use 'read' or 'write'.")


class QuerySystemJournalTool(BaseTool):
    """Deterministically queries the systemd journal via journalctl."""

    def __init__(self):
        super().__init__(
            tool_id="query_system_journal",
            name="query_system_journal",
            description=(
                "Tier 1 Systemd Journal Query: Retrieve recent systemd log entries for system services or priority levels via journalctl. "
                "Actions: query systemd logs with lines, unit, and priority filters."
            ),
        )
        self.parameters = [
            ToolParameter(
                name="lines",
                type="integer",
                description="Number of log entries to retrieve (default: 30, max: 100).",
                required=False,
                default=30,
            ),
            ToolParameter(
                name="unit",
                type="string",
                description="Systemd unit / service name to query (e.g. 'NetworkManager', 'docker', 'omadora').",
                required=False,
                default="",
            ),
            ToolParameter(
                name="priority",
                type="string",
                description="Log priority filter (e.g. 'err', 'warning', 'info', 'crit', 'alert').",
                required=False,
                default="",
            ),
        ]

    async def execute(self, params: Optional[Dict[str, Any]] = None, **kwargs) -> ToolResult:
        merged = dict(params or {})
        merged.update(kwargs)

        if not shutil.which("journalctl"):
            return ToolResult(False, error="journalctl command not found on PATH.")

        try:
            lines = int(merged.get("lines") or 30)
        except (ValueError, TypeError):
            lines = 30
        lines = min(max(lines, 1), 100)

        unit = str(merged.get("unit") or "").strip()
        priority = str(merged.get("priority") or "").strip()

        cmd = ["journalctl", "--no-pager", "-n", str(lines), "--output=short-iso"]
        if unit:
            cmd.extend(["-u", unit])
        if priority:
            cmd.extend(["-p", priority])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            log_text = stdout.decode("utf-8", errors="replace").strip()
            err_text = stderr.decode("utf-8", errors="replace").strip()

            if proc.returncode != 0 and not log_text:
                return ToolResult(False, error=f"journalctl failed: {err_text or 'Unknown error'}")

            return ToolResult(
                True,
                output={
                    "lines": lines,
                    "unit": unit or None,
                    "priority": priority or None,
                    "logs": log_text,
                    "count": len(log_text.splitlines()) if log_text else 0,
                },
            )
        except asyncio.TimeoutError:
            return ToolResult(False, error="journalctl command timed out after 5.0s.")
        except Exception as e:
            return ToolResult(False, error=f"journalctl execution failed: {e}")


class InspectNetworkTool(BaseTool):
    """Deterministically inspects Linux network interfaces, default gateway, DNS latency, and VPN status."""

    def __init__(self):
        super().__init__(
            tool_id="inspect_network",
            name="inspect_network",
            description=(
                "Tier 1 Linux Network & VPN Diagnostics: Inspect default gateway, network interfaces, "
                "local IP addresses, DNS resolution latency, and Tailscale VPN connection status."
            ),
        )
        self.parameters = [
            ToolParameter(
                name="check_target",
                type="string",
                description="Domain or IP target to measure DNS resolution latency (defaults to '1.1.1.1').",
                required=False,
                default="1.1.1.1",
            ),
        ]

    async def execute(self, params: Optional[Dict[str, Any]] = None, **kwargs) -> ToolResult:
        merged = dict(params or {})
        merged.update(kwargs)

        check_target = str(merged.get("check_target") or "1.1.1.1").strip()
        if not check_target:
            check_target = "1.1.1.1"

        # 1. Default Gateway via `ip route`
        gateway_info: Dict[str, Any] = {"gateway": "unknown", "interface": "unknown", "raw": ""}
        if shutil.which("ip"):
            try:
                proc = await asyncio.create_subprocess_exec(
                    "ip", "route", "show", "default",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)
                route_line = stdout.decode("utf-8", errors="replace").strip()
                gateway_info["raw"] = route_line
                match = re.search(r"default via ([\d\w\.:]+)\s+dev\s+([\w\.\-]+)", route_line)
                if match:
                    gateway_info["gateway"] = match.group(1)
                    gateway_info["interface"] = match.group(2)
            except Exception as e:
                gateway_info["error"] = str(e)

        # 2. Local IP Addresses & Interfaces via `ip -brief address`
        interfaces: List[Dict[str, Any]] = []
        if shutil.which("ip"):
            try:
                proc = await asyncio.create_subprocess_exec(
                    "ip", "-brief", "address",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)
                for line in stdout.decode("utf-8", errors="replace").splitlines():
                    parts = line.split()
                    if len(parts) >= 2:
                        iface = parts[0]
                        state = parts[1]
                        ips = parts[2:]
                        interfaces.append({
                            "interface": iface,
                            "state": state,
                            "ips": ips,
                        })
            except Exception as e:
                logger.warning(f"Error reading ip address: {e}")

        # 3. DNS resolution and latency
        dns_info: Dict[str, Any] = {"target": check_target}
        try:
            t0 = time.perf_counter()
            resolved = await asyncio.to_thread(socket.gethostbyname, check_target)
            lat_ms = round((time.perf_counter() - t0) * 1000, 2)
            dns_info["resolved_ip"] = resolved
            dns_info["latency_ms"] = lat_ms
            dns_info["status"] = "resolved"
        except Exception as exc:
            dns_info["resolved_ip"] = None
            dns_info["latency_ms"] = None
            dns_info["status"] = f"error: {exc}"

        # 4. Tailscale VPN status
        tailscale_info: Dict[str, Any] = {"tailscale_active": False}
        if shutil.which("tailscale"):
            try:
                proc = await asyncio.create_subprocess_exec(
                    "tailscale", "status", "--json",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3.0)
                if proc.returncode == 0:
                    ts_data = json.loads(stdout.decode("utf-8", errors="replace"))
                    backend = ts_data.get("BackendState", "Stopped")
                    tailscale_info["tailscale_active"] = (backend == "Running")
                    tailscale_info["backend_state"] = backend
                    tailscale_info["tailnet_ips"] = ts_data.get("TailscaleIPs", [])
                    self_node = ts_data.get("Self", {}) or {}
                    tailscale_info["node_name"] = self_node.get("HostName", "")
                    peers = ts_data.get("Peer", {}) or {}
                    tailscale_info["peer_count"] = len(peers)
            except Exception as ts_err:
                tailscale_info["error"] = str(ts_err)

        return ToolResult(
            True,
            output={
                "gateway": gateway_info,
                "interfaces": interfaces,
                "dns": dns_info,
                "tailscale": tailscale_info,
            },
        )

