"""Disposable OS Sandboxing Engine for Zelyr v2.5.

Intercepts shell commands dispatched by the orchestrator and routes
dangerous (state-changing) operations into disposable Docker containers
or lightweight Linux bubblewrap (bwrap) namespaces.

This is architecturally distinct from ``zelyr/plugins/sandbox.py`` (which
sandboxes third-party *plugin* code via multiprocessing) — this module
sandboxes **OS-level shell commands** that Zelyr's tool-calling loop
executes on behalf of the user.

Security tiers (auto-detected, highest-priority first):
1. Docker — ephemeral ``--rm`` containers with memory limits & no network.
2. Bubblewrap (bwrap) — Linux user-namespace isolation with read-only
   root mounts and workspace bind-mounts.
3. Bare-metal — fallback; logs a warning and runs natively.
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SandboxBackend(str, Enum):
    """Available isolation backends."""
    DOCKER = "docker"
    BWRAP = "bwrap"
    NONE = "none"


class SandboxMode(str, Enum):
    """Execution policy for the sandbox runtime."""
    STRICT = "strict"   # Sandbox ALL commands (even read-only)
    AUTO = "auto"       # Sandbox state-changing; passthrough read-only
    OFF = "off"         # Disabled — everything runs on the host


# ---------------------------------------------------------------------------
# Read-only command classifier
# ---------------------------------------------------------------------------

# Commands that are safe reconnaissance and never modify system state.
_READONLY_COMMANDS: Set[str] = frozenset({
    "ls", "cat", "head", "tail", "wc", "find", "grep", "egrep", "fgrep",
    "ps", "uname", "whoami", "hostname", "id", "env", "printenv",
    "echo", "pwd", "which", "type", "file", "stat", "df", "du",
    "free", "date", "uptime", "lsblk", "lscpu", "lsusb", "lspci",
    "ip", "ss", "netstat", "dig", "nslookup", "ping",
    "tree", "realpath", "basename", "dirname", "test", "true", "false",
    "tty", "groups", "locale", "timedatectl",
})


def _extract_base_command(command: str) -> str:
    """Extract the first token (base command) from a shell command string.

    Strips leading env vars, ``sudo``, ``nice``, ``time``, etc.
    Only skips wrapper commands when additional tokens follow them.
    """
    tokens = shlex.split(command, posix=True)
    # Skip common prefix wrappers (only when they precede another command)
    skip = {"sudo", "nice", "time", "env", "strace", "ltrace", "nohup"}
    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        # Skip env-var assignments (e.g. FOO=bar)
        if "=" in tok and not tok.startswith("-"):
            idx += 1
            continue
        # Only skip wrapper commands if more tokens follow
        if tok in skip and idx + 1 < len(tokens):
            idx += 1
            continue
        return os.path.basename(tok)
    return ""


# ---------------------------------------------------------------------------
# SandboxRuntime
# ---------------------------------------------------------------------------

class SandboxRuntime:
    """OS-level command sandbox with Docker / bwrap / bare-metal backends."""

    def __init__(
        self,
        event_bus: Any = None,
        mode: SandboxMode = SandboxMode.AUTO,
        docker_image: str = "python:3.12-slim",
        memory_limit: str = "512m",
    ) -> None:
        self.event_bus = event_bus
        self.mode = mode
        self.docker_image = docker_image
        self.memory_limit = memory_limit
        self.backend = self.detect_backend()

        if self.backend == SandboxBackend.NONE:
            logger.warning(
                "[SandboxRuntime] Neither Docker nor bubblewrap detected. "
                "Commands will execute on bare metal. Install docker or "
                "bubblewrap for OS-level isolation."
            )
        else:
            logger.info(
                f"[SandboxRuntime] Initialized with backend={self.backend.value}, "
                f"mode={self.mode.value}"
            )

    # ------------------------------------------------------------------
    # Backend detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_backend() -> SandboxBackend:
        """Probe the host for the best available isolation runtime."""
        import platform
        is_linux = platform.system() == "Linux"
        
        if shutil.which("docker"):
            return SandboxBackend.DOCKER
            
        if is_linux and shutil.which("bwrap"):
            return SandboxBackend.BWRAP
            
        return SandboxBackend.NONE

    # ------------------------------------------------------------------
    # Command classification
    # ------------------------------------------------------------------

    @staticmethod
    def is_readonly_command(command: str) -> bool:
        """Return True if *command* is safe read-only reconnaissance."""
        try:
            base = _extract_base_command(command)
        except ValueError:
            # shlex parse failure → treat as dangerous
            return False
        return base in _READONLY_COMMANDS

    # ------------------------------------------------------------------
    # Sandbox execution
    # ------------------------------------------------------------------

    def execute_in_sandbox(
        self,
        command: str,
        cwd: str = "/tmp",
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """Execute *command* inside a disposable sandbox.

        Returns a structured telemetry dict::

            {
                "stdout": "...",
                "stderr": "...",
                "exit_code": 0,
                "sandbox_backend": "docker",
                "duration": 1.23,
                "sandboxed": True,
            }
        """
        self._emit_event(command, "running")
        t0 = time.monotonic()

        try:
            if self.backend == SandboxBackend.DOCKER:
                result = self._exec_docker(command, cwd, timeout)
            elif self.backend == SandboxBackend.BWRAP:
                result = self._exec_bwrap(command, cwd, timeout)
            else:
                result = self._exec_native(command, cwd, timeout)
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - t0
            result = {
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "exit_code": -1,
                "sandbox_backend": self.backend.value,
                "duration": round(duration, 3),
                "sandboxed": self.backend != SandboxBackend.NONE,
            }

        result["duration"] = round(time.monotonic() - t0, 3)
        self._emit_event(command, "completed", result.get("exit_code", -1), result["duration"])
        return result

    def _exec_docker(self, command: str, cwd: str, timeout: int) -> Dict[str, Any]:
        """Run *command* inside an ephemeral Docker container."""
        docker_args = [
            "docker", "run", "--rm",
            f"--memory={self.memory_limit}",
            "--network=none",
            "-v", f"{cwd}:{cwd}",
            "-w", cwd,
            "-v", "/:/host:ro",
            self.docker_image,
            "bash", "-c", command,
        ]
        proc = subprocess.run(
            docker_args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
            "sandbox_backend": SandboxBackend.DOCKER.value,
            "sandboxed": True,
        }

    def _exec_bwrap(self, command: str, cwd: str, timeout: int) -> Dict[str, Any]:
        """Run *command* inside a bubblewrap namespace."""
        bwrap_args = [
            "bwrap",
            "--ro-bind", "/", "/",
            "--bind", cwd, cwd,
            "--dev", "/dev",
            "--proc", "/proc",
            "--unshare-all",
            "--die-with-parent",
            "bash", "-c", command,
        ]
        proc = subprocess.run(
            bwrap_args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
            "sandbox_backend": SandboxBackend.BWRAP.value,
            "sandboxed": True,
        }

    def _exec_native(self, command: str, cwd: str, timeout: int) -> Dict[str, Any]:
        """Fallback: run directly on the host."""
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
            "sandbox_backend": SandboxBackend.NONE.value,
            "sandboxed": False,
        }

    # ------------------------------------------------------------------
    # Routing logic (called by shell tools)
    # ------------------------------------------------------------------

    def should_sandbox(self, command: str) -> bool:
        """Decide whether *command* must be routed through the sandbox.

        Respects the current ``self.mode`` setting.
        """
        if self.mode == SandboxMode.OFF:
            return False
        if self.mode == SandboxMode.STRICT:
            return True
        # AUTO mode: sandbox unless command is read-only
        return not self.is_readonly_command(command)

    # ------------------------------------------------------------------
    # Status / introspection
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return sandbox runtime status for CLI reporting."""
        return {
            "backend": self.backend.value.upper(),
            "mode": self.mode.value,
            "docker_image": self.docker_image if self.backend == SandboxBackend.DOCKER else "N/A",
            "memory_limit": self.memory_limit if self.backend == SandboxBackend.DOCKER else "N/A",
            "available": self.backend != SandboxBackend.NONE,
        }

    # ------------------------------------------------------------------
    # EventBus helpers
    # ------------------------------------------------------------------

    def _emit_event(
        self,
        command: str,
        status: str,
        exit_code: int = 0,
        duration: float = 0.0,
    ) -> None:
        """Emit sandbox telemetry to the EventBus (if connected)."""
        if not self.event_bus:
            return
        try:
            from zelyr.core.events import Event
            event_type = "sandbox.exec" if status == "running" else "sandbox.exec.complete"
            self.event_bus.publish(Event(
                event_type=event_type,
                source="SandboxRuntime",
                data={
                    "backend": self.backend.value,
                    "command": command[:200],  # Truncate for safety
                    "status": status,
                    "exit_code": exit_code,
                    "duration": duration,
                },
            ))
        except Exception:
            pass  # Never let telemetry crash the execution path
