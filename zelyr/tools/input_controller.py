"""Wayland Input Controller using ydotool.

Provides kernel-level mouse and keyboard input simulation on Linux Wayland environments
via ydotool and the ydotoold daemon. Traditional X11 tools (pyautogui, xdotool) are forbidden
under Wayland; this module interfaces directly with Linux uinput via ydotoold.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class YdoToolError(RuntimeError):
    """Base exception for all ydotool-related failures."""


class YdoToolDaemonError(YdoToolError):
    """Raised when the ydotoold background socket or daemon cannot be reached or started."""


class YdoToolPermissionError(YdoToolError):
    """Raised when permissions are missing to access the socket or /dev/uinput."""


# Linux kernel keycodes (from <linux/input-event-codes.h>)
_KEY_NAME_TO_CODE: Dict[str, int] = {
    # Control keys
    "esc": 1,
    "escape": 1,
    "1": 2,
    "2": 3,
    "3": 4,
    "4": 5,
    "5": 6,
    "6": 7,
    "7": 8,
    "8": 9,
    "9": 10,
    "0": 11,
    "minus": 12,
    "-": 12,
    "equal": 13,
    "equals": 13,
    "=": 13,
    "backspace": 14,
    "tab": 15,
    "q": 16,
    "w": 17,
    "e": 18,
    "r": 19,
    "t": 20,
    "y": 21,
    "u": 22,
    "i": 23,
    "o": 24,
    "p": 25,
    "leftbrace": 26,
    "[": 26,
    "rightbrace": 27,
    "]": 27,
    "enter": 28,
    "return": 28,
    "leftctrl": 29,
    "ctrl": 29,
    "control": 29,
    "a": 30,
    "s": 31,
    "d": 32,
    "f": 33,
    "g": 34,
    "h": 35,
    "j": 36,
    "k": 37,
    "l": 38,
    "semicolon": 39,
    ";": 39,
    "apostrophe": 40,
    "'": 40,
    "grave": 41,
    "`": 41,
    "leftshift": 42,
    "shift": 42,
    "backslash": 43,
    "\\": 43,
    "z": 44,
    "x": 45,
    "c": 46,
    "v": 47,
    "b": 48,
    "n": 49,
    "m": 50,
    "comma": 51,
    ",": 51,
    "dot": 52,
    "period": 52,
    ".": 52,
    "slash": 53,
    "/": 53,
    "rightshift": 54,
    "kpasterisk": 55,
    "leftalt": 56,
    "alt": 56,
    "space": 57,
    "spacebar": 57,
    "capslock": 58,
    "f1": 59,
    "f2": 60,
    "f3": 61,
    "f4": 62,
    "f5": 63,
    "f6": 64,
    "f7": 65,
    "f8": 66,
    "f9": 67,
    "f10": 68,
    "numlock": 69,
    "scrolllock": 70,
    "f11": 87,
    "f12": 88,
    "rightctrl": 97,
    "rightalt": 100,
    "linefeed": 101,
    "home": 102,
    "up": 103,
    "pageup": 104,
    "left": 105,
    "right": 106,
    "end": 107,
    "down": 108,
    "pagedown": 109,
    "insert": 110,
    "delete": 111,
    "del": 111,
    "leftmeta": 125,
    "super": 125,
    "meta": 125,
    "cmd": 125,
    "win": 125,
    "windows": 125,
    "rightmeta": 126,
}

_MOUSE_BUTTONS: Dict[str, str] = {
    "left": "0xC0",
    "right": "0xC1",
    "middle": "0xC2",
    "side": "0xC3",
    "extra": "0xC4",
    "forward": "0xC5",
    "back": "0xC6",
    "task": "0xC7",
}


class WaylandInputController:
    """Wraps ydotool to execute mouse and keyboard input on Wayland."""

    def __init__(
        self,
        socket_path: Optional[str] = None,
        auto_start_daemon: bool = True,
        auto_check: bool = True,
    ) -> None:
        self.ydotool_bin = shutil.which("ydotool")
        if not self.ydotool_bin:
            raise YdoToolError(
                "Binary 'ydotool' not found in system PATH. "
                "Install ydotool package (e.g., via pacman, dnf, or apt)."
            )

        self.custom_socket = socket_path
        self.socket_path = self._resolve_socket_path(socket_path)
        self.auto_start_daemon = auto_start_daemon

        if auto_check:
            self.ensure_daemon_running()

    def _resolve_socket_path(self, explicit_path: Optional[str] = None) -> Optional[str]:
        """Resolves the best socket path based on environment or well-known locations."""
        if explicit_path:
            return explicit_path

        env_sock = os.environ.get("YDOTOOL_SOCKET")
        if env_sock and Path(env_sock).exists():
            return env_sock

        candidate_paths = [
            "/run/ydotool.socket",
            f"/run/user/{os.getuid()}/.ydotool_socket",
            "/tmp/.ydotool_socket",
        ]
        for candidate in candidate_paths:
            if Path(candidate).exists():
                return candidate

        return env_sock or candidate_paths[0]

    def _get_exec_env(self) -> Dict[str, str]:
        """Returns the process environment with YDOTOOL_SOCKET set."""
        env = os.environ.copy()
        if self.socket_path:
            env["YDOTOOL_SOCKET"] = self.socket_path
        return env

    def is_daemon_running(self) -> bool:
        """Checks if ydotoold is running and accepting commands."""
        # 1. Check if process is running
        proc_running = False
        try:
            import psutil
            for proc in psutil.process_iter(["name"]):
                if proc.info["name"] and "ydotoold" in proc.info["name"]:
                    proc_running = True
                    break
        except Exception:
            # Fallback to pgrep
            res = subprocess.run(["pgrep", "-x", "ydotoold"], capture_output=True)
            if res.returncode == 0:
                proc_running = True

        if not proc_running:
            return False

        # 2. Check if a socket exists
        if self.socket_path and not Path(self.socket_path).exists():
            # Try to discover another existing socket
            discovered = self._resolve_socket_path(None)
            if discovered and Path(discovered).exists():
                self.socket_path = discovered
            else:
                return False

        # 3. Test responsive communication with a fast probe
        try:
            res = subprocess.run(
                [self.ydotool_bin, "mousemove", "-x", "0", "-y", "0"],
                capture_output=True,
                text=True,
                env=self._get_exec_env(),
                timeout=1.5,
            )
            return res.returncode == 0
        except Exception:
            return False

    def start_daemon(self) -> None:
        """Attempts to start the ydotoold daemon via systemd user service or direct process."""
        logger.info("Attempting to start ydotoold daemon...")

        started = False
        # 1. Attempt systemctl --user start ydotool
        try:
            res = subprocess.run(
                ["systemctl", "--user", "start", "ydotool"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0:
                started = True
        except Exception as e:
            logger.debug("Failed systemctl --user start ydotool: %s", e)

        # 2. Also try systemctl --user start ydotoold if needed
        if not started:
            try:
                res = subprocess.run(
                    ["systemctl", "--user", "start", "ydotoold"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if res.returncode == 0:
                    started = True
            except Exception as e:
                logger.debug("Failed systemctl --user start ydotoold: %s", e)

        # 3. Fallback: launch ydotoold directly in background if user service not configured
        if not started:
            ydotoold_bin = shutil.which("ydotoold")
            if ydotoold_bin:
                try:
                    user_sock = f"/run/user/{os.getuid()}/.ydotool_socket"
                    subprocess.Popen(
                        [ydotoold_bin, f"--socket-path={user_sock}", "--socket-perm=0600"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    self.socket_path = user_sock
                except Exception as e:
                    logger.debug("Failed direct ydotoold invocation: %s", e)

        # Poll for daemon to become ready
        deadline = time.time() + 3.0
        while time.time() < deadline:
            time.sleep(0.15)
            # Re-resolve socket path if needed
            self.socket_path = self._resolve_socket_path(self.custom_socket)
            if self.is_daemon_running():
                logger.info("ydotoold daemon successfully verified running.")
                return

    def ensure_daemon_running(self) -> None:
        """Ensures that ydotoold is running and permissions are valid; raises descriptive exceptions."""
        if not self.is_daemon_running():
            if self.auto_start_daemon:
                self.start_daemon()

        # Re-resolve socket path
        self.socket_path = self._resolve_socket_path(self.custom_socket)

        # Run verification probe
        try:
            res = subprocess.run(
                [self.ydotool_bin, "mousemove", "-x", "0", "-y", "0"],
                capture_output=True,
                text=True,
                env=self._get_exec_env(),
                timeout=2.0,
            )
        except subprocess.TimeoutExpired:
            raise YdoToolDaemonError(
                f"ydotool timed out attempting to communicate with daemon socket '{self.socket_path}'."
            )
        except Exception as e:
            raise YdoToolError(f"Unexpected error executing ydotool probe: {e}") from e

        if res.returncode != 0:
            err = f"{res.stdout or ''}\n{res.stderr or ''}".strip()
            err_lower = err.lower()

            if "permission denied" in err_lower or "eacces" in err_lower:
                raise YdoToolPermissionError(
                    f"Permission denied accessing ydotool socket '{self.socket_path}' or /dev/uinput. "
                    "Ensure your user has read/write permissions to /dev/uinput (e.g. udev rule or input group)."
                )

            if (
                "failed to connect socket" in err_lower
                or "no such file or directory" in err_lower
                or "check if ydotoold is running" in err_lower
            ):
                raise YdoToolDaemonError(
                    f"Failed to connect to ydotool daemon socket '{self.socket_path}'. "
                    "ydotoold is not running or socket is missing. "
                    "Run 'systemctl --user start ydotool' or 'ydotoold' as background service."
                )

            raise YdoToolError(
                f"ydotool failed with exit code {res.returncode}: {err}"
            )

    def _execute(self, args: List[str], timeout: float = 5.0) -> subprocess.CompletedProcess:
        """Executes a ydotool command with error classification."""
        cmd = [self.ydotool_bin] + args
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=self._get_exec_env(),
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise YdoToolError(f"Command {' '.join(cmd)} timed out after {timeout}s") from e

        if res.returncode != 0:
            err = f"{res.stdout or ''}\n{res.stderr or ''}".strip()
            err_lower = err.lower()
            if "permission denied" in err_lower or "eacces" in err_lower:
                raise YdoToolPermissionError(
                    f"Permission denied executing '{' '.join(cmd)}': {err}"
                )
            if (
                "failed to connect socket" in err_lower
                or "no such file or directory" in err_lower
                or "check if ydotoold is running" in err_lower
            ):
                raise YdoToolDaemonError(
                    f"ydotool socket error executing '{' '.join(cmd)}': {err}"
                )
            raise YdoToolError(
                f"ydotool command '{' '.join(cmd)}' failed (exit code {res.returncode}): {err}"
            )

        return res

    def mouse_move(self, x: Union[int, float], y: Union[int, float]) -> None:
        """Moves the mouse cursor to absolute (x, y) coordinates."""
        int_x = int(round(x))
        int_y = int(round(y))
        self._execute(["mousemove", "-a", "-x", str(int_x), "-y", str(int_y)])

    def mouse_click(self, button: str = "left", clicks: int = 1) -> None:
        """Clicks a mouse button (e.g. 'left', 'right', 'middle') one or more times."""
        btn_lower = button.strip().lower()
        hex_code = _MOUSE_BUTTONS.get(btn_lower)

        if not hex_code:
            # Check if button was passed directly as hex (e.g. 0xC0)
            if re.match(r"^0x[0-9a-fA-F]+$", button):
                hex_code = button
            else:
                raise ValueError(
                    f"Unsupported mouse button '{button}'. Supported: {list(_MOUSE_BUTTONS.keys())}"
                )

        args = ["click"]
        if clicks > 1:
            args.extend(["-r", str(clicks)])
        args.append(hex_code)

        self._execute(args)

    def keyboard_type(self, text: str) -> None:
        """Types the provided text using ydotool type."""
        if not text:
            return
        # -d: delay between key events (ms)
        # -H: hold time per key (ms)
        # --: prevents text starting with dashes from parsing as flags
        self._execute(["type", "-d", "12", "-H", "12", "--", text])

    def keyboard_press(self, key: str) -> None:
        """Presses a single key or key combination (e.g., 'enter', 'tab', 'ctrl+c', 'super')."""
        key_str = key.strip()
        if not key_str:
            raise ValueError("Key string cannot be empty")

        # Parse potential combinations (e.g. 'ctrl+shift+t' or 'ctrl+c')
        parts = [p.strip().lower() for p in re.split(r"[+\-]", key_str) if p.strip()]

        keycodes: List[int] = []
        for part in parts:
            if part in _KEY_NAME_TO_CODE:
                keycodes.append(_KEY_NAME_TO_CODE[part])
            elif len(part) == 1 and part in _KEY_NAME_TO_CODE:
                keycodes.append(_KEY_NAME_TO_CODE[part])
            else:
                # Try raw integer keycode
                try:
                    code = int(part)
                    keycodes.append(code)
                except ValueError:
                    raise ValueError(
                        f"Unknown key '{part}' in combo '{key}'. Available keys: "
                        f"{', '.join(sorted(list(_KEY_NAME_TO_CODE.keys())[:25]))}..."
                    )

        if not keycodes:
            raise ValueError(f"Could not parse valid keycodes from '{key}'")

        # Build down/up sequence
        # Example for combo k1+k2: k1:1 k2:1 k2:0 k1:0
        events: List[str] = []
        for code in keycodes:
            events.append(f"{code}:1")

        for code in reversed(keycodes):
            events.append(f"{code}:0")

        self._execute(["key"] + events)


# Convenience alias
InputController = WaylandInputController

# Singleton on-demand access
_default_instance: Optional[WaylandInputController] = None


def get_input_controller() -> WaylandInputController:
    """Returns a shared default WaylandInputController instance."""
    global _default_instance
    if _default_instance is None:
        _default_instance = WaylandInputController()
    return _default_instance


# Module-level convenience functions
def mouse_move(x: Union[int, float], y: Union[int, float]) -> None:
    get_input_controller().mouse_move(x, y)


def mouse_click(button: str = "left", clicks: int = 1) -> None:
    get_input_controller().mouse_click(button=button, clicks=clicks)


def keyboard_type(text: str) -> None:
    get_input_controller().keyboard_type(text)


def keyboard_press(key: str) -> None:
    get_input_controller().keyboard_press(key)
