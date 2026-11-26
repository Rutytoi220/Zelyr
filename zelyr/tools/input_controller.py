"""Linux/Wayland input controller supporting kernel /dev/uinput, Hyprland IPC cursor, wtype, and ydotool.

Provides real mouse and keyboard actuation on Linux Wayland/Hyprland (with
ydotool/wtype/xdotool/X11 fallbacks where present), strict monitor coordinate
validation, key combo validation, bounded timeouts, and safe cleanup of
virtual input devices on close, error, or cancellation.
"""

from __future__ import annotations

import atexit
import ctypes
import fcntl
import json
import logging
import math
import os
import re
import shutil
import signal
import struct
import subprocess
import time
import weakref
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class InputControlError(RuntimeError):
    """Base structured exception for Linux desktop input controller errors."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str = "INPUT_FAILED",
        remedy_hint: str = "",
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.remedy_hint = remedy_hint


class YdoToolError(InputControlError):
    """Base exception for backend / ydotool-related failures."""

    def __init__(self, message: str, *, error_type: str = "BACKEND_UNAVAILABLE", remedy_hint: str = "") -> None:
        super().__init__(message, error_type=error_type, remedy_hint=remedy_hint)


class YdoToolDaemonError(YdoToolError):
    """Raised when the ydotoold background socket or daemon cannot be reached or started."""


class YdoToolPermissionError(YdoToolError):
    """Raised when permissions are missing to access /dev/uinput or the input socket."""

    def __init__(self, message: str, *, remedy_hint: str = "") -> None:
        super().__init__(
            message,
            error_type="PERMISSION_DENIED",
            remedy_hint=remedy_hint or "Ensure your user has read/write access to /dev/uinput.",
        )


class DesktopUnavailableError(InputControlError):
    """Raised when no active desktop display session (Wayland/X11) is present."""

    def __init__(self, message: str, *, remedy_hint: str = "") -> None:
        super().__init__(
            message,
            error_type="DESKTOP_UNAVAILABLE",
            remedy_hint=remedy_hint or "Verify WAYLAND_DISPLAY, DISPLAY, or HYPRLAND_INSTANCE_SIGNATURE.",
        )


class BackendUnavailableError(InputControlError):
    """Raised when no usable desktop input backend exists or is accessible."""

    def __init__(self, message: str, *, remedy_hint: str = "") -> None:
        super().__init__(
            message,
            error_type="BACKEND_UNAVAILABLE",
            remedy_hint=remedy_hint or "Verify /dev/uinput, wtype, or ydotool availability on this host.",
        )


class InvalidCoordinatesError(InputControlError):
    """Raised when supplied coordinates are non-numeric or outside active monitor bounds."""

    def __init__(self, message: str, *, remedy_hint: str = "") -> None:
        super().__init__(
            message,
            error_type="INVALID_COORDINATES",
            remedy_hint=remedy_hint or "Inspect monitor geometry with get_screen_size or capture_screen first.",
        )


class InvalidInputKeyError(InputControlError):
    """Raised when key names, modifier combinations, buttons, or text input are invalid."""

    def __init__(self, message: str, *, remedy_hint: str = "") -> None:
        super().__init__(
            message,
            error_type="INVALID_INPUT",
            remedy_hint=remedy_hint or "Use standard key names (e.g. 'enter', 'tab', 'ctrl+c', 'alt+tab').",
        )


class InputTimeoutError(InputControlError):
    """Raised when an external input utility exceeds its execution timeout."""

    def __init__(self, message: str, *, remedy_hint: str = "") -> None:
        super().__init__(
            message,
            error_type="INPUT_TIMEOUT",
            remedy_hint=remedy_hint or "Check compositor responsiveness and retry with smaller text/steps.",
        )


class InputFailedError(InputControlError):
    """Raised when input dispatch fails mid-execution."""

    def __init__(self, message: str, *, remedy_hint: str = "") -> None:
        super().__init__(
            message,
            error_type="INPUT_FAILED",
            remedy_hint=remedy_hint or "Verify window focus and input backend state before retrying.",
        )


# Linux kernel keycodes (from <linux/input-event-codes.h>)
_KEY_NAME_TO_CODE: Dict[str, int] = {
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
    "kpenter": 96,
    "leftctrl": 29,
    "ctrl": 29,
    "control": 29,
    "ctl": 29,
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
    "altgr": 100,
    "linefeed": 101,
    "home": 102,
    "up": 103,
    "pageup": 104,
    "pgup": 104,
    "left": 105,
    "right": 106,
    "end": 107,
    "down": 108,
    "pagedown": 109,
    "pgdn": 109,
    "insert": 110,
    "ins": 110,
    "delete": 111,
    "del": 111,
    "leftmeta": 125,
    "super": 125,
    "super_l": 125,
    "meta": 125,
    "cmd": 125,
    "win": 125,
    "windows": 125,
    "rightmeta": 126,
    "super_r": 126,
    "menu": 139,
}

# Printable US-ASCII -> (keycode, needs_shift) for kernel /dev/uinput typing fallback
_ASCII_KEYMAP: Dict[str, Tuple[int, bool]] = {
    "a": (30, False), "A": (30, True),
    "b": (48, False), "B": (48, True),
    "c": (46, False), "C": (46, True),
    "d": (32, False), "D": (32, True),
    "e": (18, False), "E": (18, True),
    "f": (33, False), "F": (33, True),
    "g": (34, False), "G": (34, True),
    "h": (35, False), "H": (35, True),
    "i": (23, False), "I": (23, True),
    "j": (36, False), "J": (36, True),
    "k": (37, False), "K": (37, True),
    "l": (38, False), "L": (38, True),
    "m": (50, False), "M": (50, True),
    "n": (49, False), "N": (49, True),
    "o": (24, False), "O": (24, True),
    "p": (25, False), "P": (25, True),
    "q": (16, False), "Q": (16, True),
    "r": (19, False), "R": (19, True),
    "s": (31, False), "S": (31, True),
    "t": (20, False), "T": (20, True),
    "u": (22, False), "U": (22, True),
    "v": (47, False), "V": (47, True),
    "w": (17, False), "W": (17, True),
    "x": (45, False), "X": (45, True),
    "y": (21, False), "Y": (21, True),
    "z": (44, False), "Z": (44, True),
    "1": (2, False), "!": (2, True),
    "2": (3, False), "@": (3, True),
    "3": (4, False), "#": (4, True),
    "4": (5, False), "$": (5, True),
    "5": (6, False), "%": (6, True),
    "6": (7, False), "^": (7, True),
    "7": (8, False), "&": (8, True),
    "8": (9, False), "*": (9, True),
    "9": (10, False), "(": (10, True),
    "0": (11, False), ")": (11, True),
    "-": (12, False), "_": (12, True),
    "=": (13, False), "+": (13, True),
    "\t": (15, False),
    "[": (26, False), "{": (26, True),
    "]": (27, False), "}": (27, True),
    "\n": (28, False), "\r": (28, False),
    ";": (39, False), ":": (39, True),
    "'": (40, False), '"': (40, True),
    "`": (41, False), "~": (41, True),
    "\\": (43, False), "|": (43, True),
    ",": (51, False), "<": (51, True),
    ".": (52, False), ">": (52, True),
    "/": (53, False), "?": (53, True),
    " ": (57, False),
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

_KERNEL_MOUSE_BUTTONS: Dict[str, int] = {
    "left": 0x110,    # BTN_LEFT
    "right": 0x111,   # BTN_RIGHT
    "middle": 0x112,  # BTN_MIDDLE
    "side": 0x113,    # BTN_SIDE
    "extra": 0x114,   # BTN_EXTRA
}

_WTYPE_MODIFIERS = {
    "ctrl": "ctrl",
    "leftctrl": "ctrl",
    "rightctrl": "ctrl",
    "control": "ctrl",
    "ctl": "ctrl",
    "alt": "alt",
    "leftalt": "alt",
    "rightalt": "alt",
    "shift": "shift",
    "leftshift": "shift",
    "rightshift": "shift",
    "super": "logo",
    "leftmeta": "logo",
    "rightmeta": "logo",
    "meta": "logo",
    "win": "logo",
    "windows": "logo",
    "cmd": "logo",
}

_WTYPE_SPECIAL_KEYS = {
    "enter": "Return",
    "return": "Return",
    "esc": "Escape",
    "escape": "Escape",
    "tab": "Tab",
    "backspace": "BackSpace",
    "space": "space",
    "spacebar": "space",
    "delete": "Delete",
    "del": "Delete",
    "insert": "Insert",
    "ins": "Insert",
    "home": "Home",
    "end": "End",
    "up": "Up",
    "down": "Down",
    "left": "Left",
    "right": "Right",
    "pageup": "Prior",
    "pgup": "Prior",
    "pagedown": "Next",
    "pgdn": "Next",
    **{f"f{i}": f"F{i}" for i in range(1, 13)},
}

# Linux uinput constants / ioctl encoding
_EV_SYN = 0x00
_EV_KEY = 0x01
_EV_REL = 0x02
_SYN_REPORT = 0
_REL_X = 0x00
_REL_Y = 0x01
_REL_HWHEEL = 0x06
_REL_WHEEL = 0x08

_UINPUT_BASE = ord("U")
_UI_DEV_CREATE = (_UINPUT_BASE << 8) | 1
_UI_DEV_DESTROY = (_UINPUT_BASE << 8) | 2
_UI_SET_EVBIT = (1 << 30) | (4 << 16) | (_UINPUT_BASE << 8) | 100
_UI_SET_KEYBIT = (1 << 30) | (4 << 16) | (_UINPUT_BASE << 8) | 101
_UI_SET_RELBIT = (1 << 30) | (4 << 16) | (_UINPUT_BASE << 8) | 102

_INPUT_EVENT_STRUCT = struct.Struct("@llHHi")


class _UInputSetup(ctypes.Structure):
    _fields_ = [
        ("id_bustype", ctypes.c_uint16),
        ("id_vendor", ctypes.c_uint16),
        ("id_product", ctypes.c_uint16),
        ("id_version", ctypes.c_uint16),
        ("name", ctypes.c_char * 80),
        ("ff_effects_max", ctypes.c_uint32),
    ]


_UI_DEV_SETUP = (1 << 30) | (ctypes.sizeof(_UInputSetup) << 16) | (_UINPUT_BASE << 8) | 3


def _pack_event(ev_type: int, code: int, value: int) -> bytes:
    """Pack a Linux struct input_event binary frame."""
    now = time.time()
    sec = int(now)
    usec = int((now - sec) * 1_000_000)
    return _INPUT_EVENT_STRUCT.pack(sec, usec, ev_type, code, value)


class _UInputBackend:
    """Direct kernel /dev/uinput virtual mouse and keyboard backend without external daemons."""

    def __init__(self, device_path: str = "/dev/uinput", settle_s: float = 0.28) -> None:
        self.device_path = device_path
        self.settle_s = settle_s
        self._pointer_fd: Optional[int] = None
        self._keyboard_fd: Optional[int] = None
        self._active_buttons: set[int] = set()
        self._active_keys: set[int] = set()
        self._closed = False
        self._weak_cleanup = weakref.finalize(self, _finalize_uinput_fds, self)

    @classmethod
    def probe_access(cls, device_path: str = "/dev/uinput") -> Tuple[bool, str]:
        """Check whether /dev/uinput exists and is writable without creating a device."""
        if not os.path.exists(device_path):
            return False, f"'{device_path}' does not exist."
        try:
            fd = os.open(device_path, os.O_WRONLY | os.O_NONBLOCK)
            os.close(fd)
            return True, ""
        except PermissionError as exc:
            return False, f"Permission denied opening '{device_path}': {exc}"
        except OSError as exc:
            return False, f"Cannot access '{device_path}': {exc}"

    def _open_fd(self) -> int:
        if not os.path.exists(self.device_path):
            raise BackendUnavailableError(
                f"Input device node '{self.device_path}' does not exist.",
                remedy_hint="Load the 'uinput' kernel module or install ydotool/wtype.",
            )
        try:
            return os.open(self.device_path, os.O_WRONLY | os.O_NONBLOCK)
        except PermissionError as exc:
            raise YdoToolPermissionError(
                f"Permission denied opening '{self.device_path}': {exc}",
                remedy_hint="Grant rw access to /dev/uinput via input group ACL or udev rule.",
            ) from exc
        except OSError as exc:
            raise InputFailedError(f"Failed to open '{self.device_path}': {exc}") from exc

    def _ensure_pointer(self) -> int:
        if self._pointer_fd is not None:
            return self._pointer_fd
        fd = self._open_fd()
        try:
            fcntl.ioctl(fd, _UI_SET_EVBIT, _EV_SYN)
            fcntl.ioctl(fd, _UI_SET_EVBIT, _EV_KEY)
            for btn_code in _KERNEL_MOUSE_BUTTONS.values():
                fcntl.ioctl(fd, _UI_SET_KEYBIT, btn_code)
            fcntl.ioctl(fd, _UI_SET_EVBIT, _EV_REL)
            for rel_code in (_REL_X, _REL_Y, _REL_WHEEL, _REL_HWHEEL):
                fcntl.ioctl(fd, _UI_SET_RELBIT, rel_code)
            setup = _UInputSetup(
                id_bustype=0x06,
                id_vendor=0x1D6B,
                id_product=0x5A01,
                id_version=1,
                name=b"zelyr-virtual-pointer",
                ff_effects_max=0,
            )
            fcntl.ioctl(fd, _UI_DEV_SETUP, setup)
            fcntl.ioctl(fd, _UI_DEV_CREATE)
            time.sleep(self.settle_s)
            self._pointer_fd = fd
            self._closed = False
            return fd
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise

    def _ensure_keyboard(self) -> int:
        if self._keyboard_fd is not None:
            return self._keyboard_fd
        fd = self._open_fd()
        try:
            fcntl.ioctl(fd, _UI_SET_EVBIT, _EV_SYN)
            fcntl.ioctl(fd, _UI_SET_EVBIT, _EV_KEY)
            for code in range(1, 249):
                fcntl.ioctl(fd, _UI_SET_KEYBIT, code)
            setup = _UInputSetup(
                id_bustype=0x06,
                id_vendor=0x1D6B,
                id_product=0x5A02,
                id_version=1,
                name=b"zelyr-virtual-keyboard",
                ff_effects_max=0,
            )
            fcntl.ioctl(fd, _UI_DEV_SETUP, setup)
            fcntl.ioctl(fd, _UI_DEV_CREATE)
            time.sleep(self.settle_s)
            self._keyboard_fd = fd
            self._closed = False
            return fd
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise

    @staticmethod
    def _emit(fd: int, events: List[Tuple[int, int, int]]) -> None:
        buf = bytearray()
        for ev_type, code, value in events:
            buf.extend(_pack_event(ev_type, code, value))
        buf.extend(_pack_event(_EV_SYN, _SYN_REPORT, 0))
        view = memoryview(buf)
        sent = 0
        while sent < len(view):
            written = os.write(fd, view[sent:])
            if written <= 0:
                raise InputFailedError("Short write emitting event to /dev/uinput.")
            sent += written

    def button_down(self, btn_code: int) -> None:
        fd = self._ensure_pointer()
        self._active_buttons.add(btn_code)
        self._emit(fd, [(_EV_KEY, btn_code, 1)])

    def button_up(self, btn_code: int) -> None:
        fd = self._ensure_pointer()
        try:
            self._emit(fd, [(_EV_KEY, btn_code, 0)])
        finally:
            self._active_buttons.discard(btn_code)

    def click(self, btn_code: int, clicks: int = 1) -> None:
        fd = self._ensure_pointer()
        for idx in range(clicks):
            try:
                self._active_buttons.add(btn_code)
                self._emit(fd, [(_EV_KEY, btn_code, 1)])
                time.sleep(0.025)
            finally:
                self._emit(fd, [(_EV_KEY, btn_code, 0)])
                self._active_buttons.discard(btn_code)
            if idx + 1 < clicks:
                time.sleep(0.06)

    def scroll(self, delta_y: int = 0, delta_x: int = 0) -> None:
        fd = self._ensure_pointer()
        events: List[Tuple[int, int, int]] = []
        if delta_y:
            events.append((_EV_REL, _REL_WHEEL, int(delta_y)))
        if delta_x:
            events.append((_EV_REL, _REL_HWHEEL, int(delta_x)))
        if events:
            self._emit(fd, events)

    def press_keycodes(self, keycodes: List[int]) -> None:
        fd = self._ensure_keyboard()
        pressed: List[int] = []
        try:
            for code in keycodes:
                self._active_keys.add(code)
                pressed.append(code)
                self._emit(fd, [(_EV_KEY, code, 1)])
                time.sleep(0.01)
            time.sleep(0.02)
        finally:
            for code in reversed(pressed):
                try:
                    self._emit(fd, [(_EV_KEY, code, 0)])
                except OSError:
                    pass
                self._active_keys.discard(code)

    def type_ascii(self, text: str) -> None:
        for ch in text:
            mapping = _ASCII_KEYMAP.get(ch)
            if mapping is None:
                raise InvalidInputKeyError(
                    f"Character {ch!r} cannot be typed via raw /dev/uinput ASCII fallback (requires wtype for non-ASCII)."
                )
            keycode, needs_shift = mapping
            seq = [42, keycode] if needs_shift else [keycode]
            self.press_keycodes(seq)
            time.sleep(0.008)

    def release_all(self) -> None:
        """Release any held mouse buttons or keyboard keys."""
        if self._pointer_fd is not None and self._active_buttons:
            for code in list(self._active_buttons):
                try:
                    self._emit(self._pointer_fd, [(_EV_KEY, code, 0)])
                except Exception:
                    pass
            self._active_buttons.clear()
        if self._keyboard_fd is not None and self._active_keys:
            for code in list(self._active_keys):
                try:
                    self._emit(self._keyboard_fd, [(_EV_KEY, code, 0)])
                except Exception:
                    pass
            self._active_keys.clear()

    def close(self) -> None:
        """Release pressed keys/buttons and destroy virtual /dev/uinput devices cleanly."""
        if self._closed:
            return
        self._closed = True
        try:
            self.release_all()
        except Exception:
            pass
        for attr in ("_pointer_fd", "_keyboard_fd"):
            fd = getattr(self, attr, None)
            if fd is not None:
                setattr(self, attr, None)
                try:
                    fcntl.ioctl(fd, _UI_DEV_DESTROY)
                except OSError:
                    pass
                try:
                    os.close(fd)
                except OSError:
                    pass


def _finalize_uinput_fds(backend: _UInputBackend) -> None:
    try:
        backend.close()
    except Exception:
        pass


def parse_key_combo(key: str) -> Tuple[List[str], List[int]]:
    """Parse and validate a key or modifier combination into canonical names and Linux keycodes."""
    if not isinstance(key, str):
        raise InvalidInputKeyError("Key specification must be a string.")
    key_str = key.strip()
    if not key_str:
        raise InvalidInputKeyError("Key string cannot be empty.")

    if key_str in _KEY_NAME_TO_CODE:
        return [key_str.lower()], [_KEY_NAME_TO_CODE[key_str]]

    if "+" in key_str:
        raw_parts = [p.strip().lower() for p in key_str.split("+")]
    elif "-" in key_str:
        raw_parts = [p.strip().lower() for p in key_str.split("-")]
    else:
        raw_parts = [key_str.lower()]

    if any(not p for p in raw_parts):
        raise InvalidInputKeyError(f"Malformed key combination '{key}'.")

    names: List[str] = []
    keycodes: List[int] = []
    for part in raw_parts:
        if part in _KEY_NAME_TO_CODE:
            names.append(part)
            keycodes.append(_KEY_NAME_TO_CODE[part])
        else:
            raise InvalidInputKeyError(
                f"Unknown or unsupported key '{part}' in '{key}'. Supported examples: enter, tab, esc, backspace, space, ctrl+c, alt+tab, super."
            )
    return names, keycodes


def _run_subprocess_bounded(
    cmd: List[str],
    *,
    timeout: float = 4.0,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    """Execute an OS input command with stdin closed and process-group kill on timeout."""
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        start_new_session=True,
    )
    try:
        stdout_s, stderr_s = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout_s, stderr_s)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            try:
                proc.kill()
            except OSError:
                pass
        try:
            proc.communicate(timeout=1.0)
        except Exception:
            pass
        raise InputTimeoutError(f"Input command '{cmd[0]}' timed out after {timeout}s.") from exc


class WaylandInputController:
    """Unified Linux/Wayland mouse & keyboard controller using Hyprland IPC, /dev/uinput, wtype, or ydotool."""

    def __init__(
        self,
        socket_path: Optional[str] = None,
        auto_start_daemon: bool = True,
        auto_check: bool = True,
        timeout: float = 4.0,
        backend: str = "auto",
        uinput_path: str = "/dev/uinput",
        subprocess_runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
        bounds_provider: Optional[Callable[[], List[Dict[str, Any]]]] = None,
    ) -> None:
        self.custom_socket = socket_path
        self.socket_path = self._resolve_socket_path(socket_path)
        self.auto_start_daemon = auto_start_daemon
        self.timeout = float(timeout)
        self.backend_preference = (backend or "auto").strip().lower()
        self.uinput_path = uinput_path
        self._runner = subprocess_runner or _run_subprocess_bounded
        self._bounds_provider = bounds_provider

        self.ydotool_bin = shutil.which("ydotool")
        self.wtype_bin = shutil.which("wtype")
        self.hyprctl_bin = shutil.which("hyprctl")
        self.xdotool_bin = shutil.which("xdotool")
        self._uinput: Optional[_UInputBackend] = None
        self._use_lua_dispatch = False
        self._closed = False
        atexit.register(self.close)

        if auto_check and subprocess_runner is None:
            self._validate_any_backend_present()

    def _has_display_session(self) -> bool:
        return bool(
            (os.environ.get("WAYLAND_DISPLAY") or "").strip()
            or (os.environ.get("DISPLAY") or "").strip()
            or (os.environ.get("HYPRLAND_INSTANCE_SIGNATURE") or "").strip()
        )

    def _ensure_display_session(self) -> None:
        if not self._has_display_session():
            raise DesktopUnavailableError(
                "No active Wayland or X11 desktop session detected (WAYLAND_DISPLAY and DISPLAY are unset)."
            )

    def _validate_any_backend_present(self) -> None:
        has_uinput = os.path.exists(self.uinput_path)
        if not (has_uinput or self.wtype_bin or self.ydotool_bin or self.hyprctl_bin or self.xdotool_bin):
            raise YdoToolError(
                "No Linux input backend found (/dev/uinput, wtype, ydotool, hyprctl, or xdotool).",
                error_type="BACKEND_UNAVAILABLE",
            )

    def _get_uinput(self) -> _UInputBackend:
        if self._uinput is None:
            self._uinput = _UInputBackend(device_path=self.uinput_path)
        return self._uinput

    def close(self) -> None:
        """Safely release held keys/buttons and close virtual uinput devices."""
        if self._uinput is not None:
            self._uinput.close()
        self._closed = True

    def __enter__(self) -> WaylandInputController:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def _resolve_socket_path(self, explicit_path: Optional[str] = None) -> Optional[str]:
        """Resolve the ydotool socket path when ydotool is configured."""
        if explicit_path:
            return explicit_path
        env_sock = os.environ.get("YDOTOOL_SOCKET")
        if env_sock and Path(env_sock).exists():
            return env_sock
        for candidate in (
            "/run/ydotool.socket",
            f"/run/user/{os.getuid()}/.ydotool_socket",
            "/tmp/.ydotool_socket",
        ):
            if Path(candidate).exists():
                return candidate
        return env_sock or f"/run/user/{os.getuid()}/.ydotool_socket"

    def _get_exec_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        if self.socket_path:
            env["YDOTOOL_SOCKET"] = self.socket_path
        return env

    def is_daemon_running(self) -> bool:
        """Check whether ydotoold is running and accepting socket commands."""
        if not self.ydotool_bin:
            return False
        try:
            res = self._runner(["pgrep", "-x", "ydotoold"], timeout=1.5)
            if res.returncode != 0:
                return False
        except Exception:
            return False
        if not self.socket_path or not Path(self.socket_path).exists():
            return False
        return True

    def ensure_daemon_running(self) -> None:
        """Verify ydotool daemon availability (retained for backward compatibility)."""
        if not self.ydotool_bin:
            raise YdoToolError("Binary 'ydotool' not found in PATH.")
        if not self.is_daemon_running():
            raise YdoToolDaemonError(
                f"Failed to connect to ydotool daemon socket '{self.socket_path}'. ydotoold is not running."
            )

    def _query_monitors(self) -> List[Dict[str, Any]]:
        """Retrieve active compositor monitor rectangles synchronously."""
        if self._bounds_provider is not None:
            mons = self._bounds_provider()
            return list(mons) if isinstance(mons, list) else []
        if not (os.environ.get("HYPRLAND_INSTANCE_SIGNATURE") or "").strip() or not self.hyprctl_bin:
            return []
        try:
            res = self._runner([self.hyprctl_bin, "monitors", "-j"], timeout=2.0)
            if res.returncode != 0 or not res.stdout:
                return []
            raw = json.loads(res.stdout)
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
                monitors.append({
                    "name": str(mon.get("name") or ""),
                    "x": int(mon.get("x") or 0),
                    "y": int(mon.get("y") or 0),
                    "width": px_w,
                    "height": px_h,
                    "logical_width": max(1, int(round(px_w / scale))) if px_w > 0 else 0,
                    "logical_height": max(1, int(round(px_h / scale))) if px_h > 0 else 0,
                    "focused": bool(mon.get("focused")),
                })
            return monitors
        except Exception:
            return []

    def validate_coordinates(
        self,
        x: Any,
        y: Any,
        *,
        monitors: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[int, int]:
        """Validate that (x, y) are finite numbers within an active display rectangle."""
        for label, val in (("x", x), ("y", y)):
            if isinstance(val, bool) or val is None:
                raise InvalidCoordinatesError(f"Coordinate '{label}' must be a numeric pixel value, got {val!r}.")
            if isinstance(val, str):
                try:
                    val = float(val.strip())
                except ValueError as exc:
                    raise InvalidCoordinatesError(f"Coordinate '{label}' must be numeric, got {val!r}.") from exc
            if not isinstance(val, (int, float)) or not math.isfinite(float(val)):
                raise InvalidCoordinatesError(f"Coordinate '{label}' must be a finite number, got {val!r}.")

        ix = int(round(float(x)))
        iy = int(round(float(y)))

        eff_monitors = monitors if monitors is not None else self._query_monitors()
        if eff_monitors:
            for mon in eff_monitors:
                mx = int(mon.get("x", 0))
                my = int(mon.get("y", 0))
                mw = int(mon.get("logical_width") or mon.get("width") or 0)
                mh = int(mon.get("logical_height") or mon.get("height") or 0)
                if mx <= ix < mx + mw and my <= iy < my + mh:
                    return ix, iy
            rects = [
                f"{m.get('name') or 'mon'}:[{m.get('x', 0)}..{int(m.get('x', 0)) + int(m.get('logical_width') or m.get('width') or 0)}, "
                f"{m.get('y', 0)}..{int(m.get('y', 0)) + int(m.get('logical_height') or m.get('height') or 0)})"
                for m in eff_monitors
            ]
            raise InvalidCoordinatesError(
                f"Coordinates ({ix}, {iy}) are outside all active monitor bounds: {', '.join(rects)}."
            )

        if ix < 0 or iy < 0:
            raise InvalidCoordinatesError(f"Coordinates ({ix}, {iy}) must be non-negative.")
        return ix, iy

    def get_cursor_position(self) -> Optional[Tuple[int, int]]:
        """Query current compositor cursor position if available."""
        if not self.hyprctl_bin or not (os.environ.get("HYPRLAND_INSTANCE_SIGNATURE") or "").strip():
            return None
        try:
            res = self._runner([self.hyprctl_bin, "cursorpos", "-j"], timeout=2.0)
            if res.returncode == 0 and res.stdout:
                data = json.loads(res.stdout)
                if isinstance(data, dict) and "x" in data and "y" in data:
                    return int(round(float(data["x"]))), int(round(float(data["y"])))
        except Exception:
            pass
        return None

    def _execute_ydotool(self, args: List[str]) -> subprocess.CompletedProcess:
        if not self.ydotool_bin:
            raise YdoToolError("Binary 'ydotool' not found in PATH.")
        cmd = [self.ydotool_bin] + args
        res = self._runner(cmd, timeout=self.timeout, env=self._get_exec_env())
        if res.returncode != 0:
            err = f"{res.stdout or ''}\n{res.stderr or ''}".strip()
            err_lower = err.lower()
            if "permission denied" in err_lower or "eacces" in err_lower:
                raise YdoToolPermissionError(f"Permission denied executing ydotool: {err}")
            if "failed to connect socket" in err_lower or "ydotoold" in err_lower:
                raise YdoToolDaemonError(f"ydotool socket error: {err}")
            raise InputFailedError(f"ydotool failed (exit {res.returncode}): {err}")
        return res

    def _hyprctl_movecursor(self, ix: int, iy: int) -> None:
        if not self.hyprctl_bin:
            raise BackendUnavailableError("hyprctl binary is not available.")
        if not self._use_lua_dispatch:
            res = self._runner([self.hyprctl_bin, "dispatch", "movecursor", str(ix), str(iy)], timeout=self.timeout)
            combined = f"{res.stdout or ''}\n{res.stderr or ''}".lower()
            if res.returncode == 0 and "hl.dispatch" not in combined and "note: dispatch in lua" not in combined:
                return
            if "hl.dispatch" in combined or "note: dispatch in lua" in combined or "expected near" in combined or res.returncode == 7:
                self._use_lua_dispatch = True
            else:
                err = f"{res.stdout or ''}\n{res.stderr or ''}".strip()
                raise InputFailedError(f"hyprctl dispatch movecursor failed (exit {res.returncode}): {err}")

        lua_expr = f"hl.dispatch(hl.dsp.cursor.move({{x={ix}, y={iy}}}))"
        res_lua = self._runner([self.hyprctl_bin, "eval", lua_expr], timeout=self.timeout)
        lua_combined = f"{res_lua.stdout or ''}\n{res_lua.stderr or ''}".strip()
        if res_lua.returncode != 0 or lua_combined.lower().startswith("error:"):
            raise InputFailedError(f"hyprctl eval cursor.move failed (exit {res_lua.returncode}): {lua_combined}")

    def mouse_move(
        self,
        x: Union[int, float],
        y: Union[int, float],
        *,
        monitors: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[int, int]:
        """Move the mouse cursor to validated logical desktop coordinates (x, y)."""
        self._ensure_display_session()
        ix, iy = self.validate_coordinates(x, y, monitors=monitors)

        if self.backend_preference == "ydotool" or (not (os.environ.get("HYPRLAND_INSTANCE_SIGNATURE") or "").strip() and self.is_daemon_running()):
            self._execute_ydotool(["mousemove", "-a", "-x", str(ix), "-y", str(iy)])
            return ix, iy

        if (os.environ.get("HYPRLAND_INSTANCE_SIGNATURE") or "").strip() and self.hyprctl_bin:
            self._hyprctl_movecursor(ix, iy)
            return ix, iy

        if self.is_daemon_running():
            self._execute_ydotool(["mousemove", "-a", "-x", str(ix), "-y", str(iy)])
            return ix, iy

        if (os.environ.get("DISPLAY") or "").strip() and self.xdotool_bin:
            res = self._runner([self.xdotool_bin, "mousemove", "--sync", str(ix), str(iy)], timeout=self.timeout)
            if res.returncode != 0:
                raise InputFailedError(f"xdotool mousemove failed: {res.stderr or res.stdout}")
            return ix, iy

        raise BackendUnavailableError(
            "No absolute pointer positioning backend available (need Hyprland IPC/hyprctl, ydotoold, or X11 xdotool)."
        )

    def _resolve_button(self, button: str) -> Tuple[str, int]:
        if not isinstance(button, str) or not button.strip():
            raise InvalidInputKeyError("Mouse button must be a non-empty string ('left', 'right', or 'middle').")
        btn_lower = button.strip().lower()
        if btn_lower not in _KERNEL_MOUSE_BUTTONS:
            raise InvalidInputKeyError(
                f"Unsupported mouse button '{button}'. Supported: left, right, middle, side, extra."
            )
        return btn_lower, _KERNEL_MOUSE_BUTTONS[btn_lower]

    def mouse_click(
        self,
        button: str = "left",
        clicks: int = 1,
        x: Optional[Union[int, float]] = None,
        y: Optional[Union[int, float]] = None,
        *,
        monitors: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Click a mouse button, optionally moving to validated (x, y) first."""
        self._ensure_display_session()
        btn_lower, btn_code = self._resolve_button(button)
        if isinstance(clicks, bool) or not isinstance(clicks, int) or clicks < 1 or clicks > 5:
            raise InvalidInputKeyError(f"Click count must be an integer between 1 and 5, got {clicks!r}.")

        if (x is None) != (y is None):
            raise InvalidCoordinatesError("Both 'x' and 'y' must be provided together when specifying click coordinates.")
        if x is not None and y is not None:
            self.mouse_move(x, y, monitors=monitors)

        if self.backend_preference == "ydotool":
            args = ["click"]
            if clicks > 1:
                args.extend(["-r", str(clicks)])
            args.append(_MOUSE_BUTTONS[btn_lower])
            self._execute_ydotool(args)
            return

        if self.backend_preference in ("auto", "uinput"):
            try:
                self._get_uinput().click(btn_code, clicks=clicks)
                return
            except (BackendUnavailableError, YdoToolPermissionError) as uinput_exc:
                if self.backend_preference == "uinput":
                    raise
                if self.is_daemon_running():
                    args = ["click"]
                    if clicks > 1:
                        args.extend(["-r", str(clicks)])
                    args.append(_MOUSE_BUTTONS[btn_lower])
                    self._execute_ydotool(args)
                    return
                if (os.environ.get("DISPLAY") or "").strip() and self.xdotool_bin:
                    x11_btn = {"left": "1", "middle": "2", "right": "3"}.get(btn_lower, "1")
                    res = self._runner([self.xdotool_bin, "click", "--repeat", str(clicks), x11_btn], timeout=self.timeout)
                    if res.returncode != 0:
                        raise InputFailedError(f"xdotool click failed: {res.stderr or res.stdout}")
                    return
                raise uinput_exc

        if self.is_daemon_running():
            args = ["click"]
            if clicks > 1:
                args.extend(["-r", str(clicks)])
            args.append(_MOUSE_BUTTONS[btn_lower])
            self._execute_ydotool(args)
            return

        raise BackendUnavailableError("No mouse click backend available (/dev/uinput or ydotoold).")

    def mouse_drag(
        self,
        start_x: Union[int, float],
        start_y: Union[int, float],
        end_x: Union[int, float],
        end_y: Union[int, float],
        button: str = "left",
        steps: int = 8,
        *,
        monitors: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Drag pointer from (start_x, start_y) to (end_x, end_y) with guaranteed button release."""
        self._ensure_display_session()
        sx, sy = self.validate_coordinates(start_x, start_y, monitors=monitors)
        ex, ey = self.validate_coordinates(end_x, end_y, monitors=monitors)
        _, btn_code = self._resolve_button(button)
        eff_steps = max(2, min(30, int(steps)))

        self.mouse_move(sx, sy, monitors=monitors)
        uinput = self._get_uinput()
        uinput.button_down(btn_code)
        try:
            for step in range(1, eff_steps + 1):
                frac = step / eff_steps
                nx = int(round(sx + (ex - sx) * frac))
                ny = int(round(sy + (ey - sy) * frac))
                self.mouse_move(nx, ny, monitors=monitors)
                time.sleep(0.015)
        finally:
            uinput.button_up(btn_code)

    def mouse_scroll(
        self,
        delta_y: int = 0,
        delta_x: int = 0,
        x: Optional[Union[int, float]] = None,
        y: Optional[Union[int, float]] = None,
        *,
        monitors: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Emit vertical or horizontal mouse wheel scroll ticks."""
        self._ensure_display_session()
        if (x is None) != (y is None):
            raise InvalidCoordinatesError("Both 'x' and 'y' must be provided together when scrolling at coordinates.")
        if x is not None and y is not None:
            self.mouse_move(x, y, monitors=monitors)
        dy = int(delta_y)
        dx = int(delta_x)
        if dy == 0 and dx == 0:
            raise InvalidInputKeyError("Scroll delta_y and delta_x cannot both be 0.")
        self._get_uinput().scroll(delta_y=dy, delta_x=dx)

    def keyboard_type(self, text: str) -> None:
        """Type the given text string via Wayland virtual keyboard or /dev/uinput."""
        self._ensure_display_session()
        if not isinstance(text, str):
            raise InvalidInputKeyError("Text to type must be a string.")
        if "\x00" in text:
            raise InvalidInputKeyError("Text to type cannot contain null bytes.")
        if not text:
            raise InvalidInputKeyError("Text to type cannot be empty.")

        use_wayland = bool((os.environ.get("WAYLAND_DISPLAY") or "").strip() or (os.environ.get("HYPRLAND_INSTANCE_SIGNATURE") or "").strip())
        if self.backend_preference in ("auto", "wtype") and use_wayland and self.wtype_bin:
            res = self._runner([self.wtype_bin, "-d", "4", "--", text], timeout=max(self.timeout, min(12.0, 2.0 + 0.02 * len(text))))
            if res.returncode == 0:
                return
            err = f"{res.stdout or ''}\n{res.stderr or ''}".strip()
            if self.backend_preference == "wtype":
                raise InputFailedError(f"wtype text input failed (exit {res.returncode}): {err}")
            logger.debug("wtype failed (%s); attempting fallback backend", err)

        if self.backend_preference in ("auto", "uinput"):
            try:
                self._get_uinput().type_ascii(text)
                return
            except (BackendUnavailableError, YdoToolPermissionError) as exc:
                if self.backend_preference == "uinput":
                    raise
                logger.debug("/dev/uinput typing unavailable (%s); trying remaining fallbacks", exc)

        if self.is_daemon_running():
            self._execute_ydotool(["type", "-d", "12", "-H", "12", "--", text])
            return

        if (os.environ.get("DISPLAY") or "").strip() and self.xdotool_bin:
            res = self._runner([self.xdotool_bin, "type", "--clearmodifiers", "--delay", "12", "--", text], timeout=self.timeout)
            if res.returncode == 0:
                return
            raise InputFailedError(f"xdotool type failed: {res.stderr or res.stdout}")

        raise BackendUnavailableError("No keyboard typing backend available (wtype, /dev/uinput, ydotoold, or xdotool).")

    def _build_wtype_key_cmd(self, names: List[str]) -> Optional[List[str]]:
        if not self.wtype_bin or not names:
            return None
        mods: List[str] = []
        keys: List[str] = []
        for idx, tok in enumerate(names):
            if idx < len(names) - 1 and tok in _WTYPE_MODIFIERS:
                mods.append(_WTYPE_MODIFIERS[tok])
            elif tok in _WTYPE_MODIFIERS and len(names) == 1:
                return None
            elif tok in _WTYPE_SPECIAL_KEYS:
                keys.append(_WTYPE_SPECIAL_KEYS[tok])
            elif len(tok) == 1 and tok.isprintable():
                keys.append(tok)
            else:
                return None
        if len(keys) != 1:
            return None
        cmd = [self.wtype_bin]
        for m in mods:
            cmd.extend(["-M", m])
        cmd.extend(["-k", keys[0]] if (keys[0] in _WTYPE_SPECIAL_KEYS.values()) else [keys[0]])
        return cmd

    def keyboard_press(self, key: str) -> None:
        """Validate and press a single key or modifier combination."""
        self._ensure_display_session()
        names, keycodes = parse_key_combo(key)

        if self.backend_preference == "ydotool":
            events = [f"{c}:1" for c in keycodes] + [f"{c}:0" for c in reversed(keycodes)]
            self._execute_ydotool(["key"] + events)
            return

        if self.backend_preference in ("auto", "uinput"):
            try:
                self._get_uinput().press_keycodes(keycodes)
                return
            except (BackendUnavailableError, YdoToolPermissionError) as uinput_exc:
                if self.backend_preference == "uinput":
                    raise
                if (os.environ.get("WAYLAND_DISPLAY") or "").strip() and self.wtype_bin:
                    wtype_cmd = self._build_wtype_key_cmd(names)
                    if wtype_cmd:
                        res = self._runner(wtype_cmd, timeout=self.timeout)
                        if res.returncode == 0:
                            return
                        raise InputFailedError(f"wtype key press failed: {res.stderr or res.stdout}")
                if self.is_daemon_running():
                    events = [f"{c}:1" for c in keycodes] + [f"{c}:0" for c in reversed(keycodes)]
                    self._execute_ydotool(["key"] + events)
                    return
                raise uinput_exc

        if self.backend_preference == "wtype" and self.wtype_bin:
            wtype_cmd = self._build_wtype_key_cmd(names)
            if not wtype_cmd:
                raise InvalidInputKeyError(f"Key combo '{key}' cannot be dispatched cleanly by wtype.")
            res = self._runner(wtype_cmd, timeout=self.timeout)
            if res.returncode != 0:
                raise InputFailedError(f"wtype key press failed: {res.stderr or res.stdout}")
            return

        if self.is_daemon_running():
            events = [f"{c}:1" for c in keycodes] + [f"{c}:0" for c in reversed(keycodes)]
            self._execute_ydotool(["key"] + events)
            return

        raise BackendUnavailableError("No keyboard press backend available (/dev/uinput, wtype, or ydotoold).")


InputController = WaylandInputController

_default_instance: Optional[WaylandInputController] = None


def get_input_controller() -> WaylandInputController:
    """Return a shared default WaylandInputController instance."""
    global _default_instance
    if _default_instance is None or _default_instance._closed:
        _default_instance = WaylandInputController()
    return _default_instance


def mouse_move(x: Union[int, float], y: Union[int, float]) -> None:
    get_input_controller().mouse_move(x, y)


def mouse_click(button: str = "left", clicks: int = 1) -> None:
    get_input_controller().mouse_click(button=button, clicks=clicks)


def keyboard_type(text: str) -> None:
    get_input_controller().keyboard_type(text)


def keyboard_press(key: str) -> None:
    get_input_controller().keyboard_press(key)


__all__ = [
    "InputControlError",
    "YdoToolError",
    "YdoToolDaemonError",
    "YdoToolPermissionError",
    "DesktopUnavailableError",
    "BackendUnavailableError",
    "InvalidCoordinatesError",
    "InvalidInputKeyError",
    "InputTimeoutError",
    "InputFailedError",
    "WaylandInputController",
    "InputController",
    "parse_key_combo",
    "get_input_controller",
    "mouse_move",
    "mouse_click",
    "keyboard_type",
    "keyboard_press",
]
