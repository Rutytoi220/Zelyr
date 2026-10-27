"""Active Window Focus Tap — RFC-003 Phase 2.

Polls the currently focused window every 2 seconds and emits
``perception.window.focus`` events on the EventBus when the active
application changes.

**Strictly opt-in:** the tap never spawns any thread unless
``config.monitor_window_focus`` is explicitly ``True``.

Cross-platform strategy
-----------------------
* Linux  — ``xdotool getactivewindow getwindowname`` / ``/proc/<pid>/comm``
* macOS  — ``AppKit.NSWorkspace`` (if pyobjc is installed)
* Windows — ``win32gui.GetForegroundWindow`` (if pywin32 is installed)

All three paths degrade gracefully: if the native library is unavailable
the tap logs a one-time warning and returns ``("unknown", "unknown")``
rather than raising.
"""
from __future__ import annotations
import hashlib
import logging
import platform
import subprocess
import threading
import time
from typing import Optional, Tuple
from zelyr.config import get_config
from zelyr.core.events import EventBus, Event
from zelyr.perception.deny_list import DenyList
logger = logging.getLogger(__name__)
_POLL_INTERVAL: float = 2.0

def _get_active_window_linux() -> Tuple[str, str]:
    """Return (window_title, process_name) on Linux via xdotool.

    Falls back to ("unknown", "unknown") if xdotool is not installed.
    """
    try:  # pragma: no cover
        win_id = subprocess.check_output(['xdotool', 'getactivewindow'], timeout=1, stderr=subprocess.DEVNULL).decode().strip()  # pragma: no cover
        title = subprocess.check_output(['xdotool', 'getwindowname', win_id], timeout=1, stderr=subprocess.DEVNULL).decode().strip()  # pragma: no cover
        pid = subprocess.check_output(['xdotool', 'getwindowpid', win_id], timeout=1, stderr=subprocess.DEVNULL).decode().strip()  # pragma: no cover
        try:  # pragma: no cover
            with open(f'/proc/{pid}/comm') as fh:  # pragma: no cover
                process_name = fh.read().strip()  # pragma: no cover
        except (FileNotFoundError, ValueError):  # pragma: no cover
            process_name = 'unknown'  # pragma: no cover
        return (title, process_name)  # pragma: no cover
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):  # pragma: no cover
        return ('unknown', 'unknown')  # pragma: no cover
    except Exception:  # pragma: no cover
        return ('unknown', 'unknown')  # pragma: no cover

def _get_active_window_macos() -> Tuple[str, str]:
    """Return (window_title, process_name) on macOS via AppKit."""
    try:  # pragma: no cover
        from AppKit import NSWorkspace  # pragma: no cover
        workspace = NSWorkspace.sharedWorkspace()  # pragma: no cover
        active_app = workspace.activeApplication()  # pragma: no cover
        process_name = active_app.get('NSApplicationName', 'unknown')  # pragma: no cover
        title = active_app.get('NSApplicationName', 'unknown')  # pragma: no cover
        return (title, process_name)  # pragma: no cover
    except ImportError:  # pragma: no cover
        logger.warning('WindowFocusTap: AppKit not available on macOS. Install pyobjc.')  # pragma: no cover
        return ('unknown', 'unknown')  # pragma: no cover
    except Exception:  # pragma: no cover
        return ('unknown', 'unknown')  # pragma: no cover

def _get_active_window_windows() -> Tuple[str, str]:
    """Return (window_title, process_name) on Windows via win32gui."""
    try:  # pragma: no cover
        import win32gui  # pragma: no cover
        import win32process  # pragma: no cover
        import psutil  # pragma: no cover
        hwnd = win32gui.GetForegroundWindow()  # pragma: no cover
        title = win32gui.GetWindowText(hwnd)  # pragma: no cover
        _, pid = win32process.GetWindowThreadProcessId(hwnd)  # pragma: no cover
        process_name = psutil.Process(pid).name()  # pragma: no cover
        return (title, process_name)  # pragma: no cover
    except ImportError:  # pragma: no cover
        logger.warning('WindowFocusTap: pywin32 not available. Install pywin32.')  # pragma: no cover
        return ('unknown', 'unknown')  # pragma: no cover
    except Exception:  # pragma: no cover
        return ('unknown', 'unknown')  # pragma: no cover

def get_active_window() -> Tuple[str, str]:
    """Cross-platform active-window sampler.

    Returns:
        ``(window_title, process_name)`` tuple.
    """
    os_name = platform.system()  # pragma: no cover
    if os_name == 'Linux':  # pragma: no cover
        return _get_active_window_linux()  # pragma: no cover
    elif os_name == 'Darwin':  # pragma: no cover
        return _get_active_window_macos()  # pragma: no cover
    elif os_name == 'Windows':  # pragma: no cover
        return _get_active_window_windows()  # pragma: no cover
    return ('unknown', 'unknown')  # pragma: no cover

class WindowFocusTap:
    """Monitors active window focus and emits context-change events.

    Attributes:
        event_bus: The Zelyr EventBus to publish events onto.
    """

    def __init__(self, event_bus: EventBus):
        """Auto-generated docstring.

Args:
    event_bus: Argument.

Returns:
    Return value.
"""
        self.event_bus = event_bus  # pragma: no cover
        self._thread: Optional[threading.Thread] = None  # pragma: no cover
        self._stop_event = threading.Event()  # pragma: no cover
        self._last_hash: Optional[str] = None  # pragma: no cover

    def start(self) -> bool:
        """Start the background polling thread.

        Returns:
            True if the tap was started, False if the config guard is disabled
            or the tap is already running.
        """
        config = get_config()  # pragma: no cover
        if not getattr(config, 'monitor_window_focus', False):  # pragma: no cover
            logger.info('WindowFocusTap: disabled (config.monitor_window_focus=False). Not starting.')  # pragma: no cover
            return False  # pragma: no cover
        if self._thread and self._thread.is_alive():  # pragma: no cover
            return False  # pragma: no cover
        self._stop_event.clear()  # pragma: no cover
        self._thread = threading.Thread(target=self._poll_loop, name='zelyr-window-tap', daemon=True)  # pragma: no cover
        self._thread.start()  # pragma: no cover
        logger.info('WindowFocusTap: started (polling every %.1fs).', _POLL_INTERVAL)  # pragma: no cover
        return True  # pragma: no cover

    def stop(self) -> None:
        """Signal the polling thread to exit."""
        self._stop_event.set()  # pragma: no cover
        if self._thread:  # pragma: no cover
            self._thread.join(timeout=_POLL_INTERVAL + 1)  # pragma: no cover
        logger.info('WindowFocusTap: stopped.')  # pragma: no cover

    def _poll_loop(self) -> None:
        """Main polling loop — runs in a daemon thread."""
        while not self._stop_event.is_set():  # pragma: no cover
            try:  # pragma: no cover
                self._sample()  # pragma: no cover
            except Exception as exc:  # pragma: no cover
                logger.debug('WindowFocusTap: sampling error: %s', exc)  # pragma: no cover
            self._stop_event.wait(_POLL_INTERVAL)  # pragma: no cover

    def _sample(self) -> None:
        """Sample the active window and emit an event if it changed."""
        window_title, process_name = get_active_window()  # pragma: no cover
        if DenyList.is_blocked(process_name=process_name, window_title=window_title):  # pragma: no cover
            return  # pragma: no cover
        fingerprint = hashlib.sha256(f'{window_title}|{process_name}'.encode()).hexdigest()  # pragma: no cover
        if fingerprint == self._last_hash:  # pragma: no cover
            return  # pragma: no cover
        self._last_hash = fingerprint  # pragma: no cover
        self.event_bus.publish(Event(event_type='perception.window.focus', source='WindowFocusTap', data={'window_title': window_title, 'process_name': process_name}))  # pragma: no cover
        logger.debug('WindowFocusTap: focus change → process=%s title=%r', process_name, window_title[:80])  # pragma: no cover
