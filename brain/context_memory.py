"""Context memory system for Zelyr - tracks running apps and active windows/tabs."""

import time
import subprocess
import psutil
from typing import Set, Dict, List, Optional
from utils.logger import get_logger
from utils.system import KNOWN_BROWSERS, is_browser

logger = get_logger(__name__)

# Maximum number of actions kept in history (ring buffer)
_MAX_HISTORY = 50


class ContextMemory:
    """Lightweight in-memory context tracker for Zelyr."""
    
    def __init__(self):
        """Initialize the context memory."""
        self.running_apps: Set[str] = set()  # Set of running app names
        self.active_browsers: Dict[str, dict] = {}  # {browser_name: {tabs, windows, etc}}
        self.open_files: Set[str] = set()  # Set of open file paths
        self.clipboard_content: Optional[str] = None  # Last copied content

        # --- NEW: action history & active window ---
        self.last_action: Optional[dict] = None  # Most recent action + result
        self.active_window: Optional[str] = None  # Currently focused window title
        self.action_history: List[dict] = []  # Capped list of past actions

        self._refresh_running_apps()
    
    def _refresh_running_apps(self) -> None:
        """Refresh the list of running applications."""
        self.running_apps.clear()
        try:
            for proc in psutil.process_iter(['name']):
                try:
                    app_name = proc.info['name'].lower()
                    # Extract base name without path
                    base_name = app_name.split('/')[-1].split('.')[0]
                    if base_name and len(base_name) > 1:  # Filter out single letters
                        self.running_apps.add(base_name)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            logger.debug(f"Error refreshing apps: {e}")
    
    def is_app_running(self, app_name: str) -> bool:
        """Check if an app is currently running.
        
        Args:
            app_name: Name of the application (e.g., 'opera', 'firefox')
        
        Returns:
            True if app is running, False otherwise
        """
        self._refresh_running_apps()
        app_lower = app_name.lower().strip()
        # Check exact match
        if app_lower in self.running_apps:
            return True
        # Check if any running app contains this name
        for running in self.running_apps:
            if app_lower in running or running in app_lower:
                return True
        return False
    
    def register_app_opened(self, app_name: str, metadata: Optional[dict] = None) -> None:
        """Register that an app was opened.
        
        Args:
            app_name: Name of the application
            metadata: Optional metadata (tabs, windows, etc)
        """
        app_lower = app_name.lower()
        self.running_apps.add(app_lower)
        if is_browser(app_lower):
            if app_lower not in self.active_browsers:
                self.active_browsers[app_lower] = {'windows': 1, 'tabs': 1, 'is_new': True}
            else:
                self.active_browsers[app_lower]['is_new'] = False
        logger.debug(f"Registered app opened: {app_lower}")
    
    def register_new_tab(self, browser_name: str) -> None:
        """Register that a new tab was opened in a browser.
        
        Args:
            browser_name: Name of the browser
        """
        browser_lower = browser_name.lower()
        if browser_lower not in self.active_browsers:
            self.active_browsers[browser_lower] = {'windows': 1, 'tabs': 1}
        self.active_browsers[browser_lower]['tabs'] = self.active_browsers[browser_lower].get('tabs', 1) + 1
        logger.debug(f"Registered new tab in {browser_lower}")
    
    def register_new_window(self, browser_name: str) -> None:
        """Register that a new window was opened in a browser.
        
        Args:
            browser_name: Name of the browser
        """
        browser_lower = browser_name.lower()
        if browser_lower not in self.active_browsers:
            self.active_browsers[browser_lower] = {'windows': 1, 'tabs': 1}
        self.active_browsers[browser_lower]['windows'] = self.active_browsers[browser_lower].get('windows', 1) + 1
        logger.debug(f"Registered new window in {browser_lower}")
    
    def get_open_browser(self) -> Optional[str]:
        """Get the name of the first open browser, if any.
        
        Returns:
            Browser name (e.g., 'opera', 'firefox') or None
        """
        self._refresh_running_apps()
        for browser in KNOWN_BROWSERS:
            if self.is_app_running(browser):
                return browser
        return None
    
    # ------------------------------------------------------------------
    # Action history (NEW)
    # ------------------------------------------------------------------

    def record_action(self, action: str, params: str, ok: bool, message: str) -> None:
        """Record an executed action in history and update last_action.

        Args:
            action: Action name (e.g., 'open_app')
            params: Parameter string
            ok: Whether it succeeded
            message: Result message
        """
        entry = {
            'action': action,
            'params': params,
            'ok': ok,
            'message': message,
            'ts': time.time(),
        }
        self.last_action = entry
        self.action_history.append(entry)
        # Cap history length
        if len(self.action_history) > _MAX_HISTORY:
            self.action_history = self.action_history[-_MAX_HISTORY:]
        logger.debug("Recorded action: %s ok=%s", action, ok)

        # Persist to disk (lazy import to avoid circular deps)
        try:
            from brain.persistent_memory import get_persistent_memory
            get_persistent_memory().save_action(action, params, ok, message)
        except Exception as e:
            logger.debug("Persistent save failed (non-blocking): %s", e)

    def get_last_action(self) -> Optional[dict]:
        """Return the most recently executed action, or None."""
        return self.last_action

    def get_action_history(self, n: int = 10) -> List[dict]:
        """Return the last *n* recorded actions.

        Args:
            n: Number of recent actions to return (default 10)
        """
        return self.action_history[-n:]

    # ------------------------------------------------------------------
    # Active window tracking (NEW)
    # ------------------------------------------------------------------

    def refresh_active_window(self) -> Optional[str]:
        """Detect the currently focused window via xdotool.

        Updates ``self.active_window`` and returns the title, or None
        if detection fails.
        """
        try:
            result = subprocess.run(
                ['xdotool', 'getactivewindow', 'getwindowname'],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode == 0 and result.stdout.strip():
                self.active_window = result.stdout.strip()
                return self.active_window
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
        return self.active_window  # return cached value on failure

    def get_active_window(self) -> Optional[str]:
        """Return the last known active window title (no subprocess call)."""
        return self.active_window

    # ------------------------------------------------------------------
    # Existing helpers
    # ------------------------------------------------------------------
    
    def set_clipboard(self, content: str) -> None:
        """Track clipboard content.
        
        Args:
            content: Content copied to clipboard
        """
        self.clipboard_content = content
        logger.debug(f"Clipboard updated: {content[:50]}...")
    
    def get_clipboard(self) -> Optional[str]:
        """Get last tracked clipboard content.
        
        Returns:
            Clipboard content or None
        """
        return self.clipboard_content
    
    def reset(self) -> None:
        """Reset all context."""
        self.running_apps.clear()
        self.active_browsers.clear()
        self.open_files.clear()
        self.clipboard_content = None
        self.last_action = None
        self.active_window = None
        self.action_history.clear()
        logger.info("Context memory reset")
    
    def to_dict(self) -> dict:
        """Export context as dictionary for logging/debugging.
        
        Returns:
            Dictionary representation of context
        """
        return {
            'running_apps': sorted(list(self.running_apps)),
            'active_browsers': dict(self.active_browsers),
            'open_files': sorted(list(self.open_files)),
            'clipboard': bool(self.clipboard_content),
            'last_action': self.last_action,
            'active_window': self.active_window,
            'history_len': len(self.action_history),
        }


# Global context instance
_global_context = None


def get_context_memory() -> ContextMemory:
    """Get the global context memory instance."""
    global _global_context
    if _global_context is None:
        _global_context = ContextMemory()
    return _global_context


def reset_context_memory() -> None:
    """Reset the global context memory."""
    global _global_context
    _global_context = None
