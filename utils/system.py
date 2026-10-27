"""Shared system utilities for Zelyr.

Single source of truth for system detection (default browser, known browsers, etc.).
Avoids duplicating detection logic across agent.py and executor.py.
"""

import os
import re
import shutil
import subprocess
from typing import Optional
from .logger import get_logger

logger = get_logger(__name__)

# Canonical list of known browser identifiers.
# Used by context_memory, executor, and intent_parser instead of hardcoded lists.
KNOWN_BROWSERS = frozenset({
    'opera', 'firefox', 'chrome', 'chromium', 'chromium-browser',
    'google-chrome', 'brave', 'brave-browser', 'vivaldi', 'epiphany',
    'microsoft-edge', 'edge',
})

# Cache to avoid repeated subprocess calls within a session
_default_browser_cache: Optional[str] = None


def is_browser(app_name: str) -> bool:
    """Check whether *app_name* is a known browser.

    Uses substring matching so that e.g. 'google-chrome-stable' still matches.
    """
    name = app_name.lower().strip()
    if name in KNOWN_BROWSERS:
        return True
    for browser in KNOWN_BROWSERS:
        if browser in name or name in browser:
            return True
    return False


def get_default_browser(force_refresh: bool = False) -> Optional[str]:
    """Detect the system default browser.

    Resolution order:
    1. ``xdg-settings get default-web-browser``
    2. ``~/.config/mimeapps.list`` [Default Applications] section
    3. First known browser found on PATH

    Results are cached for the lifetime of the process (pass
    *force_refresh=True* to invalidate).
    """
    global _default_browser_cache
    if _default_browser_cache is not None and not force_refresh:
        return _default_browser_cache

    result = _try_xdg_settings() or _try_mimeapps_list() or _try_path_fallback()
    _default_browser_cache = result
    if result:
        logger.debug("Default browser detected: %s", result)
    else:
        logger.debug("Could not detect default browser")
    return result


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _try_xdg_settings() -> Optional[str]:
    try:
        result = subprocess.run(
            ['xdg-settings', 'get', 'default-web-browser'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            desktop_file = result.stdout.strip()
            if desktop_file.endswith('.desktop'):
                return desktop_file[:-8]  # strip '.desktop'
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def _try_mimeapps_list() -> Optional[str]:
    try:
        mimeapps_path = os.path.expanduser('~/.config/mimeapps.list')
        if os.path.exists(mimeapps_path):
            with open(mimeapps_path, 'r') as f:
                content = f.read()
            match = re.search(
                r'\[Default Applications\].*?x-scheme-handler/https=([^\n]+)',
                content, re.DOTALL,
            )
            if match:
                desktop_file = match.group(1).strip()
                if desktop_file.endswith('.desktop'):
                    return desktop_file[:-8]
    except Exception:
        pass
    return None


def _try_path_fallback() -> Optional[str]:
    for browser in ['firefox', 'opera', 'chromium', 'chromium-browser', 'google-chrome']:
        if shutil.which(browser):
            return browser
    return None
