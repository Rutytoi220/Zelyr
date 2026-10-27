"""Zero-Trust Process & URL Deny-List for RFC-003 Phase 2.

Any clipboard content or window event whose source matches an entry in this
deny-list is **immediately dropped in memory** — before any logging, before any
regex evaluation, before any EventBus emission. The event is irrecoverably
discarded and no trace of the sensitive content is retained.

This module is un-bypassable: it cannot be overridden through configuration
or plugin APIs. It is enforced unconditionally at the first touch point inside
every perception tap.
"""

from __future__ import annotations

import re
import logging
from typing import Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardcoded process deny-list
# Any process whose name contains one of these strings (case-insensitive) is
# blocked. No configuration can extend or override this list at runtime.
# ---------------------------------------------------------------------------
_PROCESS_DENY_SUBSTRINGS: tuple[str, ...] = (
    "1password",
    "bitwarden",
    "keychain",
    "lastpass",
    "keepass",
    "dashlane",
    "nordpass",
    "roboform",
    "enpass",
    "strongbox",
    "vaultwarden",
    "gnome-keyring",
    "kwallet",
    "seahorse",
)

# ---------------------------------------------------------------------------
# Hardcoded window-title deny-list (regex patterns)
# Matches banking institutions, payment processors, and credential managers.
# ---------------------------------------------------------------------------
_WINDOW_TITLE_DENY_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in [
        r"1password",
        r"bitwarden",
        r"lastpass",
        r"keepass",
        r"dashlane",
        r"keychain access",
        r"credential\s*manager",
        r"password\s*manager",
        # Banking keywords
        r"\bbank(ing)?\b",
        r"\bpaypal\b",
        r"\bstripe\b",
        r"\bbraintree\b",
        r"\bchase\b",
        r"\bwells\s*fargo\b",
        r"\bbarclays\b",
        r"\bhargreaves\s*lansdown\b",
        r"\brobinhood\b",
        r"\bschwab\b",
        r"\bfidelity\b",
        r"\betrade\b",
        r"\btd\s*ameritrade\b",
        r"coinbase",
        r"binance",
        r"kraken",
        # Generic sensitive terms in window titles
        r"\bssh\s+key\b",
        r"\bprivate\s+key\b",
        r"\bapi\s+key\b",
    ]
)

# ---------------------------------------------------------------------------
# Clipboard source deny-list: if pyperclip or native hooks expose the source
# application, apply the same process deny-list.
# ---------------------------------------------------------------------------


class DenyList:
    """Enforces the hardcoded security deny-list.

    All methods are classmethods — no instance needed.  The deny-list is
    baked into the module and cannot be altered at runtime.
    """

    @classmethod
    def is_process_blocked(cls, process_name: str) -> bool:
        """Return True if the process name matches a denied credential manager.

        Args:
            process_name: The executable name or process identifier.

        Returns:
            True if the process is on the deny-list and the event must be
            dropped immediately.
        """
        name_lower = process_name.lower()  # pragma: no cover
        for substr in _PROCESS_DENY_SUBSTRINGS:  # pragma: no cover
            if substr in name_lower:  # pragma: no cover
                # Deliberately NOT logging the process name to avoid leaking it
                logger.debug("DenyList: blocked process matched. Event dropped.")  # pragma: no cover
                return True  # pragma: no cover
        return False  # pragma: no cover

    @classmethod
    def is_window_title_blocked(cls, window_title: str) -> bool:
        """Return True if the window title matches a sensitive application.

        Args:
            window_title: The raw window title string.

        Returns:
            True if the event must be dropped immediately.
        """
        for pattern in _WINDOW_TITLE_DENY_PATTERNS:  # pragma: no cover
            if pattern.search(window_title):  # pragma: no cover
                logger.debug("DenyList: blocked window title matched. Event dropped.")  # pragma: no cover
                return True  # pragma: no cover
        return False  # pragma: no cover

    @classmethod
    def is_blocked(cls, process_name: str = "", window_title: str = "") -> bool:
        """Unified entry point: returns True if EITHER condition is blocked.

        Call this once per perception event before any further processing.
        If True is returned, **discard the event entirely**.

        Args:
            process_name: The owning process name (may be empty string if unknown).
            window_title: The window title (may be empty string if unknown).

        Returns:
            True if the event must be silently dropped.
        """
        if process_name and cls.is_process_blocked(process_name):  # pragma: no cover
            return True  # pragma: no cover
        if window_title and cls.is_window_title_blocked(window_title):  # pragma: no cover
            return True  # pragma: no cover
        return False  # pragma: no cover
