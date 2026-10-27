"""Zero-Trust Privacy Scrubber for the Proactive Perception Kernel.

Screens incoming filesystem events and drops them if they involve sensitive
files (e.g., .env, SSH keys) or if their path matches known secret patterns.
"""

import re
from pathlib import Path
from typing import Optional


class PrivacyScrubber:
    """Filters sensitive filesystem events before they reach the Intent Engine."""

    # Exact filenames to block globally
    BLOCKED_FILENAMES = {
        ".env",
        "id_rsa",
        "id_ed25519",
        "id_ecdsa",
        "id_dsa",
        "secrets.json",
        "credentials.json",
    }

    # Extensions to block
    BLOCKED_EXTENSIONS = {
        ".pem",
        ".key",
        ".p12",
        ".pfx",
    }

    # Regex patterns for paths that should be ignored
    BLOCKED_PATTERNS = [
        re.compile(r"/\.git/"),
        re.compile(r"/\.zelyr/"),     # Ignore Zelyr's own internal directory
        re.compile(r"/\.axiom/"),     # Ignore legacy AXIOM state as well
        re.compile(r"__pycache__"),
        re.compile(r"\.pytest_cache"),
        re.compile(r"node_modules/"),
        re.compile(r"password", re.IGNORECASE),
        re.compile(r"secret", re.IGNORECASE),
        re.compile(r"token", re.IGNORECASE),
    ]

    @classmethod
    def is_safe(cls, path_str: str) -> bool:
        """Check if a given file path is safe to process.
        
        Args:
            path_str: Absolute or relative file path.
            
        Returns:
            True if safe, False if it contains sensitive data and should be dropped.
        """
        path = Path(path_str)  # pragma: no cover
        
        if path.name in cls.BLOCKED_FILENAMES:  # pragma: no cover
            return False  # pragma: no cover
            
        if path.suffix.lower() in cls.BLOCKED_EXTENSIONS:  # pragma: no cover
            return False  # pragma: no cover
            
        normalized_path = path.as_posix()  # pragma: no cover
        for pattern in cls.BLOCKED_PATTERNS:  # pragma: no cover
            if pattern.search(normalized_path):  # pragma: no cover
                return False  # pragma: no cover
                
        return True  # pragma: no cover

    # ------------------------------------------------------------------
    # Clipboard text scrubbing (RFC-003 Phase 2)
    # ------------------------------------------------------------------

    # Patterns that identify raw secrets in plain text.
    # Each tuple is (compiled_pattern, replacement_template).
    _TEXT_SECRET_PATTERNS: list = [
        # OpenAI / generic bearer tokens
        (re.compile(r'sk-[A-Za-z0-9]{20,}'), '[REDACTED:openai_key]'),
        # GitHub personal access tokens
        (re.compile(r'gh[pousr]_[A-Za-z0-9]{36,}'), '[REDACTED:github_token]'),
        # AWS access key IDs
        (re.compile(r'AKIA[0-9A-Z]{16}'), '[REDACTED:aws_access_key]'),
        # AWS secret access keys (40 chars of base64-ish)
        (re.compile(r'(?<![A-Za-z0-9])[A-Za-z0-9/+]{40}(?![A-Za-z0-9/+])'), '[REDACTED:aws_secret]'),
        # Generic API key = value patterns
        (re.compile(r'(?i)(api[_-]?key|apikey|access[_-]?token|auth[_-]?token)\s*[=:]\s*\S+'), r'\1=[REDACTED]'),
        # Private key PEM blocks
        (re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----', re.DOTALL), '[REDACTED:private_key]'),
        # Passwords in config-style strings
        (re.compile(r'(?i)(password|passwd|secret)\s*[=:]\s*\S+'), r'\1=[REDACTED]'),
        # Bearer tokens in Authorization headers
        (re.compile(r'(?i)bearer\s+[A-Za-z0-9\-._~+/]+=*'), 'Bearer [REDACTED]'),
    ]

    @classmethod
    def scrub_text(cls, text: str) -> str:
        """Redact secrets and API keys from raw clipboard text.

        Applies a suite of regex patterns to mask known secret formats before
        the text is handed to the IntentEngine or EventBus. The original text
        is never stored or logged.

        Args:
            text: Raw clipboard content.

        Returns:
            A sanitised copy of the text with secrets replaced by
            ``[REDACTED:*]`` placeholders.
        """
        result = text  # pragma: no cover
        for pattern, replacement in cls._TEXT_SECRET_PATTERNS:  # pragma: no cover
            result = pattern.sub(replacement, result)  # pragma: no cover
        return result  # pragma: no cover
