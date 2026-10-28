"""Standardized optional dependency boundary guards and discovery helpers.

Provides structured error reporting and zero-import availability checking
to preserve lightweight CLI startup and prevent unhandled import crashes.
"""

from __future__ import annotations

import importlib.util
import sys


class OptionalDependencyError(ImportError):
    """Raised when an optional dependency package is required but missing."""

    def __init__(self, package_name: str, extra_name: str, feature_description: str = ""):
        feature_str = f" for {feature_description}" if feature_description else ""
        message = (
            f"'{package_name}' is required{feature_str}. "
            f"Install with: pip install 'zelyr-core[{extra_name}]' or uv add --extra {extra_name} zelyr-core"
        )
        super().__init__(message)
        self.package_name = package_name
        self.extra_name = extra_name


def is_extra_available(package_name: str) -> bool:
    """Safely check if an optional package is importable without loading it into memory."""
    if sys.modules.get(package_name) is None and package_name in sys.modules:
        return False
    try:
        return importlib.util.find_spec(package_name) is not None
    except (ValueError, ModuleNotFoundError, AttributeError):
        return False


def require_extra(package_name: str, extra_name: str, feature_description: str = "") -> None:
    """Verify that an optional package is available, or raise an actionable OptionalDependencyError."""
    if not is_extra_available(package_name):
        raise OptionalDependencyError(package_name, extra_name, feature_description)
