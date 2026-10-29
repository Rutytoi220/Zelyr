"""Zelyr GUI package namespace."""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("zelyr-core")
except PackageNotFoundError:
    __version__ = "1.1.0"

__all__ = ["app", "bridge", "main_window"]
