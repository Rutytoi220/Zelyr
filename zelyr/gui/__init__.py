"""Zelyr GUI package namespace."""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("zelyr-core")
except PackageNotFoundError:
    __version__ = "3.0.0-dev"

__all__ = ["app", "bridge", "main_window"]
