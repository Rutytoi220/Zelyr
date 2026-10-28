"""Zelyr main entry point and initialization."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "1.1.0rc1"
__author__ = "Zelyr Team"

__all__ = [
    "Engine",
    "OllamaClient",
    "CLI",
    "run_cli",
    "ZelyrConfig",
    "get_config",
    "set_config",
    "__version__",
]

_MAIN_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "Engine": ("zelyr", "Engine"),
    "OllamaClient": ("zelyr", "OllamaClient"),
    "CLI": ("zelyr.api.cli", "CLI"),
    "run_cli": ("zelyr.api.cli", "run_cli"),
    "ZelyrConfig": ("zelyr", "ZelyrConfig"),
    "get_config": ("zelyr", "get_config"),
    "set_config": ("zelyr", "set_config"),
}


def __getattr__(name: str) -> Any:
    """Lazily import and cache main symbols on first access (PEP 562)."""
    if name in _MAIN_LAZY_EXPORTS:
        mod_name, attr_name = _MAIN_LAZY_EXPORTS[name]
        val = getattr(importlib.import_module(mod_name), attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module 'zelyr.main' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Ensure lazily exported symbols remain discoverable via dir()."""
    return sorted(set(list(globals().keys()) + list(_MAIN_LAZY_EXPORTS.keys())))


if TYPE_CHECKING:
    from zelyr.core import Engine
    from zelyr.llm import OllamaClient
    from zelyr.api.cli import CLI, run_cli
    from zelyr.config import ZelyrConfig, get_config, set_config


def main():
    """Run Zelyr CLI."""
    from zelyr.api.cli import run_cli

    run_cli()


if __name__ == "__main__":
    main()
