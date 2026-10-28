"""Legacy ``axiom`` import compatibility package.

The canonical package is :mod:`zelyr`.  This package lets existing scripts
keep importing ``axiom`` and its submodules while the implementation lives in
``zelyr``.  No implementation is duplicated here.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
from typing import Optional

import zelyr as _zelyr

__version__ = _zelyr.__version__

# Submodules that were themselves renamed during the identity migration.
_SUBMODULE_ALIASES = {
    "axiom.core.vision_engine.axiom_engine": "zelyr.core.vision_engine.zelyr_engine",
    "axiom.fs.axiom_fs": "zelyr.fs.zelyr_fs",
    "axiom.kernel.axiom_kernel": "zelyr.kernel.zelyr_kernel",
    "axiom.plugins.axiom_plugin": "zelyr.plugins.zelyr_plugin",
}


class _AliasLoader(importlib.abc.Loader):
    """Return an already imported canonical module under a legacy name."""

    def __init__(self, target_name: str) -> None:
        self.target_name = target_name

    def create_module(self, spec: importlib.machinery.ModuleSpec):
        return importlib.import_module(self.target_name)

    def exec_module(self, module) -> None:
        return None

    def get_code(self, fullname: str):
        """Allow ``python -m axiom.<module>`` to execute the target module."""
        target = importlib.import_module(self.target_name)
        spec = getattr(target, "__spec__", None)
        loader = getattr(spec, "loader", None)
        if loader is None:
            return None
        return loader.get_code(target.__name__)


class _AliasFinder(importlib.abc.MetaPathFinder):
    """Map ``axiom.*`` imports to the corresponding ``zelyr.*`` module."""

    def find_spec(
        self,
        fullname: str,
        path: Optional[object] = None,
        target: Optional[object] = None,
    ):
        if fullname == "axiom" or not fullname.startswith("axiom."):
            return None
        target_name = _SUBMODULE_ALIASES.get(
            fullname, f"zelyr.{fullname[len('axiom.'):]}"
        )
        try:
            target_module = importlib.import_module(target_name)
        except ModuleNotFoundError as exc:
            # Only hide a missing target module, not a missing dependency
            # imported by an existing target.
            if exc.name == target_name:
                return None
            raise
        return importlib.machinery.ModuleSpec(
            fullname,
            _AliasLoader(target_name),
            is_package=hasattr(target_module, "__path__"),
        )


def __getattr__(name: str):
    """Expose renamed public class names through the legacy package."""
    if name.startswith("Axiom"):
        target_name = f"Zelyr{name[len('Axiom'):]}"
        try:
            val = getattr(_zelyr, target_name)
            globals()[name] = val
            return val
        except AttributeError:
            pass
    try:
        val = getattr(_zelyr, name)
        globals()[name] = val
        return val
    except AttributeError as exc:
        raise AttributeError(f"module 'axiom' has no attribute {name!r}") from exc


if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

__all__ = list(getattr(_zelyr, "__all__", ())) + [
    f"Axiom{name[len('Zelyr'):]}"
    for name in getattr(_zelyr, "__all__", ())
    if name.startswith("Zelyr")
]


def __dir__() -> list[str]:
    """Ensure legacy exported symbols remain discoverable via dir() and introspection."""
    return sorted(set(list(globals().keys()) + __all__))
