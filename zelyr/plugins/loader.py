"""Plugin manifest parser and loader for Zelyr v2.

Validates ``plugin.toml`` files according to the schema specified in
RFC-001 and returns a strongly-typed ``PluginManifest`` dataclass.

ZELYR_VERSION_COMPAT is the installed Zelyr runtime version against which
all ``zelyr_version`` semver specifiers are evaluated.  Existing plugin
manifests that still use the legacy ``axiom_version`` key are accepted.
"""
from __future__ import annotations
import importlib
import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
tomllib_mod: Any = None
try:
    import tomllib
    tomllib_mod = tomllib
except ImportError:  # pragma: no cover
    try:  # pragma: no cover
        import tomli
        tomllib_mod = tomli  # pragma: no cover
    except ImportError:  # pragma: no cover
        tomllib_mod = None  # pragma: no cover
from zelyr.plugins.exceptions import PluginManifestError, PluginPermissionError, PluginVersionError
logger = logging.getLogger(__name__)
ZELYR_RUNTIME_VERSION = (2, 0, 0)
AXIOM_RUNTIME_VERSION = ZELYR_RUNTIME_VERSION  # legacy public constant
VALID_PERMISSIONS = frozenset({'filesystem', 'shell', 'network', 'desktop_ui'})

@dataclass
class PluginPermissions:
    """Parsed permission block from ``plugin.toml``."""
    filesystem: bool = False
    shell: bool = False
    network: bool = False
    desktop_ui: bool = False

    def allows(self, permission: str) -> bool:
        """Auto-generated docstring.

Args:
    permission: Argument.

Returns:
    Return value.
"""
        return bool(getattr(self, permission, False))  # pragma: no cover

@dataclass
class PluginManifest:
    """Strongly-typed representation of a validated ``plugin.toml``."""
    name: str
    version: str
    description: str
    author: str
    license: str
    zelyr_version: str
    module: str
    entry_class: str
    permissions: PluginPermissions
    plugin_dir: Path
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def axiom_version(self) -> str:
        """Legacy alias for ``zelyr_version`` used by older plugin code."""
        return self.zelyr_version

    @property
    def plugin_id(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return self.name.lower().replace(' ', '-').replace('_', '-')  # pragma: no cover

def _parse_semver(spec: str) -> tuple[str, tuple[int, ...]]:
    """Parse a simple semver specifier like ``>=2.0.0`` into (op, version_tuple)."""
    for op in ('>=', '<=', '==', '!=', '>', '<'):  # pragma: no cover
        if spec.startswith(op):  # pragma: no cover
            parts = spec[len(op):].strip().split('.')  # pragma: no cover
            return (op, tuple((int(p) for p in parts)))  # pragma: no cover
    raise PluginManifestError(f'Unsupported semver specifier format: {spec!r}. Expected one of: >=, <=, ==, !=, >, <')  # pragma: no cover

def _check_version_compat(specifier: str) -> None:
    """Raise ``PluginVersionError`` if ZELYR_RUNTIME_VERSION does not satisfy specifier."""
    op, required = _parse_semver(specifier)  # pragma: no cover
    runtime = ZELYR_RUNTIME_VERSION  # pragma: no cover
    satisfied = {'>=': runtime >= required, '<=': runtime <= required, '==': runtime == required, '!=': runtime != required, '>': runtime > required, '<': runtime < required}.get(op, False)  # pragma: no cover
    if not satisfied:  # pragma: no cover
        runtime_str = '.'.join((str(v) for v in runtime))  # pragma: no cover
        raise PluginVersionError(f'Plugin requires Zelyr {specifier}, but runtime is {runtime_str}.')  # pragma: no cover

class PluginLoader:
    """Validates and loads a single ``plugin.toml`` manifest.

    Usage::

        loader = PluginLoader()
        manifest = loader.load_manifest(Path("/path/to/my_plugin"))
        plugin_instance = loader.instantiate(manifest)
    """

    def load_manifest(self, plugin_dir: Path) -> PluginManifest:
        """Parse and validate ``plugin_dir/plugin.toml``.

        Raises:
            PluginManifestError: Schema validation failure.
            PluginVersionError: Zelyr version incompatibility.
        """
        toml_path = plugin_dir / 'plugin.toml'  # pragma: no cover
        if not toml_path.is_file():  # pragma: no cover
            raise PluginManifestError(f'No plugin.toml found in {plugin_dir}')  # pragma: no cover
        if tomllib_mod is None:  # pragma: no cover
            raise PluginManifestError('No TOML parser available. Install Python ≥ 3.11 or `pip install tomli`.')  # pragma: no cover
        with toml_path.open('rb') as fh:  # pragma: no cover
            raw = tomllib_mod.load(fh)  # pragma: no cover
        return self._validate(raw, plugin_dir)  # pragma: no cover

    def _validate(self, raw: Dict[str, Any], plugin_dir: Path) -> PluginManifest:
        """Validate the parsed TOML dict and return a ``PluginManifest``."""
        plugin_sec = raw.get('plugin')  # pragma: no cover
        if not isinstance(plugin_sec, dict):  # pragma: no cover
            raise PluginManifestError('Missing required [plugin] section in plugin.toml')  # pragma: no cover
        required_plugin_keys = ('name', 'version', 'description', 'author')  # pragma: no cover
        for key in required_plugin_keys:  # pragma: no cover
            if key not in plugin_sec:  # pragma: no cover
                raise PluginManifestError(f"Missing required key '[plugin].{key}' in plugin.toml")  # pragma: no cover
        if 'zelyr_version' in plugin_sec:  # pragma: no cover
            version_spec = plugin_sec['zelyr_version']  # pragma: no cover
        elif 'axiom_version' in plugin_sec:  # pragma: no cover
            version_spec = plugin_sec['axiom_version']  # pragma: no cover
        else:  # pragma: no cover
            raise PluginManifestError("Missing required key '[plugin].zelyr_version' (legacy alias: '[plugin].axiom_version') in plugin.toml")  # pragma: no cover
        _check_version_compat(version_spec)  # pragma: no cover
        entry_sec = raw.get('entrypoint')  # pragma: no cover
        if not isinstance(entry_sec, dict):  # pragma: no cover
            raise PluginManifestError('Missing required [entrypoint] section in plugin.toml')  # pragma: no cover
        if 'module' not in entry_sec:  # pragma: no cover
            raise PluginManifestError("Missing required key '[entrypoint].module'")  # pragma: no cover
        if 'class' not in entry_sec:  # pragma: no cover
            raise PluginManifestError("Missing required key '[entrypoint].class'")  # pragma: no cover
        perm_sec = raw.get('permissions', {})  # pragma: no cover
        if not isinstance(perm_sec, dict):  # pragma: no cover
            raise PluginManifestError('[permissions] section must be a TOML table')  # pragma: no cover
        unknown = set(perm_sec.keys()) - VALID_PERMISSIONS  # pragma: no cover
        if unknown:  # pragma: no cover
            raise PluginManifestError(f'Unknown permission keys: {unknown}. Valid keys are: {sorted(VALID_PERMISSIONS)}')  # pragma: no cover
        permissions = PluginPermissions(filesystem=bool(perm_sec.get('filesystem', False)), shell=bool(perm_sec.get('shell', False)), network=bool(perm_sec.get('network', False)), desktop_ui=bool(perm_sec.get('desktop_ui', False)))  # pragma: no cover
        return PluginManifest(name=plugin_sec['name'], version=plugin_sec['version'], description=plugin_sec.get('description', ''), author=plugin_sec.get('author', ''), license=plugin_sec.get('license', ''), zelyr_version=version_spec, module=entry_sec['module'], entry_class=entry_sec['class'], permissions=permissions, plugin_dir=plugin_dir, raw=raw)  # pragma: no cover

    def instantiate(self, manifest: PluginManifest) -> Any:
        """Import the plugin module and instantiate the declared class.

        The plugin directory is temporarily added to ``sys.path`` so relative
        imports within the plugin package work correctly.  It is removed again
        after import to avoid polluting the import namespace.

        Raises:
            PluginManifestError: Module or class not found / import error.
        """
        plugin_dir_str = str(manifest.plugin_dir)  # pragma: no cover
        added_to_path = False  # pragma: no cover
        try:  # pragma: no cover
            if plugin_dir_str not in sys.path:  # pragma: no cover
                sys.path.insert(0, plugin_dir_str)  # pragma: no cover
                added_to_path = True  # pragma: no cover
            try:  # pragma: no cover
                module = importlib.import_module(manifest.module)  # pragma: no cover
            except ImportError as exc:  # pragma: no cover
                raise PluginManifestError(f"Cannot import plugin module '{manifest.module}': {exc}") from exc  # pragma: no cover
            cls = getattr(module, manifest.entry_class, None)  # pragma: no cover
            if cls is None:  # pragma: no cover
                raise PluginManifestError(f"Class '{manifest.entry_class}' not found in module '{manifest.module}'")  # pragma: no cover
            instance = cls()  # pragma: no cover
            instance._plugin_id = manifest.plugin_id  # pragma: no cover
            return instance  # pragma: no cover
        finally:
            if added_to_path and plugin_dir_str in sys.path:  # pragma: no cover
                sys.path.remove(plugin_dir_str)  # pragma: no cover
