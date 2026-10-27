"""PluginManager — discovery, loading, and EventBus integration.

Scans ``~/.zelyr/plugins/`` (configurable) for subdirectories containing a
``plugin.toml``.  For each found manifest:

1. Validates the manifest via ``PluginLoader``.
2. Runs the static AST security scan via ``static_check_plugin_source``.
3. Instantiates the plugin class.
4. Calls ``plugin.on_load(context)``.
5. Wraps each tool in a ``SandboxedToolProxy``.
6. Registers the proxied tools in the ``core.Registry``.
7. Subscribes ``plugin.on_event`` to the global EventBus.
8. Emits ``plugin.loaded`` or ``plugin.error`` events.
"""
from __future__ import annotations
from zelyr.compat import state_dir
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from zelyr.plugins.zelyr_plugin import ZelyrPlugin
from zelyr.plugins.exceptions import PluginError, SandboxSecurityViolation
from zelyr.plugins.loader import PluginLoader, PluginManifest
from zelyr.plugins.sandbox import SandboxedToolProxy, static_check_plugin_source
logger = logging.getLogger(__name__)
DEFAULT_PLUGIN_DIR = state_dir() / 'plugins'

class _PluginLoadContext:
    """Minimal context object passed to ``plugin.on_load``."""

    class secrets:
        """Auto-generated docstring.

"""

        @staticmethod
        def get(key: str, default: Any=None) -> Any:
            """Auto-generated docstring.

Args:
    key: Argument.
    default: Argument.

Returns:
    Return value.
"""
            import os  # pragma: no cover
            return os.environ.get(key, default)  # pragma: no cover

class PluginManager:
    """Discovers, validates, sandboxes, and registers all plugins.

    Typical usage (called from ``Engine.initialize`` or the CLI)::

        manager = PluginManager(registry=engine.registry, event_bus=engine.event_bus)
        manager.load_all()
    """

    def __init__(self, registry: Any, event_bus: Any, plugin_root: Path=DEFAULT_PLUGIN_DIR, sandbox_timeout: int=30) -> None:
        """Auto-generated docstring.

Args:
    registry: Argument.
    event_bus: Argument.
    plugin_root: Argument.
    sandbox_timeout: Argument.

Returns:
    Return value.
"""
        self._registry = registry  # pragma: no cover
        self._event_bus = event_bus  # pragma: no cover
        self._plugin_root = plugin_root  # pragma: no cover
        self._sandbox_timeout = sandbox_timeout  # pragma: no cover
        self._loader = PluginLoader()  # pragma: no cover
        self._loaded: Dict[str, ZelyrPlugin] = {}  # pragma: no cover

    def load_all(self) -> Dict[str, bool]:
        """Discover and load all plugins found in ``plugin_root``.

        Returns a dict of ``{plugin_id: success}`` for reporting.
        """
        results: Dict[str, bool] = {}  # pragma: no cover
        if not self._plugin_root.is_dir():  # pragma: no cover
            logger.debug('Plugin root %s does not exist — no plugins loaded.', self._plugin_root)  # pragma: no cover
            return results  # pragma: no cover
        for candidate in sorted(self._plugin_root.iterdir()):  # pragma: no cover
            if not candidate.is_dir():  # pragma: no cover
                continue  # pragma: no cover
            toml_path = candidate / 'plugin.toml'  # pragma: no cover
            if not toml_path.is_file():  # pragma: no cover
                continue  # pragma: no cover
            try:  # pragma: no cover
                success = self._load_one(candidate)  # pragma: no cover
                results[candidate.name] = success  # pragma: no cover
            except Exception as exc:  # pragma: no cover
                logger.error('Failed to load plugin at %s: %s', candidate, exc)  # pragma: no cover
                results[candidate.name] = False  # pragma: no cover
        return results  # pragma: no cover

    def load_from_path(self, plugin_dir: Path) -> bool:
        """Load a single plugin from the given directory path."""
        return self._load_one(plugin_dir)  # pragma: no cover

    def unload(self, plugin_id: str) -> None:
        """Call ``on_shutdown`` and de-register all tools from a loaded plugin."""
        plugin = self._loaded.get(plugin_id)  # pragma: no cover
        if plugin is None:  # pragma: no cover
            logger.warning("Cannot unload unknown plugin '%s'", plugin_id)  # pragma: no cover
            return  # pragma: no cover
        try:  # pragma: no cover
            plugin.on_shutdown()  # pragma: no cover
        except Exception as exc:  # pragma: no cover
            logger.error("Error during shutdown of plugin '%s': %s", plugin_id, exc)  # pragma: no cover
        for tool_id in list(self._registry.list_tools().keys()):  # pragma: no cover
            if tool_id.startswith(f'{plugin_id}::'):  # pragma: no cover
                self._registry.unregister_tool(tool_id)  # pragma: no cover
        self._registry.unregister_plugin(plugin_id)  # pragma: no cover
        del self._loaded[plugin_id]  # pragma: no cover
        logger.info("Plugin '%s' unloaded.", plugin_id)  # pragma: no cover

    @property
    def loaded_plugins(self) -> List[str]:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return list(self._loaded.keys())  # pragma: no cover

    def _load_one(self, plugin_dir: Path) -> bool:
        """Load, validate, and register a single plugin.  Returns True on success."""
        plugin_id = '(unknown)'  # pragma: no cover
        try:  # pragma: no cover
            manifest: PluginManifest = self._loader.load_manifest(plugin_dir)  # pragma: no cover
            plugin_id = manifest.plugin_id  # pragma: no cover
            logger.info("Loading plugin '%s' v%s …", manifest.name, manifest.version)  # pragma: no cover
            static_check_plugin_source(plugin_dir, manifest.permissions, plugin_id)  # pragma: no cover
            instance: ZelyrPlugin = self._loader.instantiate(manifest)  # pragma: no cover
            instance.on_load(_PluginLoadContext())  # pragma: no cover
            self._event_bus.subscribe('*', instance.on_event)  # pragma: no cover
            for tool_def in instance.tools:  # pragma: no cover
                proxy = SandboxedToolProxy(manifest=manifest, tool_def=tool_def, timeout=self._sandbox_timeout)  # pragma: no cover
                self._registry.register_tool(proxy.tool_id, proxy)  # pragma: no cover
                logger.debug("  Registered sandboxed tool '%s' from plugin '%s'", proxy.tool_id, plugin_id)  # pragma: no cover
            self._registry.register_plugin(plugin_id, instance)  # pragma: no cover
            self._loaded[plugin_id] = instance  # pragma: no cover
            self._emit('plugin.loaded', {'plugin_id': plugin_id, 'version': manifest.version, 'tool_count': len(instance.tools)})  # pragma: no cover
            logger.info("Plugin '%s' loaded successfully (%d tools).", plugin_id, len(instance.tools))  # pragma: no cover
            return True  # pragma: no cover
        except SandboxSecurityViolation as exc:  # pragma: no cover
            logger.error("Security violation loading plugin '%s': %s", plugin_id, exc)  # pragma: no cover
            self._emit('plugin.error', {'plugin_id': plugin_id, 'reason': str(exc), 'type': 'security'})  # pragma: no cover
            return False  # pragma: no cover
        except PluginError as exc:  # pragma: no cover
            logger.error("Plugin error loading '%s': %s", plugin_id, exc)  # pragma: no cover
            self._emit('plugin.error', {'plugin_id': plugin_id, 'reason': str(exc), 'type': 'manifest'})  # pragma: no cover
            return False  # pragma: no cover
        except Exception as exc:  # pragma: no cover
            logger.error('Unexpected error loading plugin at %s: %s', plugin_dir, exc, exc_info=True)  # pragma: no cover
            self._emit('plugin.error', {'plugin_id': plugin_id, 'reason': str(exc), 'type': 'unknown'})  # pragma: no cover
            return False  # pragma: no cover

    def _emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Auto-generated docstring.

Args:
    event_type: Argument.
    data: Argument.

Returns:
    Return value.
"""
        try:  # pragma: no cover
            from zelyr.core.events import Event  # pragma: no cover
            self._event_bus.publish(Event(event_type=event_type, source='PluginManager', data=data))  # pragma: no cover
        except Exception:  # pragma: no cover
            pass  # pragma: no cover
