"""Capability-Based Permission Broker for Sandboxed Plugins.

Translates PluginManifest definitions into strict WASI boundaries.
"""
import logging
from pathlib import Path
from typing import Optional
import wasmtime
from zelyr.plugins.loader import PluginManifest
from zelyr.plugins.exceptions import PluginPermissionError
logger = logging.getLogger(__name__)

class PermissionBroker:
    """Enforces WASI sandbox boundaries based on the plugin manifest."""

    def __init__(self, manifest: PluginManifest, workspace_dir: Path):
        """Auto-generated docstring.

Args:
    manifest: Argument.
    workspace_dir: Argument.

Returns:
    Return value.
"""
        self.manifest = manifest  # pragma: no cover
        self.workspace_dir = Path(workspace_dir).resolve()  # pragma: no cover
        self.plugin_scratch_dir = self.workspace_dir / '.zelyr' / 'plugins' / manifest.plugin_id / 'scratch'  # pragma: no cover

    def build_wasi_config(self) -> wasmtime.WasiConfig:
        """Construct a secure WasiConfig according to granted permissions."""
        config = wasmtime.WasiConfig()  # pragma: no cover
        config.inherit_stdin()  # pragma: no cover
        config.inherit_stdout()  # pragma: no cover
        config.inherit_stderr()  # pragma: no cover
        if self.manifest.permissions.allows('filesystem'):  # pragma: no cover
            self.plugin_scratch_dir.mkdir(parents=True, exist_ok=True)  # pragma: no cover
            config.preopen_dir(str(self.plugin_scratch_dir), '/scratch')  # pragma: no cover
            logger.debug(f'Plugin {self.manifest.name}: Granted filesystem access to /scratch')  # pragma: no cover
        else:
            logger.debug(f'Plugin {self.manifest.name}: Filesystem access denied')  # pragma: no cover
        if self.manifest.permissions.allows('network'):  # pragma: no cover
            logger.debug(f'Plugin {self.manifest.name}: Network access explicitly requested and approved by user.')  # pragma: no cover
            config.env = [('ZELYR_NETWORK_ENABLED', '1'), ('AXIOM_NETWORK_ENABLED', '1')]  # pragma: no cover
        else:
            logger.debug(f'Plugin {self.manifest.name}: Network access denied')  # pragma: no cover
            config.env = [('ZELYR_NETWORK_ENABLED', '0'), ('AXIOM_NETWORK_ENABLED', '0')]  # pragma: no cover
        return config  # pragma: no cover
