"""Base plugin class for Zelyr."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
logger = logging.getLogger(__name__)

class BasePlugin(ABC):
    """Base class for Zelyr plugins."""

    def __init__(self, plugin_id: str, name: str, version: str='1.0.0'):
        """Auto-generated docstring.

Args:
    plugin_id: Argument.
    name: Argument.
    version: Argument.

Returns:
    Return value.
"""
        self.plugin_id = plugin_id  # pragma: no cover
        self.name = name  # pragma: no cover
        self.version = version  # pragma: no cover
        self.enabled = False  # pragma: no cover
        self.config: Dict[str, Any] = {}  # pragma: no cover

    @abstractmethod
    def initialize(self, config: Optional[Dict]=None) -> bool:
        """Initialize the plugin."""
        pass  # pragma: no cover

    @abstractmethod
    def shutdown(self) -> bool:
        """Shutdown the plugin."""
        pass  # pragma: no cover

    def enable(self) -> None:
        """Enable the plugin."""
        self.enabled = True  # pragma: no cover
        logger.info(f'Plugin {self.plugin_id} enabled')  # pragma: no cover

    def disable(self) -> None:
        """Disable the plugin."""
        self.enabled = False  # pragma: no cover
        logger.info(f'Plugin {self.plugin_id} disabled')  # pragma: no cover

    def is_enabled(self) -> bool:
        """Check if plugin is enabled."""
        return self.enabled  # pragma: no cover

    def get_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        return {'plugin_id': self.plugin_id, 'name': self.name, 'version': self.version, 'enabled': self.enabled, 'config': self.config}  # pragma: no cover
