"""Configuration Service for Zelyr.

Manages runtime configuration: loading, validation, persistence, and change notification.
Provides a central interface for getting/setting values with full audit trail.
"""

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, List, Set
from dataclasses import dataclass, asdict
from copy import deepcopy

from utils.logger import get_logger
from core.settings_registry import get_settings_registry, SettingMetadata

logger = get_logger(__name__)


@dataclass
class SettingChange:
    """Record of a setting change."""
    timestamp: float
    key: str
    old_value: Any
    new_value: Any
    changed_by: str = "system"  # user, api, plugin, system
    success: bool = True
    error: Optional[str] = None
    requires_restart: bool = False


class ConfigService:
    """Central service for runtime configuration management.
    
    Responsibilities:
    - Load configuration from JSON
    - Validate settings against registry
    - Persist changes back to JSON
    - Notify subscribers of changes
    - Support rollback on invalid updates
    - Maintain audit trail
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the config service.
        
        Args:
            config_path: Path to config.json. If None, uses default location.
        """
        if config_path is None:
            config_path = Path(__file__).resolve().parents[2] / "config" / "config.json"
        
        self.config_path = Path(config_path)
        self.registry = get_settings_registry()
        
        # Runtime configuration
        self._config: Dict[str, Any] = {}
        
        # Subscribers: key -> list of callbacks
        self._subscribers: Dict[str, List[Callable]] = {}
        
        # Audit trail
        self._change_history: List[SettingChange] = []
        
        # Load initial configuration
        self._load()
    
    def _load(self) -> None:
        """Load configuration from JSON file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            self._config = self._get_defaults()
            return
        
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            
            # Merge with defaults to ensure all keys exist
            defaults = self._get_defaults()
            self._config = self._merge_deep(defaults, loaded)
            
            logger.info(f"Loaded config from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}, using defaults")
            self._config = self._get_defaults()
    
    def _get_defaults(self) -> Dict[str, Any]:
        """Get default configuration from registry.
        
        Returns:
            Dictionary with default values for all registered settings
        """
        defaults = {}
        for key, metadata in self.registry.list_all().items():
            # Use dotted key to build nested dict
            self._set_nested_value(defaults, key, metadata.default)
        return defaults
    
    def _merge_deep(self, base: Dict, override: Dict) -> Dict:
        """Deep merge override dict into base dict.
        
        Args:
            base: Base dictionary
            override: Override values
            
        Returns:
            Merged dictionary
        """
        result = deepcopy(base)
        for key, value in override.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._merge_deep(result[key], value)
            else:
                result[key] = value
        return result
    
    def _save(self) -> None:
        """Persist current configuration to JSON file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=2)
            logger.info(f"Saved config to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value by dotted key.
        
        Args:
            key: Dotted key (e.g., "ollama.model")
            default: Default value if key not found
            
        Returns:
            Setting value or default
        """
        value = self._get_nested_value(self._config, key)
        if value is None:
            metadata = self.registry.get(key)
            if metadata is not None:
                value = metadata.default
            elif default is not None:
                value = default
        return value
    
    def update(
        self,
        key: str,
        value: Any,
        changed_by: str = "system",
        require_confirmation: bool = False,
    ) -> tuple[bool, Optional[str]]:
        """Update a setting with validation and persistence.
        
        Args:
            key: Dotted key (e.g., "ollama.model")
            value: New value
            changed_by: Source of change (user, api, plugin, system)
            require_confirmation: If True, returns without persisting
            
        Returns:
            (success, error_message)
            - If success: (True, None)
            - If failed: (False, "error reason")
        """
        # Get metadata
        metadata = self.registry.get(key)
        if metadata is None:
            error = f"Unknown setting: {key}"
            logger.warning(error)
            return False, error
        
        # Check if mutable
        if not metadata.mutable:
            error = f"Setting is immutable: {key}"
            logger.warning(error)
            return False, error
        
        # Validate
        is_valid, validation_error = metadata.validate(value)
        if not is_valid:
            error = f"Validation failed for {key}: {validation_error}"
            logger.warning(error)
            return False, error
        
        # Get old value
        old_value = self.get(key)
        
        # Create change record
        change = SettingChange(
            timestamp=time.time(),
            key=key,
            old_value=old_value,
            new_value=value,
            changed_by=changed_by,
            requires_restart=metadata.requires_restart,
        )
        
        # If confirmation required, return without persisting
        if require_confirmation:
            logger.info(f"Setting change pending confirmation: {key}")
            return True, None
        
        # Persist the change
        try:
            self._set_nested_value(self._config, key, value)
            self._save()
            self._change_history.append(change)
            logger.info(f"Updated {key}: {old_value} -> {value}")
        except Exception as e:
            error = f"Failed to update {key}: {str(e)}"
            change.success = False
            change.error = str(e)
            self._change_history.append(change)
            return False, error
        
        # Notify subscribers
        self._notify_subscribers(key, old_value, value)
        
        return True, None
    
    def subscribe(self, key: str, callback: Callable[[str, Any, Any], None]) -> None:
        """Subscribe to changes for a setting.
        
        Args:
            key: Setting key to watch (use "*" for all settings)
            callback: Function(key, old_value, new_value) to call on change
        """
        if key not in self._subscribers:
            self._subscribers[key] = []
        self._subscribers[key].append(callback)
        logger.debug(f"Subscribed to changes for: {key}")
    
    def unsubscribe(self, key: str, callback: Callable) -> None:
        """Unsubscribe from changes.
        
        Args:
            key: Setting key
            callback: Callback to remove
        """
        if key in self._subscribers:
            self._subscribers[key] = [
                cb for cb in self._subscribers[key] if cb is not callback
            ]
    
    def _notify_subscribers(self, key: str, old_value: Any, new_value: Any) -> None:
        """Notify all subscribers of a setting change.
        
        Args:
            key: Setting key that changed
            old_value: Previous value
            new_value: New value
        """
        # Notify specific key subscribers
        if key in self._subscribers:
            for callback in self._subscribers[key]:
                try:
                    callback(key, old_value, new_value)
                except Exception as e:
                    logger.exception(f"Error in setting subscriber for {key}: {e}")
        
        # Notify wildcard subscribers
        if "*" in self._subscribers:
            for callback in self._subscribers["*"]:
                try:
                    callback(key, old_value, new_value)
                except Exception as e:
                    logger.exception(f"Error in wildcard setting subscriber: {e}")
    
    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration dictionary.
        
        Returns:
            Full config dict
        """
        return deepcopy(self._config)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get a section of configuration (e.g., "ollama", "security").
        
        Args:
            section: Section name
            
        Returns:
            Section dict or empty dict if not found
        """
        return deepcopy(self._config.get(section, {}))
    
    def get_change_history(self, limit: int = 100) -> List[SettingChange]:
        """Get audit trail of setting changes.
        
        Args:
            limit: Maximum number of changes to return
            
        Returns:
            List of SettingChange records
        """
        return self._change_history[-limit:]
    
    def rollback(self, change: SettingChange) -> tuple[bool, Optional[str]]:
        """Rollback a setting change to its previous value.
        
        Args:
            change: SettingChange record to rollback
            
        Returns:
            (success, error_message)
        """
        return self.update(
            key=change.key,
            value=change.old_value,
            changed_by="system",
        )
    
    def _get_nested_value(self, obj: Dict, dotted_key: str) -> Any:
        """Get value from nested dict using dotted key.
        
        Args:
            obj: Dictionary to search
            dotted_key: Dotted key (e.g., "ollama.model")
            
        Returns:
            Value or None if not found
        """
        keys = dotted_key.split(".")
        current = obj
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
            else:
                return None
        return current
    
    def _set_nested_value(self, obj: Dict, dotted_key: str, value: Any) -> None:
        """Set value in nested dict using dotted key.
        
        Args:
            obj: Dictionary to modify
            dotted_key: Dotted key (e.g., "ollama.model")
            value: Value to set
        """
        keys = dotted_key.split(".")
        current = obj
        
        # Navigate/create nested structure
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set final value
        current[keys[-1]] = value


# Global singleton instance
_CONFIG_SERVICE: Optional[ConfigService] = None


def get_config_service() -> ConfigService:
    """Get or create the global config service.
    
    Args:
        config_path: Optional path to config file
    
    Returns:
        The global ConfigService instance
    """
    global _CONFIG_SERVICE
    if _CONFIG_SERVICE is None:
        _CONFIG_SERVICE = ConfigService()
    return _CONFIG_SERVICE


def init_config_service(config_path: Optional[Path] = None) -> ConfigService:
    """Initialize the config service with a specific path.
    
    Args:
        config_path: Path to config.json
    
    Returns:
        The ConfigService instance
    """
    global _CONFIG_SERVICE
    _CONFIG_SERVICE = ConfigService(config_path)
    return _CONFIG_SERVICE
