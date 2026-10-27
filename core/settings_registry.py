"""Settings Registry for Zelyr.

Defines the typed, centralized registry of all configurable settings.
Supports metadata, validation, and future plugin-defined settings.
"""

from typing import Any, Callable, Optional, Dict, List, Protocol, runtime_checkable
from enum import Enum
import inspect
from pathlib import Path
from zelyr.compat import resolve_file


@runtime_checkable
class SettingValidator(Protocol):
    """Protocol for custom setting validators."""
    
    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate a setting value.
        
        Args:
            value: The value to validate
            
        Returns:
            (is_valid, error_message)
            If valid: (True, None)
            If invalid: (False, "error description")
        """
        ...


class SettingCategory(str, Enum):
    """Categories for organizing settings."""
    OLLAMA = "ollama"
    AGENT = "agent"
    SECURITY = "security"
    LOGGING = "logging"
    STORAGE = "storage"
    OBSERVER = "observer"
    CONTEXT = "context"
    UI = "ui"
    ADVANCED = "advanced"


class SettingMetadata:
    """Metadata describing a single setting."""
    
    def __init__(
        self,
        key: str,
        category: SettingCategory,
        setting_type: type,
        default: Any,
        description: str,
        mutable: bool = True,
        requires_restart: bool = False,
        validator: Optional[Callable] = None,
        allowed_values: Optional[List[Any]] = None,
    ):
        """Initialize setting metadata.
        
        Args:
            key: Dotted key (e.g., "ollama.model", "security.sandbox_mode")
            category: Category for organization
            setting_type: Python type (str, int, bool, list, dict, etc.)
            default: Default value
            description: Human-readable description
            mutable: Whether this setting can be changed at runtime
            requires_restart: Whether change requires app restart
            validator: Optional callable that validates the value
            allowed_values: Optional list of allowed values
        """
        self.key = key
        self.category = category
        self.setting_type = setting_type
        self.default = default
        self.description = description
        self.mutable = mutable
        self.requires_restart = requires_restart
        self.validator = validator
        self.allowed_values = allowed_values
    
    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate a value against this setting's constraints.
        
        Args:
            value: Value to validate
            
        Returns:
            (is_valid, error_message)
        """
        # Type check
        if not isinstance(value, self.setting_type):
            return False, f"Expected {self.setting_type.__name__}, got {type(value).__name__}"
        
        # Allowed values check
        if self.allowed_values is not None:
            if value not in self.allowed_values:
                return False, f"Value must be one of: {self.allowed_values}"
        
        # Custom validator
        if self.validator is not None:
            if callable(self.validator):
                # Check if it's a protocol or a simple function
                try:
                    result = self.validator(value)
                    if isinstance(result, tuple):
                        return result
                    elif isinstance(result, bool):
                        return (result, None if result else "Validation failed")
                except Exception as e:
                    return False, f"Validation error: {str(e)}"
            else:
                return False, "Validator is not callable"
        
        return True, None


class SettingsRegistry:
    """Centralized typed registry of all Zelyr settings."""
    
    def __init__(self):
        """Initialize the settings registry with built-in settings."""
        self._settings: Dict[str, SettingMetadata] = {}
        self._register_builtin_settings()
    
    def _register_builtin_settings(self) -> None:
        """Register all built-in Zelyr settings."""
        
        # ============ OLLAMA SETTINGS ============
        self.register(SettingMetadata(
            key="ollama.url",
            category=SettingCategory.OLLAMA,
            setting_type=str,
            default="http://127.0.0.1:11434",
            description="Ollama server URL",
            mutable=True,
            requires_restart=True,
        ))
        
        self.register(SettingMetadata(
            key="ollama.model",
            category=SettingCategory.OLLAMA,
            setting_type=str,
            default="qwen2.5:14b",
            description="Default Ollama model name",
            mutable=True,
            requires_restart=False,
        ))
        
        self.register(SettingMetadata(
            key="ollama.timeout",
            category=SettingCategory.OLLAMA,
            setting_type=int,
            default=60,
            description="Ollama request timeout in seconds",
            mutable=True,
            requires_restart=False,
            validator=lambda x: (x > 0, "Timeout must be positive"),
        ))
        
        # ============ AGENT SETTINGS ============
        self.register(SettingMetadata(
            key="agent.multi_action",
            category=SettingCategory.AGENT,
            setting_type=bool,
            default=True,
            description="Allow multiple actions per request",
            mutable=True,
            requires_restart=False,
        ))
        
        self.register(SettingMetadata(
            key="agent.stop_on_error",
            category=SettingCategory.AGENT,
            setting_type=bool,
            default=True,
            description="Stop action sequence on error",
            mutable=True,
            requires_restart=False,
        ))
        
        self.register(SettingMetadata(
            key="agent.max_actions_per_request",
            category=SettingCategory.AGENT,
            setting_type=int,
            default=5,
            description="Maximum number of actions per request",
            mutable=True,
            requires_restart=False,
            validator=lambda x: (x > 0, "Max actions must be positive"),
        ))
        
        self.register(SettingMetadata(
            key="agent.enable_context_memory",
            category=SettingCategory.AGENT,
            setting_type=bool,
            default=True,
            description="Enable context memory tracking",
            mutable=True,
            requires_restart=False,
        ))
        
        # ============ SECURITY SETTINGS ============
        self.register(SettingMetadata(
            key="security.allow_any_app",
            category=SettingCategory.SECURITY,
            setting_type=bool,
            default=False,
            description="Allow any application to be opened",
            mutable=True,
            requires_restart=False,
        ))
        
        self.register(SettingMetadata(
            key="security.require_confirmation_for_commands",
            category=SettingCategory.SECURITY,
            setting_type=bool,
            default=False,
            description="Require confirmation before running commands",
            mutable=True,
            requires_restart=False,
        ))
        
        self.register(SettingMetadata(
            key="security.sandbox_mode",
            category=SettingCategory.SECURITY,
            setting_type=bool,
            default=False,
            description="Enable sandbox mode for safer execution",
            mutable=True,
            requires_restart=True,
        ))
        
        # ============ LOGGING SETTINGS ============
        self.register(SettingMetadata(
            key="logging.level",
            category=SettingCategory.LOGGING,
            setting_type=str,
            default="INFO",
            description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
            mutable=True,
            requires_restart=False,
            allowed_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        ))
        
        self.register(SettingMetadata(
            key="logging.file",
            category=SettingCategory.LOGGING,
            setting_type=str,
            default="zelyr.log",
            description="Log file path",
            mutable=True,
            requires_restart=False,
        ))
        
        self.register(SettingMetadata(
            key="logging.log_actions",
            category=SettingCategory.LOGGING,
            setting_type=bool,
            default=True,
            description="Log all executed actions",
            mutable=True,
            requires_restart=False,
        ))
        
        self.register(SettingMetadata(
            key="logging.log_errors",
            category=SettingCategory.LOGGING,
            setting_type=bool,
            default=True,
            description="Log all errors",
            mutable=True,
            requires_restart=False,
        ))
        
        # ============ STORAGE SETTINGS ============
        self.register(SettingMetadata(
            key="storage.db_path",
            category=SettingCategory.STORAGE,
            setting_type=str,
            default=str(resolve_file(Path("data/zelyr.db"), Path("data/axiom.db"))),
            description="SQLite database path",
            mutable=False,  # Should not change at runtime
            requires_restart=True,
        ))
        
        # ============ OBSERVER SETTINGS ============
        self.register(SettingMetadata(
            key="observer.enabled",
            category=SettingCategory.OBSERVER,
            setting_type=bool,
            default=True,
            description="Enable observer engine",
            mutable=True,
            requires_restart=False,
        ))
        
        self.register(SettingMetadata(
            key="observer.lookback",
            category=SettingCategory.OBSERVER,
            setting_type=int,
            default=50,
            description="Number of past events to analyze",
            mutable=True,
            requires_restart=False,
            validator=lambda x: (x > 0, "Lookback must be positive"),
        ))
        
        self.register(SettingMetadata(
            key="observer.frequency_threshold",
            category=SettingCategory.OBSERVER,
            setting_type=int,
            default=3,
            description="Threshold for frequency-based patterns",
            mutable=True,
            requires_restart=False,
            validator=lambda x: (x > 0, "Threshold must be positive"),
        ))
        
        self.register(SettingMetadata(
            key="observer.frequency_window_hours",
            category=SettingCategory.OBSERVER,
            setting_type=int,
            default=24,
            description="Time window for frequency analysis in hours",
            mutable=True,
            requires_restart=False,
            validator=lambda x: (x > 0, "Window must be positive"),
        ))
        
        self.register(SettingMetadata(
            key="observer.temporal_min_days",
            category=SettingCategory.OBSERVER,
            setting_type=int,
            default=2,
            description="Minimum days for temporal patterns",
            mutable=True,
            requires_restart=False,
            validator=lambda x: (x > 0, "Days must be positive"),
        ))
        
        self.register(SettingMetadata(
            key="observer.temporal_tolerance_minutes",
            category=SettingCategory.OBSERVER,
            setting_type=int,
            default=30,
            description="Tolerance for temporal matching in minutes",
            mutable=True,
            requires_restart=False,
            validator=lambda x: (x > 0, "Tolerance must be positive"),
        ))
        
        # ============ CONTEXT SETTINGS ============
        self.register(SettingMetadata(
            key="context.track_running_apps",
            category=SettingCategory.CONTEXT,
            setting_type=bool,
            default=True,
            description="Track running applications",
            mutable=True,
            requires_restart=False,
        ))
        
        self.register(SettingMetadata(
            key="context.track_last_folder",
            category=SettingCategory.CONTEXT,
            setting_type=bool,
            default=True,
            description="Track last accessed folder",
            mutable=True,
            requires_restart=False,
        ))
        
        self.register(SettingMetadata(
            key="context.track_last_command",
            category=SettingCategory.CONTEXT,
            setting_type=bool,
            default=True,
            description="Track last executed command",
            mutable=True,
            requires_restart=False,
        ))
    
    def register(self, metadata: SettingMetadata) -> None:
        """Register a setting.
        
        Args:
            metadata: SettingMetadata describing the setting
            
        Raises:
            ValueError: If setting key already registered
        """
        if metadata.key in self._settings:
            raise ValueError(f"Setting already registered: {metadata.key}")
        self._settings[metadata.key] = metadata
    
    def get(self, key: str) -> Optional[SettingMetadata]:
        """Get setting metadata by key.
        
        Args:
            key: Setting key (e.g., "ollama.model")
            
        Returns:
            SettingMetadata or None if not found
        """
        return self._settings.get(key)
    
    def list_all(self) -> Dict[str, SettingMetadata]:
        """Get all registered settings.
        
        Returns:
            Dictionary of key -> SettingMetadata
        """
        return dict(self._settings)
    
    def list_by_category(self, category: SettingCategory) -> Dict[str, SettingMetadata]:
        """Get all settings in a category.
        
        Args:
            category: SettingCategory to filter by
            
        Returns:
            Dictionary of key -> SettingMetadata
        """
        return {
            k: v for k, v in self._settings.items()
            if v.category == category
        }
    
    def list_mutable(self) -> Dict[str, SettingMetadata]:
        """Get all mutable settings.
        
        Returns:
            Dictionary of key -> SettingMetadata
        """
        return {k: v for k, v in self._settings.items() if v.mutable}
    
    def list_requiring_restart(self) -> Dict[str, SettingMetadata]:
        """Get all settings that require restart on change.
        
        Returns:
            Dictionary of key -> SettingMetadata
        """
        return {k: v for k, v in self._settings.items() if v.requires_restart}


# Global singleton instance
_REGISTRY: Optional[SettingsRegistry] = None


def get_settings_registry() -> SettingsRegistry:
    """Get or create the global settings registry.
    
    Returns:
        The global SettingsRegistry instance
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = SettingsRegistry()
    return _REGISTRY
