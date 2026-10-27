"""Generic settings management actions for Zelyr.

Provides framework actions for querying and modifying configuration at runtime.
These actions integrate with the action executor and are planner-friendly.

Actions:
- get_setting(key) - retrieve a setting value
- update_setting(key, value) - modify a setting with validation
- list_settings(category) - list available settings
"""

import re
from typing import Tuple, Optional, Dict, Any
from utils.logger import get_logger
from core.config_service import get_config_service
from core.settings_registry import get_settings_registry, SettingCategory

logger = get_logger(__name__)


def parse_kv_params(params: str) -> Dict[str, str]:
    """Parse key="value" parameter string.
    
    Args:
        params: String like 'key="value" other="data"'
        
    Returns:
        Dict of parsed key-value pairs
    """
    result = {}
    if not params:
        return result
    
    # Try double quotes
    for match in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', params):
        result[match.group(1)] = match.group(2)
    
    # Try single quotes
    for match in re.finditer(r"(\w+)\s*=\s*'([^']*)'", params):
        result[match.group(1)] = match.group(2)
    
    return result


def get_setting(params: str) -> Tuple[bool, str]:
    """Get a setting value.
    
    INSTRUCTION: get_setting key="ollama.model"
    
    Args:
        params: Parameters with key="..."
        
    Returns:
        (success, message)
    """
    kv = parse_kv_params(params)
    key = kv.get("key", "").strip()
    
    if not key:
        return False, "Missing key parameter"
    
    config = get_config_service()
    registry = get_settings_registry()
    
    # Check if setting exists
    metadata = registry.get(key)
    if metadata is None:
        return False, f"Unknown setting: {key}"
    
    value = config.get(key)
    return True, f"{key}: {value} ({metadata.setting_type.__name__})"


def update_setting(params: str) -> Tuple[bool, str]:
    """Update a setting value with validation.
    
    INSTRUCTION: update_setting key="ollama.model" value="qwen2.5:72b"
    
    Args:
        params: Parameters with key="..." value="..."
        
    Returns:
        (success, message)
    """
    kv = parse_kv_params(params)
    key = kv.get("key", "").strip()
    value = kv.get("value", "").strip()
    
    if not key:
        return False, "Missing key parameter"
    if not value:
        return False, "Missing value parameter"
    
    config = get_config_service()
    registry = get_settings_registry()
    
    # Check if setting exists
    metadata = registry.get(key)
    if metadata is None:
        return False, f"Unknown setting: {key}"
    
    # Check if mutable
    if not metadata.mutable:
        return False, f"Setting is immutable: {key}"
    
    # Type coercion for common types
    try:
        if metadata.setting_type == bool:
            if value.lower() in ("true", "1", "yes", "on"):
                typed_value = True
            elif value.lower() in ("false", "0", "no", "off"):
                typed_value = False
            else:
                return False, f"Invalid boolean value: {value}"
        elif metadata.setting_type == int:
            typed_value = int(value)
        elif metadata.setting_type == float:
            typed_value = float(value)
        else:
            typed_value = value
    except (ValueError, TypeError) as e:
        return False, f"Type conversion failed: {str(e)}"
    
    # Update the setting
    success, error = config.update(
        key=key,
        value=typed_value,
        changed_by="user",
    )
    
    if not success:
        return False, error
    
    msg = f"Updated {key} to {typed_value}"
    if metadata.requires_restart:
        msg += " (restart required)"
    
    return True, msg


def list_settings(params: str) -> Tuple[bool, str]:
    """List all available settings or settings in a category.
    
    INSTRUCTION: list_settings
    INSTRUCTION: list_settings category="ollama"
    
    Args:
        params: Optional category parameter
        
    Returns:
        (success, message)
    """
    registry = get_settings_registry()
    kv = parse_kv_params(params)
    category = kv.get("category", "").strip()
    
    if category:
        try:
            cat = SettingCategory(category)
            settings = registry.list_by_category(cat)
        except ValueError:
            available = [c.value for c in SettingCategory]
            return False, f"Unknown category. Available: {', '.join(available)}"
    else:
        settings = registry.list_all()
    
    if not settings:
        return True, "No settings found"
    
    # Format as human-readable list
    lines = []
    for key, metadata in sorted(settings.items()):
        mutable = "✓" if metadata.mutable else "✗"
        restart = "⟳" if metadata.requires_restart else " "
        lines.append(f"  {key:30s} [{mutable}] [{restart}] - {metadata.description}")
    
    header = f"Settings ({len(settings)})"
    if category:
        header += f" in category: {category}"
    
    return True, header + "\n" + "\n".join(lines)


def get_setting_info(params: str) -> Tuple[bool, str]:
    """Get detailed information about a specific setting.
    
    INSTRUCTION: get_setting_info key="ollama.model"
    
    Args:
        params: Parameters with key="..."
        
    Returns:
        (success, message)
    """
    kv = parse_kv_params(params)
    key = kv.get("key", "").strip()
    
    if not key:
        return False, "Missing key parameter"
    
    config = get_config_service()
    registry = get_settings_registry()
    
    metadata = registry.get(key)
    if metadata is None:
        return False, f"Unknown setting: {key}"
    
    current_value = config.get(key)
    
    lines = [
        f"Setting: {key}",
        f"Category: {metadata.category.value}",
        f"Type: {metadata.setting_type.__name__}",
        f"Description: {metadata.description}",
        f"Current value: {current_value}",
        f"Default value: {metadata.default}",
        f"Mutable: {'Yes' if metadata.mutable else 'No'}",
        f"Requires restart: {'Yes' if metadata.requires_restart else 'No'}",
    ]
    
    if metadata.allowed_values:
        lines.append(f"Allowed values: {metadata.allowed_values}")
    
    return True, "\n".join(lines)


# Action registry: these will be registered in executor
SETTINGS_ACTIONS = {
    "get_setting": get_setting,
    "update_setting": update_setting,
    "list_settings": list_settings,
    "get_setting_info": get_setting_info,
}
