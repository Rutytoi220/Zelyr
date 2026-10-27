"""Zelyr Plugin SDK.

Provides the `@zelyr.tool` decorator for external plugin developers to easily
register custom tools into the Zelyr kernel.
"""

from typing import Callable, Any, Dict, Optional
import functools

# Global registry for discovered plugin tools
_PLUGIN_REGISTRY: Dict[str, Dict[str, Any]] = {}

def tool(name: Optional[str] = None, description: Optional[str] = None):
    """Decorator to expose a Python function as an Zelyr tool.
    
    Args:
        name: Name of the tool. Defaults to the function's name.
        description: Description of what the tool does. Defaults to the docstring.
    """
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_desc = description or (func.__doc__ or "").strip()
        
        # In a real implementation, we would inspect the signature to generate a JSON Schema.
        # For simplicity in this SDK stub, we rely on the registry loader to handle it.
        _PLUGIN_REGISTRY[tool_name] = {
            "name": tool_name,
            "description": tool_desc,
            "func": func,
            "module": func.__module__
        }
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

def get_registered_plugins() -> Dict[str, Dict[str, Any]]:
    """Return all currently registered plugin tools."""
    return _PLUGIN_REGISTRY
