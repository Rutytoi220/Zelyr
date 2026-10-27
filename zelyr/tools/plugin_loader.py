"""Dynamic Plugin Loader.

Scans ~/.config/zelyr/plugins/ on boot and loads any Python scripts
found into the provided ToolRegistry.
"""
from zelyr.compat import config_dir
import os
import sys
import importlib.util
import logging
from pathlib import Path
from zelyr.tools.base import ZelyrPlugin, DecoratorTool, zelyr_tool

logger = logging.getLogger(__name__)

# Legacy names that were historically importable from this module.
AxiomPlugin = ZelyrPlugin
axiom_tool = zelyr_tool

def load_plugins(registry) -> None:
    """Scan the plugins directory, discover tools, and register them."""
    plugins_dir = config_dir() / "plugins"
    
    if not plugins_dir.exists():
        plugins_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created plugins directory at {plugins_dir}")
        _generate_example_plugin(plugins_dir)
        return

    # Check if directory is empty (other than directories/hidden)
    has_py_files = any(p.suffix == '.py' for p in plugins_dir.iterdir() if p.is_file())
    if not has_py_files:
        _generate_example_plugin(plugins_dir)

    for filepath in plugins_dir.glob("*.py"):
        _load_module_and_register(filepath, registry)

def _generate_example_plugin(plugins_dir: Path) -> None:
    example_file = plugins_dir / "system_info_plugin.py.example"
    if not example_file.exists():
        example_content = '''"""
Example Zelyr Plugin

To use this plugin, remove the '.example' extension so it becomes a '.py' file.
Zelyr will dynamically load it on the next boot!
"""
from zelyr.tools.base import zelyr_tool

@zelyr_tool(
    name="get_uptime",
    description="Returns the current system uptime",
    parameters={
        "type": "object",
        "properties": {},
        "required": []
    }
)
async def get_uptime(**kwargs):
    import subprocess
    result = subprocess.check_output("uptime -p", shell=True, text=True).strip()
    return result
'''
        example_file.write_text(example_content)

def _load_module_and_register(filepath: Path, registry) -> None:
    """Dynamically import a Python file and register its tools."""
    module_name = f"zelyr.dynamic_plugins.{filepath.stem}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(filepath))
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Inspect the module
            count = 0
            for name in dir(module):
                obj = getattr(module, name)
                
                # Check for @zelyr_tool decorators
                if callable(obj) and (getattr(obj, "__zelyr_tool__", False) or getattr(obj, "__axiom_tool__", False)):
                    tool = DecoratorTool(
                        func=obj,
                        name=obj.__tool_name__,
                        description=obj.__tool_description__,
                        parameters=obj.__tool_parameters__,
                        requires_approval=getattr(obj, '__tool_requires_approval__', False)
                    )
                    registry.register(tool)
                    count += 1
                    
                # Check for ZelyrPlugin subclasses
                elif isinstance(obj, type) and issubclass(obj, ZelyrPlugin) and obj is not ZelyrPlugin:
                    # Instantiate the plugin
                    tool = obj()
                    registry.register(tool)
                    count += 1
                    
            if count > 0:
                logger.info(f"Loaded {count} tools from plugin: {filepath.name}")
    except Exception as e:
        logger.warning(f"Failed to load plugin {filepath.name}: {e}")
