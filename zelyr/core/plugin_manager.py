from zelyr.compat import config_dir
import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from typing import Dict

from zelyr.sdk.tool import ZelyrTool
from zelyr.config import get_config

logger = logging.getLogger(__name__)

PLUGIN_DIR = config_dir() / "plugins"

class PluginManager:
    def __init__(self):
        self.active_tools: Dict[str, ZelyrTool] = {}

    def load_user_tools(self) -> list[ZelyrTool]:
        """Load and return all user-defined third-party tools."""
        self.load_plugins()
        return list(self.active_tools.values())

    def load_plugins(self):
        self.active_tools.clear()
        config = get_config()
        if not getattr(config, 'allow_third_party_plugins', False):
            logger.warning("Third-party plugins are disabled in config. Skipping plugin load.")
            return

        if not PLUGIN_DIR.exists():
            PLUGIN_DIR.mkdir(parents=True, exist_ok=True)

        for py_file in PLUGIN_DIR.glob("*.py"):
            try:
                module_name = f"zelyr_plugin_{py_file.stem}"
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    # Add to sys.modules so it can do relative imports if needed
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        from zelyr.tools.core import BaseTool
                        if (issubclass(obj, ZelyrTool) and obj is not ZelyrTool) or (
                            issubclass(obj, BaseTool) and obj is not BaseTool
                        ):
                            instance = obj()
                            from zelyr.core.plugin_sandbox import SandboxedPluginRunner
                            runner = SandboxedPluginRunner(instance)
                            self.active_tools[instance.name] = runner
                            logger.info(f"Loaded isolated plugin tool: {instance.name}")
            except Exception as e:
                logger.error(f"Failed to load plugin {py_file.name}: {e}")
