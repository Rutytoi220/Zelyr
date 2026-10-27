from zelyr.compat import data_dir
import os
import logging
import importlib.util
import inspect
from pathlib import Path
from zelyr.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

class SkillCompilerEngine:
    """Compiles verified Python code into reusable Zelyr skills."""
    
    def __init__(self, registry: ToolRegistry = None):
        self.registry = registry or ToolRegistry()
        self.skills_dir = data_dir() / "skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        
    def compile_skill(self, skill_name: str, code: str, description: str = "") -> bool:
        """
        Takes raw python code, wraps it into a BaseTool subclass,
        saves it to disk, and dynamically registers it.
        """
        # Santize skill name
        skill_name = skill_name.lower().replace(" ", "_").replace("-", "_")
        class_name = "".join(word.capitalize() for word in skill_name.split("_")) + "Tool"
        
        file_path = self.skills_dir / f"{skill_name}.py"
        
        # 1. Generate the BaseTool wrapper code
        wrapped_code = f'''\
"""Auto-compiled skill: {skill_name}"""
from typing import Dict, Any
from zelyr.tools import BaseTool, ToolResult

# Original Code:
{code}

class {class_name}(BaseTool):
    def __init__(self):
        super().__init__(
            tool_id="{skill_name}",
            name="{skill_name}",
            description="""{description or f"Auto-compiled skill {skill_name}"}"""
        )
        # Assuming single string parameter 'input' for simplicity of auto-compiled scripts
        from zelyr.tools import ToolParameter
        self.add_parameter(ToolParameter(
            name="input",
            type="string",
            description="Input for the skill"
        ))

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        try:
            # We assume the original code defines a main function matching the skill name
            # Or we just exec it if it's a script. 
            # For this simple compiler, if there's a function with the same name, call it:
            input_val = params.get("input", "")
            
            if "{skill_name}" in globals() and callable(globals()["{skill_name}"]):
                res = globals()["{skill_name}"](input_val)
                return ToolResult(success=True, output=str(res))
            else:
                return ToolResult(success=False, error="Could not find callable {skill_name} in compiled code.")
        except Exception as e:
            return ToolResult(success=False, error=str(e))
'''

        # 2. Write to disk safely
        try:
            with open(file_path, "w") as f:
                f.write(wrapped_code)
            logger.info(f"Skill '{skill_name}' written to {file_path}")
        except Exception as e:
            logger.error(f"Failed to write skill {skill_name}: {e}")
            return False
            
        # 3. Dynamically import and register
        try:
            spec = importlib.util.spec_from_file_location(f"zelyr.skills.{skill_name}", str(file_path))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find the class and instantiate it
                tool_class = getattr(module, class_name)
                tool_instance = tool_class()
                
                # Register in the active ToolRegistry
                if self.registry:
                    # Unregister if it already exists to allow overwriting
                    if tool_instance.tool_id in self.registry:
                        self.registry.unregister_tool(tool_instance.tool_id)
                    self.registry.register(tool_instance)
                
                logger.info(f"Skill '{skill_name}' successfully compiled and registered!")
                return True
        except Exception as e:
            logger.error(f"Failed to dynamically load skill {skill_name}: {e}")
            return False
            
        return False
