from abc import ABC, abstractmethod
from typing import Dict, Any, Union

class ZelyrTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """The tool name used by the LLM."""
        pass

    @property
    def tool_id(self) -> str:
        """Alias for tool name/identifier."""
        return self.name

    @property
    @abstractmethod
    def description(self) -> str:
        """The tool description."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON Schema of the tool parameters."""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> Union[dict, str]:
        """Execute the tool."""
        pass

# Legacy AXIOM API compatibility aliases
AxiomTool = ZelyrTool
