"""Execution context - shared state during task execution."""

from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import uuid


@dataclass
class ExecutionContext:
    """Shared context for task execution across agents and tools."""
    
    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_input: str = ""
    conversation_history: list = field(default_factory=list)
    tool_results: Dict[str, Any] = field(default_factory=dict)
    agent_outputs: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_tool_result(self, tool_name: str, result: Any) -> None:
        """Add result from tool execution."""
        self.tool_results[tool_name] = result
    
    def add_agent_output(self, agent_name: str, output: Any) -> None:
        """Add output from agent execution."""
        self.agent_outputs[agent_name] = output
    
    def set_variable(self, key: str, value: Any) -> None:
        """Set a context variable."""
        self.variables[key] = value
    
    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a context variable."""
        return self.variables.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "context_id": self.context_id,
            "user_input": self.user_input,
            "conversation_history": self.conversation_history,
            "tool_results": self.tool_results,
            "agent_outputs": self.agent_outputs,
            "variables": self.variables,
            "metadata": self.metadata,
        }
    
    def clear_results(self) -> None:
        """Clear tool and agent results."""
        self.tool_results.clear()
        self.agent_outputs.clear()
