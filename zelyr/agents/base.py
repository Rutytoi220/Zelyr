"""Abstract base class for Zelyr agents."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result of agent execution."""
    
    success: bool
    output: Any = None
    steps_taken: List[str] = field(default_factory=list)
    memory_keys_used: List[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"AgentResult(success={self.success}, "
            f"steps={len(self.steps_taken)}, "
            f"output_type={type(self.output).__name__})"
        )


class BaseAgent(ABC):
    """
    Abstract base class for all Zelyr agents.
    
    Agents are task-executing entities that can decompose and delegate work.
    """
    
    def __init__(self, name: str, description: str, event_bus=None, tool_registry=None):
        """
        Initialize agent.
        
        Args:
            name: Agent name
            description: Agent description
            event_bus: EventBus instance for emitting events
            tool_registry: Registry instance for accessing tools
        """
        self._name = name
        self._description = description
        self._event_bus = event_bus
        self._tool_registry = tool_registry
    
    @property
    def agent_id(self) -> str:
        """Return agent identifier."""
        return self._name
    
    @property
    def name(self) -> str:
        """Return agent name."""
        return self._name
    
    @property
    def description(self) -> str:
        """Return agent description."""
        return self._description
    
    @property
    def event_bus(self):
        """Return event bus."""
        return self._event_bus
    
    @property
    def tool_registry(self):
        """Return tool registry."""
        return self._tool_registry
    
    @abstractmethod
    async def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Execute a task.
        
        Args:
            task: Task description or command
            context: Execution context (optional)
        
        Returns:
            AgentResult with output and metadata
        """
        pass
    
    async def _emit_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        """
        Emit an event via event bus.

        Uses the synchronous publish_sync path so this works with the
        canonical :class:`zelyr.core.events.EventBus`.
        """
        if self._event_bus:
            try:
                self._event_bus.publish_sync(event_name, payload)
            except Exception as e:
                logger.warning(f"Failed to emit event {event_name}: {str(e)}")
    
    def _add_step(self, steps: List[str], step_description: str) -> None:
        """
        Add a step to the steps list.
        
        Args:
            steps: List to append to
            step_description: Description of step
        """
        steps.append(step_description)
        logger.debug(f"[{self.name}] Step: {step_description}")
    
    def set_engine_refs(self, event_bus, tool_registry=None) -> None:
        """
        Set references to engine components.
        
        Args:
            event_bus: EventBus instance for emitting events
            tool_registry: Registry instance for accessing tools (optional)
        """
        self._event_bus = event_bus
        if tool_registry is not None:
            self._tool_registry = tool_registry
        logger.debug(f"Set engine refs for agent '{self.name}'")
