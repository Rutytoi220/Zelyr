"""Regex-based task router for Zelyr agents.

This is a simple keyword-matching router, not an AI agent.
For the real LLM-backed orchestrator, see orchestrator_agent.py.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from zelyr.agents.base import BaseAgent, AgentResult

logger = logging.getLogger(__name__)


class RegexRouter(BaseAgent):
    """
    Agent that decomposes tasks and delegates to specialist agents using regex.
    
    The router:
    1. Receives a task string
    2. Decomposes into subtasks using keyword matching
    3. Delegates to registered specialist agents
    4. Collects and synthesizes results
    5. Emits events for task lifecycle
    """
    
    # Routing patterns: (regex_pattern, agent_name, capture_group)
    ROUTING_PATTERNS = [
        (r"(?:run\s+command|execute|shell|cmd)[\s:]+(.+)", "shell_agent", 1),
        (r"(?:read\s+file|get\s+file|open)[\s:]+(.+)", "file_agent", 1),
        (r"(?:write\s+file|create\s+file|save)[\s:]+(.+)", "file_agent", 1),
        (r"(?:list|ls|dir)[\s:]+(.+)", "file_agent", 1),
    ]
    
    def __init__(self, name: str = "orchestrator", description: str | None = None, 
                 event_bus=None, tool_registry=None, agent_registry=None):
        """
        Initialize orchestrator agent.
        
        Args:
            name: Agent name
            description: Agent description
            event_bus: EventBus for emitting events
            tool_registry: Registry for tools (passed to delegated agents)
            agent_registry: Registry of specialist agents to delegate to
        """
        if description is None:
            description = "Task orchestrator that decomposes and delegates work"
        
        super().__init__(name, description, event_bus, tool_registry)
        self._agent_registry = agent_registry
    
    @property
    def agent_registry(self):
        """Return agent registry."""
        return self._agent_registry
    
    async def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Execute a task by decomposing and delegating to specialists.
        
        Args:
            task: Task description
            context: Execution context
        
        Returns:
            AgentResult with synthesized output
        """
        context = context or {}
        steps: list[str] = []
        results: Dict[str, Optional[AgentResult]] = {}
        memory_keys: list[str] = []
        
        try:
            # Emit task started event
            self._add_step(steps, "Task received by orchestrator")
            await self._emit_event("agent.task.started", {
                "agent": self.name,
                "task": task,
                "context": context
            })
            
            # Decompose task into subtasks
            self._add_step(steps, "Decomposing task into subtasks")
            subtasks = self._decompose_task(task)
            logger.info(f"Decomposed into {len(subtasks)} subtasks")
            
            if not subtasks:
                return AgentResult(
                    success=False,
                    error=f"Could not decompose task: {task}",
                    steps_taken=steps
                )
            
            # Delegate each subtask
            for subtask in subtasks:
                result = await self._delegate_subtask(subtask, context, steps)
                
                agent_name = subtask.get("agent", "unknown")
                subtask_desc = subtask.get("description", "unnamed")
                results[subtask_desc] = result
                
                if result and result.memory_keys_used:
                    memory_keys.extend(result.memory_keys_used)
            
            # Synthesize results
            self._add_step(steps, "Synthesizing results from all subtasks")
            output = self._synthesize_results(results)
            
            # Emit task completed event
            await self._emit_event("agent.task.completed", {
                "agent": self.name,
                "task": task,
                "success": True,
                "subtasks": len(subtasks),
                "results_count": len(results)
            })
            
            return AgentResult(
                success=True,
                output=output,
                steps_taken=steps,
                memory_keys_used=memory_keys,
                metadata={
                    "subtasks_executed": len(subtasks),
                    "results_collected": len(results)
                }
            )
        
        except Exception as e:
            logger.error(f"Orchestrator error: {str(e)}")
            self._add_step(steps, f"Error: {str(e)}")
            
            await self._emit_event("agent.task.failed", {
                "agent": self.name,
                "task": task,
                "error": str(e)
            })
            
            return AgentResult(
                success=False,
                error=str(e),
                steps_taken=steps
            )
    
    def _decompose_task(self, task: str) -> List[Dict[str, Any]]:
        """
        Decompose a task into subtasks using pattern matching.
        
        Args:
            task: Task description
        
        Returns:
            List of subtasks with agent and extracted parameters
        """
        subtasks = []
        
        # Try each routing pattern
        for pattern, agent_name, group_idx in self.ROUTING_PATTERNS:
            match = re.search(pattern, task, re.IGNORECASE)
            if match:
                # Extract the command/parameter from the pattern
                extracted = match.group(group_idx) if match.lastindex and group_idx <= match.lastindex else task
                
                subtasks.append({
                    "agent": agent_name,
                    "description": task,
                    "parameter": extracted.strip()
                })
                break
        
        # If no pattern matched, create a generic subtask
        if not subtasks:
            subtasks.append({
                "agent": "default",
                "description": task,
                "parameter": task
            })
        
        return subtasks
    
    async def _delegate_subtask(self, subtask: Dict[str, Any], context: Dict[str, Any],
                                steps: List[str]) -> Optional[AgentResult]:
        """
        Delegate a subtask to a specialist agent.
        
        Args:
            subtask: Subtask dictionary with agent and parameters
            context: Execution context
            steps: List to track steps
        
        Returns:
            AgentResult from delegated agent or None if agent not found
        """
        agent_name = subtask.get("agent")
        description = subtask.get("description")
        parameter = subtask.get("parameter", description)
        
        self._add_step(steps, f"Delegating to {agent_name}: {description}")
        
        # Look up agent in registry
        if not self._agent_registry:
            logger.warning("No agent registry configured")
            return None
        
        try:
            # Try both get_agent() and get() methods for compatibility
            if hasattr(self._agent_registry, 'get_agent'):
                agent = self._agent_registry.get_agent(agent_name)
            else:
                agent = self._agent_registry.get(agent_name)
            
            if not agent:
                logger.warning(f"Agent not found: {agent_name}")
                return None
            
            # Execute delegated agent with the parameter
            result = await agent.run(parameter, context)
            
            if result and result.success:
                self._add_step(steps, f"✓ {agent_name} completed successfully")
            else:
                self._add_step(steps, f"✗ {agent_name} failed")
            
            return result
        
        except Exception as e:
            logger.error(f"Error delegating to {agent_name}: {str(e)}")
            self._add_step(steps, f"Error in {agent_name}: {str(e)}")
            return None
    
    def _synthesize_results(self, results: Dict[str, Optional[AgentResult]]) -> Dict[str, Any]:
        """
        Synthesize results from multiple agents.
        
        Args:
            results: Dictionary mapping task descriptions to AgentResults
        
        Returns:
            Synthesized output
        """
        synthesis: Dict[str, Any] = {
            "subtask_results": {},
            "all_successful": True,
            "total_subtasks": len(results),
            "successful_count": 0,
            "failed_count": 0
        }
        
        for task_key, result in results.items():
            if result:
                success = result.success
                synthesis["subtask_results"][task_key] = {
                    "success": success,
                    "output": result.output,
                    "error": result.error,
                    "steps": len(result.steps_taken)
                }
                
                if success:
                    synthesis["successful_count"] += 1
                else:
                    synthesis["failed_count"] += 1
                    synthesis["all_successful"] = False
            else:
                synthesis["subtask_results"][task_key] = {
                    "success": False,
                    "output": None,
                    "error": "No result returned",
                    "steps": 0
                }
                synthesis["failed_count"] += 1
                synthesis["all_successful"] = False
        
        return synthesis
