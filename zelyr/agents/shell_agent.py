"""Shell agent for executing shell commands."""

import logging
import re
from typing import Any, Dict, Optional
from zelyr.agents.base import BaseAgent, AgentResult
from zelyr.tools import ShellTool

logger = logging.getLogger(__name__)


class ShellAgent(BaseAgent):
    """
    Specialist agent for executing shell commands.
    
    Handles tasks matching patterns like:
    - "run command: <cmd>"
    - "execute <cmd>"
    - "shell <cmd>"
    """
    
    def __init__(self, name: str = "shell_agent", description: str | None = None,
                 event_bus=None, tool_registry=None, shell_tool: Optional[ShellTool] = None):
        """
        Initialize shell agent.
        
        Args:
            name: Agent name
            description: Agent description
            event_bus: EventBus for emitting events
            tool_registry: Registry for tools
            shell_tool: ShellTool instance (created if not provided)
        """
        if description is None:
            description = "Specialist agent for executing shell commands"
        
        super().__init__(name, description, event_bus, tool_registry)
        self._shell_tool = shell_tool or ShellTool()
    
    @property
    def shell_tool(self) -> ShellTool:
        """Return shell tool instance."""
        return self._shell_tool
    
    async def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Execute a shell command task.
        
        Args:
            task: Task description (should contain a command)
            context: Execution context
        
        Returns:
            AgentResult with command output
        """
        context = context or {}
        steps: list[str] = []
        
        try:
            # Parse command from task
            self._add_step(steps, f"Received task: {task}")
            command = self._extract_command(task)
            
            if not command:
                return AgentResult(
                    success=False,
                    error=f"Could not extract command from task: {task}",
                    steps_taken=steps
                )
            
            self._add_step(steps, f"Extracted command: {command}")
            
            # Emit task started event
            await self._emit_event("agent.shell.started", {
                "agent": self.name,
                "command": command
            })
            
            # Parse optional parameters from context
            timeout = context.get("timeout", 30)
            cwd = context.get("cwd")
            
            # Build execution parameters
            params = {"command": command}
            if timeout:
                params["timeout"] = timeout
            if cwd:
                params["cwd"] = cwd
            
            self._add_step(steps, f"Executing command with timeout={timeout}s")
            
            # Execute command using ShellTool
            result = await self._shell_tool.execute(params)
            
            self._add_step(steps, f"Command completed with status: {'success' if result.success else 'failed'}")
            
            # Emit completion event
            await self._emit_event("agent.shell.completed", {
                "agent": self.name,
                "command": command,
                "success": result.success,
                "returncode": result.metadata.get("returncode")
            })
            
            return AgentResult(
                success=result.success,
                output=result.output,
                steps_taken=steps,
                error=result.error,
                metadata={
                    "command": command,
                    "timeout": timeout,
                    "returncode": result.metadata.get("returncode")
                }
            )
        
        except Exception as e:
            logger.error(f"ShellAgent error: {str(e)}")
            self._add_step(steps, f"Error: {str(e)}")
            
            await self._emit_event("agent.shell.failed", {
                "agent": self.name,
                "task": task,
                "error": str(e)
            })
            
            return AgentResult(
                success=False,
                error=str(e),
                steps_taken=steps
            )
    
    def _extract_command(self, task: str) -> Optional[str]:
        """
        Extract shell command from task string.
        
        Handles patterns like:
        - "run command: echo hello"
        - "execute echo hello"
        - "shell echo hello"
        - "cmd echo hello"
        
        Args:
            task: Task description
        
        Returns:
            Extracted command or None
        """
        patterns = [
            r"(?:run\s+command|execute|shell|cmd)[\s:]+(.+)",
            r"^(.+)$"  # Fallback: entire string
        ]
        
        for pattern in patterns:
            match = re.search(pattern, task, re.IGNORECASE)
            if match:
                command = match.group(1).strip()
                if command:
                    return command
        
        return None
