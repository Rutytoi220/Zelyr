"""Echo agent implementation."""

from zelyr.agents.base import AgentResult
from zelyr.agents.simple_base import SimpleBaseAgent
from typing import Dict, Any


class EchoAgent(SimpleBaseAgent):
    """Agent that echoes back the task string."""

    def run(self, task: str) -> AgentResult:
        """Execute task by echoing it back.

        Args:
            task: Task to echo

        Returns:
            AgentResult with task as output
        """
        self._execution_count += 1
        self._emit("agent.started", {"agent": self.name, "task": task})
        steps: list[str] = []
        self._log(f"Received task: {task}", steps)

        try:
            from zelyr.tools import EchoTool

            tool = EchoTool()
            result = tool.execute({"text": task})

            self._log(f"EchoTool returned: {result.output}", steps)
            self._emit(
                "agent.completed",
                {"agent": self.name, "success": result.success}
            )

            return AgentResult(
                success=result.success,
                output=result.output,
                error=result.error,
                steps_taken=steps
            )
        except Exception as e:
            self._log(f"Error: {str(e)}", steps)
            self._emit(
                "agent.error",
                {"agent": self.name, "error": str(e)}
            )
            return AgentResult(
                success=False,
                output=None,
                error=str(e),
                steps_taken=steps
            )
