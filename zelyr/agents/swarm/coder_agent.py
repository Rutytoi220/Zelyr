"""CoderAgent for the Swarm. Specialized in file editing and syntax."""
import logging
from typing import Any, Dict, Optional
from zelyr.agents.swarm.base_swarm import BaseSubagent
from zelyr.agents.base import AgentResult
logger = logging.getLogger(__name__)

class CoderAgent(BaseSubagent):
    """Swarm worker specialized in writing code and refactoring."""

    def __init__(self, event_bus=None, tool_registry=None, llm_client=None, blackboard=None, session_id: Optional[str]=None):
        super().__init__(
            name='CoderAgent',
            description='Specialized agent for writing and editing code.',
            topic='swarm.task.code',
            event_bus=event_bus,
            tool_registry=tool_registry,
            llm_client=llm_client,
            blackboard=blackboard,
            session_id=session_id,
        )

    async def write_code(self, filepath: str, code: str) -> Dict[str, Any]:
        """A helper that attempts to use the write_file tool securely."""
        logger.info(f'[{self.name}] Attempting to write code to {filepath}')
        result = await self._execute_tool_safely(tool_name='write_file', arguments={'path': filepath, 'content': code}, timeout=5.0)
        return {'result': result}

    async def run(self, task: str, context: Optional[Dict[str, Any]]=None) -> AgentResult:
        """Execute a coding task."""
        return AgentResult(success=True, output=f'Coder processed: {task}')
