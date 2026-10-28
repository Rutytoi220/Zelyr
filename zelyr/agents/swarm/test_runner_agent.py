"""TestRunnerAgent for the Swarm. Specialized in test execution and voting."""
import logging
from typing import Any, Dict, Optional
from zelyr.agents.swarm.base_swarm import BaseSubagent
from zelyr.agents.base import AgentResult
from zelyr.core.events import Event
logger = logging.getLogger(__name__)

class TestRunnerAgent(BaseSubagent):
    """Swarm worker specialized in testing code and voting on proposals."""
    __test__ = False

    def __init__(self, event_bus=None, tool_registry=None, llm_client=None, blackboard=None, session_id: Optional[str]=None):
        super().__init__(
            name='TestRunnerAgent',
            description='Specialized agent for running tests and verifying code.',
            topic='swarm.task.test',
            event_bus=event_bus,
            tool_registry=tool_registry,
            llm_client=llm_client,
            blackboard=blackboard,
            session_id=session_id,
        )
        if self.event_bus:
            self.event_bus.subscribe('swarm.proposal', self._review_proposal)

    def _review_proposal(self, event: Event) -> None:
        """Review proposals and cast a vote on the EventBus."""
        proposal_id = event.data.get('proposal_id')
        agent = event.data.get('agent')
        tool = event.data.get('tool')
        arguments = event.data.get('arguments', {})
        if agent == self.name:
            return
        logger.info(f"[{self.name}] Reviewing proposal {proposal_id} from {agent} for tool '{tool}'")
        vote_decision = 'REJECTED'
        if tool == 'write_file':
            path = arguments.get('path', '')
            if str(path).endswith('.py'):
                vote_decision = 'APPROVED'
                logger.info(f'[{self.name}] Approving python code write.')
            else:
                logger.warning(f'[{self.name}] Rejecting write to non-python file: {path}')
        elif tool == 'shell':
            cmd = arguments.get('command', '')
            if 'pytest' in cmd:
                vote_decision = 'APPROVED'
            else:
                logger.warning(f'[{self.name}] Rejecting unsafe shell command: {cmd}')
        self.event_bus.publish(Event(event_type='swarm.vote', source=self.name, data={'proposal_id': proposal_id, 'vote': vote_decision, 'voter': self.name}))

    async def run(self, task: str, context: Optional[Dict[str, Any]]=None) -> AgentResult:
        """Execute a testing task."""
        return AgentResult(success=True, output=f'TestRunner processed: {task}')
