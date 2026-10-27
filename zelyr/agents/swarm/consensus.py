"""Consensus Protocol for Multi-Agent Swarm Execution.

Provides the `ConsensusEngine` to handle proposing, voting, and
resolving distributed tool execution consensus.
"""
import time
import uuid
import logging
from enum import Enum
from typing import Dict, Any, Optional
from zelyr.core.events import EventBus, Event
logger = logging.getLogger(__name__)

class ProposalState(Enum):
    """Auto-generated docstring.

"""
    PROPOSED = 'proposed'
    APPROVED = 'approved'
    REJECTED = 'rejected'

class ConsensusEngine:
    """Manages swarm proposals and voting logic for destructive actions."""

    def __init__(self, event_bus: EventBus, blackboard=None, memory_store=None, session_id: Optional[str]=None):
        """Auto-generated docstring.

Args:
    event_bus: Argument.
    blackboard: Argument.
    memory_store: Argument.
    session_id: Argument.

Returns:
    Return value.
"""
        self.event_bus = event_bus
        self._proposals: Dict[str, Dict[str, Any]] = {}
        self._blackboard = blackboard
        self._memory_store = memory_store
        self._session_id = session_id
        self.event_bus.subscribe('swarm.vote', self._handle_vote)

    def _handle_vote(self, event: Event) -> None:
        """Process an incoming vote."""
        proposal_id = event.data.get('proposal_id')
        vote = event.data.get('vote')
        voter = event.data.get('voter')
        if proposal_id not in self._proposals:
            return
        proposal = self._proposals[proposal_id]
        if proposal['state'] != ProposalState.PROPOSED:
            return
        if vote == 'APPROVED':
            proposal['state'] = ProposalState.APPROVED
            logger.info(f'Consensus: Proposal {proposal_id} APPROVED by {voter}')
            self._commit_blackboard_artifact(proposal)
        elif vote == 'REJECTED':
            proposal['state'] = ProposalState.REJECTED
            logger.info(f'Consensus: Proposal {proposal_id} REJECTED by {voter}')

    def propose(self, agent_name: str, tool_name: str, arguments: Dict[str, Any], timeout: float=5.0) -> bool:
        """Submit a proposal to the swarm and wait for consensus.
        
        Args:
            agent_name: Name of the agent proposing the action.
            tool_name: Name of the destructive tool.
            arguments: Arguments to be passed to the tool.
            timeout: Max seconds to wait before auto-rejecting.
            
        Returns:
            True if APPROVED, False if REJECTED or timed out.
        """
        proposal_id = str(uuid.uuid4())
        self._proposals[proposal_id] = {'id': proposal_id, 'agent': agent_name, 'tool': tool_name, 'arguments': arguments, 'state': ProposalState.PROPOSED, 'timestamp': time.time()}
        self.event_bus.publish(Event(event_type='swarm.proposal', source='ConsensusEngine', data={'proposal_id': proposal_id, 'agent': agent_name, 'tool': tool_name, 'arguments': arguments}))
        logger.info(f"Consensus: Agent '{agent_name}' proposed mutating tool '{tool_name}' (ID: {proposal_id})")
        start_time = time.time()
        while time.time() - start_time < timeout:
            state = self._proposals[proposal_id]['state']
            if state == ProposalState.APPROVED:
                return True
            if state == ProposalState.REJECTED:
                return False
            time.sleep(0.1)
        self._proposals[proposal_id]['state'] = ProposalState.REJECTED
        logger.warning(f'Consensus: Proposal {proposal_id} TIMED OUT. Automatically REJECTED to prevent deadlock.')
        return False

    def _commit_blackboard_artifact(self, proposal: Dict[str, Any]) -> None:
        """Commit the approved agent's final Blackboard artifact to persistent memory.

        Called automatically by ``_handle_vote`` when a proposal is APPROVED.
        Extracts all scratchpad keys written by the proposing agent during this
        session and stores them as canonical beliefs in the root SQLite memory
        store.

        If no Blackboard or memory_store is configured, this is a no-op.
        """
        if self._blackboard is None or self._memory_store is None or self._session_id is None:
            return
        agent_name = proposal.get('agent', 'unknown')
        tool_name = proposal.get('tool', 'unknown')
        artifact_keys = self._blackboard.list_keys(self._session_id, agent_name)
        if not artifact_keys:
            return
        for key in artifact_keys:
            value = self._blackboard.read(self._session_id, agent_name, key)
            if value is None:
                continue
            content = f'[Swarm Consensus Commit] Agent={agent_name}, Tool={tool_name}, Artifact={key}: {str(value)[:500]}'
            try:
                self._memory_store.add_message('system', content)
                logger.info("Consensus: Committed Blackboard artifact '%s' from agent '%s' to persistent memory.", key, agent_name)
            except Exception as exc:
                logger.warning("Consensus: Failed to commit artifact '%s' to memory: %s", key, exc)
