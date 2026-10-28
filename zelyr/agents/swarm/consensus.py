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

    def __init__(
        self,
        event_bus: EventBus,
        blackboard=None,
        memory_store=None,
        session_id: Optional[str] = None,
        required_quorum: float = 2 / 3,
        total_voters: int = 1,
    ):
        """Initialize ConsensusEngine.

        Args:
            event_bus: System event bus.
            blackboard: Ephemeral blackboard store.
            memory_store: Long-term memory store.
            session_id: Swarm session identifier.
            required_quorum: Ratio of positive votes required for approval (default 2/3).
            total_voters: Total count of voting agents in the swarm quorum (default 1).
        """
        self.event_bus = event_bus
        self._proposals: Dict[str, Dict[str, Any]] = {}
        self._blackboard = blackboard
        self._memory_store = memory_store
        self._session_id = session_id
        self.required_quorum = required_quorum
        self.total_voters = total_voters
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

        votes = proposal.setdefault('votes', {})
        if voter:
            votes[voter] = vote

        total = proposal.get('total_voters') or self.total_voters or 1
        quorum_ratio = proposal.get('required_quorum', self.required_quorum)

        if total <= 1:
            if vote == 'APPROVED':
                proposal['state'] = ProposalState.APPROVED
                logger.info(f'Consensus: Proposal {proposal_id} APPROVED by {voter}')
                self._commit_blackboard_artifact(proposal)
            elif vote == 'REJECTED':
                proposal['state'] = ProposalState.REJECTED
                logger.info(f'Consensus: Proposal {proposal_id} REJECTED by {voter}')
            return

        approved_count = sum(1 for v in votes.values() if v == 'APPROVED')
        rejected_count = sum(1 for v in votes.values() if v == 'REJECTED')

        if (approved_count / total) >= (quorum_ratio - 1e-6):
            proposal['state'] = ProposalState.APPROVED
            logger.info(
                f'Consensus: Proposal {proposal_id} reached quorum '
                f'({approved_count}/{total} APPROVED by {list(votes.keys())})'
            )
            self._commit_blackboard_artifact(proposal)
        elif ((total - rejected_count) / total) < (quorum_ratio - 1e-6) or (
            len(votes) >= total and (approved_count / total) < (quorum_ratio - 1e-6)
        ):
            proposal['state'] = ProposalState.REJECTED
            logger.info(
                f'Consensus: Proposal {proposal_id} failed quorum '
                f'({rejected_count}/{total} REJECTED by {list(votes.keys())})'
            )

    def propose(
        self,
        agent_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: float = 5.0,
        total_voters: Optional[int] = None,
        required_quorum: Optional[float] = None,
    ) -> bool:
        """Submit a proposal to the swarm and wait for consensus.
        
        Args:
            agent_name: Name of the agent proposing the action.
            tool_name: Name of the destructive tool.
            arguments: Arguments to be passed to the tool.
            timeout: Max seconds to wait before auto-rejecting.
            total_voters: Optional override for total voters in quorum.
            required_quorum: Optional override for quorum fraction (e.g. 0.6667).
            
        Returns:
            True if APPROVED, False if REJECTED or timed out.
        """
        proposal_id = str(uuid.uuid4())
        voters_count = total_voters if total_voters is not None else self.total_voters
        quorum = required_quorum if required_quorum is not None else self.required_quorum
        self._proposals[proposal_id] = {
            'id': proposal_id,
            'agent': agent_name,
            'tool': tool_name,
            'arguments': arguments,
            'state': ProposalState.PROPOSED,
            'timestamp': time.time(),
            'votes': {},
            'total_voters': voters_count,
            'required_quorum': quorum,
        }
        self.event_bus.publish(
            Event(
                event_type='swarm.proposal',
                source='ConsensusEngine',
                data={
                    'proposal_id': proposal_id,
                    'agent': agent_name,
                    'tool': tool_name,
                    'arguments': arguments,
                },
            )
        )
        logger.info(
            f"Consensus: Agent '{agent_name}' proposed mutating tool '{tool_name}' (ID: {proposal_id}, voters: {voters_count})"
        )
        start_time = time.time()
        while time.time() - start_time < timeout:
            state = self._proposals[proposal_id]['state']
            if state == ProposalState.APPROVED:
                return True
            if state == ProposalState.REJECTED:
                return False
            time.sleep(0.05)
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
