"""Multi-Agent Consensus Engine for high-risk operations."""
import json
import logging
import asyncio
from typing import List, Dict, Any
from zelyr.swarm.blackboard import Blackboard, Proposal, Critique
from zelyr.llm.ollama_client import OllamaClient, OllamaConfig
logger = logging.getLogger(__name__)

class ConsensusEngine:
    """Coordinates a debate among local LLMs to approve or reject high-risk tool execution."""

    def __init__(self, event_bus):
        """Auto-generated docstring.

Args:
    event_bus: Argument.

Returns:
    Return value.
"""
        self.bus = event_bus
        self.blackboard = Blackboard()
        self.proposer_llm = OllamaClient(OllamaConfig(model='qwen3-vl:2b'))
        self.critic_llm = OllamaClient(OllamaConfig(model='llama3.1:latest'))

    async def run_debate(self, task: str, context: str, pending_tools: List[Dict[str, Any]]) -> bool:
        """Run the debate protocol. Returns True if consensus reached, False otherwise."""
        if self.bus and hasattr(self.bus, 'publish_sync'):
            self.bus.publish_sync('swarm.debate.started', {'task': task, 'tool_count': len(pending_tools)})
        await self.blackboard.clear()
        max_revisions = 1
        current_revision = 0
        consensus_reached = False
        tools_json = json.dumps(pending_tools, indent=2, default=str)
        proposal = await self._generate_proposal(task, context, tools_json)
        await self.blackboard.post_proposal(proposal)
        if self.bus and hasattr(self.bus, 'publish_sync'):
            self.bus.publish_sync('swarm.proposal.submitted', {'author': proposal.author, 'proposal_id': proposal.id})
        while current_revision <= max_revisions and (not consensus_reached):
            critique = await self._generate_critique(task, proposal, tools_json)
            await self.blackboard.post_critique(critique)
            if critique.consensus_reached:
                consensus_reached = True
                break
            current_revision += 1
            if current_revision <= max_revisions:
                proposal = await self._generate_revision(task, proposal, critique, tools_json)
                await self.blackboard.post_proposal(proposal)
                if self.bus and hasattr(self.bus, 'publish_sync'):
                    self.bus.publish_sync('swarm.proposal.submitted', {'author': proposal.author, 'proposal_id': proposal.id, 'revision': current_revision})
        if self.bus and hasattr(self.bus, 'publish_sync'):
            self.bus.publish_sync('swarm.consensus.reached' if consensus_reached else 'swarm.consensus.failed', {'consensus': consensus_reached, 'revisions': current_revision})
        return consensus_reached

    async def _generate_proposal(self, task: str, context: str, tools_json: str) -> Proposal:
        """Ask the Proposer model to explain and justify the tool calls."""
        prompt = f'Task: {task}\nContext: {context}\nProposed Tool Calls:\n{tools_json}\n\nAs the Proposer, write a clear Implementation Plan that explains what these tools will do and why they are safe and correct. Be concise.'
        response = self.proposer_llm.chat([{'role': 'user', 'content': prompt}])
        return Proposal(author='qwen3-vl:2b', content=response, tool_calls=json.loads(tools_json))

    async def _generate_critique(self, task: str, proposal: Proposal, tools_json: str) -> Critique:
        """Ask the Critic model to scrutinize the proposal and tools."""
        prompt = f"Task: {task}\nImplementation Plan: {proposal.content}\nTool Calls:\n{tools_json}\n\nAs the Critic, review this plan for edge cases, bugs, or security flaws. If it is completely safe and correct, output exactly 'CONSENSUS_REACHED'. Otherwise, list the issues."
        response = self.critic_llm.chat([{'role': 'user', 'content': prompt}])
        is_consensus = 'CONSENSUS_REACHED' in response.upper()
        flags = []
        if not is_consensus:
            flags.append('Critic raised concerns.')
        return Critique(target_proposal_id=proposal.id, author='llama3.1:latest', content=response, flags=flags, consensus_reached=is_consensus)

    async def _generate_revision(self, task: str, old_proposal: Proposal, critique: Critique, tools_json: str) -> Proposal:
        """Ask the Proposer to revise based on critique."""
        prompt = f"Task: {task}\nPrevious Plan: {old_proposal.content}\nCritic's Feedback: {critique.content}\nTool Calls:\n{tools_json}\n\nAs the Proposer, revise your Implementation Plan to address the Critic's feedback."
        response = self.proposer_llm.chat([{'role': 'user', 'content': prompt}])
        return Proposal(author='qwen3-vl:2b', content=response, tool_calls=json.loads(tools_json))
