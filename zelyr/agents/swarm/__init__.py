"""Multi-Agent Swarm Orchestration & Consensus Protocol."""

from zelyr.agents.swarm.consensus import ConsensusEngine, ProposalState
from zelyr.agents.swarm.base_swarm import BaseSubagent
from zelyr.agents.swarm.coder_agent import CoderAgent
from zelyr.agents.swarm.test_runner_agent import TestRunnerAgent
from zelyr.agents.swarm.delegation import (
    DelegatedSubagent,
    SubagentDelegationTool,
    SubagentProfile,
    SubagentRole,
    SUBAGENT_PROFILES,
    get_subagent_profile,
)

__all__ = [
    "ConsensusEngine",
    "ProposalState",
    "BaseSubagent",
    "CoderAgent",
    "TestRunnerAgent",
    "SubagentDelegationTool",
    "DelegatedSubagent",
    "SubagentProfile",
    "SubagentRole",
    "SUBAGENT_PROFILES",
    "get_subagent_profile",
]
