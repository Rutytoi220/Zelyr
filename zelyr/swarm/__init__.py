"""Swarm Intelligence & Multi-Agent Coordination Module."""

from .blackboard import Blackboard, Proposal, Critique, Vote
from .consensus import ConsensusEngine

__all__ = [
    "Blackboard",
    "Proposal",
    "Critique",
    "Vote",
    "ConsensusEngine",
]
