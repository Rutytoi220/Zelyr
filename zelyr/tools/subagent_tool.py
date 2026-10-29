"""Subagent delegation tool module for Zelyr."""
from __future__ import annotations

from zelyr.agents.swarm.delegation import (
    DelegatedSubagent,
    SubagentDelegationTool,
    SubagentProfile,
    SubagentRole,
    SUBAGENT_PROFILES,
    get_subagent_profile,
)

__all__ = [
    "SubagentDelegationTool",
    "DelegatedSubagent",
    "SubagentProfile",
    "SubagentRole",
    "SUBAGENT_PROFILES",
    "get_subagent_profile",
]
