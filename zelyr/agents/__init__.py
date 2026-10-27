"""Zelyr agents module for task execution and delegation."""

from zelyr.agents.base import BaseAgent, AgentResult
from zelyr.agents.simple_base import SimpleBaseAgent
from zelyr.agents.echo_agent import EchoAgent
from zelyr.agents.regex_router import RegexRouter
from zelyr.agents.orchestrator_agent import OrchestratorAgent
from zelyr.agents.shell_agent import ShellAgent

__all__ = [
    "BaseAgent",
    "AgentResult",
    "SimpleBaseAgent",
    "EchoAgent",
    "RegexRouter",
    "OrchestratorAgent",
    "ShellAgent",
]
