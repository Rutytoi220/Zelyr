"""Zelyr - AI Orchestration Framework.

A local-first AI system with modular architecture, event-driven core,
and support for agents, tools, and plugins.
"""

from zelyr.compat import sync_legacy_env
sync_legacy_env()

from zelyr.core import Engine, EventBus, Event, Registry, ExecutionContext
from zelyr.llm import OllamaClient, OllamaConfig
from zelyr.memory import SyncMemoryStore
from zelyr.planning import ExecutionPlan, PlanError, PlanStep, StepStatus, TaskPlanner
from zelyr.agents import BaseAgent, AgentResult, OrchestratorAgent, ShellAgent
from zelyr.tools import (
    BaseTool,
    ToolResult,
    EchoTool,
    ShellTool,
    FileReadTool,
    FileWriteTool,
    SystemInfoTool,
    ToolParameter,
    FileTool,
    ShellCommandTool,
    ReadFileTool,
    WriteFileTool,
    PythonExecTool
)
from zelyr.tool_registry import ToolRegistry, ToolRegistryError
from zelyr.plugins import BasePlugin, NXBTPlugin, AutomationPlugin
from zelyr.config import ZelyrConfig, get_config, set_config

# Legacy public names retained for existing AXIOM integrations.
AxiomConfig = ZelyrConfig

__version__ = "1.1.0.dev0"
__author__ = "Zelyr Team"
__description__ = "Local-first AI orchestration framework"

__all__ = [
    # Core
    "Engine",
    "EventBus",
    "Event",
    "Registry",
    "ExecutionContext",
    # LLM
    "OllamaClient",
    "OllamaConfig",
    "SyncMemoryStore",
    # Planning
    "ExecutionPlan",
    "PlanError",
    "PlanStep",
    "StepStatus",
    "TaskPlanner",
    # Agents
    "BaseAgent",
    "AgentResult",
    "OrchestratorAgent",
    "ShellAgent",
    # Tools
    "BaseTool",
    "ToolResult",
    "EchoTool",
    "ShellTool",
    "FileReadTool",
    "FileWriteTool",
    "SystemInfoTool",
    "ToolParameter",
    "FileTool",
    "ShellCommandTool",
    "ReadFileTool",
    "WriteFileTool",
    "PythonExecTool",
    "ToolRegistry",
    "ToolRegistryError",
    # Plugins
    "BasePlugin",
    "NXBTPlugin",
    "AutomationPlugin",
    # Config
    "ZelyrConfig",
    "AxiomConfig",
    "get_config",
    "set_config",
    # Metadata
    "__version__",
    "__author__",
    "__description__",
]
