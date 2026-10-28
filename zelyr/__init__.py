"""Zelyr - AI Orchestration Framework.

A local-first AI system with modular architecture, event-driven core,
and support for agents, tools, and plugins.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from zelyr.compat import sync_legacy_env
sync_legacy_env()

__version__ = "1.1.0.dev0"
__author__ = "Zelyr Team"
__description__ = "Local-first AI orchestration framework"

# PEP 562 Lazy export registry mapping exported symbols to (module_path, target_attr).
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # Core
    "Engine": ("zelyr.core", "Engine"),
    "EventBus": ("zelyr.core", "EventBus"),
    "Event": ("zelyr.core", "Event"),
    "Registry": ("zelyr.core", "Registry"),
    "ExecutionContext": ("zelyr.core", "ExecutionContext"),
    # LLM
    "OllamaClient": ("zelyr.llm", "OllamaClient"),
    "OllamaConfig": ("zelyr.llm", "OllamaConfig"),
    # Memory
    "SyncMemoryStore": ("zelyr.memory", "SyncMemoryStore"),
    # Planning
    "ExecutionPlan": ("zelyr.planning", "ExecutionPlan"),
    "PlanError": ("zelyr.planning", "PlanError"),
    "PlanStep": ("zelyr.planning", "PlanStep"),
    "StepStatus": ("zelyr.planning", "StepStatus"),
    "TaskPlanner": ("zelyr.planning", "TaskPlanner"),
    # Agents
    "BaseAgent": ("zelyr.agents", "BaseAgent"),
    "AgentResult": ("zelyr.agents", "AgentResult"),
    "OrchestratorAgent": ("zelyr.agents", "OrchestratorAgent"),
    "ShellAgent": ("zelyr.agents", "ShellAgent"),
    # Tools
    "BaseTool": ("zelyr.tools", "BaseTool"),
    "ToolResult": ("zelyr.tools", "ToolResult"),
    "EchoTool": ("zelyr.tools", "EchoTool"),
    "ShellTool": ("zelyr.tools", "ShellTool"),
    "FileReadTool": ("zelyr.tools", "FileReadTool"),
    "FileWriteTool": ("zelyr.tools", "FileWriteTool"),
    "SystemInfoTool": ("zelyr.tools", "SystemInfoTool"),
    "ToolParameter": ("zelyr.tools", "ToolParameter"),
    "FileTool": ("zelyr.tools", "FileTool"),
    "ShellCommandTool": ("zelyr.tools", "ShellCommandTool"),
    "ReadFileTool": ("zelyr.tools", "ReadFileTool"),
    "WriteFileTool": ("zelyr.tools", "WriteFileTool"),
    "PythonExecTool": ("zelyr.tools", "PythonExecTool"),
    # Tool Registry
    "ToolRegistry": ("zelyr.tool_registry", "ToolRegistry"),
    "ToolRegistryError": ("zelyr.tool_registry", "ToolRegistryError"),
    # Plugins
    "BasePlugin": ("zelyr.plugins", "BasePlugin"),
    "NXBTPlugin": ("zelyr.plugins", "NXBTPlugin"),
    "AutomationPlugin": ("zelyr.plugins", "AutomationPlugin"),
    # Config
    "ZelyrConfig": ("zelyr.config", "ZelyrConfig"),
    "AxiomConfig": ("zelyr.config", "ZelyrConfig"),
    "get_config": ("zelyr.config", "get_config"),
    "set_config": ("zelyr.config", "set_config"),
}

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


def __getattr__(name: str) -> Any:
    """Lazily import and cache public symbols on first access (PEP 562)."""
    if name in _LAZY_EXPORTS:
        module_path, attr_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_path)
        val = getattr(module, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module 'zelyr' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Ensure lazily exported symbols remain discoverable via dir() and IDE introspection."""
    return sorted(set(list(globals().keys()) + list(_LAZY_EXPORTS.keys())))


if TYPE_CHECKING:
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
        PythonExecTool,
    )
    from zelyr.tool_registry import ToolRegistry, ToolRegistryError
    from zelyr.plugins import BasePlugin, NXBTPlugin, AutomationPlugin
    from zelyr.config import ZelyrConfig, get_config, set_config
    AxiomConfig = ZelyrConfig
