"""Zelyr - AI Orchestration Framework.

A local-first AI system with modular architecture, event-driven core,
and support for agents, tools, and plugins.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from zelyr.compat import sync_legacy_env
sync_legacy_env()


def _resolve_version() -> str:
    """Resolve package version from pyproject.toml in source checkouts or dist-info metadata."""
    from pathlib import Path
    import re

    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if pyproject_path.is_file():
        try:
            text = pyproject_path.read_text(encoding="utf-8")
            match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
            if match:
                return match.group(1)
        except OSError:
            pass
    try:
        from importlib.metadata import version

        return version("zelyr-core")
    except Exception:
        return "1.4.1"


__version__ = _resolve_version()
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
    "CompiledPlanBundle": ("zelyr.planning", "CompiledPlanBundle"),
    "ExecutionPlan": ("zelyr.planning", "ExecutionPlan"),
    "PlanCompilationError": ("zelyr.planning", "PlanCompilationError"),
    "PlanCompiler": ("zelyr.planning", "PlanCompiler"),
    "PlanConfirmationRequiredError": ("zelyr.planning", "PlanConfirmationRequiredError"),
    "PlanError": ("zelyr.planning", "PlanError"),
    "PlanStep": ("zelyr.planning", "PlanStep"),
    "PlanValidationError": ("zelyr.planning", "PlanValidationError"),
    "PlannedTaskNode": ("zelyr.planning", "PlannedTaskNode"),
    "StepStatus": ("zelyr.planning", "StepStatus"),
    "StructuredPlan": ("zelyr.planning", "StructuredPlan"),
    "TaskPlanner": ("zelyr.planning", "TaskPlanner"),
    "UnsupportedPlanVersionError": ("zelyr.planning", "UnsupportedPlanVersionError"),
    "compile_structured_plan_to_task_graph": ("zelyr.planning", "compile_structured_plan_to_task_graph"),
    "parse_structured_plan": ("zelyr.planning", "parse_structured_plan"),
    "replan_task_graph": ("zelyr.planning", "replan_task_graph"),
    # Recovery
    "CorruptedRecoveryStateError": ("zelyr.engine.recovery", "CorruptedRecoveryStateError"),
    "RecoveryAssessment": ("zelyr.engine.recovery", "RecoveryAssessment"),
    "RecoveryDecision": ("zelyr.engine.recovery", "RecoveryDecision"),
    "RecoveryEngine": ("zelyr.engine.recovery", "RecoveryEngine"),
    "RecoveryError": ("zelyr.engine.recovery", "RecoveryError"),
    "RecoveryPolicy": ("zelyr.engine.recovery", "RecoveryPolicy"),
    "UnsupportedRecoveryVersionError": ("zelyr.engine.recovery", "UnsupportedRecoveryVersionError"),
    "build_graph_recovery_overview": ("zelyr.engine.recovery", "build_graph_recovery_overview"),
    # Execution Policy
    "ConfirmationRecord": ("zelyr.engine.execution_policy", "ConfirmationRecord"),
    "ConfirmationStore": ("zelyr.engine.execution_policy", "ConfirmationStore"),
    "ExecutionPolicyConfig": ("zelyr.engine.execution_policy", "ExecutionPolicyConfig"),
    "ExecutionPolicyDecision": ("zelyr.engine.execution_policy", "ExecutionPolicyDecision"),
    "ExecutionPolicyEngine": ("zelyr.engine.execution_policy", "ExecutionPolicyEngine"),
    "PolicyConfirmationError": ("zelyr.engine.execution_policy", "PolicyConfirmationError"),
    "PolicyDecisionOutcome": ("zelyr.engine.execution_policy", "PolicyDecisionOutcome"),
    "ConfigurationDiagnosticReport": ("zelyr.engine.config_diagnostics", "ConfigurationDiagnosticReport"),
    "ConfigurationDiagnostics": ("zelyr.engine.config_diagnostics", "ConfigurationDiagnostics"),
    "diagnose_configuration": ("zelyr.engine.config_diagnostics", "diagnose_configuration"),
    "ExecutionLifecycleDiagnostics": ("zelyr.engine.diagnostics", "ExecutionLifecycleDiagnostics"),
    "ExecutionLifecycleReport": ("zelyr.engine.diagnostics", "ExecutionLifecycleReport"),
    "ExecutionMetricsReport": ("zelyr.engine.observability", "ExecutionMetricsReport"),
    "collect_execution_metrics": ("zelyr.engine.observability", "collect_execution_metrics"),
    "collect_fleet_observability": ("zelyr.engine.observability", "collect_fleet_observability"),
    "diagnose_execution_lifecycle": ("zelyr.engine.diagnostics", "diagnose_execution_lifecycle"),
    "PolicyError": ("zelyr.engine.execution_policy", "PolicyError"),
    "PolicyExplanation": ("zelyr.engine.execution_policy", "PolicyExplanation"),
    "PolicyPreviewResult": ("zelyr.engine.execution_policy", "PolicyPreviewResult"),
    "RiskClassifier": ("zelyr.engine.execution_policy", "RiskClassifier"),
    "RiskLevel": ("zelyr.engine.execution_policy", "RiskLevel"),
    "UnsupportedPolicyVersionError": ("zelyr.engine.execution_policy", "UnsupportedPolicyVersionError"),
    "inspect_policy_configuration": ("zelyr.engine.execution_policy", "inspect_policy_configuration"),
    "preview_tool_policy": ("zelyr.engine.execution_policy", "preview_tool_policy"),
    "query_policy_history": ("zelyr.engine.execution_policy", "query_policy_history"),
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
    "CompiledPlanBundle",
    "ExecutionPlan",
    "PlanCompilationError",
    "PlanCompiler",
    "PlanConfirmationRequiredError",
    "PlanError",
    "PlanStep",
    "PlanValidationError",
    "PlannedTaskNode",
    "StepStatus",
    "StructuredPlan",
    "TaskPlanner",
    "UnsupportedPlanVersionError",
    "compile_structured_plan_to_task_graph",
    "parse_structured_plan",
    "replan_task_graph",
    # Recovery
    "CorruptedRecoveryStateError",
    "RecoveryAssessment",
    "RecoveryDecision",
    "RecoveryEngine",
    "RecoveryError",
    "RecoveryPolicy",
    "UnsupportedRecoveryVersionError",
    "build_graph_recovery_overview",
    # Execution Policy
    "ConfirmationRecord",
    "ConfirmationStore",
    "ExecutionPolicyConfig",
    "ExecutionPolicyDecision",
    "ExecutionPolicyEngine",
    "PolicyConfirmationError",
    "PolicyDecisionOutcome",
    "PolicyError",
    "RiskClassifier",
    "RiskLevel",
    "UnsupportedPolicyVersionError",
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
    from zelyr.planning import (
        CompiledPlanBundle,
        ExecutionPlan,
        PlanCompilationError,
        PlanCompiler,
        PlanConfirmationRequiredError,
        PlanError,
        PlanStep,
        PlanValidationError,
        PlannedTaskNode,
        StepStatus,
        StructuredPlan,
        TaskPlanner,
        UnsupportedPlanVersionError,
        compile_structured_plan_to_task_graph,
        parse_structured_plan,
        replan_task_graph,
    )
    from zelyr.engine.recovery import (
        CorruptedRecoveryStateError,
        RecoveryAssessment,
        RecoveryDecision,
        RecoveryEngine,
        RecoveryError,
        RecoveryPolicy,
        UnsupportedRecoveryVersionError,
        build_graph_recovery_overview,
    )
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
