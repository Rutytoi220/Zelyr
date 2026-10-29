"""Zelyr Plugins Module — Extensible plugin system (v1 + v2 API)."""

# v1 legacy plugins (kept for backward compatibility)
from zelyr.plugins.base_plugin import BasePlugin
from zelyr.plugins.nxbt_plugin import NXBTPlugin
from zelyr.plugins.automation_plugin import AutomationPlugin

# v2 Plugin Ecosystem (RFC-001)
from zelyr.plugins.zelyr_plugin import ZelyrPlugin, HookResult, PluginToolDefinition

# Legacy name retained for existing AXIOM integrations.
AxiomPlugin = ZelyrPlugin
from zelyr.plugins.exceptions import (
    PluginError,
    PluginManifestError,
    PluginPermissionError,
    PluginVersionError,
    SandboxSecurityViolation,
)
from zelyr.plugins.loader import PluginLoader, PluginManifest, PluginPermissions
from zelyr.plugins.sandbox import SandboxedToolProxy, static_check_plugin_source
from zelyr.plugins.manager import PluginManager
from zelyr.core.plugin_sandbox import SandboxedPluginRunner

__all__ = [
    # v1
    "BasePlugin",
    "NXBTPlugin",
    "AutomationPlugin",
    # v2 core
    "ZelyrPlugin",
    "AxiomPlugin",
    "HookResult",
    "PluginToolDefinition",
    # exceptions
    "PluginError",
    "PluginManifestError",
    "PluginPermissionError",
    "PluginVersionError",
    "SandboxSecurityViolation",
    # loader
    "PluginLoader",
    "PluginManifest",
    "PluginPermissions",
    # sandbox
    "SandboxedToolProxy",
    "SandboxedPluginRunner",
    "static_check_plugin_source",
    # manager
    "PluginManager",
]
