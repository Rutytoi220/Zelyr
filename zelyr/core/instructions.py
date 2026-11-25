"""Zelyr Hierarchical Instruction Engine.

Manages global user custom instructions (~/.config/zelyr/instructions.md).
Legacy AXIOM instruction files (~/.config/axiom/instructions.md, ./AXIOM.md,
./.axiomrules) remain readable and are never moved.
Compiles hierarchical instructions into a unified system prompt format.
"""

from __future__ import annotations
from zelyr.compat import config_dir, config_file

import os
from pathlib import Path
from typing import Any, Optional

DEFAULT_GLOBAL_INSTRUCTIONS_TEMPLATE = """\
# Zelyr Global Custom Instructions
# Define your persistent persona, coding preferences, and global workflow directives here.
# These rules apply across all projects and sessions.
"""

from zelyr.core.system_prompt import HARDENED_SYSTEM_DIRECTIVES

MANIFEST_AUTHORITY_DIRECTIVES = """\
[RUNTIME AUTHORITY & CAPABILITY BOUNDARIES]
- Roles: You propose actions and interpret observations. The Zelyr runtime deterministically controls tool availability, permission checks, validation, execution, and verification.
- Manifest Truthfulness: Do not invent tools or claim filesystem, desktop, browser, shell, or network access that the manifest and active schemas do not substantiate.
- Locality Separation: Inference locality (local or remote endpoint) is separate from tool execution on the target host.
- Honest Outcomes: Never represent an unavailable or unauthorized tool call as successfully executed; rely strictly on returned observations and structured errors rather than inventing results."""

SYSTEM_CORE_DIRECTIVES = f"""\
[SYSTEM CORE DIRECTIVES]
You are Zelyr, a local-first AI orchestrator with a strict Three-Tier Automation Hierarchy.

{HARDENED_SYSTEM_DIRECTIVES}
"""


class InstructionManager:
    """Manages hierarchical instruction discovery and compilation."""

    def __init__(
        self,
        global_path: Optional[Path] = None,
        cwd: Optional[Path] = None,
    ) -> None:
        self.global_path = global_path or config_file("instructions.md")
        self.cwd = cwd or Path.cwd()

    def get_global_instructions(self) -> str:
        """Check for and read ~/.config/zelyr/instructions.md. If missing, create an empty template."""
        if not self.global_path.exists():
            try:
                self.global_path.parent.mkdir(parents=True, exist_ok=True)
                self.global_path.write_text(DEFAULT_GLOBAL_INSTRUCTIONS_TEMPLATE, encoding="utf-8")
            except Exception:
                return ""
        try:
            return self.global_path.read_text(encoding="utf-8").strip()
        except Exception:
            return ""

    def get_project_instructions_path(self) -> Optional[Path]:
        """Check workspace rules, preferring Zelyr names and reading legacy AXIOM files."""
        for candidate in ["ZELYR.md", ".zelyrrules", "AXIOM.md", ".axiomrules"]:
            p = self.cwd / candidate
            if p.is_file():
                return p
        return None

    def get_project_instructions(self) -> str:
        """Return workspace rules from Zelyr or legacy AXIOM instruction files."""
        p = self.get_project_instructions_path()
        if p and p.is_file():
            try:
                return p.read_text(encoding="utf-8").strip()
            except Exception:
                return ""
        return ""

    def get_compiled_instructions(
        self,
        base_directives: Optional[str] = None,
        config: Optional[Any] = None,
        tool_registry: Optional[Any] = None,
        caller_ring: Optional[int] = None,
        allowed_tools: Optional[Any] = None,
    ) -> str:
        """Compiles runtime manifest, system directives, global user instructions, and local project rules."""
        from zelyr.engine.environment import RuntimeEnvironmentManifest

        manifest = RuntimeEnvironmentManifest.build_manifest_block(
            config=config,
            tool_registry=tool_registry,
            caller_ring=caller_ring,
            allowed_tools=allowed_tools,
        )
        core = (base_directives or SYSTEM_CORE_DIRECTIVES).strip()
        global_rules = self.get_global_instructions()
        project_rules = self.get_project_instructions()

        parts = [manifest, MANIFEST_AUTHORITY_DIRECTIVES, core]
        parts.append(f"[USER CUSTOM INSTRUCTIONS]\n{global_rules if global_rules else '(No global instructions configured)'}")
        parts.append(f"[PROJECT WORKSPACE RULES]\n{project_rules if project_rules else '(No project workspace rules found)'}")

        return "\n\n".join(parts)
