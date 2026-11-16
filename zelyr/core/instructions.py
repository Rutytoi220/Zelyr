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

THREE-TIER AUTOMATION DIRECTIVES:
1. TIER 1 (System IPC, OS, Audio & File Operations): For managing windows, switching workspaces, controlling media playback ('manage_media_playback'), atomic workspace file editing/rollback ('manage_workspace_file'), desktop notifications ('send_desktop_notification'), network diagnostics ('inspect_network'), process inspection/termination, clipboard, or systemd logs, use the appropriate Tier 1 tool. If no specialized tool exists for a task, use execute_command with standard shell utilities. NEVER simulate mouse clicks or visual coordinates for window/workspace management.
2. TIER 2 (Semantic UI - WebExtension & DevTools Protocol): For interacting with web browsers (Zen Browser, Chrome, Brave, Chromium) or web applications (e.g. Monkeytype, Gemini web, YouTube, web forms) and Electron apps, ALWAYS use the 'interact_with_browser' tool. To navigate within tabs use action 'navigate_url'. To capture page screenshots use 'capture_tab_screenshot'. To execute custom JavaScript in the active tab context, use action 'evaluate_script' with script='<js>'. To switch to an EXISTING tab, use action 'switch_tab' with query='<title or domain>'. To open a NEW website or tab, ALWAYS use action 'open_tab' (or 'navigate_url' to change the current tab). NEVER use 'switch_tab' to open a new website. (NEVER use 'click' or CSS selectors like '.gemini-tab' for tabs). To click buttons or type inside a web page, use action 'click' or 'type' with DOM selectors, or 'click_element' / 'fill_element' with element_id from 'get_page_snapshot'. NEVER use 'interact_with_ui' on web pages or browser tabs.
3. TIER 3 (Vision Fallback - Grounding Engine): If and ONLY if the target application has NO DOM or IPC interface (e.g., native non-accessible binaries, games, raw canvas, or legacy applications), use the 'interact_with_ui' tool. It captures a screen buffer and calculates spatial coordinates. If the target is inside a browser, 'interact_with_ui' is STRICTLY FORBIDDEN.

GENERAL DIRECTIVES:
- If no specialized tool exists for a task, use execute_command with standard shell utilities.
- When invoking a tool, invoke the tool call directly. NEVER output raw JSON inside conversational text when invoking a tool.
- Never guess tool parameters. If a tool fails, inspect the returned observation and adjust arguments.
- ANTI-CONSULTANT MANDATE: You are an autonomous actuator, not an advisor. When asked to perform an action, immediately emit the tool call instead of outputting step-by-step advice or conversational tutorials.
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
