"""Zelyr Modern Inline Streaming CLI (REPL).

A lightweight, non-fullscreen, reactive AI streaming CLI with native terminal scrolling,
Tokyo Night / Slate aesthetics, and sub-10ms Three-Tier automation.
"""

from __future__ import annotations
from zelyr.compat import config_dir

import asyncio
import json
import os
import re
import sys
import uuid
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
import rich.box


import zelyr.agents.native_orchestrator as nat_orch
from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.config import get_config
from zelyr.core.plugins import get_tool_schemas, load_plugins
from zelyr.db.memory import add_message, create_session, get_session_messages

# ---------------------------------------------------------------------------
# Slash Command Registry & Autocompletion
# ---------------------------------------------------------------------------

COMMANDS = {
    "/help": "Show this command reference and usage guide",
    "/model": "Switch the active language or vision model",
    "/session": "Display session stats or switch/create conversation sessions",
    "/resume": "Reload and resume the previous session history",
    "/clear": "Clear the screen and reset the conversation context window",
    "/effort": "Adjust reasoning effort tier (low -> adhd)",
    "/thought": "Display the last reasoning trace (or toggle with Ctrl+O)",
    "/tools": "List active Three-Tier and built-in tools in registry",
    "/routine": "View active scheduled background routines and next run times",
    "/memory": "Manage persistent semantic memories (/memory list, /memory add <fact>, /memory <query>)",
    "/rules": "Display active global and local project instructions",
    "/status": "Display live system, bridge, desktop, and tool telemetry",
    "/doctor": "Run diagnostic health checks across all 6 subsystems",
    "/undo": "Rollback the last modified file to its previous state",
    "/exit": "Quit the Zelyr REPL cleanly",
}

slash_completer = WordCompleter(
    list(COMMANDS.keys()) + ["/memory list", "/memory add", "/thought toggle", "/session new"],
    meta_dict={
        **COMMANDS,
        "/thought toggle": "Toggle reasoning visibility (Expanded <-> Collapsed)",
        "/memory list": "Display all stored semantic memories",
        "/memory add": "Store a permanent fact in semantic memory",
        "/session new": "Start a fresh conversation session",
        "/doctor": "Run diagnostic health checks across all 6 subsystems",
        "/undo": "Rollback the last modified file to its previous state",
    },
    sentence=True,
    ignore_case=True,
)

# ---------------------------------------------------------------------------
# Tokyo Night / Slate Theme Palette (High Contrast)
# ---------------------------------------------------------------------------
# Prompt glyph:         #7aa2f7 (Soft Blue)
# Labels / Subtitles:   #a9b1d6 (Crisp Readable Slate)
# Values / Active:      #7dcfff (Vibrant Cyan)
# Session:              #bb9af7 (Soft Lavender)
# Tool execution badge: #e0af68 (Warm Amber)
# Tool outputs/Success: #9ece6a (Subtle Emerald)
# User input:           #c0caf5 (Crisp Foreground)
# Error:                #f7768e (Soft Red)

TOKYO_NIGHT_THEME = Theme({
    "prompt.glyph": "#7aa2f7 bold",
    "prompt.prefix": "#a9b1d6",
    "user.input": "#c0caf5",
    "reasoning": "#a9b1d6 italic",
    "tool.badge": "#e0af68 bold",
    "tool.name": "#e0af68",
    "tool.output": "#9ece6a",
    "success": "#9ece6a bold",
    "error": "#f7768e bold",
    "dim": "#a9b1d6",
    "label": "#a9b1d6",
    "value": "#7dcfff bold",
    "session": "#bb9af7",
    "title": "#7aa2f7 bold",
    "accent": "#bb9af7",
})

console = Console(theme=TOKYO_NIGHT_THEME, highlight=False)

PROMPT_STYLE = Style.from_dict({
    "prompt-glyph": "#7aa2f7 bold",
    "prompt-prefix": "#a9b1d6",
    "": "#c0caf5",
    "completion-menu": "bg:#1a1b26 #c0caf5",
    "completion-menu.completion": "bg:#1a1b26 #a9b1d6",
    "completion-menu.completion.current": "bg:#283457 #7aa2f7 bold",
    "completion-menu.meta": "bg:#16161e #565f89 italic",
    "completion-menu.meta.current": "bg:#283457 #7dcfff italic",
})

BANNER = (
    r"[#7aa2f7 bold]"
    r"""
     _   __  _____ ___  __  __ 
    / \  \ \/ /_ _/ _ \|  \/  |
   / _ \  \  / | | | | | |\/| |
  / ___ \ /  \ | | |_| | |  | |
 /_/   \_\_/\_\___\___/|_|  |_|
"""
    r"[/#7aa2f7 bold]"
    "\n[#a9b1d6]Local-First AI Orchestrator  ·  Three-Tier Semantic Automation  ·  Tokyo Night[/#a9b1d6]\n"
)


def clear_screen() -> None:
    """Clear terminal viewport using standard ANSI escape codes."""
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def ensure_ollama_running(base_url: str = "http://127.0.0.1:11434") -> bool:
    """Checks if Ollama daemon is responsive; if not, attempts to spawn it."""
    import urllib.request
    clean_url = (base_url or "http://127.0.0.1:11434").rstrip("/")
    try:
        req = urllib.request.Request(f"{clean_url}/api/version", method="GET")
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            if resp.status == 200:
                return True
    except Exception:
        pass

    try:
        from zelyr.services.ollama_monitor import OllamaHealthMonitor
        OllamaHealthMonitor.spawn_ollama_service()
        import time
        for _ in range(6):
            time.sleep(0.5)
            try:
                req = urllib.request.Request(f"{clean_url}/api/version", method="GET")
                with urllib.request.urlopen(req, timeout=1.0) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass
    except Exception:
        pass
    return False


def select_model_interactive(models: List[str], active_model: str) -> Optional[str]:
    """Interactive inline model selector using Tokyo Night palette and ANSI cursor keys."""
    import select
    import termios
    import tty

    if not models or not sys.stdin.isatty():
        return models[0] if models else None

    idx = 0
    if active_model in models:
        idx = models.index(active_model)

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    num_lines = len(models)

    def render_list(current_idx: int) -> None:
        for i, m in enumerate(models):
            is_active = (m == active_model)
            if i == current_idx:
                tag = " \033[38;2;187;154;247m(active)\033[0m" if is_active else ""
                sys.stdout.write(f"  \033[38;2;122;162;247m\033[1m▶ \033[0m\033[38;2;125;207;255m\033[1m{m}\033[0m{tag}\r\n")
            else:
                tag = " \033[38;2;169;177;214m\033[2m(active)\033[0m" if is_active else ""
                sys.stdout.write(f"    \033[38;2;169;177;214m{m}\033[0m{tag}\r\n")
        sys.stdout.flush()

    try:
        tty.setraw(fd)
        sys.stdout.write("\033[?25l")  # Hide cursor
        sys.stdout.write("\r\n\033[38;2;122;162;247m\033[1mSelect Model\033[0m \033[38;2;169;177;214m(↑/↓: navigate, Enter: confirm, Esc: cancel)\033[0m\r\n")
        render_list(idx)

        while True:
            r, _, _ = select.select([sys.stdin], [], [], None)
            ch = sys.stdin.read(1)

            if ch == "\x1b":
                r2, _, _ = select.select([sys.stdin], [], [], 0.05)
                if r2:
                    c2 = sys.stdin.read(1)
                    if c2 in ("[", "O"):
                        c3 = sys.stdin.read(1)
                        if c3 == "A":  # Up
                            idx = (idx - 1) % num_lines
                        elif c3 == "B":  # Down
                            idx = (idx + 1) % num_lines
                    sys.stdout.write(f"\033[{num_lines}A\r")
                    render_list(idx)
                else:
                    return None  # Standalone Esc
            elif ch in ("\r", "\n"):
                return models[idx]
            elif ch == "\x03":  # Ctrl+C
                return None
            elif ch in ("k", "K"):
                idx = (idx - 1) % num_lines
                sys.stdout.write(f"\033[{num_lines}A\r")
                render_list(idx)
            elif ch in ("j", "J"):
                idx = (idx + 1) % num_lines
                sys.stdout.write(f"\033[{num_lines}A\r")
                render_list(idx)
    finally:
        # Erase the selection lines and restore cursor
        sys.stdout.write(f"\033[{num_lines + 2}A\r\033[J")
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class InlineRepl:
    """Manages the interactive inline terminal REPL session."""

    def __init__(self, session_id: Optional[str] = None):
        self.config = get_config()
        from zelyr.memory.sessions import SessionManager

        self.session_mgr = SessionManager()
        self.session_id = session_id or self.session_mgr.create_session(
            title="CLI Session",
            model_name=getattr(self.config, "ollama_model", "qwen2.5-coder:7b"),
        )
        if not self.session_mgr.get_session(self.session_id):
            self.session_mgr.create_session(
                self.session_id,
                title="CLI Session",
                model_name=getattr(self.config, "ollama_model", "qwen2.5-coder:7b"),
            )
        create_session(self.session_id, title="CLI Session")

        self.history_dir = config_dir()
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.history_dir / "repl_history"

        self.orchestrator = NativeOrchestrator()
        load_plugins()

        self.last_thought: str = ""
        self._current_task: Optional[asyncio.Task] = None
        kb = KeyBindings()

        @kb.add("c-o")
        def _toggle_thinking(event):
            run_in_terminal(self.toggle_thinking)

        self.prompt_session: PromptSession = PromptSession(
            history=FileHistory(str(self.history_file)),
            style=PROMPT_STYLE,
            completer=slash_completer,
            complete_while_typing=True,
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=kb,
        )
        self.is_running = True
        self.bridge = None
        self._bridge_server = None
        ensure_ollama_running(self.config.ollama_base_url)

    def toggle_thinking(self) -> bool:
        """Toggles thinking visibility between expanded and collapsed, and saves to config."""
        self.config.show_thinking = not getattr(self.config, "show_thinking", True)
        self.config.save()
        state_label = "EXPANDED" if self.config.show_thinking else "COLLAPSED"
        console.print(f"\n[#7aa2f7]🧠 Thinking visibility set to:[/] [#7dcfff]{state_label}[/] [dim](saved to config)[/dim]")
        if self.config.show_thinking and self.last_thought:
            console.print(f"[dim italic #7982a9]▾ Last Thought Trace:\n{self.last_thought}[/]\n[dim #3b4261]───[/]\n")
        return self.config.show_thinking

    def print_banner(self) -> None:
        console.print(BANNER)
        think_mode = "Expanded" if getattr(self.config, "show_thinking", True) else "Collapsed"
        console.print(
            f"[label]Model:[/label] [value]{self.config.ollama_model}[/value]  ·  "
            f"[label]Session:[/label] [session]{self.session_id[:8]}[/session]  ·  "
            f"[label]Thinking:[/label] [value]{think_mode}[/value] [dim](Ctrl+O)[/dim]  ·  "
            f"[label]Type[/label] [#e0af68]/help[/#e0af68] [label]for commands or[/label] [#e0af68]/exit[/#e0af68] [label]to quit.[/label]\n"
        )

    def print_help(self) -> None:
        console.print("\n[title]Available Commands:[/title]")
        for cmd, desc in COMMANDS.items():
            console.print(f"  [#e0af68]{cmd:<16}[/#e0af68] [label]{desc}[/label]")
        console.print()

    def handle_command(self, cmd_text: str) -> bool:
        """Handles slash commands. Returns True if handled, False otherwise."""
        parts = cmd_text.strip().split()
        if not parts:
            return False

        cmd = parts[0].lower()
        if cmd in ("/exit", "/quit"):
            self.is_running = False
            console.print("[dim]Goodbye.[/dim]")
            return True

        elif cmd == "/clear":
            clear_screen()
            self.session_id = self.session_mgr.create_session(
                title="CLI Session",
                model_name=getattr(self.config, "ollama_model", "qwen2.5-coder:7b"),
            )
            create_session(self.session_id, title="CLI Session")
            self.print_banner()
            console.print(f"[dim]Context window cleared. Active session: {self.session_id[:8]}[/dim]\n")
            return True

        elif cmd == "/help":
            self.print_help()
            return True

        elif cmd == "/effort":
            if len(parts) > 1:
                tier = parts[1].lower()
                setattr(self.config, "effort_tier", tier)
                self.config.save()
                console.print(f"[success]✓ Reasoning effort tier set to '{tier}'.[/success]\n")
            else:
                current = getattr(self.config, "effort_tier", "medium")
                console.print(f"[label]Current effort tier:[/label] [value]{current}[/value]. [label]Usage: /effort <low|medium|high|ultra|adhd>[/label]\n")
            return True

        elif cmd == "/session":
            if len(parts) > 1 and parts[1].lower() in ("new", "--new"):
                self.session_id = self.session_mgr.create_session(
                    title="CLI Session",
                    model_name=getattr(self.config, "ollama_model", "qwen2.5-coder:7b"),
                )
                create_session(self.session_id, title=self.session_id)
                console.print(f"[success]✓ Started new session '{self.session_id}'.[/success]\n")
                return True
            elif len(parts) > 1:
                target_id = " ".join(parts[1:]).strip()
                if not self.session_mgr.get_session(target_id):
                    self.session_mgr.create_session(
                        target_id,
                        title=target_id,
                        model_name=getattr(self.config, "ollama_model", "qwen2.5-coder:7b"),
                    )
                self.session_id = target_id
                create_session(self.session_id, title=self.session_id)
                console.print(f"[success]✓ Switched to session '{self.session_id}'.[/success]\n")
                return True
            else:
                sess = self.session_mgr.get_session(self.session_id)
                messages = self.session_mgr.get_messages(self.session_id)
                tokens = self.session_mgr.count_tokens(messages)
                created_at = sess.get("created_at") if sess else time.time()
                import datetime

                created_str = (
                    datetime.datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M:%S")
                    if isinstance(created_at, (int, float))
                    else str(created_at)
                )

                panel_text = (
                    f"[label]Session ID:[/label] [session]{self.session_id}[/session]\n"
                    f"[label]Title:[/label] [value]{sess.get('title', 'CLI Session') if sess else 'CLI Session'}[/value]\n"
                    f"[label]Active Model:[/label] [value]{getattr(self.config, 'ollama_model', 'qwen2.5-coder:7b')}[/value]\n"
                    f"[label]Message Count:[/label] [value]{len(messages)}[/value] turns\n"
                    f"[label]Estimated Tokens:[/label] [value]{tokens}[/value]\n"
                    f"[label]Created At:[/label] [dim]{created_str}[/dim]"
                )
                console.print()
                console.print(Panel(
                    panel_text,
                    title="[title]Active Session Information[/title]",
                    border_style="#bb9af7",
                    padding=(1, 2),
                ))
                console.print()
                return True

        elif cmd == "/resume":
            messages = self.session_mgr.get_messages(self.session_id)
            if not messages:
                messages = get_session_messages(self.session_id)
            if not messages:
                console.print(f"[dim]No message history found for session '{self.session_id}'.[/dim]\n")
                return True

            console.print(f"\n[title]Resumed session {self.session_id[:8]} ({len(messages)} messages):[/title]")
            for m in messages:
                role = m.get("role")
                content = m.get("content", "")
                if role == "user":
                    console.print(f"\n[prompt.glyph]❯[/] [user.input]{content}[/]")
                elif role == "assistant":
                    console.print(f"\n[#7aa2f7]◈ Zelyr:[/]\n{content}")
            console.print()
            return True

        elif cmd == "/model":
            if len(parts) > 1:
                target_model = parts[1].strip()
                self.config.ollama_model = target_model
                self.config.save()
                console.print(f"[success]✓ Switched model to: {target_model}[/success]\n")
                return True

            import httpx
            try:
                base_url = getattr(self.config, "ollama_base_url", "http://127.0.0.1:11434").rstrip("/")
                with httpx.Client(timeout=3.0) as client:
                    resp = client.get(f"{base_url}/api/tags")
                    if resp.status_code == 200:
                        models = [m["name"] for m in resp.json().get("models", [])]
                        if not models:
                            console.print("[error]No models found in Ollama.[/error]\n")
                            return True
                        selected = select_model_interactive(models, self.config.ollama_model)
                        if selected:
                            self.config.ollama_model = selected
                            self.config.save()
                            console.print(f"[success]✓ Switched model to: {selected}[/success]\n")
                        else:
                            console.print("[dim]Model selection cancelled.[/dim]\n")
                        return True
            except Exception as e:
                console.print(f"[error]Could not connect to Ollama: {e}[/error]\n")
                return True

        elif cmd == "/tools":
            subcmd = parts[1].lower().strip() if len(parts) > 1 else "list"
            if subcmd in ("remove", "delete"):
                if len(parts) < 3:
                    console.print("[error]Usage: /tools remove <tool_name>[/error]\n")
                    return True
                target_name = parts[2].strip()
                from zelyr.tools.tool_manager import ToolManager
                res = ToolManager.delete_tool(target_name)
                if res.success:
                    console.print(f"[success]✓ Successfully removed dynamic tool '{target_name}'.[/success]\n")
                else:
                    console.print(f"[error]Failed to remove tool '{target_name}': {res.error}[/error]\n")
                return True
            else:
                from zelyr.tool_registry import ToolRegistry
                from zelyr.tools.tool_manager import ToolManager

                registry = ToolRegistry()
                builtins = registry.get_builtin_tools()
                dynamic_tools = ToolManager.list_tools()

                table = Table(
                    title="Active Zelyr Tools",
                    box=rich.box.ROUNDED,
                    show_header=True,
                    header_style="bold #7aa2f7",
                )
                table.add_column("Tool Name", style="bold #e0af68")
                table.add_column("Source", justify="center", style="#7dcfff")
                table.add_column("Ring", justify="center", style="#bb9af7")
                table.add_column("Description", style="#c0caf5")

                for bname, defn in sorted(builtins.items()):
                    table.add_row(bname, "Built-in", "0" if bname != "echo" else "1", defn.description[:75])
                for t in dynamic_tools:
                    table.add_row(t["name"], f"Dynamic (Tier {t['tier']})", str(t["ring"]), (t["description"] or "-")[:75])

                console.print()
                console.print(table)
                console.print(f"[dim]Total: {len(builtins) + len(dynamic_tools)} active tools. Use /tools remove <name> to delete dynamic plugins.[/dim]\n")
                return True

        elif cmd == "/routine":
            try:
                from zelyr.core.routine import RoutineDatabase

                rdb = RoutineDatabase()
                routines = rdb.sync_list_routines(include_disabled=True)
                if not routines:
                    console.print("[dim]No scheduled background routines found.[/dim]\n")
                else:
                    table = Table(
                        title="Scheduled Autonomous Routines",
                        box=rich.box.ROUNDED,
                        show_header=True,
                        header_style="bold #7aa2f7",
                    )
                    table.add_column("ID", style="#7dcfff")
                    table.add_column("Name", style="bold #e0af68")
                    table.add_column("Schedule", style="#bb9af7")
                    table.add_column("Status", justify="center")
                    table.add_column("Next Run", style="#9ece6a")
                    table.add_column("Action", style="#c0caf5")

                    import datetime

                    for r in routines:
                        status_str = "[green]ENABLED[/green]" if r.get("enabled", 1) else "[dim red]DISABLED[/dim red]"
                        next_run = r.get("next_run_at")
                        if isinstance(next_run, (int, float)):
                            next_run_str = datetime.datetime.fromtimestamp(next_run).strftime("%Y-%m-%d %H:%M:%S")
                        else:
                            next_run_str = str(next_run or "Pending")
                        table.add_row(
                            str(r.get("id", "")),
                            str(r.get("name", "")),
                            f"{r.get('schedule_type')}: {r.get('schedule_expr')}",
                            status_str,
                            next_run_str,
                            str(r.get("action_type", "")),
                        )
                    console.print()
                    console.print(table)
                    console.print(f"[dim]Total: {len(routines)} scheduled routines. Manage via 'zelyr routine'.[/dim]\n")
            except Exception as exc:
                console.print(f"[error]Could not query routine database: {exc}[/error]\n")
            return True

        elif cmd == "/memory":
            subcmd = parts[1].lower().strip() if len(parts) > 1 else "list"
            if subcmd == "list" or len(parts) == 1:
                from zelyr.memory.semantic import get_all_memories

                mems = get_all_memories()
                if not mems:
                    console.print("[dim]No persistent semantic memories stored yet.[/dim]\n")
                else:
                    console.print(f"\n[title]Stored Semantic Memories ({len(mems)}):[/title]")
                    for m in mems:
                        mem_id = m.get("id")
                        created = m.get("created_at", "")
                        content = m.get("content", "")
                        console.print(f"  [value]#{mem_id}[/value] [dim]({created})[/dim]: [#c0caf5]{content}[/]")
                    console.print()
                return True
            elif subcmd == "add":
                fact = cmd_text.strip().split("add", 1)[1].strip() if "add" in cmd_text else ""
                if not fact:
                    console.print("[error]Usage: /memory add <fact to remember>[/error]\n")
                    return True
                from zelyr.memory.semantic import add_memory

                mem_id = add_memory(fact)
                console.print(f"[success]✓ Remembered permanent fact (#{mem_id}):[/success] [value]{fact}[/value]\n")
                return True
            else:
                # Query / search
                search_q = cmd_text.strip().split(" ", 1)[1].strip()
                if search_q.startswith("search "):
                    search_q = search_q[len("search "):].strip()
                try:
                    from zelyr.db.memory import MemoryStore

                    store = MemoryStore()
                    results = store.search(search_q, limit=10)
                    if not results:
                        console.print(f"[dim]No memories found matching query '{search_q}'.[/dim]\n")
                    else:
                        console.print(f"\n[title]Search Results for '{search_q}' ({len(results)}):[/title]")
                        for r in results:
                            console.print(f"  [value]#{r.get('id')}[/value] [label][{r.get('category')}][/label] [#c0caf5]{r.get('content')}[/]")
                        console.print()
                except Exception as exc:
                    console.print(f"[error]Memory search error: {exc}[/error]\n")
                return True

        elif cmd == "/rules":
            from zelyr.core.instructions import InstructionManager
            im = InstructionManager()
            console.print(f"\n[title]Active Zelyr Rules & Instructions:[/title]\n")

            global_text = im.get_global_instructions()
            proj_path = im.get_project_instructions_path()
            proj_text = im.get_project_instructions()

            console.print(f"[label]Global Instructions[/label] [dim]({im.global_path}):[/dim]")
            if global_text:
                console.print(f"[#c0caf5]{global_text}[/]\n")
            else:
                console.print("[dim](None defined)[/dim]\n")

            proj_label = f"({proj_path})" if proj_path else "(None found in workspace)"
            console.print(f"[label]Project Workspace Rules[/label] [dim]{proj_label}:[/dim]")
            if proj_text:
                console.print(f"[#c0caf5]{proj_text}[/]\n")
            else:
                console.print("[dim](None defined: create ZELYR.md or .zelyrrules in workspace)[/dim]\n")
            return True

        elif cmd in ("/thought", "/think"):
            if len(parts) > 1 and parts[1].lower() in ("toggle", "t"):
                self.toggle_thinking()
                return True
            if self.last_thought:
                console.print(f"\n[dim italic #7982a9]▾ Last Thought Trace:\n{self.last_thought}[/]\n[dim #3b4261]───[/]\n")
            else:
                console.print("[dim]No thought trace stored from the last response.[/dim]\n")
            return True

        elif cmd == "/status":
            import shutil
            import subprocess
            from zelyr.tools.browser_extension import get_bridge

            # 1. WebSocket Extension Bridge Telemetry
            bridge = get_bridge()
            is_listening = bridge.server is not None
            client_count = bridge.client_count if bridge else 0
            if is_listening:
                bridge_status = f"[#9ece6a]Active (ws://{bridge.host}:{bridge.port})[/#9ece6a] · [value]{client_count}[/value] client{'s' if client_count != 1 else ''} connected"
            else:
                bridge_status = "[#f7768e]Inactive (Not listening)[/#f7768e]"

            # 2. Desktop Environment Telemetry
            hypr_sig = os.environ.get("HYPRLAND_INSTANCE_SIGNATURE", "")
            wayland_disp = os.environ.get("WAYLAND_DISPLAY", "")
            ws_str = "N/A"
            if hypr_sig and shutil.which("hyprctl"):
                try:
                    res = subprocess.run(["hyprctl", "activeworkspace", "-j"], capture_output=True, text=True, timeout=1.0)
                    if res.returncode == 0:
                        ws_data = json.loads(res.stdout)
                        ws_str = f"Workspace {ws_data.get('id', ws_data.get('name', 'Unknown'))}"
                except Exception:
                    pass
            desktop_status = f"Wayland: [value]{wayland_disp or 'None'}[/value] · Hyprland: [value]{'Active' if hypr_sig else 'Inactive'}[/value] ({ws_str})"

            # 3. Audio / Media Control Telemetry (MPRIS)
            media_status = "No active media players detected"
            if shutil.which("playerctl"):
                try:
                    res = subprocess.run(
                        ["playerctl", "metadata", "--format", "{{playerName}}: {{title}} - {{artist}} ({{status}})"],
                        capture_output=True, text=True, timeout=1.0,
                    )
                    if res.returncode == 0 and res.stdout.strip():
                        media_status = res.stdout.strip()
                except Exception:
                    pass

            # 4. Registered Tools Telemetry
            schemas = get_tool_schemas(0)
            tool_status = f"[value]{len(schemas)}[/value] dynamic tools active across Tier 1, Tier 2, and Tier 3"

            # 5. VRAM / CUDA Engine Telemetry
            cuda_status = "Torch/CUDA runtime not resident (lazy init)"
            try:
                import torch
                if torch.cuda.is_available():
                    alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                    res_mb = torch.cuda.memory_reserved() / (1024 ** 2)
                    dev_name = torch.cuda.get_device_name(0)
                    cuda_status = f"[#9ece6a]Active[/#9ece6a] ({dev_name}) · VRAM Allocated: [value]{alloc_mb:.1f} MB[/value] · Reserved: [value]{res_mb:.1f} MB[/value]"
                else:
                    cuda_status = "CPU mode (CUDA not available)"
            except ImportError:
                cuda_status = "Torch not installed in active environment"
            except Exception as e:
                cuda_status = f"Unknown ({e})"

            panel_content = (
                f"[label]WebSocket Extension Bridge:[/label] {bridge_status}\n"
                f"[label]Desktop Environment:[/label] {desktop_status}\n"
                f"[label]Audio / Media (MPRIS):[/label] [value]{media_status}[/value]\n"
                f"[label]Registered Tool Count:[/label] {tool_status}\n"
                f"[label]VRAM / CUDA Grounding:[/label] {cuda_status}"
            )
            console.print()
            console.print(Panel(
                panel_content,
                title="[title]Zelyr System & Runtime Telemetry[/title]",
                border_style="#7aa2f7",
                padding=(1, 2),
            ))
            console.print()
            return True

        elif cmd == "/undo":
            from zelyr.tools.workspace_file import ManageWorkspaceFileTool, get_last_modified_file

            if len(parts) > 1:
                target_path_str = " ".join(parts[1:]).strip()
            else:
                last_file = get_last_modified_file()
                if not last_file:
                    console.print("[dim yellow]No recent file modification detected in this session.[/dim yellow]\n")
                    return True
                target_path_str = str(last_file)

            def _run_coro(coro):
                import concurrent.futures
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop and loop.is_running():
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        return executor.submit(asyncio.run, coro).result()
                else:
                    return asyncio.run(coro)

            tool = ManageWorkspaceFileTool()
            res = _run_coro(tool.execute({"action": "rollback", "path": target_path_str}))

            if not res.success:
                err_msg = res.error or "Unknown error occurred during rollback."
                console.print(f"[error]✗ Rollback Failed: {err_msg}[/error]\n")
            else:
                out = res.output or {}
                panel_text = (
                    f"[#9ece6a bold]✓ Rollback Successful[/#9ece6a bold]\n\n"
                    f"[label]Restored:[/label] [value]{out.get('restored_path', target_path_str)}[/value]\n"
                    f"[label]From:[/label] [dim]{out.get('backup_source', 'N/A')}[/dim]\n"
                    f"[label]Size:[/label] [value]{out.get('bytes', 0)}[/value] bytes"
                )
                console.print()
                console.print(Panel(
                    panel_text,
                    title="[title]Zelyr File Rollback[/title]",
                    border_style="#9ece6a",
                    padding=(1, 2),
                ))
                console.print()
            return True

        elif cmd == "/doctor":
            import shutil
            import subprocess
            import socket

            # 1. Ollama LLM Engine Check
            ollama_status = "FAIL"
            ollama_details = ""
            base_url = getattr(self.config, "ollama_base_url", "http://127.0.0.1:11434").rstrip("/")
            try:
                import urllib.request
                req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
                with urllib.request.urlopen(req, timeout=1.5) as resp:
                    if resp.status == 200:
                        tags_data = json.loads(resp.read().decode("utf-8"))
                        models = [m.get("name", "") for m in tags_data.get("models", [])]
                        target = self.config.ollama_model
                        target_base = target.split(":")[0] if ":" in target else target
                        resident = any(target == m or m.startswith(target_base) for m in models)
                        if resident:
                            ollama_status = "PASS"
                            ollama_details = f"Online · Model '{target}' resident"
                        else:
                            ollama_status = "WARN"
                            ollama_details = f"Online · Model '{target}' not resident in tags"
            except Exception as e:
                ollama_status = "FAIL"
                ollama_details = f"Unreachable at {base_url} ({e})"

            # 2. Desktop Compositor (Hyprland / Wayland)
            hypr_sig = os.environ.get("HYPRLAND_INSTANCE_SIGNATURE", "")
            wayland_disp = os.environ.get("WAYLAND_DISPLAY", "")
            x11_disp = os.environ.get("DISPLAY", "")
            if hypr_sig and shutil.which("hyprctl"):
                try:
                    res = subprocess.run(["hyprctl", "version"], capture_output=True, text=True, timeout=1.0)
                    if res.returncode == 0:
                        desktop_check_status = "PASS"
                        desktop_check_details = f"Hyprland active ({hypr_sig[:8]}...) · Wayland {wayland_disp or 'present'}"
                    else:
                        desktop_check_status = "WARN"
                        desktop_check_details = f"Hyprland signature set but hyprctl exit code {res.returncode}"
                except Exception as e:
                    desktop_check_status = "WARN"
                    desktop_check_details = f"Hyprland signature set but probe error ({e})"
            elif wayland_disp or x11_disp:
                desktop_check_status = "WARN"
                desktop_check_details = f"Non-Hyprland display session ({wayland_disp or x11_disp})"
            else:
                desktop_check_status = "FAIL"
                desktop_check_details = "No Wayland or X11 display server detected in environment"

            # 3. Wayland Clipboard
            has_copy = shutil.which("wl-copy") is not None
            has_paste = shutil.which("wl-paste") is not None
            if has_copy and has_paste:
                try:
                    subprocess.run(["wl-paste", "--no-newline"], capture_output=True, timeout=0.5)
                    clip_status = "PASS"
                    clip_details = "wl-copy & wl-paste operational"
                except Exception:
                    clip_status = "PASS"
                    clip_details = "wl-copy & wl-paste installed"
            else:
                clip_status = "WARN"
                missing_tools = []
                if not has_copy:
                    missing_tools.append("wl-copy")
                if not has_paste:
                    missing_tools.append("wl-paste")
                clip_details = f"Missing clipboard tools: {', '.join(missing_tools)}"

            # 4. Audio / MPRIS Bus
            if shutil.which("playerctl"):
                try:
                    res = subprocess.run(["playerctl", "-l"], capture_output=True, text=True, timeout=1.0)
                    players = [p.strip() for p in res.stdout.strip().splitlines() if p.strip()]
                    if players:
                        media_check_status = "PASS"
                        media_check_details = f"playerctl active · Players: {', '.join(players)}"
                    else:
                        media_check_status = "PASS"
                        media_check_details = "playerctl installed · No active players running"
                except Exception as e:
                    media_check_status = "WARN"
                    media_check_details = f"playerctl probe error ({e})"
            else:
                media_check_status = "WARN"
                media_check_details = "playerctl not installed in PATH"

            # 5. Extension WebSocket Bridge
            from zelyr.tools.browser_extension import get_bridge
            bridge = get_bridge()
            is_listening = bridge.server is not None
            clients = bridge.client_count if bridge else 0
            port = bridge.port if bridge else 41144
            sock_bound = False
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    sock_bound = True
            except Exception:
                sock_bound = False

            if is_listening or sock_bound:
                if clients > 0:
                    bridge_check_status = "PASS"
                    bridge_check_details = f"Listening on ws://127.0.0.1:{port} · {clients} client(s) connected"
                else:
                    bridge_check_status = "WARN"
                    bridge_check_details = f"Listening on ws://127.0.0.1:{port} · 0 browser clients connected"
            else:
                bridge_check_status = "FAIL"
                bridge_check_details = f"Bridge server not listening on port {port}"

            # 6. Dynamic Tool Registry
            schemas = get_tool_schemas(0)
            tool_names = set()
            for s in schemas:
                fn = s.get("function", {})
                if fn.get("name"):
                    tool_names.add(fn["name"])
            required_tools = ["send_desktop_notification", "inspect_network", "manage_workspace_file"]
            missing_tools = [t for t in required_tools if t not in tool_names]

            if len(schemas) >= 30 and not missing_tools:
                tool_check_status = "PASS"
                tool_check_details = f"{len(schemas)} dynamic tools active across Tiers 1-3"
            elif missing_tools:
                tool_check_status = "WARN"
                tool_check_details = f"{len(schemas)} tools active · Missing: {', '.join(missing_tools)}"
            else:
                tool_check_status = "WARN"
                tool_check_details = f"Only {len(schemas)} tools active (< 30)"

            subsystems = [
                ("Ollama Engine", ollama_status, ollama_details),
                ("Compositor", desktop_check_status, desktop_check_details),
                ("Clipboard", clip_status, clip_details),
                ("Media (MPRIS)", media_check_status, media_check_details),
                ("Browser Bridge", bridge_check_status, bridge_check_details),
                ("Tool Registry", tool_check_status, tool_check_details),
            ]

            table = Table(
                title="[title]Zelyr Subsystem Health Diagnostics[/title]",
                box=rich.box.ROUNDED,
                border_style="#3b4261",
                header_style="#7dcfff bold",
                title_style="#7aa2f7 bold",
                show_lines=True,
            )
            table.add_column("Subsystem", style="#a9b1d6 bold", width=18)
            table.add_column("Status", justify="center", width=8)
            table.add_column("Diagnostic Details", style="#c0caf5")

            for name, status, details in subsystems:
                badge = (
                    "[#9ece6a bold]PASS[/#9ece6a bold]" if status == "PASS" else
                    "[#e0af68 bold]WARN[/#e0af68 bold]" if status == "WARN" else
                    "[#f7768e bold]FAIL[/#f7768e bold]"
                )
                table.add_row(name, badge, details)

            console.print()
            console.print(table)

            pass_count = sum(1 for _, s, _ in subsystems if s == "PASS")
            total_count = len(subsystems)
            if pass_count == total_count:
                console.print(f"\n[#9ece6a bold]✓ All {total_count}/{total_count} Subsystems Healthy[/#9ece6a bold]\n")
            else:
                console.print(f"\n[#e0af68 bold]⚠️ {total_count - pass_count} Subsystem(s) Require Attention ({pass_count}/{total_count} passing)[/#e0af68 bold]")
                console.print("[dim]Remediation Recommendations:[/dim]")
                if ollama_status != "PASS":
                    console.print(f"  • [label]Ollama Engine:[/label] Run [value]ollama serve[/value] and ensure [value]ollama pull {self.config.ollama_model}[/value] is complete.")
                if desktop_check_status != "PASS":
                    console.print("  • [label]Compositor:[/label] Launch Zelyr within an active Hyprland / Wayland desktop session for window management.")
                if clip_status != "PASS":
                    console.print("  • [label]Clipboard:[/label] Install Wayland clipboard utilities ([value]sudo pacman -S wl-clipboard[/value]).")
                if media_check_status != "PASS":
                    console.print("  • [label]Media:[/label] Install [value]playerctl[/value] to control Spotify, Zen, and MPRIS audio.")
                if bridge_check_status != "PASS":
                    console.print("  • [label]Browser Bridge:[/label] Install extension in Zen/Firefox from [value]/tmp/zelyr-extension.zip[/value] and open a browser window.")
                if tool_check_status != "PASS":
                    console.print("  • [label]Tool Registry:[/label] Inspect plugin manifests in [value]~/.config/zelyr/tools.d/[/value].")
                console.print()
            return True

        return False

    async def stream_response(self, user_prompt: str) -> None:
        """Streams LLM tokens and reasoning blocks directly to stdout."""
        # 0. Ensure local AI engine is responsive
        ensure_ollama_running(self.config.ollama_base_url)

        # 1. Commit user message to SessionManager and DB
        self.session_mgr.append_message(self.session_id, "user", user_prompt)
        add_message(self.session_id, "user", user_prompt)

        # 2. Build payload with dynamic tools and session history
        tier = self.orchestrator.route_tier(user_prompt)
        from zelyr.core.plugins import filter_tool_schemas_by_tier
        dynamic_tools = filter_tool_schemas_by_tier(get_tool_schemas(0), tier)
        history = self.session_mgr.get_messages(self.session_id)
        messages: List[Dict[str, Any]] = []
        for h in history:
            role = h.get("role")
            content = h.get("content")
            if role in ("user", "assistant") and content:
                msg_entry: Dict[str, Any] = {"role": role, "content": content}
                if h.get("tool_calls"):
                    msg_entry["tool_calls"] = h["tool_calls"]
                messages.append(msg_entry)
        if not messages or messages[-1].get("content") != user_prompt:
            messages.append({"role": "user", "content": user_prompt})

        payload = {
            "stream": True,
            "tools": dynamic_tools,
            "messages": messages,
            "session_id": self.session_id,
        }

        # 3. Intercept execute_tool to render transient spinners during tool dispatch
        original_execute = nat_orch.execute_tool

        async def inline_execute_tool(name: str, **kwargs):
            if in_think:
                end_thought()

            # Automatic guardrail: prevent Tier 3 regression on browser elements
            if name == "interact_with_ui" and ("element_id" in kwargs or kwargs.get("action") in ("click_element", "fill_element")):
                console.print(f"\n  [dim yellow]⚡ Tier Route Correction:[/dim yellow] Redirecting 'interact_with_ui' with element_id to Tier 2 'interact_with_browser'")
                name = "interact_with_browser"

            hint = f"Executing {name}..."
            console.print(f"\n  [tool.badge]⚡ Tool Dispatch:[/tool.badge] [tool.name]{name}[/tool.name] [dim]{json.dumps(kwargs)}[/dim]")
            from zelyr.core.plugins import get_tier_timeout
            timeout = get_tier_timeout(name)
            timeout_int = int(timeout)
            try:
                with console.status(f"  [#e0af68]{hint}[/#e0af68]", spinner="dots"):
                    res = await asyncio.wait_for(original_execute(name, **kwargs), timeout=timeout)
            except (TimeoutError, asyncio.TimeoutError):
                err_dict = {
                    "success": False,
                    "error": f"Tool execution timed out after {timeout_int}s. Try a faster Tier 1 command or verify active window.",
                }
                res = json.dumps(err_dict)
                console.print(f"  [bold red]✗ {name} timed out ({timeout_int}s)[/bold red]\n")
                return res

            # Summarize result cleanly
            res_str = str(res)
            short_res = res_str[:160] + "..." if len(res_str) > 160 else res_str
            console.print(f"  [tool.output]✓ {name} completed:[/tool.output] [dim]{short_res}[/dim]\n")
            return res

        nat_orch.execute_tool = inline_execute_tool

        # 4. Stream tokens & manage collapsible reasoning
        show_thinking = getattr(self.config, "show_thinking", True)
        in_think = False
        think_source: Optional[str] = None
        thought_start_time: Optional[float] = None
        thought_token_count = 0
        last_thought_buffer: List[str] = []
        zelyr_header_printed = False
        full_content: List[str] = []
        tool_calls_emitted: List[Dict[str, Any]] = []

        def ensure_zelyr_header(for_thinking: bool = False) -> None:
            nonlocal zelyr_header_printed
            if not zelyr_header_printed:
                if for_thinking:
                    console.print("\n[#7aa2f7]◈ Zelyr:[/#7aa2f7]")
                else:
                    console.print("\n[#7aa2f7]◈ Zelyr:[/#7aa2f7] ", end="")
                    sys.stdout.flush()
                zelyr_header_printed = True

        def start_thought(source: str = "tag") -> None:
            nonlocal in_think, thought_start_time, think_source
            if not in_think:
                in_think = True
                think_source = source
                if thought_start_time is None:
                    thought_start_time = time.time()
                ensure_zelyr_header(for_thinking=True)
                if show_thinking:
                    console.print("[dim italic #7982a9]▾ Thinking:[/]")
                else:
                    sys.stdout.write(f"\r  \033[38;2;121;130;169m\033[3mThinking... ({thought_token_count} tokens)\033[0m")
                    sys.stdout.flush()

        def append_thought(text: str) -> None:
            nonlocal thought_token_count
            if not text:
                return
            last_thought_buffer.append(text)
            words = re.findall(r"\w+|[^\w\s]", text)
            thought_token_count += len(words) if words else (1 if text.strip() else 0)
            if show_thinking:
                sys.stdout.write(f"\033[38;2;121;130;169m\033[3m{text}\033[0m")
                sys.stdout.flush()
            else:
                sys.stdout.write(f"\r\033[K  \033[38;2;121;130;169m\033[3mThinking... ({thought_token_count} tokens)\033[0m")
                sys.stdout.flush()

        def end_thought() -> None:
            nonlocal in_think, think_source
            if in_think:
                in_think = False
                think_source = None
                elapsed = max(0.1, time.time() - (thought_start_time or time.time()))
                if show_thinking:
                    console.print("\n[dim #3b4261]───[/]\n")
                else:
                    sys.stdout.write("\r\033[K")
                    sys.stdout.flush()
                    console.print(f"[dim italic #7982a9]▾ Thought for {elapsed:.1f}s ({thought_token_count} tokens) [Ctrl+O to view][/]\n")

        def append_content(text: str) -> None:
            if not text:
                return
            if in_think:
                end_thought()
            ensure_zelyr_header(for_thinking=False)
            sys.stdout.write(f"\033[38;2;192;202;245m{text}")
            sys.stdout.flush()
            full_content.append(text)

        try:
            async for chunk in self.orchestrator.generate_stream(payload):
                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})

                # Tool calls capture
                if "tool_calls" in delta:
                    if in_think:
                        end_thought()
                    for tc in delta["tool_calls"]:
                        fn = tc.get("function", {})
                        if fn.get("name"):
                            tool_calls_emitted.append(fn)

                # Reasoning delta capture (field-based)
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if reasoning:
                    if not in_think:
                        start_thought(source="field")
                    append_thought(reasoning)
                elif in_think and think_source == "field":
                    end_thought()

                raw_content = delta.get("content", "")
                if raw_content:
                    if in_think and think_source == "field":
                        end_thought()

                    cursor = 0
                    while cursor < len(raw_content):
                        if not in_think:
                            if "<think>" in raw_content[cursor:]:
                                idx = raw_content.find("<think>", cursor)
                                before = raw_content[cursor:idx]
                                if before:
                                    append_content(before)
                                start_thought(source="tag")
                                cursor = idx + len("<think>")
                            else:
                                append_content(raw_content[cursor:])
                                break
                        else:
                            if "</think>" in raw_content[cursor:]:
                                idx = raw_content.find("</think>", cursor)
                                thought_chunk = raw_content[cursor:idx]
                                if thought_chunk:
                                    append_thought(thought_chunk)
                                end_thought()
                                cursor = idx + len("</think>")
                            else:
                                append_thought(raw_content[cursor:])
                                break

        except Exception as e:
            err_str = str(e)
            if "ConnectionRefused" in err_str or "11434" in err_str or "ConnectError" in err_str:
                console.print(
                    "\n[error]✗ LLM Connection Failure:[/error] "
                    "Could not reach Ollama backend at "
                    f"{getattr(self.config, 'ollama_base_url', 'http://127.0.0.1:11434')}.\n"
                    "[dim]Remedy: Verify Ollama is running (`systemctl start ollama` or `ollama serve`).[/dim]\n"
                )
            elif "timeout" in err_str.lower():
                console.print(
                    "\n[error]✗ Model Inference Timeout:[/error] "
                    "The model provider did not respond within the allocated timeout window.\n"
                    "[dim]Remedy: Check model load status or try a smaller quantized model via /model.[/dim]\n"
                )
            else:
                console.print(f"\n[error]✗ LLM Execution Error:[/error] {e}\n")
        finally:
            if in_think:
                end_thought()
            # Reset formatting
            sys.stdout.write("\033[0m\n\n")
            sys.stdout.flush()
            nat_orch.execute_tool = original_execute

        # 5. Store accumulated reasoning trace for inspection / Ctrl+O
        if last_thought_buffer:
            self.last_thought = "".join(last_thought_buffer).strip()

        # 6. Persist assistant output to SessionManager and DB
        final_text = "".join(full_content).strip()
        if final_text:
            self.session_mgr.append_message(
                self.session_id,
                "assistant",
                final_text,
                tool_calls=tool_calls_emitted if tool_calls_emitted else None,
            )
            add_message(self.session_id, "assistant", final_text)

    async def start_services(self) -> None:
        """Eagerly starts background services including WebExtension bridge."""
        from zelyr.tools.browser_extension import get_bridge
        self.bridge = get_bridge()
        try:
            await self.bridge.start_server()
            # Keep an active reference to prevent GC
            self._bridge_server = self.bridge.server
        except Exception as exc:
            console.print(f"[dim yellow]⚠️ Warning: Failed to bind WebExtension bridge on 41144: {exc}[/dim yellow]")

    async def do_exit(self) -> None:
        """Performs cleanup and graceful teardown of background services."""
        if hasattr(self, "session_id") and self.session_id:
            try:
                from zelyr.memory.promotion import promote_session_facts
                promote_session_facts(self.session_id)
            except Exception:
                pass

        if hasattr(self, "bridge") and self.bridge:
            await self.bridge.stop_server()
            self._bridge_server = None

    async def run_loop(self) -> None:
        """Main non-blocking interactive loop."""
        clear_screen()
        await self.start_services()
        self.print_banner()

        try:
            while self.is_running:
                try:
                    # Prompt with Soft Blue glyph #7aa2f7
                    prompt_text = FormattedText([
                        ("class:prompt-glyph", "❯ "),
                    ])
                    user_input = await asyncio.to_thread(self.prompt_session.prompt, prompt_text)
                    user_input = user_input.strip()

                    if not user_input:
                        continue

                    if user_input.startswith("/"):
                        if self.handle_command(user_input):
                            continue

                    try:
                        turn_task = asyncio.create_task(self.stream_response(user_input))
                        self._current_task = turn_task
                        await turn_task
                    except (KeyboardInterrupt, asyncio.CancelledError):
                        if not turn_task.done():
                            turn_task.cancel()
                            try:
                                await turn_task
                            except (asyncio.CancelledError, KeyboardInterrupt):
                                pass
                        console.print("\n[dim yellow]⚡ Turn interrupted by user (Ctrl+C). Prior conversation context preserved.[/dim yellow]\n")
                    finally:
                        self._current_task = None

                except (KeyboardInterrupt, EOFError):
                    console.print("\n[dim]Session closed.[/dim]")
                    break
                except Exception as e:
                    console.print(f"[error]Error: {e}[/error]")
        finally:
            await self.do_exit()


def run_repl(session_id: Optional[str] = None) -> None:
    """Synchronous entry point for the inline REPL."""
    repl = InlineRepl(session_id=session_id)
    asyncio.run(repl.run_loop())


if __name__ == "__main__":
    run_repl()
