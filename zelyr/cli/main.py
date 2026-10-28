"""Zelyr Universal CLI.

Entry point: ``zelyr`` (registered in pyproject.toml).

Commands
--------
zelyr                  – Opencode-tier TUI (default when no sub-command given).
zelyr chat             – Same as default: Opencode-tier TUI.
zelyr ui               – Launch the PySide6 desktop UI.
zelyr server           – Print Zelyr Distributed Server init panel (FastAPI stub).
zelyr run <prompt>     – Autonomous agent entry point (rich spinner stub).
zelyr send <text>      – Send a prompt to the running daemon via WebSocket.
zelyr tool list        – List all registered tools.
zelyr tool run         – Execute a tool by ID.
zelyr status           – Show system + daemon telemetry.
zelyr test             – Run the pytest suite in headless mode.
zelyr eval [args...]   – Run Zelyr evaluation benchmark suite.
zelyr daemon           – Launch the background Zelyr daemon process.
zelyr serve            – Start the distributed FastAPI server.
"""

from __future__ import annotations

import os
os.environ["PYTHONWARNINGS"] = "ignore::SyntaxWarning:anyio"

import sys
import json
import asyncio
import argparse
import subprocess
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Optional rich integration — graceful degradation to plain print() if absent.
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich import box
    import time as _time

    _console = Console()
    _RICH = True
except ImportError:  # pragma: no cover
    _RICH = False
    _console = None  # type: ignore[assignment]


def _print(msg: str) -> None:
    """Print via rich Console when available, otherwise plain print."""
    if _RICH and _console:
        _console.print(msg, soft_wrap=True)
    else:
        print(msg)


def _panel(title: str, body: str, style: str = "bold violet") -> None:
    """Render a rich Panel, or a plain text fallback."""
    if _RICH and _console:
        _console.print(Panel(body, title=title, border_style=style, box=box.HEAVY))
    else:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
        print(body)
        print(f"{'=' * 60}\n")


def _spinner_context(label: str):
    """Return a context manager that shows a rich spinner or a no-op."""
    if _RICH and _console:
        from rich.live import Live
        from rich.spinner import Spinner
        from rich.text import Text

        return Live(
            Spinner("dots", text=Text(f" {label}", style="bold cyan")),
            console=_console,
            refresh_per_second=20,
            transient=True,
        )
    else:
        class _Noop:
            def __enter__(self): print(f"[…] {label}"); return self
            def __exit__(self, *_): pass
        return _Noop()


# ===========================================================================
# Command implementations
# ===========================================================================

def cmd_chat(_args: argparse.Namespace) -> None:
    """Launch the Opencode-tier TUI (default when no sub-command is given)."""
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "zelyr.cli.main", "server"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    try:
        from zelyr.cli.tui import run_tui
        run_tui()
    finally:
        server_proc.terminate()
        server_proc.wait()


def cmd_repl(_args: argparse.Namespace) -> None:
    """Launch the modern inline streaming CLI (Tokyo Night theme)."""
    from zelyr.cli.repl import run_repl
    run_repl()


def cmd_ui(_args: argparse.Namespace) -> None:
    """Launch the Zelyr PySide6 desktop UI."""
    from zelyr.utils.optional_deps import require_extra, OptionalDependencyError

    try:
        require_extra("PySide6", "gui", "the desktop graphical interface")
    except OptionalDependencyError as exc:
        try:
            from rich.markup import escape
            err_msg = escape(str(exc))
        except Exception:
            err_msg = str(exc)
        _print(f"[bold red]✗[/bold red] {err_msg}")
        sys.exit(1)

    _panel(
        "Zelyr UI",
        "[bold]Launching Zelyr desktop interface…[/bold]\n"
        "PySide6 event loop starting.",
        style="bold cyan",
    )
    try:
        from zelyr.gui.app import main as gui_main
        gui_main()
    except ImportError as exc:
        _print(f"[bold red]✗[/bold red] Could not import zelyr.gui.app: {exc}")
        sys.exit(1)


def cmd_server(_args: argparse.Namespace) -> None:
    """Boot the Zelyr Compute Node (FastAPI) on 0.0.0.0:8000."""
    import uvicorn

    _print(
        "[bold violet]Zelyr Compute Node[/bold violet] starting on "
        "[cyan]http://0.0.0.0:8000[/cyan]  "
        "([dim]docs → http://localhost:8000/docs[/dim])"
    )
    try:
        uvicorn.run(
            "zelyr.api.server:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
        )
    except KeyboardInterrupt:
        _print("[bold yellow]⏹  Zelyr Compute Node shut down.[/bold yellow]")



def cmd_run(args: argparse.Namespace) -> None:
    """Autonomous agent entry point — rich spinner stub."""
    prompt: str = args.prompt

    _panel(
        "Zelyr Autonomous Agent",
        f"[bold]Prompt:[/bold] {prompt}\n\n"
        "[dim]Planner, Memory, and Tool-Registry integration is on the roadmap.[/dim]",
        style="bold green",
    )

    with _spinner_context(f"Reasoning over: \"{prompt[:60]}{'…' if len(prompt) > 60 else ''}\""):
        if _RICH:
            _time.sleep(1.5)  # Simulate async planner startup

    _print(
        "[bold green]✓[/bold green] Agent stub complete.  "
        "Wire [cyan]zelyr.core.planner[/cyan] here when ready."
    )


# ---------------------------------------------------------------------------
# Legacy commands (preserved for backwards compatibility)
# ---------------------------------------------------------------------------

def cmd_send(args: argparse.Namespace) -> None:
    """Send a prompt to the running daemon via WebSocket."""
    async def _send() -> None:
        import websockets  # type: ignore[import]
        from zelyr.config import data_dir
        try:
            token_file = data_dir() / "daemon.token"
            headers = {}
            if token_file.is_file():
                tok = token_file.read_text(encoding="utf-8").strip()
                if tok:
                    headers["Authorization"] = f"Bearer {tok}"
            async with websockets.connect("ws://127.0.0.1:8000", additional_headers=headers) as ws:
                await ws.send(json.dumps({"action": "submit_task", "prompt": args.prompt}))
                _print(f"[green]✓[/green] Sent prompt to daemon: {args.prompt}")
        except Exception as exc:
            _print(f"[red]✗[/red] Failed to connect to daemon: {exc}")

    asyncio.run(_send())


def cmd_tool_list(args: argparse.Namespace) -> None:
    """List all registered tools."""
    from zelyr.tool_registry import ToolRegistry

    registry = ToolRegistry()
    tools: Dict[str, Any] = dict(registry.get_builtin_tools())

    if getattr(args, "include_plugins", False):
        try:
            from zelyr.core.plugin_manager import PluginManager

            pm = PluginManager()
            for ut in pm.load_user_tools():
                tools[ut.tool_id] = ut
        except Exception:
            pass

    _print("[bold]Available Tools:[/bold]")
    for tid, t in sorted(tools.items()):
        desc = getattr(t, "description", "")
        if not desc and getattr(t, "__doc__", None):
            desc = t.__doc__.strip().split("\n")[0]
        _print(f"  [cyan]•[/cyan] [bold]{tid}[/bold]: {desc}")


def cmd_tool_run(args: argparse.Namespace) -> None:
    """Execute a tool by ID."""
    from zelyr.tool_registry import ToolRegistry

    registry = ToolRegistry()
    tool = registry.get_tool(args.tool_id)
    if tool is None:
        try:
            from zelyr.core.plugin_manager import PluginManager

            for ut in PluginManager().load_user_tools():
                registry.register_tool(ut.tool_id, ut)
            tool = registry.get_tool(args.tool_id)
        except Exception:
            pass

    if tool is None:
        _print(f"[red]✗[/red] Tool '{args.tool_id}' not found.")
        return

    tool_args = json.loads(args.args) if args.args else {}

    async def _run() -> None:
        _print(f"[bold]Running[/bold] {args.tool_id} with args {tool_args}…")
        try:
            result = await registry.execute_async(args.tool_id, **tool_args)
            _print(f"[green]Result:[/green] {result}")
        except Exception as exc:
            _print(f"[red]Execution failed:[/red] {exc}")

    asyncio.run(_run())


def cmd_status(_args: argparse.Namespace) -> None:
    """Show system + daemon telemetry and optional extras status."""
    import psutil
    from zelyr.utils.optional_deps import is_extra_available

    cpu = psutil.cpu_percent(interval=None)
    if cpu == 0.0:
        try:
            load1, _, _ = os.getloadavg()
            cpu_cnt = os.cpu_count() or 1
            cpu = min(100.0, max(0.0, (load1 / cpu_cnt) * 100.0))
        except (AttributeError, OSError):
            cpu = psutil.cpu_percent(interval=0.01)
    ram = psutil.virtual_memory().percent

    extras = {
        "ml": is_extra_available("torch") and is_extra_available("transformers"),
        "browser": is_extra_available("playwright"),
        "audio": is_extra_available("faster_whisper") or is_extra_available("sounddevice"),
        "gui": is_extra_available("PySide6"),
    }
    extras_lines = []
    for extra_name, available in extras.items():
        status_text = "[green]installed[/green]" if available else "[dim yellow]missing[/dim yellow]"
        extras_lines.append(f"    • {extra_name:7s}: {status_text}")
    extras_body = "\n".join(extras_lines)

    _panel(
        "Zelyr System Status",
        f"  [cyan]CPU:[/cyan]     {cpu:.1f}%\n"
        f"  [cyan]RAM:[/cyan]     {ram:.1f}%\n"
        f"  [cyan]Daemon:[/cyan]  checking…\n\n"
        f"  [cyan]Optional Extras:[/cyan]\n"
        f"{extras_body}",
        style="bold blue",
    )

    daemon_online = False
    try:
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.01)
        try:
            s.connect(("127.0.0.1", 8000))
            daemon_online = True
        finally:
            s.close()
    except OSError:
        daemon_online = False

    if daemon_online:
        _print("[green]  Daemon: ONLINE[/green] (ws://127.0.0.1:8000)")
    else:
        _print("[yellow]  Daemon: OFFLINE[/yellow]")


def cmd_test(_args: argparse.Namespace) -> None:
    """Run the pytest suite in headless Qt mode."""
    _print("[bold]Running headless test suite…[/bold]")
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_swarm_blackboard.py",
            "tests/test_install_bootstrap.py",
            "tests/test_context_budget_fix.py",
            "tests/test_multi_turn_payload.py",
        ],
        env=env,
    )
    if res.returncode != 0:
        sys.exit(res.returncode)


def cmd_eval(args: argparse.Namespace, extra: Optional[List[str]] = None) -> None:
    """Run Zelyr evaluation benchmark suite."""
    from zelyr.eval.cli import run_benchmark_cli
    if "eval" in sys.argv:
        eval_idx = sys.argv.index("eval")
        eval_argv = sys.argv[eval_idx + 1 :]
    else:
        eval_argv = list(getattr(args, "eval_args", []) or [])
        if extra:
            eval_argv.extend(extra)
    exit_code = run_benchmark_cli(eval_argv)
    sys.exit(exit_code)


def cmd_daemon(_args: argparse.Namespace) -> None:
    """Launch the background Zelyr daemon process."""
    import runpy
    try:
        runpy.run_module("zelyr.server.daemon", run_name="__main__")
    except ImportError as exc:
        _print(f"[red]Error launching daemon:[/red] {exc}")
        sys.exit(1)


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the distributed FastAPI server."""
    from zelyr.api.server import start_server
    start_server(host=args.host, port=args.port)


def cmd_memory(args: argparse.Namespace) -> None:
    """Handle memory management subcommands."""
    mem_cmd = getattr(args, "mem_cmd", None)
    if mem_cmd == "compact":
        from zelyr.db.memory import MemoryStore

        store = MemoryStore()
        days = getattr(args, "days", 30.0)
        metrics = store.compact(older_than_days=days)
        _print(
            f"[bold green]✓[/bold green] Memory compaction complete:\n"
            f"  • Total Scanned: {metrics['total_scanned']}\n"
            f"  • Merged:        {metrics['merged_count']}\n"
            f"  • Decayed:       {metrics['decayed_count']}\n"
            f"  • Purged:        {metrics['purged_count']}\n"
            f"  • Duration:      {metrics['duration_seconds']:.3f}s"
        )
    else:
        _print("[yellow]Usage: zelyr memory compact [--days N][/yellow]")


# ===========================================================================
# Argument parser
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    from zelyr import __version__

    parser = argparse.ArgumentParser(
        prog="zelyr",
        description="Zelyr — Local-First AI Operating System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  zelyr                           Launch modern inline streaming CLI (default)\n"
            "  zelyr repl                      Same as default\n"
            "  zelyr chat                      Opencode-tier full-screen TUI\n"
            "  zelyr ui                        Launch desktop UI\n"
            "  zelyr server                    Show backend stub panel\n"
            "  zelyr run 'summarise my notes'  Run autonomous agent\n"
            "  zelyr status                    Show daemon + system status\n"
        ),
    )
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # ── Primary commands ────────────────────────────────────────────────────
    sub.add_parser("repl",   help="Modern inline streaming CLI (Tokyo Night, default)")
    sub.add_parser("chat",   help="Opencode-tier full-screen TUI")
    sub.add_parser("ui",     aliases=["gui"], help="Launch the PySide6 desktop UI")
    sub.add_parser("server", help="Print Zelyr Distributed Server panel (stub)")

    p_run = sub.add_parser("run", help="Autonomous agent entry point")
    p_run.add_argument("prompt", type=str, help="Natural-language task prompt")

    # ── Legacy commands (backwards-compatible) ──────────────────────────────
    p_send = sub.add_parser("send", help="Send a prompt to the daemon")
    p_send.add_argument("prompt", type=str)

    p_tool = sub.add_parser("tool", help="Manage tools")
    tool_sub = p_tool.add_subparsers(dest="tool_cmd")
    p_tool_list = tool_sub.add_parser("list", help="List all available tools")
    p_tool_list.add_argument(
        "--all",
        "--plugins",
        dest="include_plugins",
        action="store_true",
        default=False,
        help="Include custom third-party plugins in tool listing",
    )
    p_tool_run = tool_sub.add_parser("run", help="Run a specific tool")
    p_tool_run.add_argument("tool_id", type=str)
    p_tool_run.add_argument("--args", type=str, default="{}")

    p_mem = sub.add_parser("memory", help="Manage persistent memory and compaction")
    mem_sub = p_mem.add_subparsers(dest="mem_cmd")
    p_compact = mem_sub.add_parser("compact", help="Trigger REM sleep memory compaction")
    p_compact.add_argument("--days", type=float, default=30.0, help="Retention threshold in days")

    sub.add_parser("status", help="Show system + daemon status")
    sub.add_parser("test",   help="Run the automated test suite")
    p_eval = sub.add_parser("eval", help="Run Zelyr evaluation benchmark suite")
    p_eval.add_argument("eval_args", nargs=argparse.REMAINDER, help="Arguments passed to evaluation runner")
    sub.add_parser("daemon", help="Run the background daemon")

    p_serve = sub.add_parser("serve", help="Start the distributed FastAPI server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=9412)

    return parser


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in ("-v", "--version"):
        from zelyr import __version__
        print(f"zelyr {__version__}")
        return

    from zelyr.config import initialize_directories
    initialize_directories()

    parser = _build_parser()
    args, extra = parser.parse_known_args()

    dispatch = {
        "repl":   cmd_repl,
        "chat":   cmd_chat,
        "ui":     cmd_ui,
        "gui":    cmd_ui,
        "server": cmd_server,
        "run":    cmd_run,
        "send":   cmd_send,
        "status": cmd_status,
        "test":   cmd_test,
        "eval":   cmd_eval,
        "daemon": cmd_daemon,
        "serve":  cmd_serve,
    }

    if args.command == "tool":
        if args.tool_cmd == "list":
            cmd_tool_list(args)
        elif args.tool_cmd == "run":
            cmd_tool_run(args)
        else:
            parser.parse_args(["tool", "--help"])
    elif args.command == "memory":
        cmd_memory(args)
    elif args.command in dispatch:
        if args.command == "eval":
            cmd_eval(args, extra)
        else:
            if extra:
                parser.error(f"unrecognized arguments: {' '.join(extra)}")
            dispatch[args.command](args)
    else:
        # No sub-command → launch the modern inline streaming CLI.
        cmd_repl(args)


if __name__ == "__main__":
    main()
