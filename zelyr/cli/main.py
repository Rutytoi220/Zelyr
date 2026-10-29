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


def cmd_mcp(args: argparse.Namespace) -> None:
    """Handle MCP (Model Context Protocol) gateway subcommands."""
    mcp_cmd = getattr(args, "mcp_cmd", None)
    if mcp_cmd != "serve":
        _print("[yellow]Usage: zelyr mcp serve [--transport stdio|http|sse] [--host HOST] [--port PORT][/yellow]")
        return

    transport = getattr(args, "transport", "stdio")
    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8000)
    allowed_tools_raw = getattr(args, "allowed_tools", None)
    allowed_tools = [t.strip() for t in allowed_tools_raw.split(",")] if allowed_tools_raw else None
    caller_ring = getattr(args, "caller_ring", 1)
    expose_memory = not getattr(args, "no_memory", False)

    try:
        import anyio
        from zelyr.mcp.server import ZelyrMCPServerGateway

        gateway = ZelyrMCPServerGateway(
            caller_ring=caller_ring,
            allowed_tools=allowed_tools,
            expose_memory=expose_memory,
        )

        if transport == "stdio":
            anyio.run(gateway.run_stdio_async)
        else:
            _print(f"[bold cyan]Starting Zelyr MCP Gateway on {transport}://{host}:{port}[/bold cyan]")
            anyio.run(gateway.run_http_async, host, port, transport)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        _print(f"[bold red]Error running MCP server:[/bold red] {exc}")
        sys.exit(1)


def _notify_daemon_reload_routines() -> None:
    """Best-effort notification to running daemon to reload routines."""
    async def _notify() -> None:
        try:
            import websockets
            from zelyr.config import data_dir
            token_file = data_dir() / "daemon.token"
            headers = {}
            if token_file.is_file():
                tok = token_file.read_text(encoding="utf-8").strip()
                if tok:
                    headers["Authorization"] = f"Bearer {tok}"
            async with websockets.connect("ws://127.0.0.1:8000", additional_headers=headers) as ws:
                await ws.send(json.dumps({"action": "reload_routines"}))
        except Exception:
            pass

    try:
        asyncio.run(_notify())
    except Exception:
        pass


def cmd_routine(args: argparse.Namespace) -> None:
    """Dispatch routine subcommand."""
    cmd = getattr(args, "routine_cmd", None)
    if cmd in ("list", "ls") or not cmd:
        cmd_routine_list(args)
    elif cmd == "add":
        cmd_routine_add(args)
    elif cmd == "run":
        cmd_routine_run(args)
    elif cmd in ("remove", "delete", "rm"):
        cmd_routine_remove(args)
    elif cmd == "history":
        cmd_routine_history(args)
    else:
        _build_parser().parse_args(["routine", "--help"])


def cmd_routine_list(args: argparse.Namespace) -> None:
    """List scheduled routines in a formatted table."""
    from datetime import datetime
    from zelyr.core.routine import RoutineDatabase

    db = RoutineDatabase()
    include_disabled = getattr(args, "all", False)
    routines = db.sync_list_routines(include_disabled=True)

    if not include_disabled:
        routines = [r for r in routines if r.get("enabled", 1)]

    if not routines:
        _print("[dim]No routines configured. Use 'zelyr routine add' to schedule one.[/dim]")
        return

    history = {h.get("routine_id"): h.get("status") for h in db.sync_get_history(limit=100)}

    if _RICH and _console:
        from rich.table import Table

        table = Table(title="Autonomous Background Routines", box=box.ROUNDED)
        table.add_column("Name", style="bold cyan", no_wrap=True)
        table.add_column("Schedule", style="yellow")
        table.add_column("Action", style="magenta")
        table.add_column("Status", justify="center")
        table.add_column("Last Run", style="dim")
        table.add_column("Next Run", style="bold green")
        table.add_column("Last Result", justify="center")

        for r in routines:
            sched = f"{r['schedule_type']} ({r['schedule_expr']})"
            act_type = r.get("action_type", "prompt")
            payload = r.get("action_payload", "")
            if isinstance(payload, dict):
                act_str = f"{act_type}: {payload.get('tool', payload.get('hook', str(payload)))[:24]}"
            else:
                act_str = f"{act_type}: {str(payload)[:24]}"

            status_str = "[green]ENABLED[/green]" if r.get("enabled", 1) else "[yellow]DISABLED[/yellow]"

            last_run = r.get("last_run_at")
            last_run_str = datetime.fromtimestamp(last_run).strftime("%Y-%m-%d %H:%M:%S") if last_run else "Never"

            next_run = r.get("next_run_at")
            if not r.get("enabled", 1):
                next_run_str = "[yellow]Paused[/yellow]"
            elif next_run:
                next_run_str = datetime.fromtimestamp(next_run).strftime("%Y-%m-%d %H:%M:%S")
            else:
                next_run_str = "[dim]Pending[/dim]"

            last_st = history.get(r["id"], "-")
            if last_st == "success":
                res_str = "[green]SUCCESS[/green]"
            elif last_st == "failure":
                res_str = "[red]FAILURE[/red]"
            elif last_st == "timeout":
                res_str = "[yellow]TIMEOUT[/yellow]"
            else:
                res_str = "[dim]-[/dim]"

            table.add_row(r["name"], sched, act_str, status_str, last_run_str, next_run_str, res_str)

        _console.print(table)
    else:
        print(f"{'NAME':<20} {'SCHEDULE':<18} {'ACTION':<24} {'STATUS':<10} {'NEXT RUN':<20}")
        print("-" * 92)
        for r in routines:
            sched = f"{r['schedule_type']} ({r['schedule_expr']})"
            act = f"{r.get('action_type', '')}: {str(r.get('action_payload', ''))[:16]}"
            stat = "ENABLED" if r.get("enabled", 1) else "DISABLED"
            nxt = datetime.fromtimestamp(r["next_run_at"]).strftime("%Y-%m-%d %H:%M:%S") if r.get("next_run_at") else "Paused"
            print(f"{r['name']:<20} {sched:<18} {act:<24} {stat:<10} {nxt:<20}")


def cmd_routine_add(args: argparse.Namespace) -> None:
    """Add a new background routine."""
    from datetime import datetime
    from zelyr.core.routine import RoutineEngine

    engine = RoutineEngine()
    enabled = not getattr(args, "disabled", False)

    try:
        routine = asyncio.run(
            engine.add_routine_async(
                name=args.name,
                schedule_expr=args.schedule,
                schedule_type=getattr(args, "type", None),
                action_type=getattr(args, "action", "prompt"),
                action_payload=args.target,
                timeout_seconds=getattr(args, "timeout", 120.0),
                description=getattr(args, "description", ""),
                enabled=enabled,
            )
        )
        nxt_str = (
            datetime.fromtimestamp(routine.next_run_at).strftime("%Y-%m-%d %H:%M:%S")
            if routine.next_run_at
            else "Paused"
        )
        _print(f"[green]✓[/green] Added routine [bold]{routine.name}[/bold] ({routine.schedule_type}: '{routine.schedule_expr}')")
        _print(f"  ID: {routine.id} | Action: {routine.action_type} | Next Run: {nxt_str}")
        _notify_daemon_reload_routines()
    except Exception as exc:
        _print(f"[bold red]Failed to add routine:[/bold red] {exc}")
        sys.exit(1)


def cmd_routine_run(args: argparse.Namespace) -> None:
    """Manually trigger immediate execution of a routine."""
    from zelyr.core.routine import RoutineDatabase, RoutineEngine

    db = RoutineDatabase()
    routine = asyncio.run(db.get_routine(args.name_or_id))
    if not routine:
        _print(f"[bold red]Error:[/bold red] Routine '{args.name_or_id}' not found.")
        sys.exit(1)

    _print(f"[cyan]▶[/cyan] Executing routine [bold]{routine.name}[/bold] ({routine.action_type})...")
    engine = RoutineEngine()
    run = asyncio.run(engine.execute_routine_once(routine))

    if run.status == "success":
        _print(f"[bold green]✓ Execution succeeded[/bold green] ({run.duration_seconds:.2f}s)")
        if run.output:
            _print(f"[bold]Output:[/bold]\n{run.output}")
    elif run.status == "timeout":
        _print(f"[bold yellow]⚠ Execution timed out[/bold yellow] after {run.duration_seconds:.2f}s")
        if run.error:
            _print(f"[yellow]{run.error}[/yellow]")
        sys.exit(1)
    else:
        _print(f"[bold red]✗ Execution failed[/bold red] ({run.duration_seconds:.2f}s)")
        if run.error:
            _print(f"[red]{run.error}[/red]")
        sys.exit(1)


def cmd_routine_remove(args: argparse.Namespace) -> None:
    """Remove a routine by name or ID prefix."""
    from zelyr.core.routine import RoutineDatabase

    db = RoutineDatabase()
    target = args.name_or_id
    success = asyncio.run(db.delete_routine(target))
    if success:
        _print(f"[green]✓[/green] Routine [bold]{target}[/bold] removed successfully.")
        _notify_daemon_reload_routines()
    else:
        _print(f"[bold red]Error:[/bold red] Routine '{target}' not found.")
        sys.exit(1)


def cmd_routine_history(args: argparse.Namespace) -> None:
    """Show execution history of routines."""
    from datetime import datetime
    from zelyr.core.routine import RoutineDatabase

    db = RoutineDatabase()
    limit = getattr(args, "limit", 20) or 20
    target = getattr(args, "name_or_id", None)
    runs = db.sync_get_history(name_or_id=target, limit=limit)

    if not runs:
        _print("[dim]No routine execution history recorded.[/dim]")
        return

    if _RICH and _console:
        from rich.table import Table

        table = Table(title="Routine Execution History", box=box.ROUNDED)
        table.add_column("Run ID", style="dim")
        table.add_column("Routine Name", style="bold cyan")
        table.add_column("Started At", style="dim")
        table.add_column("Duration", justify="right")
        table.add_column("Status", justify="center")
        table.add_column("Summary / Error", style="italic")

        for r in runs:
            st = r.get("status", "unknown")
            if st == "success":
                status_str = "[green]SUCCESS[/green]"
                msg = str(r.get("output", "") or "")[:40]
            elif st == "timeout":
                status_str = "[yellow]TIMEOUT[/yellow]"
                msg = str(r.get("error", "") or "")[:40]
            else:
                status_str = "[red]FAILURE[/red]"
                msg = str(r.get("error", "") or "")[:40]

            started_str = (
                datetime.fromtimestamp(r["started_at"]).strftime("%Y-%m-%d %H:%M:%S")
                if r.get("started_at")
                else "-"
            )
            dur_str = f"{r.get('duration_seconds', 0.0):.2f}s"
            table.add_row(r["id"][:8], r["routine_name"], started_str, dur_str, status_str, msg)

        _console.print(table)
    else:
        print(f"{'RUN ID':<10} {'ROUTINE':<20} {'STARTED':<20} {'DURATION':<10} {'STATUS':<10} {'SUMMARY'}")
        print("-" * 85)
        for r in runs:
            st = r.get("status", "unknown").upper()
            started_str = datetime.fromtimestamp(r["started_at"]).strftime("%Y-%m-%d %H:%M:%S") if r.get("started_at") else "-"
            dur_str = f"{r.get('duration_seconds', 0.0):.2f}s"
            msg = (r.get("output") or r.get("error") or "")[:30]
            print(f"{r['id'][:8]:<10} {r['routine_name']:<20} {started_str:<20} {dur_str:<10} {st:<10} {msg}")


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

    p_mcp = sub.add_parser("mcp", help="Model Context Protocol (MCP) server gateway")
    mcp_sub = p_mcp.add_subparsers(dest="mcp_cmd")
    p_mcp_serve = mcp_sub.add_parser("serve", help="Start the MCP server gateway")
    p_mcp_serve.add_argument(
        "--transport",
        choices=["stdio", "http", "sse"],
        default="stdio",
        help="Transport type (stdio, http, or sse, default: stdio)",
    )
    p_mcp_serve.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    p_mcp_serve.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    p_mcp_serve.add_argument("--allowed-tools", default=None, help="Comma-separated list of allowed tool IDs")
    p_mcp_serve.add_argument(
        "--caller-ring",
        type=int,
        default=1,
        help="Caller security ring (0=root, 1=user, default: 1)",
    )
    p_mcp_serve.add_argument(
        "--no-memory",
        action="store_true",
        default=False,
        help="Disable read-only memory resource endpoints",
    )

    # ── Routine management ───────────────────────────────────────────────────
    p_routine = sub.add_parser("routine", help="Manage autonomous background routines and schedules")
    routine_sub = p_routine.add_subparsers(dest="routine_cmd")

    p_rt_list = routine_sub.add_parser("list", aliases=["ls"], help="List all scheduled routines")
    p_rt_list.add_argument("--all", "-a", action="store_true", help="Include disabled routines")

    p_rt_add = routine_sub.add_parser("add", help="Add a new background routine")
    p_rt_add.add_argument("name", help="Unique identifier/name for the routine")
    p_rt_add.add_argument(
        "--schedule",
        "-s",
        required=True,
        help="Cron expression (e.g. '0 8 * * *') or interval seconds (e.g. '300')",
    )
    p_rt_add.add_argument(
        "--type",
        choices=["cron", "interval"],
        default=None,
        help="Schedule type (auto-detected if omitted)",
    )
    p_rt_add.add_argument(
        "--action",
        choices=["prompt", "tool", "maintenance"],
        default="prompt",
        help="Action type to execute (default: prompt)",
    )
    p_rt_add.add_argument(
        "--target",
        required=True,
        help="Target prompt, tool payload, or maintenance hook",
    )
    p_rt_add.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Execution timeout in seconds (default: 120.0)",
    )
    p_rt_add.add_argument(
        "--description",
        "-d",
        default="",
        help="Optional description of the routine",
    )
    p_rt_add.add_argument(
        "--disabled",
        action="store_true",
        default=False,
        help="Create routine in disabled state",
    )

    p_rt_run = routine_sub.add_parser("run", help="Trigger immediate execution of a routine")
    p_rt_run.add_argument("name_or_id", help="Routine name or ID prefix")

    p_rt_del = routine_sub.add_parser("remove", aliases=["delete", "rm"], help="Remove a scheduled routine")
    p_rt_del.add_argument("name_or_id", help="Routine name or ID prefix to delete")

    p_rt_hist = routine_sub.add_parser("history", help="Show execution history of routines")
    p_rt_hist.add_argument("name_or_id", nargs="?", default=None, help="Optional routine name or ID to filter")
    p_rt_hist.add_argument(
        "--limit",
        "-n",
        type=int,
        default=20,
        help="Maximum number of history records to show (default: 20)",
    )

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
    elif args.command == "mcp":
        cmd_mcp(args)
    elif args.command == "routine":
        cmd_routine(args)
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
