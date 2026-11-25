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
from pathlib import Path
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

def cmd_chat(args: argparse.Namespace) -> None:
    """Launch the interactive streaming chat REPL with multi-turn persistence."""
    if getattr(args, "tui", False):
        server_proc = subprocess.Popen(
            [sys.executable, "-m", "zelyr.cli.main", "server"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            from zelyr.cli.tui import run_tui
            run_tui()
        finally:
            server_proc.terminate()
            server_proc.wait()
        return

    from zelyr.memory.sessions import SessionManager

    session_mgr = SessionManager()

    # 1. Handle --list
    if getattr(args, "list", False):
        sessions = session_mgr.list_sessions(limit=50)
        if not sessions:
            _print("[dim]No conversation sessions found.[/dim]")
            return

        from rich.table import Table
        import rich.box

        table = Table(
            title="Stored Conversation Sessions",
            box=rich.box.ROUNDED if _RICH else None,
            show_header=True,
            header_style="bold #7aa2f7",
        )
        table.add_column("Session ID", style="bold #7dcfff")
        table.add_column("Title", style="#c0caf5")
        table.add_column("Model", style="#bb9af7")
        table.add_column("Messages", justify="center", style="#9ece6a")
        table.add_column("Updated", style="#a9b1d6")

        import datetime

        for s in sessions:
            up = s.get("updated_at") or s.get("created_at") or 0
            up_str = datetime.datetime.fromtimestamp(up).strftime("%Y-%m-%d %H:%M:%S") if up else "-"
            table.add_row(
                s["session_id"],
                (s.get("title") or "New Session")[:40],
                s.get("model_name") or "-",
                str(s.get("message_count", 0)),
                up_str,
            )
        if _console:
            _console.print(table)
        else:
            _print(str(sessions))
        return

    # 2. Handle --delete
    delete_id = getattr(args, "delete", None)
    if delete_id:
        deleted = session_mgr.delete_session(delete_id)
        if deleted:
            _print(f"[bold green]✓[/bold green] Deleted session [value]{delete_id}[/value].")
        else:
            _print(f"[bold red]✗[/bold red] Session not found: [value]{delete_id}[/value].")
        return

    # 3. Determine session ID
    session_id = None
    if getattr(args, "new", False):
        session_id = session_mgr.create_session(title="New Session", model_name=getattr(args, "model", None))
    elif getattr(args, "session", None):
        session_id = args.session

    # 4. Handle model override
    model_override = getattr(args, "model", None)
    if model_override:
        from zelyr.config import get_config

        cfg = get_config()
        cfg.ollama_model = model_override
        cfg.save()

    # 5. Launch inline REPL
    from zelyr.cli.repl import run_repl

    run_repl(session_id=session_id)


def cmd_repl(args: argparse.Namespace) -> None:
    """Launch the modern inline streaming CLI (alias for cmd_chat)."""
    cmd_chat(args)


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



def _redact_cli_args(args: Any, *, verbose: bool = False) -> str:
    """Serialize tool arguments with secret redaction and concise previews for CLI visibility."""
    from zelyr.engine.normalizer import _is_sensitive_param_path, _redact_sensitive_text

    def _clean(val: Any, key_path: str = "") -> Any:
        if _is_sensitive_param_path(key_path):
            return "[REDACTED]"
        if isinstance(val, dict):
            return {
                str(k): _clean(v, f"{key_path}.{k}" if key_path else str(k))
                for k, v in val.items()
            }
        if isinstance(val, list):
            items = [_clean(v, f"{key_path}[]") for v in (val if verbose else val[:8])]
            if not verbose and len(val) > 8:
                items.append(f"... (+{len(val) - 8} more)")
            return items
        if isinstance(val, str):
            scrubbed = _redact_sensitive_text(val)
            if not verbose and len(scrubbed) > 120:
                one_line = scrubbed.replace("\n", "\\n")
                return f"{one_line[:90]}... ({len(scrubbed)} chars)"
            return scrubbed
        return val

    try:
        cleaned = _clean(args if isinstance(args, dict) else {"value": args})
        rendered = json.dumps(cleaned, ensure_ascii=False)
    except Exception:
        rendered = _redact_sensitive_text(str(args))
    if not verbose and len(rendered) > 220:
        return rendered[:217] + "..."
    return rendered


def _summarize_cli_tool_result(res_text: str, *, verbose: bool = False) -> tuple[str, Optional[int], str]:
    """Return concise (status_label, exit_code, redacted_summary) from raw tool output."""
    from zelyr.engine.normalizer import _redact_sensitive_text

    text = (res_text or "").strip()
    exit_code: Optional[int] = None
    status_label = "ok"
    summary = ""

    if text.startswith("Exit code "):
        lines = text.splitlines()
        head = lines[0].strip()
        try:
            exit_code = int(head.split("Exit code ", 1)[1].split()[0])
        except Exception:
            exit_code = None
        if exit_code is not None and exit_code != 0:
            status_label = "error"
        body = " | ".join(line.strip() for line in lines[1:] if line.strip())
        summary = body or head
    elif text.startswith("{") and text.endswith("}"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                if parsed.get("success") is False or str(parsed.get("status", "")).lower() in ("error", "failed", "rejected", "denied"):
                    status_label = "error"
                exit_val = parsed.get("exit_code")
                err_msg = parsed.get("error") or parsed.get("detail")
                out_obj = parsed.get("output") if "output" in parsed else parsed.get("result")
                if isinstance(out_obj, dict):
                    if exit_val is None and isinstance(out_obj.get("exit_code"), int):
                        exit_val = out_obj.get("exit_code")
                    if "output" in out_obj and out_obj["output"] is not None:
                        if not err_msg and out_obj.get("error"):
                            err_msg = out_obj.get("error")
                        out_obj = out_obj["output"]
                if isinstance(exit_val, int):
                    exit_code = exit_val
                    if exit_val != 0:
                        status_label = "error"
                if err_msg:
                    summary = str(err_msg)
                elif isinstance(out_obj, dict):
                    if "message" in out_obj:
                        summary = str(out_obj["message"])
                    elif "entries" in out_obj:
                        entries = out_obj.get("entries") or []
                        names = [e.get("name") for e in entries[:8] if isinstance(e, dict) and e.get("name")]
                        extra = f" (+{len(entries) - len(names)} more)" if len(entries) > len(names) else ""
                        summary = f"{len(entries)} entries: {', '.join(names)}{extra}"
                    elif "lines_read" in out_obj:
                        summary = f"read {out_obj.get('lines_read')}/{out_obj.get('total_lines')} lines from {Path(str(out_obj.get('path', ''))).name}"
                    else:
                        summary = json.dumps(out_obj, ensure_ascii=False)
                else:
                    summary = str(out_obj if out_obj is not None else parsed.get("status") or "ok")
        except Exception:
            summary = text
    else:
        if any(marker in text.lower() for marker in ("error:", "tool execution failed", "unsupported file action")):
            status_label = "error"
        summary = " | ".join(line.strip() for line in text.splitlines() if line.strip())

    scrubbed = _redact_sensitive_text(summary or status_label)
    if not verbose and len(scrubbed) > 160:
        scrubbed = scrubbed[:157] + "..."
    return status_label, exit_code, scrubbed


async def _execute_headless_run(
    prompt: str,
    model_override: Optional[str] = None,
    output_json: bool = False,
    orchestrator: Optional[Any] = None,
    verify_spec: Optional[str] = None,
    base_url_override: Optional[str] = None,
    max_steps: Optional[int] = None,
    db_path: Optional[str] = None,
    verbose: bool = False,
) -> int:
    """Execute a one-shot prompt headlessly via NativeOrchestrator."""
    import re
    from zelyr.agents.native_orchestrator import NativeOrchestrator
    from zelyr.config import ZelyrConfig, get_config
    from zelyr.core.plugins import filter_tool_schemas_by_tier, get_tool_schemas, load_plugins
    from zelyr.engine.execution_ledger import (
        extract_global_mutation_prohibition,
        is_diagnostic_test_command,
    )

    clean_prompt = (prompt or "").strip()
    if not clean_prompt:
        sys.stderr.write("Error: prompt cannot be empty.\n")
        sys.stderr.flush()
        return 1

    base_cfg = getattr(orchestrator, "config", None) or get_config()
    active_model = model_override or getattr(base_cfg, "ollama_model", "qwen2.5-coder:7b")

    load_plugins()
    cli_ledger = None
    owns_cli_ledger = False
    eff_max_steps = max_steps if max_steps is not None else (getattr(orchestrator, "max_steps", 5) if orchestrator is not None else 12)
    if orchestrator is not None:
        orch = orchestrator
        if max_steps is not None:
            orch.max_steps = max_steps
        if base_url_override and getattr(orch, "config", None) is None:
            run_cfg = ZelyrConfig.from_dict(base_cfg.to_dict()) if hasattr(base_cfg, "to_dict") else ZelyrConfig()
            run_cfg.ollama_base_url = base_cfg.get_effective_ollama_base_url(base_url_override) if hasattr(base_cfg, "get_effective_ollama_base_url") else base_url_override.rstrip("/")
            orch.config = run_cfg
    else:
        run_cfg = None
        if base_url_override:
            run_cfg = ZelyrConfig.from_dict(base_cfg.to_dict()) if hasattr(base_cfg, "to_dict") else ZelyrConfig()
            run_cfg.ollama_base_url = base_cfg.get_effective_ollama_base_url(base_url_override) if hasattr(base_cfg, "get_effective_ollama_base_url") else base_url_override.rstrip("/")
        cli_ledger, owns_cli_ledger = _open_cli_ledger(db_path)
        orch = NativeOrchestrator(
            config=run_cfg,
            central_ledger=cli_ledger,
            max_steps=eff_max_steps,
        )
    tier = orch.route_tier(clean_prompt)
    dynamic_tools = filter_tool_schemas_by_tier(get_tool_schemas(0), tier)

    payload: Dict[str, Any] = {
        "model": active_model,
        "stream": True,
        "tools": dynamic_tools,
        "messages": [{"role": "user", "content": clean_prompt}],
    }
    if verify_spec:
        payload["acceptance_criteria"] = verify_spec
        orch.acceptance_criteria = verify_spec

    original_raw_execute = getattr(orch, "_execute_tool", None)
    original_execute = orch.execute_tool
    tool_calls_emitted: List[Dict[str, Any]] = []
    exec_step_counter = 0

    if not output_json:
        p_preview = clean_prompt.replace("\n", " ")
        if not verbose and len(p_preview) > 120:
            p_preview = p_preview[:117] + "..."
        sys.stderr.write(f"[task] {p_preview} | model={active_model} | cwd={Path.cwd()}\n")
        sys.stderr.flush()

    async def headless_execute_tool(name: str, **kwargs: Any) -> str:
        nonlocal exec_step_counter
        from zelyr.tool_registry import authorize_and_validate_tool_call, canonicalize_tool_invocation

        exec_step_counter += 1
        step_no = exec_step_counter
        clean_name = NativeOrchestrator.normalize_tool_name(name)
        clean_name, norm_kwargs = canonicalize_tool_invocation(clean_name, kwargs)
        norm_kwargs = NativeOrchestrator.normalize_tool_arguments(
            clean_name, norm_kwargs, original_name=name, active_schemas=dynamic_tools
        )
        decision = authorize_and_validate_tool_call(
            clean_name,
            norm_kwargs,
            caller_ring=getattr(orch, "caller_ring", None),
            allowed_tools=getattr(orch, "allowed_tools", None),
            active_schemas=dynamic_tools,
            tool_registry=getattr(orch, "tool_registry", None),
            validate_schema=False,
            argument_source="model",
        )
        if not decision.allowed:
            tool_calls_emitted.append({"name": clean_name, "arguments": norm_kwargs, "status": "blocked"})
            if not output_json:
                sys.stderr.write(f"[step {step_no}] tool={clean_name} args={_redact_cli_args(norm_kwargs, verbose=verbose)}\n")
                sys.stderr.write(f"[result {step_no}] tool={clean_name} status=blocked reason={decision.reason or 'authorization_denied'}\n")
                sys.stderr.flush()
            return json.dumps({
                "success": False,
                "error": decision.detail,
                "error_type": decision.error_type or "AUTHORIZATION_DENIED",
                "reason": decision.reason or "authorization_denied",
            })
        clean_name = decision.tool_name
        norm_kwargs = decision.normalized_arguments

        from zelyr.engine.execution_policy import (
            authorized_policy_context,
            evaluate_direct_tool_policy_async,
        )
        from zelyr.tool_registry import resolve_tool_ring

        matched_dyn_schema = next(
            (s for s in (dynamic_tools or []) if isinstance(s, dict) and (s.get("function") or {}).get("name") == clean_name),
            None,
        )
        reg_tool_cli = (
            orch.tool_registry.get_tool(clean_name)
            if getattr(orch, "tool_registry", None) is not None and callable(getattr(orch.tool_registry, "get_tool", None))
            else None
        )
        pol_dec = await evaluate_direct_tool_policy_async(
            clean_name,
            norm_kwargs,
            tool_obj=reg_tool_cli,
            schema=matched_dyn_schema,
            required_ring=resolve_tool_ring(clean_name, tool=reg_tool_cli),
            auth_decision=decision,
            is_registered=bool(decision.allowed),
            is_external=bool(getattr(reg_tool_cli, "is_external", False) or str(clean_name).startswith("ext__")),
            confirm_callback=getattr(orch, "confirm_callback", None),
            policy_engine=getattr(orch, "execution_policy_engine", None),
            policy_config=getattr(orch, "execution_policy_config", None),
            zelyr_config=orch._active_config(),
            source_surface="cli.headless_execute_tool",
        )
        if not pol_dec.allowed:
            tool_calls_emitted.append({"name": clean_name, "arguments": norm_kwargs, "status": "blocked"})
            if not output_json:
                sys.stderr.write(f"[step {step_no}] tool={clean_name} args={_redact_cli_args(norm_kwargs, verbose=verbose)}\n")
                sys.stderr.write(f"[result {step_no}] tool={clean_name} status=blocked reason={pol_dec.reason}\n")
                sys.stderr.flush()
            return json.dumps(pol_dec.to_tool_error_payload())

        cur_ledger = getattr(orch, "current_ledger", None)
        rec_attempts = int(getattr(cur_ledger, "total_recovery_attempts", 0) or 0) if cur_ledger is not None else 0
        if not output_json:
            has_real_unres = bool(
                cur_ledger is not None
                and any(
                    (getattr(getattr(r, "state", None), "value", None) == "failed" or getattr(r, "verification_failed", False))
                    for r in cur_ledger.unresolved_failures()
                )
            )
            if rec_attempts > 0 or has_real_unres:
                sys.stderr.write(f"[recovery] attempt={max(1, rec_attempts)} tool={clean_name}\n")
            sys.stderr.write(f"[step {step_no}] tool={clean_name} args={_redact_cli_args(norm_kwargs, verbose=verbose)}\n")
            sys.stderr.flush()

        with authorized_policy_context(pol_dec):
            res = await original_execute(clean_name, **norm_kwargs)
        res_str = str(res)
        r_status, r_exit, r_summary = _summarize_cli_tool_result(res_str, verbose=verbose)
        tool_calls_emitted.append({"name": clean_name, "arguments": norm_kwargs, "status": r_status})
        if not output_json:
            exit_tag = f"exit={r_exit} " if r_exit is not None else ""
            sys.stderr.write(f"[result {step_no}] tool={clean_name} status={r_status} {exit_tag}summary={r_summary}\n")
            sys.stderr.flush()
        return res_str

    orch.execute_tool = headless_execute_tool
    final_answer = ""

    is_truncated = False
    try:
        max_steps = getattr(orch, "max_steps", 5)
        for step_idx in range(max_steps):
            step_content: List[str] = []
            step_tcs: List[Dict[str, Any]] = []
            last_finish_reason: Optional[str] = None

            if (step_idx > 0 or getattr(orch, "_preserve_current_ledger", False)) and getattr(orch, "current_ledger", None) is not None:
                payload["_reuse_current_ledger"] = True
            async for chunk in orch.generate_stream(payload):
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                choice = choices[0]
                fr = choice.get("finish_reason") or chunk.get("done_reason") or chunk.get("finish_reason")
                if fr:
                    last_finish_reason = fr

                delta = choice.get("delta", {})
                if "tool_calls" in delta:
                    # Clear any pre-tool text or leaked JSON buffered in this turn
                    step_content.clear()
                    for tc in delta["tool_calls"]:
                        fn = tc.get("function", {})
                        if fn.get("name"):
                            clean_fn_name = NativeOrchestrator.normalize_tool_name(fn["name"])
                            norm_fn_args = NativeOrchestrator.normalize_tool_arguments(
                                clean_fn_name, fn.get("arguments", {}), original_name=fn["name"], active_schemas=dynamic_tools
                            )
                            norm_fn = {
                                "name": clean_fn_name,
                                "arguments": norm_fn_args,
                            }
                            step_tcs.append(norm_fn)
                            tool_calls_emitted.append(norm_fn)
                raw_text = delta.get("content", "")
                if raw_text:
                    step_content.append(raw_text)

            raw_joined = "".join(step_content)
            clean_step = re.sub(r"<think>.*?</think>", "", raw_joined, flags=re.DOTALL)
            clean_step = re.sub(r"<think>.*$", "", clean_step, flags=re.DOTALL).strip()

            if last_finish_reason is None and hasattr(orch, "last_finish_reason"):
                last_finish_reason = orch.last_finish_reason

            if not step_tcs and clean_step:
                known_tools = orch.get_known_tool_names(payload.get("tools"))
                fallback_tc = orch.extract_first_tool_call(clean_step, known_tools)
                if fallback_tc:
                    raw_fb_name = fallback_tc.get("function", {}).get("name") or fallback_tc.get("name")
                    fn_name = NativeOrchestrator.normalize_tool_name(raw_fb_name)
                    fn_args = fallback_tc.get("function", {}).get("arguments") or fallback_tc.get("arguments", {})
                    fn_args = NativeOrchestrator.normalize_tool_arguments(
                        fn_name, fn_args, original_name=raw_fb_name, active_schemas=dynamic_tools
                    )
                    obs = await headless_execute_tool(fn_name, **fn_args)
                    call_id = f"call_headless_fallback_{step_idx}"
                    framed_obs = NativeOrchestrator.format_tool_observation(fn_name, obs)
                    from zelyr.engine.observation_framing import ObservationFramingEngine
                    fb_parsed = ObservationFramingEngine.parse_observation(framed_obs)
                    fb_status = "error" if (
                        fb_parsed.get("status") == "error"
                        or (fb_parsed.get("exit_code") is not None and fb_parsed.get("exit_code") != 0)
                    ) else "success"
                    fb_verified = fb_parsed.get("verified") is True
                    fb_ver_failed = bool(fb_parsed.get("verification_failed"))
                    fb_ver_detail = fb_parsed.get("verification_detail")

                    fb_failed_targets: List[str] = []
                    fb_verified_targets: List[str] = []
                    if verify_spec:
                        from zelyr.engine.execution_ledger import canonicalize_target, parse_acceptance_criteria, has_remaining_explicit_steps
                        from zelyr.engine.verifier import OutcomeVerifier
                        parsed_c = parse_acceptance_criteria(verify_spec)
                        if parsed_c:
                            c_ver_results = []
                            for c in parsed_c:
                                vr = OutcomeVerifier.verify(check=c["check"], target=c["target"], expected=c.get("expected"))
                                c_ver_results.append((c, vr))
                                if vr.success and hasattr(orch, "current_ledger") and orch.current_ledger is not None:
                                    orch.current_ledger.record_target_verified(c["target"], detail=f"{c['check']} verified")
                                elif not vr.success and hasattr(orch, "current_ledger") and orch.current_ledger is not None:
                                    orch.current_ledger.invalidate_target_verification(c["target"], check_type=c.get("check"))
                            if fb_status == "success":
                                if all(vr.success for _, vr in c_ver_results):
                                    fb_verified = True
                                    fb_ver_failed = False
                                    fb_verified_targets = [
                                        canonicalize_target(str(c["target"]), check_type=c.get("check"))
                                        for c, _ in c_ver_results
                                        if c.get("target")
                                    ]
                                else:
                                    fb_verified = False
                                    fb_ver_failed = True
                                    fb_failed_targets = [
                                        canonicalize_target(str(c["target"]), check_type=c.get("check"))
                                        for c, vr in c_ver_results
                                        if not vr.success and c.get("target")
                                    ]
                                    failed_c, failed_vr = next((c, vr) for c, vr in c_ver_results if not vr.success)
                                    fb_ver_detail = failed_vr.error or f"Verification failed on {failed_c.get('target')}"
                                    framed_obs = ObservationFramingEngine.frame_observation(
                                        fn_name,
                                        {
                                            "output": fb_parsed.get("output") or obs,
                                            "status": "success",
                                            "verified": False,
                                            "verification_failed": True,
                                            "target": failed_c.get("target"),
                                            "failed_criterion": f"{failed_c.get('check')} on {failed_c.get('target')}",
                                            "verifier_reason": fb_ver_detail,
                                        },
                                    )

                    if hasattr(orch, "current_ledger") and orch.current_ledger is not None:
                        if orch.current_ledger.get_record(call_id) is not None:
                            suffix = 2
                            while orch.current_ledger.get_record(f"{call_id}_{suffix}") is not None:
                                suffix += 1
                            call_id = f"{call_id}_{suffix}"
                        orch.current_ledger.record_proposed(call_id, fn_name, fn_args)
                        orch.current_ledger.record_validated(call_id)
                        orch.current_ledger.record_dispatched(call_id)
                        fb_ind_ver = None
                        if verify_spec and fb_ver_failed:
                            fb_ind_ver = {
                                "passed": False,
                                "check": verify_spec.split(":", 1)[0] if ":" in str(verify_spec) else "outcome",
                                "target": str(fb_failed_targets[0]) if fb_failed_targets else "",
                                "error": fb_ver_detail,
                            }
                        elif verify_spec and fb_verified:
                            fb_ind_ver = {
                                "passed": True,
                                "check": verify_spec.split(":", 1)[0] if ":" in str(verify_spec) else "outcome",
                                "target": str(fb_verified_targets[0]) if fb_verified_targets else "",
                            }
                        fb_struct = ObservationFramingEngine.frame_structured(
                            fn_name,
                            obs,
                            requested_tool=fn_name,
                            requested_args=fn_args if isinstance(fn_args, dict) else {},
                            dispatched=True,
                            execution_id=getattr(orch.current_ledger, "_execution_id", None),
                            correlation_id=call_id,
                            independent_verification=fb_ind_ver,
                        )
                        orch.current_ledger.record_observation(
                            call_id,
                            status=fb_status,
                            output=framed_obs,
                            error=fb_parsed.get("error"),
                            exit_code=fb_parsed.get("exit_code"),
                            verified=fb_verified,
                            verification_detail=fb_ver_detail,
                            verification_failed=fb_ver_failed,
                            failed_targets=fb_failed_targets,
                            verified_targets=fb_verified_targets,
                            structured_observation=fb_struct,
                        )
                        if verify_spec and fb_verified and not fb_ver_failed:
                            from zelyr.engine.execution_ledger import has_remaining_explicit_steps
                            tool_target = NativeOrchestrator._extract_target_from_args(fn_name, fn_args)
                            if not has_remaining_explicit_steps(clean_prompt, orch.current_ledger, current_target=tool_target):
                                payload["tools"] = []
                    payload["messages"].append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": call_id,
                            "type": "function",
                            "function": {"name": fn_name, "arguments": fn_args},
                        }],
                    })
                    payload["messages"].append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": fn_name,
                        "content": framed_obs,
                    })
                    continue

            # Detect truncation or exhaustion
            if last_finish_reason in ("length", "max_tokens"):
                is_truncated = True
                if not clean_step and not step_tcs:
                    final_answer = (
                        "⚠️ [Model Error] Model generation truncated: "
                        "token limit reached during reasoning without producing terminal response or tool call."
                    )
                    break
                else:
                    final_answer = clean_step
                    break

            if not step_tcs and not clean_step:
                final_answer = "⚠️ [Model Error] Empty response received from model without tool execution."
                break

            final_answer = clean_step
            break
    except Exception as exc:
        if hasattr(orch, "current_ledger") and orch.current_ledger is not None and orch.current_ledger.has_pending_proposals():
            orch.current_ledger.discard_pending_proposals(reason="headless_error")
        sys.stderr.write(f"Error executing headless run: {exc}\n")
        sys.stderr.flush()
        return 1
    finally:
        orch.execute_tool = original_raw_execute
        if owns_cli_ledger and cli_ledger is not None and verify_spec is None:
            cli_ledger.close()
            cli_ledger = None

    reached_step_limit = "[Notice: Reached maximum autonomous ReAct steps" in (final_answer or "")

    if final_answer.startswith("⚠️ [Connection Error]") or final_answer.startswith("⚠️ [API Error]") or final_answer.startswith("⚠️ [Request Error]") or final_answer.startswith("⚠️ [Model Error]"):
        sys.stderr.write(f"{final_answer}\n")
        sys.stderr.flush()
        if owns_cli_ledger and cli_ledger is not None:
            cli_ledger.close()
            cli_ledger = None
        if output_json:
            from zelyr.engine.execution_ledger import TurnExecutionLedger
            payload_dict = {
                "prompt": clean_prompt,
                "model": active_model,
                "response": final_answer,
                "tool_calls": tool_calls_emitted,
                "error": final_answer,
            }
            if hasattr(orch, "current_ledger") and orch.current_ledger is not None:
                payload_dict["execution_ledger"] = orch.current_ledger.to_dict()
            else:
                payload_dict["execution_ledger"] = TurnExecutionLedger().to_dict()
            sys.stdout.write(json.dumps(payload_dict) + "\n")
            sys.stdout.flush()
        return 1

    if is_truncated:
        sys.stderr.write("⚠️ [Model Truncated] Generation reached token length limit before completion.\n")
        sys.stderr.flush()
        if owns_cli_ledger and cli_ledger is not None:
            cli_ledger.close()
            cli_ledger = None
        if output_json:
            from zelyr.engine.execution_ledger import TurnExecutionLedger
            payload_dict = {
                "prompt": clean_prompt,
                "model": active_model,
                "response": final_answer,
                "tool_calls": tool_calls_emitted,
                "truncated": True,
            }
            if hasattr(orch, "current_ledger") and orch.current_ledger is not None:
                payload_dict["execution_ledger"] = orch.current_ledger.to_dict()
            else:
                payload_dict["execution_ledger"] = TurnExecutionLedger().to_dict()
            sys.stdout.write(json.dumps(payload_dict) + "\n")
            sys.stdout.flush()
        else:
            sys.stdout.write(final_answer.rstrip("\n") + "\n")
            sys.stdout.flush()
        return 1

    v_results = []
    if verify_spec is not None:
        if not verify_spec.strip():
            if owns_cli_ledger and cli_ledger is not None:
                cli_ledger.close()
                cli_ledger = None
            err_msg = "[verification failed] Empty verification specification provided.\n"
            sys.stdout.write(err_msg)
            sys.stderr.write(err_msg)
            return 2
        from zelyr.engine.execution_ledger import parse_acceptance_criteria
        from zelyr.engine.verifier import OutcomeVerifier

        parsed_specs = parse_acceptance_criteria(verify_spec)
        if parsed_specs:
            for item in parsed_specs:
                v_res = OutcomeVerifier.verify(
                    check=item.get("check", ""),
                    target=item.get("target", ""),
                    expected=item.get("expected"),
                )
                v_results.append((item.get("check", ""), item.get("target", ""), v_res))
        else:
            if owns_cli_ledger and cli_ledger is not None:
                cli_ledger.close()
                cli_ledger = None
            err_msg = f"[verification failed] Unable to parse verification specification: '{verify_spec}'\n"
            sys.stdout.write(err_msg)
            sys.stderr.write(err_msg)
            return 2

    ledger = getattr(orch, "current_ledger", None)
    has_unresolved_ledger = False
    unresolved_recs: List[Any] = []
    if ledger is not None:
        unresolved_fn = getattr(ledger, "unresolved_failures", None)
        if callable(unresolved_fn):
            unresolved_recs = list(unresolved_fn())
            has_unresolved_ledger = bool(unresolved_recs)
        else:
            has_fn = getattr(ledger, "has_unresolved_failures", None)
            has_unresolved_ledger = bool(has_fn()) if callable(has_fn) else False

    diagnostic_failures_only = False
    if has_unresolved_ledger and verify_spec is None and not reached_step_limit and unresolved_recs:
        from zelyr.engine.execution_ledger import (
            _extract_action_target,
            _is_inspection_action,
            detect_false_success,
        )
        global_prohib = extract_global_mutation_prohibition(clean_prompt)
        if (
            global_prohib.get("no_file_modifications")
            and global_prohib.get("is_test_diagnostic")
            and bool(final_answer.strip())
            and not detect_false_success(final_answer, has_unresolved=True)
        ):
            succ_read_names = set()
            for sr in (ledger.successful_actions() if hasattr(ledger, "successful_actions") else []):
                if _is_inspection_action(sr.tool_name, sr.arguments):
                    t_s = _extract_action_target(sr.tool_name, sr.arguments)
                    if t_s:
                        succ_read_names.add(Path(t_s).name)

            def _is_resolved_diag_rec(rec: Any) -> bool:
                if is_diagnostic_test_command(rec.tool_name, rec.arguments):
                    return True
                if _is_inspection_action(rec.tool_name, rec.arguments):
                    t_f = _extract_action_target(rec.tool_name, rec.arguments)
                    if t_f and Path(t_f).name in succ_read_names:
                        return True
                return False

            if all(_is_resolved_diag_rec(r) for r in unresolved_recs):
                diagnostic_failures_only = True
                has_unresolved_ledger = False

    if ledger is not None and getattr(ledger, "_central_ledger", None) is not None and getattr(ledger, "_execution_id", None):
        from zelyr.engine.execution_ledger import (
            ExecutionStatus,
            LedgerEventType,
            VerificationStatus,
        )

        c_ledger = ledger._central_ledger
        c_exec_id = ledger._execution_id
        if v_results:
            all_v_ok = all(res.success for _, _, res in v_results)
            for v_check, v_target, v_res in v_results:
                c_ledger.record_event(
                    c_exec_id,
                    LedgerEventType.VERIFICATION_RECORDED,
                    status=VerificationStatus.VERIFIED.value if v_res.success else VerificationStatus.VERIFICATION_FAILED.value,
                    metadata={"check": v_check, "target": v_target, "error": getattr(v_res, "error", None)},
                )
            if all_v_ok and not has_unresolved_ledger and not reached_step_limit:
                ledger.set_verification_status(VerificationStatus.VERIFIED)
                c_ledger.finalize_execution(
                    c_exec_id,
                    status=ExecutionStatus.COMPLETED,
                    verification_status=VerificationStatus.VERIFIED,
                    final_summary=final_answer,
                )
            else:
                ledger.set_verification_status(VerificationStatus.VERIFICATION_FAILED)
                c_ledger.finalize_execution(
                    c_exec_id,
                    status=ExecutionStatus.FAILED,
                    verification_status=VerificationStatus.VERIFICATION_FAILED,
                    error="Post-run outcome verification failed",
                    final_summary=final_answer,
                )

    if owns_cli_ledger and cli_ledger is not None:
        cli_ledger.close()
        cli_ledger = None

    if output_json:
        from zelyr.engine.execution_ledger import TurnExecutionLedger

        payload_dict = {
            "prompt": clean_prompt,
            "model": active_model,
            "response": final_answer,
            "tool_calls": tool_calls_emitted,
        }
        if reached_step_limit:
            payload_dict["step_limit_exceeded"] = True
        if diagnostic_failures_only:
            payload_dict["diagnostic_test_failed"] = True
        if v_results:
            all_success = all(res.success for _, _, res in v_results)
            if len(v_results) == 1:
                payload_dict["verification"] = v_results[0][2].to_dict()
            else:
                payload_dict["verification"] = {
                    "success": all_success,
                    "checks": [res.to_dict() for _, _, res in v_results],
                }
        if ledger is not None:
            payload_dict["execution_ledger"] = ledger.to_dict()
        else:
            payload_dict["execution_ledger"] = TurnExecutionLedger().to_dict()
        json_out = json.dumps(payload_dict)
        sys.stdout.write(json_out.rstrip("\n") + "\n")
        sys.stdout.flush()
        if reached_step_limit:
            return 2
        if v_results and not all(res.success for _, _, res in v_results):
            return 2
        if has_unresolved_ledger:
            return 2
    else:
        sys.stdout.write(final_answer.rstrip("\n") + "\n")
        sys.stdout.flush()
        any_failed = False
        if v_results:
            for v_check, v_target, v_res in v_results:
                if v_res.success:
                    sys.stdout.write(f"[verified] {v_check} on {v_target}: PASS\n")
                    sys.stdout.flush()
                else:
                    any_failed = True
                    err_msg = f"[verification failed] {v_check} on {v_target}: FAIL ({v_res.error})\n"
                    sys.stdout.write(err_msg)
                    sys.stdout.flush()
                    sys.stderr.write(err_msg)
                    sys.stderr.flush()
        total_rec = int(getattr(ledger, "total_recovery_attempts", 0) or 0) if ledger is not None else 0
        total_acts = int(ledger.total_actions()) if ledger is not None and hasattr(ledger, "total_actions") else len(tool_calls_emitted)
        if v_results:
            ver_status_str = "FAIL" if any_failed else "PASS"
        elif diagnostic_failures_only:
            ver_status_str = "DIAGNOSTIC_TEST_FAILED"
        elif ledger is not None and getattr(ledger, "verified_actions", None) and ledger.verified_actions():
            ver_status_str = "VERIFIED"
        else:
            ver_status_str = "NOT_RUN"
        if reached_step_limit:
            outcome_str = "step_limit_exceeded"
        elif any_failed or has_unresolved_ledger:
            outcome_str = "failed"
        else:
            outcome_str = "completed"
        sys.stderr.write(
            f"[verification] {ver_status_str} | [outcome] {outcome_str} "
            f"(steps={exec_step_counter}, actions={total_acts}, recoveries={total_rec})\n"
        )
        sys.stderr.flush()
        if reached_step_limit or any_failed or has_unresolved_ledger:
            return 2

    return 0


async def _execute_planned_headless_run(
    prompt: str,
    *,
    model_override: Optional[str] = None,
    output_json: bool = False,
    orchestrator: Optional[Any] = None,
    verify_spec: Optional[str] = None,
    base_url_override: Optional[str] = None,
    plan_only: bool = False,
    confirmed: Optional[bool] = None,
    max_concurrency: int = 1,
    db_path: Optional[str] = None,
) -> int:
    """Compile a complex objective into a TaskGraph, present review summary, and execute if permitted."""
    from zelyr.agents.native_orchestrator import NativeOrchestrator
    from zelyr.config import ZelyrConfig, get_config
    from zelyr.core.plugins import get_tool_schemas, load_plugins
    from zelyr.engine.execution_ledger import CentralExecutionLedger
    from zelyr.planning.compiler import PlanConfirmationRequiredError, PlanError
    from zelyr.tool_registry import ToolRegistry

    clean_prompt = (prompt or "").strip()
    if not clean_prompt:
        sys.stderr.write("Error: prompt cannot be empty.\n")
        sys.stderr.flush()
        return 1

    if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int) or max_concurrency < 1:
        sys.stderr.write(f"Error: --max-concurrency must be an integer >= 1 (got {max_concurrency!r}).\n")
        sys.stderr.flush()
        return 1

    load_plugins()
    base_cfg = getattr(orchestrator, "config", None) or get_config()
    active_model = model_override or getattr(base_cfg, "ollama_model", "qwen2.5-coder:7b")
    ledger, owns_ledger = (
        (CentralExecutionLedger(db_path=db_path), True)
        if db_path
        else _open_cli_ledger(None)
    )
    try:
        if orchestrator is not None:
            orch = orchestrator
            if getattr(orch, "central_ledger", None) is None:
                orch.central_ledger = ledger
        else:
            run_cfg = None
            if base_url_override:
                run_cfg = ZelyrConfig.from_dict(base_cfg.to_dict()) if hasattr(base_cfg, "to_dict") else ZelyrConfig()
                run_cfg.ollama_base_url = (
                    base_cfg.get_effective_ollama_base_url(base_url_override)
                    if hasattr(base_cfg, "get_effective_ollama_base_url")
                    else base_url_override.rstrip("/")
                )
            orch = NativeOrchestrator(
                config=run_cfg,
                tool_registry=ToolRegistry(),
                central_ledger=ledger,
                max_concurrency=max_concurrency,
            )
        schemas = getattr(orch, "tool_schemas", None) or get_tool_schemas(0)

        def _cli_confirm(bundle_obj: Any) -> bool:
            if not output_json:
                print(bundle_obj.summary_text)
            if not sys.stdin.isatty():
                return False
            try:
                from rich.prompt import Confirm

                return Confirm.ask("Execute this compiled plan?", default=False)
            except (EOFError, KeyboardInterrupt):
                return False

        eff_confirmed: Optional[bool] = confirmed
        eff_cb = None
        if not plan_only and eff_confirmed is None:
            if sys.stdin.isatty():
                eff_cb = _cli_confirm
            else:
                eff_confirmed = False

        try:
            res = await orch.run_planned_task(
                clean_prompt,
                central_ledger=ledger,
                tool_schemas=schemas,
                max_concurrency=max_concurrency,
                acceptance_criteria=verify_spec,
                confirmed=eff_confirmed,
                confirm_callback=eff_cb,
                plan_only=plan_only,
                model=active_model,
            )
        except PlanConfirmationRequiredError as exc:
            b_obj = getattr(exc, "bundle", None)
            if output_json:
                out_err: Dict[str, Any] = {
                    "status": "awaiting_confirmation",
                    "executed": False,
                    "error": str(exc),
                }
                if b_obj is not None:
                    out_err["execution_id"] = b_obj.task_graph.execution_id
                    out_err["graph_id"] = b_obj.task_graph.graph_id
                    out_err["task_graph"] = b_obj.task_graph.to_dict()
                    out_err["parallel_batches"] = b_obj.parallel_batches
                    out_err["summary_text"] = b_obj.summary_text
                print(json.dumps(out_err, indent=2))
            else:
                if b_obj is not None:
                    print(b_obj.summary_text)
                sys.stderr.write(f"{exc}\nRe-run with --confirm / --yes to execute.\n")
                sys.stderr.flush()
            return 2
        except PlanError as exc:
            sys.stderr.write(f"Plan compilation error: {exc}\n")
            sys.stderr.flush()
            return 1

        if output_json:
            bundle = res.get("bundle")
            out_payload = {
                "execution_id": res.get("execution_id"),
                "graph_id": res.get("graph_id"),
                "executed": bool(res.get("executed", False)),
                "status": res.get("graph_status") or res.get("status"),
                "task_graph": res["task_graph"].to_dict() if res.get("task_graph") is not None else None,
            }
            if bundle is not None:
                out_payload["parallel_batches"] = bundle.parallel_batches
                out_payload["independent_node_ids"] = bundle.independent_node_ids
                out_payload["summary_text"] = bundle.summary_text
            if res.get("executed"):
                out_payload["results"] = res.get("results", [])
            print(json.dumps(out_payload, indent=2))
        else:
            bundle = res.get("bundle")
            if bundle is not None:
                print(bundle.summary_text)
            if res.get("executed"):
                print(f"Execution '{res.get('execution_id')}' finished with graph status: {res.get('graph_status')}")

        if res.get("executed") and res.get("graph_status") in ("failed", "blocked", "cancelled"):
            return 2
        return 0
    finally:
        if owns_ledger:
            ledger.close()


def cmd_run(args: argparse.Namespace) -> None:
    """Execute a one-shot prompt headlessly via NativeOrchestrator and exit."""
    prompt: str = getattr(args, "prompt", "")
    model_override: Optional[str] = getattr(args, "model", None)
    output_json: bool = bool(getattr(args, "json", False))
    verify_spec: Optional[str] = getattr(args, "verify", None)
    base_url_override: Optional[str] = getattr(args, "base_url", None)

    if getattr(args, "plan", False) or getattr(args, "plan_only", False):
        eff_confirmed: Optional[bool] = True if getattr(args, "confirm", False) else None
        exit_code = asyncio.run(
            _execute_planned_headless_run(
                prompt=prompt,
                model_override=model_override,
                output_json=output_json,
                verify_spec=verify_spec,
                base_url_override=base_url_override,
                plan_only=bool(getattr(args, "plan_only", False)),
                confirmed=eff_confirmed,
                max_concurrency=int(getattr(args, "max_concurrency", 1) or 1),
                db_path=getattr(args, "db_path", None),
            )
        )
        if exit_code != 0:
            sys.exit(exit_code)
        return

    exit_code = asyncio.run(
        _execute_headless_run(
            prompt=prompt,
            model_override=model_override,
            output_json=output_json,
            verify_spec=verify_spec,
            base_url_override=base_url_override,
            max_steps=getattr(args, "max_steps", None),
            db_path=getattr(args, "db_path", None),
            verbose=bool(getattr(args, "verbose", False)),
        )
    )
    if exit_code != 0:
        sys.exit(exit_code)


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
    """Execute a tool by ID under unified execution policy enforcement."""
    from zelyr.config import get_config
    from zelyr.engine.execution_policy import (
        ExecutionPolicyConfig,
        ExecutionPolicyEngine,
        PolicyDecisionOutcome,
        get_default_confirmation_store,
    )
    from zelyr.tool_registry import ToolRegistry, resolve_tool_ring

    cfg = get_config()
    eff_mode = getattr(args, "policy_mode", None) or getattr(cfg, "execution_policy_mode", None) or "conservative"
    pol_cfg = ExecutionPolicyConfig(
        policy_mode=eff_mode,
        risk_defaults=dict(getattr(cfg, "execution_policy_risk_defaults", {}) or {}),
        tool_overrides=dict(getattr(cfg, "execution_policy_tool_overrides", {}) or {}),
        resource_restrictions=dict(getattr(cfg, "execution_policy_resource_restrictions", {}) or {}),
        confirmation_ttl_seconds=getattr(cfg, "execution_policy_confirmation_ttl_seconds", 300.0),
        max_confirmation_lifetime_seconds=getattr(cfg, "execution_policy_max_confirmation_lifetime_seconds", 900.0),
    )
    pol_engine = ExecutionPolicyEngine(config=pol_cfg, confirmation_store=get_default_confirmation_store())
    use_json = bool(getattr(args, "json", False))
    caller_ring = getattr(args, "caller_ring", None)
    confirm_flag = bool(getattr(args, "confirm", False))

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

    try:
        tool_args = json.loads(args.args) if args.args else {}
    except Exception as exc:
        if use_json:
            print(json.dumps({"success": False, "error": f"Invalid --args JSON: {exc}"}))
        else:
            _print(f"[red]✗[/red] Invalid --args JSON: {exc}")
        sys.exit(1)

    if tool is None:
        pol_dec = pol_engine.evaluate(
            args.tool_id,
            tool_args if isinstance(tool_args, dict) else {},
            is_registered=False,
            consume_confirmation=False,
            source_surface="cli.tool_run",
        )
        if use_json:
            print(json.dumps({"success": False, "error": f"Tool '{args.tool_id}' not found.", "policy_decision": pol_dec.to_safe_dict()}))
        else:
            _print(f"[red]✗[/red] Tool '{args.tool_id}' not found ({pol_dec.reason}).")
        sys.exit(2)

    req_ring = resolve_tool_ring(args.tool_id, tool=tool)
    if caller_ring is not None and req_ring is not None and int(caller_ring) > int(req_ring):
        msg = f"AuthorizationError: Tool '{args.tool_id}' requires caller_ring <= {req_ring}, got {caller_ring}."
        if use_json:
            print(json.dumps({"success": False, "error_type": "AUTHORIZATION_DENIED", "error": msg}))
        else:
            _print(f"[red]✗[/red] {msg}")
        sys.exit(2)

    pre_dec = pol_engine.evaluate(
        args.tool_id,
        tool_args,
        tool_obj=tool,
        required_ring=req_ring,
        is_registered=True,
        is_external=bool(getattr(tool, "is_external", False) or str(args.tool_id).startswith("ext__")),
        consume_confirmation=False,
        source_surface="cli.tool_run",
    )
    if pre_dec.outcome == PolicyDecisionOutcome.DENY:
        if use_json:
            print(json.dumps(pre_dec.to_tool_error_payload()))
        else:
            _print(f"[red]✗ Policy denied ({pre_dec.risk_level.value}):[/red] {pre_dec.reason}")
        sys.exit(2)

    conf_token: Optional[str] = None
    conf_id: Optional[str] = None
    if pre_dec.outcome == PolicyDecisionOutcome.REQUIRE_CONFIRMATION:
        approved = False
        if confirm_flag:
            approved = True
        elif sys.stdin.isatty():
            _print(
                f"[yellow]⚠ Confirmation required[/yellow] tool={pre_dec.tool_id} risk={pre_dec.risk_level.value} "
                f"digest={pre_dec.arg_digest[:12]} scope={pre_dec.resource_scope}"
            )
            try:
                ans = input("Proceed with this exact action? [y/N]: ").strip().lower()
                approved = ans in ("y", "yes")
            except (EOFError, KeyboardInterrupt):
                approved = False
        if not approved:
            if use_json:
                print(json.dumps(pre_dec.to_tool_error_payload()))
            else:
                _print(
                    f"[yellow]⏸ Confirmation required ({pre_dec.risk_level.value}):[/yellow] {pre_dec.reason} "
                    "(pass --confirm to approve this exact normalized invocation)."
                )
            sys.exit(2)
        conf_token, conf_rec = pol_engine.confirmation_store.issue_confirmation(
            tool_id=pre_dec.tool_id,
            arg_digest=pre_dec.arg_digest,
            resource_scope=pre_dec.resource_scope,
            ttl_seconds=pol_cfg.confirmation_ttl_seconds,
            max_lifetime_seconds=pol_cfg.max_confirmation_lifetime_seconds,
            policy_version=pol_cfg.policy_version,
            approved_by="cli_tool_run",
        )
        conf_id = conf_rec.confirmation_id

    async def _run() -> int:
        if not use_json:
            _print(f"[bold]Running[/bold] {args.tool_id} with args {tool_args} (risk={pre_dec.risk_level.value})…")
        try:
            result = await registry.execute_async(
                args.tool_id,
                caller_ring=caller_ring if caller_ring is not None else req_ring,
                allowed_tools=[args.tool_id],
                confirmation_token=conf_token,
                confirmation_id=conf_id,
                policy_engine=pol_engine,
                **tool_args,
            )
            if use_json:
                out_d = result.to_dict(tool=args.tool_id, arguments=tool_args) if hasattr(result, "to_dict") else {"success": bool(result), "output": result}
                print(json.dumps(out_d, default=str))
            else:
                _print(f"[green]Result:[/green] {result}")
            return 0 if getattr(result, "success", True) else 2
        except Exception as exc:
            if use_json:
                print(json.dumps({"success": False, "error": str(exc)}))
            else:
                _print(f"[red]Execution failed:[/red] {exc}")
            return 1

    rc = asyncio.run(_run())
    if rc != 0:
        sys.exit(rc)


def cmd_status(args: argparse.Namespace) -> None:
    """Show system + daemon telemetry and optional extras status."""
    if getattr(args, "manifest", False) or getattr(args, "json", False):
        cmd_env(args)
        return

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


def cmd_env(args: argparse.Namespace) -> None:
    """Inspect the structured runtime environment manifest via the CLI."""
    from zelyr.engine.environment import RuntimeEnvironmentManifest
    from zelyr.tool_registry import ToolRegistry

    caller_ring = getattr(args, "caller_ring", None)
    allowed_raw = getattr(args, "allowed_tools", None)
    allowed_tools = (
        [item.strip() for item in allowed_raw.split(",") if item.strip()]
        if isinstance(allowed_raw, str) and allowed_raw.strip()
        else None
    )
    probe_services = not getattr(args, "no_probe", False)
    probe_timeout = float(getattr(args, "timeout", 0.5) or 0.5)
    refresh = bool(getattr(args, "refresh", False))

    registry = ToolRegistry()
    for builtin_id in registry.get_builtin_tools():
        registry.get_tool(builtin_id)

    if getattr(args, "compact", False):
        block = RuntimeEnvironmentManifest.build_manifest_block(
            tool_registry=registry,
            caller_ring=caller_ring,
            allowed_tools=allowed_tools,
        )
        print(block)
        return

    manifest = RuntimeEnvironmentManifest.collect_manifest(
        tool_registry=registry,
        caller_ring=caller_ring,
        allowed_tools=allowed_tools,
        probe_services=probe_services,
        probe_timeout=probe_timeout,
        refresh=refresh,
    )
    compat_report = None
    if getattr(args, "model_check", False):
        from zelyr.engine.model_compatibility import (
            check_model_compatibility,
            format_model_compatibility_report,
        )

        compat_report = check_model_compatibility(
            probe=probe_services,
            timeout=probe_timeout,
        )
        if isinstance(manifest, dict):
            manifest = dict(manifest)
            manifest["model_compatibility"] = compat_report.to_dict()

    if getattr(args, "json", False):
        print(json.dumps(manifest, indent=2))
        return

    report = RuntimeEnvironmentManifest.format_diagnostic_report(manifest)
    print(report)
    if compat_report is not None:
        print()
        print(format_model_compatibility_report(compat_report))


def _open_cli_ledger(db_path: Optional[str]) -> tuple[Any, bool]:
    """Open ledger instance lazily for CLI diagnostics commands."""
    from zelyr.engine.execution_ledger import CentralExecutionLedger, get_default_central_ledger

    if db_path:
        return CentralExecutionLedger(db_path=db_path, raise_on_persistence_error=False), True
    return get_default_central_ledger(), False


def cmd_executions(args: argparse.Namespace) -> None:
    """List recent persistent execution ledger records (or interrupted/resumable tasks)."""
    import datetime

    ledger, owns_conn = _open_cli_ledger(getattr(args, "db_path", None))
    try:
        only_resumable = bool(getattr(args, "resumable", False))
        if only_resumable:
            from zelyr.engine.checkpoint import CheckpointManager

            records = ledger.list_resumable_executions(limit=int(getattr(args, "limit", 20) or 20))
            assessed = []
            for r in records:
                a = CheckpointManager.assess_resume(ledger, r.execution_id, verify_outcomes=False)
                assessed.append((r, a))
            if getattr(args, "json", False):
                payload = {
                    "count": len(assessed),
                    "db_path": str(ledger.db_path),
                    "persistence_errors": ledger.get_persistence_errors(),
                    "resumable_tasks": [
                        {
                            **r.to_dict(),
                            "checkpoint_id": a.checkpoint_id,
                            "checkpoint_version": a.checkpoint_version,
                            "resumability_status": a.resumability_status,
                            "can_resume": a.can_resume,
                            "requires_explicit_resume": a.requires_explicit_resume,
                            "pending_proposal_count": a.to_dict()["pending_proposal_count"],
                            "pending_actions": [p.to_dict() for p in a.pending_actions],
                            "blocking_reasons": a.blocking_reasons,
                            "warnings": a.warnings,
                        }
                        for r, a in assessed
                    ],
                }
                print(json.dumps(payload, indent=2))
                return
            if not assessed:
                print("No interrupted or resumable tasks found.")
                return
            print(f"Interrupted / Resumable Tasks ({ledger.db_path}) — {len(assessed)} record(s):")
            for r, a in assessed:
                ts_str = datetime.datetime.fromtimestamp(r.updated_at).strftime("%Y-%m-%d %H:%M:%S")
                reason = (a.blocking_reasons[0] if a.blocking_reasons else (a.warnings[0] if a.warnings else "ready"))[:70]
                prompt_snip = (r.task_prompt or "").replace("\n", " ")[:48]
                pending_cnt = a.to_dict()["pending_proposal_count"]
                print(
                    f"  {r.execution_id}  status={r.status}  resume={a.resumability_status}  "
                    f"pending={pending_cnt}  cp={a.checkpoint_id or 'none'}  at={ts_str}  note={reason}  {prompt_snip}"
                )
            return

        records = ledger.list_executions(
            limit=int(getattr(args, "limit", 20) or 20),
            status=getattr(args, "status", None),
            tool_id=getattr(args, "tool", None),
            event_type=getattr(args, "event_type", None),
        )
        if getattr(args, "json", False):
            payload = {
                "count": len(records),
                "db_path": str(ledger.db_path),
                "persistence_errors": ledger.get_persistence_errors(),
                "executions": [r.to_dict() for r in records],
            }
            print(json.dumps(payload, indent=2))
            return

        if not records:
            print("No execution records found.")
            return

        print(f"Execution Ledger ({ledger.db_path}) — {len(records)} record(s):")
        for rec in records:
            ts_str = datetime.datetime.fromtimestamp(rec.created_at).strftime("%Y-%m-%d %H:%M:%S")
            prompt_snip = (rec.task_prompt or "").replace("\n", " ")[:60]
            print(
                f"  {rec.execution_id}  "
                f"status={rec.status}  "
                f"verify={rec.verification_status}  "
                f"events={rec.event_count}  "
                f"src={rec.source_component}  "
                f"at={ts_str}  "
                f"{prompt_snip}"
            )
    finally:
        if owns_conn:
            ledger.close()


def _resolve_policy_config_for_cli(args: argparse.Namespace) -> Any:
    """Resolve ExecutionPolicyConfig lazily from --config-path or default ZelyrConfig."""
    from pathlib import Path
    from zelyr.config import ZelyrConfig, get_config
    from zelyr.engine.execution_policy import ExecutionPolicyConfig

    cfg_path_arg = getattr(args, "config_path", None)
    if cfg_path_arg:
        cfg_path = Path(cfg_path_arg).expanduser()
        if not cfg_path.exists():
            print(f"Error: Config file not found: {cfg_path}", file=sys.stderr)
            sys.exit(1)
        try:
            z_cfg = ZelyrConfig.load(cfg_path)
        except Exception as exc:
            print(f"Error: Failed to load config file '{cfg_path}': {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        z_cfg = get_config()

    pol_cfg = ExecutionPolicyConfig.from_zelyr_config(z_cfg)
    mode_override = getattr(args, "policy_mode", None)
    if mode_override:
        sources = dict(pol_cfg.config_sources)
        sources["policy_mode"] = "explicit:cli"
        pol_cfg = ExecutionPolicyConfig(
            policy_version=pol_cfg.policy_version,
            policy_mode=str(mode_override),
            risk_defaults=pol_cfg.risk_defaults,
            tool_overrides=pol_cfg.tool_overrides,
            resource_restrictions=pol_cfg.resource_restrictions,
            confirmation_ttl_seconds=pol_cfg.confirmation_ttl_seconds,
            max_confirmation_lifetime_seconds=pol_cfg.max_confirmation_lifetime_seconds,
            governor_strict_mode=pol_cfg.governor_strict_mode,
            config_sources=sources,
            _config_file_path=pol_cfg._config_file_path,
            _auth_mode_str=pol_cfg._auth_mode_str,
        )
    return pol_cfg


def cmd_policy(args: argparse.Namespace) -> None:
    """Inspect active execution policy, preview tool decisions, or query ledger policy history."""
    subcmd = getattr(args, "policy_cmd", None)
    if subcmd == "show":
        from zelyr.engine.execution_policy import (
            format_policy_configuration,
            inspect_policy_configuration,
        )

        pol_cfg = _resolve_policy_config_for_cli(args)
        view = inspect_policy_configuration(policy_config=pol_cfg)
        if getattr(args, "json", False):
            print(json.dumps(view, indent=2))
            return
        print(format_policy_configuration(view))
        return

    if subcmd == "explain":
        from zelyr.engine.execution_policy import (
            format_policy_explanation,
            preview_tool_policy,
        )

        pol_cfg = _resolve_policy_config_for_cli(args)
        allowed_raw = getattr(args, "allowed_tools", None)
        allowed_tools = (
            [t.strip() for t in str(allowed_raw).split(",") if t.strip()]
            if allowed_raw is not None
            else None
        )
        raw_args = getattr(args, "args", None)
        eff_args: Any = {} if raw_args is None else raw_args
        preview_res = preview_tool_policy(
            args.tool_id,
            eff_args,
            caller_ring=getattr(args, "caller_ring", None),
            allowed_tools=allowed_tools,
            workspace_root=getattr(args, "workspace_root", None),
            policy_config=pol_cfg,
            source_surface="cli_policy_explain",
        )
        if getattr(args, "json", False):
            print(json.dumps(preview_res.to_safe_dict(), indent=2))
            return
        print(format_policy_explanation(preview_res))
        return

    if subcmd == "history":
        import datetime
        from zelyr.engine.execution_policy import query_policy_history

        ledger, owns_conn = _open_cli_ledger(getattr(args, "db_path", None))
        try:
            hist = query_policy_history(
                ledger,
                limit=int(getattr(args, "limit", 20) or 20),
                outcome=getattr(args, "outcome", None),
                tool_id=getattr(args, "tool", None),
                execution_id=getattr(args, "execution_id", None),
            )
            if getattr(args, "json", False):
                print(json.dumps(hist, indent=2))
                if hist.get("persistence_errors"):
                    sys.exit(1)
                return

            if hist.get("persistence_errors"):
                for perr in hist["persistence_errors"]:
                    print(
                        f"Error: Ledger persistence failure ({perr.get('operation', 'query')}): {perr.get('error', 'unknown error')}",
                        file=sys.stderr,
                    )
                sys.exit(1)

            entries = hist.get("entries") or []
            if not entries:
                print("No policy authorization decisions recorded.")
                return

            print(f"Policy Authorization History ({hist.get('db_path')}) — {len(entries)} decision(s):")
            for entry in entries:
                ts_val = float(entry.get("timestamp") or 0.0)
                ts_str = datetime.datetime.fromtimestamp(ts_val).strftime("%Y-%m-%d %H:%M:%S")
                tags: List[str] = []
                if entry.get("legacy_metadata"):
                    tags.append("legacy")
                if entry.get("malformed_metadata"):
                    tags.append("malformed")
                tag_str = f" [{' '.join(tags)}]" if tags else ""
                print(
                    f"  {ts_str}  outcome={entry.get('outcome')}  tool={entry.get('tool_id')}  "
                    f"risk={entry.get('risk_level')}  stage={entry.get('precedence_stage')}  "
                    f"reason_code={entry.get('reason_code')}  exec={entry.get('execution_id')}  "
                    f"scope={entry.get('resource_scope')}{tag_str}"
                )
            return
        finally:
            if owns_conn:
                ledger.close()


def cmd_execution(args: argparse.Namespace) -> None:
    """Inspect a single execution record, ordered timeline, and optional checkpoint details."""
    import datetime

    exec_query = getattr(args, "execution_id", "")
    ledger, owns_conn = _open_cli_ledger(getattr(args, "db_path", None))
    try:
        record, matches = ledger.find_execution_by_prefix(exec_query)
        if record is None:
            if len(matches) > 1:
                sys.stderr.write(
                    f"Ambiguous execution ID prefix '{exec_query}': matched {', '.join(matches[:5])}\n"
                )
            else:
                sys.stderr.write(f"Execution '{exec_query}' not found.\n")
            sys.stderr.flush()
            sys.exit(1)

        if getattr(args, "graph", False):
            from zelyr.engine.checkpoint import CheckpointManager
            from zelyr.tool_registry import ToolRegistry

            assess_g = CheckpointManager.assess_resume(
                ledger,
                record.execution_id,
                tool_registry=ToolRegistry(),
                verify_outcomes=True,
            )
            if getattr(args, "json", False):
                from zelyr.engine.recovery import build_graph_recovery_overview

                print(
                    json.dumps(
                        {
                            "execution_id": record.execution_id,
                            "resumability_status": assess_g.resumability_status,
                            "can_resume": assess_g.can_resume,
                            "task_graph": assess_g.task_graph.to_dict() if assess_g.task_graph is not None else None,
                            "recovery_overview": (
                                build_graph_recovery_overview(assess_g.task_graph)
                                if assess_g.task_graph is not None
                                else None
                            ),
                        },
                        indent=2,
                    )
                )
            else:
                _print_task_graph_view(assess_g, empty_notice=f"No task dependency graph recorded for execution '{record.execution_id}'.")
            return

        if getattr(args, "diagnose", False):
            _print_execution_diagnostics(
                ledger,
                record.execution_id,
                verify=bool(getattr(args, "verify", False)),
                output_json=bool(getattr(args, "json", False)),
            )
            return

        if getattr(args, "metrics", False):
            from zelyr.engine.observability import collect_execution_metrics, format_observability_summary

            metrics = collect_execution_metrics(ledger, record.execution_id)
            if getattr(args, "json", False):
                print(json.dumps(metrics, indent=2))
                if metrics.get("persistence_errors"):
                    sys.exit(1)
                return
            print(format_observability_summary(metrics))
            if metrics.get("persistence_errors"):
                sys.exit(1)
            return

        if getattr(args, "checkpoint", False):
            _print_checkpoint_assessment(ledger, record.execution_id, output_json=bool(getattr(args, "json", False)))
            return

        summary = ledger.get_execution_summary(
            record.execution_id,
            event_type=getattr(args, "event_type", None),
            tool_id=getattr(args, "tool", None),
        )
        if summary is None:
            sys.stderr.write(f"Execution '{record.execution_id}' not found.\n")
            sys.stderr.flush()
            sys.exit(1)

        if getattr(args, "json", False):
            try:
                from zelyr.engine.checkpoint import CheckpointManager

                assess_sum = CheckpointManager.assess_resume(ledger, record.execution_id, verify_outcomes=False)
                if assess_sum.task_graph is not None:
                    summary["task_graph"] = assess_sum.task_graph.to_dict()
            except Exception:
                pass
            print(json.dumps(summary, indent=2))
            return

        rec_d = summary["execution"]
        created_str = datetime.datetime.fromtimestamp(rec_d["created_at"]).strftime("%Y-%m-%d %H:%M:%S")
        print(f"Execution:           {rec_d['execution_id']}")
        print(f"Status:              {rec_d['status']}")
        print(f"Verification Status: {rec_d['verification_status']}")
        print(f"Source Component:    {rec_d['source_component']}")
        print(f"Created At:          {created_str}")
        if rec_d.get("caller_ring") is not None:
            print(f"Caller Ring:         {rec_d['caller_ring']}")
        if rec_d.get("task_local_scoped"):
            print("Task Local Scoped:   true")
        if rec_d.get("task_prompt"):
            print(f"Task Prompt:         {rec_d['task_prompt']}")
        if rec_d.get("final_summary"):
            print(f"Final Summary:       {rec_d['final_summary']}")
        if rec_d.get("error"):
            print(f"Error:               {rec_d['error']}")
        if summary["tools_used"]:
            print(f"Tools Used:          {', '.join(summary['tools_used'])}")
        if summary["external_tools"]:
            ext_desc = ", ".join(
                f"{e['tool_id']} ({e.get('source_type')}:{e.get('source_id')})"
                for e in summary["external_tools"]
            )
            print(f"External Tools:      {ext_desc}")
        if summary["persistence_errors"]:
            print(f"Persistence Errors:  {len(summary['persistence_errors'])}")
        events = summary["events"]
        print(f"Events ({len(events)}):")
        for ev in events:
            t_str = datetime.datetime.fromtimestamp(ev["timestamp"]).strftime("%H:%M:%S.%f")[:-3]
            ext_tag = f" [ext:{ev['source_id']}]" if ev.get("is_external") else ""
            tool_part = f" tool={ev['tool_id']}" if ev.get("tool_id") else ""
            flags = []
            if ev.get("redacted"):
                flags.append("redacted")
            if ev.get("truncated"):
                flags.append("truncated")
            flag_str = f" ({','.join(flags)})" if flags else ""
            print(
                f"  #{ev['sequence_number']:03d} [{t_str}] {ev['event_type']} "
                f"status={ev['status']}{tool_part}{ext_tag}{flag_str}"
            )
    finally:
        if owns_conn:
            ledger.close()


def _print_execution_diagnostics(
    ledger: Any,
    execution_id: str,
    *,
    verify: bool = False,
    output_json: bool = False,
) -> None:
    """Render read-only end-to-end execution lifecycle diagnostics in JSON or text form."""
    from zelyr.engine.diagnostics import (
        diagnose_execution_lifecycle,
        format_execution_lifecycle_report,
    )

    tool_reg = None
    if verify:
        from zelyr.tool_registry import ToolRegistry

        tool_reg = ToolRegistry()

    report = diagnose_execution_lifecycle(
        ledger,
        execution_id,
        verify_outcomes=verify,
        tool_registry=tool_reg,
    )
    if output_json:
        print(json.dumps(report.to_dict(), indent=2))
        if report.persistence_errors or not report.found:
            sys.exit(1)
        return
    print(format_execution_lifecycle_report(report))
    if report.persistence_errors or not report.found:
        sys.exit(1)


def cmd_browser(args: argparse.Namespace) -> None:
    """Install or inspect Chromium Native Messaging host and browser bridge status."""
    from pathlib import Path
    from zelyr.tools.browser_native_host import (
        get_native_host_status,
        install_native_messaging_host,
    )

    sub = getattr(args, "browser_cmd", None)
    user_data_dir = Path(args.user_data_dir) if getattr(args, "user_data_dir", None) else None
    output_json = bool(getattr(args, "json", False))

    if sub == "install-host":
        allowed_hosts = None
        if getattr(args, "allowed_hosts", None):
            allowed_hosts = [h.strip() for h in str(args.allowed_hosts).split(",") if h.strip()]
        res = install_native_messaging_host(
            extension_id=getattr(args, "extension_id", None),
            user_data_dir=user_data_dir,
            allowed_hosts=allowed_hosts,
        )
        if output_json:
            print(json.dumps(res, indent=2))
        else:
            print(f"Installed Native Messaging host '{res['host_name']}':")
            print(f"  Extension Dir:   {res['extension_dir']}")
            print(f"  Extension ID:    {res['extension_id']}")
            print(f"  Allowed Origins: {', '.join(res['allowed_origins'])}")
            print(f"  Manifest Path:   {res['manifest_path']}")
            print(f"  Launcher Path:   {res['launcher_path']}")
        return

    status = get_native_host_status(user_data_dir=user_data_dir)
    if output_json:
        print(json.dumps(status, indent=2))
        return
    print(f"Browser Native Host Status ({status['host_name']}):")
    print(f"  Extension Dir:      {status['extension_dir']} (exists={status['extension_exists']})")
    print(f"  Unpacked Ext ID:    {status['extension_id']}")
    print(f"  Manifest Path:      {status['manifest_path']} (installed={status['manifest_installed']})")
    print(f"  Allowed Origins:    {status['allowed_origins'] or '(none)'}")
    print(f"  Native IPC Socket:  {status['socket_path']} (active={status['socket_present']})")


def cmd_diagnose(args: argparse.Namespace) -> None:
    """Run read-only end-to-end execution lifecycle diagnostics for an execution ID or prefix."""
    exec_query = getattr(args, "execution_id", "")
    ledger, owns_conn = _open_cli_ledger(getattr(args, "db_path", None))
    try:
        record, matches = ledger.find_execution_by_prefix(exec_query)
        if record is None:
            perrs = ledger.get_persistence_errors() if hasattr(ledger, "get_persistence_errors") else []
            msg = (
                f"Ambiguous execution ID prefix '{exec_query}': matched {', '.join(matches[:5])}"
                if len(matches) > 1
                else f"Execution '{exec_query}' not found."
            )
            if perrs:
                msg = f"{msg} (persistence error: {perrs[-1].get('error', 'database error')})"
            if getattr(args, "json", False):
                print(json.dumps({"found": False, "execution_id": exec_query, "error": msg, "persistence_errors": perrs}, indent=2))
            else:
                sys.stderr.write(msg + "\n")
                sys.stderr.flush()
            sys.exit(1)

        _print_execution_diagnostics(
            ledger,
            record.execution_id,
            verify=bool(getattr(args, "verify", False)),
            output_json=bool(getattr(args, "json", False)),
        )
    finally:
        if owns_conn:
            ledger.close()


def cmd_metrics(args: argparse.Namespace) -> None:
    """Summarize measured execution performance & policy observability metrics."""
    from zelyr.engine.observability import (
        collect_execution_metrics,
        collect_fleet_observability,
        format_observability_summary,
    )

    exec_query = getattr(args, "execution_id", None)
    ledger, owns_conn = _open_cli_ledger(getattr(args, "db_path", None))
    try:
        if exec_query:
            record, matches = ledger.find_execution_by_prefix(exec_query)
            if record is None:
                perrs = ledger.get_persistence_errors() if hasattr(ledger, "get_persistence_errors") else []
                msg = (
                    f"Ambiguous execution ID prefix '{exec_query}': matched {', '.join(matches[:5])}"
                    if len(matches) > 1
                    else f"Execution '{exec_query}' not found."
                )
                if getattr(args, "json", False):
                    print(json.dumps({"found": False, "execution_id": exec_query, "error": msg, "persistence_errors": perrs}, indent=2))
                else:
                    sys.stderr.write(msg + "\n")
                    sys.stderr.flush()
                sys.exit(1)
            summary = collect_execution_metrics(ledger, record.execution_id)
        else:
            summary = collect_fleet_observability(ledger, limit=int(getattr(args, "limit", 20) or 20))

        if getattr(args, "json", False):
            print(json.dumps(summary, indent=2))
            if summary.get("persistence_errors"):
                sys.exit(1)
            return
        print(format_observability_summary(summary))
        if summary.get("persistence_errors"):
            for perr in summary["persistence_errors"]:
                sys.stderr.write(f"Persistence error: {perr.get('error', 'database error')}\n")
            sys.stderr.flush()
            sys.exit(1)
    finally:
        if owns_conn:
            ledger.close()


def cmd_config(args: argparse.Namespace) -> None:
    """Inspect and validate effective Zelyr configuration, sources, unknown keys, and ranges."""
    from zelyr.engine.config_diagnostics import (
        diagnose_configuration,
        format_configuration_diagnostics,
    )

    report = diagnose_configuration(config_path=getattr(args, "config_path", None))
    subcmd = getattr(args, "config_cmd", None) or "show"
    strict = bool(getattr(args, "strict", False)) or (subcmd == "validate")

    if getattr(args, "json", False):
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_configuration_diagnostics(report))

    if report.parse_error or not report.valid:
        sys.exit(1)
    if strict and (report.validation_issues or report.unknown_keys):
        sys.exit(1)


def cmd_model(args: argparse.Namespace) -> None:
    """Assess model and provider compatibility for a Zelyr workflow."""
    from zelyr.engine.environment import CapabilityStatus
    from zelyr.engine.model_compatibility import (
        check_model_compatibility,
        format_model_compatibility_report,
    )

    try:
        report = check_model_compatibility(
            model=getattr(args, "model", None),
            base_url=getattr(args, "base_url", None),
            workflow=str(getattr(args, "workflow", None) or "planned_graph"),
            probe=not bool(getattr(args, "no_probe", False)),
            timeout=float(getattr(args, "timeout", 2.0) or 2.0),
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if getattr(args, "json", False):
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_model_compatibility_report(report))

    if report.overall_status == CapabilityStatus.UNAVAILABLE.value or report.workflow_compatible is False:
        sys.exit(1)
    if getattr(args, "strict", False) and (
        report.overall_status != CapabilityStatus.AVAILABLE.value or report.workflow_compatible is not True
    ):
        sys.exit(1)


def _print_task_graph_view(assessment: Any, *, empty_notice: Optional[str] = None) -> None:
    """Render readable dependency-aware task graph summary, nodes, edges, and blockers."""
    tg = getattr(assessment, "task_graph", None)
    if tg is None:
        if empty_notice:
            print(empty_notice)
        return
    g_sum = tg.summary or {}
    next_node = tg.get_next_eligible_node() if hasattr(tg, "get_next_eligible_node") else None
    max_conc = getattr(tg, "max_concurrency", 1)
    running_nodes = (
        tg.get_running_nodes()
        if hasattr(tg, "get_running_nodes")
        else [n for n in tg.nodes if n.execution_status == "running"]
    )
    ready_nodes = (
        tg.get_ready_nodes()
        if hasattr(tg, "get_ready_nodes")
        else [n for n in tg.nodes if n.execution_status == "ready" and getattr(n, "reserved_by", None) is None]
    )
    print(f"Task Graph:          {tg.graph_id} (v{tg.graph_version}, status={tg.status})")
    print(f"Max Concurrency:     {max_conc}")
    if tg.objective:
        print(f"Graph Objective:     {tg.objective}")
    print(
        f"Graph Summary:       verified={g_sum.get('completed_verified_count', 0)}/{g_sum.get('total_nodes', len(tg.nodes))} "
        f"unverified={g_sum.get('completed_unverified_count', 0)} "
        f"ready={g_sum.get('ready_count', 0)} pending={g_sum.get('pending_count', 0)} "
        f"running={g_sum.get('running_count', 0)} failed={g_sum.get('failed_count', 0)} "
        f"blocked={g_sum.get('blocked_count', 0)} cancelled={g_sum.get('cancelled_count', 0)}"
    )
    running_desc = (
        ", ".join(f"{n.node_id}(owner={getattr(n, 'reserved_by', None) or 'unowned'})" for n in running_nodes)
        if running_nodes
        else "none"
    )
    ready_desc = ", ".join(n.node_id for n in ready_nodes) if ready_nodes else "none"
    print(f"Running Nodes:       {running_desc}")
    print(f"Ready Nodes:         {ready_desc}")
    print(f"Next Eligible Node:  {next_node.node_id if next_node is not None else 'none'}")
    try:
        from zelyr.planning.compiler import _compute_parallel_batches

        waves = _compute_parallel_batches(tg)
        if waves:
            wave_parts = [
                f"W{i + 1}[{','.join(w)}]{'(parallel)' if len(w) > 1 else ''}"
                for i, w in enumerate(waves)
            ]
            print(f"Execution Waves:     {' -> '.join(wave_parts)}")
    except Exception:
        pass
    active_res = getattr(tg, "active_reservations", None) or {}
    if active_res:
        print(f"Active Reservations ({len(active_res)}):")
        for rnid, rinfo in sorted(active_res.items()):
            r_owner = rinfo.get("owner_id", "-") if isinstance(rinfo, dict) else "-"
            r_id = rinfo.get("reservation_id", "-") if isinstance(rinfo, dict) else "-"
            r_locks = ",".join(rinfo.get("resources") or []) if isinstance(rinfo, dict) else ""
            lock_s = f" locks=[{r_locks}]" if r_locks else ""
            print(f"  - {rnid}: owner={r_owner} reservation={r_id}{lock_s}")
    if tg.nodes:
        print(f"Nodes ({len(tg.nodes)}):")
        for node in tg.nodes:
            deps = ",".join(node.dependency_ids) if node.dependency_ids else "-"
            props = ",".join(node.proposal_ids) if node.proposal_ids else "-"
            owner_s = f" owner={node.reserved_by}" if getattr(node, "reserved_by", None) else ""
            c_res = getattr(node, "conflict_resources", None) or []
            locks_s = f" locks=[{','.join(c_res)}]" if c_res else ""
            ver_req_s = f" verify_req={node.verification_requirement}" if getattr(node, "verification_requirement", None) else ""
            exp_out_s = f" outcome={node.expected_outcome!r}" if getattr(node, "expected_outcome", None) else ""
            rs = getattr(node, "recovery_state", None) or {}
            rec_s = ""
            if isinstance(rs, dict) and rs:
                att_c = int(rs.get("attempt_count") or 0)
                max_a = rs.get("max_attempts")
                att_str = f"{att_c}/{max_a}" if max_a is not None else str(att_c)
                rec_st = rs.get("recovery_status") or "-"
                rec_dec = rs.get("recovery_decision") or "-"
                nxt_act = rs.get("next_recovery_action") or "-"
                rec_s = f" rec_status={rec_st} decision={rec_dec} attempts={att_str} next_action={nxt_act}"
            blk = f" reason={node.blocked_reason}" if node.blocked_reason else ""
            fail = f" err={(node.failure_info or {}).get('error')}" if node.failure_info and not blk else ""
            print(
                f"  - [{node.order_index}] {node.node_id} exec={node.execution_status} "
                f"verify={node.verification_status}{owner_s}{locks_s} proposals=[{props}] deps=[{deps}] "
                f"desc={node.description!r}{exp_out_s}{ver_req_s}{rec_s}{blk}{fail}"
            )
    from zelyr.engine.recovery import build_graph_recovery_overview

    rec_ov = build_graph_recovery_overview(tg)
    blocked_nodes = tg.get_blocked_nodes() if hasattr(tg, "get_blocked_nodes") else []
    if blocked_nodes:
        print("Blocked Nodes:")
        for bn in blocked_nodes:
            deps_str = ",".join(bn.dependency_ids) if bn.dependency_ids else "-"
            print(f"  - {bn.node_id} (deps=[{deps_str}]): {bn.blocked_reason or 'blocked'}")
    failed_nodes = tg.get_failed_nodes() if hasattr(tg, "get_failed_nodes") else []
    if failed_nodes:
        print("Failed Prerequisites:")
        for fn in failed_nodes:
            rs = getattr(fn, "recovery_state", None) or {}
            err_s = (fn.failure_info or {}).get("error") or fn.blocked_reason or "failed"
            att_c = int(rs.get("attempt_count") or 0) if isinstance(rs, dict) else 0
            rec_st = rs.get("recovery_status") if isinstance(rs, dict) and rs.get("recovery_status") else "unassessed"
            rec_dec = rs.get("recovery_decision") if isinstance(rs, dict) and rs.get("recovery_decision") else "-"
            nxt_act = rs.get("next_recovery_action") if isinstance(rs, dict) and rs.get("next_recovery_action") else "-"
            print(
                f"  - {fn.node_id}: {err_s} (recoverable={fn.recoverable}, recovery_status={rec_st}, "
                f"decision={rec_dec}, attempts={att_c}, next_action={nxt_act})"
            )
    if rec_ov.get("failed_nodes") or rec_ov.get("recovering_nodes") or rec_ov.get("blocked_dependents"):
        print(
            f"Recovery Status:     total_attempts={rec_ov.get('total_attempts', 0)} "
            f"failed={len(rec_ov.get('failed_nodes', []))} "
            f"recovering={len(rec_ov.get('recovering_nodes', []))} "
            f"blocked_dependents={len(rec_ov.get('blocked_dependents', []))}"
        )
        for rn in rec_ov.get("recovering_nodes", []):
            print(
                f"  - {rn['node_id']}: status={rn.get('recovery_status') or '-'} "
                f"decision={rn.get('recovery_decision') or '-'} "
                f"attempts={rn.get('attempt_count', 0)}"
                f"{'/' + str(rn['max_attempts']) if rn.get('max_attempts') is not None else ''} "
                f"next_action={rn.get('next_recovery_action') or '-'}"
            )
        if rec_ov.get("blocked_dependents"):
            print("Blocked Dependents:")
            for bd in rec_ov["blocked_dependents"]:
                b_pre = ",".join(bd.get("blocking_prerequisites") or []) or "-"
                print(f"  - {bd['node_id']} (blocked_by=[{b_pre}]): {bd.get('reason') or 'blocked'}")


def _print_checkpoint_assessment(
    ledger: Any,
    execution_id: str,
    *,
    output_json: bool = False,
    explicit_resume: bool = False,
) -> Any:
    """Inspect a task's checkpoint & resume safety assessment and render human or JSON output."""
    from zelyr.engine.checkpoint import CheckpointManager
    from zelyr.tool_registry import ToolRegistry

    assessment = CheckpointManager.assess_resume(
        ledger,
        execution_id,
        tool_registry=ToolRegistry(),
        explicit_resume=explicit_resume,
        verify_outcomes=True,
    )
    if output_json:
        print(json.dumps(assessment.to_dict(), indent=2))
        return assessment

    print(f"Execution:           {assessment.execution_id}")
    print(f"Checkpoint ID:       {assessment.checkpoint_id or 'reconstructed'}")
    print(f"Checkpoint Version:  {assessment.checkpoint_version or 'N/A'}")
    print(f"Task Status:         {assessment.task_status}")
    print(f"Verification Status: {assessment.verification_status}")
    print(f"Resumability:        {assessment.resumability_status} (can_resume={assessment.can_resume})")
    print(
        f"Operations:          verified={len(assessment.completed_verified_ops)} "
        f"unverified={len(assessment.completed_unverified_ops)} "
        f"in_flight={len(assessment.in_flight_ambiguous_ops)} "
        f"failed={len(assessment.failed_ops)} "
        f"contradictory={len(assessment.contradictory_ops)} "
        f"pending={len(assessment.pending_ops)}"
    )
    active_pa_cnt = assessment.to_dict()["pending_proposal_count"]
    print(f"Pending Proposals:   {active_pa_cnt} active ({len(assessment.pending_actions)} total)")
    if assessment.pending_actions:
        for pa in assessment.pending_actions:
            deps_str = ",".join(pa.depends_on) if pa.depends_on else "-"
            blk_str = f" reason={pa.blocked_reason}" if pa.blocked_reason else ""
            print(
                f"  - [{pa.proposal_order}] {pa.proposal_id} tool={pa.tool_id} status={pa.status} "
                f"auth={pa.authorization_status} dep={pa.dependency_status} depends_on=[{deps_str}] "
                f"dispatched={pa.dispatch_attempted}{blk_str}"
            )
    if assessment.task_graph is not None:
        _print_task_graph_view(assessment)
    if assessment.blocking_reasons:
        print("Blocking Reasons:")
        for reason in assessment.blocking_reasons:
            print(f"  - {reason}")
    if assessment.warnings:
        print("Warnings / Safety Notes:")
        for warn in assessment.warnings:
            print(f"  - {warn}")
    if assessment.environment_diff.get("changed"):
        print(f"Environment Drift:   {json.dumps(assessment.environment_diff.get('changes', []))}")
    return assessment


def cmd_checkpoint(args: argparse.Namespace) -> None:
    """Manage and inspect task continuity checkpoints (`list`, `inspect`, `abandon`)."""
    subcmd = getattr(args, "checkpoint_cmd", None)
    if subcmd == "list":
        setattr(args, "resumable", not bool(getattr(args, "all", False)))
        cmd_executions(args)
        return

    exec_query = getattr(args, "execution_id", "")
    ledger, owns_conn = _open_cli_ledger(getattr(args, "db_path", None))
    try:
        record, matches = ledger.find_execution_by_prefix(exec_query)
        if record is None:
            msg = (
                f"Ambiguous execution ID prefix '{exec_query}': matched {', '.join(matches[:5])}"
                if len(matches) > 1
                else f"Execution '{exec_query}' not found."
            )
            if getattr(args, "json", False):
                print(json.dumps({"error": msg}))
            else:
                sys.stderr.write(msg + "\n")
                sys.stderr.flush()
            sys.exit(1)

        if subcmd == "inspect":
            _print_checkpoint_assessment(ledger, record.execution_id, output_json=bool(getattr(args, "json", False)))
            return

        if subcmd == "abandon":
            from zelyr.engine.checkpoint import CheckpointError, CheckpointManager

            try:
                updated = CheckpointManager.abandon_task(
                    ledger,
                    record.execution_id,
                    reason=str(getattr(args, "reason", None) or "Abandoned via CLI"),
                    strict=True,
                )
            except (CheckpointError, Exception) as exc:
                if getattr(args, "json", False):
                    print(json.dumps({"error": str(exc), "execution_id": record.execution_id}))
                else:
                    sys.stderr.write(f"Error abandoning task '{record.execution_id}': {exc}\n")
                    sys.stderr.flush()
                sys.exit(1)
            if getattr(args, "json", False):
                print(json.dumps({"status": "abandoned", "execution": updated.to_dict() if updated else None}, indent=2))
            else:
                print(f"Abandoned execution '{record.execution_id}'.")
            return

        sys.stderr.write("Usage: zelyr checkpoint {list|inspect|abandon} ...\n")
        sys.stderr.flush()
        sys.exit(1)
    finally:
        if owns_conn:
            ledger.close()


def cmd_resume(args: argparse.Namespace) -> None:
    """List resumable tasks, explain resume blocks, or resume an eligible checkpointed task."""
    if getattr(args, "list", False) or not getattr(args, "execution_id", None):
        setattr(args, "resumable", True)
        cmd_executions(args)
        return

    exec_query = str(getattr(args, "execution_id") or "")
    ledger, owns_conn = _open_cli_ledger(getattr(args, "db_path", None))
    try:
        record, matches = ledger.find_execution_by_prefix(exec_query)
        if record is None:
            msg = (
                f"Ambiguous execution ID prefix '{exec_query}': matched {', '.join(matches[:5])}"
                if len(matches) > 1
                else f"Execution '{exec_query}' not found."
            )
            if getattr(args, "json", False):
                print(json.dumps({"error": msg}))
            else:
                sys.stderr.write(msg + "\n")
                sys.stderr.flush()
            sys.exit(1)

        if getattr(args, "abandon", False):
            setattr(args, "checkpoint_cmd", "abandon")
            cmd_checkpoint(args)
            return

        explicit_resume = bool(getattr(args, "confirm", False))
        allow_unverified_retry = bool(getattr(args, "allow_unverified_retry", False))
        output_json = bool(getattr(args, "json", False))

        if getattr(args, "inspect", False):
            assessment = _print_checkpoint_assessment(
                ledger,
                record.execution_id,
                output_json=output_json,
                explicit_resume=explicit_resume,
            )
            if not assessment.can_resume:
                sys.exit(2)
            return

        from zelyr.agents.native_orchestrator import NativeOrchestrator
        from zelyr.config import ZelyrConfig, get_config
        from zelyr.core.plugins import get_tool_schemas, load_plugins
        from zelyr.engine.checkpoint import CheckpointError, CheckpointManager
        from zelyr.tool_registry import ToolRegistry

        load_plugins()
        base_cfg = get_config()
        base_url_override = getattr(args, "base_url", None)
        run_cfg = None
        if base_url_override:
            run_cfg = ZelyrConfig.from_dict(base_cfg.to_dict()) if hasattr(base_cfg, "to_dict") else ZelyrConfig()
            run_cfg.ollama_base_url = (
                base_cfg.get_effective_ollama_base_url(base_url_override)
                if hasattr(base_cfg, "get_effective_ollama_base_url")
                else base_url_override.rstrip("/")
            )

        orch = NativeOrchestrator(config=run_cfg, tool_registry=ToolRegistry(), central_ledger=ledger)
        schemas = get_tool_schemas(0)
        pre_assess = CheckpointManager.assess_resume(
            ledger,
            record.execution_id,
            tool_registry=orch.tool_registry,
            active_schemas=schemas,
            explicit_resume=explicit_resume,
            allow_unverified_retry=allow_unverified_retry,
            verify_outcomes=True,
        )
        if not pre_assess.can_resume:
            if output_json:
                print(json.dumps(pre_assess.to_dict(), indent=2))
            else:
                sys.stderr.write(
                    f"Cannot safely resume '{record.execution_id}' ({pre_assess.resumability_status}):\n"
                    + "\n".join(f"  - {r}" for r in pre_assess.blocking_reasons)
                    + "\n"
                )
                sys.stderr.flush()
            sys.exit(2)

        try:
            extra_payload = {}
            if getattr(args, "verify", None):
                extra_payload["acceptance_criteria"] = args.verify
            _, chunks = asyncio.run(
                CheckpointManager.resume_execution(
                    orch,
                    ledger,
                    record.execution_id,
                    explicit_resume=explicit_resume,
                    allow_unverified_retry=allow_unverified_retry,
                    tool_schemas=schemas,
                    model=getattr(args, "model", None),
                    extra_payload=extra_payload or None,
                )
            )
        except CheckpointError as exc:
            if output_json:
                print(json.dumps({"error": str(exc), "execution_id": record.execution_id}))
            else:
                sys.stderr.write(f"Resume blocked for '{record.execution_id}': {exc}\n")
                sys.stderr.flush()
            sys.exit(2)

        text_parts = []
        for ch in chunks:
            for choice in ch.get("choices", []) if isinstance(ch, dict) else []:
                c_txt = (choice.get("delta") or {}).get("content")
                if isinstance(c_txt, str):
                    text_parts.append(c_txt)
        final_text = "".join(text_parts).strip()
        updated_rec = ledger.get_execution(record.execution_id)
        if output_json:
            print(
                json.dumps(
                    {
                        "execution_id": record.execution_id,
                        "assessment": pre_assess.to_dict(),
                        "execution": updated_rec.to_dict() if updated_rec else None,
                        "response": final_text,
                    },
                    indent=2,
                )
            )
        else:
            if final_text:
                print(final_text)
            if updated_rec:
                print(f"[{record.execution_id}] final status={updated_rec.status} verification={updated_rec.verification_status}")
    finally:
        if owns_conn:
            ledger.close()


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
    try:
        exit_code = run_benchmark_cli(eval_argv)
    finally:
        try:
            import concurrent.futures
            # Cleanly shutdown any pending futures or unblock exit
        except Exception:
            pass
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
        if transport == "stdio":
            sys.stderr.write(f"Error running MCP server: {exc}\n")
        else:
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
    p_chat = sub.add_parser("chat", aliases=["repl"], help="Interactive streaming chat REPL with multi-turn persistence")
    p_chat.add_argument("--session", "-s", type=str, default=None, help="Resume an existing session by ID or slug")
    p_chat.add_argument("--new", "-n", action="store_true", help="Start a new conversation session")
    p_chat.add_argument("--list", "-l", action="store_true", help="List recent conversation sessions")
    p_chat.add_argument("--delete", "-d", type=str, default=None, help="Delete a stored session by ID")
    p_chat.add_argument("--model", "-m", type=str, default=None, help="Override model for the chat session")
    p_chat.add_argument("--tui", action="store_true", default=False, help="Launch full-screen TUI interface")

    sub.add_parser("ui",     aliases=["gui"], help="Launch the PySide6 desktop UI")
    sub.add_parser("server", help="Print Zelyr Distributed Server panel (stub)")

    p_run = sub.add_parser("run", help="Execute a one-shot prompt headlessly via NativeOrchestrator")
    p_run.add_argument("prompt", type=str, help="Natural-language task prompt")
    p_run.add_argument("--model", "-m", type=str, default=None, help="Override model for this execution")
    p_run.add_argument("--base-url", type=str, default=None, help="Override Ollama base URL for this execution")
    p_run.add_argument("--json", action="store_true", default=False, help="Output structured JSON response")
    p_run.add_argument(
        "--verify",
        type=str,
        default=None,
        help="Deterministic outcome assertion formatted as CHECK:TARGET[:EXPECTED]",
    )
    p_run.add_argument(
        "--plan",
        action="store_true",
        default=False,
        help="Compile the prompt into a validated TaskGraph and execute via graph scheduler",
    )
    p_run.add_argument(
        "--plan-only",
        "--inspect-plan",
        dest="plan_only",
        action="store_true",
        default=False,
        help="Compile and display the proposed TaskGraph plan without executing it",
    )
    p_run.add_argument(
        "--confirm",
        "--yes",
        "-y",
        action="store_true",
        default=False,
        help="Explicitly confirm executing a compiled mutating task plan",
    )
    p_run.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum concurrent independent graph nodes to execute (default: 1)",
    )
    p_run.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Override execution ledger SQLite DB path",
    )
    p_run.add_argument(
        "--max-steps",
        type=int,
        default=12,
        help="Maximum autonomous ReAct steps for headless execution (default: 12)",
    )
    p_run.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Display detailed un-truncated operational traces on stderr",
    )

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
    p_tool_run.add_argument("--caller-ring", type=int, default=None, help="Explicit caller security ring (0-2)")
    p_tool_run.add_argument(
        "--policy-mode",
        type=str,
        default=None,
        choices=["conservative", "strict", "balanced", "basic", "autopilot"],
        help="Execution policy mode override for this invocation",
    )
    p_tool_run.add_argument("--confirm", action="store_true", default=False, help="Approve policy confirmation for this exact invocation")
    p_tool_run.add_argument("--json", action="store_true", default=False, help="Emit structured JSON output")

    p_mem = sub.add_parser("memory", help="Manage persistent memory and compaction")
    mem_sub = p_mem.add_subparsers(dest="mem_cmd")
    p_compact = mem_sub.add_parser("compact", help="Trigger REM sleep memory compaction")
    p_compact.add_argument("--days", type=float, default=30.0, help="Retention threshold in days")

    p_status = sub.add_parser("status", help="Show system + daemon status")
    p_status.add_argument(
        "--manifest",
        "-m",
        action="store_true",
        default=False,
        help="Display the full runtime environment manifest",
    )
    p_status.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output runtime environment manifest as JSON",
    )
    p_status.add_argument(
        "--no-probe",
        action="store_true",
        default=False,
        help="Skip live endpoint probes when inspecting manifest",
    )

    p_env = sub.add_parser(
        "env",
        aliases=["manifest"],
        help="Inspect the structured runtime environment and capability manifest",
    )
    p_env.add_argument("--json", action="store_true", default=False, help="Output structured JSON manifest")
    p_env.add_argument(
        "--compact",
        action="store_true",
        default=False,
        help="Output compact planning manifest block",
    )
    p_env.add_argument(
        "--no-probe",
        action="store_true",
        default=False,
        help="Skip live model endpoint health probes",
    )
    p_env.add_argument(
        "--timeout",
        type=float,
        default=0.5,
        help="Bounded timeout in seconds for external endpoint probes (default: 0.5)",
    )
    p_env.add_argument(
        "--caller-ring",
        type=int,
        default=None,
        help="Optional caller security ring (0..3) to scope tool authorization",
    )
    p_env.add_argument(
        "--allowed-tools",
        type=str,
        default=None,
        help="Optional comma-separated tool allowlist to scope tool authorization",
    )
    p_env.add_argument(
        "--refresh",
        action="store_true",
        default=False,
        help="Force-refresh cached manifest state",
    )
    p_env.add_argument(
        "--model-check",
        action="store_true",
        default=False,
        help="Include honest model/provider compatibility diagnostics in the manifest",
    )

    p_execs = sub.add_parser(
        "executions",
        help="List recent centralized execution ledger records",
    )
    p_execs.add_argument("--limit", "-n", type=int, default=20, help="Maximum executions to list (default: 20)")
    p_execs.add_argument("--status", type=str, default=None, help="Filter executions by status")
    p_execs.add_argument("--tool", type=str, default=None, help="Filter executions involving a tool ID")
    p_execs.add_argument("--event-type", type=str, default=None, help="Filter executions containing an event type")
    p_execs.add_argument(
        "--resumable",
        action="store_true",
        default=False,
        help="Show only interrupted or incomplete tasks and their resume eligibility",
    )
    p_execs.add_argument("--json", action="store_true", default=False, help="Output structured JSON")
    p_execs.add_argument("--db-path", type=str, default=None, help="Override execution ledger SQLite DB path")

    p_exec = sub.add_parser(
        "execution",
        help="Inspect an execution ledger record and its ordered events",
    )
    p_exec.add_argument("execution_id", type=str, help="Execution ID or unique prefix to inspect")
    p_exec.add_argument("--event-type", type=str, default=None, help="Filter timeline events by event type")
    p_exec.add_argument("--tool", type=str, default=None, help="Filter timeline events by tool ID")
    p_exec.add_argument(
        "--checkpoint",
        action="store_true",
        default=False,
        help="Display latest durable checkpoint and resume safety assessment",
    )
    p_exec.add_argument(
        "--graph",
        action="store_true",
        default=False,
        help="Display dependency-aware task graph, node states, and blockers",
    )
    p_exec.add_argument(
        "--diagnose",
        action="store_true",
        default=False,
        help="Display unified end-to-end execution lifecycle diagnostics",
    )
    p_exec.add_argument(
        "--verify",
        action="store_true",
        default=False,
        help="Reconcile live OutcomeVerifier checks when running --diagnose",
    )
    p_exec.add_argument(
        "--metrics",
        action="store_true",
        default=False,
        help="Display execution and policy observability metrics",
    )
    p_exec.add_argument("--json", action="store_true", default=False, help="Output structured JSON summary")
    p_exec.add_argument("--db-path", type=str, default=None, help="Override execution ledger SQLite DB path")

    p_diag = sub.add_parser(
        "diagnose",
        help="Run read-only end-to-end lifecycle diagnostics for an execution",
    )
    p_diag.add_argument("execution_id", type=str, help="Execution ID or unique prefix to diagnose")
    p_diag.add_argument("--verify", action="store_true", default=False, help="Reconcile live OutcomeVerifier checks (read-only)")
    p_diag.add_argument("--json", action="store_true", default=False, help="Output structured JSON diagnostic report")
    p_diag.add_argument("--db-path", type=str, default=None, help="Override execution ledger SQLite DB path")

    p_metrics = sub.add_parser(
        "metrics",
        aliases=["observability"],
        help="Summarize measured execution timings, counts, and policy outcomes",
    )
    p_metrics.add_argument("execution_id", nargs="?", default=None, help="Optional execution ID or prefix (omit for recent summary)")
    p_metrics.add_argument("--limit", "-n", type=int, default=20, help="Maximum recent executions to aggregate (default: 20)")
    p_metrics.add_argument("--json", action="store_true", default=False, help="Output structured JSON metrics")
    p_metrics.add_argument("--db-path", type=str, default=None, help="Override execution ledger SQLite DB path")

    p_config = sub.add_parser(
        "config",
        help="Inspect and validate configuration provenance, types, ranges, and unknown keys",
    )
    config_sub = p_config.add_subparsers(dest="config_cmd")
    for cfg_sub_name, cfg_help in (
        ("show", "Show effective configuration settings and sources"),
        ("validate", "Validate configuration types, bounds, unknown keys, and divergence"),
    ):
        p_cfg_sub = config_sub.add_parser(cfg_sub_name, help=cfg_help)
        p_cfg_sub.add_argument("--config-path", type=str, default=None, help="Inspect an explicit config JSON file")
        p_cfg_sub.add_argument("--strict", action="store_true", default=False, help="Exit non-zero on warnings or unknown keys")
        p_cfg_sub.add_argument("--json", action="store_true", default=False, help="Output structured JSON")
    p_config.add_argument("--config-path", type=str, default=None, help="Inspect an explicit config JSON file")
    p_config.add_argument("--strict", action="store_true", default=False, help="Exit non-zero on warnings or unknown keys")
    p_config.add_argument("--json", action="store_true", default=False, help="Output structured JSON")

    p_model = sub.add_parser(
        "model",
        help="Inspect model and provider compatibility for Zelyr workflows",
    )
    model_sub = p_model.add_subparsers(dest="model_cmd")
    p_model_check = model_sub.add_parser(
        "check",
        help="Assess model capabilities, context window, and workflow readiness",
    )
    p_model_check.add_argument("--model", type=str, default=None, help="Model name to assess (defaults to configured model)")
    p_model_check.add_argument("--base-url", type=str, default=None, help="Provider endpoint URL override")
    p_model_check.add_argument(
        "--workflow",
        choices=["planned_graph", "react_tools", "chat", "vision"],
        default="planned_graph",
        help="Target Zelyr workflow to evaluate (default: planned_graph)",
    )
    p_model_check.add_argument("--no-probe", action="store_true", default=False, help="Skip live HTTP probes (honestly report unknown)")
    p_model_check.add_argument("--timeout", type=float, default=2.0, help="HTTP probe timeout in seconds (default: 2.0)")
    p_model_check.add_argument("--strict", action="store_true", default=False, help="Exit non-zero unless status is available and workflow_compatible is true")
    p_model_check.add_argument("--json", action="store_true", default=False, help="Output structured JSON")

    p_cp = sub.add_parser(
        "checkpoint",
        help="List, inspect, or abandon task continuity checkpoints",
    )
    cp_sub = p_cp.add_subparsers(dest="checkpoint_cmd")
    p_cp_list = cp_sub.add_parser("list", aliases=["ls"], help="List interrupted and resumable tasks")
    p_cp_list.add_argument("--limit", "-n", type=int, default=20, help="Maximum tasks to list")
    p_cp_list.add_argument("--all", "-a", action="store_true", default=False, help="Include completed/abandoned records")
    p_cp_list.add_argument("--json", action="store_true", default=False, help="Output structured JSON")
    p_cp_list.add_argument("--db-path", type=str, default=None, help="Override execution ledger SQLite DB path")

    p_cp_inspect = cp_sub.add_parser("inspect", help="Inspect a task checkpoint and explain resume eligibility")
    p_cp_inspect.add_argument("execution_id", type=str, help="Execution ID or unique prefix")
    p_cp_inspect.add_argument("--json", action="store_true", default=False, help="Output structured JSON")
    p_cp_inspect.add_argument("--db-path", type=str, default=None, help="Override execution ledger SQLite DB path")

    p_cp_abandon = cp_sub.add_parser("abandon", help="Mark an interrupted task as abandoned")
    p_cp_abandon.add_argument("execution_id", type=str, help="Execution ID or unique prefix")
    p_cp_abandon.add_argument("--reason", type=str, default="Abandoned via CLI", help="Reason recorded in execution ledger")
    p_cp_abandon.add_argument("--json", action="store_true", default=False, help="Output structured JSON")
    p_cp_abandon.add_argument("--db-path", type=str, default=None, help="Override execution ledger SQLite DB path")

    p_resume = sub.add_parser(
        "resume",
        help="List interrupted tasks, explain resume blocks, or explicitly resume a task",
    )
    p_resume.add_argument("execution_id", nargs="?", default=None, help="Execution ID or unique prefix to resume")
    p_resume.add_argument("--list", "-l", action="store_true", default=False, help="List interrupted and resumable tasks")
    p_resume.add_argument("--inspect", "-i", action="store_true", default=False, help="Inspect checkpoint and explain why task can/cannot resume")
    p_resume.add_argument("--abandon", action="store_true", default=False, help="Abandon the specified task")
    p_resume.add_argument("--confirm", "--yes", "-y", action="store_true", default=False, help="Explicitly confirm resuming a cancelled or flagged task")
    p_resume.add_argument("--allow-unverified-retry", action="store_true", default=False, help="Allow retrying previously ambiguous unverified operations")
    p_resume.add_argument("--verify", type=str, default=None, help="Optional OutcomeVerifier criteria string")
    p_resume.add_argument("--model", "-m", type=str, default=None, help="Override Ollama model when resuming")
    p_resume.add_argument("--base-url", type=str, default=None, help="Override Ollama base URL when resuming")
    p_resume.add_argument("--reason", type=str, default="Abandoned via CLI", help="Reason when --abandon is used")
    p_resume.add_argument("--limit", "-n", type=int, default=20, help="Maximum tasks when listing")
    p_resume.add_argument("--json", action="store_true", default=False, help="Output structured JSON")
    p_resume.add_argument("--db-path", type=str, default=None, help="Override execution ledger SQLite DB path")

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

    # ── Execution Policy inspection, preview & history ───────────────────────
    p_policy = sub.add_parser(
        "policy",
        help="Inspect effective execution policy, preview tool decisions, and view policy history",
    )
    policy_sub = p_policy.add_subparsers(dest="policy_cmd")

    p_pol_show = policy_sub.add_parser("show", help="Show effective execution policy settings and precedence")
    p_pol_show.add_argument("--config-path", type=str, default=None, help="Load settings from an explicit ZelyrConfig JSON file")
    p_pol_show.add_argument(
        "--policy-mode",
        choices=["conservative", "strict", "balanced", "basic", "autopilot"],
        default=None,
        help="Preview effective defaults under a specific policy mode",
    )
    p_pol_show.add_argument("--json", action="store_true", default=False, help="Output structured JSON")

    p_pol_explain = policy_sub.add_parser("explain", help="Preview and explain policy outcome for a tool without executing")
    p_pol_explain.add_argument("tool_id", help="Tool ID to evaluate in non-executing preview mode")
    p_pol_explain.add_argument("--args", type=str, default=None, help="JSON object string of tool arguments to preview")
    p_pol_explain.add_argument("--caller-ring", type=int, default=None, help="Simulated caller ring (0-3)")
    p_pol_explain.add_argument("--allowed-tools", type=str, default=None, help="Comma-separated allowed tool IDs")
    p_pol_explain.add_argument("--workspace-root", type=str, default=None, help="Workspace root path for containment preview")
    p_pol_explain.add_argument("--config-path", type=str, default=None, help="Load settings from an explicit ZelyrConfig JSON file")
    p_pol_explain.add_argument(
        "--policy-mode",
        choices=["conservative", "strict", "balanced", "basic", "autopilot"],
        default=None,
        help="Override policy mode for explanation preview",
    )
    p_pol_explain.add_argument("--json", action="store_true", default=False, help="Output structured JSON")

    p_pol_hist = policy_sub.add_parser("history", help="Query redacted AUTHORIZATION_DECISION history from execution ledger")
    p_pol_hist.add_argument("--limit", "-n", type=int, default=20, help="Maximum history entries to show (default: 20)")
    p_pol_hist.add_argument("--outcome", type=str, default=None, help="Filter by outcome (allow, require_confirmation, deny)")
    p_pol_hist.add_argument("--tool", type=str, default=None, help="Filter by tool ID")
    p_pol_hist.add_argument("--execution-id", type=str, default=None, help="Filter by execution ID")
    p_pol_hist.add_argument("--db-path", type=str, default=None, help="Override execution ledger SQLite DB path")
    p_pol_hist.add_argument("--json", action="store_true", default=False, help="Output structured JSON")

    # ── Browser Extension & Native Messaging host ────────────────────────────
    p_browser = sub.add_parser(
        "browser",
        help="Install or inspect Chromium Native Messaging host and extension bridge status",
    )
    browser_sub = p_browser.add_subparsers(dest="browser_cmd")

    p_br_install = browser_sub.add_parser("install-host", help="Install user-scoped Chromium NativeMessagingHosts manifest")
    p_br_install.add_argument("--extension-id", type=str, default=None, help="Override Chromium extension ID")
    p_br_install.add_argument("--user-data-dir", type=str, default=None, help="Target custom Chromium user-data-dir")
    p_br_install.add_argument("--allowed-hosts", type=str, default=None, help="Comma-separated host allowlist for native bridge")
    p_br_install.add_argument("--json", action="store_true", default=False, help="Output structured JSON")

    p_br_status = browser_sub.add_parser("status", help="Inspect browser extension and Native Messaging host status")
    p_br_status.add_argument("--user-data-dir", type=str, default=None, help="Inspect custom Chromium user-data-dir")
    p_br_status.add_argument("--json", action="store_true", default=False, help="Output structured JSON")

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
        "run":      cmd_run,
        "send":     cmd_send,
        "status":   cmd_status,
        "env":      cmd_env,
        "manifest": cmd_env,
        "executions": cmd_executions,
        "execution":     cmd_execution,
        "diagnose":      cmd_diagnose,
        "metrics":       cmd_metrics,
        "observability": cmd_metrics,
        "config":        cmd_config,
        "checkpoint": cmd_checkpoint,
        "resume":     cmd_resume,
        "test":     cmd_test,
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
    elif args.command == "policy":
        if getattr(args, "policy_cmd", None):
            cmd_policy(args)
        else:
            parser.parse_args(["policy", "--help"])
    elif args.command == "browser":
        if getattr(args, "browser_cmd", None):
            cmd_browser(args)
        else:
            parser.parse_args(["browser", "--help"])
    elif args.command == "model":
        if getattr(args, "model_cmd", None) == "check":
            cmd_model(args)
        else:
            parser.parse_args(["model", "--help"])
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
