"""Isolated subprocess worker for Zelyr distributed evaluation tasks.

Executes a single benchmark task inside an isolated temporary workspace using
real `_execute_headless_run`, `NativeOrchestrator`, isolated `CentralExecutionLedger`,
`ExecutionPolicyEngine`, `TurnExecutionLedger`, and `OutcomeVerifier`.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.cli.main import _execute_headless_run
from zelyr.config import ZelyrConfig, get_config, initialize_directories, set_config
from zelyr.engine.error_classifier import _redact_secrets
from zelyr.engine.execution_ledger import (
    CentralExecutionLedger,
    ExecutionStatus,
    TurnExecutionLedger,
    VerificationStatus,
    execution_scope,
    parse_acceptance_criteria,
)
from zelyr.engine.execution_policy import (
    ConfirmationStore,
    ExecutionPolicyConfig,
    ExecutionPolicyEngine,
    query_policy_history,
    sanitize_policy_metadata,
)
from zelyr.engine.verifier import OutcomeVerifier


_WORKER_ENV_KEYS = (
    "HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "ZELYR_CONFIG_DIR",
    "ZELYR_LEDGER_DB",
    "AXIOM_CONFIG_DIR",
    "ZELYR_TESTING",
    "ZELYR_AUTOPILOT",
    "QT_QPA_PLATFORM",
    "ZELYR_OLLAMA_BASE_URL",
    "AXIOM_OLLAMA_BASE_URL",
    "ZELYR_NUM_CTX",
    "AXIOM_NUM_CTX",
    "ZELYR_OLLAMA_KEEP_ALIVE",
    "AXIOM_OLLAMA_KEEP_ALIVE",
    "ZELYR_OLLAMA_READ_TIMEOUT",
    "AXIOM_OLLAMA_READ_TIMEOUT",
)


def _attach_stream_telemetry(orch: NativeOrchestrator) -> Dict[str, Any]:
    """Capture Ollama terminal chunk timings, token counters, thinking tokens, and live inference intervals."""
    telemetry: Dict[str, Any] = {
        "turn_metrics": [],
        "inference_intervals": [],
        "thinking_chunks": 0,
        "thinking_chars": 0,
    }
    orig_parser = orch._parse_stream_line_to_events
    orig_stream_events = orch._stream_model_events

    def tapped_parser(line: str):
        stripped = (line or "").strip()
        if stripped.startswith("{"):
            try:
                raw = json.loads(stripped)
                if isinstance(raw, dict):
                    msg = raw.get("message")
                    if isinstance(msg, dict) and msg.get("thinking"):
                        telemetry["thinking_chunks"] += 1
                        telemetry["thinking_chars"] += len(str(msg.get("thinking")))
                    if raw.get("done") is True:
                        telemetry["turn_metrics"].append({
                            "model": raw.get("model"),
                            "total_duration_ms": round(raw.get("total_duration", 0) / 1e6, 1) if raw.get("total_duration") else None,
                            "load_duration_ms": round(raw.get("load_duration", 0) / 1e6, 1) if raw.get("load_duration") else None,
                            "prompt_eval_count": raw.get("prompt_eval_count"),
                            "prompt_eval_duration_ms": round(raw.get("prompt_eval_duration", 0) / 1e6, 1) if raw.get("prompt_eval_duration") else None,
                            "eval_count": raw.get("eval_count"),
                            "eval_duration_ms": round(raw.get("eval_duration", 0) / 1e6, 1) if raw.get("eval_duration") else None,
                            "done_reason": raw.get("done_reason"),
                        })
            except Exception:
                pass
        return orig_parser(line)

    async def tapped_stream_events(payload: dict, base_url: str, stream_provider: Optional[Any] = None):
        turn_idx = len(telemetry["inference_intervals"]) + 1
        start_wall = time.time()
        start_mon = time.monotonic()
        first_chunk_wall: Optional[float] = None
        first_chunk_mon: Optional[float] = None
        chunk_count = 0
        interval_rec: Dict[str, Any] = {
            "turn": turn_idx,
            "model": str(payload.get("model") or ""),
            "endpoint": _redact_secrets(str(base_url or "")),
            "start_ts": start_wall,
            "first_chunk_ts": None,
            "end_ts": None,
            "duration_seconds": 0.0,
            "first_chunk_latency_seconds": None,
            "chunk_count": 0,
        }
        try:
            async for item in orig_stream_events(payload, base_url, stream_provider=stream_provider):
                if first_chunk_wall is None:
                    first_chunk_wall = time.time()
                    first_chunk_mon = time.monotonic()
                    interval_rec["first_chunk_ts"] = first_chunk_wall
                    interval_rec["first_chunk_latency_seconds"] = round(first_chunk_mon - start_mon, 4)
                chunk_count += 1
                yield item
        finally:
            end_wall = time.time()
            interval_rec["end_ts"] = end_wall
            interval_rec["duration_seconds"] = round(max(0.0, time.monotonic() - start_mon), 4)
            interval_rec["chunk_count"] = chunk_count
            telemetry["inference_intervals"].append(interval_rec)

    orch._parse_stream_line_to_events = tapped_parser  # type: ignore[method-assign]
    orch._stream_model_events = tapped_stream_events  # type: ignore[method-assign]
    return telemetry


_INPROCESS_ENV_LOCK_DEPTH = 0
_INPROCESS_BASE_STATE: Optional[Tuple[Dict[str, Optional[str]], Any, Path]] = None


async def _run_worker(spec: Dict[str, Any]) -> Dict[str, Any]:
    global _INPROCESS_ENV_LOCK_DEPTH, _INPROCESS_BASE_STATE
    if not isinstance(spec, dict):
        return {
            "exit_code": 1,
            "execution_id": None,
            "cli_json": {},
            "cli_json_valid": False,
            "worker_error": f"Invalid worker spec type: {type(spec).__name__}",
            "malformed_worker_output": True,
            "raw_stderr": "",
            "files_on_disk": {},
            "verifier_checks": [],
            "execution_ledger": {},
            "ledger_transitions": [],
            "policy_decisions": [],
            "checkpoint_snapshot": None,
            "messages_transcript": [],
            "telemetry": {},
        }
    for req_key in ("endpoint", "model", "workspace_dir", "prompt"):
        val = spec.get(req_key)
        if not isinstance(val, str) or not val.strip():
            return {
                "exit_code": 1,
                "execution_id": None,
                "cli_json": {},
                "cli_json_valid": False,
                "worker_error": f"Invalid or missing required worker spec field '{req_key}'",
                "malformed_worker_output": True,
                "raw_stderr": "",
                "files_on_disk": {},
                "verifier_checks": [],
                "execution_ledger": {},
                "ledger_transitions": [],
                "policy_decisions": [],
                "checkpoint_snapshot": None,
                "messages_transcript": [],
                "telemetry": {},
            }

    if _INPROCESS_ENV_LOCK_DEPTH == 0 or _INPROCESS_BASE_STATE is None:
        _INPROCESS_BASE_STATE = (
            {k: os.environ.get(k) for k in _WORKER_ENV_KEYS},
            get_config(),
            Path.cwd(),
        )
    _INPROCESS_ENV_LOCK_DEPTH += 1

    ws = Path(spec["workspace_dir"]).resolve()
    ws.mkdir(parents=True, exist_ok=True)

    raw_state_dir = spec.get("isolated_state_dir")
    owns_state_dir = False
    if isinstance(raw_state_dir, str) and raw_state_dir.strip():
        state_root = Path(raw_state_dir).resolve()
        state_root.mkdir(parents=True, exist_ok=True)
    else:
        state_root = Path(tempfile.mkdtemp(prefix=f"zdist_wstate_{os.getpid()}_")).resolve()
        owns_state_dir = True

    isolated_home = state_root / "home"
    isolated_cfg = state_root / "config"
    isolated_xdg_cfg = state_root / "xdg_config"
    isolated_xdg_data = state_root / "xdg_data"
    for d in (isolated_home, isolated_cfg, isolated_xdg_cfg, isolated_xdg_data):
        d.mkdir(parents=True, exist_ok=True)

    raw_ledger_db = spec.get("ledger_db_path")
    ledger_db_path = Path(raw_ledger_db).resolve() if isinstance(raw_ledger_db, str) and raw_ledger_db.strip() else (isolated_cfg / "worker_executions.db")
    ledger_db_path.parent.mkdir(parents=True, exist_ok=True)

    eff_ctx = int(spec.get("num_ctx", 8192))
    eff_keep_alive = str(spec.get("keep_alive", "5m"))
    eff_read_timeout = float(spec.get("read_timeout", 1800.0))

    # Clear inherited endpoint/context overrides so worker strictly hits its assigned endpoint.
    for clear_key in (
        "AXIOM_CONFIG_DIR",
        "AXIOM_OLLAMA_BASE_URL",
        "AXIOM_NUM_CTX",
        "AXIOM_OLLAMA_KEEP_ALIVE",
        "AXIOM_OLLAMA_READ_TIMEOUT",
    ):
        os.environ.pop(clear_key, None)

    os.environ["HOME"] = str(isolated_home)
    os.environ["XDG_CONFIG_HOME"] = str(isolated_xdg_cfg)
    os.environ["XDG_DATA_HOME"] = str(isolated_xdg_data)
    os.environ["ZELYR_CONFIG_DIR"] = str(isolated_cfg)
    os.environ["ZELYR_LEDGER_DB"] = str(ledger_db_path)
    os.environ["ZELYR_TESTING"] = "1"
    os.environ["ZELYR_AUTOPILOT"] = "1"
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["ZELYR_OLLAMA_BASE_URL"] = spec["endpoint"]
    os.environ["ZELYR_NUM_CTX"] = str(eff_ctx)
    os.environ["ZELYR_OLLAMA_KEEP_ALIVE"] = eff_keep_alive
    os.environ["ZELYR_OLLAMA_READ_TIMEOUT"] = str(eff_read_timeout)

    central_ledger: Optional[CentralExecutionLedger] = None
    worker_exec_id: Optional[str] = None
    try:
        initialize_directories(isolated_cfg)
        if _INPROCESS_ENV_LOCK_DEPTH == 1:
            os.chdir(ws)

        policy_mode = str(spec.get("policy_mode") or "basic").strip().lower()
        cfg = ZelyrConfig(
            ollama_base_url=spec["endpoint"],
            ollama_num_ctx=eff_ctx,
            ollama_keep_alive=eff_keep_alive,
            ollama_read_timeout=eff_read_timeout,
            ollama_model=spec["model"],
            execution_policy_mode=policy_mode,
        )
        if _INPROCESS_ENV_LOCK_DEPTH == 1:
            set_config(cfg)

        central_ledger = CentralExecutionLedger(
            db_path=ledger_db_path,
            raise_on_persistence_error=False,
        )
        conf_store = ConfirmationStore(db_path=isolated_cfg / "worker_confirmations.db")
        pol_cfg = ExecutionPolicyConfig(
            policy_mode=policy_mode,
            resource_restrictions={
                "enforce_workspace_containment": True,
                "workspace_root": str(ws),
            },
        )
        pol_engine = ExecutionPolicyEngine(config=pol_cfg, confirmation_store=conf_store)

        raw_allowed = spec.get("allowed_tools")
        allowed_tools = [str(t) for t in raw_allowed] if isinstance(raw_allowed, list) else None
        raw_ring = spec.get("caller_ring")
        caller_ring = int(raw_ring) if raw_ring is not None else None

        exec_rec = central_ledger.create_execution(
            spec["prompt"],
            source_component="distributed_worker",
            caller_ring=caller_ring,
            status=ExecutionStatus.RUNNING,
            metadata={
                "model": spec["model"],
                "worker_id": str(spec.get("worker_id") or ""),
                "task_id": str(spec.get("task_id") or ""),
            },
        )
        worker_exec_id = exec_rec.execution_id

        turn_ledger = TurnExecutionLedger(
            execution_id=worker_exec_id,
            central_ledger=central_ledger,
            caller_ring=caller_ring,
            source_component="distributed_worker",
        )

        orch = NativeOrchestrator(
            config=cfg,
            caller_ring=caller_ring,
            allowed_tools=allowed_tools,
            central_ledger=central_ledger,
        )
        orch.current_ledger = turn_ledger
        orch._preserve_current_ledger = True  # type: ignore[attr-defined]
        orch.workspace_root = ws
        orch.execution_policy_config = pol_cfg
        orch.execution_policy_engine = pol_engine
        orch.confirmation_store = conf_store
        orch.max_steps = int(spec.get("max_steps", 8))
        orch.max_replanning_attempts = int(spec.get("max_replanning", 2))
        telemetry = _attach_stream_telemetry(orch)

        model_options = dict(spec.get("model_options") or {})
        spec_think = spec.get("think") if "think" in spec else None
        captured_payloads: List[Dict[str, Any]] = []
        orig_generate_stream = orch.generate_stream

        async def instrumented_generate_stream(payload: dict, *args: Any, **kwargs: Any):
            if spec_think is not None and "think" not in payload:
                payload["think"] = bool(spec_think)
            if model_options:
                opts = dict(payload.get("options") or {})
                opts.update(model_options)
                payload["options"] = opts
            if "ledger" not in kwargs or kwargs["ledger"] is None:
                kwargs["ledger"] = getattr(orch, "current_ledger", None) or turn_ledger
            if kwargs.get("depth", 0) == 0:
                captured_payloads.append(payload)
            if "max_replanning_attempts" not in kwargs:
                kwargs["max_replanning_attempts"] = int(spec.get("max_replanning", 2))
            async for chunk in orig_generate_stream(payload, *args, **kwargs):
                yield chunk

        orch.generate_stream = instrumented_generate_stream  # type: ignore[method-assign]

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        exit_code: Any = 1
        worker_error: Optional[str] = None
        worker_cancelled = False

        try:
            with execution_scope(
                central_ledger,
                worker_exec_id,
                caller_ring=caller_ring,
                source_component="distributed_worker",
                owned_by_outer=True,
            ):
                with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                    exit_code = await _execute_headless_run(
                        prompt=spec["prompt"],
                        model_override=spec["model"],
                        output_json=True,
                        orchestrator=orch,
                        verify_spec=spec.get("verify_spec"),
                    )
        except asyncio.CancelledError:
            worker_cancelled = True
            worker_error = "CancelledError: Worker task cancelled during execution."
            stderr_buf.write(f"\n{worker_error}\n")
            exit_code = 1
        except Exception as exc:
            worker_error = f"{type(exc).__name__}: {_redact_secrets(str(exc))}"
            stderr_buf.write(f"\nWorker exception: {worker_error}\n")
            exit_code = 1

        raw_stdout = stdout_buf.getvalue()
        raw_stderr = _redact_secrets(stderr_buf.getvalue())

        cli_json: Dict[str, Any] = {}
        cli_json_valid = False
        for line in reversed(raw_stdout.splitlines()):
            s = line.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, dict):
                        cli_json = parsed
                        cli_json_valid = True
                        break
                except Exception:
                    pass

        files_on_disk: Dict[str, str] = {}
        for p in sorted(ws.rglob("*")):
            if p.is_file():
                try:
                    files_on_disk[str(p.relative_to(ws))] = p.read_text(encoding="utf-8")[:500]
                except Exception as exc:
                    files_on_disk[str(p.relative_to(ws))] = f"<read error: {exc}>"

        verifier_checks: List[Dict[str, Any]] = []
        verify_spec = spec.get("verify_spec")
        if verify_spec:
            verifier_checks = [
                OutcomeVerifier.verify(
                    check=item["check"],
                    target=item["target"],
                    expected=item.get("expected"),
                ).to_dict()
                for item in parse_acceptance_criteria(verify_spec)
            ]

        active_turn_ledger = getattr(orch, "current_ledger", None) or turn_ledger
        if not getattr(active_turn_ledger, "_execution_id", None):
            active_turn_ledger.attach_central_ledger(
                central_ledger,
                worker_exec_id,
                caller_ring=caller_ring,
                source_component="distributed_worker",
            )
        exec_id = active_turn_ledger._execution_id or worker_exec_id

        cur_exec_rec = central_ledger.get_execution(exec_id)
        if cur_exec_rec is None or cur_exec_rec.status in (
            ExecutionStatus.INITIALIZED.value,
            ExecutionStatus.RUNNING.value,
        ):
            if worker_cancelled:
                central_ledger.finalize_execution(
                    exec_id,
                    status=ExecutionStatus.CANCELLED,
                    verification_status=VerificationStatus.INTERRUPTED,
                    error="Worker task cancelled during execution.",
                )
            elif worker_error:
                central_ledger.finalize_execution(
                    exec_id,
                    status=ExecutionStatus.INTERRUPTED,
                    verification_status=VerificationStatus.INTERRUPTED,
                    error=_redact_secrets(worker_error),
                )
            elif exit_code == 0 and not active_turn_ledger.has_unresolved_failures():
                v_stat = (
                    VerificationStatus.VERIFIED
                    if (
                        (verifier_checks and all(vc.get("success") for vc in verifier_checks))
                        or bool(active_turn_ledger.verified_actions())
                    )
                    else VerificationStatus.NOT_VERIFIED
                )
                central_ledger.finalize_execution(
                    exec_id,
                    status=ExecutionStatus.COMPLETED if v_stat == VerificationStatus.VERIFIED or not verify_spec else ExecutionStatus.COMPLETED_UNVERIFIED,
                    verification_status=v_stat,
                )
            else:
                central_ledger.finalize_execution(
                    exec_id,
                    status=ExecutionStatus.FAILED,
                    verification_status=(
                        VerificationStatus.VERIFICATION_FAILED
                        if verifier_checks and any(not vc.get("success") for vc in verifier_checks)
                        else VerificationStatus.NOT_VERIFIED
                    ),
                    error=_redact_secrets(raw_stderr[:240] or f"Worker exited with code {exit_code}"),
                )

        ledger_dict = active_turn_ledger.to_dict()
        ledger_dict["execution_id"] = exec_id

        # Query strictly by this worker execution's explicit execution_id — never infer or scan unrelated runs.
        ledger_transitions: List[Dict[str, Any]] = [
            {
                "event_id": ev.event_id,
                "execution_id": ev.execution_id,
                "sequence_number": ev.sequence_number,
                "event_type": ev.event_type,
                "tool_id": ev.tool_id,
                "correlation_id": ev.correlation_id,
                "status": ev.status,
                "timestamp": ev.timestamp,
                "error_type": (ev.metadata or {}).get("error_type") if isinstance(ev.metadata, dict) else None,
                "redacted": bool(ev.redacted),
                "metadata": dict(ev.metadata or {}),
            }
            for ev in central_ledger.get_events(exec_id)
        ]
        hist = query_policy_history(central_ledger, execution_id=exec_id, limit=200)
        policy_decisions: List[Dict[str, Any]] = list(hist.get("entries") or [])

        checkpoint_snapshot: Optional[Dict[str, Any]] = None
        try:
            raw_cp = central_ledger.get_latest_checkpoint(exec_id, verify_integrity=False)
            if isinstance(raw_cp, dict):
                checkpoint_snapshot = sanitize_policy_metadata({
                    "checkpoint_id": raw_cp.get("checkpoint_id"),
                    "checkpoint_seq": raw_cp.get("checkpoint_seq"),
                    "status": raw_cp.get("status"),
                    "verification_status": raw_cp.get("verification_status"),
                    "last_event_seq": raw_cp.get("last_event_seq"),
                    "verified_targets": raw_cp.get("verified_targets"),
                })
        except Exception:
            checkpoint_snapshot = None

        messages_transcript = captured_payloads[0].get("messages", []) if captured_payloads else []

        return {
            "exit_code": exit_code,
            "execution_id": exec_id,
            "cancelled": worker_cancelled,
            "cli_json": cli_json,
            "cli_json_valid": cli_json_valid,
            "worker_error": worker_error,
            "raw_stderr": raw_stderr,
            "files_on_disk": files_on_disk,
            "verifier_checks": verifier_checks,
            "execution_ledger": ledger_dict,
            "ledger_transitions": ledger_transitions,
            "policy_decisions": policy_decisions,
            "checkpoint_snapshot": checkpoint_snapshot,
            "messages_transcript": messages_transcript,
            "telemetry": telemetry,
        }
    finally:
        if central_ledger is not None:
            try:
                central_ledger.close()
            except Exception:
                pass
        _INPROCESS_ENV_LOCK_DEPTH = max(0, _INPROCESS_ENV_LOCK_DEPTH - 1)
        if _INPROCESS_ENV_LOCK_DEPTH == 0 and _INPROCESS_BASE_STATE is not None:
            prev_env, prev_cfg, prev_cwd = _INPROCESS_BASE_STATE
            _INPROCESS_BASE_STATE = None
            try:
                set_config(prev_cfg)
            except Exception:
                pass
            for k, old_v in prev_env.items():
                if old_v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old_v
            try:
                os.chdir(prev_cwd)
            except Exception:
                pass
        if owns_state_dir:
            shutil.rmtree(state_root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Zelyr Distributed Eval Worker")
    parser.add_argument("--spec-file", required=True)
    parser.add_argument("--out-file", required=True)
    args = parser.parse_args()

    spec = json.loads(Path(args.spec_file).read_text(encoding="utf-8"))
    try:
        res = asyncio.run(_run_worker(spec))
    finally:
        if spec.get("cleanup_workspace", False):
            shutil.rmtree(spec["workspace_dir"], ignore_errors=True)
    Path(args.out_file).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
