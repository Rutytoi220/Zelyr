"""Isolated subprocess worker for Zelyr distributed evaluation tasks.

Executes a single benchmark task inside an isolated temporary workspace using
real `_execute_headless_run`, `NativeOrchestrator`, tool execution, `TurnExecutionLedger`,
and `OutcomeVerifier`.
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
from typing import Any, Dict, List, Optional

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.cli.main import _execute_headless_run
from zelyr.config import ZelyrConfig
from zelyr.engine.execution_ledger import parse_acceptance_criteria
from zelyr.engine.verifier import OutcomeVerifier


def _attach_stream_telemetry(orch: NativeOrchestrator) -> Dict[str, Any]:
    """Capture Ollama terminal chunk timings, token counters, and thinking tokens non-invasively."""
    telemetry: Dict[str, Any] = {
        "turn_metrics": [],
        "thinking_chunks": 0,
        "thinking_chars": 0,
    }
    orig_parser = orch._parse_stream_line_to_events

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

    orch._parse_stream_line_to_events = tapped_parser  # type: ignore[method-assign]
    return telemetry


async def _run_worker(spec: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        return {
            "exit_code": 1,
            "cli_json": {},
            "cli_json_valid": False,
            "worker_error": f"Invalid worker spec type: {type(spec).__name__}",
            "malformed_worker_output": True,
            "raw_stderr": "",
            "files_on_disk": {},
            "verifier_checks": [],
            "execution_ledger": {},
            "messages_transcript": [],
            "telemetry": {},
        }
    for req_key in ("endpoint", "model", "workspace_dir", "prompt"):
        val = spec.get(req_key)
        if not isinstance(val, str) or not val.strip():
            return {
                "exit_code": 1,
                "cli_json": {},
                "cli_json_valid": False,
                "worker_error": f"Invalid or missing required worker spec field '{req_key}'",
                "malformed_worker_output": True,
                "raw_stderr": "",
                "files_on_disk": {},
                "verifier_checks": [],
                "execution_ledger": {},
                "messages_transcript": [],
                "telemetry": {},
            }

    os.environ["ZELYR_TESTING"] = "1"
    os.environ["ZELYR_AUTOPILOT"] = "1"
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    prev_cwd = Path.cwd()
    ws = Path(spec["workspace_dir"]).resolve()
    ws.mkdir(parents=True, exist_ok=True)
    os.chdir(ws)

    try:
        cfg = ZelyrConfig(
            ollama_base_url=spec["endpoint"],
            ollama_num_ctx=int(spec.get("num_ctx", 8192)),
            ollama_keep_alive=spec.get("keep_alive", "5m"),
            ollama_read_timeout=float(spec.get("read_timeout", 1800.0)),
            ollama_model=spec["model"],
        )
        raw_allowed = spec.get("allowed_tools")
        allowed_tools = [str(t) for t in raw_allowed] if isinstance(raw_allowed, list) else None
        raw_ring = spec.get("caller_ring")
        caller_ring = int(raw_ring) if raw_ring is not None else None
        orch = NativeOrchestrator(config=cfg, caller_ring=caller_ring, allowed_tools=allowed_tools)
        orch.max_steps = int(spec.get("max_steps", 8))
        orch.max_replanning_attempts = int(spec.get("max_replanning", 2))
        telemetry = _attach_stream_telemetry(orch)

        model_options = dict(spec.get("model_options") or {})
        captured_payloads: List[Dict[str, Any]] = []
        orig_generate_stream = orch.generate_stream

        async def instrumented_generate_stream(payload: dict, *args: Any, **kwargs: Any):
            if model_options:
                opts = dict(payload.get("options") or {})
                opts.update(model_options)
                payload["options"] = opts
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

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exit_code = await _execute_headless_run(
                    prompt=spec["prompt"],
                    model_override=spec["model"],
                    output_json=True,
                    orchestrator=orch,
                    verify_spec=spec.get("verify_spec"),
                )
        except Exception as exc:
            worker_error = f"{type(exc).__name__}: {exc}"
            stderr_buf.write(f"\nWorker exception: {worker_error}\n")
            exit_code = 1

        raw_stdout = stdout_buf.getvalue()
        raw_stderr = stderr_buf.getvalue()

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

        ledger_dict = (
            orch.current_ledger.to_dict()
            if getattr(orch, "current_ledger", None) is not None
            else cli_json.get("execution_ledger", {})
        )
        messages_transcript = captured_payloads[0].get("messages", []) if captured_payloads else []

        return {
            "exit_code": exit_code,
            "cli_json": cli_json,
            "cli_json_valid": cli_json_valid,
            "worker_error": worker_error,
            "raw_stderr": raw_stderr,
            "files_on_disk": files_on_disk,
            "verifier_checks": verifier_checks,
            "execution_ledger": ledger_dict,
            "messages_transcript": messages_transcript,
            "telemetry": telemetry,
        }
    finally:
        try:
            os.chdir(prev_cwd)
        except Exception:
            pass


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
