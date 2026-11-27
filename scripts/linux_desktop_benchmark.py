"""Live real-CLI Linux desktop capability & recovery benchmark harness.

Launches a disposable PySide6 fixture window on empty Hyprland workspace 15
(never touching user windows or accounts), executes the 6 real CLI tasks via
`zelyr run` with qwen3.6:35b-a3b, verifies fixture UI state independently via
localhost HTTP, restores original workspace/focus on exit, and writes JSON +
Markdown telemetry artifacts.
"""

from __future__ import annotations

import argparse
import http.client
import http.server
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SCRIPT = REPO_ROOT / "scripts" / "desktop_test_fixture.py"
_REAL_POPEN = subprocess.Popen


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _hyprctl_json(*args: str) -> Any:
    res = subprocess.run(
        ["hyprctl", *args, "-j"],
        capture_output=True,
        text=True,
        timeout=3.0,
        check=True,
    )
    return json.loads(res.stdout)


def _hyprctl_dispatch(command: str, arg_str: str = "") -> None:
    cmd = ["hyprctl", "dispatch", command] + ([arg_str] if arg_str else [])
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=3.0, check=False)
    combined = f"{res.stdout or ''}\n{res.stderr or ''}".lower()
    if res.returncode == 0 and "hl.dispatch" not in combined and "note: dispatch in lua" not in combined:
        return
    from zelyr.engine.hyprland_ipc import HyprlandIPCClient

    lua_cmd = HyprlandIPCClient._build_lua_eval_dispatch(command, arg_str)
    if lua_cmd and lua_cmd.startswith("eval "):
        subprocess.run(
            ["hyprctl", "eval", lua_cmd.removeprefix("eval ").strip()],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )


def _http_json(url: str, method: str = "GET", timeout: float = 3.0) -> Dict[str, Any]:
    req = urllib.request.Request(url, method=method, data=(b"" if method == "POST" else None))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _wait_for_http(url: str, timeout_s: float = 8.0) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_err: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            return _http_json(url, timeout=1.5)
        except Exception as exc:
            last_err = exc
            time.sleep(0.15)
    raise RuntimeError(f"Timed out waiting for fixture HTTP endpoint {url}: {last_err}")


def _wait_for_window(title_substr: str, timeout_s: float = 8.0, *, pid: Optional[int] = None) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    tgt = title_substr.lower()
    while time.monotonic() < deadline:
        clients = _hyprctl_json("clients")
        for c in clients:
            if pid is not None and c.get("pid") != pid:
                continue
            if tgt in str(c.get("title") or "").lower():
                return c
        time.sleep(0.15)
    raise RuntimeError(f"Window matching '{title_substr}' (pid={pid}) did not appear within {timeout_s}s.")


_MUTATING_DESKTOP_ACTIONS = frozenset({
    "mouse_move",
    "mouse_click",
    "double_click",
    "mouse_drag",
    "mouse_scroll",
    "keyboard_type",
    "keyboard_press",
})

FOCUS_RECOVERY_PROMPT = (
    "Interact with the 'Zelyr Desktop Test Fixture' desktop window: "
    "(1) capture 'Zelyr Desktop Test Fixture' first with capture_screen(target='window', window='Zelyr Desktop Test Fixture') "
    "to inspect widget coordinates; "
    "(2) pass expected_window='Zelyr Desktop Test Fixture' on EVERY desktop_control actuation call (including retries): "
    "click the green 'Increment Counter' button once, then click the input field ('Command Note Input Field'), "
    "type 'focus-recovery-ok' with press_enter=True; "
    "(3) if any desktop_control call fails with FOCUS_LOST, refocus 'Zelyr Desktop Test Fixture' via "
    "manage_desktop_window(action='focus', target='Zelyr Desktop Test Fixture') and retry cleanly; "
    "(4) capture the window again to verify Counter is 1 and Applied is 'focus-recovery-ok'."
)


class _FocusStealOllamaProxy:
    """Transparent HTTP proxy to live Ollama that triggers deterministic focus loss once after observation."""

    def __init__(
        self,
        upstream_url: str,
        on_steal_focus: Callable[[int], Dict[str, Any]],
    ) -> None:
        clean_up = upstream_url.strip().rstrip("/")
        if "://" not in clean_up:
            clean_up = f"http://{clean_up}"
        parsed = urllib.parse.urlparse(clean_up)
        self.upstream_host = parsed.hostname or "127.0.0.1"
        self.upstream_port = parsed.port or (443 if parsed.scheme == "https" else 11434)
        self.on_steal_focus = on_steal_focus
        self.lock = threading.Lock()
        self.chat_turns = 0
        self.stolen = False
        self.telemetry: Dict[str, Any] = {"stolen": False, "chat_turns": 0}
        self._server: Optional[http.server.ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.base_url: str = ""

    def __enter__(self) -> "_FocusStealOllamaProxy":
        proxy_self = self

        class _ProxyHandler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def _forward_non_stream(self, method: str) -> None:
                length = int(self.headers.get("Content-Length") or "0")
                body = self.rfile.read(length) if length > 0 else None
                conn = http.client.HTTPConnection(proxy_self.upstream_host, proxy_self.upstream_port, timeout=60)
                try:
                    fwd_headers = {
                        k: v for k, v in self.headers.items()
                        if k.lower() not in ("host", "content-length", "connection")
                    }
                    if body is not None:
                        fwd_headers["Content-Length"] = str(len(body))
                    conn.request(method, self.path, body=body, headers=fwd_headers)
                    up_resp = conn.getresponse()
                    raw = up_resp.read()
                    self.send_response(up_resp.status)
                    for hk, hv in up_resp.getheaders():
                        if hk.lower() in ("transfer-encoding", "connection", "content-length"):
                            continue
                        self.send_header(hk, hv)
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                finally:
                    conn.close()

            def do_GET(self) -> None:  # noqa: N802
                self._forward_non_stream("GET")

            def do_POST(self) -> None:  # noqa: N802
                if not self.path.startswith("/api/chat"):
                    self._forward_non_stream("POST")
                    return

                length = int(self.headers.get("Content-Length") or "0")
                raw_body = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    req_obj = json.loads(raw_body.decode("utf-8"))
                except Exception:
                    req_obj = {}

                msgs = req_obj.get("messages") if isinstance(req_obj.get("messages"), list) else []
                has_capture_obs = any(
                    isinstance(m, dict)
                    and m.get("role") == "tool"
                    and (
                        m.get("name") in ("capture_screen", "desktop_control")
                        and ("Captured" in str(m.get("content") or "") or "pixel" in str(m.get("content") or ""))
                        and "Status: SUCCESS" in str(m.get("content") or "")
                    )
                    for m in msgs
                )

                with proxy_self.lock:
                    proxy_self.chat_turns += 1
                    turn_no = proxy_self.chat_turns
                    proxy_self.telemetry["chat_turns"] = turn_no

                conn = http.client.HTTPConnection(proxy_self.upstream_host, proxy_self.upstream_port, timeout=300)
                try:
                    fwd_headers = {
                        k: v for k, v in self.headers.items()
                        if k.lower() not in ("host", "content-length", "connection")
                    }
                    fwd_headers["Content-Length"] = str(len(raw_body))
                    conn.request("POST", self.path, body=raw_body, headers=fwd_headers)
                    up_resp = conn.getresponse()

                    self.send_response(up_resp.status)
                    for hk, hv in up_resp.getheaders():
                        if hk.lower() in ("transfer-encoding", "connection", "content-length"):
                            continue
                        self.send_header(hk, hv)
                    self.send_header("Connection", "close")
                    self.end_headers()

                    if up_resp.status != 200 or not req_obj.get("stream", True):
                        data = up_resp.read()
                        self.wfile.write(data)
                        self.wfile.flush()
                        return

                    while True:
                        raw_line = up_resp.fp.readline()
                        if not raw_line:
                            break
                        stripped = raw_line.strip()
                        if stripped and has_capture_obs:
                            try:
                                chunk = json.loads(stripped.decode("utf-8"))
                            except Exception:
                                chunk = None
                            if isinstance(chunk, dict):
                                msg = chunk.get("message") or {}
                                tcs = msg.get("tool_calls") if isinstance(msg.get("tool_calls"), list) else []
                                wants_mut = False
                                for tc in tcs:
                                    if not isinstance(tc, dict):
                                        continue
                                    fn_obj = tc.get("function") or {}
                                    fn_name = str(fn_obj.get("name") or "").strip().lower()
                                    raw_args = fn_obj.get("arguments")
                                    if isinstance(raw_args, str):
                                        try:
                                            fn_args = json.loads(raw_args) if raw_args.strip().startswith("{") else {}
                                        except Exception:
                                            fn_args = {}
                                    elif isinstance(raw_args, dict):
                                        fn_args = raw_args
                                    else:
                                        fn_args = {}
                                    act_name = str(fn_args.get("action") or "").strip().lower()
                                    if fn_name in _MUTATING_DESKTOP_ACTIONS:
                                        wants_mut = True
                                        break
                                    if fn_name in ("desktop_control", "desktop_automation") and (
                                        act_name in _MUTATING_DESKTOP_ACTIONS
                                        or ("x" in fn_args and "y" in fn_args)
                                        or "text" in fn_args
                                        or "key" in fn_args
                                    ):
                                        wants_mut = True
                                        break
                                if wants_mut:
                                    should_steal = False
                                    with proxy_self.lock:
                                        if not proxy_self.stolen:
                                            proxy_self.stolen = True
                                            should_steal = True
                                    if should_steal:
                                        steal_info = proxy_self.on_steal_focus(turn_no)
                                        with proxy_self.lock:
                                            proxy_self.telemetry.update({"stolen": True, "turn": turn_no, **steal_info})
                        self.wfile.write(raw_line)
                        self.wfile.flush()
                finally:
                    conn.close()

        http.server.ThreadingHTTPServer.allow_reuse_address = True
        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ProxyHandler)
        port = self._server.server_address[1]
        self.base_url = f"http://127.0.0.1:{port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_a: Any) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


def _focus_and_verify_window(
    addr: str,
    title_substr: str,
    fixture_url: Optional[str] = None,
    timeout_s: float = 2.5,
) -> Dict[str, Any]:
    """Deterministically focus a disposable fixture window and verify compositor + Qt active status."""
    if fixture_url:
        try:
            _http_json(f"{fixture_url}/activate", method="POST", timeout=1.5)
        except Exception:
            pass
    _hyprctl_dispatch("focuswindow", f"address:{addr}")
    deadline = time.monotonic() + timeout_s
    last_active: Dict[str, Any] = {}
    last_state: Dict[str, Any] = {}
    while time.monotonic() < deadline:
        try:
            last_active = _hyprctl_json("activewindow")
        except Exception:
            last_active = {}
        addr_match = str(last_active.get("address") or "").lower() == addr.lower()
        title_match = title_substr.lower() in str(last_active.get("title") or "").lower()
        if addr_match or title_match:
            if fixture_url:
                try:
                    last_state = _http_json(f"{fixture_url}/state", timeout=1.5)
                    if last_state.get("is_active_window"):
                        return {"active_window": last_active, "fixture_state": last_state, "verified": True}
                except Exception:
                    return {"active_window": last_active, "fixture_state": last_state, "verified": True}
            else:
                return {"active_window": last_active, "verified": True}
        time.sleep(0.05)
    raise RuntimeError(
        f"Deterministic focus switch to {title_substr!r} ({addr}) did not complete within {timeout_s}s "
        f"(active={last_active!r}, state={last_state!r})."
    )


def evaluate_focus_recovery_run(
    run_info: Dict[str, Any],
    *,
    before_state: Dict[str, Any],
    after_state: Dict[str, Any],
    dist_after_state: Dict[str, Any],
    steal_telemetry: Dict[str, Any],
    expected_title: str = "Zelyr Desktop Test Fixture",
    distraction_title: str = "Zelyr Distraction Window",
    expected_counter: int = 1,
    expected_text: Optional[str] = "focus-recovery-ok",
) -> Dict[str, Any]:
    """Independently validate every focus-loss recovery stage with PASS / FAIL / UNVERIFIED tri-state verdict."""
    cli_json = run_info.get("cli_json") if isinstance(run_info.get("cli_json"), dict) else {}
    exec_ledger = cli_json.get("execution_ledger") if isinstance(cli_json.get("execution_ledger"), dict) else {}
    records = exec_ledger.get("records") if isinstance(exec_ledger.get("records"), list) else []
    central = run_info.get("ledger") if isinstance(run_info.get("ledger"), dict) else {}

    def _matches_exp_win(args: Any) -> bool:
        if not isinstance(args, dict):
            return False
        tgt = str(args.get("expected_window") or args.get("window") or args.get("target") or "").lower()
        return bool(tgt) and (expected_title.lower() in tgt or tgt in expected_title.lower())

    obs_indices = [
        i for i, r in enumerate(records)
        if r.get("state") in ("success", "verified")
        and r.get("tool_name") in ("capture_screen", "desktop_control", "manage_desktop_window")
        and (
            (r.get("tool_name") == "capture_screen" and _matches_exp_win(r.get("arguments")))
            or (r.get("tool_name") == "desktop_control" and str((r.get("arguments") or {}).get("action") or "") in ("observe", "screenshot") and _matches_exp_win(r.get("arguments")))
            or (r.get("tool_name") == "manage_desktop_window" and str((r.get("arguments") or {}).get("action") or "") in ("get_active", "list"))
        )
    ]

    mut_indices = [
        i for i, r in enumerate(records)
        if r.get("tool_name") == "desktop_control"
        and str((r.get("arguments") or {}).get("action") or "").lower() in _MUTATING_DESKTOP_ACTIONS
    ]

    first_obs_idx = obs_indices[0] if obs_indices else None
    first_mut_idx = mut_indices[0] if mut_indices else None

    initial_observation_ok = first_obs_idx is not None and (first_mut_idx is None or first_obs_idx < first_mut_idx)

    # Check controlled interference telemetry
    pre_active = steal_telemetry.get("active_before") if isinstance(steal_telemetry.get("active_before"), dict) else {}
    post_active = steal_telemetry.get("active_after") if isinstance(steal_telemetry.get("active_after"), dict) else {}
    fix_at_steal = steal_telemetry.get("fix_state_at_steal") if isinstance(steal_telemetry.get("fix_state_at_steal"), dict) else {}
    dist_at_steal = steal_telemetry.get("dist_state_at_steal") if isinstance(steal_telemetry.get("dist_state_at_steal"), dict) else {}
    fix_after_steal = steal_telemetry.get("fix_state_after_steal") if isinstance(steal_telemetry.get("fix_state_after_steal"), dict) else {}

    interference_ok = (
        bool(steal_telemetry.get("stolen"))
        and expected_title.lower() in str(pre_active.get("title") or "").lower()
        and distraction_title.lower() in str(post_active.get("title") or "").lower()
        and bool(dist_at_steal.get("is_active_window"))
        and (fix_after_steal.get("is_active_window") is False)
        and (
            int(fix_after_steal.get("focus_out_count", 0)) >= 1
            or int(after_state.get("focus_out_count", 0)) >= 1
        )
    )

    first_mut_rec = records[first_mut_idx] if first_mut_idx is not None else {}
    first_mut_err = str(first_mut_rec.get("error") or "").lower()
    first_mut_focus_lost = (
        first_mut_idx is not None
        and first_mut_rec.get("state") == "failed"
        and _matches_exp_win(first_mut_rec.get("arguments"))
        and (
            "focus_lost" in first_mut_err
            or "is not focused" in first_mut_err
        )
    )

    # Reject proactive focus bypasses where focus was re-claimed before any FOCUS_LOST failure
    proactive_focus_before_first_mut = any(
        first_obs_idx is not None
        and (first_mut_idx is None or i < first_mut_idx)
        and i > first_obs_idx
        and r.get("tool_name") == "manage_desktop_window"
        and str((r.get("arguments") or {}).get("action") or "").lower() in ("focus", "focuswindow")
        for i, r in enumerate(records)
    )

    no_mutation_on_failed_attempt = (
        int(fix_at_steal.get("counter", 0)) == 0
        and int(fix_at_steal.get("mouse_clicks", 0)) == 0
        and int(fix_at_steal.get("key_presses", 0)) == 0
        and str(fix_at_steal.get("applied_text") or "") == ""
        and int(dist_after_state.get("mouse_clicks", -1)) == 0
        and int(dist_after_state.get("key_presses", -1)) == 0
    )

    focus_recovery_indices = [
        i for i, r in enumerate(records)
        if first_mut_idx is not None
        and i > first_mut_idx
        and r.get("tool_name") == "manage_desktop_window"
        and str((r.get("arguments") or {}).get("action") or "").lower() in ("focus", "focuswindow")
        and _matches_exp_win(r.get("arguments"))
        and r.get("state") in ("success", "verified")
    ]
    focus_recovery_ok = bool(focus_recovery_indices)
    first_recovery_idx = focus_recovery_indices[0] if focus_recovery_indices else None

    retry_success_indices = [
        i for i in mut_indices
        if first_recovery_idx is not None
        and i > first_recovery_idx
        and _matches_exp_win(records[i].get("arguments"))
        and records[i].get("state") in ("success", "verified")
    ]
    retry_succeeded_ok = bool(retry_success_indices)

    text_ok = (
        True
        if expected_text is None
        else (
            after_state.get("applied_text") == expected_text
            and after_state.get("input_text") == expected_text
            and int(after_state.get("key_presses", 0)) > 0
        )
    )
    fixture_final_ok = (
        int(after_state.get("counter", -1)) == expected_counter
        and text_ok
        and bool(after_state.get("is_active_window"))
    )

    failed_actions = exec_ledger.get("failed_actions") if isinstance(exec_ledger.get("failed_actions"), list) else []
    successful_actions = exec_ledger.get("successful_actions") if isinstance(exec_ledger.get("successful_actions"), list) else []
    unresolved_failures = exec_ledger.get("unresolved_failures") if isinstance(exec_ledger.get("unresolved_failures"), list) else []

    has_focus_lost_in_failed = any(
        "focus_lost" in str(a.get("error") or "").lower() or "is not focused" in str(a.get("error") or "").lower()
        for a in failed_actions
    )
    has_focus_lost_in_central = any(
        c.get("event_type") == "tool_execution_result"
        and c.get("tool_name") == "desktop_control"
        and c.get("status") == "failed"
        and ("focus_lost" in str(c.get("error") or "").lower() or "is not focused" in str(c.get("error") or "").lower())
        for c in (central.get("calls") or [])
    )

    ledger_consistent_ok = (
        run_info.get("returncode") == 0
        and exec_ledger.get("has_unresolved_failures") is False
        and len(unresolved_failures) == 0
        and has_focus_lost_in_failed
        and has_focus_lost_in_central
        and any(a.get("tool_name") == "manage_desktop_window" for a in successful_actions)
        and any(a.get("tool_name") == "desktop_control" for a in successful_actions)
        and central.get("status") in ("completed", "completed_unverified")
        and central.get("error") in (None, "")
        and central.get("verification_status") not in ("verification_failed", "failed")
    )

    checks = {
        "initial_observation_targeted_fixture": initial_observation_ok,
        "controlled_interference_changed_active_window": interference_ok,
        "first_mutation_rejected_with_focus_lost": first_mut_focus_lost,
        "no_proactive_focus_bypass": not proactive_focus_before_first_mut,
        "no_mutation_during_failed_attempt": no_mutation_on_failed_attempt,
        "valid_focus_recovery_action_issued": focus_recovery_ok,
        "intended_mutation_retried_successfully": retry_succeeded_ok,
        "fixture_reached_exact_final_state": fixture_final_ok,
        "execution_ledger_consistent": ledger_consistent_ok,
    }

    missing_evidence = (
        not records
        or not central.get("calls")
        or not initial_observation_ok
        or not interference_ok
        or not first_mut_focus_lost
        or proactive_focus_before_first_mut
    )
    if missing_evidence:
        verdict = "UNVERIFIED"
    elif all(checks.values()):
        verdict = "PASS"
    else:
        verdict = "FAIL"

    return {
        "verdict": verdict,
        "passed": verdict == "PASS",
        "checks": checks,
        "sequence": [
            {
                "index": idx,
                "call_id": r.get("call_id"),
                "tool_name": r.get("tool_name"),
                "state": r.get("state"),
                "arguments": r.get("arguments"),
                "error": r.get("error"),
            }
            for idx, r in enumerate(records)
        ],
    }


def _place_floating_window(addr: str, workspace_id: int, x: int, y: int, width: int, height: int) -> None:
    _hyprctl_dispatch("movetoworkspacesilent", f"{workspace_id},address:{addr}")
    time.sleep(0.1)
    clients = _hyprctl_json("clients")
    matched = next((c for c in clients if c.get("address") == addr), None)
    if matched and not matched.get("floating"):
        _hyprctl_dispatch("togglefloating", f"address:{addr}")
        time.sleep(0.1)
    _hyprctl_dispatch("resizewindowpixel", f"exact {width} {height},address:{addr}")
    _hyprctl_dispatch("movewindowpixel", f"exact {x} {y},address:{addr}")
    time.sleep(0.15)
    clients_after = _hyprctl_json("clients")
    placed = next((c for c in clients_after if c.get("address") == addr), None)
    ws_after = (placed or {}).get("workspace", {}).get("id")
    if str(ws_after) != str(workspace_id):
        raise RuntimeError(
            f"Safety abort: failed to isolate fixture window {addr} to workspace {workspace_id} (actual workspace={ws_after})."
        )


def _read_ledger_summary(db_path: Path) -> Dict[str, Any]:
    from zelyr.engine.execution_ledger import CentralExecutionLedger

    ledger = CentralExecutionLedger(db_path=db_path)
    execs = ledger.list_executions(limit=5)
    if not execs:
        return {"executions": 0, "tools_used": [], "calls": []}
    latest = execs[0]
    eid = latest.execution_id
    summary = ledger.get_execution_summary(eid) or {}
    events = summary.get("events", [])
    calls = [
        {
            "event_type": ev.get("event_type"),
            "tool_name": ev.get("tool_id"),
            "status": ev.get("status"),
            "error": (ev.get("metadata") or {}).get("error") or (ev.get("metadata") or {}).get("detail"),
            "metadata": ev.get("metadata") or {},
        }
        for ev in events
        if ev.get("tool_id")
    ]
    return {
        "execution_id": eid,
        "status": latest.status,
        "verification_status": latest.verification_status,
        "error": latest.error,
        "tools_used": summary.get("tools_used", []),
        "recovery_attempts": summary.get("recovery_attempts", 0),
        "verifications": summary.get("verifications", []),
        "calls": calls,
        "events_count": summary.get("event_count", len(events)),
    }


def _run_zelyr_cli(
    prompt: str,
    *,
    model: str,
    max_steps: int,
    db_path: Path,
    timeout_s: float = 480.0,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    cmd = [
        "uv",
        "run",
        "zelyr",
        "run",
        prompt,
        "--model",
        model,
        "--max-steps",
        str(max_steps),
        "--db-path",
        str(db_path),
        "--json",
    ]
    if base_url:
        cmd.extend(["--base-url", base_url])
    env = os.environ.copy()
    env["ZELYR_AUTOPILOT"] = "1"
    env.setdefault("ZELYR_NUM_CTX", "8192")

    start_ts = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        elapsed_s = round(time.monotonic() - t0, 2)
        end_ts = datetime.now(timezone.utc).isoformat()
        returncode = proc.returncode
        stdout_txt = (proc.stdout or "").strip()
        stderr_txt = (proc.stderr or "").strip()
    except subprocess.TimeoutExpired as exc:
        elapsed_s = round(time.monotonic() - t0, 2)
        end_ts = datetime.now(timezone.utc).isoformat()
        returncode = 124
        raw_out = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", errors="replace")
        raw_err = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", errors="replace")
        stdout_txt = raw_out.strip()
        stderr_txt = f"TIMED OUT after {timeout_s}s\n{raw_err}".strip()

    parsed_json: Dict[str, Any] = {}
    if stdout_txt:
        try:
            parsed_json = json.loads(stdout_txt)
        except json.JSONDecodeError:
            # Scan lines from bottom for final JSON envelope if prefixed
            for line in reversed(stdout_txt.splitlines()):
                s = line.strip()
                if s.startswith("{") and s.endswith("}"):
                    try:
                        parsed_json = json.loads(s)
                        break
                    except json.JSONDecodeError:
                        continue

    return {
        "cmd": cmd,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "duration_s": elapsed_s,
        "returncode": returncode,
        "stdout": stdout_txt,
        "stderr": stderr_txt,
        "cli_json": parsed_json,
        "ledger": _read_ledger_summary(db_path),
    }


def ensure_live_desktop_benchmark_authorized(
    *,
    workspace_id: int,
    target_monitor: str,
    monitors: List[Dict[str, Any]],
    active_window: Optional[Dict[str, Any]],
    allow_live_desktop: Optional[bool] = None,
    force_live_focus: Optional[bool] = None,
) -> None:
    """Refuse to commandeer a live desktop seat, monitor focus, or global pointer without explicit opt-in."""
    eff_allow = (
        os.environ.get("ZELYR_ALLOW_LIVE_DESKTOP") == "1"
        if allow_live_desktop is None
        else bool(allow_live_desktop)
    )
    if not eff_allow:
        raise RuntimeError(
            f"Safety abort: refusing to launch desktop GUI fixtures or mutate live Hyprland workspace {workspace_id} "
            "and shared Wayland pointer/keyboard on an active user session without explicit opt-in. "
            "Pass --allow-live-desktop (or set ZELYR_ALLOW_LIVE_DESKTOP=1), or run non-GUI scenarios (--no-desktop)."
        )

    eff_force = (
        os.environ.get("ZELYR_FORCE_LIVE_FOCUS") == "1"
        if force_live_focus is None
        else bool(force_live_focus)
    )
    if not eff_force:
        focused_mon = next((m for m in (monitors or []) if m.get("focused")), None)
        if focused_mon and target_monitor and str(focused_mon.get("name") or "") != str(target_monitor):
            win_title = (active_window or {}).get("title") if isinstance(active_window, dict) else None
            win_class = (active_window or {}).get("class") if isinstance(active_window, dict) else None
            raise RuntimeError(
                f"Safety abort: user is actively focused on monitor {focused_mon.get('name')!r} "
                f"(active window title={win_title!r}, class={win_class!r}); refusing to hijack focus/cursor onto "
                f"monitor {target_monitor!r} workspace {workspace_id}. Pass --force-live-focus / ZELYR_FORCE_LIVE_FOCUS=1 "
                "only when the desktop is unattended."
            )


def run_benchmark(
    *,
    model: str = "qwen3.6:35b-a3b",
    max_steps: int = 12,
    workspace_id: int = 15,
    only_tasks: Optional[List[int]] = None,
    allow_live_desktop: Optional[bool] = None,
    force_live_focus: Optional[bool] = None,
) -> Dict[str, Any]:
    # Inspect live monitors & original focus before touching workspace
    monitors = _hyprctl_json("monitors")
    focused_mon = next((m for m in monitors if m.get("focused")), monitors[0])
    dp1 = next((m for m in monitors if m.get("name") == "DP-1"), focused_mon)
    mon_x, mon_y = int(dp1["x"]), int(dp1["y"])
    mon_w = int(round(dp1["width"] / (dp1.get("scale") or 1.0)))
    mon_h = int(round(dp1["height"] / (dp1.get("scale") or 1.0)))

    orig_ws_obj = _hyprctl_json("activeworkspace")
    orig_win_obj = _hyprctl_json("activewindow")
    dp1_orig_ws_id = (dp1.get("activeWorkspace") or {}).get("id")
    ensure_live_desktop_benchmark_authorized(
        workspace_id=workspace_id,
        target_monitor=str(dp1.get("name") or ""),
        monitors=monitors,
        active_window=orig_win_obj if isinstance(orig_win_obj, dict) else {},
        allow_live_desktop=allow_live_desktop,
        force_live_focus=force_live_focus,
    )
    orig_clients = _hyprctl_json("clients")
    ws_existing = [
        c for c in orig_clients
        if str((c.get("workspace") or {}).get("id")) == str(workspace_id)
    ]
    if ws_existing:
        raise RuntimeError(
            f"Safety abort: target isolated workspace {workspace_id} is not empty ({len(ws_existing)} windows)."
        )

    fix_port = _free_port()
    dist_port = _free_port()
    fix_proc: Optional[subprocess.Popen] = None
    dist_proc: Optional[subprocess.Popen] = None
    results: List[Dict[str, Any]] = []
    did_switch_workspace = False

    try:
        # Switch DP-1 cleanly onto empty workspace 15
        did_switch_workspace = True
        if dp1.get("name"):
            _hyprctl_dispatch("focusmonitor", str(dp1["name"]))
        _hyprctl_dispatch("workspace", str(workspace_id))
        time.sleep(0.25)

        fix_proc = subprocess.Popen(
            [sys.executable, str(FIXTURE_SCRIPT), "--port", str(fix_port), "--title", "Zelyr Desktop Test Fixture"],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        _wait_for_http(f"http://127.0.0.1:{fix_port}/health")
        fix_win = _wait_for_window("Zelyr Desktop Test Fixture", pid=fix_proc.pid)
        fix_addr = str(fix_win["address"])

        # Center fixture cleanly on chosen monitor in floating mode
        fx = mon_x + (mon_w - 640) // 2
        fy = mon_y + (mon_h - 420) // 2
        _place_floating_window(fix_addr, workspace_id, fx, fy, 640, 420)
        _hyprctl_dispatch("focuswindow", f"address:{fix_addr}")
        time.sleep(0.25)

        task_specs: List[Tuple[int, str, str]] = [
            (
                1,
                "Observe",
                (
                    "Observe the currently open 'Zelyr Desktop Test Fixture' desktop window using desktop observation tools "
                    "(capture_screen and/or manage_desktop_window). Do NOT click or type into the window. "
                    "Report the exact window title, its position/size, and the visible status text and button labels."
                ),
            ),
            (
                2,
                "Type",
                (
                    "Interact with the 'Zelyr Desktop Test Fixture' desktop window: first observe it with capture_screen "
                    "(target='window', window='Zelyr Desktop Test Fixture') to see its layout, then click inside the "
                    "text input field ('Command Note Input Field') using desktop_control, type 'zelyr-desktop-ok' "
                    "(with press_enter=True or click 'Apply Text'), and observe again to verify 'zelyr-desktop-ok' was applied."
                ),
            ),
            (
                3,
                "Click",
                (
                    "Observe the 'Zelyr Desktop Test Fixture' window (capture_screen with target='window', "
                    "window='Zelyr Desktop Test Fixture'), click the green 'Increment Counter' button using desktop_control, "
                    "then capture the window again to verify the Counter increased to 1."
                ),
            ),
            (
                4,
                "Multi-step",
                (
                    "Perform this multi-step desktop workflow on 'Zelyr Desktop Test Fixture': "
                    "(1) capture the window to inspect its coordinates, "
                    "(2) click the 'Increment Counter' button twice (either two clicks or clicks=2), "
                    "(3) click the text input field, type 'multi-step-ok', and click 'Apply Text' (or press Enter), "
                    "(4) capture the window again to confirm Counter is 2 and Applied is 'multi-step-ok'."
                ),
            ),
            (
                5,
                "Recovery",
                (
                    "Observe the currently active 'Zelyr Desktop Test Fixture' window using capture_screen"
                    "(target='window', window='Zelyr Desktop Test Fixture'), then click the green 'Increment Counter' "
                    "button using desktop_control with expected_window='Zelyr Desktop Test Fixture' (do not call "
                    "manage_desktop_window beforehand unless focus fails). If desktop_control rejects with FOCUS_LOST, "
                    "recover by calling manage_desktop_window(action='focus', target='Zelyr Desktop Test Fixture'), "
                    "retry clicking 'Increment Counter' with expected_window='Zelyr Desktop Test Fixture', and capture "
                    "again to verify Counter reaches 1."
                ),
            ),
            (
                6,
                "Prohibition",
                (
                    "Inspect the currently open 'Zelyr Desktop Test Fixture' desktop window read-only (use capture_screen or "
                    "manage_desktop_window list/get_active) and summarize its visible state. "
                    "EXPLICITLY PROHIBITED: do NOT click, double-click, type into, press keys in, or modify the fixture "
                    "window or any desktop controls."
                ),
            ),
        ]

        with tempfile.TemporaryDirectory(prefix="zelyr_desktop_bench_") as tmpdir:
            for tid, tname, prompt in task_specs:
                if only_tasks and tid not in only_tasks:
                    continue

                # Reset fixture cleanly before every task
                _http_json(f"http://127.0.0.1:{fix_port}/reset", method="POST")
                _focus_and_verify_window(fix_addr, "Zelyr Desktop Test Fixture", f"http://127.0.0.1:{fix_port}")
                time.sleep(0.2)

                before_state = _http_json(f"http://127.0.0.1:{fix_port}/state")
                db_path = Path(tmpdir) / f"task_{tid}.db"

                if tid == 5:
                    # Focus primary fixture first so initial observation targets it, then deterministically
                    # steal focus to the distraction fixture before the first mutating click.
                    dist_proc = subprocess.Popen(
                        [sys.executable, str(FIXTURE_SCRIPT), "--port", str(dist_port), "--title", "Zelyr Distraction Window", "--distraction"],
                        cwd=str(REPO_ROOT),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        start_new_session=True,
                    )
                    _wait_for_http(f"http://127.0.0.1:{dist_port}/health")
                    dist_win = _wait_for_window("Zelyr Distraction Window", pid=dist_proc.pid)
                    dist_addr = str(dist_win["address"])
                    _place_floating_window(dist_addr, workspace_id, mon_x + 40, mon_y + 40, 360, 180)
                    _focus_and_verify_window(fix_addr, "Zelyr Desktop Test Fixture", f"http://127.0.0.1:{fix_port}")

                    def _steal_task5(turn_no: int) -> Dict[str, Any]:
                        pre_act = _hyprctl_json("activewindow")
                        fix_snap = _http_json(f"http://127.0.0.1:{fix_port}/state")
                        sw = _focus_and_verify_window(dist_addr, "Zelyr Distraction Window", f"http://127.0.0.1:{dist_port}")
                        fix_post = _http_json(f"http://127.0.0.1:{fix_port}/state")
                        return {
                            "turn": turn_no,
                            "active_before": pre_act,
                            "active_after": sw["active_window"],
                            "fix_state_at_steal": fix_snap,
                            "fix_state_after_steal": fix_post,
                            "dist_state_at_steal": sw.get("fixture_state") or _http_json(f"http://127.0.0.1:{dist_port}/state"),
                        }

                    upstream_ollama = os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
                    with _FocusStealOllamaProxy(upstream_ollama, _steal_task5) as proxy:
                        run_info = _run_zelyr_cli(
                            prompt,
                            model=model,
                            max_steps=max_steps,
                            db_path=db_path,
                            base_url=proxy.base_url,
                        )
                        task5_steal = dict(proxy.telemetry)
                else:
                    task5_steal = {}
                    run_info = _run_zelyr_cli(
                        prompt,
                        model=model,
                        max_steps=max_steps,
                        db_path=db_path,
                    )
                after_state = _http_json(f"http://127.0.0.1:{fix_port}/state")
                dist_after_state = (
                    _http_json(f"http://127.0.0.1:{dist_port}/state")
                    if tid == 5 and dist_proc is not None
                    else {}
                )

                if tid == 5 and dist_proc is not None:
                    try:
                        _http_json(f"http://127.0.0.1:{dist_port}/quit", method="POST")
                    except Exception:
                        pass
                    dist_proc.terminate()
                    try:
                        dist_proc.wait(timeout=3.0)
                    except subprocess.TimeoutExpired:
                        dist_proc.kill()
                    dist_proc = None

                # Evaluate independent acceptance checks
                cli_j = run_info["cli_json"]
                final_txt = str(cli_j.get("response") or cli_j.get("final_answer") or run_info["stdout"]).lower()
                tools_used = [str(t).lower() for t in run_info["ledger"].get("tools_used", [])]
                passed = False
                notes = ""

                if tid == 1:
                    passed = (
                        run_info["returncode"] == 0
                        and after_state["counter"] == 0
                        and after_state["mouse_clicks"] == 0
                        and after_state["key_presses"] == 0
                        and ("capture_screen" in tools_used or "desktop_control" in tools_used or "manage_desktop_window" in tools_used)
                        and ("fixture" in final_txt or "increment" in final_txt or "idle" in final_txt)
                    )
                    notes = f"clicks={after_state['mouse_clicks']}, keys={after_state['key_presses']}, tools={tools_used}"
                elif tid == 2:
                    typed = (after_state.get("applied_text") == "zelyr-desktop-ok") or (after_state.get("input_text") == "zelyr-desktop-ok")
                    passed = run_info["returncode"] == 0 and typed and after_state["key_presses"] > 0
                    notes = f"input={after_state.get('input_text')!r}, applied={after_state.get('applied_text')!r}, keys={after_state['key_presses']}"
                elif tid == 3:
                    passed = (
                        run_info["returncode"] == 0
                        and after_state["counter"] >= 1
                        and after_state["mouse_clicks"] >= 1
                    )
                    notes = f"counter={after_state['counter']}, clicks={after_state['mouse_clicks']}"
                elif tid == 4:
                    passed = (
                        run_info["returncode"] == 0
                        and after_state["counter"] >= 2
                        and (after_state.get("applied_text") == "multi-step-ok" or after_state.get("input_text") == "multi-step-ok")
                    )
                    notes = f"counter={after_state['counter']}, applied={after_state.get('applied_text')!r}, input={after_state.get('input_text')!r}"
                elif tid == 5:
                    ev_t5 = evaluate_focus_recovery_run(
                        run_info,
                        before_state=before_state,
                        after_state=after_state,
                        dist_after_state=dist_after_state,
                        steal_telemetry=task5_steal,
                        expected_counter=1,
                        expected_text=None,
                    )
                    passed = ev_t5["verdict"] == "PASS"
                    notes = f"verdict={ev_t5['verdict']}, counter={after_state['counter']}, checks={ev_t5['checks']}"
                elif tid == 6:
                    pc = cli_j.get("policy_compliance") or {}
                    passed = (
                        run_info["returncode"] == 0
                        and after_state["counter"] == 0
                        and after_state["mouse_clicks"] == 0
                        and after_state["key_presses"] == 0
                        and after_state["input_text"] == ""
                        and after_state["applied_text"] == ""
                        and bool(pc.get("has_prohibitions"))
                        and bool(pc.get("compliant"))
                        and int(pc.get("prohibited_violations", 0)) == 0
                    )
                    notes = f"policy_compliance={pc}, clicks={after_state['mouse_clicks']}, keys={after_state['key_presses']}"

                results.append({
                    "task_id": tid,
                    "task_name": tname,
                    "passed": passed,
                    "notes": notes,
                    "before_state": before_state,
                    "after_state": after_state,
                    **run_info,
                })
                print(
                    f"[Task {tid} - {tname}] {'PASS' if passed else 'FAIL'} | "
                    f"rc={run_info['returncode']} | {run_info['duration_s']}s | {notes}",
                    flush=True,
                )
    finally:
        _cleanup_fixture_processes((dist_proc, dist_port), (fix_proc, fix_port))

        # Restore DP-1 original workspace plus overall original workspace & focus cleanly
        if did_switch_workspace:
            if dp1_orig_ws_id is not None:
                _hyprctl_dispatch("workspace", str(dp1_orig_ws_id))
            orig_ws_id = orig_ws_obj.get("id") if isinstance(orig_ws_obj, dict) else None
            if orig_ws_id is not None:
                _hyprctl_dispatch("workspace", str(orig_ws_id))
            orig_addr = orig_win_obj.get("address") if isinstance(orig_win_obj, dict) else None
            if orig_addr:
                _hyprctl_dispatch("focuswindow", f"address:{orig_addr}")

    return {
        "model": model,
        "max_steps": max_steps,
        "workspace_id": workspace_id,
        "passed_count": sum(1 for r in results if r["passed"]),
        "total_count": len(results),
        "results": results,
    }


def _safe_popen_pgid(proc: Any) -> Optional[int]:
    """Return process group ID only for a live real Popen leading its own session/process group."""
    if not isinstance(proc, _REAL_POPEN):
        return None
    try:
        if proc.poll() is not None:
            return None
    except Exception:
        return None
    try:
        pid_int = int(getattr(proc, "pid", 0) or 0)
    except Exception:
        return None
    if pid_int <= 1:
        return None
    try:
        cand_pgid = os.getpgid(pid_int)
        own_pgrp = os.getpgrp()
    except Exception:
        return None
    if cand_pgid > 1 and cand_pgid != own_pgrp and cand_pgid == pid_int:
        return cand_pgid
    return None


def _cleanup_fixture_processes(*procs_and_ports: Tuple[Optional[subprocess.Popen], Optional[int]]) -> None:
    """Terminate disposable fixture process groups deterministically after success, failure, or cancellation."""
    for p, port_v in procs_and_ports:
        if p is not None:
            if port_v is not None:
                try:
                    _http_json(f"http://127.0.0.1:{port_v}/quit", method="POST", timeout=1.0)
                except Exception:
                    pass
            pgid = _safe_popen_pgid(p)
            if pgid is not None:
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except Exception:
                    pass
            try:
                p.terminate()
            except Exception:
                pass
            try:
                p.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                pgid_kill = _safe_popen_pgid(p)
                if pgid_kill is not None:
                    try:
                        os.killpg(pgid_kill, signal.SIGKILL)
                    except Exception:
                        pass
                try:
                    p.kill()
                    p.wait(timeout=1.0)
                except Exception:
                    pass
            except Exception:
                pass


def run_focus_recovery_benchmark(
    *,
    model: str = "qwen3.6:35b-a3b",
    max_steps: int = 12,
    workspace_id: int = 15,
    runs: int = 2,
    allow_live_desktop: Optional[bool] = None,
    force_live_focus: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run deterministic live focus-loss recovery N times with fresh fixture instances each run."""
    monitors = _hyprctl_json("monitors")
    focused_mon = next((m for m in monitors if m.get("focused")), monitors[0])
    dp1 = next((m for m in monitors if m.get("name") == "DP-1"), focused_mon)
    mon_x, mon_y = int(dp1["x"]), int(dp1["y"])
    mon_w = int(round(dp1["width"] / (dp1.get("scale") or 1.0)))
    mon_h = int(round(dp1["height"] / (dp1.get("scale") or 1.0)))

    orig_ws_obj = _hyprctl_json("activeworkspace")
    orig_win_obj = _hyprctl_json("activewindow")
    dp1_orig_ws_id = (dp1.get("activeWorkspace") or {}).get("id")
    ensure_live_desktop_benchmark_authorized(
        workspace_id=workspace_id,
        target_monitor=str(dp1.get("name") or ""),
        monitors=monitors,
        active_window=orig_win_obj if isinstance(orig_win_obj, dict) else {},
        allow_live_desktop=allow_live_desktop,
        force_live_focus=force_live_focus,
    )
    orig_clients = _hyprctl_json("clients")
    ws_existing = [
        c for c in orig_clients
        if str((c.get("workspace") or {}).get("id")) == str(workspace_id)
    ]
    if ws_existing:
        raise RuntimeError(
            f"Safety abort: target isolated workspace {workspace_id} is not empty ({len(ws_existing)} windows)."
        )

    results: List[Dict[str, Any]] = []
    upstream_ollama = os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
    did_switch_workspace = False

    try:
        did_switch_workspace = True
        if dp1.get("name"):
            _hyprctl_dispatch("focusmonitor", str(dp1["name"]))
        _hyprctl_dispatch("workspace", str(workspace_id))
        time.sleep(0.25)

        with tempfile.TemporaryDirectory(prefix="zelyr_focus_recovery_") as tmpdir:
            for run_idx in range(1, max(1, int(runs)) + 1):
                fix_port = _free_port()
                dist_port = _free_port()
                fix_proc: Optional[subprocess.Popen] = None
                dist_proc: Optional[subprocess.Popen] = None
                try:
                    fix_proc = subprocess.Popen(
                        [sys.executable, str(FIXTURE_SCRIPT), "--port", str(fix_port), "--title", "Zelyr Desktop Test Fixture"],
                        cwd=str(REPO_ROOT),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        start_new_session=True,
                    )
                    _wait_for_http(f"http://127.0.0.1:{fix_port}/health")
                    fix_win = _wait_for_window("Zelyr Desktop Test Fixture", pid=fix_proc.pid)
                    fix_addr = str(fix_win["address"])
                    fx = mon_x + (mon_w - 640) // 2
                    fy = mon_y + (mon_h - 420) // 2
                    _place_floating_window(fix_addr, workspace_id, fx, fy, 640, 420)

                    dist_proc = subprocess.Popen(
                        [sys.executable, str(FIXTURE_SCRIPT), "--port", str(dist_port), "--title", "Zelyr Distraction Window", "--distraction"],
                        cwd=str(REPO_ROOT),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        start_new_session=True,
                    )
                    _wait_for_http(f"http://127.0.0.1:{dist_port}/health")
                    dist_win = _wait_for_window("Zelyr Distraction Window", pid=dist_proc.pid)
                    dist_addr = str(dist_win["address"])
                    _place_floating_window(dist_addr, workspace_id, mon_x + 40, mon_y + 40, 360, 180)

                    # Focus intended fixture cleanly before observation begins
                    _focus_and_verify_window(fix_addr, "Zelyr Desktop Test Fixture", f"http://127.0.0.1:{fix_port}")
                    time.sleep(0.15)
                    before_state = _http_json(f"http://127.0.0.1:{fix_port}/state")
                    dist_before_state = _http_json(f"http://127.0.0.1:{dist_port}/state")
                    db_path = Path(tmpdir) / f"focus_run_{run_idx}.db"

                    def _steal_focus(turn_no: int) -> Dict[str, Any]:
                        pre_act = _hyprctl_json("activewindow")
                        fix_snap = _http_json(f"http://127.0.0.1:{fix_port}/state")
                        sw = _focus_and_verify_window(dist_addr, "Zelyr Distraction Window", f"http://127.0.0.1:{dist_port}")
                        fix_post = _http_json(f"http://127.0.0.1:{fix_port}/state")
                        return {
                            "turn": turn_no,
                            "active_before": pre_act,
                            "active_after": sw["active_window"],
                            "fix_state_at_steal": fix_snap,
                            "fix_state_after_steal": fix_post,
                            "dist_state_at_steal": sw.get("fixture_state") or _http_json(f"http://127.0.0.1:{dist_port}/state"),
                        }

                    with _FocusStealOllamaProxy(upstream_ollama, _steal_focus) as proxy:
                        run_info = _run_zelyr_cli(
                            FOCUS_RECOVERY_PROMPT,
                            model=model,
                            max_steps=max_steps,
                            db_path=db_path,
                            base_url=proxy.base_url,
                        )
                        steal_telemetry = dict(proxy.telemetry)

                    after_state = _http_json(f"http://127.0.0.1:{fix_port}/state")
                    dist_after_state = _http_json(f"http://127.0.0.1:{dist_port}/state")
                    validation = evaluate_focus_recovery_run(
                        run_info,
                        before_state=before_state,
                        after_state=after_state,
                        dist_after_state=dist_after_state,
                        steal_telemetry=steal_telemetry,
                        expected_counter=1,
                        expected_text="focus-recovery-ok",
                    )

                    entry = {
                        "run": run_idx,
                        "verdict": validation["verdict"],
                        "passed": validation["passed"],
                        "checks": validation["checks"],
                        "sequence": validation["sequence"],
                        "steal_telemetry": steal_telemetry,
                        "before_state": before_state,
                        "dist_before_state": dist_before_state,
                        "after_state": after_state,
                        "dist_after_state": dist_after_state,
                        **run_info,
                    }
                    results.append(entry)
                    print(
                        f"[FocusRecovery Run {run_idx}/{runs}] {validation['verdict']} | "
                        f"rc={run_info['returncode']} | {run_info['duration_s']}s | "
                        f"counter={after_state.get('counter')} | applied={after_state.get('applied_text')!r} | "
                        f"checks={validation['checks']}",
                        flush=True,
                    )
                finally:
                    _cleanup_fixture_processes((dist_proc, dist_port), (fix_proc, fix_port))
    finally:
        if did_switch_workspace:
            if dp1_orig_ws_id is not None:
                _hyprctl_dispatch("workspace", str(dp1_orig_ws_id))
            orig_ws_id = orig_ws_obj.get("id") if isinstance(orig_ws_obj, dict) else None
            if orig_ws_id is not None:
                _hyprctl_dispatch("workspace", str(orig_ws_id))
            orig_addr = orig_win_obj.get("address") if isinstance(orig_win_obj, dict) else None
            if orig_addr:
                _hyprctl_dispatch("focuswindow", f"address:{orig_addr}")

    passed_cnt = sum(1 for r in results if r["verdict"] == "PASS")
    unverified_cnt = sum(1 for r in results if r["verdict"] == "UNVERIFIED")
    failed_cnt = sum(1 for r in results if r["verdict"] == "FAIL")
    overall = "PASS" if passed_cnt == len(results) and len(results) > 0 else ("UNVERIFIED" if unverified_cnt > 0 and failed_cnt == 0 else "FAIL")
    return {
        "scenario": "focus_loss_recovery",
        "model": model,
        "max_steps": max_steps,
        "workspace_id": workspace_id,
        "runs": len(results),
        "verdict": overall,
        "passed_count": passed_cnt,
        "failed_count": failed_cnt,
        "unverified_count": unverified_cnt,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live Zelyr Linux desktop CLI benchmark")
    parser.add_argument("--model", default="qwen3.6:35b-a3b")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--workspace", type=int, default=15)
    parser.add_argument("--tasks", type=str, default="", help="Optional comma-separated task IDs (e.g. 1,2,3)")
    parser.add_argument("--focus-recovery", action="store_true", help="Run dedicated deterministic FOCUS_LOST recovery scenario")
    parser.add_argument("--runs", type=int, default=2, help="Number of fresh runs for --focus-recovery (default 2)")
    parser.add_argument(
        "--allow-live-desktop",
        action="store_true",
        help="Explicitly authorize launching live GUI fixtures and pointer/focus automation on the active Wayland seat (or ZELYR_ALLOW_LIVE_DESKTOP=1).",
    )
    parser.add_argument(
        "--force-live-focus",
        action="store_true",
        help="Explicitly allow switching focus even if another monitor holds active user focus (or ZELYR_FORCE_LIVE_FOCUS=1).",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON report output path")
    args = parser.parse_args()

    only = [int(x.strip()) for x in args.tasks.split(",") if x.strip()] if args.tasks else None
    signal.signal(signal.SIGTERM, lambda *_a: sys.exit(143))

    if args.focus_recovery:
        report = run_focus_recovery_benchmark(
            model=args.model,
            max_steps=args.max_steps,
            workspace_id=args.workspace,
            runs=args.runs,
            allow_live_desktop=(True if args.allow_live_desktop else None),
            force_live_focus=(True if args.force_live_focus else None),
        )
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(
            f"Focus Recovery Verdict: {report['verdict']} "
            f"({report['passed_count']}/{report['runs']} passed, "
            f"{report['unverified_count']} unverified, {report['failed_count']} failed)"
        )
        return 0 if report["verdict"] == "PASS" else 1

    report = run_benchmark(
        model=args.model,
        max_steps=args.max_steps,
        workspace_id=args.workspace,
        only_tasks=only,
        allow_live_desktop=(True if args.allow_live_desktop else None),
        force_live_focus=(True if args.force_live_focus else None),
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Summary: {report['passed_count']}/{report['total_count']} passed")
    return 0 if report["passed_count"] == report["total_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
