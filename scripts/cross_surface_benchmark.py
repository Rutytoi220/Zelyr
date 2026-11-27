"""Live isolated Cross-Surface Autonomous Workflow & Recovery Benchmark for Zelyr.

Exercises one realistic sequential task across four surfaces through real `zelyr run`:
  1. Browser automation (isolated headless Chromium + Native Messaging AF_UNIX socket)
     against a disposable local HTTP web app (dashboard + record detail + state mutation).
  2. Read-only HTTP API verification against the local web app API.
  3. Filesystem output generation (`out/report.json` and `out/report.md`) in an isolated workspace.
  4. Desktop GUI actuation & visual verification on an isolated PySide6 fixture window in
     empty Hyprland workspace 15.

Also validates two deterministic failure-and-recovery variants:
  - Scenario A (`browser_failure`): deterministic stale DOM element invalidation causing real
    `ELEMENT_NOT_FOUND`, requiring snapshot refresh and retry.
  - Scenario B (`desktop_focus_loss`): deterministic Hyprland focus steal onto a disposable
    distraction window causing real `FOCUS_LOST`, requiring window refocus and retry.
"""

from __future__ import annotations

import argparse
import copy
import functools
import http.server
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.linux_desktop_benchmark import (  # noqa: E402
    _cleanup_fixture_processes,
    _focus_and_verify_window,
    _FocusStealOllamaProxy,
    _free_port,
    _http_json,
    _hyprctl_dispatch,
    _hyprctl_json,
    _MUTATING_DESKTOP_ACTIONS,
    _place_floating_window,
    _read_ledger_summary,
    _wait_for_http,
    _wait_for_window,
)

FIXTURE_SCRIPT = REPO_ROOT / "scripts" / "desktop_test_fixture.py"
DEFAULT_DESKTOP_TITLE = "Zelyr Cross-Surface Confirmation"
DEFAULT_DISTRACTION_TITLE = "Zelyr Distraction Window"

INITIAL_RECORDS: Dict[str, Dict[str, Any]] = {
    "REC-101": {
        "record_id": "REC-101",
        "service": "alpha-ingress",
        "owner": "ops-team-a",
        "region": "eu-west-1",
        "score": 91,
        "confirmation_token": "ALPHA-101-KEEP",
        "status": "ACTIVE",
    },
    "REC-204": {
        "record_id": "REC-204",
        "service": "billing-ledger",
        "owner": "fin-core",
        "region": "eu-central-1",
        "score": 98,
        "confirmation_token": "BL-204-OK",
        "status": "PENDING_REVIEW",
    },
    "REC-309": {
        "record_id": "REC-309",
        "service": "media-transcoder",
        "owner": "edge-pipeline",
        "region": "us-east-1",
        "score": 74,
        "confirmation_token": "MT-309-HOLD",
        "status": "MONITORING",
    },
}


class CrossSurfaceWebApp:
    """Disposable local HTTP dashboard, detail page, and REST API server on 127.0.0.1."""

    def __init__(self, *, target_record: str = "REC-204", fail_browser_dom: bool = False) -> None:
        self.target_record = target_record
        self.fail_browser_dom = fail_browser_dom
        self._lock = threading.Lock()
        self.records: Dict[str, Dict[str, Any]] = copy.deepcopy(INITIAL_RECORDS)
        self.requests: List[Dict[str, Any]] = []
        self.dom_invalidated_events: List[Dict[str, Any]] = []
        self.dom_invalidation_consumed = False
        self._server: Optional[http.server.ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.base_url: str = ""

    def _record_request(self, method: str, path: str, status: int) -> None:
        with self._lock:
            self.requests.append({
                "idx": len(self.requests),
                "ts": round(time.time(), 4),
                "method": method,
                "path": path,
                "status": status,
            })

    def audit_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "target_record": self.target_record,
                "fail_browser_dom": self.fail_browser_dom,
                "dom_invalidation_consumed": self.dom_invalidation_consumed,
                "dom_invalidated_events": list(self.dom_invalidated_events),
                "records": copy.deepcopy(self.records),
                "requests": list(self.requests),
            }

    def __enter__(self) -> "CrossSurfaceWebApp":
        app_self = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def _send_html(self, code: int, html: str, *, log_req: bool = True) -> None:
                raw = html.encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                if log_req:
                    app_self._record_request(self.command, self.path, code)

            def _send_json(self, code: int, payload: Dict[str, Any], *, log_req: bool = True) -> None:
                raw = json.dumps(payload, indent=2).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                if log_req:
                    app_self._record_request(self.command, self.path, code)

            def do_GET(self) -> None:  # noqa: N802
                parsed = urllib.parse.urlparse(self.path)
                path = parsed.path

                if path == "/health":
                    self._send_json(200, {"ok": True}, log_req=False)
                    return

                if path == "/api/internal/audit":
                    self._send_json(200, app_self.audit_snapshot(), log_req=False)
                    return

                if path == "/api/records":
                    with app_self._lock:
                        recs = list(copy.deepcopy(app_self.records).values())
                    self._send_json(200, {"records": recs})
                    return

                if path.startswith("/api/records/"):
                    rec_id = path.removeprefix("/api/records/").strip("/")
                    with app_self._lock:
                        rec = copy.deepcopy(app_self.records.get(rec_id))
                    if rec is None:
                        self._send_json(404, {"error": f"Unknown record {rec_id}"})
                        return
                    self._send_json(200, rec)
                    return

                if path == "/":
                    with app_self._lock:
                        recs = list(copy.deepcopy(app_self.records).values())
                    rows = []
                    for r in recs:
                        rid = r["record_id"]
                        rows.append(
                            f"<tr id='row-{rid}'>"
                            f"<td><strong>{rid}</strong></td>"
                            f"<td>{r['service']}</td>"
                            f"<td>{r['owner']}</td>"
                            f"<td>{r['region']}</td>"
                            f"<td>{r['score']}</td>"
                            f"<td><span class='badge' id='dash-status-{rid}'>{r['status']}</span></td>"
                            f"<td><a id='link-{rid}' href='/records/{rid}'>Open {rid} Detail</a></td>"
                            f"</tr>"
                        )
                    html = (
                        "<!doctype html><html><head><meta charset='utf-8'>"
                        "<title>Operations Records Dashboard</title>"
                        "<style>body{font-family:sans-serif;background:#0f172a;color:#f8fafc;padding:24px;}"
                        "table{border-collapse:collapse;width:100%;max-width:900px;background:#1e293b;}"
                        "th,td{border:1px solid #334155;padding:10px 12px;text-align:left;}"
                        "th{background:#0b1220;color:#7dd3fc;}"
                        "a{color:#38bdf8;font-weight:bold;}</style></head><body>"
                        "<h1>Operations Records Dashboard</h1>"
                        "<p id='dash-note'>Select a record link to inspect metrics and manage its status.</p>"
                        "<table><thead><tr><th>Record ID</th><th>Service</th><th>Owner</th>"
                        "<th>Region</th><th>Score</th><th>Status</th><th>Detail</th></tr></thead>"
                        f"<tbody>{''.join(rows)}</tbody></table>"
                        "</body></html>"
                    )
                    self._send_html(200, html)
                    return

                if path.startswith("/records/"):
                    rec_id = path.removeprefix("/records/").strip("/")
                    with app_self._lock:
                        rec = copy.deepcopy(app_self.records.get(rec_id))
                        inject_stale = (
                            app_self.fail_browser_dom
                            and rec_id == app_self.target_record
                            and not app_self.dom_invalidation_consumed
                            and rec is not None
                            and rec["status"] != "APPROVED"
                        )
                    if rec is None:
                        self._send_html(404, "<html><body><h1>Record Not Found</h1></body></html>")
                        return

                    if inject_stale:
                        action_html = (
                            "<div id='action-slot'>"
                            "<button id='stale-approve-btn' type='button' onclick='return false;'>Approve Record</button>"
                            "</div>"
                            "<script>"
                            "(function(){"
                            "  var done = false;"
                            "  var slot = document.getElementById('action-slot');"
                            "  function swapToLive(){"
                            "    if (done || !slot) return;"
                            "    done = true;"
                            "    fetch('/api/internal/dom-invalidated', {method:'POST', body:'{}'}).catch(function(){});"
                            "    slot.innerHTML = '<button id=\"confirm-approve-btn\" type=\"button\" onclick=\"approveRecord()\">Confirm Approval</button>';"
                            "  }"
                            "  var obs = new MutationObserver(function(muts){"
                            "    for (var i = 0; i < muts.length; i++){"
                            "      if (muts[i].type === 'attributes' && muts[i].attributeName === 'data-zelyr-id'){"
                            "        setTimeout(swapToLive, 80);"
                            "        obs.disconnect();"
                            "        break;"
                            "      }"
                            "    }"
                            "  });"
                            "  obs.observe(document.body, {subtree:true, attributes:true, attributeFilter:['data-zelyr-id']});"
                            "})();"
                            "</script>"
                        )
                    elif rec["status"] == "APPROVED":
                        action_html = (
                            "<div id='action-slot'>"
                            "<button id='approve-btn' type='button' disabled>Record Approved</button>"
                            "</div>"
                        )
                    else:
                        action_html = (
                            "<div id='action-slot'>"
                            "<button id='approve-btn' type='button' onclick='approveRecord()'>Approve Record</button>"
                            "</div>"
                        )

                    html = (
                        "<!doctype html><html><head><meta charset='utf-8'>"
                        f"<title>Record {rec_id} Detail</title>"
                        "<style>body{font-family:sans-serif;background:#0f172a;color:#f8fafc;padding:24px;}"
                        ".card{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:18px;max-width:620px;}"
                        ".row{margin:8px 0;font-size:15px;} button{background:#16a34a;color:#fff;border:none;"
                        "border-radius:6px;padding:10px 16px;font-size:15px;font-weight:bold;cursor:pointer;}"
                        "a{color:#38bdf8;}</style></head><body>"
                        f"<h1>Record Detail: {rec_id}</h1>"
                        "<div class='card'>"
                        f"<div class='row'>Record ID: <span id='field-record-id'>{rec['record_id']}</span></div>"
                        f"<div class='row'>Service: <span id='field-service'>{rec['service']}</span></div>"
                        f"<div class='row'>Owner: <span id='field-owner'>{rec['owner']}</span></div>"
                        f"<div class='row'>Region: <span id='field-region'>{rec['region']}</span></div>"
                        f"<div class='row'>Score: <span id='field-score'>{rec['score']}</span></div>"
                        f"<div class='row'>Confirmation Token: <span id='field-token'>{rec['confirmation_token']}</span></div>"
                        f"<div class='row'>Status: <strong id='status-badge'>{rec['status']}</strong></div>"
                        f"{action_html}"
                        "<p id='action-feedback'></p>"
                        "<p><a id='back-dashboard' href='/'>Back to Dashboard</a></p>"
                        "</div>"
                        "<script>"
                        "async function approveRecord(){"
                        f"  var resp = await fetch('/api/records/{rec_id}/approve', {{method:'POST'}});"
                        "  var data = await resp.json();"
                        "  if (data && data.record){"
                        "    document.getElementById('status-badge').textContent = data.record.status;"
                        "    document.getElementById('action-feedback').textContent = 'Record status updated to ' + data.record.status;"
                        "    var btn = document.querySelector('#action-slot button');"
                        "    if (btn){ btn.textContent = 'Record Approved'; btn.disabled = true; }"
                        "  }"
                        "}"
                        "</script>"
                        "</body></html>"
                    )
                    self._send_html(200, html)
                    return

                self._send_html(404, "<html><body><h1>Not Found</h1></body></html>")

            def do_POST(self) -> None:  # noqa: N802
                parsed = urllib.parse.urlparse(self.path)
                path = parsed.path
                length = int(self.headers.get("Content-Length") or "0")
                if length > 0:
                    _ = self.rfile.read(length)

                if path == "/api/internal/dom-invalidated":
                    with app_self._lock:
                        app_self.dom_invalidation_consumed = True
                        app_self.dom_invalidated_events.append({"ts": round(time.time(), 4)})
                    self._send_json(200, {"ok": True}, log_req=False)
                    return

                if path.startswith("/api/records/") and path.endswith("/approve"):
                    rec_id = path.removeprefix("/api/records/").removesuffix("/approve").strip("/")
                    with app_self._lock:
                        rec = app_self.records.get(rec_id)
                        if rec is None:
                            self._send_json(404, {"error": f"Unknown record {rec_id}"})
                            return
                        rec["status"] = "APPROVED"
                        snap = copy.deepcopy(rec)
                    self._send_json(200, {"ok": True, "record": snap})
                    return

                self._send_json(404, {"error": "not found"})

        http.server.ThreadingHTTPServer.allow_reuse_address = True
        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        port = self._server.server_address[1]
        self.base_url = f"http://127.0.0.1:{port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_a: Any) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


class IsolatedChromiumSession:
    """Manages an isolated headless Chromium + Native Messaging AF_UNIX IPC session."""

    def __init__(self, tmp_root: Path) -> None:
        self.tmp_root = tmp_root
        self.home_dir = tmp_root / "chrom_home"
        self.prof_dir = tmp_root / "chrom_profile"
        self.socket_path = tmp_root / "zelyr_native.sock"
        self.wrapper_path = tmp_root / "bin" / "zelyr-browser-host"
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self.extension_id: str = ""

    def _kill_proc(self) -> None:
        if self._proc is None:
            return
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
        except Exception:
            try:
                self._proc.terminate()
            except Exception:
                pass
        try:
            self._proc.wait(timeout=4.0)
        except Exception:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None

    def __enter__(self) -> "IsolatedChromiumSession":
        import asyncio
        from zelyr.tools.browser_native_host import (
            NativeBrowserClient,
            compute_unpacked_extension_id,
            install_native_messaging_host,
        )

        chromium_path = shutil.which("chromium") or shutil.which("chromium-browser") or shutil.which("google-chrome")
        if not chromium_path:
            raise RuntimeError("Chromium executable not found on PATH for isolated browser session.")

        ext_dir = REPO_ROOT / "zelyr" / "tools" / "extension"
        self.extension_id = compute_unpacked_extension_id(ext_dir)
        self.prof_dir.mkdir(parents=True, exist_ok=True)
        install_native_messaging_host(
            extension_dir=ext_dir,
            hosts_dir=self.home_dir / ".config" / "chromium" / "NativeMessagingHosts",
            wrapper_path=self.wrapper_path,
            allowed_hosts=["127.0.0.1"],
        )
        install_native_messaging_host(
            extension_dir=ext_dir,
            user_data_dir=self.prof_dir,
            wrapper_path=self.wrapper_path,
            allowed_hosts=["127.0.0.1"],
        )
        self.wrapper_path.write_text(
            self.wrapper_path.read_text(encoding="utf-8").replace(
                "#!/bin/sh\n",
                f'#!/bin/sh\nexport ZELYR_BROWSER_NATIVE_SOCKET="{self.socket_path}"\n',
            ),
            encoding="utf-8",
        )
        os.chmod(self.wrapper_path, 0o755)

        env = {
            **os.environ,
            "HOME": str(self.home_dir),
            "XDG_CONFIG_HOME": str(self.home_dir / ".config"),
            "ZELYR_BROWSER_NATIVE_SOCKET": str(self.socket_path),
            "ZELYR_BROWSER_ALLOWED_HOSTS": "127.0.0.1",
        }

        def _launch() -> subprocess.Popen[bytes]:
            return subprocess.Popen(
                [
                    chromium_path,
                    "--headless=new",
                    "--no-sandbox",
                    "--disable-gpu",
                    f"--user-data-dir={self.prof_dir}",
                    f"--load-extension={ext_dir}",
                    "about:blank",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                start_new_session=True,
            )

        self._proc = _launch()
        for _ in range(80):
            if self.socket_path.exists():
                break
            time.sleep(0.1)
        if not self.socket_path.exists():
            # Chromium MV3 service worker cold-start occasionally requires one relaunch after profile init.
            self._kill_proc()
            self._proc = _launch()
            for _ in range(100):
                if self.socket_path.exists():
                    break
                time.sleep(0.1)

        if not self.socket_path.exists():
            raise RuntimeError(f"Isolated Native Messaging socket did not appear at {self.socket_path}.")

        client = NativeBrowserClient(socket_path=self.socket_path)
        if not asyncio.run(client.is_available()):
            raise RuntimeError(f"Isolated Native Messaging host at {self.socket_path} did not answer ping.")
        return self

    def __exit__(self, *_a: Any) -> None:
        self._kill_proc()


def setup_cross_surface_workspace(
    ws_dir: Path,
    base_url: str,
    *,
    target_record: str = "REC-204",
    desktop_title: str = DEFAULT_DESKTOP_TITLE,
) -> Dict[str, Any]:
    """Create isolated filesystem workspace with empty out/, task_spec.json, and expected_result.json."""
    ws_dir.mkdir(parents=True, exist_ok=True)
    out_dir = ws_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_json = out_dir / "report.json"
    report_md = out_dir / "report.md"
    task_spec_path = ws_dir / "task_spec.json"
    expected_result_path = ws_dir / "expected_result.json"

    task_spec = {
        "workflow": "cross_surface_verification",
        "web_dashboard_url": f"{base_url}/",
        "target_record_id": target_record,
        "requested_status": "APPROVED",
        "api_endpoint_template": f"{base_url}/api/records/{{record_id}}",
        "output_dir": str(out_dir),
        "report_json_path": str(report_json),
        "report_md_path": str(report_md),
        "required_json_keys": [
            "record_id",
            "service",
            "owner",
            "region",
            "score",
            "confirmation_token",
            "status",
        ],
        "desktop_window_title": desktop_title,
        "desktop_confirmation_button": "Confirm",
        "confirmation_payload_template": "<record_id>:<confirmation_token>:<status>",
    }
    task_spec_path.write_text(json.dumps(task_spec, indent=2) + "\n", encoding="utf-8")

    expected_rec = copy.deepcopy(INITIAL_RECORDS[target_record])
    expected_rec["status"] = "APPROVED"
    expected_result = {
        "target_record_id": target_record,
        "expected_record": expected_rec,
        "unrelated_records": {
            k: copy.deepcopy(v) for k, v in INITIAL_RECORDS.items() if k != target_record
        },
        "expected_desktop_confirmation_text": (
            f"{target_record}:{expected_rec['confirmation_token']}:APPROVED"
        ),
        "expected_desktop_status_prefix": "CONFIRMED:",
    }
    expected_result_path.write_text(json.dumps(expected_result, indent=2) + "\n", encoding="utf-8")

    return {
        "workspace_dir": ws_dir,
        "out_dir": out_dir,
        "task_spec_path": task_spec_path,
        "expected_result_path": expected_result_path,
        "report_json_path": report_json,
        "report_md_path": report_md,
        "expected_result": expected_result,
    }


def build_cross_surface_prompt(base_url: str, ws_info: Dict[str, Any], *, desktop_title: str) -> str:
    """Build a crisp, non-prescriptive cross-surface workflow prompt."""
    return (
        f"Complete this cross-surface verification workflow:\n"
        f"1. Use Chromium browser automation (do NOT use curl or shell commands to inspect web pages "
        f"or approve records): open {base_url}/, navigate to REC-204's detail page, and click the "
        f"approval control on the page so its status becomes APPROVED. Leave all other records unchanged.\n"
        f"2. Run a read-only HTTP GET command against {base_url}/api/records/REC-204 to verify the stored "
        f"fields and APPROVED status.\n"
        f"3. Write the verified record JSON to {ws_info['report_json_path']} with keys "
        f"record_id, service, owner, region, score, confirmation_token, status, and write a concise "
        f"Markdown summary with those same verified details to {ws_info['report_md_path']}.\n"
        f"4. On your Linux desktop (outside Chromium), observe window '{desktop_title}', enforce focused "
        f"window '{desktop_title}' on every desktop input action, enter "
        f"'<record_id>:<confirmation_token>:<status>' (using the live verified values) into its input field "
        f"and submit via 'Confirm' or Enter, then capture a zoomed window screenshot of '{desktop_title}' "
        f"afterward to confirm completion."
    )


def _run_cross_surface_cli(
    prompt: str,
    *,
    model: str,
    max_steps: int,
    db_path: Path,
    native_socket: Path,
    timeout_s: float = 540.0,
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
    env["ZELYR_BROWSER_NATIVE_SOCKET"] = str(native_socket)
    env["ZELYR_BROWSER_ALLOWED_HOSTS"] = "127.0.0.1"

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


def evaluate_cross_surface_run(
    *,
    scenario: str,
    run_info: Dict[str, Any],
    web_audit: Dict[str, Any],
    ws_info: Dict[str, Any],
    desktop_before: Dict[str, Any],
    desktop_after: Dict[str, Any],
    distraction_after: Optional[Dict[str, Any]] = None,
    steal_telemetry: Optional[Dict[str, Any]] = None,
    desktop_title: str = DEFAULT_DESKTOP_TITLE,
    distraction_title: str = DEFAULT_DISTRACTION_TITLE,
) -> Dict[str, Any]:
    """Fail-closed independent validator returning PASS, FAIL, or UNVERIFIED."""
    from zelyr.engine.execution_ledger import _is_inspection_action

    cli_json = run_info.get("cli_json") if isinstance(run_info.get("cli_json"), dict) else {}
    exec_ledger = cli_json.get("execution_ledger") if isinstance(cli_json.get("execution_ledger"), dict) else {}
    records = exec_ledger.get("records") if isinstance(exec_ledger.get("records"), list) else []
    central = run_info.get("ledger") if isinstance(run_info.get("ledger"), dict) else {}
    expected = ws_info["expected_result"]
    exp_rec = expected["expected_record"]

    # 1. Web app state & unrelated record preservation
    web_recs = web_audit.get("records") if isinstance(web_audit.get("records"), dict) else {}
    target_web_ok = web_recs.get("REC-204") == exp_rec
    unrelated_web_ok = all(
        web_recs.get(uid) == udata for uid, udata in expected["unrelated_records"].items()
    )

    reqs = web_audit.get("requests") if isinstance(web_audit.get("requests"), list) else []
    visited_dash = any(r.get("method") == "GET" and r.get("path") == "/" and r.get("status") == 200 for r in reqs)
    visited_detail = any(
        r.get("method") == "GET" and r.get("path") == "/records/REC-204" and r.get("status") == 200
        for r in reqs
    )
    approve_indices = [
        i for i, r in enumerate(reqs)
        if r.get("method") == "POST" and r.get("path") == "/api/records/REC-204/approve" and r.get("status") == 200
    ]
    unrelated_approve = any(
        r.get("method") == "POST" and r.get("path") != "/api/records/REC-204/approve"
        for r in reqs
    )
    first_approve_req_idx = approve_indices[0] if approve_indices else None
    api_verify_after_mut = (
        first_approve_req_idx is not None
        and any(
            i > first_approve_req_idx
            and r.get("method") == "GET"
            and r.get("path") == "/api/records/REC-204"
            and r.get("status") == 200
            for i, r in enumerate(reqs)
        )
    )

    # 2. Browser + API tool ledger evidence
    browser_recs = [r for r in records if r.get("tool_name") == "interact_with_browser"]
    browser_ok = bool(browser_recs) and visited_dash and visited_detail and target_web_ok and unrelated_web_ok and not unrelated_approve

    api_cmd_recs = [
        r for r in records
        if r.get("tool_name") == "execute_command"
        and r.get("state") in ("success", "verified")
        and "/api/records/REC-204" in str((r.get("arguments") or {}).get("command") or "")
        and re.match(
            r"^\s*(?:curl|wget|http|python3?|fetch)\b",
            str((r.get("arguments") or {}).get("command") or ""),
            flags=re.IGNORECASE,
        )
    ]
    api_classified_inspection = bool(api_cmd_recs) and all(
        _is_inspection_action(r.get("tool_name") or "", r.get("arguments") or {})
        for r in api_cmd_recs
    )
    read_only_api_verified = bool(api_verify_after_mut and api_cmd_recs and api_classified_inspection)

    # 3. Filesystem report artifacts & accuracy
    report_json_path = Path(ws_info["report_json_path"])
    report_md_path = Path(ws_info["report_md_path"])
    json_ok = False
    parsed_report_json: Any = None
    if report_json_path.is_file():
        try:
            parsed_report_json = json.loads(report_json_path.read_text(encoding="utf-8"))
            if isinstance(parsed_report_json, dict):
                json_ok = all(parsed_report_json.get(k) == v for k, v in exp_rec.items())
        except Exception:
            json_ok = False

    md_ok = False
    if report_md_path.is_file():
        md_text = report_md_path.read_text(encoding="utf-8")
        md_ok = (
            bool(re.search(r"^\s*#", md_text, flags=re.MULTILINE))
            and "REC-204" in md_text
            and exp_rec["service"] in md_text
            and exp_rec["owner"] in md_text
            and exp_rec["region"] in md_text
            and str(exp_rec["score"]) in md_text
            and exp_rec["confirmation_token"] in md_text
            and "APPROVED" in md_text
        )

    out_files = sorted(p.name for p in Path(ws_info["out_dir"]).iterdir()) if Path(ws_info["out_dir"]).is_dir() else []
    workspace_clean = set(out_files) == {"report.json", "report.md"}

    # 4. Desktop confirmation state & unrelated distraction safety
    exp_desktop_text = expected["expected_desktop_confirmation_text"]
    desktop_confirmed = (
        desktop_after.get("applied_text") == exp_desktop_text
        and str(desktop_after.get("status_text") or "").startswith("CONFIRMED:")
        and int(desktop_after.get("key_presses", 0)) >= 1
    )
    desktop_observed_after = False
    last_mut_idx = -1
    last_obs_idx = -1
    for i, rec in enumerate(records):
        if rec.get("state") not in ("success", "verified"):
            continue
        tn = rec.get("tool_name")
        act = str((rec.get("arguments") or {}).get("action") or "").lower()
        if tn == "desktop_control" and act in _MUTATING_DESKTOP_ACTIONS:
            last_mut_idx = i
        elif tn == "capture_screen" or (tn == "desktop_control" and act in ("observe", "screenshot")):
            last_obs_idx = i
    desktop_observed_after = last_mut_idx >= 0 and last_obs_idx > last_mut_idx

    distraction_untouched = True
    if distraction_after is not None:
        distraction_untouched = (
            int(distraction_after.get("mouse_clicks", 0)) == 0
            and int(distraction_after.get("key_presses", 0)) == 0
        )

    # 5. Ledger & CLI integrity
    unresolved_list = exec_ledger.get("unresolved_failures") if isinstance(exec_ledger.get("unresolved_failures"), list) else []
    has_unresolved_flag = exec_ledger.get("has_unresolved_failures")
    if has_unresolved_flag is None:
        has_unresolved_flag = bool(unresolved_list)
    cli_outcome_ok = (
        run_info.get("returncode") == 0
        and has_unresolved_flag is False
        and len(unresolved_list) == 0
        and central.get("status") in ("completed", "completed_unverified")
        and central.get("error") in (None, "")
        and central.get("verification_status") not in ("verification_failed", "failed")
        and len(records) >= 4
    )

    checks: Dict[str, Any] = {
        "evidence_present": bool(cli_json and records and reqs),
        "browser_dashboard_visited": visited_dash,
        "browser_detail_visited": visited_detail,
        "target_web_approved": target_web_ok,
        "unrelated_web_untouched": unrelated_web_ok and not unrelated_approve,
        "read_only_api_verified": read_only_api_verified,
        "api_check_classified_inspection": api_classified_inspection,
        "report_json_accurate": json_ok,
        "report_md_accurate": md_ok,
        "workspace_clean": workspace_clean,
        "desktop_confirmed": desktop_confirmed,
        "desktop_visual_verified_after_mut": desktop_observed_after,
        "distraction_untouched": distraction_untouched,
        "cli_and_ledger_consistent": cli_outcome_ok,
    }

    unverified_reasons: List[str] = []
    fail_reasons: List[str] = []

    if scenario == "browser_failure":
        dom_events = web_audit.get("dom_invalidated_events") or []
        first_dom_ts = float(dom_events[0]["ts"]) if dom_events else None
        elem_fail_idx = next(
            (
                i for i, r in enumerate(records)
                if r.get("tool_name") == "interact_with_browser"
                and r.get("state") == "failed"
                and str((r.get("arguments") or {}).get("action") or "").lower() in ("click_element", "click", "fill_element")
                and "element" in str(r.get("error") or "").lower()
                and "not found" in str(r.get("error") or "").lower()
            ),
            None,
        )
        browser_fail_classified = elem_fail_idx is not None
        refresh_after_fail = any(
            elem_fail_idx is not None
            and i > elem_fail_idx
            and r.get("tool_name") == "interact_with_browser"
            and str((r.get("arguments") or {}).get("action") or "").lower() == "get_page_snapshot"
            and r.get("state") in ("success", "verified")
            for i, r in enumerate(records)
        )
        retry_after_refresh_idx = next(
            (
                i for i, r in enumerate(records)
                if elem_fail_idx is not None
                and i > elem_fail_idx
                and r.get("tool_name") == "interact_with_browser"
                and str((r.get("arguments") or {}).get("action") or "").lower() in ("click_element", "click")
                and r.get("state") in ("success", "verified")
            ),
            None,
        )
        no_approve_before_invalidation = (
            first_dom_ts is not None
            and bool(approve_indices)
            and float(reqs[approve_indices[0]]["ts"]) > first_dom_ts
        )
        checks.update({
            "browser_dom_invalidated": bool(dom_events),
            "browser_failed_element_not_found": browser_fail_classified,
            "browser_snapshot_refreshed_after_fail": refresh_after_fail,
            "browser_retry_succeeded": retry_after_refresh_idx is not None,
            "no_mutation_before_invalidation": no_approve_before_invalidation,
        })
        if not dom_events or not browser_fail_classified:
            unverified_reasons.append("browser failure was bypassed or not encountered in ledger")
        if bool(approve_indices) and not no_approve_before_invalidation:
            unverified_reasons.append("web mutation occurred before DOM invalidation")

    elif scenario == "desktop_focus_loss":
        steal = steal_telemetry or {}
        pre_active = steal.get("active_before") if isinstance(steal.get("active_before"), dict) else {}
        post_active = steal.get("active_after") if isinstance(steal.get("active_after"), dict) else {}
        fix_at_steal = steal.get("fix_state_at_steal") if isinstance(steal.get("fix_state_at_steal"), dict) else {}

        def _matches_fix(args: Any) -> bool:
            if not isinstance(args, dict):
                return False
            tgt = str(args.get("expected_window") or args.get("window") or args.get("target") or "").lower()
            return bool(tgt) and (desktop_title.lower() in tgt or tgt in desktop_title.lower())

        def _matches_initial_obs(args: Any) -> bool:
            if _matches_fix(args):
                return True
            if not isinstance(args, dict):
                return False
            tgt_scope = str(args.get("target") or "").strip().lower()
            return (
                tgt_scope in ("active_monitor", "active_window", "full")
                and desktop_title.lower() in str(pre_active.get("title") or "").lower()
            )

        obs_indices = [
            i for i, r in enumerate(records)
            if r.get("state") in ("success", "verified")
            and (
                (r.get("tool_name") == "capture_screen" and _matches_initial_obs(r.get("arguments")))
                or (
                    r.get("tool_name") == "desktop_control"
                    and str((r.get("arguments") or {}).get("action") or "").lower() in ("observe", "screenshot")
                    and _matches_initial_obs(r.get("arguments"))
                )
            )
        ]
        mut_indices = [
            i for i, r in enumerate(records)
            if r.get("tool_name") == "desktop_control"
            and str((r.get("arguments") or {}).get("action") or "").lower() in _MUTATING_DESKTOP_ACTIONS
        ]
        state_changing_mut_indices = [
            i for i in mut_indices
            if str((records[i].get("arguments") or {}).get("action") or "").lower()
            not in ("mouse_move", "move_cursor", "hover")
        ]
        focus_lost_indices = [
            i for i in mut_indices
            if records[i].get("state") == "failed"
            and _matches_fix(records[i].get("arguments"))
            and (
                "focus_lost" in str(records[i].get("error") or "").lower()
                or "is not focused" in str(records[i].get("error") or "").lower()
            )
        ]
        first_obs_idx = obs_indices[0] if obs_indices else None
        first_mut_idx = mut_indices[0] if mut_indices else None
        first_focus_lost_idx = focus_lost_indices[0] if focus_lost_indices else None
        no_pre_focus_loss_mutation = (
            first_focus_lost_idx is not None
            and not any(
                i < first_focus_lost_idx and records[i].get("state") in ("success", "verified")
                for i in state_changing_mut_indices
            )
        )
        first_mut_focus_lost = bool(first_focus_lost_idx is not None and no_pre_focus_loss_mutation)
        proactive_focus_bypass = any(
            first_obs_idx is not None
            and (first_focus_lost_idx is None or i < first_focus_lost_idx)
            and i > first_obs_idx
            and r.get("tool_name") == "manage_desktop_window"
            and str((r.get("arguments") or {}).get("action") or "").lower() in ("focus", "focuswindow")
            for i, r in enumerate(records)
        )
        interference_ok = (
            bool(steal.get("stolen"))
            and desktop_title.lower() in str(pre_active.get("title") or "").lower()
            and distraction_title.lower() in str(post_active.get("title") or "").lower()
            and str(fix_at_steal.get("applied_text") or "") == ""
            and int(fix_at_steal.get("mouse_clicks", 0)) == 0
            and int(fix_at_steal.get("key_presses", 0)) == 0
        )
        refocus_indices = [
            i for i, r in enumerate(records)
            if first_focus_lost_idx is not None
            and i > first_focus_lost_idx
            and r.get("tool_name") == "manage_desktop_window"
            and str((r.get("arguments") or {}).get("action") or "").lower() in ("focus", "focuswindow")
            and _matches_fix(r.get("arguments"))
            and r.get("state") in ("success", "verified")
        ]
        retry_ok = any(
            bool(refocus_indices)
            and i > refocus_indices[0]
            and _matches_fix(records[i].get("arguments"))
            and records[i].get("state") in ("success", "verified")
            for i in state_changing_mut_indices
        )
        checks.update({
            "desktop_initial_observation": first_obs_idx is not None and (first_mut_idx is None or first_obs_idx < first_mut_idx),
            "desktop_controlled_interference": interference_ok,
            "desktop_focus_lost_rejected": first_mut_focus_lost,
            "no_proactive_focus_bypass": not proactive_focus_bypass,
            "desktop_refocus_recovered": bool(refocus_indices),
            "desktop_mutation_retried_ok": retry_ok,
        })
        if not interference_ok or not first_mut_focus_lost or proactive_focus_bypass:
            unverified_reasons.append("FOCUS_LOST was bypassed or deterministic focus-steal evidence incomplete")

    if not checks["evidence_present"]:
        unverified_reasons.append("missing CLI JSON, ledger records, or HTTP server telemetry")

    for k, v in checks.items():
        if not v:
            fail_reasons.append(k)

    if unverified_reasons:
        verdict = "UNVERIFIED"
    elif all(checks.values()):
        verdict = "PASS"
    else:
        verdict = "FAIL"

    return {
        "scenario": scenario,
        "verdict": verdict,
        "checks": checks,
        "unverified_reasons": unverified_reasons,
        "failed_checks": fail_reasons,
        "report_json_parsed": parsed_report_json,
        "desktop_before": desktop_before,
        "desktop_after": desktop_after,
    }


def _save_run_artifacts(
    output_dir: Path,
    report: Dict[str, Any],
    transcripts: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "cross_surface_report.json"
    md_path = output_dir / "cross_surface_summary.md"
    transcript_dir = output_dir / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)

    transcript_paths: Dict[str, str] = {}
    for sc_name, tdata in transcripts.items():
        tpath = transcript_dir / f"transcript_{sc_name}.json"
        tpath.write_text(json.dumps(tdata, indent=2) + "\n", encoding="utf-8")
        transcript_paths[sc_name] = str(tpath)

    full_report = {**report, "transcript_files": transcript_paths}
    json_path.write_text(json.dumps(full_report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Zelyr Cross-Surface Autonomous Workflow Benchmark",
        "",
        f"- **Timestamp:** {report['timestamp']}",
        f"- **Model:** `{report['model']}` (`num_ctx={report['num_ctx']}`, `max_steps={report['max_steps']}`)",
        f"- **Passed Scenarios:** {report['passed_count']}/{report['total_count']}",
        "",
        "| Scenario | Verdict | CLI Exit | Duration (s) | Ledger Status | Unresolved |",
        "|---|---|---:|---:|---|---:|",
    ]
    for r in report["scenarios"]:
        cli_j = r.get("cli_json") if isinstance(r.get("cli_json"), dict) else {}
        ex_l = cli_j.get("execution_ledger") if isinstance(cli_j.get("execution_ledger"), dict) else {}
        cen_l = r.get("ledger") if isinstance(r.get("ledger"), dict) else {}
        unres = len(ex_l.get("unresolved_failures") or []) if isinstance(ex_l.get("unresolved_failures"), list) else -1
        lines.append(
            f"| `{r['scenario']}` | **{r['verdict']}** | {r['returncode']} | {r['duration_s']} | "
            f"`{cen_l.get('status')}` | {unres} |"
        )
    lines.append("")
    for r in report["scenarios"]:
        lines.extend([
            f"## Scenario `{r['scenario']}` — `{r['verdict']}`",
            "",
            f"- **Failed checks:** {r['validation'].get('failed_checks')}",
            f"- **Unverified reasons:** {r['validation'].get('unverified_reasons')}",
            f"- **Transcript:** `{transcript_paths.get(r['scenario'], '')}`",
            "",
        ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "json": str(json_path),
        "md": str(md_path),
        "transcripts": json.dumps(transcript_paths),
    }


def run_single_cross_surface_scenario(
    scenario: str,
    *,
    model: str,
    max_steps: int,
    workspace_id: int,
    mon_x: int,
    mon_y: int,
    mon_w: int,
    mon_h: int,
    upstream_ollama: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run one complete isolated cross-surface scenario with dedicated fixtures."""
    if scenario not in ("normal", "browser_failure", "desktop_focus_loss"):
        raise ValueError(f"Unsupported cross-surface scenario: {scenario!r}")

    with tempfile.TemporaryDirectory(prefix=f"zelyr_xsurf_{scenario}_") as tmp_str:
        tmp_dir = Path(tmp_str)
        ws_dir = tmp_dir / "workspace"
        db_path = tmp_dir / f"{scenario}.db"
        fix_port = _free_port()
        dist_port = _free_port() if scenario == "desktop_focus_loss" else None
        fix_proc: Optional[subprocess.Popen[bytes]] = None
        dist_proc: Optional[subprocess.Popen[bytes]] = None
        steal_telemetry: Dict[str, Any] = {}

        with CrossSurfaceWebApp(
            target_record="REC-204",
            fail_browser_dom=(scenario == "browser_failure"),
        ) as web_app, IsolatedChromiumSession(tmp_dir) as browser_sess:
            ws_info = setup_cross_surface_workspace(
                ws_dir,
                web_app.base_url,
                target_record="REC-204",
                desktop_title=DEFAULT_DESKTOP_TITLE,
            )
            prompt = build_cross_surface_prompt(
                web_app.base_url,
                ws_info,
                desktop_title=DEFAULT_DESKTOP_TITLE,
            )
            try:
                fix_proc = subprocess.Popen(
                    [
                        sys.executable,
                        str(FIXTURE_SCRIPT),
                        "--port",
                        str(fix_port),
                        "--title",
                        DEFAULT_DESKTOP_TITLE,
                        "--cross-surface",
                    ],
                    cwd=str(REPO_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                )
                _wait_for_http(f"http://127.0.0.1:{fix_port}/health")
                fix_win = _wait_for_window(DEFAULT_DESKTOP_TITLE, pid=fix_proc.pid)
                fix_addr = str(fix_win["address"])
                fx = mon_x + (mon_w - 640) // 2
                fy = mon_y + (mon_h - 420) // 2
                _place_floating_window(fix_addr, workspace_id, fx, fy, 640, 420)

                dist_addr: Optional[str] = None
                if scenario == "desktop_focus_loss" and dist_port is not None:
                    dist_proc = subprocess.Popen(
                        [
                            sys.executable,
                            str(FIXTURE_SCRIPT),
                            "--port",
                            str(dist_port),
                            "--title",
                            DEFAULT_DISTRACTION_TITLE,
                            "--distraction",
                        ],
                        cwd=str(REPO_ROOT),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        start_new_session=True,
                    )
                    _wait_for_http(f"http://127.0.0.1:{dist_port}/health")
                    dist_win = _wait_for_window(DEFAULT_DISTRACTION_TITLE, pid=dist_proc.pid)
                    dist_addr = str(dist_win["address"])
                    _place_floating_window(dist_addr, workspace_id, mon_x + 40, mon_y + 40, 360, 180)

                _hyprctl_dispatch("workspace", str(workspace_id))
                _focus_and_verify_window(fix_addr, DEFAULT_DESKTOP_TITLE, f"http://127.0.0.1:{fix_port}")
                time.sleep(0.15)
                desktop_before = _http_json(f"http://127.0.0.1:{fix_port}/state")

                if scenario == "desktop_focus_loss" and dist_addr and dist_port is not None:
                    def _steal_focus(turn_no: int) -> Dict[str, Any]:
                        pre_act = _hyprctl_json("activewindow")
                        fix_snap = _http_json(f"http://127.0.0.1:{fix_port}/state")
                        sw = _focus_and_verify_window(
                            dist_addr,
                            DEFAULT_DISTRACTION_TITLE,
                            f"http://127.0.0.1:{dist_port}",
                        )
                        return {
                            "turn": turn_no,
                            "active_before": pre_act,
                            "active_after": sw["active_window"],
                            "fix_state_at_steal": fix_snap,
                            "dist_state_at_steal": sw.get("fixture_state") or _http_json(f"http://127.0.0.1:{dist_port}/state"),
                        }

                    with _FocusStealOllamaProxy(upstream_ollama, _steal_focus) as proxy:
                        run_info = _run_cross_surface_cli(
                            prompt,
                            model=model,
                            max_steps=max_steps,
                            db_path=db_path,
                            native_socket=browser_sess.socket_path,
                            base_url=proxy.base_url,
                        )
                        steal_telemetry = dict(proxy.telemetry)
                else:
                    run_info = _run_cross_surface_cli(
                        prompt,
                        model=model,
                        max_steps=max_steps,
                        db_path=db_path,
                        native_socket=browser_sess.socket_path,
                    )

                desktop_after = _http_json(f"http://127.0.0.1:{fix_port}/state")
                distraction_after = (
                    _http_json(f"http://127.0.0.1:{dist_port}/state")
                    if (scenario == "desktop_focus_loss" and dist_port is not None)
                    else None
                )
                web_audit = web_app.audit_snapshot()

                validation = evaluate_cross_surface_run(
                    scenario=scenario,
                    run_info=run_info,
                    web_audit=web_audit,
                    ws_info=ws_info,
                    desktop_before=desktop_before,
                    desktop_after=desktop_after,
                    distraction_after=distraction_after,
                    steal_telemetry=steal_telemetry,
                    desktop_title=DEFAULT_DESKTOP_TITLE,
                    distraction_title=DEFAULT_DISTRACTION_TITLE,
                )

                summary_entry = {
                    "scenario": scenario,
                    "verdict": validation["verdict"],
                    "returncode": run_info["returncode"],
                    "duration_s": run_info["duration_s"],
                    "cli_json": run_info["cli_json"],
                    "ledger": run_info["ledger"],
                    "validation": validation,
                }
                raw_transcript = {
                    "scenario": scenario,
                    "prompt": prompt,
                    "base_url": web_app.base_url,
                    "web_audit": web_audit,
                    "desktop_before": desktop_before,
                    "desktop_after": desktop_after,
                    "distraction_after": distraction_after,
                    "steal_telemetry": steal_telemetry,
                    "run_info": run_info,
                    "validation": validation,
                }
                return summary_entry, raw_transcript
            finally:
                _cleanup_fixture_processes((dist_proc, dist_port), (fix_proc, fix_port))


def run_cross_surface_benchmark(
    *,
    scenarios: Optional[List[str]] = None,
    model: str = "qwen3.6:35b-a3b",
    max_steps: int = 24,
    workspace_id: int = 15,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run all selected cross-surface scenarios safely on empty Hyprland workspace 15."""
    monitors = _hyprctl_json("monitors")
    focused_mon = next((m for m in monitors if m.get("focused")), monitors[0])
    dp1 = next((m for m in monitors if m.get("name") == "DP-1"), focused_mon)
    mon_x, mon_y = int(dp1["x"]), int(dp1["y"])
    mon_w = int(round(dp1["width"] / (dp1.get("scale") or 1.0)))
    mon_h = int(round(dp1["height"] / (dp1.get("scale") or 1.0)))

    orig_ws_obj = _hyprctl_json("activeworkspace")
    orig_win_obj = _hyprctl_json("activewindow")
    dp1_orig_ws_id = (dp1.get("activeWorkspace") or {}).get("id")
    orig_clients = _hyprctl_json("clients")
    ws_existing = [
        c for c in orig_clients
        if str((c.get("workspace") or {}).get("id")) == str(workspace_id)
    ]
    if ws_existing:
        raise RuntimeError(
            f"Safety abort: target isolated workspace {workspace_id} is not empty ({len(ws_existing)} windows)."
        )

    chosen = scenarios or ["normal", "browser_failure", "desktop_focus_loss"]
    upstream_ollama = os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
    scenario_summaries: List[Dict[str, Any]] = []
    transcripts: Dict[str, Dict[str, Any]] = {}

    try:
        if dp1.get("name"):
            _hyprctl_dispatch("focusmonitor", str(dp1["name"]))
        _hyprctl_dispatch("workspace", str(workspace_id))
        time.sleep(0.25)

        for sc in chosen:
            summary_entry, raw_transcript = run_single_cross_surface_scenario(
                sc,
                model=model,
                max_steps=max_steps,
                workspace_id=workspace_id,
                mon_x=mon_x,
                mon_y=mon_y,
                mon_w=mon_w,
                mon_h=mon_h,
                upstream_ollama=upstream_ollama,
            )
            scenario_summaries.append(summary_entry)
            transcripts[sc] = raw_transcript
            val = summary_entry["validation"]
            print(
                f"[Cross-Surface: {sc}] {summary_entry['verdict']} | rc={summary_entry['returncode']} | "
                f"{summary_entry['duration_s']}s | failed={val['failed_checks']} | unverified={val['unverified_reasons']}",
                flush=True,
            )
    finally:
        if dp1_orig_ws_id is not None:
            _hyprctl_dispatch("workspace", str(dp1_orig_ws_id))
        orig_ws_id = orig_ws_obj.get("id") if isinstance(orig_ws_obj, dict) else None
        if orig_ws_id is not None:
            _hyprctl_dispatch("workspace", str(orig_ws_id))
        orig_addr = orig_win_obj.get("address") if isinstance(orig_win_obj, dict) else None
        if orig_addr:
            _hyprctl_dispatch("focuswindow", f"address:{orig_addr}")

    report: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "num_ctx": int(os.environ.get("ZELYR_NUM_CTX") or "8192"),
        "max_steps": max_steps,
        "workspace_id": workspace_id,
        "passed_count": sum(1 for r in scenario_summaries if r["verdict"] == "PASS"),
        "total_count": len(scenario_summaries),
        "scenarios": scenario_summaries,
    }
    if output_dir is not None:
        artifact_paths = _save_run_artifacts(output_dir, report, transcripts)
        report["artifacts"] = artifact_paths
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Zelyr isolated Cross-Surface Autonomous Workflow Benchmark")
    parser.add_argument(
        "--scenario",
        choices=["all", "normal", "browser_failure", "desktop_focus_loss"],
        default="all",
        help="Which scenario(s) to execute",
    )
    parser.add_argument("--model", default="qwen3.6:35b-a3b")
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--workspace-id", type=int, default=15)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/zelyr_cross_surface_benchmark_reports"),
        help="Directory to write cross_surface_report.json, cross_surface_summary.md, and transcripts",
    )
    args = parser.parse_args()

    scenarios = (
        ["normal", "browser_failure", "desktop_focus_loss"]
        if args.scenario == "all"
        else [args.scenario]
    )
    res = run_cross_surface_benchmark(
        scenarios=scenarios,
        model=args.model,
        max_steps=args.max_steps,
        workspace_id=args.workspace_id,
        output_dir=args.output_dir,
    )
    print(f"Cross-Surface Summary: {res['passed_count']}/{res['total_count']} PASSED", flush=True)
    return 0 if res["passed_count"] == res["total_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
