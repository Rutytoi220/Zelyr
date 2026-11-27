"""Offline security regression tests for zelyr.server.node_api POST /execute_shell.

Covers:
1. Unauthenticated requests & missing/weak/insecure server tokens
2. Invalid credentials & untrusted/spoofed client IP addresses
3. Authorized requests (env token & 0600 file token; loopback/LAN/Tailscale)
4. Malformed request bodies, NUL bytes, extra fields, out-of-range timeouts
5. Oversized request payloads (declared Content-Length and streamed) and capped output
6. Execution timeout and process-group reaping
7. Working-directory restrictions (workspace default, subdirs, traversal, symlinks)
8. Environment-variable filtering (secret & loader stripping)
9. Subprocess errors and cancellation cleanup
10. Concurrent request limits (429 Too Many Requests and slot recovery)
11. Secret redaction across output, stderr, errors, and logs
12. Backward compatibility for legacy authorized {command, timeout} callers
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
from pathlib import Path
import signal
from typing import Any, Dict, List
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import pytest

from zelyr.server import node_api


TEST_TOKEN = "zelyr-remote-shell-test-token-9f4c2a8b1d"


class _FakeByteStream:
    """Async chunked stream drop-in for subprocess stdout/stderr."""

    def __init__(self, data: bytes = b"", *, chunk_size: int = 4096, hang_after_eof: bool = False) -> None:
        self._data = data
        self._offset = 0
        self._chunk_size = chunk_size
        self._hang_after_eof = hang_after_eof

    async def read(self, n: int = -1) -> bytes:
        size = self._chunk_size if n <= 0 else min(n, self._chunk_size)
        if self._offset >= len(self._data):
            if self._hang_after_eof:
                await asyncio.sleep(60.0)
            return b""
        chunk = self._data[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


class _FakeProcess:
    """Mock async subprocess recording calls cleanly without spawning real OS processes."""

    def __init__(
        self,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int = 0,
        pid: int = 999991,
        delay: float = 0.0,
        hang: bool = False,
        chunk_size: int = 4096,
        hang_stderr: bool = False,
    ) -> None:
        self.stdout = _FakeByteStream(stdout, chunk_size=chunk_size)
        self.stderr = _FakeByteStream(stderr, chunk_size=chunk_size, hang_after_eof=hang_stderr)
        self.returncode: int | None = None
        self._final_rc = returncode
        self.pid = pid
        self._delay = delay
        self._hang = hang
        self.killed = False
        self.waited = False

    async def wait(self) -> int:
        if self._hang:
            await asyncio.sleep(60.0)
        elif self._delay > 0:
            await asyncio.sleep(self._delay)
        self.waited = True
        self.returncode = self._final_rc
        return self._final_rc

    def kill(self) -> None:
        self.killed = True
        if self.returncode is None:
            self.returncode = -9


@pytest.fixture(autouse=True)
def _isolated_node_api_shell(tmp_path: Path, monkeypatch):
    """Isolate state_dir, auth token, concurrency counter, and prevent real subprocesses."""
    monkeypatch.setenv("ZELYR_TESTING", "1")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr(node_api, "state_dir", lambda: tmp_path)
    monkeypatch.setattr(node_api, "state_file", lambda *parts: tmp_path / parts[0])
    monkeypatch.setattr(node_api.state, "remote_shell_token", None)
    monkeypatch.delenv("ZELYR_REMOTE_SHELL_TOKEN", raising=False)
    monkeypatch.delenv("ZELYR_REMOTE_SHELL_TOKEN_FILE", raising=False)
    monkeypatch.delenv("ZELYR_REMOTE_SHELL_CWD", raising=False)
    monkeypatch.delenv("ZELYR_REMOTE_SHELL_ALLOWED_CWDS", raising=False)
    monkeypatch.delenv("ZELYR_REMOTE_SHELL_MAX_CONCURRENT", raising=False)
    monkeypatch.setitem(node_api._SHELL_CONCURRENCY_STATE, "active", 0)
    monkeypatch.setattr(node_api.app.router, "on_startup", [])
    monkeypatch.setattr(node_api.app.router, "on_shutdown", [])
    yield


def _auth_headers(token: str = TEST_TOKEN) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ── 1. Unauthenticated & Unconfigured Failures ────────────────────────────────


def test_unauthenticated_and_unconfigured_requests_rejected(tmp_path: Path, monkeypatch) -> None:
    spawned: List[str] = []

    async def _fail_spawn(*_a, **_kw):
        spawned.append("spawned")
        raise AssertionError("Subprocess must never spawn for unauthenticated requests")

    monkeypatch.setattr(asyncio, "create_subprocess_shell", _fail_spawn)
    monkeypatch.setenv("ZELYR_REMOTE_SHELL_TOKEN", TEST_TOKEN)

    client = TestClient(node_api.app, client=("127.0.0.1", 50000))

    # Missing header -> 401
    r_missing = client.post("/execute_shell", json={"command": "echo hi"})
    assert r_missing.status_code == 401
    assert r_missing.headers.get("www-authenticate") == "Bearer"

    # Non-Bearer header -> 401
    r_basic = client.post(
        "/execute_shell",
        headers={"Authorization": "Basic dXNlcjpwYXNz"},
        json={"command": "echo hi"},
    )
    assert r_basic.status_code == 401

    # Query-string token must NEVER authenticate
    r_query = client.post(f"/execute_shell?token={TEST_TOKEN}", json={"command": "echo hi"})
    assert r_query.status_code == 401

    # Missing/unconfigured server token -> fails closed with 503
    monkeypatch.delenv("ZELYR_REMOTE_SHELL_TOKEN", raising=False)
    r_unconf = client.post("/execute_shell", headers=_auth_headers(), json={"command": "echo hi"})
    assert r_unconf.status_code == 503

    # Weak short token (< 16 chars) -> fails closed with 503
    monkeypatch.setenv("ZELYR_REMOTE_SHELL_TOKEN", "short")
    r_short = client.post("/execute_shell", headers=_auth_headers("short"), json={"command": "echo hi"})
    assert r_short.status_code == 503

    # World-readable remote_shell.token file (0644) -> refused (503)
    monkeypatch.delenv("ZELYR_REMOTE_SHELL_TOKEN", raising=False)
    tok_file = tmp_path / "remote_shell.token"
    tok_file.write_text(TEST_TOKEN, encoding="utf-8")
    os.chmod(tok_file, 0o644)
    r_perm = client.post("/execute_shell", headers=_auth_headers(), json={"command": "echo hi"})
    assert r_perm.status_code == 503

    # Symlink remote_shell.token -> refused (503)
    real_tok = tmp_path / "real_tok"
    real_tok.write_text(TEST_TOKEN, encoding="utf-8")
    os.chmod(real_tok, 0o600)
    tok_file.unlink()
    tok_file.symlink_to(real_tok)
    r_sym = client.post("/execute_shell", headers=_auth_headers(), json={"command": "echo hi"})
    assert r_sym.status_code == 503

    assert spawned == []


# ── 2. Invalid Credentials & Untrusted / Spoofed IP Addresses ─────────────────


def test_invalid_credentials_and_ip_perimeter_checks(monkeypatch) -> None:
    spawned: List[str] = []
    digest_calls: List[tuple[bytes, bytes]] = []
    orig_compare = hmac.compare_digest

    def _spy_compare(a, b):
        digest_calls.append((a, b))
        return orig_compare(a, b)

    async def _mock_spawn(*_a, **_kw):
        spawned.append("spawned")
        return _FakeProcess(stdout=b"ok\n", returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_shell", _mock_spawn)
    monkeypatch.setattr(node_api.hmac, "compare_digest", _spy_compare)
    monkeypatch.setenv("ZELYR_REMOTE_SHELL_TOKEN", TEST_TOKEN)

    client = TestClient(node_api.app, client=("127.0.0.1", 50000))
    r_bad = client.post("/execute_shell", headers=_auth_headers("wrong-token-1234567890"), json={"command": "id"})
    assert r_bad.status_code == 401
    assert digest_calls, "Expected constant-time hmac.compare_digest check"

    # Untrusted / spoofed IPs MUST be rejected with 403 even with a valid Bearer token:
    # - 8.8.8.8 public Internet
    # - 100.1.2.3 public non-Tailscale 100.x IP (previously matched startswith("100.")!)
    # - 169.254.169.254 link-local cloud metadata (is_private=True in Python 3.11!)
    # - 0.0.0.0 unspecified (is_private=True in Python 3.11!)
    # - testclient non-IP host
    for bad_ip in ("8.8.8.8", "100.1.2.3", "169.254.169.254", "0.0.0.0", "testclient"):
        bad_client = TestClient(node_api.app, client=(bad_ip, 50000))
        r_ip = bad_client.post("/execute_shell", headers=_auth_headers(), json={"command": "id"})
        assert r_ip.status_code == 403, f"Expected 403 for untrusted IP {bad_ip!r}"

    assert spawned == []

    # Trusted Tailscale CGNAT (100.64.0.42), RFC 1918 LAN, ::1, and ::ffff:127.0.0.1 succeed with valid token
    for good_ip in ("127.0.0.1", "::1", "::ffff:127.0.0.1", "10.0.1.5", "192.168.1.50", "100.64.0.42"):
        good_client = TestClient(node_api.app, client=(good_ip, 50000))
        r_ok = good_client.post("/execute_shell", headers=_auth_headers(), json={"command": "echo ok"})
        assert r_ok.status_code == 200, f"Expected 200 for trusted IP {good_ip!r}"
        assert r_ok.json()["stdout"] == "ok\n"

    assert len(spawned) == 6


# ── 3 & 12. Authorized Token File & Backward-Compatible Legacy Clients ────────


def test_authorized_0600_token_file_and_legacy_client_compat(tmp_path: Path, monkeypatch) -> None:
    captured_calls: List[Dict[str, Any]] = []

    async def _mock_spawn(command: str, **kwargs: Any):
        captured_calls.append({"command": command, **kwargs})
        return _FakeProcess(stdout=b"service active\n", stderr=b"", returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_shell", _mock_spawn)

    tok_file = tmp_path / "remote_shell.token"
    tok_file.write_text(TEST_TOKEN + "\n", encoding="utf-8")
    os.chmod(tok_file, 0o600)

    client = TestClient(node_api.app, client=("100.64.5.10", 50000))
    # Legacy payload with only {command, timeout} (no cwd specified)
    resp = client.post(
        "/execute_shell",
        headers=_auth_headers(),
        json={"command": "systemctl status ollama | head -n 5", "timeout": 30},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["stdout"] == "service active\n"
    assert body["stderr"] == ""
    assert body["returncode"] == 0
    assert body["truncated"] is False
    assert captured_calls[0]["command"] == "systemctl status ollama | head -n 5"
    assert captured_calls[0]["stdin"] is asyncio.subprocess.DEVNULL
    assert captured_calls[0]["start_new_session"] is True


# ── 4 & 5. Malformed Bodies, Oversized Requests, and Bounded Output ───────────


def test_malformed_requests_and_oversized_payload_rejected(monkeypatch) -> None:
    spawned: List[str] = []

    async def _fail_spawn(*_a, **_kw):
        spawned.append("spawned")
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_shell", _fail_spawn)
    monkeypatch.setenv("ZELYR_REMOTE_SHELL_TOKEN", TEST_TOKEN)

    client = TestClient(node_api.app, client=("127.0.0.1", 50000))

    # Malformed JSON -> 400
    r_bad_json = client.post(
        "/execute_shell",
        headers={**_auth_headers(), "Content-Type": "application/json"},
        content=b"{invalid_json",
    )
    assert r_bad_json.status_code == 400

    # Non-object JSON list -> 422
    r_list = client.post("/execute_shell", headers=_auth_headers(), json=["echo", "hi"])
    assert r_list.status_code == 422

    # Invalid fields, extra keys, NUL byte, bad timeouts, empty command -> 422
    for bad_payload in (
        {"command": ""},
        {"command": "   "},
        {"command": "echo hi\x00whoami"},
        {"command": "echo hi", "timeout": 0},
        {"command": "echo hi", "timeout": -10},
        {"command": "echo hi", "timeout": 9999},
        {"command": "echo hi", "timeout": True},
        {"command": "echo hi", "timeout": "10"},
        {"command": "echo hi", "env": {"LD_PRELOAD": "/tmp/evil.so"}},
        {"command": "a" * (node_api.MAX_SHELL_COMMAND_BYTES + 1)},
    ):
        r_val = client.post("/execute_shell", headers=_auth_headers(), json=bad_payload)
        assert r_val.status_code == 422, f"Expected 422 for {bad_payload!r}"

    # Oversized raw request body (streamed) -> 413
    oversized = b'{"command": "' + (b"x" * (node_api.MAX_SHELL_REQUEST_BYTES + 100)) + b'"}'
    r_big = client.post(
        "/execute_shell",
        headers={**_auth_headers(), "Content-Type": "application/json"},
        content=oversized,
    )
    assert r_big.status_code == 413

    # Declared oversized Content-Length -> rejected immediately with 413 before reading stream
    r_hdr_big = client.post(
        "/execute_shell",
        headers={**_auth_headers(), "Content-Length": str(node_api.MAX_SHELL_REQUEST_BYTES + 1), "Content-Type": "application/json"},
        content=b'{"command": "echo hi"}',
    )
    assert r_hdr_big.status_code == 413

    assert spawned == []


def test_oversized_subprocess_output_truncated_and_reaped(monkeypatch) -> None:
    monkeypatch.setenv("ZELYR_REMOTE_SHELL_TOKEN", TEST_TOKEN)
    # Create output larger than MAX_SHELL_OUTPUT_BYTES; simulate stderr hanging so truncation must not stall
    proc = _FakeProcess(
        stdout=b"A" * (node_api.MAX_SHELL_OUTPUT_BYTES + 8192),
        stderr=b"err_line\n",
        returncode=0,
        chunk_size=4096,
        hang_stderr=True,
    )

    async def _mock_spawn(*_a, **_kw):
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_shell", _mock_spawn)
    killpg_spy = MagicMock()
    monkeypatch.setattr(node_api.os, "killpg", killpg_spy)
    monkeypatch.setattr(node_api.os, "getpgid", lambda pid: pid)

    client = TestClient(node_api.app, client=("127.0.0.1", 50000))
    resp = client.post("/execute_shell", headers=_auth_headers(), json={"command": "yes", "timeout": 10})
    assert resp.status_code == 200
    data = resp.json()
    assert data["truncated"] is True
    assert len(data["stdout"].encode("utf-8")) <= node_api.MAX_SHELL_OUTPUT_BYTES
    assert "OUTPUT TRUNCATED" in data["stderr"]
    assert proc.killed is True
    killpg_spy.assert_called_with(proc.pid, signal.SIGKILL)


# ── 6 & 9. Timeout, Process-Group Cleanup, Errors, and Cancellation ───────────


def test_execution_timeout_kills_process_group_and_releases_slot(monkeypatch) -> None:
    monkeypatch.setenv("ZELYR_REMOTE_SHELL_TOKEN", TEST_TOKEN)
    proc = _FakeProcess(hang=True, pid=888777)

    async def _mock_spawn(*_a, **_kw):
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_shell", _mock_spawn)
    killpg_calls: List[tuple[int, int]] = []
    monkeypatch.setattr(node_api.os, "killpg", lambda pgid, sig: killpg_calls.append((pgid, sig)))
    monkeypatch.setattr(node_api.os, "getpgid", lambda pid: pid)

    client = TestClient(node_api.app, client=("127.0.0.1", 50000))
    resp = client.post("/execute_shell", headers=_auth_headers(), json={"command": "sleep 60", "timeout": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["returncode"] == -1
    assert body["timed_out"] is True
    assert "timed out after 1 seconds" in body["stderr"]
    assert proc.killed is True
    assert killpg_calls == [(888777, signal.SIGKILL)]
    assert node_api._SHELL_CONCURRENCY_STATE["active"] == 0


@pytest.mark.asyncio
async def test_subprocess_spawn_error_and_cancellation_cleanup(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ZELYR_REMOTE_SHELL_TOKEN", TEST_TOKEN)

    class _MockRequest:
        client = MagicMock(host="127.0.0.1")
        headers = {"authorization": f"Bearer {TEST_TOKEN}"}

        async def stream(self):
            yield b'{"command": "echo secret_cmd_sk-secret1234567890", "timeout": 5}'

    # 1. Subprocess error redacts secret error text & releases concurrency slot
    async def _err_spawn(*_a, **_kw):
        raise OSError(f"boom with {TEST_TOKEN} and sk-abcdefghijklmnop")

    monkeypatch.setattr(asyncio, "create_subprocess_shell", _err_spawn)
    with pytest.raises(node_api.HTTPException) as exc_info:
        await node_api.execute_shell(_MockRequest())
    assert exc_info.value.status_code == 500
    assert TEST_TOKEN not in exc_info.value.detail
    assert "sk-abcdefghijklmnop" not in exc_info.value.detail
    assert node_api._SHELL_CONCURRENCY_STATE["active"] == 0

    # 2. Task cancellation terminates subprocess & releases concurrency slot cleanly
    proc = _FakeProcess(hang=True, pid=777666)

    async def _hang_spawn(*_a, **_kw):
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_shell", _hang_spawn)
    killpg_calls: List[int] = []
    monkeypatch.setattr(node_api.os, "killpg", lambda pgid, _sig: killpg_calls.append(pgid))
    monkeypatch.setattr(node_api.os, "getpgid", lambda pid: pid)

    task = asyncio.create_task(node_api.execute_shell(_MockRequest()))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert proc.killed is True
    assert killpg_calls == [777666]
    assert node_api._SHELL_CONCURRENCY_STATE["active"] == 0


# ── 7. Working-Directory Restrictions ─────────────────────────────────────────


def test_working_directory_restrictions(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ZELYR_REMOTE_SHELL_TOKEN", TEST_TOKEN)
    captured: List[Dict[str, Any]] = []

    async def _mock_spawn(command: str, **kwargs: Any):
        captured.append({"command": command, **kwargs})
        return _FakeProcess(stdout=b"pwd ok\n", returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_shell", _mock_spawn)

    workspace = tmp_path / "swarm_workspace"
    sub = workspace / "project"
    sub.mkdir(parents=True)
    outside = tmp_path / "outside_secret_dir"
    outside.mkdir()
    escape_link = workspace / "link_out"
    escape_link.symlink_to(outside)

    client = TestClient(node_api.app, client=("127.0.0.1", 50000))

    # Relative subdir inside swarm_workspace -> allowed
    r_ok = client.post("/execute_shell", headers=_auth_headers(), json={"command": "pwd", "cwd": "project"})
    assert r_ok.status_code == 200
    assert captured[-1]["cwd"] == str(sub.resolve())

    # Path traversal, outside dir, symlink escape -> 403
    for bad_cwd in ("../outside_secret_dir", str(outside), "/etc", str(escape_link)):
        r_bad = client.post("/execute_shell", headers=_auth_headers(), json={"command": "pwd", "cwd": bad_cwd})
        assert r_bad.status_code == 403, f"Expected 403 for cwd={bad_cwd!r}"

    # Missing directory -> 400
    r_missing = client.post("/execute_shell", headers=_auth_headers(), json={"command": "pwd", "cwd": "does_not_exist"})
    assert r_missing.status_code == 400
    assert len(captured) == 1


# ── 8. Environment-Variable Filtering ─────────────────────────────────────────


def test_sanitized_environment_strips_host_secrets(monkeypatch) -> None:
    monkeypatch.setenv("ZELYR_REMOTE_SHELL_TOKEN", TEST_TOKEN)
    monkeypatch.setenv("ZELYR_AUTH_TOKEN", "daemon-secret-value-xyz")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-openai-key-12345")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret-value")
    monkeypatch.setenv("LD_PRELOAD", "/tmp/injected.so")
    monkeypatch.setenv("PYTHONPATH", "/tmp/untrusted_py")

    captured_env: Dict[str, str] = {}

    async def _mock_spawn(_command: str, **kwargs: Any):
        captured_env.update(kwargs.get("env") or {})
        return _FakeProcess(stdout=b"ok", returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_shell", _mock_spawn)
    client = TestClient(node_api.app, client=("127.0.0.1", 50000))
    resp = client.post("/execute_shell", headers=_auth_headers(), json={"command": "env"})
    assert resp.status_code == 200

    for forbidden in (
        "ZELYR_REMOTE_SHELL_TOKEN",
        "ZELYR_AUTH_TOKEN",
        "OPENAI_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "LD_PRELOAD",
        "PYTHONPATH",
    ):
        assert forbidden not in captured_env
    assert captured_env["PAGER"] == "cat"
    assert captured_env["GIT_TERMINAL_PROMPT"] == "0"
    assert "PATH" in captured_env


# ── 10. Concurrent Request Limits ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_execution_limit_returns_429_and_recovers(monkeypatch) -> None:
    monkeypatch.setenv("ZELYR_REMOTE_SHELL_TOKEN", TEST_TOKEN)
    monkeypatch.setenv("ZELYR_REMOTE_SHELL_MAX_CONCURRENT", "2")

    gate = asyncio.Event()
    started = 0

    class _GatedProc(_FakeProcess):
        async def wait(self) -> int:
            nonlocal started
            started += 1
            await gate.wait()
            self.returncode = 0
            return 0

    async def _mock_spawn(*_a, **_kw):
        return _GatedProc()

    monkeypatch.setattr(asyncio, "create_subprocess_shell", _mock_spawn)

    def _mk_req():
        req = MagicMock()
        req.client = MagicMock(host="127.0.0.1")
        req.headers = {"authorization": f"Bearer {TEST_TOKEN}"}

        async def _stream():
            yield b'{"command": "echo concurrent", "timeout": 10}'

        req.stream = _stream
        return req

    t1 = asyncio.create_task(node_api.execute_shell(_mk_req()))
    t2 = asyncio.create_task(node_api.execute_shell(_mk_req()))
    await asyncio.sleep(0.05)
    assert node_api._SHELL_CONCURRENCY_STATE["active"] == 2

    # 3rd concurrent request MUST fail fast with 429 Too Many Requests
    with pytest.raises(node_api.HTTPException) as exc_429:
        await node_api.execute_shell(_mk_req())
    assert exc_429.value.status_code == 429
    assert exc_429.value.headers.get("Retry-After") == "1"

    # Unblock active executions and confirm slot recovery
    gate.set()
    r1, r2 = await asyncio.gather(t1, t2)
    assert r1["returncode"] == 0 and r2["returncode"] == 0
    assert node_api._SHELL_CONCURRENCY_STATE["active"] == 0

    r3 = await node_api.execute_shell(_mk_req())
    assert r3["returncode"] == 0


# ── 11. Secret Redaction Across Output, Errors, and Logs ──────────────────────


def test_secret_redaction_in_output_and_audit_logs(monkeypatch, caplog) -> None:
    monkeypatch.setenv("ZELYR_REMOTE_SHELL_TOKEN", TEST_TOKEN)
    monkeypatch.setenv("MY_CUSTOM_API_KEY", "custom-api-secret-xyz987")

    async def _mock_spawn(*_a, **_kw):
        return _FakeProcess(
            stdout=(
                f"token={TEST_TOKEN}\n"
                "openai=sk-1234567890abcdef1234\n"
                "gh=ghp_1234567890abcdefghij123456\n"
                "env_leak=custom-api-secret-xyz987\n"
            ).encode("utf-8"),
            stderr=b"Authorization: Bearer super_secret_bearer_token_999\n",
            returncode=0,
        )

    monkeypatch.setattr(asyncio, "create_subprocess_shell", _mock_spawn)
    client = TestClient(node_api.app, client=("127.0.0.1", 50000))

    with caplog.at_level(logging.INFO, logger="zelyr.node"):
        resp = client.post(
            "/execute_shell",
            headers=_auth_headers(),
            json={"command": "API_KEY=cmd_inline_secret_abc123 echo sk-1234567890abcdef1234"},
        )

    assert resp.status_code == 200
    body = resp.json()
    for secret_val in (
        TEST_TOKEN,
        "sk-1234567890abcdef1234",
        "ghp_1234567890abcdefghij123456",
        "custom-api-secret-xyz987",
    ):
        assert secret_val not in body["stdout"]
        assert secret_val not in caplog.text

    assert "super_secret_bearer_token_999" not in body["stderr"]
    assert "cmd_inline_secret_abc123" not in caplog.text
    assert "sha256=" in caplog.text
