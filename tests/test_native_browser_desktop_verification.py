"""Focused offline regression tests for native browser and desktop verification contracts."""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from scripts.cross_surface_benchmark import (
    DEFAULT_DESKTOP_TITLE,
    INITIAL_RECORDS,
    build_cross_surface_verify_spec,
    evaluate_cross_surface_run,
    setup_cross_surface_workspace,
)
from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.cli.main import _execute_headless_run
from zelyr.engine.execution_ledger import (
    ActionState,
    CentralExecutionLedger,
    ExecutionStatus,
    TurnExecutionLedger,
    VerificationStatus,
    parse_acceptance_criteria,
    resolve_root_execution_outcome,
)
from zelyr.engine.hyprland_ipc import HyprlandUnavailableError
from zelyr.engine.verifier import (
    OutcomeVerifier,
    VERDICT_FAIL,
    VERDICT_PASS,
    VERDICT_UNSUPPORTED,
    VERDICT_UNVERIFIED,
    VerificationResult,
)
from zelyr.tools.browser_native_host import (
    ALLOWED_NATIVE_ACTIONS,
    NativeBrowserResult,
    validate_browser_command,
)
from zelyr.tools.verify_outcome import VerifyOutcomeTool


class _FakeNativeBrowserClient:
    """In-memory NativeBrowserClient double recording sent verify_dom_state commands."""

    def __init__(self, responder: Any) -> None:
        self.responder = responder
        self.calls: List[Dict[str, Any]] = []

    def is_socket_present(self) -> bool:
        return True

    async def send_command(self, action: str, timeout: float = 10.0, **params: Any) -> NativeBrowserResult:
        self.calls.append({"action": action, "timeout": timeout, **params})
        resp = self.responder(action, params) if callable(self.responder) else dict(self.responder)
        ok = bool(resp.get("success", False))
        return NativeBrowserResult(success=ok, output=resp, error=resp.get("error"))


class _FakeHyprlandClient:
    """Read-only HyprlandIPCClient double; fails loudly if any write/dispatch method is attempted."""

    def __init__(
        self,
        *,
        clients: Optional[List[Dict[str, Any]]] = None,
        active: Optional[Dict[str, Any]] = None,
        available: bool = True,
        raise_exc: Optional[Exception] = None,
    ) -> None:
        self._clients = list(clients or [])
        self._active = dict(active or {})
        self._available = available
        self._raise_exc = raise_exc
        self.dispatch_calls: List[Any] = []

    def can_connect(self) -> bool:
        return self._available

    async def get_clients(self) -> List[Dict[str, Any]]:
        if self._raise_exc is not None:
            raise self._raise_exc
        return list(self._clients)

    async def get_active_window(self) -> Dict[str, Any]:
        if self._raise_exc is not None:
            raise self._raise_exc
        return dict(self._active)

    async def dispatch(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        self.dispatch_calls.append((args, kwargs))
        raise AssertionError("Desktop verification MUST NOT invoke compositor dispatch/input!")


# ── 1. Browser Verification & Boundary Tests ─────────────────────────────────


def test_native_host_allows_and_validates_verify_dom_state() -> None:
    assert "verify_dom_state" in ALLOWED_NATIVE_ACTIONS

    ok, _ = validate_browser_command({
        "id": "v1",
        "action": "verify_dom_state",
        "mode": "visible_text",
        "selector": "#status-badge",
        "expected_url": "http://127.0.0.1:43210/records/REC-204",
        "tab_id": 12,
    })
    assert ok is True

    # Reject forbidden cookie / localStorage / eval / oversized selectors & bad modes
    for bad_payload in (
        {"id": "b1", "action": "verify_dom_state", "mode": "eval"},
        {"id": "b2", "action": "verify_dom_state", "mode": "visible_text", "selector": ""},
        {"id": "b3", "action": "verify_dom_state", "mode": "visible_text", "selector": "document.cookie"},
        {"id": "b4", "action": "verify_dom_state", "mode": "field_value", "selector": "window.localStorage"},
        {"id": "b5", "action": "verify_dom_state", "mode": "url", "expected_url": "javascript:alert(1)"},
        {"id": "b6", "action": "verify_dom_state", "mode": "url", "tab_id": 0},
        {"id": "b7", "action": "verify_dom_state", "mode": "visible_text", "selector": "a" * 2048},
    ):
        bad_ok, _ = validate_browser_command(bad_payload)
        assert bad_ok is False, f"Expected fail-closed validation rejection for {bad_payload}"


@pytest.mark.asyncio
async def test_browser_checks_pass_across_url_title_dom_text_state_and_value() -> None:
    client = _FakeNativeBrowserClient(lambda _act, params: {
        "success": True,
        "tab_id": 42,
        "url": "http://127.0.0.1:45678/records/REC-204",
        "origin": "http://127.0.0.1:45678",
        "title": "Record REC-204 Approval",
        "ready_state": "complete",
        "match_count": 0 if params.get("selector") == "#missing-el" else 1,
        "visible_count": 0 if params.get("selector") == "#missing-el" else 1,
        "matches": [] if params.get("selector") == "#missing-el" else [{
            "tag": "select" if params.get("selector") == "#tier" else "button",
            "visible": True,
            "disabled": True,
            "checked": True,
            "selected": True,
            "focused": False,
            "sensitive_redacted": False,
            "visible_text": "APPROVED",
            "text_content": "APPROVED",
            "value": "prod-tier",
            "selected_options": [{"value": "prod-tier", "text": "Production Tier"}],
        }],
    })

    checks = [
        ("browser_url", "http://127.0.0.1:45678/records/REC-204", None),
        ("browser_navigation", "http://127.0.0.1:45678/records/REC-204", "http://127.0.0.1:45678/records/REC-204"),
        ("browser_origin", "http://127.0.0.1:45678", None),
        ("browser_title", "active", "REC-204"),
        ("browser_element_present", "#status-badge", "visible"),
        ("browser_element_absent", "#missing-el", None),
        ("browser_visible_text", "#status-badge", "APPROVED"),
        ("browser_element_state", "#approve-btn", "disabled"),
        ("browser_element_state", "#agree", "checked"),
        ("browser_element_state", "#tier", "selected_option=prod-tier"),
        ("browser_field_value", "#tier", "prod-tier"),
    ]
    for chk, tgt, exp in checks:
        res = await OutcomeVerifier.verify_async(chk, tgt, expected=exp, browser_client=client)
        assert res.success is True, f"{chk} failed: {res.error}"
        assert res.result == VERDICT_PASS
        assert res.evidence_source == "browser_native_messaging"
        assert res.evidence_collected is True
        assert res.criterion["check"] in (chk, "browser_url")


@pytest.mark.asyncio
async def test_browser_checks_fail_closed_on_disconnected_stale_wrong_state_and_secrets(monkeypatch) -> None:
    monkeypatch.delenv("ZELYR_BROWSER_NATIVE_SOCKET", raising=False)
    # 1. Extension disconnected -> UNVERIFIED, evidence_collected=False
    disc_res = await OutcomeVerifier.verify_async("browser_visible_text", "#status-badge", expected="APPROVED")
    assert disc_res.success is False
    assert disc_res.result == VERDICT_UNVERIFIED
    assert disc_res.evidence_collected is False

    # 2. Tab loading / stale / closed -> UNVERIFIED
    stale_client = _FakeNativeBrowserClient({
        "success": False,
        "stale": True,
        "error_type": "PAGE_STALE_LOADING",
        "error": "Target tab 7 is still loading.",
    })
    stale_res = await OutcomeVerifier.verify_async("browser_visible_text", "#status-badge", expected="APPROVED", browser_client=stale_client)
    assert stale_res.success is False
    assert stale_res.result == VERDICT_UNVERIFIED
    assert stale_res.evidence_collected is False

    # 3. Context mismatch (wrong URL/page active) -> FAIL with observed URL
    ctx_client = _FakeNativeBrowserClient({
        "success": False,
        "context_mismatch": True,
        "tab_id": 7,
        "url": "http://127.0.0.1:45678/",
        "origin": "http://127.0.0.1:45678",
        "title": "Dashboard",
        "error": "Active tab URL does not match requested context URL.",
    })
    ctx_res = await OutcomeVerifier.verify_async(
        "browser_visible_text",
        "#status-badge",
        expected="APPROVED",
        expected_url="http://127.0.0.1:45678/records/REC-204",
        browser_client=ctx_client,
    )
    assert ctx_res.success is False
    assert ctx_res.result == VERDICT_FAIL
    assert ctx_res.evidence_collected is True

    # 4. Hidden element does NOT pass browser_visible_text
    hidden_client = _FakeNativeBrowserClient({
        "success": True,
        "tab_id": 8,
        "url": "http://127.0.0.1:45678/records/REC-204",
        "origin": "http://127.0.0.1:45678",
        "title": "Detail",
        "match_count": 1,
        "visible_count": 0,
        "matches": [{"tag": "span", "visible": False, "visible_text": "", "text_content": "APPROVED"}],
    })
    hidden_res = await OutcomeVerifier.verify_async("browser_visible_text", "#status-badge", expected="APPROVED", browser_client=hidden_client)
    assert hidden_res.success is False
    assert hidden_res.result == VERDICT_FAIL

    # 5. Sensitive password field never exposes raw value & fails closed
    secret_client = _FakeNativeBrowserClient({
        "success": True,
        "tab_id": 8,
        "url": "http://127.0.0.1:45678/login",
        "origin": "http://127.0.0.1:45678",
        "title": "Login",
        "match_count": 1,
        "visible_count": 1,
        "matches": [{"tag": "input", "input_type": "password", "visible": True, "sensitive_redacted": True, "value": None}],
    })
    secret_res = await OutcomeVerifier.verify_async("browser_field_value", "input[type='password']", expected="hunter2", browser_client=secret_client)
    assert secret_res.success is False
    assert secret_res.status == "rejected"
    assert "hunter2" not in json.dumps(secret_res.observed)

    # 6. Invalid selector fails closed even on browser_element_absent
    invalid_sel_client = _FakeNativeBrowserClient({
        "success": False,
        "invalid_selector": True,
        "url": "http://127.0.0.1:45678/",
        "error": "Invalid CSS selector '###bad': SyntaxError",
    })
    absent_err = await OutcomeVerifier.verify_async("browser_element_absent", "###bad", browser_client=invalid_sel_client)
    assert absent_err.success is False
    assert absent_err.status == "invalid_spec"
    assert absent_err.evidence_collected is False


@pytest.mark.asyncio
async def test_browser_verification_works_through_real_native_socket_server(tmp_path: Path, monkeypatch) -> None:
    """Exercise VerifyOutcomeTool + OutcomeVerifier over a real AF_UNIX Native Messaging socket."""
    sock_path = tmp_path / "native_browser.sock"
    monkeypatch.setenv("ZELYR_BROWSER_NATIVE_SOCKET", str(sock_path))

    async def handle_conn(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        line = await reader.readline()
        req = json.loads(line.decode("utf-8"))
        assert req["action"] == "verify_dom_state"
        reply = {
            "id": req["id"],
            "success": True,
            "action": "verify_dom_state",
            "tab_id": 3,
            "url": "http://127.0.0.1:50001/records/REC-204",
            "origin": "http://127.0.0.1:50001",
            "title": "REC-204 Detail",
            "match_count": 1,
            "visible_count": 1,
            "matches": [{"tag": "strong", "visible": True, "visible_text": "APPROVED"}],
        }
        writer.write((json.dumps(reply) + "\n").encode("utf-8"))
        await writer.drain()
        writer.close()

    server = await asyncio.start_unix_server(handle_conn, path=str(sock_path))
    try:
        tool = VerifyOutcomeTool()
        tr = await tool.execute(
            check="browser_visible_text",
            target="#status-badge",
            expected="APPROVED",
            url="http://127.0.0.1:50001/records/REC-204",
        )
        assert tr.success is True
        vmeta = tr.metadata["verification"]
        assert vmeta["result"] == VERDICT_PASS
        assert vmeta["evidence_source"] == "browser_native_messaging"
        assert vmeta["evidence_collected"] is True
        assert vmeta["observed"]["visible_text"] == "APPROVED"
    finally:
        server.close()
        await server.wait_closed()


# ── 2. Desktop Verification & Privacy / Honesty Tests ────────────────────────


@pytest.mark.asyncio
async def test_desktop_compositor_checks_are_read_only_privacy_scoped_and_honest() -> None:
    hypr = _FakeHyprlandClient(
        clients=[
            {
                "address": "0xdeadbeef",
                "title": "Secret Banking - Personal Private Email",
                "class": "private-mail",
                "workspace": {"id": 1},
                "mapped": True,
            },
            {
                "address": "0xabc123",
                "title": "Zelyr Cross-Surface Confirmation",
                "class": "python3",
                "workspace": {"id": 15},
                "mapped": True,
            },
        ],
        active={
            "address": "0xdeadbeef",
            "title": "Secret Banking - Personal Private Email",
            "class": "private-mail",
            "workspace": {"id": 1},
        },
    )

    exists_res = await OutcomeVerifier.verify_async(
        "desktop_window_exists",
        "Zelyr Cross-Surface Confirmation",
        hyprland_client=hypr,
    )
    assert exists_res.success is True
    assert exists_res.result == VERDICT_PASS
    assert exists_res.evidence_source == "hyprland_ipc"
    assert exists_res.evidence_collected is True
    assert "Secret Banking" not in json.dumps(exists_res.observed)
    assert hypr.dispatch_calls == []

    title_res = await OutcomeVerifier.verify_async(
        "desktop_window_title",
        "0xabc123",
        expected="Cross-Surface Confirmation",
        hyprland_client=hypr,
    )
    assert title_res.success is True
    assert title_res.result == VERDICT_PASS

    # Active window mismatches target -> FAIL and MUST NOT leak private unrelated window title
    active_fail = await OutcomeVerifier.verify_async(
        "desktop_active_window",
        "Zelyr Cross-Surface Confirmation",
        hyprland_client=hypr,
    )
    assert active_fail.success is False
    assert active_fail.result == VERDICT_FAIL
    assert active_fail.evidence_collected is True
    assert active_fail.observed["focus_only"] is True
    assert active_fail.observed["semantic_outcome_verified"] is False
    assert "Secret Banking" not in json.dumps(active_fail.observed)
    assert "Secret Banking" not in (active_fail.error or "")

    # Unavailable compositor -> UNVERIFIED
    hypr_down = _FakeHyprlandClient(available=False)
    down_res = await OutcomeVerifier.verify_async(
        "desktop_window_exists",
        "Zelyr Cross-Surface Confirmation",
        hyprland_client=hypr_down,
    )
    assert down_res.success is False
    assert down_res.result == VERDICT_UNVERIFIED
    assert down_res.evidence_collected is False

    # IPC error -> UNVERIFIED
    hypr_err = _FakeHyprlandClient(available=True, raise_exc=HyprlandUnavailableError("socket gone"))
    err_res = await OutcomeVerifier.verify_async("desktop_active_window", "Win", hyprland_client=hypr_err)
    assert err_res.success is False
    assert err_res.result == VERDICT_UNVERIFIED
    assert err_res.evidence_collected is False


@pytest.mark.asyncio
async def test_desktop_semantic_widget_check_returns_unsupported_and_focus_cannot_promote_root() -> None:
    class _UnavailableA11yBackend:
        backend_name = "atspi2_dbus_jeepney"

        def is_available(self):
            return False, "AT-SPI2 session bus unavailable in test double"

        def query_widget(self, **_kwargs):
            raise AssertionError("query_widget must not run when backend is unavailable")

    sem_res = await OutcomeVerifier.verify_async(
        "desktop_visible_text",
        "Zelyr Cross-Surface Confirmation",
        selector="#statusLabel",
        expected="CONFIRMED:REC-204",
        desktop_accessibility_backend=_UnavailableA11yBackend(),
    )
    assert sem_res.success is False
    assert sem_res.result == VERDICT_UNSUPPORTED
    assert sem_res.status == "unsupported"
    assert sem_res.evidence_collected is False
    assert sem_res.evidence_source == "desktop_accessibility"

    # Window focus verification alone MUST NOT promote root without explicit --verify criteria
    ledger = TurnExecutionLedger()
    ledger.record_proposed("call_1", "verify_outcome", {"check": "desktop_active_window", "target": "Zelyr Cross-Surface Confirmation"})
    ledger.record_dispatched("call_1")
    ledger.record_observation("call_1", "success", {"status": "success", "verified": True}, verified=True)
    ledger.record_target_verified("Zelyr Cross-Surface Confirmation", check_type="desktop_active_window")
    outcome = resolve_root_execution_outcome(
        prompt="Confirm desktop window",
        ledger=ledger,
        final_text="Focused window and done.",
    )
    assert outcome.verification_status != VerificationStatus.VERIFIED
    assert outcome.execution_status == ExecutionStatus.COMPLETED_UNVERIFIED


# ── 3. Lifecycle, Pre-Execution Rejection & Benchmark Integrity Tests ─────────


@pytest.mark.asyncio
async def test_invalid_verify_specs_rejected_before_task_execution(tmp_path: Path, capsys) -> None:
    invalid_specs = [
        "   ",
        "[not valid json",
        "unknown_check_type:something",
        "browser_visible_text:#status-badge",
        "browser_visible_text:document.cookie:abc",
        "desktop_window_title:MyWindow",
        "desktop_visible_text:MyWindow:CONFIRMED",
    ]
    for spec in invalid_specs:
        stream_calls = 0

        async def mock_stream(_payload):
            nonlocal stream_calls
            stream_calls += 1
            yield {"choices": [{"delta": {"content": "Should never execute"}, "finish_reason": "stop"}]}

        central = CentralExecutionLedger(db_path=tmp_path / f"inv_{abs(hash(spec))}.db", raise_on_persistence_error=True)
        orch = NativeOrchestrator(central_ledger=central, stream_provider=mock_stream, max_steps=4)
        try:
            rc = await _execute_headless_run("Run task", orchestrator=orch, verify_spec=spec, output_json=True)
        finally:
            central.close()

        assert rc == 2, f"Expected rc=2 for invalid spec {spec!r}"
        assert stream_calls == 0, f"Model MUST NOT run for invalid verify spec {spec!r}"
        payload = json.loads(capsys.readouterr().out.strip())
        assert payload["invalid_verification_spec"] is True
        assert payload["root_execution_status"] == "failed"
        assert payload["root_verification_status"] == "verification_failed"


@pytest.mark.asyncio
async def test_unsupported_criterion_blocks_root_verified_status(tmp_path: Path, capsys, monkeypatch) -> None:
    from zelyr.engine.desktop_accessibility import AtspiDbusAccessibilityBackend

    monkeypatch.setattr(
        AtspiDbusAccessibilityBackend,
        "is_available",
        lambda _self: (False, "AT-SPI2 disabled in offline test"),
    )
    out_file = tmp_path / "ok.txt"
    out_file.write_text("READY\n", encoding="utf-8")
    spec = json.dumps([
        {"check": "file_contains", "target": str(out_file), "expected": "READY"},
        {
            "check": "desktop_visible_text",
            "window": "Zelyr Cross-Surface Confirmation",
            "selector": "#statusLabel",
            "expected": "CONFIRMED",
        },
    ])

    async def mock_stream(_payload):
        yield {"choices": [{"delta": {"content": "Finished everything!"}, "finish_reason": "stop"}]}

    central = CentralExecutionLedger(db_path=tmp_path / "unsup.db", raise_on_persistence_error=True)
    orch = NativeOrchestrator(central_ledger=central, stream_provider=mock_stream, max_steps=4)
    try:
        rc = await _execute_headless_run("Verify file and desktop", orchestrator=orch, verify_spec=spec, output_json=True)
    finally:
        central.close()

    assert rc == 2
    cli_json = json.loads(capsys.readouterr().out.strip())
    assert cli_json["root_execution_status"] == "failed"
    assert cli_json["root_verification_status"] == "verification_failed"
    assert cli_json["verification"]["success"] is False
    assert cli_json["verification"]["result"] == VERDICT_UNSUPPORTED
    assert cli_json["verification"]["checks"][0]["result"] == VERDICT_PASS
    assert cli_json["verification"]["checks"][1]["result"] == VERDICT_UNSUPPORTED


@pytest.mark.asyncio
async def test_browser_failure_alongside_filesystem_pass_fails_closed(tmp_path: Path, capsys, monkeypatch) -> None:
    report_json = tmp_path / "report.json"
    report_json.write_text(json.dumps({"record_id": "REC-204", "status": "APPROVED"}), encoding="utf-8")
    monkeypatch.delenv("ZELYR_BROWSER_NATIVE_SOCKET", raising=False)

    spec = json.dumps([
        {"check": "file_contains", "target": str(report_json), "expected": "APPROVED"},
        {"check": "browser_visible_text", "target": "#status-badge", "expected": "APPROVED"},
    ])

    async def mock_stream(_payload):
        yield {"choices": [{"delta": {"content": "Report written and approved!"}, "finish_reason": "stop"}]}

    central = CentralExecutionLedger(db_path=tmp_path / "mixed.db", raise_on_persistence_error=True)
    orch = NativeOrchestrator(central_ledger=central, stream_provider=mock_stream, max_steps=4)
    try:
        rc = await _execute_headless_run("Approve and write report", orchestrator=orch, verify_spec=spec, output_json=True)
    finally:
        central.close()

    assert rc == 2
    cli_json = json.loads(capsys.readouterr().out.strip())
    assert cli_json["root_execution_status"] == "failed"
    assert cli_json["verification"]["success"] is False
    assert cli_json["verification"]["result"] == VERDICT_UNVERIFIED
    assert cli_json["verification"]["checks"][0]["result"] == VERDICT_PASS
    assert cli_json["verification"]["checks"][1]["result"] == VERDICT_UNVERIFIED


@pytest.mark.asyncio
async def test_later_browser_and_desktop_mutations_invalidate_prior_verifications(tmp_path: Path, monkeypatch) -> None:
    """If step 1 verifies browser/desktop state and step 2 mutates browser/desktop without re-verifying,
    prior verifications MUST be invalidated."""
    sock_path = tmp_path / "sock_inv.sock"
    monkeypatch.setenv("ZELYR_BROWSER_NATIVE_SOCKET", str(sock_path))

    async def handle_conn(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        line = await reader.readline()
        req = json.loads(line.decode("utf-8"))
        reply = {
            "id": req["id"],
            "success": True,
            "action": "verify_dom_state",
            "tab_id": 1,
            "url": "http://127.0.0.1:45555/records/REC-204",
            "origin": "http://127.0.0.1:45555",
            "title": "Detail",
            "match_count": 1,
            "visible_count": 1,
            "matches": [{"tag": "strong", "visible": True, "visible_text": "APPROVED"}],
        }
        writer.write((json.dumps(reply) + "\n").encode("utf-8"))
        await writer.drain()
        writer.close()

    server = await asyncio.start_unix_server(handle_conn, path=str(sock_path))
    turn = 0

    async def stream_fn(payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        nonlocal turn
        turn += 1
        seq = [
            ("v1", "verify_outcome", {"check": "browser_visible_text", "target": "#status-badge", "expected": "APPROVED"}),
            ("m1", "interact_with_browser", {"action": "navigate_url", "url": "http://127.0.0.1:45555/other"}),
        ]
        if turn <= len(seq):
            cid, name, args = seq[turn - 1]
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": cid,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }]
                    },
                    "finish_reason": "tool_calls",
                }]
            }
        else:
            yield {"choices": [{"delta": {"content": "Navigated elsewhere."}, "finish_reason": "stop"}]}

    async def exec_fn(name: str, **kw: Any) -> str:
        if name == "verify_outcome":
            tr = await VerifyOutcomeTool().execute(**kw)
            return tr.output or tr.error or ""
        return json.dumps({"status": "success", "success": True})

    central = CentralExecutionLedger(db_path=tmp_path / "invalidate.db", raise_on_persistence_error=True)
    orch = NativeOrchestrator(central_ledger=central, stream_provider=stream_fn, tool_executor=exec_fn, max_steps=6)
    try:
        chunks = [c async for c in orch.generate_stream({"model": "test", "messages": [{"role": "user", "content": "Check then navigate."}]})]
        assert chunks
        ledger = orch.current_ledger
        assert ledger is not None
        assert not any(vt.startswith("browser:") for vt in ledger.get_verified_targets())
        vrec = ledger.get_record("v1")
        assert vrec is not None
        assert vrec.state == ActionState.UNVERIFIED
        assert vrec.verified is False
    finally:
        central.close()
        server.close()
        await server.wait_closed()


def test_desktop_surface_invalidation_direct_ledger() -> None:
    ledger = TurnExecutionLedger()
    ledger.record_proposed(
        "dv1",
        "verify_outcome",
        {"check": "desktop_window_exists", "target": "Confirmation"},
    )
    ledger.record_dispatched("dv1")
    ledger.record_observation("dv1", "success", {"status": "success", "verified": True}, verified=True)
    ledger.record_target_verified("Confirmation", check_type="desktop_window_exists")
    assert ledger.is_target_verified("Confirmation", check_type="desktop_window_exists") is True
    ledger.invalidate_surface_verifications("desktop")
    assert ledger.is_target_verified("Confirmation", check_type="desktop_window_exists") is False
    rec = ledger.get_record("dv1")
    assert rec is not None and rec.state == ActionState.UNVERIFIED


def test_cross_surface_benchmark_spec_and_independent_external_validation(tmp_path: Path) -> None:
    ws_dir = tmp_path / "ws"
    ws_info = setup_cross_surface_workspace(ws_dir, "http://127.0.0.1:43210", requires_desktop=True)
    spec_str = build_cross_surface_verify_spec(
        ws_info,
        base_url="http://127.0.0.1:43210",
        requires_desktop=True,
        verify_mode="cross_surface",
    )
    parsed = parse_acceptance_criteria(spec_str)
    checks = [c["check"] for c in parsed]
    assert "file_json" in checks
    assert "file_contains" in checks
    assert "browser_url" in checks
    assert "browser_element_present" in checks
    assert "browser_visible_text" in checks
    assert "browser_element_state" in checks
    assert "desktop_window_exists" in checks
    # External oracle URL (/state) MUST NEVER be embedded inside internal verification criteria
    assert "/state" not in spec_str
    assert "expected_result.json" not in spec_str

    # Even if internal multi-surface checks pass, external validator still independently rejects forged confirmation token / unapproved API
    exp_rec = copy.deepcopy(INITIAL_RECORDS["REC-204"])
    exp_rec["status"] = "APPROVED"
    forged = copy.deepcopy(exp_rec)
    forged["confirmation_token"] = "FAKE-TOK"
    ws_info["report_json_path"].write_text(json.dumps(forged, indent=2), encoding="utf-8")
    ws_info["report_md_path"].write_text("# Report\n\n- REC-204 billing-ledger fin-core eu-central-1 98 FAKE-TOK APPROVED\n", encoding="utf-8")
    web_audit = {
        "records": {**copy.deepcopy(INITIAL_RECORDS), "REC-204": exp_rec},
        "requests": [
            {"method": "GET", "path": "/", "status": 200},
            {"method": "GET", "path": "/records/REC-204", "status": 200},
            {"method": "POST", "path": "/api/records/REC-204/approve", "status": 200},
            {"method": "GET", "path": "/api/records/REC-204", "status": 200},
        ],
    }
    res = evaluate_cross_surface_run(
        scenario="normal",
        run_info={
            "returncode": 0,
            "verify_spec": spec_str,
            "cli_json": {
                "root_execution_status": "completed",
                "root_verification_status": "verified",
                "verification": {
                    "success": True,
                    "scope": {"surfaces": ["filesystem", "browser", "desktop_compositor"], "overall_result": "PASS"},
                    "checks": [
                        {"check": "file_json", "success": True, "result": "PASS"},
                        {"check": "file_contains", "success": True, "result": "PASS"},
                        {"check": "browser_url", "success": True, "result": "PASS"},
                        {"check": "desktop_window_exists", "success": True, "result": "PASS"},
                    ],
                },
                "execution_ledger": {
                    "has_unresolved_failures": False,
                    "unresolved_failures": [],
                    "records": [
                        {"tool_name": "interact_with_browser", "arguments": {"action": "open_tab"}, "state": "success"},
                        {"tool_name": "execute_command", "arguments": {"command": "curl -sS http://127.0.0.1:43210/api/records/REC-204"}, "state": "success"},
                        {"tool_name": "desktop_control", "arguments": {"action": "keyboard_type"}, "state": "success"},
                        {"tool_name": "capture_screen", "arguments": {"target": "window"}, "state": "success"},
                    ],
                },
            },
            "ledger": {"status": "completed", "verification_status": "verified", "verifications": [{"status": "verified"}]},
        },
        web_audit=web_audit,
        ws_info=ws_info,
        desktop_after={"applied_text": "WRONG-DESKTOP-TEXT", "status_text": "PENDING", "key_presses": 1},
    )
    assert res["internal_verification"]["scope"] == "filesystem+browser+desktop_compositor"
    assert "web_dom" not in res["internal_verification"]["unsupported_surfaces"]
    assert "desktop_widget_semantic_text" in res["internal_verification"]["unsupported_surfaces"]
    assert res["internal_verification"]["passed"] is True
    assert res["external_validation_status"] == "FAIL"
    assert res["benchmark_status"] == "FAIL"
