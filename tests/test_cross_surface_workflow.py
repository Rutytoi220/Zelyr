"""Offline regression & contract tests for Zelyr Cross-Surface Autonomous Workflows."""

from __future__ import annotations

import copy
import http.client
import json
import urllib.parse
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.cross_surface_benchmark import (
    CrossSurfaceWebApp,
    DEFAULT_DESKTOP_TITLE,
    DEFAULT_DISTRACTION_TITLE,
    INITIAL_RECORDS,
    build_cross_surface_prompt,
    evaluate_cross_surface_run,
    setup_cross_surface_workspace,
)
from zelyr.agents.actuator_arbiter import ActuatorArbiter
from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.core.plugins import filter_tool_schemas_by_tier, get_tool_schemas, load_plugins
from zelyr.engine.error_classifier import ErrorClassifier
from zelyr.engine.execution_ledger import (
    ExecutionStatus,
    TurnExecutionLedger,
    VerificationStatus,
    _is_inspection_action,
    resolve_root_execution_outcome,
)
from zelyr.tools.browser_cdp import InteractWithBrowserTool


def _make_base_eval_inputs(tmp_path: Path, *, approved: bool = True):
    ws_info = setup_cross_surface_workspace(tmp_path / "ws", "http://127.0.0.1:43210")
    recs = copy.deepcopy(INITIAL_RECORDS)
    if approved:
        recs["REC-204"]["status"] = "APPROVED"
    exp = ws_info["expected_result"]["expected_record"]
    ws_info["report_json_path"].write_text(json.dumps(exp, indent=2), encoding="utf-8")
    ws_info["report_md_path"].write_text(
        f"# Verification Report\n\n"
        f"- Record: `{exp['record_id']}`\n"
        f"- Service: `{exp['service']}`\n"
        f"- Owner: `{exp['owner']}`\n"
        f"- Region: `{exp['region']}`\n"
        f"- Score: `{exp['score']}`\n"
        f"- Token: `{exp['confirmation_token']}`\n"
        f"- Status: `{exp['status']}`\n",
        encoding="utf-8",
    )
    web_audit = {
        "target_record": "REC-204",
        "fail_browser_dom": False,
        "dom_invalidation_consumed": False,
        "dom_invalidated_events": [],
        "records": recs,
        "requests": [
            {"idx": 0, "ts": 100.0, "method": "GET", "path": "/", "status": 200},
            {"idx": 1, "ts": 101.0, "method": "GET", "path": "/records/REC-204", "status": 200},
            {"idx": 2, "ts": 102.0, "method": "POST", "path": "/api/records/REC-204/approve", "status": 200},
            {"idx": 3, "ts": 103.0, "method": "GET", "path": "/api/records/REC-204", "status": 200},
        ],
    }
    ledger_records = [
        {"tool_name": "interact_with_browser", "arguments": {"action": "open_tab", "url": "http://127.0.0.1:43210/"}, "state": "success"},
        {"tool_name": "interact_with_browser", "arguments": {"action": "get_page_snapshot"}, "state": "success"},
        {"tool_name": "interact_with_browser", "arguments": {"action": "click_element", "element_id": 2}, "state": "success"},
        {"tool_name": "execute_command", "arguments": {"command": "curl -sS http://127.0.0.1:43210/api/records/REC-204"}, "state": "success"},
        {"tool_name": "manage_workspace_file", "arguments": {"action": "write", "path": str(ws_info["report_json_path"])}, "state": "success"},
        {"tool_name": "manage_workspace_file", "arguments": {"action": "write", "path": str(ws_info["report_md_path"])}, "state": "success"},
        {"tool_name": "capture_screen", "arguments": {"target": "window", "window": DEFAULT_DESKTOP_TITLE}, "state": "success"},
        {"tool_name": "desktop_control", "arguments": {"action": "keyboard_type", "expected_window": DEFAULT_DESKTOP_TITLE, "text": "REC-204:BL-204-OK:APPROVED", "press_enter": True}, "state": "success"},
        {"tool_name": "capture_screen", "arguments": {"target": "window", "window": DEFAULT_DESKTOP_TITLE}, "state": "success"},
    ]
    run_info = {
        "returncode": 0,
        "cli_json": {
            "prompt": "test",
            "model": "qwen3.6:35b-a3b",
            "response": "Done.",
            "execution_ledger": {
                "has_unresolved_failures": False,
                "unresolved_failures": [],
                "records": ledger_records,
            },
        },
        "ledger": {"status": "completed", "verification_status": "verified"},
    }
    desktop_before = {"applied_text": "", "status_text": "AWAITING_CONFIRMATION", "mouse_clicks": 0, "key_presses": 0}
    desktop_after = {
        "applied_text": "REC-204:BL-204-OK:APPROVED",
        "status_text": "CONFIRMED:REC-204:BL-204-OK:APPROVED",
        "mouse_clicks": 1,
        "key_presses": 5,
    }
    return ws_info, web_audit, run_info, desktop_before, desktop_after


@pytest.mark.asyncio
async def test_cross_surface_tool_selection_preserves_tools_across_window_transitions():
    """Verify cross-surface prompts expose both browser and desktop tools when desktop is active
    while still hiding interact_with_ui/desktop_control on browser-focused turns."""
    load_plugins()
    all_schemas = get_tool_schemas(0)
    prompt = (
        "Using Chromium open http://127.0.0.1:8123/, inspect REC-204, verify GET API, "
        "write report.json, then observe desktop window 'Zelyr Cross-Surface Confirmation', "
        "enter confirmation text, click Confirm, and capture a screenshot."
    )
    assert ActuatorArbiter.has_cross_surface_browser_and_desktop_intent(prompt) is True

    # 1. During active Chromium window turn, negative gating purges desktop_control/interact_with_ui
    browser_turn = filter_tool_schemas_by_tier(
        all_schemas,
        "tier2_browser",
        is_browser=True,
        include_tiers={2, 3},
    )
    browser_names = {s["function"]["name"] for s in browser_turn}
    assert "interact_with_browser" in browser_names
    assert "desktop_control" not in browser_names
    assert "interact_with_ui" not in browser_names

    # 2. Orchestrator recovers desktop tools from _base_payload_tools when active window switches to desktop
    orch = NativeOrchestrator()
    observed_tool_sets: list[set[str]] = []
    windows = [
        {"class": "chromium", "title": "Operations Dashboard"},
        {"class": "python3", "title": DEFAULT_DESKTOP_TITLE},
    ]

    async def fake_exec(name: str, **_kw):
        return json.dumps({"success": True, "status": "success", "output": {"ok": True}})

    orch.execute_tool = fake_exec
    turn = 0

    async def mock_stream(payload, _base_url, **_kw):
        nonlocal turn
        turn += 1
        names = {
            (t.get("function") or {}).get("name") or t.get("name")
            for t in (payload.get("tools") or [])
        }
        observed_tool_sets.append(names)
        if turn == 1:
            yield {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "b1", "function": {"name": "interact_with_browser", "arguments": {"action": "get_page_snapshot"}}},
            ]}, "finish_reason": "tool_calls"}]}
        else:
            yield {"choices": [{"delta": {"content": "Completed cross-surface workflow."}, "finish_reason": "stop"}]}

    async def fake_active_window():
        return windows[min(len(observed_tool_sets), len(windows) - 1)]

    with (
        patch.object(orch, "_stream_model_events", side_effect=mock_stream),
        patch.object(orch, "get_active_window", side_effect=fake_active_window),
    ):
        async for _ in orch.generate_stream({"messages": [{"role": "user", "content": prompt}], "tools": list(all_schemas)}):
            pass

    assert len(observed_tool_sets) >= 2
    assert "interact_with_browser" in observed_tool_sets[0]
    assert "desktop_control" not in observed_tool_sets[0]
    assert {"interact_with_browser", "desktop_control", "capture_screen", "manage_workspace_file", "execute_command"} <= observed_tool_sets[1]


@pytest.mark.asyncio
async def test_isolated_native_socket_takes_priority_over_global_ws_bridge(monkeypatch):
    """When ZELYR_BROWSER_NATIVE_SOCKET is set, NativeBrowserClient handles calls without touching port 41144."""
    monkeypatch.setenv("ZELYR_BROWSER_NATIVE_SOCKET", "/tmp/zelyr_isolated_test.sock")
    tool = InteractWithBrowserTool()
    mock_client = MagicMock()
    mock_client.is_socket_present.return_value = True
    mock_client.is_available = AsyncMock(return_value=True)
    mock_client.send_command = AsyncMock(return_value={"success": True, "url": "http://127.0.0.1:8123/"})

    with (
        patch("zelyr.tools.browser_native_host.NativeBrowserClient", return_value=mock_client),
        patch("zelyr.tools.browser_extension.get_bridge", side_effect=AssertionError("global WS bridge must not be called")),
    ):
        res = await tool.execute(action="get_active_tab")

    assert res.success is True
    mock_client.send_command.assert_awaited_once_with("get_active_tab")

    # When isolated socket is unavailable, fail closed without falling back to global WS bridge
    mock_client.is_socket_present.return_value = False
    with (
        patch("zelyr.tools.browser_native_host.NativeBrowserClient", return_value=mock_client),
        patch("zelyr.tools.browser_extension.get_bridge", side_effect=AssertionError("global WS bridge must not be called")),
    ):
        res_down = await tool.execute(action="get_active_tab")
    assert res_down.success is False
    assert "No browser extension connected" in (res_down.error or "")


def test_web_to_filesystem_consistency_and_read_only_api_classification(tmp_path):
    """Validator enforces live web-to-JSON/Markdown consistency and treats curl GET as inspection."""
    # Read-only curl GET is inspection; POST/output curl is mutation
    assert _is_inspection_action("execute_command", {"command": "curl -sS http://127.0.0.1:9000/api/records/REC-204"}) is True
    assert _is_inspection_action("execute_command", {"command": "curl -X POST http://127.0.0.1:9000/api/records/REC-204/approve"}) is False
    assert _is_inspection_action("execute_command", {"command": "curl -sS http://127.0.0.1:9000/api/records/REC-204 > out.json"}) is False
    assert _is_inspection_action("interact_with_browser", {"action": "get_page_snapshot"}) is True
    assert _is_inspection_action("interact_with_browser", {"action": "click_element", "element_id": 2}) is False

    ws_info, web_audit, run_info, desktop_before, desktop_after = _make_base_eval_inputs(tmp_path)
    ok_eval = evaluate_cross_surface_run(
        scenario="normal",
        run_info=run_info,
        web_audit=web_audit,
        ws_info=ws_info,
        desktop_before=desktop_before,
        desktop_after=desktop_after,
    )
    assert ok_eval["verdict"] == "PASS"

    # Mutating API command instead of read-only GET fails classification check
    bad_api_run = copy.deepcopy(run_info)
    bad_api_run["cli_json"]["execution_ledger"]["records"][3]["arguments"]["command"] = (
        "curl -sS -X POST http://127.0.0.1:43210/api/records/REC-204"
    )
    bad_api_eval = evaluate_cross_surface_run(
        scenario="normal",
        run_info=bad_api_run,
        web_audit=web_audit,
        ws_info=ws_info,
        desktop_before=desktop_before,
        desktop_after=desktop_after,
    )
    assert bad_api_eval["verdict"] == "FAIL"
    assert bad_api_eval["checks"]["api_check_classified_inspection"] is False

    # Stale/mismatched status in report.json fails filesystem consistency check
    bad_report = copy.deepcopy(ws_info["expected_result"]["expected_record"])
    bad_report["status"] = "PENDING_REVIEW"
    ws_info["report_json_path"].write_text(json.dumps(bad_report), encoding="utf-8")
    stale_eval = evaluate_cross_surface_run(
        scenario="normal",
        run_info=run_info,
        web_audit=web_audit,
        ws_info=ws_info,
        desktop_before=desktop_before,
        desktop_after=desktop_after,
    )
    assert stale_eval["verdict"] == "FAIL"
    assert stale_eval["checks"]["report_json_accurate"] is False


def test_desktop_confirmation_and_unrelated_record_protection(tmp_path):
    """Validator rejects unconfirmed desktop state and rejects any mutation to unrelated web/desktop targets."""
    ws_info, web_audit, run_info, desktop_before, desktop_after = _make_base_eval_inputs(tmp_path)

    # Unconfirmed desktop input
    unconfirmed = evaluate_cross_surface_run(
        scenario="normal",
        run_info=run_info,
        web_audit=web_audit,
        ws_info=ws_info,
        desktop_before=desktop_before,
        desktop_after={"applied_text": "", "status_text": "AWAITING_CONFIRMATION", "mouse_clicks": 0, "key_presses": 0},
    )
    assert unconfirmed["verdict"] == "FAIL"
    assert unconfirmed["checks"]["desktop_confirmed"] is False

    # Unrelated web record modified
    dirty_web = copy.deepcopy(web_audit)
    dirty_web["records"]["REC-101"]["status"] = "APPROVED"
    dirty_eval = evaluate_cross_surface_run(
        scenario="normal",
        run_info=run_info,
        web_audit=dirty_web,
        ws_info=ws_info,
        desktop_before=desktop_before,
        desktop_after=desktop_after,
    )
    assert dirty_eval["verdict"] == "FAIL"
    assert dirty_eval["checks"]["unrelated_web_untouched"] is False

    # Distraction window received input clicks
    dirty_dist = evaluate_cross_surface_run(
        scenario="normal",
        run_info=run_info,
        web_audit=web_audit,
        ws_info=ws_info,
        desktop_before=desktop_before,
        desktop_after=desktop_after,
        distraction_after={"mouse_clicks": 1, "key_presses": 0},
    )
    assert dirty_dist["verdict"] == "FAIL"
    assert dirty_dist["checks"]["distraction_untouched"] is False


def test_scenario_a_browser_failure_classification_ledger_and_validator(tmp_path):
    """Scenario A: ErrorClassifier & ledger handle stale browser element failure -> refresh -> retry cleanly."""
    clf_sel = ErrorClassifier.classify(
        "interact_with_browser",
        error="Element not found for selector: '#stale-approve-btn'",
    )
    assert clf_sel.error_type == "ELEMENT_NOT_FOUND"
    assert "interact_with_browser" in clf_sel.allowed_actions

    clf_to = ErrorClassifier.classify("interact_with_browser", error="Operation timed out")
    assert clf_to.error_type == "TIMEOUT"
    assert "interact_with_browser" in clf_to.allowed_actions

    # Ledger: click_element failure is NOT resolved by get_page_snapshot alone, ONLY by retried click_element
    ledger = TurnExecutionLedger()
    ledger.record_proposed("b_click1", "interact_with_browser", {"action": "click_element", "element_id": 4})
    ledger.record_observation("b_click1", "failed", {"success": False}, error="Element ID 4 not found on page.")
    assert ledger.has_unresolved_failures() is True

    ledger.record_proposed("b_snap", "interact_with_browser", {"action": "get_page_snapshot"})
    ledger.record_observation("b_snap", "success", {"elements": [{"id": 7}]})
    assert ledger.has_unresolved_failures() is True  # Snapshot refresh alone does not substitute for action retry

    ledger.record_proposed("b_click2", "interact_with_browser", {"action": "click_element", "element_id": 7})
    ledger.record_observation("b_click2", "success", {"clicked": True})
    assert ledger.has_unresolved_failures() is False

    # Validator rejects bypassed Scenario A (no DOM invalidation or failure recorded) as UNVERIFIED
    ws_info, web_audit, run_info, desktop_before, desktop_after = _make_base_eval_inputs(tmp_path)
    bypassed = evaluate_cross_surface_run(
        scenario="browser_failure",
        run_info=run_info,
        web_audit=web_audit,
        ws_info=ws_info,
        desktop_before=desktop_before,
        desktop_after=desktop_after,
    )
    assert bypassed["verdict"] == "UNVERIFIED"

    # Validator passes genuine Scenario A recovery evidence even if an earlier typo failed schema validation
    valid_audit = copy.deepcopy(web_audit)
    valid_audit["dom_invalidated_events"] = [{"ts": 101.5}]
    valid_run = copy.deepcopy(run_info)
    valid_run["ledger"] = {"status": "completed_unverified", "verification_status": "not_verified", "error": None}
    valid_run["cli_json"]["execution_ledger"]["records"] = [
        {"tool_name": "interact_with_browser", "arguments": {"action": "open_tab", "url": "http://127.0.0.1:43210/"}, "state": "success"},
        {"tool_name": "interact_with_browser", "arguments": {"action": "get_page_snapshot"}, "state": "success"},
        {"tool_name": "interact_with_browser", "arguments": {"action": "click_element", "id": "2"}, "state": "failed", "error": "[schema_validation_error] Unknown parameter 'id'"},
        {"tool_name": "interact_with_browser", "arguments": {"action": "click_element", "element_id": 2}, "state": "success"},
        {"tool_name": "interact_with_browser", "arguments": {"action": "get_page_snapshot"}, "state": "success"},
        {"tool_name": "interact_with_browser", "arguments": {"action": "click_element", "element_id": 4}, "state": "failed", "error": "Element ID 4 not found on page."},
        {"tool_name": "interact_with_browser", "arguments": {"action": "get_page_snapshot"}, "state": "success"},
        {"tool_name": "interact_with_browser", "arguments": {"action": "click_element", "element_id": 7}, "state": "success"},
        *run_info["cli_json"]["execution_ledger"]["records"][3:],
    ]
    passed_a = evaluate_cross_surface_run(
        scenario="browser_failure",
        run_info=valid_run,
        web_audit=valid_audit,
        ws_info=ws_info,
        desktop_before=desktop_before,
        desktop_after=desktop_after,
    )
    assert passed_a["verdict"] == "PASS"


def test_scenario_b_focus_loss_rejects_proactive_bypass_and_validates_real_recovery(tmp_path):
    """Scenario B rejects proactive focus bypass as UNVERIFIED and passes genuine FOCUS_LOST recovery."""
    ws_info, web_audit, run_info, desktop_before, desktop_after = _make_base_eval_inputs(tmp_path)

    # Bypassed focus loss (never failed with FOCUS_LOST) -> UNVERIFIED
    bypass_eval = evaluate_cross_surface_run(
        scenario="desktop_focus_loss",
        run_info=run_info,
        web_audit=web_audit,
        ws_info=ws_info,
        desktop_before=desktop_before,
        desktop_after=desktop_after,
        distraction_after={"mouse_clicks": 0, "key_presses": 0},
        steal_telemetry={},
    )
    assert bypass_eval["verdict"] == "UNVERIFIED"

    # Real FOCUS_LOST -> refocus -> retry -> PASS
    real_run = copy.deepcopy(run_info)
    recs = real_run["cli_json"]["execution_ledger"]["records"]
    real_recs = [
        *recs[:7],
        {
            "tool_name": "desktop_control",
            "arguments": {"action": "mouse_click", "x": 300, "y": 200, "expected_window": DEFAULT_DESKTOP_TITLE},
            "state": "failed",
            "error": "FOCUS_LOST: Target window is not focused",
        },
        {
            "tool_name": "manage_desktop_window",
            "arguments": {"action": "focus", "target": DEFAULT_DESKTOP_TITLE},
            "state": "success",
        },
        recs[7],
        recs[8],
    ]
    real_run["cli_json"]["execution_ledger"]["records"] = real_recs
    steal_info = {
        "stolen": True,
        "active_before": {"title": DEFAULT_DESKTOP_TITLE},
        "active_after": {"title": DEFAULT_DISTRACTION_TITLE},
        "fix_state_at_steal": {"applied_text": "", "mouse_clicks": 0, "key_presses": 0},
    }
    ok_b = evaluate_cross_surface_run(
        scenario="desktop_focus_loss",
        run_info=real_run,
        web_audit=web_audit,
        ws_info=ws_info,
        desktop_before=desktop_before,
        desktop_after=desktop_after,
        distraction_after={"mouse_clicks": 0, "key_presses": 0},
        steal_telemetry=steal_info,
    )
    assert ok_b["verdict"] == "PASS"

    # Passive cursor move before FOCUS_LOST on mouse_click still validates cleanly as PASS,
    # whereas pre-FOCUS_LOST mouse_click success is rejected as UNVERIFIED.
    passive_move_run = copy.deepcopy(real_run)
    passive_move_run["cli_json"]["execution_ledger"]["records"].insert(
        7,
        {"tool_name": "desktop_control", "arguments": {"action": "mouse_move", "x": 300, "y": 200}, "state": "success"},
    )
    ok_move = evaluate_cross_surface_run(
        scenario="desktop_focus_loss",
        run_info=passive_move_run,
        web_audit=web_audit,
        ws_info=ws_info,
        desktop_before=desktop_before,
        desktop_after=desktop_after,
        distraction_after={"mouse_clicks": 0, "key_presses": 0},
        steal_telemetry=steal_info,
    )
    assert ok_move["verdict"] == "PASS"

    premature_click_run = copy.deepcopy(real_run)
    premature_click_run["cli_json"]["execution_ledger"]["records"].insert(
        7,
        {"tool_name": "desktop_control", "arguments": {"action": "mouse_click", "x": 300, "y": 200, "expected_window": DEFAULT_DESKTOP_TITLE}, "state": "success"},
    )
    bad_premature = evaluate_cross_surface_run(
        scenario="desktop_focus_loss",
        run_info=premature_click_run,
        web_audit=web_audit,
        ws_info=ws_info,
        desktop_before=desktop_before,
        desktop_after=desktop_after,
        distraction_after={"mouse_clicks": 0, "key_presses": 0},
        steal_telemetry=steal_info,
    )
    assert bad_premature["verdict"] == "UNVERIFIED"
    assert bad_premature["checks"]["desktop_focus_lost_rejected"] is False


def test_cross_surface_web_app_dom_invalidation_and_clean_teardown(tmp_path):
    """CrossSurfaceWebApp serves deterministic DOM invalidation once on fail_browser_dom and cleans up on exit."""
    with CrossSurfaceWebApp(target_record="REC-204", fail_browser_dom=True) as app:
        parsed = urllib.parse.urlparse(app.base_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 80

        conn = http.client.HTTPConnection(host, port, timeout=2.0)
        try:
            conn.request("GET", "/records/REC-204")
            resp1 = conn.getresponse()
            html_1 = resp1.read().decode("utf-8")
            assert resp1.status == 200
            assert "stale-approve-btn" in html_1
            assert "MutationObserver" in html_1

            conn.request("POST", "/api/internal/dom-invalidated", body=b"{}", headers={"Content-Length": "2"})
            resp_inv = conn.getresponse()
            _ = resp_inv.read()
            assert resp_inv.status == 200

            conn.request("GET", "/records/REC-204")
            resp2 = conn.getresponse()
            html_2 = resp2.read().decode("utf-8")
            assert resp2.status == 200
            assert "stale-approve-btn" not in html_2
            assert 'id="approve-btn"' in html_2 or "id='approve-btn'" in html_2
        finally:
            conn.close()

    # Server cleanly closed after context exit
    with pytest.raises(Exception):
        conn_closed = http.client.HTTPConnection(host, port, timeout=0.5)
        try:
            conn_closed.request("GET", "/health")
            conn_closed.getresponse()
        finally:
            conn_closed.close()

    # Prompt remains non-prescriptive and never exposes secret oracle field values
    ws_info = setup_cross_surface_workspace(tmp_path / "ws2", "http://127.0.0.1:9999")
    prompt = build_cross_surface_prompt("http://127.0.0.1:9999", ws_info, desktop_title=DEFAULT_DESKTOP_TITLE)
    assert "do not use curl" in prompt.lower()
    assert "BL-204-OK" not in prompt
    assert "billing-ledger" not in prompt
    assert "expected_result.json" not in prompt


def test_honest_root_outcome_rejects_unresolved_or_cancelled_cross_surface_runs():
    """resolve_root_execution_outcome must never report COMPLETED when cancelled or unresolved failures remain."""
    ledger = TurnExecutionLedger()
    ledger.record_proposed("b1", "interact_with_browser", {"action": "open_tab", "url": "http://127.0.0.1:8000/"})
    ledger.record_observation("b1", "success", {"ok": True})
    ledger.record_proposed("d1", "desktop_control", {"action": "keyboard_type", "text": "REC-204:BL-204-OK:APPROVED"})
    ledger.record_observation("d1", "failed", {"success": False}, error="FOCUS_LOST")

    res_fail = resolve_root_execution_outcome(prompt="Approve REC-204 and confirm desktop", ledger=ledger)
    assert res_fail.execution_status == ExecutionStatus.FAILED
    assert res_fail.verification_status != VerificationStatus.VERIFIED
    assert res_fail.exit_code != 0

    res_cancel = resolve_root_execution_outcome(
        prompt="Approve REC-204 and confirm desktop",
        ledger=ledger,
        was_cancelled=True,
    )
    assert res_cancel.execution_status == ExecutionStatus.CANCELLED
    assert res_cancel.exit_code == 1


def test_url_paths_never_extracted_as_unfulfilled_file_targets():
    """URL route paths like http://127.0.0.1:8000/api/records/REC-204 must not trigger bogus replanning."""
    from zelyr.engine.execution_ledger import (
        extract_file_targets_from_text,
        find_unfulfilled_task_requirements,
    )

    prompt = (
        "Run a read-only HTTP GET command against http://127.0.0.1:45707/api/records/REC-204 "
        "to verify REC-204, then write /tmp/xsurf_demo/out/report.json and /tmp/xsurf_demo/out/report.md."
    )
    targets = extract_file_targets_from_text(prompt)
    assert "/api/records/REC-204" not in targets
    assert "/tmp/xsurf_demo/out/report.json" in targets
    assert "/tmp/xsurf_demo/out/report.md" in targets

    ledger = TurnExecutionLedger()
    ledger.record_proposed("w1", "manage_workspace_file", {"action": "write", "path": "/tmp/xsurf_demo/out/report.json", "content": "{}"})
    ledger.record_observation("w1", "success", {"ok": True})
    ledger.record_proposed("w2", "manage_workspace_file", {"action": "write", "path": "/tmp/xsurf_demo/out/report.md", "content": "# ok"})
    ledger.record_observation("w2", "success", {"ok": True})
    _, unfulfilled = find_unfulfilled_task_requirements(prompt, ledger)
    assert unfulfilled == []

    # Also verify mistaken browser switch_tab onto desktop window resolves cleanly when desktop surface runs
    ledger_sw = TurnExecutionLedger()
    ledger_sw.record_proposed("b_sw", "interact_with_browser", {"action": "switch_tab", "query": DEFAULT_DESKTOP_TITLE})
    ledger_sw.record_observation("b_sw", "failed", {"success": False}, error=f"No open tab matching query: '{DEFAULT_DESKTOP_TITLE.lower()}'")
    assert ledger_sw.has_unresolved_failures() is True
    ledger_sw.record_proposed("d_obs", "capture_screen", {"target": "window", "window": DEFAULT_DESKTOP_TITLE})
    ledger_sw.record_observation("d_obs", "success", {"success": True})
    assert ledger_sw.has_unresolved_failures() is False


def test_budget_manager_preserves_verified_data_facts_across_stage2_pruning():
    """Stage 2 turn summarization must preserve verified JSON/API facts instead of erasing them."""
    from zelyr.engine.budget_manager import ContextBudgetEngine

    engine = ContextBudgetEngine(default_max_tokens=8192)
    turns = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "execute_command", "arguments": {"command": "curl -s http://127.0.0.1:8000/api/records/REC-204"}}}]},
        {"role": "tool", "name": "execute_command", "tool_call_id": "c1", "content": "[Output Trace]:\n{\"record_id\": \"REC-204\", \"service\": \"billing-ledger\", \"confirmation_token\": \"BL-204-OK\", \"status\": \"APPROVED\"}"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c2", "function": {"name": "manage_workspace_file", "arguments": {"action": "write", "path": "/tmp/ws/out/report.json", "content": "{\"record_id\": \"REC-204\", \"confirmation_token\": \"BL-204-OK\"}"}}}]},
        {"role": "tool", "name": "manage_workspace_file", "tool_call_id": "c2", "content": "{\"success\": true}"},
    ]
    summary = engine.summarize_turns(turns, max_summary_tokens=250)
    assert "Completed earlier (do not repeat):" in summary
    assert "wrote:report.json" in summary
    assert "Key verified data:" in summary
    assert "BL-204-OK" in summary
    assert "billing-ledger" in summary

    norm_coord = NativeOrchestrator.normalize_tool_arguments(
        "desktop_control",
        {"action": "mouse_click", "coordinate": "[1060, 1410]", "expected_window": DEFAULT_DESKTOP_TITLE},
    )
    assert norm_coord["x"] == 1060 and norm_coord["y"] == 1410
    assert "coordinate" not in norm_coord


def test_http_read_failure_resolved_by_matching_curl_and_stage175_preserves_facts():
    """Verify SSRF-blocked fetch_web_page resolves via matching read-only curl GET,
    while Stage 1.75/2 preserve REC-204 verified JSON without trivial status overwrites."""
    from zelyr.engine.budget_manager import ContextBudgetEngine

    ledger = TurnExecutionLedger()
    ledger.record_proposed("f1", "fetch_web_page", {"url": "http://127.0.0.1:41033/api/records/REC-204"})
    ledger.record_observation("f1", "error", {"error": "SSRF blocked"}, error="SSRF blocked: loopback disallowed")
    assert ledger.has_unresolved_failures() is True

    # Different URL must NOT resolve f1
    ledger.record_proposed("c_other", "execute_command", {"command": "curl -s http://127.0.0.1:41033/api/records/REC-101"})
    ledger.record_observation("c_other", "success", {"output": "{}"})
    assert ledger.has_unresolved_failures() is True

    # Exact matching read-only curl GET resolves f1
    ledger.record_proposed("c_match", "execute_command", {"command": "curl -s http://127.0.0.1:41033/api/records/REC-204"})
    ledger.record_observation("c_match", "success", {"output": '{"record_id": "REC-204", "status": "APPROVED"}'})
    assert ledger.has_unresolved_failures() is False

    # Stage 2 summary must skip trivial {"success": true} from mkdir/desktop and retain REC-204 JSON + URLs
    engine = ContextBudgetEngine(default_max_tokens=8192)
    turns = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "b0", "function": {"name": "interact_with_browser", "arguments": {"action": "open_tab", "url": "http://127.0.0.1:41033/"}}}]},
        {"role": "tool", "name": "interact_with_browser", "tool_call_id": "b0", "content": "[TOOL OBSERVATION: interact_with_browser]\n- Status: SUCCESS\n- Summary: ok\n- Evidence Level: REPORTED\n- Ledger Ref: x\n\n[Output]:\n{\"success\": true}"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "b1", "function": {"name": "interact_with_browser", "arguments": {"action": "get_page_snapshot"}}}]},
        {"role": "tool", "name": "interact_with_browser", "tool_call_id": "b1", "content": "[Output]:\n{\"success\": true, \"title\": \"REC-204\", \"snapshot\": \"REC-204 PENDING_REVIEW " + ("row " * 40) + "\"}"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "execute_command", "arguments": {"command": "curl -s http://127.0.0.1:41033/api/records/REC-204"}}}]},
        {"role": "tool", "name": "execute_command", "tool_call_id": "c1", "content": "[Output Trace]:\n{\"record_id\": \"REC-204\", \"confirmation_token\": \"BL-204-OK\", \"status\": \"APPROVED\"}"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c2", "function": {"name": "execute_command", "arguments": {"command": "mkdir -p /tmp/ws/out"}}}]},
        {"role": "tool", "name": "execute_command", "tool_call_id": "c2", "content": "[Output Trace]:\n{\"success\": true}"},
    ]
    compacted_obs, n_obs = engine.stage1_75_compact_observation_headers(turns, recent_turns_to_preserve=1)
    assert n_obs >= 1
    assert "- Evidence Level:" not in compacted_obs[1]["content"]
    assert "[TOOL OBSERVATION: interact_with_browser]" in compacted_obs[1]["content"]

    summary = engine.summarize_turns(turns, max_summary_tokens=240)
    assert "url=http://127.0.0.1:41033/" in summary
    assert "BL-204-OK" in summary
    assert "APPROVED" in summary
    assert "PENDING_REVIEW" not in summary
    assert "{\"success\": true}" not in summary
    from zelyr.engine.budget_manager import estimate_tokens as _est_tok
    assert _est_tok(summary) <= 240

    # Read-only wget -qO- counts as HTTP inspection and historical capture_screen compacts bulky arrays
    from zelyr.engine.budget_manager import _truncate_to_tokens, estimate_tokens

    assert _is_inspection_action("execute_command", {"command": "wget -qO- http://127.0.0.1:41033/api/records/REC-204"}) is True
    assert _is_inspection_action("execute_command", {"command": "wget http://127.0.0.1:41033/api/records/REC-204 -O out.json"}) is False
    url_heavy = "Completed earlier: " + ", ".join(
        f"interact_with_browser(open_tab url=http://127.0.0.1:45139/very/long/path/segment/{i})" for i in range(15)
    )
    assert estimate_tokens(_truncate_to_tokens(url_heavy, 250)) <= 250

    cap_turns = [
        {
            "role": "tool",
            "name": "capture_screen",
            "tool_call_id": "cap1",
            "content": (
                "[TOOL OBSERVATION: capture_screen]\n- Status: SUCCESS\n- Evidence Level: REPORTED\n\n[Output]:\n"
                + json.dumps({
                    "success": True,
                    "target": "window",
                    "captured_region": {"x": 640, "y": 1500, "width": 640, "height": 420},
                    "active_window": {"title": DEFAULT_DESKTOP_TITLE},
                    "monitors": [{"name": "DP-1", "x": 0, "y": 1080}] * 3,
                    "visible_windows": [{"title": f"win-{i}"} for i in range(10)],
                    "sha256": "a" * 64,
                })
            ),
        },
        {"role": "tool", "name": "desktop_control", "tool_call_id": "d2", "content": "[TOOL OBSERVATION: desktop_control]\n- Status: SUCCESS"},
    ]
    comp_cap, n_cap = engine.stage1_75_compact_observation_headers(cap_turns, recent_turns_to_preserve=1)
    assert n_cap == 1
    assert "monitors" not in comp_cap[0]["content"]
    assert "captured_region" in comp_cap[0]["content"]


@pytest.mark.asyncio
async def test_browser_dom_refresh_resets_element_cycle_ceiling_and_fail_closed_reply(tmp_path):
    """After click_element fails & immediate retry hits cycle_detected, get_page_snapshot
    resets the retry window so click_element(element_id=1) executes after DOM refresh."""
    orch = NativeOrchestrator()
    executed_calls: list[tuple[str, dict]] = []

    async def fake_exec(name: str, **kw):
        executed_calls.append((name, dict(kw)))
        act = kw.get("action")
        if act == "click_element" and len([c for c in executed_calls if c[1].get("action") == "click_element"]) == 1:
            return json.dumps({"success": False, "status": "error", "error": "Element ID 1 not found on page"})
        return json.dumps({"success": True, "status": "success", "output": {"ok": True}})

    orch.execute_tool = fake_exec
    step = 0

    async def mock_stream(_payload, _base_url, **_kw):
        nonlocal step
        step += 1
        seq = [
            {"name": "interact_with_browser", "arguments": {"action": "click_element", "element_id": 1}},
            {"name": "interact_with_browser", "arguments": {"action": "click_element", "element_id": 1}},
            {"name": "interact_with_browser", "arguments": {"action": "get_page_snapshot"}},
            {"name": "interact_with_browser", "arguments": {"action": "click_element", "element_id": 1}},
        ]
        if step <= len(seq):
            yield {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": f"tc_{step}", "function": seq[step - 1]},
            ]}, "finish_reason": "tool_calls"}]}
        else:
            yield {"choices": [{"delta": {"content": "Recovered click cleanly."}, "finish_reason": "stop"}]}

    load_plugins()
    all_s = get_tool_schemas(0)
    with (
        patch.object(orch, "_stream_model_events", side_effect=mock_stream),
        patch.object(orch, "get_active_window", AsyncMock(return_value={"class": "chromium", "title": "Record REC-204"})),
    ):
        async for _ in orch.generate_stream({"messages": [{"role": "user", "content": "Approve REC-204 in Chromium"}], "tools": list(all_s)}):
            pass

    click_execs = [c for c in executed_calls if c[1].get("action") == "click_element"]
    # Step 1 failed -> Step 2 blocked pre-exec -> Step 3 snapshot -> Step 4 retried & succeeded!
    assert len(click_execs) == 2
    assert any(c[1].get("action") == "get_page_snapshot" for c in executed_calls)

    # Zero web approvals must report FAIL(target_web_approved), not spurious mutation-before-invalidation
    ws_info, web_audit, run_info, dbefore, dafter = _make_base_eval_inputs(tmp_path, approved=False)
    web_audit["fail_browser_dom"] = True
    web_audit["dom_invalidated_events"] = [{"ts": 101.5}]
    web_audit["requests"] = [r for r in web_audit["requests"] if r["method"] != "POST"]
    recs = run_info["cli_json"]["execution_ledger"]["records"]
    recs.insert(2, {"tool_name": "interact_with_browser", "arguments": {"action": "click_element", "element_id": 1}, "state": "failed", "error": "Element ID 1 not found on page"})
    res = evaluate_cross_surface_run(
        scenario="browser_failure",
        run_info=run_info,
        web_audit=web_audit,
        ws_info=ws_info,
        desktop_before=dbefore,
        desktop_after=dafter,
    )
    assert res["verdict"] == "FAIL"
    assert "web mutation occurred before DOM invalidation" not in res["unverified_reasons"]


@pytest.mark.asyncio
async def test_desktop_refocus_resets_identical_call_ceiling_after_focus_lost():
    """After FOCUS_LOST + unrefocused cycle block, manage_desktop_window(focus)
    resets the identical-call ceiling so retrying desktop_control at the same coordinates succeeds."""
    orch = NativeOrchestrator()
    executed_calls: list[tuple[str, dict]] = []

    async def fake_exec(name: str, **kw):
        executed_calls.append((name, dict(kw)))
        if name == "desktop_control" and len([c for c in executed_calls if c[0] == "desktop_control"]) == 1:
            return json.dumps({
                "success": False,
                "status": "error",
                "error": "FOCUS_LOST: Target window 'Zelyr Cross-Surface Confirmation' is not focused",
            })
        return json.dumps({"success": True, "status": "success", "output": {"ok": True}})

    orch.execute_tool = fake_exec
    step = 0

    async def mock_stream(_payload, _base_url, **_kw):
        nonlocal step
        step += 1
        seq = [
            {"name": "desktop_control", "arguments": {"action": "mouse_click", "x": 960, "y": 1675, "expected_window": DEFAULT_DESKTOP_TITLE}},
            {"name": "desktop_control", "arguments": {"action": "mouse_click", "x": 960, "y": 1675, "expected_window": DEFAULT_DESKTOP_TITLE}},
            {"name": "manage_desktop_window", "arguments": {"action": "focus", "target": DEFAULT_DESKTOP_TITLE, "verify": True}},
            {"name": "desktop_control", "arguments": {"action": "mouse_click", "x": 960, "y": 1675, "expected_window": DEFAULT_DESKTOP_TITLE}},
        ]
        if step <= len(seq):
            yield {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": f"dc_{step}", "function": seq[step - 1]},
            ]}, "finish_reason": "tool_calls"}]}
        else:
            yield {"choices": [{"delta": {"content": "Desktop click recovered cleanly."}, "finish_reason": "stop"}]}

    load_plugins()
    all_s = get_tool_schemas(0)
    with (
        patch.object(orch, "_stream_model_events", side_effect=mock_stream),
        patch.object(orch, "get_active_window", AsyncMock(return_value={"class": "python3", "title": DEFAULT_DESKTOP_TITLE})),
    ):
        async for _ in orch.generate_stream({"messages": [{"role": "user", "content": f"Confirm desktop window '{DEFAULT_DESKTOP_TITLE}'"}], "tools": list(all_s)}):
            pass

    dc_execs = [c for c in executed_calls if c[0] == "desktop_control"]
    # Step 1 FOCUS_LOST -> Step 2 blocked pre-exec -> Step 3 focus -> Step 4 retried & succeeded!
    assert len(dc_execs) == 2
    assert any(c[0] == "manage_desktop_window" and c[1].get("action") == "focus" for c in executed_calls)
