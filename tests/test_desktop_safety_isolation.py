"""Regression tests ensuring normal Zelyr tasks and benchmarks never steal user focus or mouse input."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess
from unittest.mock import AsyncMock, MagicMock

import pytest

from scripts import cross_surface_benchmark as cross_bench
from scripts import linux_desktop_benchmark as desk_bench
from zelyr.agents.actuator_arbiter import ActuatorArbiter
from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.cli.main import _execute_headless_run
from zelyr.core.plugins import get_tool_schemas, load_plugins
from zelyr.engine.execution_ledger import CentralExecutionLedger, TurnExecutionLedger
from zelyr.engine.hyprland_ipc import HyprlandIPCClient
from zelyr.tools.core import ToolResult
from zelyr.tools.desktop_control import DesktopAutomationTool
from zelyr.tools.input_controller import WaylandInputController


# ── 1. Ordinary CLI Tasks Never Expose Desktop Mouse/Fixture Execution ────────


def test_arbiter_keeps_code_and_web_button_prompts_out_of_desktop_mouse_tier() -> None:
    assert (
        ActuatorArbiter.arbitrate(
            "Inspect the submit button label in src/Checkout.tsx and fix the bug",
            active_window={"class": "kitty", "title": "nvim"},
            bridge_connected=False,
        )
        == "tier1_ipc"
    )
    assert ActuatorArbiter.has_explicit_desktop_gui_intent(
        "Inspect the confirmation dialog in modal.html and run pytest"
    ) is False
    assert ActuatorArbiter.has_explicit_window_control_intent(
        "Read ~/notes/sport.md and summarize my checklist"
    ) is False

    # Explicit desktop prompts stay supported
    assert (
        ActuatorArbiter.arbitrate(
            "Observe the Zelyr Desktop Test Fixture window and click Confirm",
            active_window=None,
            bridge_connected=False,
        )
        == "tier3_vision"
    )
    assert ActuatorArbiter.has_explicit_desktop_gui_intent(
        "Click Confirm in the Zelyr Desktop Test Fixture window"
    ) is True
    assert ActuatorArbiter.has_explicit_window_control_intent("Switch to workspace 4") is True
    assert ActuatorArbiter.has_explicit_desktop_gui_intent("Switch to workspace 4") is False

    # Coding/workspace/DOM/browser-only phrases containing "workspace" or "window" stay safe
    for safe_prompt, expected_tier in (
        ("Update README.md in my workspace", "tier1_ipc"),
        ("Debug window.location in src/App.tsx", "tier1_ipc"),
        ("Open https://example.com in Firefox and click Place Your Order", "tier2_browser"),
    ):
        assert ActuatorArbiter.arbitrate(safe_prompt, active_window={"class": "kitty", "title": "term"}, bridge_connected=False) == expected_tier
        assert ActuatorArbiter.has_explicit_desktop_gui_intent(safe_prompt) is False
        assert ActuatorArbiter.has_explicit_window_control_intent(safe_prompt) is False


@pytest.mark.asyncio
async def test_ordinary_cli_tasks_never_launch_fixture_or_dispatch_desktop_control(tmp_path: Path, monkeypatch) -> None:
    load_plugins()
    real_popen = subprocess.Popen
    popen_cmds: list[object] = []

    def _guarded_popen(*a, **kw):
        cmd_repr = " ".join(str(x) for x in (a[0] if a and isinstance(a[0], (list, tuple)) else [a[0] if a else kw.get("args", "")]))
        popen_cmds.append(cmd_repr)
        if "desktop_test_fixture" in cmd_repr or "chromium" in cmd_repr or "wtype" in cmd_repr or "ydotool" in cmd_repr:
            raise AssertionError(f"Unexpected GUI/fixture Popen during ordinary CLI task: {a}")
        return real_popen(*a, **kw)

    monkeypatch.setattr(subprocess, "Popen", _guarded_popen)

    prompts = [
        "Inspect the submit button in src/App.tsx",
        "List open processes and summarize CPU usage",
        "Fetch https://127.0.0.1:8080/health and check status",
    ]
    for idx, prompt in enumerate(prompts):
        seen_tools: list[list[str]] = []
        dispatched: list[tuple[str, dict]] = []
        turn = 0

        async def mock_stream(payload):
            nonlocal turn
            turn += 1
            seen_tools.append([
                (t.get("function") or {}).get("name") or t.get("name")
                for t in (payload.get("tools") or [])
                if isinstance(t, dict)
            ])
            if turn == 1:
                # Simulate confused model trying to hijack mouse during an ordinary coding task
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "dc_bad", "type": "function", "function": {"name": "desktop_control", "arguments": json.dumps({"action": "mouse_click", "x": 500, "y": 500})}},
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }]
                }
            elif turn == 2:
                # And then trying to steal focus
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "fw_bad", "type": "function", "function": {"name": "manage_desktop_window", "arguments": json.dumps({"action": "focus", "target": "Checkout"})}},
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Completed safely without desktop input."}, "finish_reason": "stop"}]}

        async def mock_exec(name: str, args: dict):
            dispatched.append((name, dict(args)))
            return ToolResult(True, output={"status": "success"})

        central = CentralExecutionLedger(db_path=tmp_path / f"ord_{idx}.db", raise_on_persistence_error=True)
        orch = NativeOrchestrator(central_ledger=central, stream_provider=mock_stream, tool_executor=mock_exec, max_steps=4)
        try:
            rc = await _execute_headless_run(prompt, orchestrator=orch, output_json=True)
        finally:
            central.close()

        assert rc == 2
        assert dispatched == [], f"Ordinary task {prompt!r} must never dispatch mouse/focus calls: {dispatched}"
        assert all("desktop_control" not in ts and "interact_with_ui" not in ts for ts in seen_tools)
        ledger = orch.current_ledger
        assert ledger is not None
        assert any(r.rejection_reason in ("authorization_denied", "unrequested_desktop_control", "unrequested_window_focus") for r in ledger._records.values())


@pytest.mark.asyncio
async def test_orchestrator_blocks_hallucinated_mouse_focus_even_with_full_schemas() -> None:
    """Even if caller supplies full schemas, NativeOrchestrator blocks unrequested desktop input/focus."""
    executed: list[str] = []
    turn = 0

    async def mock_stream(_payload):
        nonlocal turn
        turn += 1
        if turn == 1:
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "m1", "type": "function", "function": {"name": "desktop_control", "arguments": json.dumps({"action": "mouse_move", "x": 100, "y": 200})}},
                        ]
                    },
                    "finish_reason": "tool_calls",
                }]
            }
        elif turn == 2:
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "m2", "type": "function", "function": {"name": "manage_desktop_window", "arguments": json.dumps({"action": "focus", "target": "Zen Browser"})}},
                        ]
                    },
                    "finish_reason": "tool_calls",
                }]
            }
        elif turn == 3:
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "m3", "type": "function", "function": {"name": "manage_desktop_window", "arguments": json.dumps({"action": "list"})}},
                        ]
                    },
                    "finish_reason": "tool_calls",
                }]
            }
        else:
            yield {"choices": [{"delta": {"content": "Done read-only check."}, "finish_reason": "stop"}]}

    async def mock_exec(name: str, _args: dict):
        executed.append(name)
        return ToolResult(True, output={"status": "success"})

    load_plugins()
    schemas = [
        s for s in get_tool_schemas(0)
        if (s.get("function") or {}).get("name") in ("desktop_control", "manage_desktop_window", "execute_command")
    ]
    orch = NativeOrchestrator(stream_provider=mock_stream, tool_executor=mock_exec, enable_cycle_detection=False)
    ledger = TurnExecutionLedger()
    async for _ in orch.generate_stream(
        {"messages": [{"role": "user", "content": "Check my python unit tests."}], "tools": schemas},
        ledger=ledger,
    ):
        pass

    # Only read-only manage_desktop_window(action='list') ran; mouse_move and focus were blocked!
    assert executed == ["manage_desktop_window"]
    assert ledger._records["m1"].rejection_reason == "unrequested_desktop_control"
    assert ledger._records["m2"].rejection_reason == "unrequested_window_focus"


# ── 2. Explicit Desktop Tasks Preserve Interaction ────────────────────────────


@pytest.mark.asyncio
async def test_explicit_desktop_task_preserves_focus_and_click(monkeypatch) -> None:
    monkeypatch.setattr(
        NativeOrchestrator,
        "get_active_window_sync",
        classmethod(lambda cls: {"class": "python3", "title": "Fixture"}),
    )
    executed: list[tuple[str, str | None]] = []
    turn = 0

    async def mock_stream(_payload):
        nonlocal turn
        turn += 1
        if turn == 1:
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "e1", "type": "function", "function": {"name": "manage_desktop_window", "arguments": json.dumps({"action": "focus", "target": "Fixture"})}},
                        ]
                    },
                    "finish_reason": "tool_calls",
                }]
            }
        elif turn == 2:
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "e2", "type": "function", "function": {"name": "desktop_control", "arguments": json.dumps({"action": "mouse_click", "x": 300, "y": 400, "expected_window": "Fixture"})}},
                        ]
                    },
                    "finish_reason": "tool_calls",
                }]
            }
        else:
            yield {"choices": [{"delta": {"content": "Clicked button in Fixture window."}, "finish_reason": "stop"}]}

    async def mock_exec(name: str, args: dict):
        executed.append((name, args.get("action")))
        return ToolResult(True, output={"status": "success", "verified": True if name == "manage_desktop_window" else False})

    load_plugins()
    schemas = [
        s for s in get_tool_schemas(0)
        if (s.get("function") or {}).get("name") in ("desktop_control", "manage_desktop_window", "capture_screen")
    ]
    orch = NativeOrchestrator(stream_provider=mock_stream, tool_executor=mock_exec)
    async for _ in orch.generate_stream(
        {"messages": [{"role": "user", "content": "Focus the Fixture window and click Confirm."}], "tools": schemas},
    ):
        pass

    assert executed == [("manage_desktop_window", "focus"), ("desktop_control", "mouse_click")]


# ── 3 & 4. Desktop Operations Never Focus or Click Unrelated User Windows ─────


@pytest.mark.asyncio
async def test_desktop_control_refuses_blind_input_without_expected_window_on_live_hyprland(monkeypatch) -> None:
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    monkeypatch.setenv("HYPRLAND_INSTANCE_SIGNATURE", "sig")

    active_user_win = {
        "address": "0x5582dc003160",
        "title": "Place Your Order – Amazon Checkout — Zen Browser",
        "class": "app.zen_browser.zen",
    }
    mock_ipc = AsyncMock(spec=HyprlandIPCClient)
    mock_ipc.can_connect.return_value = True
    mock_ipc.get_active_window.return_value = active_user_win
    mock_ipc.get_monitors.return_value = [
        {"id": 2, "name": "HDMI-A-1", "x": 0, "y": 0, "width": 1920, "height": 1080, "logical_width": 1920, "logical_height": 1080, "scale": 1.0, "focused": True}
    ]

    ctrl = WaylandInputController(auto_check=False)
    ctrl.mouse_click = MagicMock()
    ctrl.mouse_move = MagicMock()
    ctrl.keyboard_type = MagicMock()
    tool = DesktopAutomationTool(wayland_controller=ctrl, ipc_client=mock_ipc)

    # Missing expected_window -> MUST refuse instead of clicking active user browser!
    res_no_target = await tool.execute(action="mouse_click", x=400, y=400)
    assert res_no_target.success is False
    assert res_no_target.error_type == "FOCUS_TARGET_REQUIRED"
    ctrl.mouse_click.assert_not_called()

    res_move_no_target = await tool.execute(action="mouse_move", x=400, y=400)
    assert res_move_no_target.success is False
    assert res_move_no_target.error_type == "FOCUS_TARGET_REQUIRED"
    ctrl.mouse_move.assert_not_called()

    # Mismatched expected_window against active user browser -> FOCUS_LOST, zero input
    res_mismatch = await tool.execute(action="keyboard_type", text="hello", expected_window="Zelyr Fixture")
    assert res_mismatch.success is False
    assert res_mismatch.error_type == "FOCUS_LOST"
    ctrl.keyboard_type.assert_not_called()

    ctrl.close = MagicMock()
    tool.close()
    ctrl.close.assert_called_once()


def test_live_desktop_benchmarks_refuse_unprompted_takeover_and_active_user_monitor(monkeypatch) -> None:
    monkeypatch.delenv("ZELYR_ALLOW_LIVE_DESKTOP", raising=False)
    monkeypatch.delenv("ZELYR_FORCE_LIVE_FOCUS", raising=False)

    monitors = [
        {"id": 1, "name": "DP-1", "x": 0, "y": 1080, "width": 1920, "height": 1080, "scale": 1.0, "focused": False, "activeWorkspace": {"id": 11}},
        {"id": 2, "name": "HDMI-A-1", "x": 0, "y": 0, "width": 1920, "height": 1080, "scale": 1.0, "focused": True, "activeWorkspace": {"id": 1}},
    ]
    active_user_win = {
        "address": "0x5582dc003160",
        "title": "Place Your Order – Amazon Checkout — Zen Browser",
        "class": "app.zen_browser.zen",
    }

    dispatched: list[tuple[str, str]] = []
    monkeypatch.setattr(cross_bench, "_hyprctl_json", lambda arg: monitors if arg == "monitors" else (active_user_win if arg == "activewindow" else []))
    monkeypatch.setattr(cross_bench, "_hyprctl_dispatch", lambda cmd, arg="": dispatched.append((cmd, arg)))
    monkeypatch.setattr(desk_bench, "_hyprctl_json", lambda arg: monitors if arg == "monitors" else (active_user_win if arg == "activewindow" else []))
    monkeypatch.setattr(desk_bench, "_hyprctl_dispatch", lambda cmd, arg="": dispatched.append((cmd, arg)))
    monkeypatch.setattr(
        cross_bench.subprocess,
        "Popen",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Fixture must not spawn without opt-in")),
    )

    # 1. Default run without --allow-live-desktop aborts BEFORE touching Hyprland or spawning fixtures
    with pytest.raises(RuntimeError, match="ZELYR_ALLOW_LIVE_DESKTOP"):
        cross_bench.run_cross_surface_benchmark(scenarios=["normal"], workspace_id=15)
    with pytest.raises(RuntimeError, match="ZELYR_ALLOW_LIVE_DESKTOP"):
        desk_bench.run_benchmark(workspace_id=15)
    with pytest.raises(RuntimeError, match="ZELYR_ALLOW_LIVE_DESKTOP"):
        desk_bench.run_focus_recovery_benchmark(workspace_id=15)
    assert dispatched == []

    # 2. Even with allow_live_desktop=True, active user focus on HDMI-A-1 blocks stealing focus onto DP-1
    with pytest.raises(RuntimeError, match="HDMI-A-1"):
        cross_bench.run_cross_surface_benchmark(
            scenarios=["normal"],
            workspace_id=15,
            allow_live_desktop=True,
        )
    assert dispatched == []


def test_fixture_no_activate_mode_and_process_group_cleanup(monkeypatch) -> None:
    from scripts.desktop_test_fixture import FixtureWindow, _make_handler
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    import http.client
    import http.server
    import threading

    app = QApplication.instance() or QApplication([])
    win = FixtureWindow(
        title="Passive Fixture",
        cross_surface=True,
        initial_status="AWAITING_CONFIRMATION",
        no_activate=True,
    )
    assert win.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating) is True
    assert bool(win.windowFlags() & Qt.WindowType.WindowDoesNotAcceptFocus) is True

    # activateWindow slot is a no-op and /activate HTTP endpoint returns 403 in no_activate mode
    win._activate_ui_slot()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(win, app))
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
        try:
            conn.request("POST", "/activate", body=b"{}")
            resp = conn.getresponse()
            assert resp.status == 403
        finally:
            conn.close()
    finally:
        server.shutdown()
        server.server_close()
        win.close()

    # Verify process-group kill cleanly terminates timed-out child process groups as well as normal procs
    killed_groups: list[tuple[int, int]] = []
    proc = MagicMock(spec=desk_bench._REAL_POPEN)
    proc.pid = 424242
    proc.poll.return_value = None
    proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="fixture", timeout=3.0), 0]
    monkeypatch.setattr(desk_bench.os, "getpgid", lambda pid: 424242 if pid == 424242 else 1)
    monkeypatch.setattr(desk_bench.os, "getpgrp", lambda: 111111)
    monkeypatch.setattr(desk_bench.os, "killpg", lambda pgid, sig: killed_groups.append((pgid, sig)))

    desk_bench._cleanup_fixture_processes((proc, None))
    assert any(pg == 424242 and sig == desk_bench.signal.SIGTERM for pg, sig in killed_groups)
    assert any(pg == 424242 and sig == desk_bench.signal.SIGKILL for pg, sig in killed_groups)
    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()


@pytest.mark.asyncio
async def test_cancelled_error_cleans_up_benchmark_fixtures(tmp_path: Path, monkeypatch) -> None:
    class _FakeSess:
        def __init__(self, root: Path):
            self.socket_path = root / "sock"

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    class _FakeWeb:
        def __init__(self, **_kw):
            self.base_url = "http://127.0.0.1:45558"

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    monkeypatch.setattr(cross_bench, "IsolatedChromiumSession", _FakeSess)
    monkeypatch.setattr(cross_bench, "CrossSurfaceWebApp", _FakeWeb)
    monkeypatch.setattr(cross_bench, "_wait_for_http", lambda *_a, **_kw: {"ok": True})
    monkeypatch.setattr(cross_bench, "_wait_for_window", lambda title, pid=0: {"address": f"0x{pid:x}", "title": title})
    monkeypatch.setattr(cross_bench, "_place_floating_window", lambda *_a, **_kw: None)
    monkeypatch.setattr(cross_bench, "_hyprctl_dispatch", lambda *_a, **_kw: None)
    monkeypatch.setattr(cross_bench, "_focus_and_verify_window", lambda *_a, **_kw: {"active_window": {"title": "x"}})
    monkeypatch.setattr(cross_bench, "_http_json", lambda *_a, **_kw: {"applied_text": ""})
    monkeypatch.setattr(
        cross_bench,
        "_run_cross_surface_cli",
        MagicMock(side_effect=asyncio.CancelledError()),
    )

    killed_pgids: list[tuple[int, int]] = []
    monkeypatch.setattr(desk_bench.os, "killpg", lambda pgid, sig: killed_pgids.append((pgid, sig)))
    monkeypatch.setattr(cross_bench.os, "killpg", lambda pgid, sig: killed_pgids.append((pgid, sig)))
    monkeypatch.setattr(desk_bench.os, "getpgid", lambda pid: (_ for _ in ()).throw(AssertionError(f"Unexpected getpgid({pid}) on mock")))
    monkeypatch.setattr(cross_bench.os, "getpgid", lambda pid: (_ for _ in ()).throw(AssertionError(f"Unexpected getpgid({pid}) on mock")))

    spawned: list[MagicMock] = []
    monkeypatch.setattr(cross_bench.subprocess, "Popen", lambda *_a, **_kw: spawned.append(MagicMock(pid=7001)) or spawned[-1])

    with pytest.raises(asyncio.CancelledError):
        cross_bench.run_single_cross_surface_scenario(
            "normal",
            model="qwen3.6:35b-a3b",
            max_steps=4,
            workspace_id=15,
            mon_x=0,
            mon_y=0,
            mon_w=1920,
            mon_h=1080,
            upstream_ollama="http://127.0.0.1:11434",
        )
    assert len(spawned) == 1
    assert killed_pgids == []
    spawned[0].terminate.assert_called_once()


# ── 5. Audit Hardening: History Isolation, Fallbacks, Mutators & PGID Safety ──


@pytest.mark.asyncio
async def test_repl_history_bleed_interact_with_ui_and_fullscreen_close_blocked() -> None:
    """Prior desktop turn must not unlock desktop/window mutations in a subsequent ordinary coding turn."""
    executed: list[str] = []
    turn = 0

    async def mock_stream(_payload):
        nonlocal turn
        turn += 1
        if turn == 1:
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "d1", "type": "function", "function": {"name": "interact_with_ui", "arguments": json.dumps({"instruction": "click Checkout"})}},
                        ]
                    },
                    "finish_reason": "tool_calls",
                }]
            }
        elif turn == 2:
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "w1", "type": "function", "function": {"name": "manage_desktop_window", "arguments": json.dumps({"action": "fullscreen"})}},
                        ]
                    },
                    "finish_reason": "tool_calls",
                }]
            }
        elif turn == 3:
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "w2", "type": "function", "function": {"name": "manage_desktop_window", "arguments": json.dumps({"action": "close", "target": "Zen Browser"})}},
                        ]
                    },
                    "finish_reason": "tool_calls",
                }]
            }
        elif turn == 4:
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "w3", "type": "function", "function": {"name": "manage_desktop_window", "arguments": json.dumps({"action": "list"})}},
                        ]
                    },
                    "finish_reason": "tool_calls",
                }]
            }
        else:
            yield {"choices": [{"delta": {"content": "Handled read-only safely."}, "finish_reason": "stop"}]}

    async def mock_exec(name: str, _args: dict):
        executed.append(name)
        return ToolResult(True, output={"status": "success"})

    load_plugins()
    schemas = [
        s for s in get_tool_schemas(0)
        if (s.get("function") or {}).get("name") in ("interact_with_ui", "desktop_control", "manage_desktop_window", "execute_command")
    ]
    orch = NativeOrchestrator(stream_provider=mock_stream, tool_executor=mock_exec, enable_cycle_detection=False)
    ledger = TurnExecutionLedger()
    messages = [
        {"role": "user", "content": "Focus the Fixture window and click Confirm."},
        {"role": "assistant", "content": "Confirmed earlier task."},
        {"role": "user", "content": "Fix README.md in my workspace."},
    ]
    async for _ in orch.generate_stream({"messages": messages, "tools": schemas}, ledger=ledger):
        pass

    assert executed == ["manage_desktop_window"]
    assert ledger._records["d1"].rejection_reason == "unrequested_desktop_control"
    assert ledger._records["w1"].rejection_reason == "unrequested_window_focus"
    assert ledger._records["w2"].rejection_reason == "unrequested_window_focus"


@pytest.mark.asyncio
async def test_headless_text_fallback_blocks_unrequested_window_focus(tmp_path: Path) -> None:
    """Text-fallback tool parsing in CLI headless mode cannot bypass the desktop/window intent guard."""
    load_plugins()
    dispatched: list[tuple[str, dict]] = []
    turn = 0

    async def mock_stream(_payload):
        nonlocal turn
        turn += 1
        if turn == 1:
            # Plain JSON text tool call that bypasses structured tool_calls and hits headless_execute_tool directly
            res_direct = await orch.execute_tool("manage_desktop_window", action="focus", target="Zen Browser")
            assert "unrequested_window_focus" in str(res_direct)
            yield {
                "choices": [{
                    "delta": {
                        "content": '<tool_call>{"name": "manage_desktop_window", "arguments": {"action": "focus", "target": "Zen Browser"}}</tool_call>'
                    },
                    "finish_reason": "stop",
                }]
            }
        else:
            yield {"choices": [{"delta": {"content": "Stopped without focusing window."}, "finish_reason": "stop"}]}

    async def mock_exec(name: str, args: dict):
        dispatched.append((name, dict(args)))
        return ToolResult(True, output={"status": "success"})

    central = CentralExecutionLedger(db_path=tmp_path / "fallback.db", raise_on_persistence_error=True)
    orch = NativeOrchestrator(central_ledger=central, stream_provider=mock_stream, tool_executor=mock_exec, max_steps=2)
    try:
        rc = await _execute_headless_run("Read README.md in my workspace", orchestrator=orch, output_json=True)
    finally:
        central.close()

    assert rc == 2
    assert dispatched == []


def test_cleanup_never_killpgs_inherited_pgid_exited_proc_or_unmocked_chromium_mock(tmp_path: Path, monkeypatch) -> None:
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(desk_bench.os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))
    monkeypatch.setattr(desk_bench.os, "getpgrp", lambda: 111111)

    # 1. Real Popen inheriting an unrelated compositor/session PGID (pgid != pid) MUST NOT killpg
    inherited_proc = MagicMock(spec=desk_bench._REAL_POPEN)
    inherited_proc.pid = 424242
    inherited_proc.poll.return_value = None
    monkeypatch.setattr(desk_bench.os, "getpgid", lambda _pid: 888888)
    desk_bench._cleanup_fixture_processes((inherited_proc, None))
    assert killed == []
    inherited_proc.terminate.assert_called_once()

    # 2. Already-exited Popen MUST NOT call getpgid or killpg
    exited_proc = MagicMock(spec=desk_bench._REAL_POPEN)
    exited_proc.pid = 434343
    exited_proc.poll.return_value = 0
    monkeypatch.setattr(desk_bench.os, "getpgid", lambda pid: (_ for _ in ()).throw(AssertionError(f"getpgid({pid}) on exited proc")))
    desk_bench._cleanup_fixture_processes((exited_proc, None))
    assert killed == []

    # 3. Unmocked IsolatedChromiumSession._kill_proc never invokes real os.killpg/getpgid on a mock _proc
    monkeypatch.setattr(cross_bench.os, "getpgid", lambda pid: (_ for _ in ()).throw(AssertionError(f"getpgid({pid}) on mock chromium")))
    sess = cross_bench.IsolatedChromiumSession(tmp_path / "chrom_safe")
    fake_proc = MagicMock(pid=7777)
    sess._proc = fake_proc
    sess._kill_proc()
    assert sess._proc is None
    fake_proc.terminate.assert_called_once()
