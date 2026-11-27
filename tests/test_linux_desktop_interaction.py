"""Offline Section 9 regression tests for Zelyr Linux desktop observation, input, safety, and recovery."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from zelyr.agents.actuator_arbiter import ActuatorArbiter
from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.core.plugins import filter_tool_schemas_by_tier, get_tool_schemas, load_plugins
from zelyr.engine.error_classifier import ErrorClassifier
from zelyr.engine.execution_ledger import (
    TurnExecutionLedger,
    _is_inspection_action,
    compute_policy_compliance,
    extract_desktop_interaction_prohibition,
    is_mutating_tool_call,
)
from zelyr.engine.execution_policy import (
    ExecutionPolicyConfig,
    ExecutionPolicyEngine,
    PolicyDecisionOutcome,
    RiskLevel,
)
from zelyr.engine.hyprland_ipc import HyprlandIPCClient
from zelyr.engine.observation_framing import ObservationFramingEngine
from zelyr.services.os_controller.hyprland import HyprlandController
from zelyr.tools.desktop_control import DesktopAutomationTool
from zelyr.tools.input_controller import (
    BackendUnavailableError,
    DesktopUnavailableError,
    InputTimeoutError,
    InvalidCoordinatesError,
    InvalidInputKeyError,
    WaylandInputController,
    YdoToolPermissionError,
    parse_key_combo,
)
from zelyr.tools.screen_perception import CaptureScreenTool, capture_desktop_observation


def _make_png_bytes(color: tuple[int, int, int] = (20, 120, 80), size: tuple[int, int] = (160, 90)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


SAMPLE_MONITORS = [
    {"id": 0, "name": "HDMI-A-1", "x": 0, "y": 0, "width": 1920, "height": 1080, "logical_width": 1920, "logical_height": 1080, "scale": 1.0, "focused": False},
    {"id": 1, "name": "DP-1", "x": 0, "y": 1080, "width": 1920, "height": 1080, "logical_width": 1920, "logical_height": 1080, "scale": 1.0, "focused": True},
    {"id": 2, "name": "eDP-1", "x": 1920, "y": 540, "width": 1920, "height": 1080, "logical_width": 1920, "logical_height": 1080, "scale": 1.0, "focused": False},
]


# ---------------------------------------------------------------------------
# 1. Environment, Backend, Permission & Error Taxonomy Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_desktop_unavailable_when_display_missing(monkeypatch):
    for var in ("WAYLAND_DISPLAY", "DISPLAY", "HYPRLAND_INSTANCE_SIGNATURE"):
        monkeypatch.delenv(var, raising=False)

    obs = await capture_desktop_observation()
    assert obs["success"] is False
    assert obs["error_type"] == "DESKTOP_UNAVAILABLE"

    ctrl = WaylandInputController(auto_check=False)
    with pytest.raises(DesktopUnavailableError) as exc_info:
        ctrl.mouse_move(100, 100)
    assert exc_info.value.error_type == "DESKTOP_UNAVAILABLE"

    tool = DesktopAutomationTool(wayland_controller=ctrl)
    res = await tool.execute(action="mouse_click", button="left")
    assert res.success is False
    assert res.error_type == "DESKTOP_UNAVAILABLE"

    cl = ErrorClassifier.classify("desktop_control", output=json.dumps(res.output))
    assert cl.error_type == "DESKTOP_UNAVAILABLE"
    assert cl.is_recoverable is False


@pytest.mark.asyncio
async def test_backend_unavailable_when_binaries_and_uinput_missing(monkeypatch):
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)

    with patch("zelyr.tools.screen_perception.shutil.which", return_value=None):
        obs = await capture_desktop_observation()
    assert obs["success"] is False
    assert obs["error_type"] == "BACKEND_UNAVAILABLE"
    assert ErrorClassifier.classify("capture_screen", output=json.dumps(obs)).is_recoverable is False

    with patch("zelyr.tools.input_controller.shutil.which", return_value=None):
        ctrl = WaylandInputController(auto_check=False, uinput_path="/nonexistent/zelyr_uinput")
    with pytest.raises(BackendUnavailableError) as exc_info:
        ctrl.mouse_click("left")
    assert exc_info.value.error_type == "BACKEND_UNAVAILABLE"


@pytest.mark.asyncio
async def test_uinput_permission_denied_surfaces_structured_error(monkeypatch, tmp_path):
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    fake_uinput = tmp_path / "uinput"
    fake_uinput.write_bytes(b"")

    with patch("zelyr.tools.input_controller.shutil.which", return_value=None):
        ctrl = WaylandInputController(auto_check=False, uinput_path=str(fake_uinput))

    with patch("zelyr.tools.input_controller.os.open", side_effect=PermissionError(13, "Permission denied")):
        with pytest.raises(YdoToolPermissionError) as exc_info:
            ctrl.mouse_click("left")
        assert exc_info.value.error_type == "PERMISSION_DENIED"

        tool = DesktopAutomationTool(wayland_controller=ctrl)
        res = await tool.execute(action="keyboard_press", key="enter")
        assert res.success is False
        assert res.error_type == "PERMISSION_DENIED"


# ---------------------------------------------------------------------------
# 2. Coordinate Bounds, Key Combo Validation, Timeouts & Button Release
# ---------------------------------------------------------------------------


def test_coordinate_validation_across_multi_monitor_rectangles():
    ctrl = WaylandInputController(auto_check=False, bounds_provider=lambda: SAMPLE_MONITORS)
    assert ctrl.validate_coordinates(960, 1620) == (960, 1620)
    assert ctrl.validate_coordinates(2400, 900) == (2400, 900)

    for bad_x, bad_y in [
        (-5, 100),
        (9999, 9999),
        (2500, 100),  # Dead corner above eDP-1
        (float("nan"), 100),
        (True, 100),
        ("not-a-num", 100),
        (None, 50),
    ]:
        with pytest.raises(InvalidCoordinatesError) as exc_info:
            ctrl.validate_coordinates(bad_x, bad_y)
        assert exc_info.value.error_type == "INVALID_COORDINATES"


def test_key_combo_parsing_and_invalid_input_rejection():
    names, codes = parse_key_combo("ctrl+shift+t")
    assert names == ["ctrl", "shift", "t"]
    assert codes == [29, 42, 20]

    names2, codes2 = parse_key_combo("return")
    assert names2 == ["return"]
    assert codes2 == [28]

    assert parse_key_combo("-")[1] == [12]

    ctrl = WaylandInputController(auto_check=False)
    for bad in ("", "   ", "unknown_foo_key", "ctrl+"):
        with pytest.raises(InvalidInputKeyError):
            parse_key_combo(bad)
    with patch.dict("os.environ", {"WAYLAND_DISPLAY": "wayland-1"}):
        with pytest.raises(InvalidInputKeyError):
            ctrl.mouse_click(button="invalid_btn")
        with pytest.raises(InvalidInputKeyError):
            ctrl.mouse_click(button="left", clicks=0)
        with pytest.raises(InvalidInputKeyError):
            ctrl.keyboard_type("")


@pytest.mark.asyncio
async def test_input_timeout_and_modifier_release_on_failure(monkeypatch):
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    monkeypatch.setenv("HYPRLAND_INSTANCE_SIGNATURE", "sig")

    def _timeout_runner(*_a, **_kw):
        raise InputTimeoutError("Input command timed out")

    ctrl = WaylandInputController(
        auto_check=False,
        subprocess_runner=_timeout_runner,
        bounds_provider=lambda: SAMPLE_MONITORS,
    )
    ctrl.hyprctl_bin = "/usr/bin/hyprctl"
    tool = DesktopAutomationTool(wayland_controller=ctrl)
    res = await tool.execute(action="mouse_move", x=500, y=1500)
    assert res.success is False
    assert res.error_type == "INPUT_TIMEOUT"
    cl = ErrorClassifier.classify("desktop_control", output=json.dumps(res.output))
    assert cl.error_type == "INPUT_TIMEOUT"


@pytest.mark.asyncio
async def test_cancelled_error_closes_uinput_and_propagates(monkeypatch):
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    ctrl = WaylandInputController(auto_check=False)
    ctrl.close = MagicMock(wraps=ctrl.close)

    tool = DesktopAutomationTool(wayland_controller=ctrl)
    with patch.object(tool, "_action_mouse_click", side_effect=asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError):
            await tool.execute(action="mouse_click", button="left")
    ctrl.close.assert_called()


# ---------------------------------------------------------------------------
# 3. Screenshot Observation, Diffing, Timeout & Corrupt Image Handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_capture_observation_diffing_and_clean_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    monkeypatch.setenv("HYPRLAND_INSTANCE_SIGNATURE", "sig")

    png1 = _make_png_bytes((20, 40, 80), (640, 360))
    png2 = _make_png_bytes((220, 40, 40), (640, 360))
    frames = [png1, png1, png2]

    def _fake_runner(cmd, _timeout):
        out_path = cmd[-1]
        Path(out_path).write_bytes(frames.pop(0))
        return 0, b"", ""

    mock_ipc = AsyncMock(spec=HyprlandIPCClient)
    mock_ipc.can_connect.return_value = True
    mock_ipc.get_monitors.return_value = SAMPLE_MONITORS
    mock_ipc.get_clients.return_value = [
        {"address": "0xabc", "title": "Fixture", "class": "fixture", "at": [100, 1200], "size": [600, 400], "workspace": {"id": 15, "name": "15"}, "monitor": 1, "focused": True}
    ]
    mock_ipc.get_active_window.return_value = mock_ipc.get_clients.return_value[0]
    mock_ipc.get_cursor_pos.return_value = (400, 1400)

    with patch("zelyr.tools.screen_perception.shutil.which", return_value="/usr/bin/grim"), \
         patch("zelyr.tools.screen_perception._SCREENSHOT_DIR", tmp_path):
        obs1 = await capture_desktop_observation(target="active_window", runner=_fake_runner, ipc_client=mock_ipc)
        obs2 = await capture_desktop_observation(target="active_window", runner=_fake_runner, ipc_client=mock_ipc)
        obs3 = await capture_desktop_observation(target="active_window", runner=_fake_runner, ipc_client=mock_ipc)

    assert obs1["success"] is True
    assert obs1["captured_region"] == {"x": 100, "y": 1200, "width": 600, "height": 400, "scale": 1.0, "window": "Fixture"}
    assert obs1["changed_since_last"] is None
    assert obs2["changed_since_last"] is False
    assert obs2["diff_ratio"] == 0.0
    assert obs3["changed_since_last"] is True
    assert obs3["diff_ratio"] > 0.1
    assert isinstance(obs3.pop("_vision_b64"), str)

    # ObservationFramingEngine strips any _vision_b64 cleanly
    framed = ObservationFramingEngine.frame_observation("capture_screen", {**obs3, "_vision_b64": "A" * 5000})
    assert "AAAA" not in framed
    assert "omitted" in framed


@pytest.mark.asyncio
async def test_capture_timeout_corrupt_image_and_missing_window(monkeypatch):
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    monkeypatch.setenv("HYPRLAND_INSTANCE_SIGNATURE", "sig")

    mock_ipc = AsyncMock(spec=HyprlandIPCClient)
    mock_ipc.can_connect.return_value = True
    mock_ipc.get_monitors.return_value = SAMPLE_MONITORS
    mock_ipc.get_clients.return_value = []
    mock_ipc.get_active_window.return_value = {}
    mock_ipc.get_cursor_pos.return_value = (10, 10)

    with patch("zelyr.tools.screen_perception.shutil.which", return_value="/usr/bin/grim"):
        obs_missing = await capture_desktop_observation(target="window", window="NonExistentWin", ipc_client=mock_ipc)
        assert obs_missing["error_type"] == "WINDOW_NOT_FOUND"

        obs_to = await capture_desktop_observation(
            target="full",
            runner=lambda *_a: (_ for _ in ()).throw(subprocess.TimeoutExpired("grim", 4.0)),
            ipc_client=mock_ipc,
        )
        assert obs_to["error_type"] == "SCREENSHOT_FAILED"
        assert "timed out" in obs_to["error"].lower()

        def _corrupt_runner(cmd, _t):
            Path(cmd[-1]).write_bytes(b"not-a-valid-png")
            return 0, b"", ""

        obs_corrupt = await capture_desktop_observation(target="full", runner=_corrupt_runner, ipc_client=mock_ipc)
        assert obs_corrupt["error_type"] == "SCREENSHOT_FAILED"


# ---------------------------------------------------------------------------
# 4. Focus Loss Detection & Recovery + Hyprland Geometry + Controller Delegation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_focus_loss_blocks_input_then_recovers_after_refocus(monkeypatch):
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    monkeypatch.setenv("HYPRLAND_INSTANCE_SIGNATURE", "sig")

    active_state = {"title": "Other App", "class": "other", "address": "0x222", "at": [0, 0], "size": [300, 300]}
    mock_ipc = AsyncMock(spec=HyprlandIPCClient)
    mock_ipc.can_connect.return_value = True
    mock_ipc.get_monitors.return_value = SAMPLE_MONITORS
    mock_ipc.get_active_window.side_effect = lambda: dict(active_state)

    ctrl = WaylandInputController(auto_check=False, bounds_provider=lambda: SAMPLE_MONITORS)
    ctrl.mouse_click = MagicMock()
    ctrl.keyboard_type = MagicMock()
    ctrl.keyboard_press = MagicMock()
    tool = DesktopAutomationTool(wayland_controller=ctrl, ipc_client=mock_ipc)

    res_lost = await tool.execute(
        action="mouse_click",
        x=400,
        y=1400,
        button="left",
        expected_window="Zelyr Fixture",
    )
    assert res_lost.success is False
    assert res_lost.error_type == "FOCUS_LOST"
    assert "FOCUS_LOST" in (res_lost.error or "")
    ctrl.mouse_click.assert_not_called()

    # Keyboard mutations must also fail closed with zero input while focus validation fails
    res_type_lost = await tool.execute(action="keyboard_type", text="abc", expected_window="Zelyr Fixture")
    assert res_type_lost.success is False and res_type_lost.error_type == "FOCUS_LOST"
    ctrl.keyboard_type.assert_not_called()

    res_key_lost = await tool.execute(action="keyboard_press", key="enter", expected_window="Zelyr Fixture")
    assert res_key_lost.success is False and res_key_lost.error_type == "FOCUS_LOST"
    ctrl.keyboard_press.assert_not_called()

    cl = ErrorClassifier.classify("desktop_control", output=json.dumps(res_lost.output))
    assert cl.error_type == "FOCUS_LOST"
    assert "manage_desktop_window" in cl.allowed_actions

    # Simulate refocus recovery
    active_state["title"] = "Zelyr Fixture"
    active_state["address"] = "0x111"
    res_ok = await tool.execute(
        action="mouse_click",
        x=400,
        y=1400,
        button="left",
        expected_window="Zelyr Fixture",
    )
    assert res_ok.success is True
    ctrl.mouse_click.assert_called_once()


@pytest.mark.asyncio
async def test_hyprland_geometry_and_controller_delegation():
    client = HyprlandIPCClient(socket_path=None)
    raw_mons = json.dumps([
        {"id": 0, "name": "DP-1", "x": 0, "y": 1080, "width": 3840, "height": 2160, "scale": 2.0, "focused": True, "activeWorkspace": {"id": 15, "name": "15"}},
    ])
    with patch.object(client, "send_command", AsyncMock(return_value=raw_mons)):
        geom = await client.get_desktop_geometry()
    assert geom["monitors"][0]["logical_width"] == 1920
    assert geom["monitors"][0]["logical_height"] == 1080
    assert HyprlandIPCClient.point_in_monitors(100, 1200, geom["monitors"]) is not None
    assert HyprlandIPCClient.point_in_monitors(100, 100, geom["monitors"]) is None

    hc = HyprlandController()
    fake_ctrl = MagicMock()
    hc._ctrl = fake_ctrl
    hc.click(500, 1500, button="left", clicks=2)
    fake_ctrl.mouse_click.assert_called_once_with(button="left", clicks=2, x=500, y=1500)


@pytest.mark.asyncio
async def test_hyprland_057_lua_dispatch_fallback_and_compact_schema_enums(monkeypatch):
    """Verify Hyprland 0.57+ Lua eval fallback and preservation of enum actions in CompactSchemaGenerator."""
    from zelyr.engine.schema_compact import CompactSchemaGenerator

    calls: list[str] = []

    client = HyprlandIPCClient(socket_path=None)

    async def fake_send(cmd: str) -> str:
        calls.append(cmd)
        if cmd.startswith("dispatch "):
            return 'error: 3 [string "return hl.dispatch(...)"]:1: \')\' expected near \'x\'\n -> Note: dispatch in lua is a shorthand for hl.dispatch'
        if cmd.startswith("eval "):
            return "ok"
        if cmd == "j/activeworkspace":
            return json.dumps({"id": 15, "name": "15"})
        return "{}"

    with patch.object(client, "send_command", side_effect=fake_send):
        res = await client.dispatch("workspace", "15", verify=True)
    assert res["verified"] is True
    assert calls[0] == "dispatch workspace 15"
    assert calls[1] == 'eval hl.dispatch(hl.dsp.focus({workspace="15"}))'

    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    monkeypatch.setenv("HYPRLAND_INSTANCE_SIGNATURE", "sig")
    runner_cmds: list[list[str]] = []

    def fake_cursor_runner(cmd, **_kw):
        runner_cmds.append(list(cmd))
        if len(runner_cmds) == 1:
            return subprocess.CompletedProcess(cmd, 7, "Note: dispatch in lua is a shorthand for hl.dispatch", "")
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    ctrl = WaylandInputController(auto_check=False, subprocess_runner=fake_cursor_runner, bounds_provider=lambda: SAMPLE_MONITORS)
    ctrl.hyprctl_bin = "/usr/bin/hyprctl"
    ctrl.mouse_move(400, 1400)
    ctrl.mouse_move(450, 1450)
    assert runner_cmds[0][:3] == ["/usr/bin/hyprctl", "dispatch", "movecursor"]
    assert runner_cmds[1] == ["/usr/bin/hyprctl", "eval", "hl.dispatch(hl.dsp.cursor.move({x=400, y=1400}))"]
    # Second call reuses cached Lua mode directly without extra dispatch trial
    assert runner_cmds[2] == ["/usr/bin/hyprctl", "eval", "hl.dispatch(hl.dsp.cursor.move({x=450, y=1450}))"]

    compact = CompactSchemaGenerator.compact_schema({"type": "function", "function": {"name": "desktop_control", "description": DesktopAutomationTool().description, "parameters": DesktopAutomationTool().schema}})
    enum_actions = compact["function"]["parameters"]["properties"]["action"].get("enum") or []
    assert {"mouse_click", "keyboard_type", "keyboard_press", "mouse_move"} <= set(enum_actions)


# ---------------------------------------------------------------------------
# 5. Arbiter, Schema Registration, Policy, Prohibition Enforcement & Vision Wire
# ---------------------------------------------------------------------------


def test_arbiter_routes_desktop_gui_tasks_preserving_browser_and_os_tiers():
    load_plugins()
    schemas = get_tool_schemas(0)
    t3_names = {s["function"]["name"] for s in filter_tool_schemas_by_tier(schemas, "tier3_vision")}
    assert {"capture_screen", "desktop_control", "manage_desktop_window", "execute_command"} <= t3_names

    t2_names = {s["function"]["name"] for s in filter_tool_schemas_by_tier(schemas, "tier2_browser", is_browser=True)}
    assert "desktop_control" not in t2_names
    assert "interact_with_ui" not in t2_names

    assert ActuatorArbiter.arbitrate(
        "Observe the Zelyr Desktop Test Fixture window on workspace 15 using capture_screen and summarize its layout.",
        active_window={"class": "kitty", "title": "terminal"},
        bridge_connected=False,
    ) == "tier3_vision"
    assert ActuatorArbiter.arbitrate(
        "Click the Increment Counter button in the Zelyr Desktop Test Fixture window.",
        active_window={"class": "kitty", "title": "terminal"},
        bridge_connected=False,
    ) == "tier3_vision"
    assert ActuatorArbiter.arbitrate(
        "Do NOT click, type into, or interact with the fixture window. Inspect it read-only.",
        active_window=None,
        bridge_connected=False,
    ) == "tier3_vision"
    assert ActuatorArbiter.arbitrate(
        "Pause my music playback",
        active_window={"class": "zen", "title": "YouTube"},
        bridge_connected=True,
    ) == "tier1_ipc"
    assert ActuatorArbiter.arbitrate(
        "Click the submit button",
        active_window={"class": "google-chrome", "title": "Checkout"},
        bridge_connected=True,
    ) == "tier2_browser"


@pytest.mark.asyncio
async def test_policy_and_prohibition_allow_read_only_block_desktop_mutation():
    cons_cfg = ExecutionPolicyConfig(policy_mode="conservative")
    eng = ExecutionPolicyEngine(config=cons_cfg)

    dec_cap = eng.evaluate("capture_screen", {"target": "active_window"}, required_ring=1, is_registered=True)
    assert dec_cap.allowed is True
    assert dec_cap.risk_level == RiskLevel.READ_ONLY

    dec_obs = eng.evaluate("desktop_control", {"action": "get_screen_size"}, required_ring=0, is_registered=True)
    assert dec_obs.allowed is True
    assert dec_obs.risk_level == RiskLevel.READ_ONLY

    dec_click = eng.evaluate("desktop_control", {"action": "mouse_click", "x": 10, "y": 20}, required_ring=0, is_registered=True)
    assert dec_click.outcome == PolicyDecisionOutcome.REQUIRE_CONFIRMATION
    assert dec_click.risk_level == RiskLevel.PRIVILEGED

    prohib_prompt = "Inspect the open desktop fixture window read-only. Do NOT click, type into, or modify the fixture window."
    assert extract_desktop_interaction_prohibition(prohib_prompt) is True
    assert _is_inspection_action("capture_screen", {"target": "active_window"}) is True
    assert _is_inspection_action("desktop_control", {"action": "observe"}) is True
    assert _is_inspection_action("manage_desktop_window", {"action": "list"}) is True
    assert is_mutating_tool_call("desktop_control", {"action": "mouse_click", "x": 1, "y": 2})[0] is True
    assert is_mutating_tool_call("execute_command", {"command": "wtype hello"})[0] is True
    assert is_mutating_tool_call("execute_command", {"command": "hyprctl dispatch focuswindow title:foo"})[0] is True

    # Verify orchestrator blocks prohibited desktop_control mutation before dispatch while allowing capture_screen
    orch = NativeOrchestrator()
    dispatched: list[str] = []

    async def fake_exec(name: str, **kwargs):
        dispatched.append(name)
        return json.dumps({"success": True, "status": "success", "output": {"message": "ok"}})

    orch.execute_tool = fake_exec

    step = 0

    async def mock_stream(_payload, _base_url, **_kw):
        nonlocal step
        step += 1
        if step == 1:
            yield {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "c1", "function": {"name": "desktop_control", "arguments": {"action": "mouse_click", "x": 200, "y": 200}}},
            ]}, "finish_reason": "tool_calls"}]}
        elif step == 2:
            yield {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "c2", "function": {"name": "capture_screen", "arguments": {"target": "active_monitor", "include_image": False}}},
            ]}, "finish_reason": "tool_calls"}]}
        else:
            yield {"choices": [{"delta": {"content": "Inspected read-only."}, "finish_reason": "stop"}]}

    with patch.object(orch, "_stream_model_events", side_effect=mock_stream):
        payload = {
            "messages": [{"role": "user", "content": prohib_prompt}],
            "tools": get_tool_schemas(0),
        }
        async for _ in orch.generate_stream(payload):
            pass

    assert dispatched == ["capture_screen"]
    comp = compute_policy_compliance(prohib_prompt, orch.current_ledger)
    assert comp["has_prohibitions"] is True
    assert comp["compliant"] is True
    assert comp["prohibited_attempts_blocked"] >= 1
    assert comp["prohibited_violations"] == 0


@pytest.mark.asyncio
async def test_reobserve_before_after_click_allowed_and_repeated_failing_click_blocked():
    """Verify capture_screen can run before & after desktop_control, while identical failing clicks hit cycle_detected."""
    orch = NativeOrchestrator()
    executed: list[tuple[str, dict]] = []

    async def fake_exec(name: str, **kwargs):
        executed.append((name, dict(kwargs)))
        if name == "desktop_control" and kwargs.get("action") == "mouse_click" and kwargs.get("expected_window") == "MissingFocus":
            return json.dumps({"success": False, "status": "error", "error_type": "FOCUS_LOST", "error": "focus lost"})
        return json.dumps({"success": True, "status": "success", "output": {"message": "ok"}})

    orch.execute_tool = fake_exec

    # Part 1: capture -> click -> capture works cleanly
    turn1 = 0

    async def stream_ok(_p, _b, **_kw):
        nonlocal turn1
        turn1 += 1
        if turn1 == 1:
            yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "o1", "function": {"name": "capture_screen", "arguments": {"target": "window", "window": "Fix"}}}]}, "finish_reason": "tool_calls"}]}
        elif turn1 == 2:
            yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "o2", "function": {"name": "desktop_control", "arguments": {"action": "mouse_click", "x": 500, "y": 1500}}}]}, "finish_reason": "tool_calls"}]}
        elif turn1 == 3:
            yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "o3", "function": {"name": "capture_screen", "arguments": {"target": "window", "window": "Fix"}}}]}, "finish_reason": "tool_calls"}]}
        else:
            yield {"choices": [{"delta": {"content": "Verified."}, "finish_reason": "stop"}]}

    with patch.object(orch, "_stream_model_events", side_effect=stream_ok):
        async for _ in orch.generate_stream({"messages": [{"role": "user", "content": "Click button and verify."}], "tools": get_tool_schemas(0)}):
            pass

    assert [n for n, _ in executed] == ["capture_screen", "desktop_control", "capture_screen"]

    # Part 2: repeated identical failing desktop click is blocked as cycle_detected
    executed.clear()
    orch2 = NativeOrchestrator()
    orch2.execute_tool = fake_exec
    turn2 = 0

    async def stream_fail(_p, _b, **_kw):
        nonlocal turn2
        turn2 += 1
        if turn2 in (1, 2):
            yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": f"f{turn2}", "function": {"name": "desktop_control", "arguments": {"action": "mouse_click", "x": 500, "y": 1500, "expected_window": "MissingFocus"}}}]}, "finish_reason": "tool_calls"}]}
        else:
            yield {"choices": [{"delta": {"content": "Blocked duplicate failing click."}, "finish_reason": "stop"}]}

    with patch.object(orch2, "_stream_model_events", side_effect=stream_fail):
        async for _ in orch2.generate_stream({"messages": [{"role": "user", "content": "Click button."}], "tools": get_tool_schemas(0)}):
            pass

    assert len(executed) == 1
    assert any(r.rejection_reason == "cycle_detected" for r in orch2.current_ledger._records.values())


@pytest.mark.asyncio
async def test_transient_vision_image_attached_once_on_wire_without_history_bloat():
    orch = NativeOrchestrator()
    orch._vision_support_cache[("http://127.0.0.1:11434", "qwen3.6:35b-a3b")] = True
    wire_calls = []

    class FakeResp:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            pass

        async def aiter_lines(self):
            yield json.dumps({"message": {"role": "assistant", "content": "Saw button."}, "done": True, "done_reason": "stop"})

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            pass

        async def post(self, *_a, **_kw):
            return MagicMock(status_code=200)

        def stream(self, _method, _url, json=None):
            wire_calls.append(json)
            return FakeResp()

    b64 = base64.b64encode(b"fakejpeg").decode("ascii")
    payload = {
        "model": "qwen3.6:35b-a3b",
        "messages": [{"role": "user", "content": "What is shown?"}],
        "_pending_vision_b64": b64,
    }
    with patch("httpx.AsyncClient", return_value=FakeClient()):
        async for _ in orch.generate_stream(payload):
            pass

    # payload itself must NOT keep _pending_vision_b64 or 'images' in payload["messages"]
    assert "_pending_vision_b64" not in payload
    assert all("images" not in m for m in payload["messages"])
    # Wire payload had the single transient image attached
    assert len(wire_calls) == 1
    assert wire_calls[0]["messages"][-1]["images"] == [b64]


@pytest.mark.asyncio
async def test_focus_recovery_after_focus_lost_not_blocked_as_redundant():
    """Stateful manage_desktop_window(focus) recovers after FOCUS_LOST while immediate duplicates block."""
    from zelyr.agents.native_orchestrator import NativeOrchestrator
    from zelyr.engine.execution_ledger import TurnExecutionLedger

    turn = 0
    seen_tools_per_turn = []

    async def provider(payload):
        nonlocal turn
        turn += 1
        seen_tools_per_turn.append([
            (t.get("function") or {}).get("name") or t.get("name")
            for t in (payload.get("tools") or [])
            if isinstance(t, dict)
        ])
        if turn == 1:
            call = ("call_f1", "manage_desktop_window", '{"action": "focus", "target": "Fixture"}')
        elif turn == 2:
            call = ("call_c1", "desktop_control", '{"action": "mouse_click", "x": 100, "y": 100, "expected_window": "Fixture"}')
        elif turn == 3:
            # Must NOT be blocked as redundant_action because FOCUS_LOST intervened!
            call = ("call_f2", "manage_desktop_window", '{"action": "focus", "target": "Fixture"}')
        elif turn == 4:
            # Consecutive back-to-back duplicate focus STILL must be blocked as redundant_action
            call = ("call_f3", "manage_desktop_window", '{"action": "focus", "target": "Fixture"}')
        else:
            yield {"choices": [{"delta": {"content": "Recovered focus."}}]}
            return

        yield {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": call[0],
                        "type": "function",
                        "function": {"name": call[1], "arguments": call[2]},
                    }]
                }
            }]
        }

    executed = []

    async def mock_executor(name: str, args: dict):
        executed.append((name, args.get("action")))
        if name == "desktop_control":
            return {"success": False, "error": "FOCUS_LOST: expected Fixture", "error_type": "FOCUS_LOST"}
        return {"success": True, "output": "focused"}

    schemas = [
        {"type": "function", "function": {"name": "manage_desktop_window", "parameters": {"type": "object", "properties": {"action": {"type": "string"}, "target": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "capture_screen", "parameters": {"type": "object", "properties": {"target": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "desktop_control", "parameters": {"type": "object", "properties": {"action": {"type": "string"}, "x": {"type": "integer"}, "y": {"type": "integer"}, "expected_window": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "execute_command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}},
    ]

    ledger = TurnExecutionLedger()
    orch = NativeOrchestrator(stream_provider=provider, tool_executor=mock_executor, enable_cycle_detection=True)
    payload = {
        "messages": [{"role": "user", "content": "Click button and recover focus if FOCUS_LOST."}],
        "tools": schemas,
    }
    with patch.object(
        NativeOrchestrator,
        "get_active_window_sync",
        return_value={"class": "python3", "title": "Zelyr Desktop Test Fixture"},
    ):
        async for _ in orch.generate_stream(payload, ledger=ledger):
            pass

    assert executed == [
        ("manage_desktop_window", "focus"),
        ("desktop_control", "mouse_click"),
        ("manage_desktop_window", "focus"),
    ]
    assert "redundant_action" in str(ledger._records["call_f3"].error)
    assert "desktop_control" in seen_tools_per_turn[2]
    assert "capture_screen" in seen_tools_per_turn[2]
    assert "manage_desktop_window" in seen_tools_per_turn[2]


@pytest.mark.asyncio
async def test_disabled_interact_with_ui_returns_tool_unavailable_and_keeps_desktop_tools():
    """Calling omitted interact_with_ui when desktop_control is active guides back without dropping desktop tools."""
    from unittest.mock import patch
    from zelyr.agents.native_orchestrator import NativeOrchestrator
    from zelyr.engine.execution_ledger import TurnExecutionLedger

    turn = 0
    seen_tools_after_rejection = []

    async def provider(payload):
        nonlocal turn
        turn += 1
        seen_tools_after_rejection.append([
            (t.get("function") or {}).get("name") or t.get("name")
            for t in (payload.get("tools") or [])
            if isinstance(t, dict)
        ])
        if turn == 1:
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_ui_1",
                            "type": "function",
                            "function": {"name": "interact_with_ui", "arguments": '{"instruction": "click button"}'},
                        }]
                    }
                }]
            }
        else:
            yield {"choices": [{"delta": {"content": "Used desktop_control instead."}}]}

    executed = []

    async def mock_executor(name: str, args: dict):
        executed.append(name)
        return {"success": True}

    schemas = [
        {"type": "function", "function": {"name": "capture_screen", "parameters": {"type": "object", "properties": {"target": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "desktop_control", "parameters": {"type": "object", "properties": {"action": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "manage_desktop_window", "parameters": {"type": "object", "properties": {"action": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "execute_command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}},
    ]

    ledger = TurnExecutionLedger()
    orch = NativeOrchestrator(stream_provider=provider, tool_executor=mock_executor, enable_cycle_detection=True)
    payload = {
        "messages": [{"role": "user", "content": "Click Apply Text in the window."}],
        "tools": schemas,
    }
    with patch.object(
        NativeOrchestrator,
        "get_active_window_sync",
        return_value={"class": "python3", "title": "Zelyr Desktop Test Fixture"},
    ):
        async for _ in orch.generate_stream(payload, ledger=ledger):
            pass

    assert executed == []
    tool_msgs = [m for m in payload["messages"] if m.get("role") == "tool"]
    assert tool_msgs
    assert "desktop_control" in tool_msgs[-1]["content"]
    assert "desktop_control" in seen_tools_after_rejection[-1]
    assert "capture_screen" in seen_tools_after_rejection[-1]

    with patch("zelyr.core.vision_engine.singleton.is_spotless_arms_available", return_value=False):
        assert NativeOrchestrator.get_tier_tool("Click Apply Text in desktop app") == "desktop_control"


@pytest.mark.asyncio
async def test_nested_vision_b64_extraction_and_capture_arg_normalization():
    """Nested _vision_b64 inside ToolResult and capture_screen window target aliases work cleanly."""
    from zelyr.agents.native_orchestrator import NativeOrchestrator
    from zelyr.engine.execution_ledger import TurnExecutionLedger
    from zelyr.tools.core import ToolResult

    captured_wire_vis = []
    turn = 0

    async def stream_provider(payload):
        nonlocal turn
        turn += 1
        captured_wire_vis.append((list(payload.get("messages") or []), payload.get("_pending_vision_b64")))
        if turn == 1:
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_dc_ss",
                            "type": "function",
                            "function": {
                                "name": "desktop_control",
                                "arguments": '{"action": "screenshot", "target": "Zelyr Desktop Test Fixture"}',
                            },
                        }]
                    }
                }]
            }
        else:
            yield {"choices": [{"delta": {"content": "Saw nested screenshot."}}]}

    async def mock_executor(name: str, args: dict):
        return ToolResult(True, output={"status": "success", "_vision_b64": "QUJDRA==", "message": "ok"})

    schemas = [
        {"type": "function", "function": {"name": "desktop_control", "parameters": {"type": "object", "properties": {"action": {"type": "string"}, "window": {"type": "string"}}, "additionalProperties": False}}},
        {"type": "function", "function": {"name": "capture_screen", "parameters": {"type": "object", "properties": {"target": {"type": "string"}, "window": {"type": "string"}}, "additionalProperties": False}}},
    ]
    norm_cap = NativeOrchestrator.normalize_tool_arguments(
        "capture_screen", {"target": "Zelyr Desktop Test Fixture"}, active_schemas=schemas
    )
    assert norm_cap == {"target": "window", "window": "Zelyr Desktop Test Fixture"}

    ledger = TurnExecutionLedger()
    orch = NativeOrchestrator(stream_provider=stream_provider, tool_executor=mock_executor)
    orch._vision_support_cache[("http://localhost:11434", "qwen3.6:35b-a3b")] = True
    payload = {
        "model": "qwen3.6:35b-a3b",
        "messages": [{"role": "user", "content": "Take screenshot."}],
        "tools": schemas,
    }
    async for _ in orch.generate_stream(payload, ledger=ledger):
        pass

    assert len(captured_wire_vis) == 2
    assert captured_wire_vis[1][1] == "QUJDRA=="
    tool_msgs = [m for m in payload["messages"] if m.get("role") == "tool"]
    assert "QUJDRA==" not in tool_msgs[-1]["content"]


def test_failed_capture_screen_resolved_by_later_valid_capture_while_preserving_mutation_rule():
    """Later valid capture_screen clears earlier capture_screen typo without relaxing mutation rules."""
    from zelyr.engine.execution_ledger import TurnExecutionLedger

    ledger = TurnExecutionLedger()
    ledger.record_proposed("cap_bad", "capture_screen", {"target": "Zelyr Desktop Test Fixture"})
    ledger.record_dispatched("cap_bad")
    ledger.record_observation("cap_bad", status="failed", output="", error="Unsupported capture target", exit_code=1)

    ledger.record_proposed("cap_ok", "capture_screen", {"target": "window", "window": "Zelyr Desktop Test Fixture"})
    ledger.record_dispatched("cap_ok")
    ledger.record_observation("cap_ok", status="success", output="ok", exit_code=None)
    assert ledger.unresolved_failures() == []

    # Mutating shell failure must NOT be cleared by read-only capture_screen!
    ledger.record_proposed("mut_bad", "execute_command", {"command": "false"})
    ledger.record_dispatched("mut_bad")
    ledger.record_observation("mut_bad", status="failed", output="", error="exit 1", exit_code=1)
    ledger.record_proposed("cap_ok2", "capture_screen", {"target": "window", "window": "Zelyr Desktop Test Fixture"})
    ledger.record_dispatched("cap_ok2")
    ledger.record_observation("cap_ok2", status="success", output="ok", exit_code=None)
    assert len(ledger.unresolved_failures()) == 1


@pytest.mark.asyncio
async def test_focus_loss_end_to_end_recovery_and_failed_recovery_remains_failure(monkeypatch):
    """End-to-end orchestrator + ledger check: recovered FOCUS_LOST passes while unrecovered FOCUS_LOST fails."""
    from zelyr.engine.execution_ledger import ExecutionStatus, TurnExecutionLedger, resolve_root_execution_outcome

    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    monkeypatch.setenv("HYPRLAND_INSTANCE_SIGNATURE", "sig")

    active_state = {"title": "Zelyr Fixture", "class": "python3", "address": "0x111", "at": [100, 1100], "size": [640, 420]}
    mock_ipc = AsyncMock(spec=HyprlandIPCClient)
    mock_ipc.can_connect.return_value = True
    mock_ipc.get_monitors.return_value = SAMPLE_MONITORS
    mock_ipc.get_active_window.side_effect = lambda: dict(active_state)
    mock_ipc.get_cursor_pos.return_value = (400, 1400)

    ctrl = WaylandInputController(auto_check=False, bounds_provider=lambda: SAMPLE_MONITORS)
    ctrl.mouse_click = MagicMock()
    dc_tool = DesktopAutomationTool(wayland_controller=ctrl, ipc_client=mock_ipc)

    async def _run_case(*, recover_focus: bool, retry_after_focus: bool):
        active_state["title"] = "Zelyr Fixture"
        active_state["address"] = "0x111"
        ctrl.mouse_click.reset_mock()
        turn = 0
        final_chunks = []

        async def provider(_payload):
            nonlocal turn
            turn += 1
            if turn == 1:
                call = ("c_obs", "capture_screen", '{"target": "window", "window": "Zelyr Fixture"}')
            elif turn == 2:
                # Focus shifted to auxiliary distraction window before first click executes
                active_state["title"] = "Zelyr Distraction Window"
                active_state["address"] = "0x222"
                call = ("c_clk1", "desktop_control", '{"action": "mouse_click", "x": 400, "y": 1400, "expected_window": "Zelyr Fixture"}')
            elif turn == 3:
                call = ("c_foc", "manage_desktop_window", '{"action": "focus", "target": "Zelyr Fixture"}')
            elif turn == 4 and retry_after_focus:
                call = ("c_clk2", "desktop_control", '{"action": "mouse_click", "x": 400, "y": 1400, "expected_window": "Zelyr Fixture"}')
            else:
                yield {"choices": [{"delta": {"content": "Finished run."}}]}
                return
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": call[0],
                            "type": "function",
                            "function": {"name": call[1], "arguments": call[2]},
                        }]
                    }
                }]
            }

        async def executor(name: str, args: dict):
            if name == "capture_screen":
                return {"success": True, "status": "success", "target": "window", "message": "Captured Zelyr Fixture."}
            if name == "manage_desktop_window":
                if recover_focus:
                    active_state["title"] = "Zelyr Fixture"
                    active_state["address"] = "0x111"
                    return {"status": "success", "action": "focus", "target": "Zelyr Fixture", "verified": True}
                return {"status": "error", "error_type": "WINDOW_NOT_FOUND", "error": "Target window not found"}
            res = await dc_tool.execute(args)
            return res.to_dict(tool=name, arguments=args)

        ledger = TurnExecutionLedger()
        orch = NativeOrchestrator(stream_provider=provider, tool_executor=executor, enable_cycle_detection=True)
        schemas = [
            s for s in get_tool_schemas(0)
            if (s.get("function") or {}).get("name") in ("capture_screen", "desktop_control", "manage_desktop_window", "execute_command")
        ]
        async for ch in orch.generate_stream(
            {"messages": [{"role": "user", "content": "Click Increment Counter cleanly."}], "tools": schemas},
            ledger=ledger,
        ):
            final_chunks.append(ch)
        outcome = resolve_root_execution_outcome(
            prompt="Click Increment Counter cleanly.",
            ledger=ledger,
            final_text="Finished run.",
        )
        return ledger, outcome

    # Case 1: Genuine FOCUS_LOST -> refocus -> successful retry
    ledger_ok, outcome_ok = await _run_case(recover_focus=True, retry_after_focus=True)
    assert ctrl.mouse_click.call_count == 1
    assert any("FOCUS_LOST" in str(r.error or "") for r in ledger_ok.failed_actions())
    assert ledger_ok.unresolved_failures() == []
    assert outcome_ok.exit_code == 0
    assert outcome_ok.execution_status in (ExecutionStatus.COMPLETED, ExecutionStatus.COMPLETED_UNVERIFIED)

    # Case 2: Refocus happens, but retry is NOT performed -> remains failure!
    ledger_noretry, outcome_noretry = await _run_case(recover_focus=True, retry_after_focus=False)
    assert ctrl.mouse_click.call_count == 0
    assert len(ledger_noretry.unresolved_failures()) == 1
    assert outcome_noretry.exit_code == 2
    assert outcome_noretry.execution_status == ExecutionStatus.FAILED

    # Case 3: Refocus fails -> retry also fails -> remains failure!
    ledger_badrefocus, outcome_badrefocus = await _run_case(recover_focus=False, retry_after_focus=True)
    assert ctrl.mouse_click.call_count == 0
    assert len(ledger_badrefocus.unresolved_failures()) >= 1
    assert outcome_badrefocus.exit_code == 2
    assert outcome_badrefocus.execution_status == ExecutionStatus.FAILED


@pytest.mark.asyncio
async def test_cancelled_error_releases_and_destroys_active_uinput_fds(monkeypatch):
    """CancelledError mid-actuation releases held keys/buttons, calls UI_DEV_DESTROY, and closes fds."""
    from zelyr.tools.input_controller import _UI_DEV_DESTROY, _UInputBackend

    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    uinput = _UInputBackend(settle_s=0.0)
    uinput._pointer_fd = 101
    uinput._keyboard_fd = 102
    uinput._active_buttons = {272}
    uinput._active_keys = {29}

    writes = []
    ioctls = []
    closed_fds = []

    monkeypatch.setattr("os.write", lambda fd, data: (writes.append((fd, len(data))), len(data))[1])
    monkeypatch.setattr("fcntl.ioctl", lambda fd, req, *_a: ioctls.append((fd, req)) or 0)
    monkeypatch.setattr("os.close", lambda fd: closed_fds.append(fd))

    ctrl = WaylandInputController(auto_check=False, bounds_provider=lambda: SAMPLE_MONITORS)
    ctrl._uinput = uinput
    tool = DesktopAutomationTool(wayland_controller=ctrl)

    with patch.object(tool, "_action_mouse_click", side_effect=asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError):
            await tool.execute(action="mouse_click", button="left")

    assert uinput._closed is True
    assert uinput._active_buttons == set()
    assert uinput._active_keys == set()
    assert (101, _UI_DEV_DESTROY) in ioctls
    assert (102, _UI_DEV_DESTROY) in ioctls
    assert set(closed_fds) == {101, 102}
    assert any(fd == 101 for fd, _ in writes)
    assert any(fd == 102 for fd, _ in writes)


def test_focus_recovery_validator_tri_state_and_fixture_cleanup(monkeypatch):
    """Validator enforces PASS/FAIL/UNVERIFIED semantics and fixtures clean up after success and failure."""
    import importlib.util

    bench_path = Path(__file__).resolve().parents[1] / "scripts" / "linux_desktop_benchmark.py"
    spec = importlib.util.spec_from_file_location("linux_desktop_benchmark", bench_path)
    assert spec is not None and spec.loader is not None
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

    base_steal = {
        "stolen": True,
        "active_before": {"address": "0x111", "title": "Zelyr Desktop Test Fixture"},
        "active_after": {"address": "0x222", "title": "Zelyr Distraction Window"},
        "fix_state_at_steal": {"counter": 0, "mouse_clicks": 0, "key_presses": 0, "applied_text": ""},
        "fix_state_after_steal": {"is_active_window": False},
        "dist_state_at_steal": {"is_active_window": True},
    }
    good_records = [
        {"call_id": "c1", "tool_name": "capture_screen", "state": "success", "arguments": {"target": "window", "window": "Zelyr Desktop Test Fixture"}, "error": None},
        {"call_id": "c2", "tool_name": "desktop_control", "state": "failed", "arguments": {"action": "mouse_click", "expected_window": "Zelyr Desktop Test Fixture"}, "error": "FOCUS_LOST: not focused"},
        {"call_id": "c3", "tool_name": "manage_desktop_window", "state": "success", "arguments": {"action": "focus", "target": "Zelyr Desktop Test Fixture"}, "error": None},
        {"call_id": "c4", "tool_name": "desktop_control", "state": "success", "arguments": {"action": "mouse_click", "expected_window": "Zelyr Desktop Test Fixture"}, "error": None},
    ]
    good_run = {
        "returncode": 0,
        "cli_json": {
            "execution_ledger": {
                "has_unresolved_failures": False,
                "unresolved_failures": [],
                "failed_actions": [good_records[1]],
                "successful_actions": [good_records[0], good_records[2], good_records[3]],
                "records": good_records,
            }
        },
        "ledger": {
            "status": "completed_unverified",
            "verification_status": "not_verified",
            "error": None,
            "calls": [{"event_type": "tool_execution_result", "tool_name": "desktop_control", "status": "failed", "error": "FOCUS_LOST: not focused"}],
        },
    }
    good_after = {"counter": 1, "applied_text": "focus-recovery-ok", "input_text": "focus-recovery-ok", "key_presses": 18, "is_active_window": True, "focus_out_count": 1}
    dist_clean = {"mouse_clicks": 0, "key_presses": 0}

    ev_pass = bench.evaluate_focus_recovery_run(
        good_run,
        before_state={"counter": 0},
        after_state=good_after,
        dist_after_state=dist_clean,
        steal_telemetry=base_steal,
    )
    assert ev_pass["verdict"] == "PASS"

    # Proactive focus bypass (no FOCUS_LOST encountered) MUST NOT count as PASS -> UNVERIFIED
    proactive_records = [
        good_records[0],
        good_records[2],  # focused before mutating click!
        good_records[3],
    ]
    proactive_run = {
        "returncode": 0,
        "cli_json": {
            "execution_ledger": {
                "has_unresolved_failures": False,
                "unresolved_failures": [],
                "failed_actions": [],
                "successful_actions": proactive_records,
                "records": proactive_records,
            }
        },
        "ledger": {"status": "completed", "verification_status": "verified", "error": None, "calls": [{"event_type": "tool_execution_result", "tool_name": "desktop_control", "status": "succeeded"}]},
    }
    ev_proactive = bench.evaluate_focus_recovery_run(
        proactive_run,
        before_state={"counter": 0},
        after_state=good_after,
        dist_after_state=dist_clean,
        steal_telemetry=base_steal,
    )
    assert ev_proactive["verdict"] == "UNVERIFIED"
    assert ev_proactive["passed"] is False

    # Distraction click leakage during failure -> FAIL
    ev_leak = bench.evaluate_focus_recovery_run(
        good_run,
        before_state={"counter": 0},
        after_state=good_after,
        dist_after_state={"mouse_clicks": 1, "key_presses": 0},
        steal_telemetry=base_steal,
    )
    assert ev_leak["verdict"] == "FAIL"

    # Verify _cleanup_fixture_processes tears down processes even when wait times out
    quit_calls = []
    monkeypatch.setattr(bench, "_http_json", lambda url, **_kw: quit_calls.append(url) or {"ok": True})
    fake_p1 = MagicMock()
    fake_p1.wait.side_effect = [subprocess.TimeoutExpired("x", 3.0), 0]
    fake_p2 = MagicMock()
    fake_p2.wait.return_value = 0
    bench._cleanup_fixture_processes((fake_p1, 9001), (fake_p2, 9002))
    assert len(quit_calls) == 2
    fake_p1.terminate.assert_called_once()
    fake_p1.kill.assert_called_once()
    fake_p2.terminate.assert_called_once()

    # Verify run_focus_recovery_benchmark cleans up fixtures and restores workspace after mid-run failure
    dispatched = []
    cleaned_procs = []
    monkeypatch.setattr(bench, "_hyprctl_json", lambda cmd, *_a: [{"name": "DP-1", "x": 0, "y": 0, "width": 1920, "height": 1080, "scale": 1.0, "focused": True, "activeWorkspace": {"id": 11}}] if cmd == "monitors" else ({"id": 11} if cmd == "activeworkspace" else ({"address": "0x999"} if cmd == "activewindow" else [])))
    monkeypatch.setattr(bench, "_hyprctl_dispatch", lambda c, a="": dispatched.append((c, a)))
    monkeypatch.setattr(bench, "_wait_for_http", lambda *_a, **_kw: {"ok": True})
    monkeypatch.setattr(bench, "_wait_for_window", lambda title, **_kw: {"address": "0x111" if "Test Fixture" in title else "0x222"})
    monkeypatch.setattr(bench, "_place_floating_window", lambda *_a, **_kw: None)
    monkeypatch.setattr(bench, "_focus_and_verify_window", lambda *_a, **_kw: {"active_window": {"address": "0x111", "title": "Zelyr Desktop Test Fixture"}, "verified": True})

    fake_procs = []

    def _fake_popen(*_a, **_kw):
        pr = MagicMock(pid=1000 + len(fake_procs))
        fake_procs.append(pr)
        return pr

    monkeypatch.setattr(bench.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(bench, "_cleanup_fixture_processes", lambda *items: cleaned_procs.extend([p for p, _ in items if p is not None]))
    monkeypatch.setattr(bench, "_run_zelyr_cli", MagicMock(side_effect=RuntimeError("boom mid-run")))

    with pytest.raises(RuntimeError, match="boom mid-run"):
        bench.run_focus_recovery_benchmark(runs=1, allow_live_desktop=True)

    assert len(cleaned_procs) == 2
    assert ("workspace", "11") in dispatched
    assert ("focuswindow", "address:0x999") in dispatched


@pytest.mark.asyncio
async def test_focus_recovery_preserves_vision_across_failure_and_refocus_without_history_bloat():
    """Transient vision survives FOCUS_LOST -> manage_desktop_window(focus) -> retry, then clears on success."""
    from zelyr.tools.core import ToolResult

    orch = NativeOrchestrator()
    orch.max_steps = 8
    ledger = TurnExecutionLedger()
    seen_images_per_turn = []
    call_idx = 0

    async def mock_stream(payload: dict):
        nonlocal call_idx
        call_idx += 1
        seen_images_per_turn.append(payload.get("_pending_vision_b64"))
        if call_idx == 1:
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "c_obs",
                            "type": "function",
                            "function": {
                                "name": "capture_screen",
                                "arguments": json.dumps({"target": "window", "window": "TestWin"}),
                            },
                        }]
                    }
                }]
            }
        elif call_idx == 2:
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "c_fail",
                            "type": "function",
                            "function": {
                                "name": "desktop_control",
                                "arguments": json.dumps({"action": "mouse_click", "x": 100, "y": 200, "expected_window": "TestWin"}),
                            },
                        }]
                    }
                }]
            }
        elif call_idx == 3:
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "c_focus",
                            "type": "function",
                            "function": {
                                "name": "manage_desktop_window",
                                "arguments": json.dumps({"action": "focus", "target": "TestWin", "verify": True}),
                            },
                        }]
                    }
                }]
            }
        elif call_idx == 4:
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "c_retry",
                            "type": "function",
                            "function": {
                                "name": "desktop_control",
                                "arguments": json.dumps({"action": "mouse_click", "x": 100, "y": 200, "expected_window": "TestWin"}),
                            },
                        }]
                    }
                }]
            }
        elif call_idx == 5:
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "c_post_obs",
                            "type": "function",
                            "function": {
                                "name": "capture_screen",
                                "arguments": json.dumps({"target": "window", "window": "TestWin"}),
                            },
                        }]
                    }
                }]
            }
        else:
            yield {"choices": [{"delta": {"content": "Completed and verified."}}]}

    async def mock_executor(name: str, args: dict):
        if name == "capture_screen":
            return ToolResult(
                True,
                output={"status": "success"},
                metadata={"_vision_b64": "FRAME_BASE64_1" if call_idx == 1 else "FRAME_BASE64_2"},
            )
        if name == "desktop_control":
            if call_idx == 2:
                return ToolResult(
                    False,
                    error="FOCUS_LOST: Target window 'TestWin' is not focused",
                    output={"success": False, "error_type": "FOCUS_LOST", "error": "FOCUS_LOST: Target window 'TestWin' is not focused"},
                )
            return ToolResult(True, output={"status": "success", "action": args.get("action")})
        if name == "manage_desktop_window":
            return ToolResult(True, output={"status": "success", "verified": True, "action": "focus"})
        raise AssertionError(f"Unexpected tool {name}")

    payload = {
        "model": "qwen3.6:35b-a3b",
        "_force_vision_image": True,
        "messages": [{"role": "user", "content": "Capture TestWin, click button, and capture TestWin again to verify."}],
    }
    schemas = [
        {"type": "function", "function": {"name": "capture_screen", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "desktop_control", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "manage_desktop_window", "parameters": {"type": "object"}}},
    ]

    with patch.object(orch, "_model_supports_vision", return_value=True):
        chunks = [
            c
            async for c in orch.generate_stream(
                payload,
                stream_provider=mock_stream,
                tool_schemas=schemas,
                tool_executor=mock_executor,
                ledger=ledger,
            )
        ]

    assert chunks
    assert "_pending_vision_b64" not in payload
    assert all("images" not in m for m in payload["messages"] if isinstance(m, dict))
    assert seen_images_per_turn == [
        None,              # before initial capture
        "FRAME_BASE64_1",  # initial click attempt (receives FOCUS_LOST)
        "FRAME_BASE64_1",  # manage_desktop_window(focus) keeps frame
        "FRAME_BASE64_1",  # retry click still sees frame
        None,              # cleared immediately after mutating click succeeded
        "FRAME_BASE64_2",  # fresh frame from post-mutation capture
    ]


@pytest.mark.asyncio
async def test_desktop_trailing_unobserved_mutation_gating_and_outcome_resolution():
    """Trailing desktop_control click after capture_screen triggers replanning and cannot claim VERIFIED."""
    from PIL import Image
    from zelyr.engine.execution_ledger import (
        ExecutionStatus,
        TurnExecutionLedger,
        VerificationStatus,
        find_unfulfilled_task_requirements,
        resolve_root_execution_outcome,
    )
    from zelyr.tools.core import ToolResult
    from zelyr.tools.screen_perception import _overlay_coordinate_grid

    # Grid helper handles narrow windows symmetrically without raising
    sample_img = Image.new("RGB", (640, 420), (15, 23, 42))
    gridded = _overlay_coordinate_grid(sample_img, {"x": 640, "y": 1410, "width": 640, "height": 420})
    assert gridded.size == (640, 420)

    # Ledger unit invariant: Turn 2 focus verify=True cannot mark later trailing click as VERIFIED
    l_stale = TurnExecutionLedger()
    l_stale.record_proposed("c0", "capture_screen", {"target": "window", "window": "TestWin"})
    l_stale.record_dispatched("c0")
    l_stale.record_observation("c0", status="success", output="ok")
    l_stale.record_proposed("c1", "manage_desktop_window", {"action": "focus", "target": "TestWin", "verify": True})
    l_stale.record_dispatched("c1")
    l_stale.record_observation("c1", status="success", output="ok", verified=True)
    l_stale.record_proposed("c2", "desktop_control", {"action": "mouse_click", "x": 960, "y": 1670})
    l_stale.record_dispatched("c2")
    l_stale.record_observation("c2", status="success", output="clicked")

    _, unfulfilled = find_unfulfilled_task_requirements(
        "Capture TestWin, click field, type 'abc', and capture window again to verify.",
        l_stale,
    )
    assert any("Trailing desktop GUI action" in msg for msg in unfulfilled)

    outcome_unobs = resolve_root_execution_outcome(
        ledger=l_stale,
        final_text="All interactions completed successfully.",
        prompt="Capture TestWin and verify.",
    )
    assert outcome_unobs.verification_status == VerificationStatus.CLAIMED_UNVERIFIED
    assert outcome_unobs.execution_status == ExecutionStatus.COMPLETED_UNVERIFIED

    # End-to-end orchestrator Stage 4 replanning catches premature completion after trailing click
    orch = NativeOrchestrator()
    orch.max_steps = 8
    ledger_e2e = TurnExecutionLedger()
    turn_no = 0

    async def mock_stream_e2e(payload: dict):
        nonlocal turn_no
        turn_no += 1
        if turn_no == 1:
            yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "t1", "type": "function", "function": {"name": "capture_screen", "arguments": json.dumps({"target": "window", "window": "TestWin"})}}]}}]}
        elif turn_no == 2:
            yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "t2", "type": "function", "function": {"name": "desktop_control", "arguments": json.dumps({"action": "mouse_click", "x": 960, "y": 1670})}}]}}]}
        elif turn_no == 3:
            # Model tries to claim completion right after trailing click!
            yield {"choices": [{"delta": {"content": "I have completed all interactions."}}]}
        elif turn_no == 4:
            # After Stage 4 directive, model types and then re-captures
            yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "t4", "type": "function", "function": {"name": "desktop_control", "arguments": json.dumps({"action": "keyboard_type", "text": "abc", "press_enter": True})}}]}}]}
        elif turn_no == 5:
            yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "t5", "type": "function", "function": {"name": "capture_screen", "arguments": json.dumps({"target": "window", "window": "TestWin"})}}]}}]}
        else:
            yield {"choices": [{"delta": {"content": "Typed abc and verified via final capture_screen."}}]}

    async def mock_exec_e2e(name: str, args: dict):
        return ToolResult(True, output={"status": "success", "action": args.get("action")})

    payload_e2e = {
        "model": "qwen3.6:35b-a3b",
        "messages": [{"role": "user", "content": "Capture TestWin, click input field, type 'abc', then capture window again to verify."}],
    }
    schemas_e2e = [
        {"type": "function", "function": {"name": "capture_screen", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "desktop_control", "parameters": {"type": "object"}}},
    ]
    out_chunks = [
        c
        async for c in orch.generate_stream(
            payload_e2e,
            stream_provider=mock_stream_e2e,
            tool_schemas=schemas_e2e,
            tool_executor=mock_exec_e2e,
            ledger=ledger_e2e,
        )
    ]
    final_txt = "".join(
        c.get("choices", [{}])[0].get("delta", {}).get("content", "")
        for c in out_chunks
    )
    assert "Typed abc and verified" in final_txt
    assert [
        ledger_e2e._records[cid].tool_name
        for cid in ledger_e2e._order
        if cid in ledger_e2e._records
    ] == ["capture_screen", "desktop_control", "desktop_control", "capture_screen"]
    _, unfulfilled_after = find_unfulfilled_task_requirements(
        payload_e2e["messages"][1]["content"] if len(payload_e2e["messages"]) > 1 else "",
        ledger_e2e,
    )
    assert unfulfilled_after == []


@pytest.mark.asyncio
async def test_consecutive_desktop_control_mutation_after_focus_lost_requires_refocus():
    """Retrying desktop_control with jittered coordinates right after FOCUS_LOST is blocked until manage_desktop_window refocuses."""
    from zelyr.tools.core import ToolResult

    orch = NativeOrchestrator()
    orch.max_steps = 8
    orch.enable_cycle_detection = True
    ledger = TurnExecutionLedger()
    executed = []
    turn_no = 0

    async def mock_stream(payload: dict):
        nonlocal turn_no
        turn_no += 1
        if turn_no == 1:
            yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "type": "function", "function": {"name": "desktop_control", "arguments": json.dumps({"action": "mouse_click", "x": 960, "y": 1650, "expected_window": "Fix"})}}]}}]}
        elif turn_no == 2:
            # Tweaked coords without refocusing first -> must be blocked pre-dispatch!
            yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c2", "type": "function", "function": {"name": "desktop_control", "arguments": json.dumps({"action": "mouse_click", "x": 962, "y": 1680, "expected_window": "Fix"})}}]}}]}
        elif turn_no == 3:
            yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c3", "type": "function", "function": {"name": "manage_desktop_window", "arguments": json.dumps({"action": "focus", "target": "Fix", "verify": True})}}]}}]}
        elif turn_no == 4:
            yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c4", "type": "function", "function": {"name": "desktop_control", "arguments": json.dumps({"action": "mouse_click", "x": 960, "y": 1660, "expected_window": "Fix"})}}]}}]}
        else:
            yield {"choices": [{"delta": {"content": "Refocused and retried cleanly."}}]}

    async def mock_exec(name: str, args: dict):
        executed.append((name, args.get("action"), args.get("x"), args.get("y")))
        if name == "desktop_control" and len(executed) == 1:
            return ToolResult(False, error="FOCUS_LOST: Target window 'Fix' is not focused", output={"success": False, "error_type": "FOCUS_LOST"})
        return ToolResult(True, output={"status": "success", "verified": True if name == "manage_desktop_window" else False})

    schemas = [
        {"type": "function", "function": {"name": "desktop_control", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "manage_desktop_window", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "capture_screen", "parameters": {"type": "object"}}},
    ]
    payload = {"model": "qwen3.6:35b-a3b", "messages": [{"role": "user", "content": "Click button in Fix."}]}
    async for _ in orch.generate_stream(payload, stream_provider=mock_stream, tool_schemas=schemas, tool_executor=mock_exec, ledger=ledger):
        pass

    assert executed == [
        ("desktop_control", "mouse_click", 960, 1650),
        ("manage_desktop_window", "focus", None, None),
        ("desktop_control", "mouse_click", 960, 1660),
    ]
    assert ledger._records["c2"].rejection_reason == "cycle_detected"
    assert "manage_desktop_window" in str(ledger._records["c2"].error)
