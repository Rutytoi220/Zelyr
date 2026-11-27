"""Comprehensive offline tests for read-only Linux AT-SPI2 desktop accessibility verification."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from scripts.cross_surface_benchmark import (
    INITIAL_RECORDS,
    build_cross_surface_verify_spec,
    evaluate_cross_surface_run,
    setup_cross_surface_workspace,
)
from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.cli.main import _execute_headless_run
from zelyr.engine.desktop_accessibility import (
    ATSPI_STATE_BITS,
    AtspiDbusAccessibilityBackend,
    MAX_INSPECTED_DESCENDANTS,
    MAX_RETURNED_TEXT_LEN,
    decode_atspi_states,
    is_sensitive_widget,
    parse_desktop_selector,
    split_window_and_widget_target,
)
from zelyr.engine.execution_ledger import (
    ActionState,
    CentralExecutionLedger,
    ExecutionStatus,
    TurnExecutionLedger,
    VerificationStatus,
    criterion_canonical_target_raw,
    parse_acceptance_criteria,
    resolve_root_execution_outcome,
)
from zelyr.engine.verifier import (
    OutcomeVerifier,
    VERDICT_FAIL,
    VERDICT_PASS,
    VERDICT_UNSUPPORTED,
    VERDICT_UNVERIFIED,
)
from zelyr.tools.verify_outcome import VerifyOutcomeTool


# ── Fake D-Bus / AT-SPI2 Test Infrastructure ─────────────────────────────────


def _encode_states(*names: str) -> List[int]:
    w0 = 0
    w1 = 0
    for name in names:
        bit = ATSPI_STATE_BITS[name]
        if bit < 32:
            w0 |= 1 << bit
        else:
            w1 |= 1 << (bit - 32)
    return [w0, w1]


@dataclass
class _FakeNode:
    bus: str
    path: str
    role: str
    name: str = ""
    description: str = ""
    accessible_id: str = ""
    states: Tuple[str, ...] = ("enabled", "sensitive", "visible", "showing")
    attributes: Dict[str, str] = field(default_factory=dict)
    has_text: bool = False
    text: Optional[str] = None
    children: List["_FakeNode"] = field(default_factory=list)
    fail_on: Optional[str] = None


class _FakeReply:
    def __init__(self, body: Tuple[Any, ...]) -> None:
        self.body = body


class _FakeDBusAddress:
    def __init__(self, object_path: str, bus_name: Optional[str] = None, interface: Optional[str] = None) -> None:
        self.object_path = object_path
        self.bus_name = bus_name
        self.interface = interface


def _fake_new_method_call(addr: _FakeDBusAddress, method: str, signature: str = "", body: Tuple[Any, ...] = ()) -> Dict[str, Any]:
    return {
        "bus": addr.bus_name or "",
        "path": addr.object_path,
        "iface": addr.interface or "",
        "method": method,
        "sig": signature,
        "args": body,
    }


class _FakeAtspiRegistryWorld:
    """In-memory AT-SPI2 registry tree backing fake jeepney D-Bus connections."""

    def __init__(self, apps: List[Tuple[int, _FakeNode]], *, bus_addr: str = "unix:path=/tmp/fake-atspi") -> None:
        self.bus_addr = bus_addr
        self.apps = apps
        self.calls: List[Dict[str, Any]] = []
        self._by_bus_path: Dict[Tuple[str, str], _FakeNode] = {}
        self._pid_by_bus: Dict[str, int] = {}
        for pid, app_node in apps:
            self._pid_by_bus[app_node.bus] = pid
            self._index_node(app_node)

    def _index_node(self, node: _FakeNode) -> None:
        self._by_bus_path[(node.bus, node.path)] = node
        for ch in node.children:
            self._index_node(ch)

    def open_dbus_connection(self, bus: str = "SESSION"):
        world = self

        class _Conn:
            def __enter__(self):
                return self

            def __exit__(self, _et, _ev, _tb):
                return False

            def send_and_get_reply(self, msg: Dict[str, Any], timeout: float = 3.0) -> _FakeReply:
                world.calls.append({"bus_mode": bus, **msg})
                m_bus = msg["bus"]
                m_path = msg["path"]
                m_iface = msg["iface"]
                method = msg["method"]
                args = msg["args"]

                if bus == "SESSION" and m_bus == "org.a11y.Bus" and method == "GetAddress":
                    return _FakeReply((world.bus_addr,))

                if (
                    m_bus in ("org.a11y.atspi.Registry", "org.a11y.atspi.Accessible")
                    and m_path == "/org/a11y/atspi/accessible/root"
                    and method == "GetChildren"
                ):
                    return _FakeReply(([
                        (app.bus, app.path) for _, app in world.apps
                    ],))

                if m_bus == "org.freedesktop.DBus" and method == "GetConnectionUnixProcessID":
                    target_bus = str(args[0])
                    if target_bus not in world._pid_by_bus:
                        raise RuntimeError(f"Unknown bus {target_bus}")
                    return _FakeReply((world._pid_by_bus[target_bus],))

                node = world._by_bus_path.get((m_bus, m_path))
                if node is None:
                    raise RuntimeError(f"Unknown object {m_bus}{m_path}")

                if node.fail_on and node.fail_on == method:
                    raise RuntimeError(f"Injected AT-SPI2 failure on {method}")

                if m_iface == "org.freedesktop.DBus.Properties" and method == "Get":
                    prop_iface, prop_name = str(args[0]), str(args[1])
                    if prop_iface == "org.a11y.atspi.Accessible":
                        mapping = {
                            "Name": node.name,
                            "Description": node.description,
                            "AccessibleId": node.accessible_id,
                        }
                        if prop_name in mapping:
                            return _FakeReply((("s", mapping[prop_name]),))
                    elif prop_iface == "org.a11y.atspi.Text":
                        if not node.has_text:
                            raise RuntimeError("No Text interface")
                        if prop_name == "CharacterCount":
                            return _FakeReply((("i", len(node.text or "")),))
                    raise RuntimeError(f"Unsupported property {prop_iface}.{prop_name}")

                if m_iface == "org.a11y.atspi.Accessible":
                    if method == "GetChildren":
                        return _FakeReply(([(c.bus, c.path) for c in node.children],))
                    if method == "GetRoleName":
                        return _FakeReply((node.role,))
                    if method == "GetState":
                        return _FakeReply((_encode_states(*node.states),))
                    if method == "GetAttributes":
                        return _FakeReply((dict(node.attributes),))
                    if method == "GetInterfaces":
                        ifaces = ["org.a11y.atspi.Accessible"]
                        if node.has_text:
                            ifaces.append("org.a11y.atspi.Text")
                        return _FakeReply((ifaces,))

                if m_iface == "org.a11y.atspi.Text" and method == "GetText":
                    if not node.has_text:
                        raise RuntimeError("No Text interface")
                    if node.fail_on == "GetText":
                        raise RuntimeError("Injected GetText error")
                    start_i, end_i = int(args[0]), int(args[1])
                    return _FakeReply(((node.text or "")[start_i:end_i],))

                raise RuntimeError(f"Unhandled D-Bus call {m_iface}.{method}")

        return _Conn()


class _FakeHyprlandClient:
    def __init__(
        self,
        clients: List[Dict[str, Any]],
        *,
        active: Optional[Dict[str, Any]] = None,
        available: bool = True,
    ) -> None:
        self._clients = clients
        self._active = active or {}
        self._available = available
        self.calls: List[str] = []

    def can_connect(self) -> bool:
        return self._available

    async def get_clients(self) -> List[Dict[str, Any]]:
        self.calls.append("get_clients")
        return list(self._clients)

    async def get_active_window(self) -> Dict[str, Any]:
        self.calls.append("get_active_window")
        return dict(self._active)


def _wire_backend_with_world(monkeypatch, world: _FakeAtspiRegistryWorld) -> AtspiDbusAccessibilityBackend:
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/tmp/fake-session")
    monkeypatch.delenv("AT_SPI_BUS_ADDRESS", raising=False)
    backend = AtspiDbusAccessibilityBackend(timeout=1.0)
    monkeypatch.setattr(
        backend,
        "_import_jeepney",
        lambda: (_FakeDBusAddress, _fake_new_method_call, world.open_dbus_connection),
    )
    return backend


def _build_fixture_world() -> Tuple[_FakeAtspiRegistryWorld, List[Dict[str, Any]]]:
    target_bus = ":1.42"
    other_bus = ":1.99"

    lbl = _FakeNode(
        bus=target_bus,
        path="/org/a11y/atspi/accessible/10",
        role="label",
        name="Status: CONFIRMED:REC-204",
        description="Live fixture status indicator",
        accessible_id="QApplication.FixtureWindow.statusLabel",
        states=("enabled", "sensitive", "visible", "showing"),
    )
    inp = _FakeNode(
        bus=target_bus,
        path="/org/a11y/atspi/accessible/11",
        role="text",
        name="Confirmation Payload Input",
        accessible_id="QApplication.FixtureWindow.inputField",
        has_text=True,
        text="CONFIRMED:REC-204",
        states=("enabled", "sensitive", "editable", "focusable", "focused", "visible", "showing"),
    )
    empty_inp = _FakeNode(
        bus=target_bus,
        path="/org/a11y/atspi/accessible/12",
        role="text",
        name="Secondary Notes Input",
        accessible_id="QApplication.FixtureWindow.notesField",
        has_text=True,
        text="",
        states=("enabled", "sensitive", "editable", "focusable", "visible", "showing"),
    )
    apply_btn = _FakeNode(
        bus=target_bus,
        path="/org/a11y/atspi/accessible/13",
        role="push button",
        name="Confirm",
        description="Confirm text input into status indicator",
        accessible_id="QApplication.FixtureWindow.applyBtn",
        states=("enabled", "sensitive", "focusable", "visible", "showing"),
    )
    chk = _FakeNode(
        bus=target_bus,
        path="/org/a11y/atspi/accessible/14",
        role="check box",
        name="Auto-sync",
        accessible_id="QApplication.FixtureWindow.autoSyncCheck",
        states=("enabled", "sensitive", "checkable", "checked", "focusable", "visible", "showing"),
    )
    pwd = _FakeNode(
        bus=target_bus,
        path="/org/a11y/atspi/accessible/15",
        role="password text",
        name="API Secret Key",
        accessible_id="QApplication.FixtureWindow.secretInput",
        has_text=True,
        text="super-secret-value",
        attributes={"input-purpose": "password"},
        states=("enabled", "sensitive", "editable", "focusable", "visible", "showing"),
    )
    target_win = _FakeNode(
        bus=target_bus,
        path="/org/a11y/atspi/accessible/2",
        role="frame",
        name="Zelyr Cross-Surface Confirmation",
        accessible_id="QApplication.FixtureWindow",
        children=[lbl, inp, empty_inp, apply_btn, chk, pwd],
    )
    target_app = _FakeNode(
        bus=target_bus,
        path="/org/a11y/atspi/accessible/root_app",
        role="application",
        name="desktop_test_fixture.py",
        children=[target_win],
    )

    unrelated_secret = _FakeNode(
        bus=other_bus,
        path="/org/a11y/atspi/accessible/901",
        role="label",
        name="Private Notes Secret Token",
        accessible_id="privateLabel",
        states=("enabled", "sensitive", "visible", "showing"),
    )
    unrelated_win = _FakeNode(
        bus=other_bus,
        path="/org/a11y/atspi/accessible/900",
        role="frame",
        name="Keepass Private Vault",
        children=[unrelated_secret],
    )
    unrelated_app = _FakeNode(
        bus=other_bus,
        path="/org/a11y/atspi/accessible/root_other",
        role="application",
        name="keepassxc",
        children=[unrelated_win],
    )

    clients = [
        {
            "address": "0xaa001",
            "title": "Zelyr Cross-Surface Confirmation",
            "initialTitle": "Zelyr Cross-Surface Confirmation",
            "class": "desktop_test_fixture.py",
            "pid": 4242,
            "workspace": {"id": 15, "name": "15"},
        },
        {
            "address": "0xbb002",
            "title": "Keepass Private Vault",
            "initialTitle": "Keepass Private Vault",
            "class": "keepassxc",
            "pid": 9999,
            "workspace": {"id": 2, "name": "2"},
        },
    ]
    world = _FakeAtspiRegistryWorld([(9999, unrelated_app), (4242, target_app)])
    return world, clients


# ── 1 & 2. Backend Availability / Unavailability ──────────────────────────────


@pytest.mark.asyncio
async def test_backend_available_and_5field_compat(monkeypatch) -> None:
    world, clients = _build_fixture_world()
    backend = _wire_backend_with_world(monkeypatch, world)
    hypr = _FakeHyprlandClient(clients)

    res = await OutcomeVerifier.verify_async(
        "desktop_visible_text",
        "Zelyr Cross-Surface Confirmation",
        selector="#statusLabel",
        expected="CONFIRMED:REC-204",
        desktop_accessibility_backend=backend,
        hyprland_client=hypr,
    )
    assert res.success is True
    assert res.result == VERDICT_PASS
    assert res.evidence_collected is True
    assert res.evidence_source == "desktop_accessibility"
    assert res.observed["accessibility_backend"] == "atspi2_dbus_jeepney"

    as_dict = res.to_dict()
    assert set(as_dict.keys()) == {"success", "check", "target", "observed", "error"}
    assert as_dict["success"] is True


@pytest.mark.asyncio
async def test_backend_unavailable_paths_return_unsupported(monkeypatch) -> None:
    world, clients = _build_fixture_world()
    hypr = _FakeHyprlandClient(clients)

    # Case A: Missing DBUS_SESSION_BUS_ADDRESS and AT_SPI_BUS_ADDRESS
    monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
    monkeypatch.delenv("AT_SPI_BUS_ADDRESS", raising=False)
    backend_no_env = AtspiDbusAccessibilityBackend()
    monkeypatch.setattr(backend_no_env, "_import_jeepney", lambda: (_FakeDBusAddress, _fake_new_method_call, world.open_dbus_connection))
    res_env = await OutcomeVerifier.verify_async(
        "desktop_visible_text",
        "Zelyr Cross-Surface Confirmation",
        selector="#statusLabel",
        expected="CONFIRMED:REC-204",
        desktop_accessibility_backend=backend_no_env,
        hyprland_client=hypr,
    )
    assert res_env.result == VERDICT_UNSUPPORTED
    assert res_env.evidence_collected is False

    # Case B: Missing jeepney binding
    monkeypatch.setenv("AT_SPI_BUS_ADDRESS", "unix:path=/tmp/fake")
    backend_no_jeep = AtspiDbusAccessibilityBackend()
    from zelyr.engine.desktop_accessibility import AccessibilityBackendUnavailable

    monkeypatch.setattr(
        backend_no_jeep,
        "_import_jeepney",
        lambda: (_ for _ in ()).throw(AccessibilityBackendUnavailable("jeepney missing")),
    )
    res_imp = await OutcomeVerifier.verify_async(
        "desktop_widget_state",
        "Zelyr Cross-Surface Confirmation",
        selector="#applyBtn",
        expected="enabled",
        desktop_accessibility_backend=backend_no_jeep,
        hyprland_client=hypr,
    )
    assert res_imp.result == VERDICT_UNSUPPORTED
    assert res_imp.evidence_collected is False

    # Case C: Unreachable D-Bus socket
    monkeypatch.setenv("AT_SPI_BUS_ADDRESS", "unix:path=/tmp/nonexistent-bus")
    backend_down = AtspiDbusAccessibilityBackend()

    def _boom_open(bus: str = "SESSION"):
        raise OSError("Connection refused")

    monkeypatch.setattr(backend_down, "_import_jeepney", lambda: (_FakeDBusAddress, _fake_new_method_call, _boom_open))
    res_down = await OutcomeVerifier.verify_async(
        "desktop_visible_text",
        "Zelyr Cross-Surface Confirmation",
        selector="#statusLabel",
        expected="CONFIRMED:REC-204",
        desktop_accessibility_backend=backend_down,
        hyprland_client=hypr,
    )
    assert res_down.result == VERDICT_UNSUPPORTED
    assert res_down.evidence_collected is False


# ── 3, 4 & 5. Window Resolution, Ambiguity Rejection & Read-Only Scoping ──────


@pytest.mark.asyncio
async def test_pid_scoping_never_inspects_unrelated_app_or_injects_input(monkeypatch) -> None:
    world, clients = _build_fixture_world()
    backend = _wire_backend_with_world(monkeypatch, world)
    hypr = _FakeHyprlandClient(clients)

    res = await OutcomeVerifier.verify_async(
        "desktop_visible_text",
        "Zelyr Cross-Surface Confirmation",
        selector="#statusLabel",
        expected="CONFIRMED:REC-204",
        desktop_accessibility_backend=backend,
        hyprland_client=hypr,
    )
    assert res.result == VERDICT_PASS
    # Verify NO D-Bus call ever accessed unrelated app's bus (:1.99) beyond root GetChildren / PID check!
    unrelated_obj_calls = [c for c in world.calls if c["bus"] == ":1.99"]
    assert unrelated_obj_calls == []
    # Verify ONLY read-only Hyprland and AT-SPI2 methods ran
    assert hypr.calls == ["get_clients"]
    forbidden_methods = {"DoAction", "GrabFocus", "SetTextContents", "dispatch", "focuswindow"}
    assert all(c["method"] not in forbidden_methods for c in world.calls)


@pytest.mark.asyncio
async def test_fallback_top_level_window_match_without_compositor(monkeypatch) -> None:
    world, _ = _build_fixture_world()
    backend = _wire_backend_with_world(monkeypatch, world)
    hypr_down = _FakeHyprlandClient([], available=False)

    res = await OutcomeVerifier.verify_async(
        "desktop_visible_text",
        "Zelyr Cross-Surface Confirmation",
        selector="#statusLabel",
        expected="CONFIRMED:REC-204",
        desktop_accessibility_backend=backend,
        hyprland_client=hypr_down,
    )
    assert res.result == VERDICT_PASS
    assert res.observed["window_pid"] == 4242
    # Top-level scan inspected keepass top-level frame Name/AccessibleId ONLY, NEVER its private child node 901!
    unrelated_child_calls = [
        c for c in world.calls if c["bus"] == ":1.99" and "/901" in str(c["path"])
    ]
    assert unrelated_child_calls == []


@pytest.mark.asyncio
async def test_ambiguous_windows_and_ambiguous_widgets_return_unverified(monkeypatch) -> None:
    world, clients = _build_fixture_world()
    backend = _wire_backend_with_world(monkeypatch, world)

    # 1. Ambiguous Hyprland windows (two windows share substring match and neither is exact)
    dup_clients = list(clients) + [{
        "address": "0xaa099",
        "title": "Zelyr Cross-Surface Confirmation Copy",
        "initialTitle": "Zelyr Cross-Surface Confirmation Copy",
        "class": "desktop_test_fixture.py",
        "pid": 4243,
    }]
    res_amb_win = await OutcomeVerifier.verify_async(
        "desktop_visible_text",
        "Zelyr Cross-Surface",
        selector="#statusLabel",
        expected="CONFIRMED",
        desktop_accessibility_backend=backend,
        hyprland_client=_FakeHyprlandClient(dup_clients),
    )
    assert res_amb_win.result == VERDICT_UNVERIFIED
    assert res_amb_win.observed["ambiguous"] is True
    assert res_amb_win.evidence_collected is False
    # Zero AT-SPI2 GetRoleName calls because ambiguity rejected before tree walk
    assert not any(c["method"] == "GetRoleName" for c in world.calls)

    # 2. Ambiguous widget selector inside one window (2 text fields match role=text)
    hypr = _FakeHyprlandClient(clients)
    res_amb_widget = await OutcomeVerifier.verify_async(
        "desktop_widget_state",
        "Zelyr Cross-Surface Confirmation",
        selector="role=text",
        expected="editable",
        desktop_accessibility_backend=backend,
        hyprland_client=hypr,
    )
    assert res_amb_widget.result == VERDICT_UNVERIFIED
    assert res_amb_widget.observed["ambiguous"] is True
    assert res_amb_widget.evidence_collected is False

    # 3. Disambiguated by explicit index=0
    res_idx = await OutcomeVerifier.verify_async(
        "desktop_visible_text",
        "Zelyr Cross-Surface Confirmation",
        selector="role=text, index=0",
        expected="CONFIRMED:REC-204",
        desktop_accessibility_backend=backend,
        hyprland_client=hypr,
    )
    assert res_idx.result == VERDICT_PASS


# ── 6 & 7. Text Interface vs Name, Hidden Fail, State Bits & Checkable ────────


@pytest.mark.asyncio
async def test_visible_text_exact_mismatch_hidden_and_empty_text_interface(monkeypatch) -> None:
    world, clients = _build_fixture_world()
    # Add a hidden label for visibility check
    hidden_lbl = _FakeNode(
        bus=":1.42",
        path="/org/a11y/atspi/accessible/19",
        role="label",
        name="CONFIRMED:SECRET_HIDDEN",
        accessible_id="hiddenLabel",
        states=("enabled", "sensitive"),
    )
    world.apps[1][1].children[0].children.append(hidden_lbl)
    world._index_node(world.apps[1][1])
    backend = _wire_backend_with_world(monkeypatch, world)
    hypr = _FakeHyprlandClient(clients)

    # Exact match passes and wrong text fails honestly
    ok = await OutcomeVerifier.verify_async(
        "desktop_visible_text",
        "Zelyr Cross-Surface Confirmation",
        selector="#statusLabel",
        expected="Status: CONFIRMED:REC-204",
        match="exact",
        desktop_accessibility_backend=backend,
        hyprland_client=hypr,
    )
    assert ok.result == VERDICT_PASS

    wrong = await OutcomeVerifier.verify_async(
        "desktop_text",
        "Zelyr Cross-Surface Confirmation",
        selector="#statusLabel",
        expected="WRONG-VALUE",
        desktop_accessibility_backend=backend,
        hyprland_client=hypr,
    )
    assert wrong.result == VERDICT_FAIL
    assert wrong.evidence_collected is True

    # Hidden widget MUST fail even if accessible Name contains expected string
    hidden = await OutcomeVerifier.verify_async(
        "desktop_status",
        "Zelyr Cross-Surface Confirmation",
        selector="#hiddenLabel",
        expected="CONFIRMED",
        desktop_accessibility_backend=backend,
        hyprland_client=hypr,
    )
    assert hidden.result == VERDICT_FAIL
    assert "hidden" in (hidden.error or "").lower()

    # Empty Text-interface field (QLineEdit) must NOT fall back to static accessible Name
    empty_res = await OutcomeVerifier.verify_async(
        "desktop_visible_text",
        "Zelyr Cross-Surface Confirmation",
        selector="#notesField",
        expected="Secondary Notes Input",
        desktop_accessibility_backend=backend,
        hyprland_client=hypr,
    )
    assert empty_res.result == VERDICT_FAIL


@pytest.mark.asyncio
async def test_desktop_widget_state_decoding_and_checkable_validation(monkeypatch) -> None:
    world, clients = _build_fixture_world()
    backend = _wire_backend_with_world(monkeypatch, world)
    hypr = _FakeHyprlandClient(clients)

    dec = decode_atspi_states(_encode_states("enabled", "sensitive", "editable", "focused", "visible", "showing"))
    assert dec["enabled"] is True and dec["disabled"] is False
    assert dec["editable"] is True and dec["focused"] is True and dec["visible"] is True

    # Button disabled if enabled=True but sensitive=False
    dec_insens = decode_atspi_states(_encode_states("enabled", "visible", "showing"))
    assert dec_insens["enabled"] is False and dec_insens["disabled"] is True

    # Dict expectation on focused editable field
    ok_inp = await OutcomeVerifier.verify_async(
        "desktop_ui_state",
        "Zelyr Cross-Surface Confirmation",
        selector="#inputField",
        expected={"enabled": True, "editable": True, "focused": True, "visible": True},
        desktop_accessibility_backend=backend,
        hyprland_client=hypr,
    )
    assert ok_inp.result == VERDICT_PASS

    # String expectation and multi-key dict expectation on checkbox
    ok_chk = await OutcomeVerifier.verify_async(
        "desktop_widget_state",
        "Zelyr Cross-Surface Confirmation",
        selector="#autoSyncCheck",
        expected="checked",
        desktop_accessibility_backend=backend,
        hyprland_client=hypr,
    )
    assert ok_chk.result == VERDICT_PASS
    ok_chk_dict = await OutcomeVerifier.verify_async(
        "desktop_widget_state",
        "Zelyr Cross-Surface Confirmation",
        selector="#autoSyncCheck",
        expected={"checked": True, "enabled": True},
        desktop_accessibility_backend=backend,
        hyprland_client=hypr,
    )
    assert ok_chk_dict.result == VERDICT_PASS

    # Non-checkable role (push button) rejected when expecting checked state
    bad_chk = await OutcomeVerifier.verify_async(
        "desktop_widget_state",
        "Zelyr Cross-Surface Confirmation",
        selector="#applyBtn",
        expected="checked",
        desktop_accessibility_backend=backend,
        hyprland_client=hypr,
    )
    assert bad_chk.result == VERDICT_FAIL
    assert "not checkable" in (bad_chk.error or "").lower()


# ── 8 & 9. Absent vs Inaccessible vs Sensitive Redaction ───────────────────────


@pytest.mark.asyncio
async def test_absent_widget_vs_inaccessible_vs_sensitive_exclusion(monkeypatch) -> None:
    world, clients = _build_fixture_world()
    backend = _wire_backend_with_world(monkeypatch, world)
    hypr = _FakeHyprlandClient(clients)

    # Absent widget in accessible window -> FAIL (evidence_collected=True)
    missing_w = await OutcomeVerifier.verify_async(
        "desktop_visible_text",
        "Zelyr Cross-Surface Confirmation",
        selector="#doesNotExist",
        expected="ANY",
        desktop_accessibility_backend=backend,
        hyprland_client=hypr,
    )
    assert missing_w.result == VERDICT_FAIL
    assert missing_w.evidence_collected is True
    assert missing_w.observed["widget_found"] is False

    # Absent window in compositor -> FAIL (evidence_collected=True)
    missing_win = await OutcomeVerifier.verify_async(
        "desktop_visible_text",
        "NoSuchWindowTitle",
        selector="#doesNotExist",
        expected="ANY",
        desktop_accessibility_backend=backend,
        hyprland_client=hypr,
    )
    assert missing_win.result == VERDICT_FAIL
    assert missing_win.observed["window_found"] is False

    # Non-accessible app existing in Hyprland (pid=7777 not on AT-SPI2 bus) -> UNVERIFIED
    no_a11y_clients = list(clients) + [{
        "address": "0xcc777",
        "title": "SDL Unexposed App",
        "initialTitle": "SDL Unexposed App",
        "class": "sdl_app",
        "pid": 7777,
    }]
    inacc = await OutcomeVerifier.verify_async(
        "desktop_visible_text",
        "SDL Unexposed App",
        selector="#statusLabel",
        expected="READY",
        desktop_accessibility_backend=backend,
        hyprland_client=_FakeHyprlandClient(no_a11y_clients),
    )
    assert inacc.result == VERDICT_UNVERIFIED
    assert inacc.evidence_collected is False
    assert inacc.observed["inaccessible"] is True

    # Sensitive/password field: MUST refuse and NEVER call GetText or leak payload
    world.calls.clear()
    sec_res = await OutcomeVerifier.verify_async(
        "desktop_visible_text",
        "Zelyr Cross-Surface Confirmation",
        selector="#secretInput",
        expected="super-secret-value",
        desktop_accessibility_backend=backend,
        hyprland_client=hypr,
    )
    assert sec_res.result == VERDICT_FAIL
    assert sec_res.evidence_collected is False
    assert sec_res.observed["safety_blocked"] is True
    assert not any(c["method"] == "GetText" and c["path"].endswith("/15") for c in world.calls)
    assert "super-secret-value" not in json.dumps(sec_res.to_dict())

    # Bullet-masked text in plain 'text' role (e.g. Qt EchoMode.Password) also blocked
    assert is_sensitive_widget(role="text", name="Pass", text_value="\u2022\u2022\u2022\u2022\u2022") is True


# ── 10. Bounded Traversal & Text Length Limits ────────────────────────────────


@pytest.mark.asyncio
async def test_descendant_and_text_length_bounds(monkeypatch) -> None:
    target_bus = ":1.55"
    big_txt = "X" * 5000
    long_lbl = _FakeNode(
        bus=target_bus,
        path="/org/a11y/atspi/accessible/long",
        role="text",
        name="Long Output",
        accessible_id="longField",
        has_text=True,
        text=big_txt,
        states=("enabled", "sensitive", "visible", "showing"),
    )
    # Add 220 decoy labels before unreachable deep node so BFS hits MAX_INSPECTED_DESCENDANTS=200
    many_nodes = [
        _FakeNode(
            bus=target_bus,
            path=f"/org/a11y/atspi/accessible/decoy/{i}",
            role="label",
            name=f"Decoy {i}",
            accessible_id=f"decoy_{i}",
        )
        for i in range(220)
    ]
    win = _FakeNode(
        bus=target_bus,
        path="/org/a11y/atspi/accessible/win",
        role="frame",
        name="Bounded Window",
        children=[long_lbl] + many_nodes,
    )
    app = _FakeNode(
        bus=target_bus,
        path="/org/a11y/atspi/accessible/app",
        role="application",
        name="bounded_app",
        children=[win],
    )
    world = _FakeAtspiRegistryWorld([(5555, app)])
    backend = _wire_backend_with_world(monkeypatch, world)
    hypr = _FakeHyprlandClient([{"address": "0x55", "title": "Bounded Window", "pid": 5555}])

    # Text truncation at MAX_RETURNED_TEXT_LEN (2048)
    long_res = await OutcomeVerifier.verify_async(
        "desktop_visible_text",
        "Bounded Window",
        selector="#longField",
        expected="X" * 100,
        desktop_accessibility_backend=backend,
        hyprland_client=hypr,
    )
    assert long_res.result == VERDICT_PASS
    assert len(long_res.observed["widget"]["visible_text"]) == MAX_RETURNED_TEXT_LEN
    get_text_calls = [c for c in world.calls if c["method"] == "GetText"]
    assert get_text_calls and get_text_calls[0]["args"] == (0, MAX_RETURNED_TEXT_LEN)

    # Missing target in oversized tree truncates cleanly at 200 descendants -> UNVERIFIED
    trunc_res = await OutcomeVerifier.verify_async(
        "desktop_visible_text",
        "Bounded Window",
        selector="#decoy_219",
        expected="Decoy 219",
        desktop_accessibility_backend=backend,
        hyprland_client=hypr,
    )
    assert trunc_res.result == VERDICT_UNVERIFIED
    assert trunc_res.observed["truncated"] is True
    assert trunc_res.observed["inspected_descendants"] == MAX_INSPECTED_DESCENDANTS


# ── 11. Malformed Selector / Spec Pre-Execution Rejection & Target Identity ─────


def test_selector_parsing_and_multi_widget_target_scoping() -> None:
    sel_ok, err = parse_desktop_selector("#statusLabel")
    assert err is None and sel_ok is not None and sel_ok.widget_id == "statusLabel"

    w, s = split_window_and_widget_target("Main Window::#statusLabel")
    assert w == "Main Window" and s == "#statusLabel"

    for bad in ("", "   ", "*", "xpath=//button", "eval=1+1", "text=CONFIRMED", "index=-1", "[role=label]", "button[name=OK]"):
        parsed, p_err = parse_desktop_selector(bad)
        assert parsed is None and p_err

    # Two semantic criteria on SAME window MUST produce distinct canonical raw targets
    spec = parse_acceptance_criteria([
        {"check": "desktop_visible_text", "window": "W", "selector": "#lbl1", "expected": "A"},
        {"check": "desktop_visible_text", "window": "W", "selector": "#lbl2", "expected": "B"},
    ])
    assert criterion_canonical_target_raw(spec[0]) == "W::#lbl1"
    assert criterion_canonical_target_raw(spec[1]) == "W::#lbl2"


@pytest.mark.asyncio
async def test_invalid_desktop_semantic_specs_rejected_before_model(tmp_path: Path, capsys) -> None:
    bad_specs = [
        json.dumps([{"check": "desktop_visible_text", "window": "Confirm", "expected": "OK"}]),
        json.dumps([{"check": "desktop_visible_text", "window": "Confirm", "selector": "xpath=//div", "expected": "OK"}]),
        json.dumps([{"check": "desktop_widget_state", "window": "Confirm", "selector": "#applyBtn", "expected": "telepathic"}]),
        json.dumps([{"check": "desktop_visible_text", "window": "*", "selector": "#statusLabel", "expected": "OK"}]),
    ]
    for spec in bad_specs:
        stream_calls = 0

        async def mock_stream(_payload):
            nonlocal stream_calls
            stream_calls += 1
            yield {"choices": [{"delta": {"content": "Should never run"}, "finish_reason": "stop"}]}

        central = CentralExecutionLedger(db_path=tmp_path / f"bad_{abs(hash(spec))}.db", raise_on_persistence_error=True)
        orch = NativeOrchestrator(central_ledger=central, stream_provider=mock_stream, max_steps=4)
        try:
            rc = await _execute_headless_run("Verify", orchestrator=orch, verify_spec=spec, output_json=True)
        finally:
            central.close()
        assert rc == 2
        assert stream_calls == 0
        payload = json.loads(capsys.readouterr().out.strip())
        assert payload["invalid_verification_spec"] is True


# ── 12 & 13. Scoped Mutation Invalidation & Focus-Only Separation ─────────────


def test_desktop_mutation_invalidates_only_target_window_preserving_other_surfaces() -> None:
    ledger = TurnExecutionLedger()
    ledger.record_proposed("fv1", "verify_outcome", {"check": "file_exists", "target": "/tmp/report.json"})
    ledger.record_dispatched("fv1")
    ledger.record_observation("fv1", "success", {"status": "success", "verified": True}, verified=True)
    ledger.record_target_verified("/tmp/report.json", check_type="file_exists")

    ledger.record_proposed(
        "dv_a",
        "verify_outcome",
        {"check": "desktop_visible_text", "window": "Window Alpha", "selector": "#statusLabel", "expected": "OK"},
    )
    ledger.record_dispatched("dv_a")
    ledger.record_observation("dv_a", "success", {"status": "success", "verified": True}, verified=True)
    ledger.record_target_verified("Window Alpha::#statusLabel", check_type="desktop_visible_text")

    ledger.record_proposed(
        "dv_b",
        "verify_outcome",
        {"check": "desktop_visible_text", "window": "Window Beta", "selector": "#statusLabel", "expected": "OK"},
    )
    ledger.record_dispatched("dv_b")
    ledger.record_observation("dv_b", "success", {"status": "success", "verified": True}, verified=True)
    ledger.record_target_verified("Window Beta::#statusLabel", check_type="desktop_visible_text")

    # Mutate ONLY Window Alpha -> Alpha invalidated, Beta & file preserved!
    ledger.invalidate_surface_verifications("desktop", window_target="Window Alpha")
    assert ledger.is_target_verified("Window Alpha::#statusLabel", check_type="desktop_visible_text") is False
    assert ledger.is_target_verified("Window Beta::#statusLabel", check_type="desktop_visible_text") is True
    assert ledger.is_target_verified("/tmp/report.json", check_type="file_exists") is True
    assert ledger.get_record("dv_a").state == ActionState.UNVERIFIED
    assert ledger.get_record("dv_b").state == ActionState.VERIFIED
    assert ledger.get_record("fv1").state == ActionState.VERIFIED

    # Focus IPC alone still cannot verify a semantic task without verification criteria
    focus_ledger = TurnExecutionLedger()
    focus_ledger.record_proposed("f1", "manage_desktop_window", {"action": "focus", "window": "Window Alpha"})
    focus_ledger.record_dispatched("f1")
    focus_ledger.record_observation("f1", "success", {"status": "success"}, verified=False)
    outcome = resolve_root_execution_outcome(
        prompt="Confirm Window Alpha shows CONFIRMED:REC-204",
        ledger=focus_ledger,
        final_text="Focused Window Alpha and confirmed.",
    )
    assert outcome.verification_status != VerificationStatus.VERIFIED
    assert outcome.execution_status == ExecutionStatus.COMPLETED_UNVERIFIED


# ── 14. VerifyOutcomeTool, CLI --verify --json & Oracle Independence ──────────


@pytest.mark.asyncio
async def test_verify_outcome_tool_cli_and_external_oracle_independence(tmp_path: Path, capsys, monkeypatch) -> None:
    world, clients = _build_fixture_world()
    hypr = _FakeHyprlandClient(clients)

    tool = VerifyOutcomeTool()
    param_names = {p.name for p in tool.parameters}
    assert {"window", "selector"} <= param_names

    import zelyr.engine.desktop_accessibility as da_mod

    orig_query = da_mod.query_desktop_widget_via_accessibility

    async def _injected_query(*, window_target, selector, backend=None, hyprland_client=None, hyprland_socket_path=None, timeout=3.0):
        return await orig_query(
            window_target=window_target,
            selector=selector,
            backend=backend or _wire_backend_with_world(monkeypatch, world),
            hyprland_client=hyprland_client or hypr,
            hyprland_socket_path=hyprland_socket_path,
            timeout=timeout,
        )

    monkeypatch.setattr(da_mod, "query_desktop_widget_via_accessibility", _injected_query)

    ok_res = await tool.execute(
        check="desktop_visible_text",
        window="Zelyr Cross-Surface Confirmation",
        selector="#statusLabel",
        expected="CONFIRMED:REC-204",
    )
    assert ok_res.success is True
    assert ok_res.output.startswith("[VERIFIED]")

    bad_res = await tool.execute(
        check="desktop_visible_text",
        window="Zelyr Cross-Surface Confirmation",
        selector="#statusLabel",
        expected="WRONG-TEXT",
    )
    assert bad_res.success is False
    assert "WRONG-TEXT" in (bad_res.error or "")

    # End-to-end CLI --verify --json with desktop accessibility checks passing
    report_f = tmp_path / "done.txt"
    report_f.write_text("CONFIRMED:REC-204\n", encoding="utf-8")
    spec = json.dumps([
        {"check": "file_contains", "target": str(report_f), "expected": "CONFIRMED:REC-204"},
        {
            "check": "desktop_visible_text",
            "window": "Zelyr Cross-Surface Confirmation",
            "selector": "#statusLabel",
            "expected": "CONFIRMED:REC-204",
        },
        {
            "check": "desktop_widget_state",
            "window": "Zelyr Cross-Surface Confirmation",
            "selector": "#applyBtn",
            "expected": "enabled",
        },
    ])

    async def mock_stream(_payload):
        yield {"choices": [{"delta": {"content": "Verified file and desktop."}, "finish_reason": "stop"}]}

    central = CentralExecutionLedger(db_path=tmp_path / "cli_a11y.db", raise_on_persistence_error=True)
    orch = NativeOrchestrator(central_ledger=central, stream_provider=mock_stream, max_steps=4)
    try:
        rc = await _execute_headless_run("Verify output and desktop", orchestrator=orch, verify_spec=spec, output_json=True)
    finally:
        central.close()
    assert rc == 0
    cli_json = json.loads(capsys.readouterr().out.strip())
    assert cli_json["root_execution_status"] == "completed"
    assert cli_json["root_verification_status"] == "verified"
    assert "desktop_accessibility" in cli_json["verification"]["scope"]["surfaces"]

    # Benchmark spec now natively checks desktop_visible_text + desktop_widget_state without touching /state,
    # while external validation still independently enforces HTTP/desktop /state consistency.
    ws_info = setup_cross_surface_workspace(tmp_path / "bench_ws", "http://127.0.0.1:43210", requires_desktop=True)
    bench_spec = build_cross_surface_verify_spec(
        ws_info, base_url="http://127.0.0.1:43210", requires_desktop=True, verify_mode="cross_surface"
    )
    assert "/state" not in bench_spec
    parsed_bench = parse_acceptance_criteria(bench_spec)
    assert any(c["check"] == "desktop_visible_text" for c in parsed_bench)
    assert any(c["check"] == "desktop_widget_state" for c in parsed_bench)

    import copy

    exp_rec = copy.deepcopy(INITIAL_RECORDS["REC-204"])
    exp_rec["status"] = "APPROVED"
    ws_info["report_json_path"].write_text(json.dumps(exp_rec, indent=2), encoding="utf-8")
    ws_info["report_md_path"].write_text(
        f"# Report\n\n- REC-204 billing-ledger fin-core eu-central-1 98 {exp_rec['confirmation_token']} APPROVED\n",
        encoding="utf-8",
    )
    run_info_ext = {
        "returncode": 0,
        "verify_spec": bench_spec,
        "cli_json": {
            **cli_json,
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
    }
    ext_eval = evaluate_cross_surface_run(
        scenario="normal",
        run_info=run_info_ext,
        web_audit={
            "records": {**copy.deepcopy(INITIAL_RECORDS), "REC-204": exp_rec},
            "requests": [
                {"method": "GET", "path": "/", "status": 200},
                {"method": "GET", "path": "/records/REC-204", "status": 200},
                {"method": "POST", "path": "/api/records/REC-204/approve", "status": 200},
                {"method": "GET", "path": "/api/records/REC-204", "status": 200},
            ],
        },
        ws_info=ws_info,
        desktop_after={"applied_text": "WRONG", "status_text": "WRONG", "key_presses": 1},
    )
    assert ext_eval["external_validation_status"] == "FAIL"
    assert ext_eval["benchmark_status"] == "FAIL"
    assert "desktop_widget_semantic_text" not in ext_eval["internal_verification"]["unsupported_surfaces"]
