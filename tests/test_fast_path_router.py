"""Unit & Integration tests for Hybrid Fastest Path Tool Router."""

from __future__ import annotations

import shutil
from unittest.mock import MagicMock, patch

import pytest

from zelyr.core.fast_path_router import FastPathPlan, FastestPathRouter
from instruction_parser import execute_instruction_fast_path, parse as parse_instruction


@pytest.fixture(autouse=True)
def guard_desktop_processes(monkeypatch):
    """Guard against unmocked browser spawns or desktop commands during router tests."""
    mock_popen = MagicMock()
    mock_run = MagicMock(return_value=MagicMock(returncode=0, stdout="", stderr=""))
    mock_open = MagicMock(return_value=True)

    monkeypatch.setattr("subprocess.Popen", mock_popen)
    monkeypatch.setattr("subprocess.run", mock_run)
    monkeypatch.setattr("webbrowser.open", mock_open, raising=False)
    monkeypatch.setattr("webbrowser.open_new_tab", mock_open, raising=False)
    yield


@patch("webbrowser.open", return_value=True)
def test_url_routing(mock_browser):
    """Verify web URLs are intercepted and routed to xdg-open."""
    urls = [
        "https://github.com/ChienGPT/Zelyr",
        "http://localhost:8080/dashboard",
        "www.google.com",
    ]
    for url in urls:
        plan = FastestPathRouter.can_route_to_cli("open_browser", {"url": url})
        assert plan is not None, f"Failed to route URL: {url}"
        assert plan.command[0] == "xdg-open"
        assert url in plan.command[1] or ("https://" + url) in plan.command[1]
        assert plan.category == "browser"


@patch("shutil.which", return_value="/usr/bin/kitty")
def test_app_launch_routing(mock_which):
    """Verify application launches prioritize CLI binaries or gtk-launch."""
    plan = FastestPathRouter.can_route_to_cli("open_app", {"name": "terminal"})
    assert plan is not None
    assert plan.category == "app"
    cmd0 = plan.command[0]
    assert cmd0.endswith("kitty") or cmd0.endswith("gtk-launch") or cmd0.endswith("terminal")


def test_systemctl_routing():
    """Verify system service management routes to systemctl."""
    plan = FastestPathRouter.can_route_to_cli("service_action", {"command": "restart service bluetooth"})
    if shutil.which("systemctl"):
        assert plan is not None
        assert plan.command == ["systemctl", "restart", "bluetooth"]
        assert plan.category == "service"


def test_audio_routing():
    """Verify audio volume commands route to wpctl."""
    plan = FastestPathRouter.can_route_to_cli("set_volume", {"text": "volume up"})
    if shutil.which("wpctl"):
        assert plan is not None
        assert plan.command[0] == "wpctl"
        assert plan.category == "audio"


def test_gui_fallback_preserved_for_canvas_elements():
    """Verify raw mouse and keyboard clicks on visually locked widgets remain GUI fallback."""
    # A click at coordinates (x, y) with no CLI intent should return None
    plan = FastestPathRouter.can_route_to_cli("mouse_click", {"x": 350, "y": 720, "button": "left"})
    assert plan is None, "Pure mouse_click must NOT be intercepted by CLI router!"

    plan_move = FastestPathRouter.can_route_to_cli("mouse_move", {"x": 100, "y": 200})
    assert plan_move is None, "Pure mouse_move must NOT be intercepted by CLI router!"


def test_instruction_parser_fast_path_integration():
    """Verify instruction_parser tags fast path plans onto parsed LLM instructions."""
    text = "[INSTRUCTION]open_url https://duckduckgo.com[/INSTRUCTION]"
    res = parse_instruction(text)
    assert res.get("type") == "instruction"
    assert res.get("fast_path") is True
    assert res.get("cli_command")[0] == "xdg-open"
    assert "https://duckduckgo.com" in res.get("cli_command")[1]


def test_instruction_parser_pure_gui_fallback():
    """Verify instruction_parser preserves pure GUI instructions without CLI override."""
    text = "[INSTRUCTION]click_mouse 500 300[/INSTRUCTION]"
    res = parse_instruction(text)
    assert res.get("type") == "instruction"
    assert not res.get("fast_path", False)
    assert "cli_command" not in res


@pytest.mark.asyncio
async def test_desktop_automation_tool_fast_path_interception():
    """Verify DesktopAutomationTool delegates fast path actions instead of calling motor controls."""
    from zelyr.tools.desktop_control import DesktopAutomationTool

    tool = DesktopAutomationTool()
    with patch("subprocess.Popen") as mock_popen, patch("webbrowser.open", return_value=True, create=True):
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc
        # Intercepted action:
        res = await tool.execute("open_url", url="https://example.com")
        assert res.success is True
        assert res.metadata.get("fast_path") is True
        assert res.metadata.get("command") == ["xdg-open", "https://example.com"]
        mock_popen.assert_called_once()
        assert mock_popen.call_args[0][0] == ["xdg-open", "https://example.com"]
        assert mock_popen.call_args[1].get("start_new_session") is True
