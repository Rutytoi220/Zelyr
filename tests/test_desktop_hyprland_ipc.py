"""Comprehensive test suite for direct Hyprland UNIX domain socket IPC actuator and state verification."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from zelyr.engine.error_classifier import ErrorClassifier
from zelyr.engine.hyprland_ipc import HyprlandIPCClient, HyprlandUnavailableError
from zelyr.tools.hyprland_ipc import ManageDesktopWindowTool, execute


class MockHyprlandServer:
    """In-memory async mock UNIX domain socket server emulating Hyprland IPC (.socket.sock)."""

    def __init__(self, socket_path: Path):
        self.socket_path = socket_path
        self.server: asyncio.AbstractServer | None = None
        self.active_workspace = {"id": 1, "name": "1"}
        self.active_window = {"address": "0x55a8e8690210", "class": "kitty", "title": "Terminal"}
        self.clients = [
            {
                "address": "0x55a8e8690210",
                "class": "kitty",
                "title": "Terminal",
                "workspace": {"id": 1, "name": "1"},
                "at": [0, 0],
                "size": [1920, 1080],
                "floating": False,
                "fullscreen": False,
            },
            {
                "address": "0x55a8e8690330",
                "class": "firefox",
                "title": "Mozilla Firefox",
                "workspace": {"id": 2, "name": "2"},
                "at": [100, 100],
                "size": [800, 600],
                "floating": False,
                "fullscreen": False,
            },
        ]
        self.fail_workspace_update = False

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        data = await reader.read(4096)
        cmd = data.decode("utf-8").strip()

        if cmd == "j/clients":
            resp = json.dumps(self.clients)
        elif cmd == "j/activewindow":
            resp = json.dumps(self.active_window)
        elif cmd == "j/activeworkspace":
            resp = json.dumps(self.active_workspace)
        elif cmd == "j/workspaces":
            resp = json.dumps([{"id": 1, "name": "1"}, {"id": 2, "name": "2"}])
        elif cmd.startswith("dispatch workspace "):
            target = cmd.removeprefix("dispatch workspace ").strip()
            if not self.fail_workspace_update:
                self.active_workspace = {"id": int(target) if target.isdigit() else target, "name": target}
            resp = "ok"
        elif cmd.startswith("dispatch focuswindow "):
            arg = cmd.removeprefix("dispatch focuswindow ").strip()
            addr = arg.removeprefix("address:").strip()
            for c in self.clients:
                if c["address"] == addr or arg.lower() in c["class"].lower() or arg.lower() in c["title"].lower():
                    self.active_window = {"address": c["address"], "class": c["class"], "title": c["title"]}
                    break
            resp = "ok"
        elif cmd.startswith("dispatch movetoworkspace "):
            target = cmd.removeprefix("dispatch movetoworkspace ").strip()
            ws_val = int(target) if target.isdigit() else target
            for c in self.clients:
                if c["address"] == self.active_window.get("address"):
                    c["workspace"] = {"id": ws_val, "name": str(ws_val)}
                    break
            resp = "ok"
        elif cmd.startswith("dispatch "):
            resp = "ok"
        else:
            resp = "unknown request"

        writer.write(resp.encode("utf-8"))
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def start(self) -> None:
        self.server = await asyncio.start_unix_server(self.handle_client, path=str(self.socket_path))

    async def stop(self) -> None:
        if self.server:
            self.server.close()
            await self.server.wait_closed()


class MockHyprlandContext:
    """Async context manager for spinning up an isolated mock Hyprland socket server."""

    def __init__(self, tmp_path: Path):
        self.sock_dir = tmp_path / "hypr" / "test_sig_123"
        self.sock_dir.mkdir(parents=True, exist_ok=True)
        self.sock_path = self.sock_dir / ".socket.sock"
        self.server = MockHyprlandServer(self.sock_path)

    async def __aenter__(self) -> tuple[MockHyprlandServer, Path]:
        await self.server.start()
        return self.server, self.sock_path

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.server.stop()


@pytest.mark.asyncio
async def test_hyprland_ipc_queries(tmp_path):
    """Verify that HyprlandIPCClient queries j/clients, j/activewindow, and j/activeworkspace cleanly."""
    async with MockHyprlandContext(tmp_path) as (_server, sock_path):
        client = HyprlandIPCClient(socket_path=sock_path)

        assert client.is_available() is True
        assert client.can_connect() is True

        clients = await client.get_clients()
        assert len(clients) == 2
        assert clients[0]["class"] == "kitty"

        active = await client.get_active_window()
        assert active["class"] == "kitty"
        assert active["address"] == "0x55a8e8690210"

        ws = await client.get_active_workspace()
        assert ws["id"] == 1


@pytest.mark.asyncio
async def test_dispatch_workspace_state_verification_success(tmp_path):
    """Verify workspace dispatch passes verification when active workspace changes to target."""
    async with MockHyprlandContext(tmp_path) as (_server, sock_path):
        client = HyprlandIPCClient(socket_path=sock_path)

        res = await client.dispatch("workspace", "2", verify=True)
        assert res["status"] == "success"
        assert res["verified"] is True
        assert str(res["observed"].get("id")) == "2"


@pytest.mark.asyncio
async def test_dispatch_workspace_state_verification_failure(tmp_path):
    """Verify workspace dispatch reports verified=False when active workspace remains unchanged."""
    async with MockHyprlandContext(tmp_path) as (server, sock_path):
        server.fail_workspace_update = True
        client = HyprlandIPCClient(socket_path=sock_path)

        res = await client.dispatch("workspace", "2", verify=True)
        assert res["status"] == "failed"
        assert res["verified"] is False
        assert str(res["observed"].get("id")) == "1"


@pytest.mark.asyncio
async def test_dispatch_focus_window_state_verification(tmp_path):
    """Verify focuswindow dispatch updates active window and verifies target address."""
    async with MockHyprlandContext(tmp_path) as (_server, sock_path):
        client = HyprlandIPCClient(socket_path=sock_path)

        res = await client.dispatch("focuswindow", "address:0x55a8e8690330", verify=True)
        assert res["status"] == "success"
        assert res["verified"] is True
        assert res["observed"].get("address") == "0x55a8e8690330"


@pytest.mark.asyncio
async def test_degradation_when_signature_unset_and_socket_absent(monkeypatch):
    """Verify clean DESKTOP_UNAVAILABLE error envelope when Hyprland environment is missing."""
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)

    with patch("shutil.which", return_value=None):
        client = HyprlandIPCClient()
        assert client.is_available() is False
        assert client.can_connect() is False

        with pytest.raises(HyprlandUnavailableError):
            await client.send_command("j/clients")

        # Dynamic tool execution returns structured DESKTOP_UNAVAILABLE JSON
        raw_output = await execute(action="workspace", target="1", client=client)
        parsed = json.loads(raw_output)
        assert parsed["status"] == "error"
        assert parsed["error_type"] == "DESKTOP_UNAVAILABLE"
        assert "Hyprland session not detected" in parsed["error"]
        assert "remedy_hint" in parsed

        # Error classifier parses it into DESKTOP_UNAVAILABLE
        classification = ErrorClassifier.classify("manage_desktop_window", output=raw_output)
        assert classification.error_type == "DESKTOP_UNAVAILABLE"
        assert "execute_command" in classification.allowed_actions


@pytest.mark.asyncio
async def test_manage_desktop_window_tool_actions(tmp_path):
    """Test manage_desktop_window tool handler across list, focus, workspace, and invalid actions."""
    async with MockHyprlandContext(tmp_path) as (_server, sock_path):
        client = HyprlandIPCClient(socket_path=sock_path)

        # 1. Action 'list'
        out_list = json.loads(await execute(action="list", client=client))
        assert out_list["status"] == "success"
        assert out_list["count"] == 2
        assert out_list["windows"][0]["class"] == "kitty"

        # 2. Action 'get_active'
        out_active = json.loads(await execute(action="get_active", client=client))
        assert out_active["status"] == "success"
        assert out_active["active_window"]["class"] == "kitty"

        # 3. Action 'focus' with fuzzy class match
        out_focus = json.loads(await execute(action="focus", target="firefox", client=client))
        assert out_focus["status"] == "success"
        assert out_focus["verified"] is True
        assert out_focus["observed"]["class"] == "firefox"

        # 4. Action 'focus' window not found
        out_fnf = json.loads(await execute(action="focus", target="unknown_app_xyz", client=client))
        assert out_fnf["status"] == "error"
        assert out_fnf["error_type"] == "WINDOW_NOT_FOUND"

        classification = ErrorClassifier.classify("manage_desktop_window", output=json.dumps(out_fnf))
        assert classification.error_type == "WINDOW_NOT_FOUND"
        assert "manage_desktop_window" in classification.allowed_actions

        # 5. Action 'workspace'
        out_ws = json.loads(await execute(action="workspace", target="2", client=client))
        assert out_ws["status"] == "success"
        assert out_ws["verified"] is True
        assert out_ws["workspace"] == "2"

        # 6. Invalid action
        out_invalid = json.loads(await execute(action="non_existent_action", client=client))
        assert out_invalid["status"] == "error"
        assert "Unknown action" in out_invalid["error"]


@pytest.mark.asyncio
async def test_tool_instance_execution_with_client(tmp_path):
    """Verify ManageDesktopWindowTool BaseTool instance works with mock client."""
    async with MockHyprlandContext(tmp_path) as (_server, sock_path):
        client = HyprlandIPCClient(socket_path=sock_path)
        tool = ManageDesktopWindowTool(client=client)

        res = await tool.execute({"action": "workspace", "target": "2"})
        assert res.success is True
        assert res.output["action"] == "workspace"
        assert res.output["workspace"] == "2"
        assert res.output["verified"] is True
