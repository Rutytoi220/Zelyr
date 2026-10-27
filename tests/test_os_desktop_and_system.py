"""Test suite for Tier 1 Linux Desktop Media, Desktop Notifications, and Network Inspection."""

import asyncio
import json
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from zelyr.core.plugins import get_tier_timeout
from zelyr.tools.hyprland_ipc import ManageDesktopWindowTool
from zelyr.tools.os_desktop import ManageMediaPlaybackTool, SendDesktopNotificationTool
from zelyr.tools.os_system import (
    InspectNetworkTool,
    ManageSystemClipboardTool,
    ManageSystemProcessTool,
    QuerySystemJournalTool,
)


class TestSendDesktopNotificationTool(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tool = SendDesktopNotificationTool()

    async def test_notification_success(self):
        """Verify successful desktop notification dispatch with mocked subprocess."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (b"", b"")

        with patch("shutil.which", return_value="/usr/bin/notify-send"), patch(
            "asyncio.create_subprocess_exec", return_value=mock_proc
        ) as mock_exec:
            res = await self.tool.execute({
                "title": "Build Finished",
                "message": "All 146 tests passed successfully.",
                "urgency": "low",
                "app_name": "Zelyr-Test",
            })

            self.assertTrue(res.success)
            self.assertTrue(res.output.get("delivered"))
            self.assertEqual(res.output.get("title"), "Build Finished")
            self.assertEqual(res.output.get("urgency"), "low")

            mock_exec.assert_called_once_with(
                "notify-send",
                "-u",
                "low",
                "-a",
                "Zelyr-Test",
                "Build Finished",
                "All 146 tests passed successfully.",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

    async def test_notification_missing_parameters(self):
        """Verify missing title or message produces validation errors."""
        res_no_title = await self.tool.execute({"title": "", "message": "Some message"})
        self.assertFalse(res_no_title.success)
        self.assertIn("Title is required", res_no_title.error)

        res_no_msg = await self.tool.execute({"title": "Some Title", "message": ""})
        self.assertFalse(res_no_msg.success)
        self.assertIn("Message body is required", res_no_msg.error)

    async def test_notification_subprocess_failure(self):
        """Verify non-zero return code from notify-send returns error envelope."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate.return_value = (b"", b"Error: GDBus.Error:org.freedesktop.Notifications.MaxNotifications")

        with patch("shutil.which", return_value="/usr/bin/notify-send"), patch(
            "asyncio.create_subprocess_exec", return_value=mock_proc
        ):
            res = await self.tool.execute({"title": "Test", "message": "Failed notification"})
            self.assertFalse(res.success)
            self.assertIn("notify-send failed", res.error)

    async def test_notification_fallback_when_binary_missing(self):
        """Verify fallback to console logging when notify-send binary is missing."""
        with patch("shutil.which", return_value=None):
            res = await self.tool.execute({"title": "Fallback", "message": "Console log only"})
            self.assertTrue(res.success)
            self.assertEqual(res.output.get("fallback"), "console")
            self.assertFalse(res.output.get("delivered"))


class TestManageMediaPlaybackTool(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tool = ManageMediaPlaybackTool()

    async def test_media_playback_status_and_metadata(self):
        """Verify status query correctly parses metadata JSON and volume."""
        mock_metadata = json.dumps({
            "player": "spotify",
            "status": "Playing",
            "title": "Midnight City",
            "artist": "M83",
            "album": "Hurry Up, We're Dreaming",
        })

        async def fake_run(*args, player=""):
            if "metadata" in args:
                return (0, mock_metadata, "")
            if "volume" in args:
                return (0, "0.80", "")
            return (0, "", "")

        with patch("shutil.which", return_value="/usr/bin/playerctl"), patch.object(
            self.tool, "_run_playerctl", side_effect=fake_run
        ):
            res = await self.tool.execute({"action": "status"})
            self.assertTrue(res.success)
            self.assertEqual(res.output.get("player"), "spotify")
            self.assertEqual(res.output.get("status"), "Playing")
            self.assertEqual(res.output.get("title"), "Midnight City")
            self.assertEqual(res.output.get("volume"), 0.8)
            self.assertEqual(res.output.get("volume_percent"), "80%")

    async def test_media_playback_controls(self):
        """Verify playback state actions (play_pause, next, previous)."""
        for act in ("play_pause", "next", "previous"):
            with patch("shutil.which", return_value="/usr/bin/playerctl"), patch.object(
                self.tool, "_run_playerctl", return_value=(0, "ok", "")
            ):
                res = await self.tool.execute({"action": act})
                self.assertTrue(res.success)
                self.assertEqual(res.output.get("action"), act)

    async def test_media_playback_volume_adjustment(self):
        """Verify volume adjustments handle percentage formatting."""
        async def fake_run(*args, player=""):
            if "50%" in args or "0.5" in args:
                return (0, "", "")
            if "volume" in args:
                return (0, "0.50", "")
            return (0, "", "")

        with patch("shutil.which", return_value="/usr/bin/playerctl"), patch.object(
            self.tool, "_run_playerctl", side_effect=fake_run
        ):
            res = await self.tool.execute({"action": "volume", "value": "50%"})
            self.assertTrue(res.success)
            self.assertEqual(res.output.get("applied"), "0.5")

    async def test_media_playback_no_players(self):
        """Verify graceful envelope when no active media players are running."""
        with patch("shutil.which", return_value="/usr/bin/playerctl"), patch.object(
            self.tool, "_run_playerctl", return_value=(1, "", "No players found")
        ):
            res = await self.tool.execute({"action": "play_pause"})
            self.assertTrue(res.success)
            self.assertIn("No active media players", res.output.get("status", ""))

    async def test_media_playback_binary_missing(self):
        """Verify graceful response when playerctl utility is not installed."""
        with patch("shutil.which", return_value=None):
            res = await self.tool.execute({"action": "status"})
            self.assertTrue(res.success)
            self.assertIn("No active media players", res.output.get("status", ""))
            self.assertIn("not installed", res.output.get("note", ""))

    async def test_media_playback_unsupported_action(self):
        """Verify unsupported action returns error envelope."""
        with patch("shutil.which", return_value="/usr/bin/playerctl"):
            res = await self.tool.execute({"action": "burn_cd"})
            self.assertFalse(res.success)
            self.assertIn("Unsupported media action", res.error)


class TestInspectNetworkTool(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tool = InspectNetworkTool()

    async def test_inspect_network_offline_mocked(self):
        """Verify gateway, interfaces, DNS, and Tailscale diagnostics with mocked subprocesses."""
        mock_route_out = b"default via 192.168.1.1 dev wlan0 proto dhcp src 192.168.1.50 metric 600\n"
        mock_ip_brief = (
            b"lo               UNKNOWN        127.0.0.1/8 ::1/128\n"
            b"wlan0            UP             192.168.1.50/24 fe80::1/64\n"
        )
        mock_tailscale = json.dumps({
            "BackendState": "Running",
            "TailscaleIPs": ["100.64.0.1"],
            "Self": {"HostName": "laptop"},
            "Peer": {"peer1": {}},
        }).encode("utf-8")

        def fake_which(cmd):
            if cmd in ("ip", "tailscale"):
                return f"/usr/bin/{cmd}"
            return None

        async def fake_subprocess(*cmd, **kwargs):
            proc = AsyncMock()
            if "route" in cmd:
                proc.returncode = 0
                proc.communicate.return_value = (mock_route_out, b"")
            elif "-brief" in cmd:
                proc.returncode = 0
                proc.communicate.return_value = (mock_ip_brief, b"")
            elif "tailscale" in cmd:
                proc.returncode = 0
                proc.communicate.return_value = (mock_tailscale, b"")
            else:
                proc.returncode = 0
                proc.communicate.return_value = (b"", b"")
            return proc

        with patch("shutil.which", side_effect=fake_which), patch(
            "asyncio.create_subprocess_exec", side_effect=fake_subprocess
        ), patch("socket.gethostbyname", return_value="1.1.1.1"):
            res = await self.tool.execute({"check_target": "1.1.1.1"})

            self.assertTrue(res.success)
            data = res.output

            # 1. Gateway
            gw = data.get("gateway", {})
            self.assertEqual(gw.get("gateway"), "192.168.1.1")
            self.assertEqual(gw.get("interface"), "wlan0")

            # 2. Interfaces
            ifaces = data.get("interfaces", [])
            self.assertEqual(len(ifaces), 2)
            self.assertEqual(ifaces[0]["interface"], "lo")
            self.assertEqual(ifaces[1]["interface"], "wlan0")
            self.assertEqual(ifaces[1]["state"], "UP")

            # 3. DNS
            dns = data.get("dns", {})
            self.assertEqual(dns.get("target"), "1.1.1.1")
            self.assertEqual(dns.get("resolved_ip"), "1.1.1.1")
            self.assertEqual(dns.get("status"), "resolved")

            # 4. Tailscale
            tailscale = data.get("tailscale", {})
            self.assertTrue(tailscale.get("tailscale_active"))


class TestManageSystemProcessTool(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tool = ManageSystemProcessTool()

    async def test_process_list(self):
        """Verify process listing returns non-zero processes with required keys."""
        res = await self.tool.execute({"action": "list"})
        self.assertTrue(res.success)
        procs = res.output.get("processes", [])
        self.assertGreater(len(procs), 0)
        first = procs[0]
        self.assertIn("pid", first)
        self.assertIn("comm", first)

    async def test_process_inspect_self(self):
        """Verify inspecting self PID returns process details."""
        self_pid = os.getpid()
        res = await self.tool.execute({"action": "inspect", "target": str(self_pid)})
        self.assertTrue(res.success)
        self.assertEqual(res.output.get("pid"), self_pid)
        self.assertIn("comm", res.output)
        self.assertIn("args", res.output)

    async def test_process_inspect_by_name(self):
        """Verify inspecting by name returns match list."""
        res = await self.tool.execute({"action": "inspect", "target": "python"})
        self.assertTrue(res.success)
        self.assertIn("matches", res.output)
        self.assertGreaterEqual(res.output.get("count", 0), 1)

    async def test_process_protection_guards(self):
        """Verify PID 1, self, and parent process are strictly protected against termination."""
        # 1. PID 1
        res_1 = await self.tool.execute({"action": "kill", "target": "1"})
        self.assertFalse(res_1.success)
        self.assertTrue(any(w in res_1.error.lower() for w in ("refused", "protected")))

        # 2. Self PID
        res_self = await self.tool.execute({"action": "kill", "target": str(os.getpid())})
        self.assertFalse(res_self.success)
        self.assertTrue(any(w in res_self.error.lower() for w in ("refused", "current", "protected")))

        # 3. Parent PID
        res_parent = await self.tool.execute({"action": "kill", "target": str(os.getppid())})
        self.assertFalse(res_parent.success)
        self.assertTrue(any(w in res_parent.error.lower() for w in ("refused", "parent", "protected")))


class TestManageSystemClipboardTool(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tool = ManageSystemClipboardTool()

    async def test_clipboard_write_and_read_mocked(self):
        """Verify clipboard write and read via mocked subprocess."""
        mock_proc_write = AsyncMock()
        mock_proc_write.returncode = 0
        mock_proc_write.communicate.return_value = (b"", b"")

        mock_proc_read = AsyncMock()
        mock_proc_read.returncode = 0
        mock_proc_read.communicate.return_value = (b"test_token_42", b"")

        def fake_which(cmd):
            return f"/usr/bin/{cmd}" if cmd in ("wl-copy", "wl-paste", "xclip") else None

        with patch("shutil.which", side_effect=fake_which), patch(
            "asyncio.create_subprocess_exec", side_effect=[mock_proc_write, mock_proc_read]
        ), patch.dict(os.environ, {"WAYLAND_DISPLAY": "wayland-0"}):
            res_write = await self.tool.execute({"action": "write", "content": "test_token_42"})
            self.assertTrue(res_write.success)
            self.assertEqual(res_write.output.get("content"), "test_token_42")

            res_read = await self.tool.execute({"action": "read"})
            self.assertTrue(res_read.success)
            self.assertEqual(res_read.output.get("content"), "test_token_42")

    async def test_clipboard_unavailable_when_binaries_missing(self):
        """Verify graceful error envelope when no clipboard utility is installed."""
        with patch("shutil.which", return_value=None):
            res = await self.tool.execute({"action": "read"})
            self.assertFalse(res.success)
            self.assertIn("No supported clipboard utility found", res.error)


class TestQuerySystemJournalTool(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tool = QuerySystemJournalTool()

    async def test_journal_query_mocked(self):
        """Verify journalctl log retrieval and structured envelope."""
        mock_log = (
            b"2026-10-15T22:00:00+02:00 host systemd[1]: Started Service.\n"
            b"2026-10-15T22:00:01+02:00 host app[123]: Ready.\n"
        )
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (mock_log, b"")

        with patch("shutil.which", return_value="/usr/bin/journalctl"), patch(
            "asyncio.create_subprocess_exec", return_value=mock_proc
        ):
            res = await self.tool.execute({"lines": 2})
            self.assertTrue(res.success)
            self.assertEqual(res.output.get("lines"), 2)
            self.assertEqual(res.output.get("count"), 2)
            self.assertIn("logs", res.output)

    async def test_journal_binary_missing(self):
        """Verify graceful error envelope when journalctl binary is missing."""
        with patch("shutil.which", return_value=None):
            res = await self.tool.execute({"lines": 10})
            self.assertFalse(res.success)
            self.assertIn("journalctl command not found", res.error)


class TestTierTimeouts(unittest.TestCase):
    def test_tier_timeouts(self):
        """Verify standard per-tier execution latency budget timeouts."""
        self.assertEqual(get_tier_timeout("manage_system_process"), 8.0)
        self.assertEqual(get_tier_timeout("manage_system_clipboard"), 8.0)
        self.assertEqual(get_tier_timeout("query_system_journal"), 8.0)
        self.assertEqual(get_tier_timeout("manage_desktop_window"), 8.0)
        self.assertEqual(get_tier_timeout("execute_command"), 8.0)
        self.assertEqual(get_tier_timeout("interact_with_browser"), 10.0)
        self.assertEqual(get_tier_timeout("interact_with_ui"), 35.0)
        self.assertEqual(get_tier_timeout("vision"), 35.0)
        self.assertEqual(get_tier_timeout("capture_som_screen"), 35.0)
        self.assertEqual(get_tier_timeout("click_tag"), 35.0)


class TestManageDesktopWindowTool(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tool = ManageDesktopWindowTool()

    async def test_hyprland_workspace_dispatch_mocked(self):
        """Verify Hyprland IPC workspace switching and window management with mocked subprocess."""
        async def fake_run_hyprctl(command: str):
            if "clients" in command:
                return 0, json.dumps([{"address": "0x123", "class": "zen", "title": "Zen Browser", "workspace": {"name": "1"}}]), ""
            elif "activewindow" in command:
                return 0, json.dumps({"address": "0x123", "title": "Zen Browser"}), ""
            return 0, "ok", ""

        with patch("shutil.which", return_value="/usr/bin/hyprctl"), patch.dict(
            os.environ, {"HYPRLAND_INSTANCE_SIGNATURE": "fake_sig_123"}
        ), patch.object(self.tool, "_run_hyprctl", side_effect=fake_run_hyprctl):
            # 1. Switch workspace
            res_ws = await self.tool.execute({"action": "workspace", "target": "2"})
            self.assertTrue(res_ws.success)
            self.assertEqual(res_ws.output.get("action"), "workspace")
            self.assertEqual(res_ws.output.get("workspace"), "2")

            # 2. Move to workspace
            res_move = await self.tool.execute({"action": "move_to_workspace", "target": "3"})
            self.assertTrue(res_move.success)
            self.assertEqual(res_move.output.get("action"), "move_to_workspace")
            self.assertEqual(res_move.output.get("target_workspace"), "3")

            # 3. Toggle floating
            res_float = await self.tool.execute({"action": "toggle_floating"})
            self.assertTrue(res_float.success)
            self.assertEqual(res_float.output.get("action"), "toggle_floating")

            # 4. Fullscreen
            res_fs = await self.tool.execute({"action": "fullscreen"})
            self.assertTrue(res_fs.success)
            self.assertEqual(res_fs.output.get("action"), "fullscreen")

            # 5. Missing target validation
            res_missing = await self.tool.execute({"action": "workspace"})
            self.assertFalse(res_missing.success)
            self.assertIn("Target workspace", res_missing.error)

    async def test_hyprland_missing_environment(self):
        """Verify graceful error when Hyprland environment or hyprctl binary is not present."""
        with patch("shutil.which", return_value=None), patch.dict(os.environ, {}, clear=True):
            res = await self.tool.execute({"action": "workspace", "target": "1"})
            self.assertFalse(res.success)
            self.assertTrue(any(w in res.error.lower() for w in ("hyprland", "not found", "unavailable")))


if __name__ == "__main__":
    unittest.main()
