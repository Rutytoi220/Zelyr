"""Tests for Shell authorization and safety boundaries in zelyr.tools."""

import os
import shutil
import sys
import unittest
from unittest.mock import patch

from zelyr.config import get_config, set_config, ZelyrConfig, AuthMode
from zelyr.tools import _request_shell_authorization, ShellTool, ShellCommandTool


class TestShellAuthorization(unittest.TestCase):
    """Verify shell authorization contracts across interactive and non-interactive environments."""

    def setUp(self):
        self._orig_env = os.environ.copy()
        os.environ.pop("ZELYR_AUTOPILOT", None)
        os.environ.pop("ZELYR_TESTING", None)
        os.environ.pop("PYTEST_CURRENT_TEST", None)
        set_config(ZelyrConfig(auth_mode=AuthMode.BASIC))

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._orig_env)
        set_config(ZelyrConfig())

    def test_non_interactive_fails_closed(self):
        """In a non-interactive environment without autopilot, authorization fails closed."""
        with patch.object(sys.stdin, "isatty", return_value=False):
            authorized = _request_shell_authorization("shell", "echo test")
            self.assertFalse(authorized, "Non-interactive session without autopilot must NOT be authorized")

    def test_interactive_approved(self):
        """When an interactive user answers 'yes', authorization succeeds."""
        with patch.object(sys.stdin, "isatty", return_value=True):
            with patch("rich.prompt.Confirm.ask", return_value=True):
                authorized = _request_shell_authorization("shell", "echo test")
                self.assertTrue(authorized, "Explicit user confirmation must authorize execution")

    def test_interactive_denied(self):
        """When an interactive user answers 'no', authorization is denied."""
        with patch.object(sys.stdin, "isatty", return_value=True):
            with patch("rich.prompt.Confirm.ask", return_value=False):
                authorized = _request_shell_authorization("shell", "echo test")
                self.assertFalse(authorized, "User denial must reject execution")

    def test_interactive_keyboard_interrupt_or_eof(self):
        """EOF or KeyboardInterrupt during prompt must fail closed."""
        with patch.object(sys.stdin, "isatty", return_value=True):
            with patch("rich.prompt.Confirm.ask", side_effect=EOFError):
                self.assertFalse(_request_shell_authorization("shell", "echo test"))
            with patch("rich.prompt.Confirm.ask", side_effect=KeyboardInterrupt):
                self.assertFalse(_request_shell_authorization("shell", "echo test"))

    def test_autopilot_env_bypass(self):
        """ZELYR_AUTOPILOT=1 allows execution without interactive confirmation."""
        os.environ["ZELYR_AUTOPILOT"] = "1"
        with patch.object(sys.stdin, "isatty", return_value=False):
            self.assertTrue(_request_shell_authorization("shell", "echo test"))

    def test_testing_env_bypass(self):
        """ZELYR_TESTING=1 allows execution during automated test harnesses."""
        os.environ["ZELYR_TESTING"] = "1"
        with patch.object(sys.stdin, "isatty", return_value=False):
            self.assertTrue(_request_shell_authorization("shell", "echo test"))

    async def async_test_shell_tool_non_interactive_blocked(self):
        """Async ShellTool execution is aborted if non-interactive and unauthorized."""
        with patch.object(sys.stdin, "isatty", return_value=False):
            tool = ShellTool(allow_dangerous=False)
            tool._allow_dangerous = False  # override any test-env auto-enabling
            res = await tool.execute({"command": "echo 'should not run'"})
            self.assertFalse(res.success)
            self.assertEqual(res.error, "Command execution aborted by user")

    def test_shell_tool_execution_aborted_when_unauthorized(self):
        """Run async test in event loop."""
        import asyncio
        asyncio.run(self.async_test_shell_tool_non_interactive_blocked())

    def test_sync_shell_command_tool_non_interactive_blocked(self):
        """Sync ShellCommandTool execution is aborted if non-interactive and unauthorized."""
        with patch.object(sys.stdin, "isatty", return_value=False):
            tool = ShellCommandTool()
            res = tool.execute("echo 'should not run'")
            self.assertFalse(res.success)
            self.assertEqual(res.error, "Command execution aborted by user")

    def test_sync_shell_command_tool_autopilot_allowed(self):
        """Sync ShellCommandTool execution proceeds when in autopilot mode."""
        os.environ["ZELYR_AUTOPILOT"] = "1"
        with patch.object(sys.stdin, "isatty", return_value=False), \
             patch("shutil.which", side_effect=lambda x: None if x == "bwrap" else shutil.which(x)):
            tool = ShellCommandTool()
            res = tool.execute("echo 'hello from autopilot'")
            self.assertTrue(res.success)
            self.assertIn("hello from autopilot", res.output.get("stdout", ""))


if __name__ == "__main__":
    unittest.main()
