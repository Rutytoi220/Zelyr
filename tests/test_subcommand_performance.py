"""Regression and performance test suite for CLI subcommands.

Covers:
1. zelyr status execution latency under 120ms with valid telemetry fields.
2. zelyr tool list outputs all 10 registered built-in tools.
3. zelyr tool list does not import PluginManager or watchdog during default invocation.
4. zelyr tool list --all / --plugins discovers plugins on demand.
5. zelyr tool run executes built-in tools correctly.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import unittest

from zelyr.tool_registry import BUILTIN_TOOLS, ToolRegistry


class TestSubcommandPerformance(unittest.TestCase):
    """Verify latency and lazy-loading boundaries of CLI subcommands."""

    def test_status_latency_and_output(self):
        """zelyr status must complete under 120ms and display telemetry fields."""
        cmd = [sys.executable, "-m", "zelyr.cli.main", "status"]
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"

        # Warm-up run to prime pyc / disk cache
        subprocess.run(cmd, capture_output=True, env=env, check=True)

        latencies = []
        last_stdout = ""
        for _ in range(3):
            t0 = time.perf_counter()
            res = subprocess.run(cmd, capture_output=True, text=True, env=env)
            latencies.append(time.perf_counter() - t0)
            self.assertEqual(res.returncode, 0, res.stderr)
            last_stdout = res.stdout

        min_latency = min(latencies)
        self.assertLess(
            min_latency,
            0.120,
            f"zelyr status minimum latency ({min_latency*1000:.2f}ms) exceeded 120ms target",
        )
        self.assertIn("CPU:", last_stdout)
        self.assertIn("RAM:", last_stdout)
        self.assertIn("Daemon:", last_stdout)
        self.assertIn("Optional Extras:", last_stdout)

    def test_tool_list_outputs_all_builtin_tools(self):
        """zelyr tool list must output all 10 built-in tools."""
        cmd = [sys.executable, "-m", "zelyr.cli.main", "tool", "list"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

        for tool_id in BUILTIN_TOOLS:
            self.assertIn(
                f"• {tool_id}:",
                res.stdout,
                f"Built-in tool '{tool_id}' missing from tool list output",
            )

    def test_tool_list_does_not_import_plugin_manager_or_watchdog(self):
        """Default zelyr tool list must not import PluginManager or watchdog."""
        code = (
            "import sys\n"
            "from zelyr.cli.main import main\n"
            "sys.argv = ['zelyr', 'tool', 'list']\n"
            "try:\n"
            "    main()\n"
            "except SystemExit:\n"
            "    pass\n"
            "assert 'watchdog' not in sys.modules, 'watchdog unexpectedly loaded'\n"
            "assert 'zelyr.core.plugin_manager' not in sys.modules, 'PluginManager unexpectedly loaded'\n"
        )
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(
            res.returncode,
            0,
            f"Default tool list imported heavy modules: {res.stdout}\n{res.stderr}",
        )

    def test_tool_list_all_flag_discovers_plugins(self):
        """zelyr tool list --all invokes PluginManager for custom plugins."""
        code = (
            "import sys\n"
            "from zelyr.cli.main import main\n"
            "sys.argv = ['zelyr', 'tool', 'list', '--all']\n"
            "try:\n"
            "    main()\n"
            "except SystemExit:\n"
            "    pass\n"
            "assert 'zelyr.core.plugin_manager' in sys.modules, 'PluginManager was not loaded with --all'\n"
        )
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"tool list --all failed: {res.stdout}\n{res.stderr}")

    def test_tool_registry_builtin_resolution_and_execution(self):
        """ToolRegistry can resolve and instantiate builtins on demand."""
        registry = ToolRegistry()
        builtins = registry.get_builtin_tools()
        self.assertEqual(len(builtins), 10)
        self.assertIn("echo", builtins)

        # Lazy instantiation of echo
        tool = registry.get_tool("echo")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.tool_id, "echo")

        result = registry.execute("echo", text="test_hello")
        self.assertTrue(result.success)
        self.assertEqual(result.output, "test_hello")

    def test_tool_run_echo_via_cli(self):
        """zelyr tool run echo executes cleanly from CLI."""
        cmd = [
            sys.executable,
            "-m",
            "zelyr.cli.main",
            "tool",
            "run",
            "echo",
            "--args",
            '{"text": "perf_test"}',
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("Result:", res.stdout)
        self.assertIn("perf_test", res.stdout)


if __name__ == "__main__":
    unittest.main()
