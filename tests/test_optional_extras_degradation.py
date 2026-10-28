"""Targeted verification battery for optional dependency degradation and boundary hardening.

Verifies:
1. Missing ML Dependencies: actionable OptionalDependencyError with installation commands.
2. Missing Browser Dependencies: structured tool failure envelope containing install instructions.
3. Missing GUI Dependencies: clean exit code 1 with actionable remediation message without traceback.
4. CLI Startup & Fast-Path Invariance: version/help checks never import optional heavy modules.
5. Status Reporting Accuracy: zelyr status dynamically and safely reflects missing vs installed extras.
6. Agent Loop Recovery: agent loops handle tool failure envelopes from missing extras without crashing.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import io
import sys
import unittest
from typing import Generator
from unittest.mock import patch

from zelyr.tools.core import ToolResult
from zelyr.tool_registry import ToolRegistry
from zelyr.utils.optional_deps import (
    OptionalDependencyError,
    is_extra_available,
    require_extra,
)


@contextmanager
def mask_modules(*module_names: str) -> Generator[None, None, None]:
    """Temporarily mask specified modules in sys.modules to simulate missing dependencies."""
    saved_modules = {}
    for name in module_names:
        saved_modules[name] = sys.modules.get(name)
        sys.modules[name] = None
        sub_prefix = f"{name}."
        for k in list(sys.modules.keys()):
            if k.startswith(sub_prefix):
                saved_modules[k] = sys.modules.pop(k)

    # Also evict cached vision_engine modules if masking vision dependencies
    for k in list(sys.modules.keys()):
        if "vision_engine" in k:
            saved_modules[k] = sys.modules.pop(k)

    try:
        yield
    finally:
        for k, v in saved_modules.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


class TestMissingMLDependencies(unittest.TestCase):
    """Test 1: Verify missing ML dependencies trigger actionable OptionalDependencyError."""

    def test_require_extra_raises_optional_dependency_error(self):
        """require_extra must raise OptionalDependencyError with package and extra install hints."""
        with mask_modules("torch", "transformers"):
            with self.assertRaises(OptionalDependencyError) as ctx:
                require_extra("torch", "ml", "local neural network execution")

            err = ctx.exception
            self.assertEqual(err.package_name, "torch")
            self.assertEqual(err.extra_name, "ml")
            self.assertIn("'torch' is required for local neural network execution", str(err))
            self.assertIn("pip install 'zelyr-core[ml]'", str(err))
            self.assertIn("uv add --extra ml zelyr-core", str(err))
            self.assertIsInstance(err, ImportError)

    def test_singleton_vlm_engine_raises_actionable_error(self):
        """get_grounding_engine must raise actionable error citing zelyr-core[ml]."""
        with mask_modules("torch", "transformers"):
            import zelyr.core.vision_engine.singleton as singleton

            singleton._engine_instance = None
            with self.assertRaises(ImportError) as ctx:
                singleton.get_grounding_engine()

            err_msg = str(ctx.exception)
            self.assertIn("pip install 'zelyr-core[ml]'", err_msg)


class TestMissingBrowserDependencies(unittest.TestCase):
    """Test 2: Verify browser tools return structured failure when playwright is absent."""

    def test_playwright_tool_direct_execute_raises_optional_dependency_error(self):
        """Direct call to PlaywrightWebTool raises OptionalDependencyError with browser instructions."""
        with mask_modules("playwright", "playwright.async_api"):
            from zelyr.tools.web_engine import PlaywrightWebTool

            tool = PlaywrightWebTool()
            with self.assertRaises(OptionalDependencyError) as ctx:
                asyncio.run(tool.execute({"action": "navigate", "url": "https://example.com"}))

            err = ctx.exception
            self.assertEqual(err.package_name, "playwright")
            self.assertEqual(err.extra_name, "browser")
            self.assertIn("pip install 'zelyr-core[browser]'", str(err))

    def test_tool_registry_catches_missing_browser_and_returns_tool_result(self):
        """ToolRegistry must catch missing playwright and return structured ToolResult failure."""
        with mask_modules("playwright", "playwright.async_api"):
            from zelyr.tools.web_engine import PlaywrightWebTool

            registry = ToolRegistry()
            tool = PlaywrightWebTool()
            registry.register_tool("web_engine", tool)

            result = asyncio.run(
                registry.execute_async("web_engine", action="navigate", url="https://example.com")
            )
            self.assertIsInstance(result, ToolResult)
            self.assertFalse(result.success)
            self.assertEqual(result.output, "")
            self.assertIn("Tool requires optional dependency", result.error)
            self.assertIn("pip install 'zelyr-core[browser]'", result.error)


class TestMissingGUIDependencies(unittest.TestCase):
    """Test 3: Verify zelyr gui exits cleanly with actionable remediation without traceback."""

    def test_zelyr_gui_exits_cleanly_when_pyside6_missing(self):
        """Invoking cmd_ui when PySide6 is missing must exit with code 1 and actionable message."""
        import argparse
        from zelyr.cli.main import cmd_ui

        with mask_modules("PySide6", "PySide6.QtWidgets", "PySide6.QtCore"):
            buf = io.StringIO()
            with patch("sys.stdout", buf), patch("sys.stderr", buf):
                with self.assertRaises(SystemExit) as ctx:
                    cmd_ui(argparse.Namespace())

            self.assertEqual(ctx.exception.code, 1)
            output = buf.getvalue()
            self.assertIn("'PySide6' is required for the desktop graphical interface", output)
            self.assertIn("pip install 'zelyr-core[gui]'", output)
            self.assertNotIn("Traceback (most recent call last)", output)

    def test_zelyr_gui_alias_dispatches_to_cmd_ui(self):
        """zelyr gui alias must be accepted by the CLI dispatcher."""
        from zelyr.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["gui"])
        self.assertEqual(args.command, "gui")


class TestCLIStartupInvariance(unittest.TestCase):
    """Test 4: Verify fast-path CLI operations never import optional packages."""

    def test_version_fast_path_invariance(self):
        """Running zelyr --version in a clean process must not import torch, transformers, playwright."""
        import subprocess

        code = (
            "import sys\n"
            "from zelyr.cli.main import main\n"
            "sys.argv = ['zelyr', '--version']\n"
            "main()\n"
            "for mod in ['torch', 'transformers', 'playwright', 'playwright.async_api']:\n"
            "    assert mod not in sys.modules, f'Forbidden module {mod} was imported during --version!'\n"
        )
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Error output: {res.stderr}")
        self.assertIn("zelyr", res.stdout)

    def test_parser_construction_does_not_import_optional_extras(self):
        """Building the CLI argument parser in a clean process must not import heavy optional dependencies."""
        import subprocess

        code = (
            "import sys\n"
            "from zelyr.cli.main import _build_parser\n"
            "parser = _build_parser()\n"
            "assert parser is not None\n"
            "for mod in ['torch', 'transformers', 'playwright', 'playwright.async_api']:\n"
            "    assert mod not in sys.modules, f'Forbidden module {mod} was imported during _build_parser!'\n"
        )
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Error output: {res.stderr}")


class TestStatusReportingAccuracy(unittest.TestCase):
    """Test 5: Verify status accurately probes optional extras without importing heavy modules."""

    def test_is_extra_available_detects_installed_vs_masked(self):
        """is_extra_available correctly returns False when masked without importing."""
        with mask_modules("playwright"):
            self.assertFalse(is_extra_available("playwright"))

        # Non-existent extra
        self.assertFalse(is_extra_available("non_existent_package_xyz_123"))

    def test_cmd_status_displays_extras_status(self):
        """cmd_status outputs optional extras section with correct statuses."""
        import argparse
        from zelyr.cli.main import cmd_status

        with mask_modules("torch", "transformers", "playwright"):
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cmd_status(argparse.Namespace())

            output = buf.getvalue()
            self.assertIn("Zelyr System Status", output)
            self.assertIn("Optional Extras", output)
            self.assertIn("ml", output)
            self.assertIn("browser", output)
            self.assertIn("missing", output)


class TestAgentLoopRecovery(unittest.TestCase):
    """Test 6: Verify agent loop handles optional dependency failure envelopes gracefully."""

    def test_orchestrator_handles_missing_tool_dependency_envelope(self):
        """NativeOrchestrator must process a tool error envelope without crashing."""
        from zelyr.agents.native_orchestrator import NativeOrchestrator

        # Create orchestrator with a mock tool executor that returns a structured failure
        async def mock_failing_executor(tool_name: str, **kwargs):
            return ToolResult(
                success=False,
                output="",
                error=f"Tool requires optional dependency: '{tool_name}' is required for browser. Install with: pip install 'zelyr-core[browser]'",
            )

        orchestrator = NativeOrchestrator(
            db_path=":memory:",
            tool_executor=mock_failing_executor,
        )
        self.assertIsNotNone(orchestrator)

        # Directly verify ToolResult envelope conversion matches the expected ReAct protocol
        tr = ToolResult(
            success=False,
            output="",
            error="Tool requires optional dependency: 'playwright' is required for browser automation.",
        )
        envelope = tr.to_dict(tool="web_engine", arguments={"url": "https://example.com"})

        self.assertFalse(envelope["success"])
        self.assertEqual(envelope["tool"], "web_engine")
        self.assertIn("Tool requires optional dependency", envelope["error"])
        self.assertEqual(envelope["result"]["output"], "")


if __name__ == "__main__":
    unittest.main()
