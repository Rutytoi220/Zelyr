"""Test suite for REPL /doctor diagnostic health suite and /undo command."""

import asyncio
import io
import shutil
import tempfile
import unittest
from pathlib import Path

from rich.console import Console

import zelyr.cli.repl as repl_module
from zelyr.cli.repl import InlineRepl
from zelyr.tools.workspace_file import ManageWorkspaceFileTool


class TestReplDoctorAndUndo(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.repl = InlineRepl()
        self.string_io = io.StringIO()
        # Redirect the REPL console for the duration of this test only. The
        # module-level console is shared global state, so it must be restored
        # in tearDown() or every later test in the session loses REPL output.
        self._original_console = repl_module.console
        repl_module.console = Console(file=self.string_io, force_terminal=False, color_system=None)

    def tearDown(self):
        repl_module.console = self._original_console
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_repl_doctor_command(self):
        handled = self.repl.handle_command("/doctor")
        self.assertTrue(handled)
        output = self.string_io.getvalue()

        self.assertIn("Zelyr Subsystem Health Diagnostics", output)
        self.assertIn("Ollama Engine", output)
        self.assertIn("Compositor", output)
        self.assertIn("Clipboard", output)
        self.assertIn("Media (MPRIS)", output)
        self.assertIn("Browser Bridge", output)
        self.assertIn("Tool Registry", output)

    def test_repl_undo_with_explicit_path(self):
        test_file = Path(self.test_dir) / "doc.txt"
        tool = ManageWorkspaceFileTool()

        asyncio.run(tool.execute({
            "action": "write",
            "path": str(test_file),
            "content": "Original Content",
        }))

        asyncio.run(tool.execute({
            "action": "write",
            "path": str(test_file),
            "content": "Modified Content",
        }))

        self.assertEqual(test_file.read_text(encoding="utf-8"), "Modified Content")

        handled = self.repl.handle_command(f"/undo {test_file}")
        self.assertTrue(handled)
        self.assertEqual(test_file.read_text(encoding="utf-8"), "Original Content")
        output = self.string_io.getvalue()
        self.assertIn("Rollback Successful", output)

    def test_repl_undo_with_implicit_session_path(self):
        test_file = Path(self.test_dir) / "implicit_doc.txt"
        tool = ManageWorkspaceFileTool()

        asyncio.run(tool.execute({
            "action": "write",
            "path": str(test_file),
            "content": "Before Change",
        }))
        asyncio.run(tool.execute({
            "action": "write",
            "path": str(test_file),
            "content": "After Change",
        }))

        handled = self.repl.handle_command("/undo")
        self.assertTrue(handled)
        self.assertEqual(test_file.read_text(encoding="utf-8"), "Before Change")
        output = self.string_io.getvalue()
        self.assertIn("Rollback Successful", output)

    def test_repl_bridge_lifecycle_and_gc_retention(self):
        async def run_lifecycle():
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                free_port = s.getsockname()[1]

            from zelyr.tools.browser_extension import BrowserExtensionBridge, get_bridge
            BrowserExtensionBridge.reset_instance()
            BrowserExtensionBridge.get_instance(port=free_port)
            repl = InlineRepl(session_id="test_bridge_binding_session")
            self.assertIsNone(repl.bridge)
            self.assertIsNone(repl._bridge_server)

            await repl.start_services()

            self.assertIsNotNone(repl.bridge)
            self.assertIsNotNone(repl.bridge.server)
            self.assertIsNotNone(repl._bridge_server)
            self.assertIs(repl._bridge_server, repl.bridge.server)
            self.assertIsNotNone(get_bridge().server)
            self.assertTrue(get_bridge().server.is_serving())

            ws_uri = f"ws://{repl.bridge.host}:{repl.bridge.port}"
            import websockets
            async with websockets.connect(ws_uri):
                await asyncio.sleep(0.05)
                self.assertTrue(repl.bridge.is_connected())

            await repl.do_exit()
            self.assertIsNone(repl._bridge_server)
            self.assertIsNone(repl.bridge.server)
            self.assertFalse(repl.bridge.is_connected())

            BrowserExtensionBridge.reset_instance()

        asyncio.run(run_lifecycle())


if __name__ == "__main__":
    unittest.main()
