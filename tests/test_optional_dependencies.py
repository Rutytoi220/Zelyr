"""Targeted test battery for optional dependency decoupling and import guards.

Verifies that core modules, CLI entry points, and agents import cleanly without
requiring optional machine learning (`torch`, `transformers`) or browser (`playwright`)
dependencies, and that accessing optional features without their respective extra
produces clear, actionable error messages.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import importlib
import sys
import unittest
from typing import Generator

from zelyr.tool_registry import ToolRegistry


@contextmanager
def mask_modules(*module_names: str) -> Generator[None, None, None]:
    """Temporarily mask specified modules so any import attempts raise ImportError."""
    saved_modules = {}
    for name in module_names:
        saved_modules[name] = sys.modules.get(name)
        sys.modules[name] = None
        sub_prefix = f"{name}."
        for k in list(sys.modules.keys()):
            if k.startswith(sub_prefix):
                saved_modules[k] = sys.modules.pop(k)

    # Also evict cached vision_engine modules so fresh imports exercise the guards
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


class TestOptionalDependenciesCoreImports(unittest.TestCase):
    """Verify that core package surfaces import cleanly without ML or browser packages."""

    def test_core_package_imports_without_ml_and_browser(self):
        """Core modules must import without torch, transformers, or playwright."""
        with mask_modules(
            "torch",
            "torchvision",
            "transformers",
            "bitsandbytes",
            "accelerate",
            "playwright",
        ):
            # Test package root
            import zelyr

            self.assertTrue(hasattr(zelyr, "__version__"))

            # Test CLI dispatch
            import zelyr.cli.main as cli_main

            self.assertTrue(callable(cli_main.main))

            # Test native orchestrator
            import zelyr.agents.native_orchestrator as orchestrator

            self.assertTrue(hasattr(orchestrator, "NativeOrchestrator"))

            # Test core engine & registry
            import zelyr.core.engine as engine
            import zelyr.tool_registry as tr

            self.assertTrue(hasattr(engine, "Engine"))
            self.assertTrue(hasattr(tr, "ToolRegistry"))

            # Test memory store
            import zelyr.db.memory as memory

            self.assertTrue(callable(memory.get_db_path))

            # Test tools module and tool definitions
            from zelyr.tools.web_engine import PlaywrightWebTool
            from zelyr.tools.vision import InteractWithUITool
            from zelyr.tools.voice_engine import VoiceDictationEngine

            tool = PlaywrightWebTool()
            self.assertEqual(tool.name, "web_engine")

            ui_tool = InteractWithUITool()
            self.assertEqual(ui_tool.tool_id, "interact_with_ui")

            voice = VoiceDictationEngine()
            self.assertIsNone(voice.model)


class TestMLExtrasGuards(unittest.TestCase):
    """Verify that ML/VLM features raise actionable error messages when ml extra is missing."""

    def test_singleton_get_grounding_engine_raises_actionable_error(self):
        """get_grounding_engine() must raise actionable ImportError citing zelyr-core[ml]."""
        with mask_modules("torch", "transformers"):
            import zelyr.core.vision_engine.singleton as singleton

            singleton._engine_instance = None

            with self.assertRaises(ImportError) as ctx:
                singleton.get_grounding_engine()

            err_msg = str(ctx.exception)
            self.assertIn("torch", err_msg)
            self.assertIn("transformers", err_msg)
            self.assertIn("pip install 'zelyr-core[ml]'", err_msg)

    def test_evict_grounding_engine_safe_without_torch(self):
        """evict_grounding_engine() must degrade gracefully when torch is missing."""
        with mask_modules("torch"):
            import zelyr.core.vision_engine.singleton as singleton

            # Should not raise any exception
            singleton.evict_grounding_engine()

    def test_zelyr_engine_import_raises_actionable_error(self):
        """Direct import of zelyr_engine must raise actionable ImportError."""
        with mask_modules("torch", "transformers"):
            with self.assertRaises(ImportError) as ctx:
                importlib.import_module("zelyr.core.vision_engine.zelyr_engine")

            err_msg = str(ctx.exception)
            self.assertIn("pip install 'zelyr-core[ml]'", err_msg)

    def test_spotless_vlm_import_raises_actionable_error(self):
        """Direct import of spotless_vlm must raise actionable ImportError."""
        with mask_modules("torch", "transformers"):
            with self.assertRaises(ImportError) as ctx:
                importlib.import_module("zelyr.core.vision_engine.models.spotless_vlm")

            err_msg = str(ctx.exception)
            self.assertIn("pip install 'zelyr-core[ml]'", err_msg)


class TestBrowserExtrasGuards(unittest.TestCase):
    """Verify that browser automation raises actionable error messages when browser extra is missing."""

    def test_playwright_web_tool_call_raises_actionable_error(self):
        """Calling PlaywrightWebTool must raise actionable ImportError citing zelyr-core[browser]."""
        with mask_modules("playwright", "playwright.async_api"):
            from zelyr.tools.web_engine import PlaywrightWebTool

            tool = PlaywrightWebTool()
            with self.assertRaises(ImportError) as ctx:
                asyncio.run(tool(action="navigate", url="https://example.com"))

            err_msg = str(ctx.exception)
            self.assertIn("playwright", err_msg)
            self.assertIn("pip install 'zelyr-core[browser]'", err_msg)

    def test_playwright_web_tool_execute_raises_actionable_error(self):
        """Executing PlaywrightWebTool via execute must raise actionable ImportError."""
        with mask_modules("playwright", "playwright.async_api"):
            from zelyr.tools.web_engine import PlaywrightWebTool

            tool = PlaywrightWebTool()
            with self.assertRaises(ImportError) as ctx:
                asyncio.run(tool.execute({"action": "navigate", "url": "https://example.com"}))

            err_msg = str(ctx.exception)
            self.assertIn("pip install 'zelyr-core[browser]'", err_msg)

    def test_tool_registry_execution_captures_actionable_error(self):
        """Executing PlaywrightWebTool via ToolRegistry returns ToolResult with error message."""
        with mask_modules("playwright", "playwright.async_api"):
            from zelyr.tools.web_engine import PlaywrightWebTool

            registry = ToolRegistry()
            tool = PlaywrightWebTool()
            registry.register_tool("web_engine", tool)

            result = asyncio.run(
                registry.execute_async("web_engine", action="navigate", url="https://example.com")
            )
            self.assertFalse(result.success)
            self.assertIn("pip install 'zelyr-core[browser]'", result.error)


class TestVoiceExtrasGuards(unittest.TestCase):
    """Verify that voice engine features raise actionable error messages when dependencies are missing."""

    def test_voice_dictation_load_model_raises_actionable_error(self):
        """VoiceDictationEngine._load_model() must raise actionable ImportError citing zelyr-core[ml,audio]."""
        with mask_modules("torch", "faster_whisper"):
            from zelyr.tools.voice_engine import VoiceDictationEngine

            engine = VoiceDictationEngine()
            with self.assertRaises(ImportError) as ctx:
                engine._load_model()

            err_msg = str(ctx.exception)
            self.assertIn("pip install 'zelyr-core[ml,audio]'", err_msg)


if __name__ == "__main__":
    unittest.main()
