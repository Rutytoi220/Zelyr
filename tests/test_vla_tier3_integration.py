"""Deterministic offline test suite for Spotless-Arms Tier 3 VLA integration.

Covers:
1. Enabled integration: caching, cleanup, telemetry structure, and Ollama keep_alive=0 eviction.
2. Environment opt-out (ZELYR_USE_SPOTLESS_ARMS) & strict cache invalidation.
3. Missing optional package (sys.modules["spotless_arms"] = None) without ValueError and with legacy fallback.
4. Broken installation propagation (transitive ModuleNotFoundError(name="some_missing_subdep") is re-raised).
5. Browser isolation & tool authorization boundaries (Tier 2 negative gating, web-element rejection, ring/allowlist auth).
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as _np  # noqa: F401 — ensure C-extension is in sys.modules before patch.dict snapshots
from PIL import Image

import zelyr.core.vision_engine.singleton as singleton
from zelyr.agents.actuator_arbiter import ActuatorArbiter
from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.core.plugins import filter_tool_schemas_by_tier
from zelyr.tool_registry import ToolRegistry
from zelyr.tools.core import ToolResult
from zelyr.tools.vision import InteractWithUITool, trusted_ui_execution_context


def _build_fake_spotless_modules(
    mock_vla_cls: Any,
    mock_runner_cls: Any,
    mock_buf_cls: Any,
    mock_arm_cls: Any,
) -> dict[str, ModuleType]:
    """Construct synthetic in-memory spotless_arms modules for GPU-free offline testing."""
    pkg = ModuleType("spotless_arms")
    models_pkg = ModuleType("spotless_arms.models")
    latent_mod = ModuleType("spotless_arms.models.latent_core")
    motor_mod = ModuleType("spotless_arms.models.motor_head")
    motor_mod.SpotlessVLA = mock_vla_cls  # type: ignore[attr-defined]

    act_pkg = ModuleType("spotless_arms.actuators")
    arm_mod = ModuleType("spotless_arms.actuators.uinput_arm")
    arm_mod.LinuxVirtualArm = mock_arm_cls  # type: ignore[attr-defined]

    pipe_pkg = ModuleType("spotless_arms.pipeline")
    runner_mod = ModuleType("spotless_arms.pipeline.pipelined_runner")
    runner_mod.PipelinedVLARunner = mock_runner_cls  # type: ignore[attr-defined]
    runner_mod.AsyncScreenBuffer = mock_buf_cls  # type: ignore[attr-defined]

    return {
        "spotless_arms": pkg,
        "spotless_arms.models": models_pkg,
        "spotless_arms.models.latent_core": latent_mod,
        "spotless_arms.models.motor_head": motor_mod,
        "spotless_arms.actuators": act_pkg,
        "spotless_arms.actuators.uinput_arm": arm_mod,
        "spotless_arms.pipeline": pipe_pkg,
        "spotless_arms.pipeline.pipelined_runner": runner_mod,
    }


class TestVLATier3Integration(unittest.IsolatedAsyncioTestCase):
    """Offline tests for optional Spotless-Arms boundary, cache invalidation, and Tier 3 safety."""

    def setUp(self) -> None:
        singleton._vla_instance = None
        singleton._engine_instance = None
        singleton._vla_active_count = 0
        self._orig_env = os.environ.get("ZELYR_USE_SPOTLESS_ARMS")
        os.environ.pop("ZELYR_USE_SPOTLESS_ARMS", None)

    def tearDown(self) -> None:
        singleton._vla_instance = None
        singleton._engine_instance = None
        singleton._vla_active_count = 0
        if self._orig_env is None:
            os.environ.pop("ZELYR_USE_SPOTLESS_ARMS", None)
        else:
            os.environ["ZELYR_USE_SPOTLESS_ARMS"] = self._orig_env

    def _make_mock_spotless_stack(self):
        mock_vla_instance = MagicMock(name="SpotlessVLAInstance")
        mock_vla_cls = MagicMock(name="SpotlessVLA")
        mock_vla_cls.from_pretrained.return_value = mock_vla_instance

        mock_buf_cm = MagicMock(name="AsyncScreenBufferCM")
        mock_buf_cls = MagicMock(name="AsyncScreenBuffer")
        mock_buf_cls.return_value.__enter__.return_value = mock_buf_cm

        mock_arm_cm = MagicMock(name="LinuxVirtualArmCM")
        mock_arm_cm.backend = "dry_run"
        mock_arm_cls = MagicMock(name="LinuxVirtualArm")
        mock_arm_cls.return_value.__enter__.return_value = mock_arm_cm

        sample_telemetry = [
            {
                "tick": 1,
                "k_steps": 2,
                "action": {
                    "px": 1440,
                    "py": 900,
                    "x_norm": 0.48,
                    "y_norm": 0.4688,
                    "clicked": "left",
                    "backend": "dry_run",
                },
                "halted_reason": "click:left",
            }
        ]
        mock_runner_instance = MagicMock(name="PipelinedVLARunnerInstance")
        mock_runner_instance.run.return_value = sample_telemetry
        mock_runner_instance.last_summary = {
            "ticks_executed": 1,
            "converged": True,
            "converged_tick": 1,
            "jitter_px": 0.0,
        }
        mock_runner_cls = MagicMock(name="PipelinedVLARunner", return_value=mock_runner_instance)

        fake_mods = _build_fake_spotless_modules(
            mock_vla_cls, mock_runner_cls, mock_buf_cls, mock_arm_cls
        )
        return (
            fake_mods,
            mock_vla_cls,
            mock_vla_instance,
            mock_runner_cls,
            mock_runner_instance,
            mock_buf_cls,
            mock_arm_cls,
        )

    # ------------------------------------------------------------------
    # 1. Enabled Integration
    # ------------------------------------------------------------------
    async def test_enabled_engine_caching_eviction_and_ollama_keep_alive_zero(self) -> None:
        (
            fake_mods,
            mock_vla_cls,
            mock_vla_instance,
            mock_runner_cls,
            mock_runner_instance,
            mock_buf_cls,
            mock_arm_cls,
        ) = self._make_mock_spotless_stack()

        mock_http_client = MagicMock()
        mock_ps_resp = MagicMock()
        mock_ps_resp.status_code = 200
        mock_ps_resp.json.return_value = {"models": [{"name": "qwen2.5-coder:7b"}]}
        mock_http_client.get.return_value = mock_ps_resp

        mock_http_cm = MagicMock()
        mock_http_cm.__enter__.return_value = mock_http_client

        mock_cfg = SimpleNamespace(
            ollama_base_url="http://127.0.0.1:11434",
            ollama_model="qwen3:8b",
        )

        with (
            patch.dict(sys.modules, fake_mods),
            patch("httpx.Client", return_value=mock_http_cm),
            patch("zelyr.config.get_config", return_value=mock_cfg),
            patch("time.sleep"),
        ):
            self.assertTrue(singleton.is_spotless_arms_available())

            # First call initializes and caches engine; without established Ollama ownership,
            # pre-existing or unrelated models in /api/ps are NEVER unloaded.
            with self.assertLogs(singleton.logger, level="WARNING") as cm:
                eng1 = singleton.get_vla_engine(device="cpu")
            eng2 = singleton.get_vla_engine(device="cpu")
            self.assertIs(eng1, mock_vla_instance)
            self.assertIs(eng2, mock_vla_instance)
            self.assertEqual(mock_vla_cls.from_pretrained.call_count, 1)

            mock_http_client.get.assert_called_once_with("http://127.0.0.1:11434/api/ps", timeout=2.0)
            self.assertEqual(mock_http_client.post.call_count, 0)
            self.assertTrue(
                any("safe Zelyr session ownership cannot be established" in msg for msg in cm.output)
            )

            # Verify run_reflex_action telemetry contract
            telemetry = singleton.run_reflex_action(
                instruction="Click Confirm",
                max_ticks=4,
                move_only=True,
                dry_run_arm=True,
            )
            self.assertEqual(telemetry["status"], "ok")
            self.assertEqual(telemetry["instruction"], "Click Confirm")
            self.assertEqual(telemetry["ticks"], 1)
            self.assertEqual(telemetry["k_schedule"], [2])
            self.assertEqual(telemetry["final_xy"], [1440, 900])
            self.assertEqual(telemetry["clicked"], "left")
            self.assertTrue(telemetry["converged"])
            self.assertEqual(telemetry["halted_reason"], "click:left")
            self.assertTrue(telemetry["move_only"])
            mock_runner_instance.run.assert_called_once_with(
                prompt="Click Confirm",
                max_ticks=4,
                move_only=True,
                verbose=False,
            )

            # Verify evict_vla_engine cleans up and clears cache
            singleton.evict_vla_engine()
            self.assertIsNone(singleton._vla_instance)
            mock_vla_instance.unload.assert_called_once()

    # ------------------------------------------------------------------
    # 2. Environment Opt-Out & Cache Invalidation
    # ------------------------------------------------------------------
    async def test_env_opt_out_invalidates_cached_engine_and_falls_back(self) -> None:
        (
            fake_mods,
            mock_vla_cls,
            mock_vla_instance,
            mock_runner_cls,
            _,
            _,
            _,
        ) = self._make_mock_spotless_stack()

        with (
            patch.dict(sys.modules, fake_mods),
            patch.object(singleton, "_evict_ollama_vram"),
        ):
            # Step 1: Initialize and cache engine while enabled
            cached = singleton.get_vla_engine(device="cpu")
            self.assertIs(cached, mock_vla_instance)
            self.assertIs(singleton._vla_instance, mock_vla_instance)

            # Step 2: Set ZELYR_USE_SPOTLESS_ARMS=0 (and test case-insensitive variants)
            for disabled_val in ("0", "false", "FALSE", "No", "OFF"):
                os.environ["ZELYR_USE_SPOTLESS_ARMS"] = disabled_val
                self.assertFalse(singleton.is_spotless_arms_available())

            os.environ["ZELYR_USE_SPOTLESS_ARMS"] = "0"
            # Cached instance must already be evicted and must not be returned or used
            self.assertIsNone(singleton._vla_instance)
            mock_vla_instance.unload.assert_called_once()

            with self.assertRaises(RuntimeError):
                singleton.get_vla_engine(device="cpu")
            self.assertIsNone(singleton._vla_instance)

            with self.assertRaises(RuntimeError):
                singleton.run_reflex_action("Click Submit", move_only=True, dry_run_arm=True)
            mock_runner_cls.assert_not_called()

            # Step 3: InteractWithUITool must not use cached VLA engine and must use legacy fallback
            tool = InteractWithUITool()
            buf = io.BytesIO()
            Image.new("RGB", (64, 64), color="black").save(buf, format="PNG")
            fake_png = buf.getvalue()

            mock_legacy_engine = MagicMock()
            mock_legacy_engine.predict_action.return_value = {
                "action": "unknown",
                "point": [0, 0],
                "raw_output": "unparsed",
            }

            with (
                patch.object(tool, "_capture_wayland_grim", AsyncMock(return_value=fake_png)),
                patch("zelyr.tools.vision.get_grounding_engine", return_value=mock_legacy_engine),
                patch("httpx.AsyncClient"),
                patch("asyncio.sleep", AsyncMock()),
                trusted_ui_execution_context(caller_ring=0, allowed_tools=["interact_with_ui"]),
            ):
                res = await tool.execute({"instruction": "Click Submit"})

            self.assertIsInstance(res, ToolResult)
            self.assertFalse(res.success)
            self.assertIn("Engine failed to parse action from raw output", res.error)
            mock_runner_cls.assert_not_called()

            # Step 4: Re-enabling ZELYR_USE_SPOTLESS_ARMS=1 must initialize a fresh instance
            os.environ["ZELYR_USE_SPOTLESS_ARMS"] = "1"
            self.assertTrue(singleton.is_spotless_arms_available())
            self.assertIsNone(singleton._vla_instance)
            fresh = singleton.get_vla_engine(device="cpu")
            self.assertIs(fresh, mock_vla_instance)
            self.assertEqual(mock_vla_cls.from_pretrained.call_count, 2)

    # ------------------------------------------------------------------
    # 3. Missing Optional Package (sys.modules["spotless_arms"] = None)
    # ------------------------------------------------------------------
    async def test_missing_optional_package_masked_in_sys_modules_degrades_safely(self) -> None:
        masked = {name: None for name in singleton._SPOTLESS_REQUIRED_MODULES}

        with patch.dict(sys.modules, masked):
            # Must return False cleanly without raising ValueError from find_spec
            self.assertFalse(singleton.is_spotless_arms_available())

            tool = InteractWithUITool()
            buf = io.BytesIO()
            Image.new("RGB", (64, 64), color="black").save(buf, format="PNG")
            fake_png = buf.getvalue()

            mock_legacy_engine = MagicMock()
            mock_legacy_engine.predict_action.return_value = {
                "action": "click",
                "point": [200, 100],
                "text": None,
            }
            mock_proc = MagicMock()
            mock_proc.wait = AsyncMock(return_value=0)

            with (
                patch.object(tool, "_capture_wayland_grim", AsyncMock(return_value=fake_png)),
                patch("zelyr.tools.vision.get_grounding_engine", return_value=mock_legacy_engine),
                patch("zelyr.tools.vision.get_hyprland_scale", AsyncMock(return_value=1.0)),
                patch("asyncio.create_subprocess_shell", AsyncMock(return_value=mock_proc)),
                patch("httpx.AsyncClient"),
                patch("asyncio.sleep", AsyncMock()),
                trusted_ui_execution_context(caller_ring=0, allowed_tools=["interact_with_ui"]),
            ):
                res = await tool.execute({"instruction": "Click OK"})

            self.assertTrue(res.success)
            self.assertIn("Action 'click' executed at coordinates [200, 100]", res.output)

    # ------------------------------------------------------------------
    # 4. Broken Installation (Missing Internal / Transitive Dependency)
    # ------------------------------------------------------------------
    async def test_broken_installation_missing_subdependency_propagates(self) -> None:
        def _broken_import(name: str, package: str | None = None):
            if name.startswith("spotless_arms"):
                raise ModuleNotFoundError(
                    "No module named 'some_missing_subdep'",
                    name="some_missing_subdep",
                )
            return importlib.__import__(name)

        import importlib

        with (
            patch.dict(sys.modules, {"spotless_arms": ModuleType("spotless_arms")}),
            patch("importlib.import_module", side_effect=_broken_import),
        ):
            with self.assertRaises(ModuleNotFoundError) as ctx1:
                singleton.is_spotless_arms_available()
            self.assertEqual(ctx1.exception.name, "some_missing_subdep")

            with self.assertRaises(ModuleNotFoundError) as ctx2:
                singleton.get_vla_engine()
            self.assertEqual(ctx2.exception.name, "some_missing_subdep")

            with self.assertRaises(ModuleNotFoundError) as ctx3:
                singleton.run_reflex_action("Click Save")
            self.assertEqual(ctx3.exception.name, "some_missing_subdep")

            tool = InteractWithUITool()
            with (
                trusted_ui_execution_context(caller_ring=0, allowed_tools=["interact_with_ui"]),
                self.assertRaises(ModuleNotFoundError) as ctx4,
            ):
                await tool.execute({"instruction": "Click Save"})
            self.assertEqual(ctx4.exception.name, "some_missing_subdep")

    # ------------------------------------------------------------------
    # 5. Browser Isolation & Authorization Boundary
    # ------------------------------------------------------------------
    async def test_browser_isolation_and_authorization_prevent_tier3_reflex(self) -> None:
        (
            fake_mods,
            _,
            _,
            mock_runner_cls,
            _,
            _,
            _,
        ) = self._make_mock_spotless_stack()

        all_schemas = [
            {"type": "function", "function": {"name": "execute_command", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "interact_with_browser", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "interact_with_ui", "parameters": {"type": "object", "properties": {}}}},
        ]

        with (
            patch.dict(sys.modules, fake_mods),
            patch("zelyr.core.plugins.load_plugins"),
        ):
            # 5a. Tier 2 browser focus + bridge connected routes to tier2_browser and purges interact_with_ui
            browser_window = {"class": "app.zen_browser.zen", "title": "Zen Browser"}
            tier = ActuatorArbiter.arbitrate(
                task="Click the checkout button",
                active_window=browser_window,
                bridge_connected=True,
            )
            self.assertEqual(tier, "tier2_browser")
            filtered = filter_tool_schemas_by_tier(all_schemas, tier, is_browser=True)
            names = {s.get("function", {}).get("name") or s.get("name") for s in filtered}
            self.assertNotIn("interact_with_ui", names)

            # 5b. NativeOrchestrator.generate_stream purges interact_with_ui when browser is active
            orch = NativeOrchestrator(db_path=":memory:")
            payload = {
                "messages": [{"role": "user", "content": "Click submit"}],
                "tools": list(all_schemas),
            }

            @contextlib.asynccontextmanager
            async def _mock_stream(method, url, **kwargs):
                resp = AsyncMock()
                resp.status_code = 200

                async def _lines():
                    yield 'data: {"choices": [{"delta": {"content": "Done."}}]}'
                    yield "data: [DONE]"

                resp.aiter_lines = _lines
                yield resp

            with (
                patch.object(orch, "get_active_window", AsyncMock(return_value=browser_window)),
                patch("zelyr.tools.browser_extension.get_bridge") as mock_bridge_fn,
                patch("httpx.AsyncClient") as mock_http_cls,
            ):
                mock_bridge_fn.return_value.is_connected.return_value = True
                mock_http_cls.return_value.__aenter__.return_value.stream = _mock_stream
                mock_http_cls.return_value.__aenter__.return_value.post = AsyncMock()

                async for _ in orch.generate_stream(payload):
                    pass

            exposed_names = {
                t.get("function", {}).get("name") or t.get("name")
                for t in payload.get("tools", [])
            }
            self.assertNotIn("interact_with_ui", exposed_names)

            # 5c. Direct web-element calls to InteractWithUITool are rejected before run_reflex_action
            tool = InteractWithUITool()
            r1 = await tool.execute({"instruction": "Click submit", "element_id": 12})
            self.assertFalse(r1.success)
            self.assertIn("interact_with_browser", r1.error)

            r2 = await tool.execute({"instruction": "Click element #7 on page"})
            self.assertFalse(r2.success)
            self.assertIn("interact_with_browser", r2.error)
            mock_runner_cls.assert_not_called()

            # 5d. ToolRegistry authorization boundary blocks unauthorized callers before tool.execute
            registry = ToolRegistry()
            registry.register_tool("interact_with_ui", tool)
            denied = await registry.execute_async(
                "interact_with_ui",
                allowed_tools={"echo"},
                instruction="Click Submit",
            )
            self.assertFalse(denied.success)
            mock_runner_cls.assert_not_called()

    # ------------------------------------------------------------------
    # 6. Repeated Availability Checks & Single-Pass Component Import
    # ------------------------------------------------------------------
    async def test_repeated_availability_checks_do_not_initialize_engine(self) -> None:
        (
            fake_mods,
            mock_vla_cls,
            _,
            _,
            _,
            _,
            _,
        ) = self._make_mock_spotless_stack()

        with (
            patch.dict(sys.modules, fake_mods),
            patch.object(singleton, "_evict_ollama_vram") as mock_evict_ollama,
        ):
            for _ in range(5):
                self.assertTrue(singleton.is_spotless_arms_available())

            self.assertIsNone(singleton._vla_instance)
            mock_vla_cls.from_pretrained.assert_not_called()
            mock_evict_ollama.assert_not_called()

            # Verify run_reflex_action imports components once without duplicate import in get_vla_engine
            with patch.object(
                singleton,
                "_import_spotless_components",
                wraps=singleton._import_spotless_components,
            ) as spy_import:
                singleton.run_reflex_action("Click Confirm", move_only=True, dry_run_arm=True)
                self.assertEqual(spy_import.call_count, 1)

    # ------------------------------------------------------------------
    # 7. Missing Optional Submodule & Corrupted Submodule Symbol
    # ------------------------------------------------------------------
    async def test_missing_optional_submodule_degrades_safely(self) -> None:
        (
            fake_mods,
            _,
            mock_vla_instance,
            mock_runner_cls,
            _,
            _,
            _,
        ) = self._make_mock_spotless_stack()

        # Case A: Top-level package exists, but submodule spotless_arms.pipeline.pipelined_runner is None
        partial_mods = dict(fake_mods)
        partial_mods["spotless_arms.pipeline.pipelined_runner"] = None  # type: ignore[assignment]
        singleton._vla_instance = mock_vla_instance

        with patch.dict(sys.modules, partial_mods):
            self.assertFalse(singleton.is_spotless_arms_available())
            self.assertIsNone(singleton._vla_instance)
            mock_vla_instance.unload.assert_called_once()

            tool = InteractWithUITool()
            buf = io.BytesIO()
            Image.new("RGB", (64, 64), color="black").save(buf, format="PNG")
            fake_png = buf.getvalue()

            mock_legacy_engine = MagicMock()
            mock_legacy_engine.predict_action.return_value = {
                "action": "hover",
                "point": [300, 150],
                "text": None,
            }
            mock_proc = MagicMock()
            mock_proc.wait = AsyncMock(return_value=0)

            with (
                patch.object(tool, "_capture_wayland_grim", AsyncMock(return_value=fake_png)),
                patch("zelyr.tools.vision.get_grounding_engine", return_value=mock_legacy_engine),
                patch("zelyr.tools.vision.get_hyprland_scale", AsyncMock(return_value=1.0)),
                patch("asyncio.create_subprocess_shell", AsyncMock(return_value=mock_proc)),
                patch("httpx.AsyncClient"),
                patch("asyncio.sleep", AsyncMock()),
                trusted_ui_execution_context(caller_ring=0, allowed_tools=["interact_with_ui"]),
            ):
                res = await tool.execute({"instruction": "Hover over menu"})

            self.assertTrue(res.success)
            self.assertIn("Action 'hover' executed at coordinates [300, 150]", res.output)
            mock_runner_cls.assert_not_called()

        # Case B: Submodule exists in sys.modules but is missing required symbol (corrupted install)
        corrupted_mods = dict(fake_mods)
        corrupted_mods["spotless_arms.models.motor_head"] = ModuleType(
            "spotless_arms.models.motor_head"
        )
        with patch.dict(sys.modules, corrupted_mods):
            with self.assertRaises(ImportError):
                singleton.is_spotless_arms_available()

    # ------------------------------------------------------------------
    # 8. Submodule Internal Dependency Failure
    # ------------------------------------------------------------------
    async def test_submodule_internal_dependency_import_failure_propagates(self) -> None:
        (
            fake_mods,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = self._make_mock_spotless_stack()

        import importlib

        orig_import = importlib.import_module

        def _submodule_dep_failure(name: str, package: str | None = None):
            if name == "spotless_arms.pipeline.pipelined_runner":
                raise ModuleNotFoundError(
                    "No module named 'evdev_internal_missing'",
                    name="evdev_internal_missing",
                )
            return orig_import(name, package)

        with (
            patch.dict(sys.modules, fake_mods),
            patch("importlib.import_module", side_effect=_submodule_dep_failure),
        ):
            with self.assertRaises(ModuleNotFoundError) as ctx:
                singleton.is_spotless_arms_available()
            self.assertEqual(ctx.exception.name, "evdev_internal_missing")

    # ------------------------------------------------------------------
    # 9. Partial Initialization Failure & Unload Cleanup Exception Safety
    # ------------------------------------------------------------------
    async def test_partial_initialization_failure_and_unload_exception_safety(self) -> None:
        (
            fake_mods,
            mock_vla_cls,
            mock_vla_instance,
            _,
            _,
            _,
            _,
        ) = self._make_mock_spotless_stack()

        mock_vla_cls.from_pretrained.side_effect = [
            RuntimeError("CUDA out of memory during SpotlessVLA init"),
            mock_vla_instance,
        ]

        with (
            patch.dict(sys.modules, fake_mods),
            patch.object(singleton, "_evict_ollama_vram"),
        ):
            # First call fails partway through initialization; _vla_instance must stay None
            with self.assertRaises(RuntimeError) as ctx:
                singleton.get_vla_engine(device="cpu")
            self.assertIn("CUDA out of memory", str(ctx.exception))
            self.assertIsNone(singleton._vla_instance)

            # Second call retries cleanly and succeeds
            recovered = singleton.get_vla_engine(device="cpu")
            self.assertIs(recovered, mock_vla_instance)
            self.assertEqual(mock_vla_cls.from_pretrained.call_count, 2)

            # If unload() raises during eviction, evict_vla_engine() still clears _vla_instance
            mock_vla_instance.unload.side_effect = RuntimeError("cleanup boom")
            singleton.evict_vla_engine()
            self.assertIsNone(singleton._vla_instance)
            mock_vla_instance.unload.assert_called_once()

    # ------------------------------------------------------------------
    # 10. Runtime Exception During Reflex Execution
    # ------------------------------------------------------------------
    async def test_reflex_execution_runtime_exception_propagates_without_legacy_fallback(
        self,
    ) -> None:
        (
            fake_mods,
            _,
            _,
            _,
            mock_runner_instance,
            _,
            _,
        ) = self._make_mock_spotless_stack()
        mock_runner_instance.run.side_effect = RuntimeError(
            "AsyncScreenBuffer timed out waiting for first frame"
        )

        with (
            patch.dict(sys.modules, fake_mods),
            patch.object(singleton, "_evict_ollama_vram"),
            patch("zelyr.tools.vision.get_grounding_engine") as mock_legacy_get,
        ):
            tool = InteractWithUITool()
            with (
                trusted_ui_execution_context(caller_ring=0, allowed_tools=["interact_with_ui"]),
                self.assertRaises(RuntimeError) as ctx,
            ):
                await tool.execute({"instruction": "Click Apply"})
            self.assertIn("AsyncScreenBuffer timed out", str(ctx.exception))
            mock_legacy_get.assert_not_called()

            # ToolRegistry wraps the propagated runtime exception into a structured ToolResult
            registry = ToolRegistry()
            registry.register_tool("interact_with_ui", tool)
            res = await registry.execute_async(
                "interact_with_ui",
                caller_ring=0,
                allowed_tools=["interact_with_ui"],
                instruction="Click Apply",
            )
            self.assertFalse(res.success)
            self.assertIn("AsyncScreenBuffer timed out", res.error)
            mock_legacy_get.assert_not_called()

    # ------------------------------------------------------------------
    # 11. Conservative Ollama Non-Eviction Policy (No Name-Based Inference)
    # ------------------------------------------------------------------
    async def test_ollama_conservative_no_unload_preserves_existing_session(self) -> None:
        mock_http_client = MagicMock()
        mock_ps_resp = MagicMock()
        mock_ps_resp.status_code = 200
        mock_ps_resp.json.return_value = {
            "models": [
                {"name": "preloaded:7b"},
                {"name": "unrelated:14b"},
                {"name": "qwen3:8b"},
            ]
        }
        mock_http_client.get.return_value = mock_ps_resp

        mock_http_cm = MagicMock()
        mock_http_cm.__enter__.return_value = mock_http_client

        mock_cfg = SimpleNamespace(
            ollama_base_url="http://127.0.0.1:11434",
            ollama_model="qwen3:8b",
        )

        # Confirm dead name-based globals do not exist on singleton
        self.assertFalse(hasattr(singleton, "_preexisting_ollama_models"))
        self.assertFalse(hasattr(singleton, "_vla_owned_ollama_models"))

        with (
            patch("httpx.Client", return_value=mock_http_cm),
            patch("zelyr.config.get_config", return_value=mock_cfg),
            patch("time.sleep"),
        ):
            with self.assertLogs(singleton.logger, level="WARNING") as cm:
                singleton._evict_ollama_vram()
            mock_http_client.get.assert_called_once_with(
                "http://127.0.0.1:11434/api/ps", timeout=2.0
            )
            self.assertEqual(mock_http_client.post.call_count, 0)
            self.assertTrue(
                any(
                    "preloaded:7b" in m and "unrelated:14b" in m and "qwen3:8b" in m
                    for m in cm.output
                )
            )

    # ------------------------------------------------------------------
    # 12. Engine Lock Ordering & Deadlock Regression
    # ------------------------------------------------------------------
    async def test_concurrent_grounding_and_vla_lock_ordering_no_deadlock(self) -> None:
        import threading
        import time

        (
            fake_mods,
            mock_vla_cls,
            _,
            _,
            _,
            _,
            _,
        ) = self._make_mock_spotless_stack()

        # Verify canonical lock ordering (_engine_lock and _vla_lock held together) on both paths
        lock_state_observations: list[tuple[str, bool, bool]] = []
        orig_evict_vla_unlocked = singleton._evict_vla_engine_unlocked
        orig_evict_grounding_unlocked = singleton._evict_grounding_engine_unlocked

        def _spy_evict_vla_unlocked():
            lock_state_observations.append(
                (
                    threading.current_thread().name,
                    singleton._engine_lock.locked(),
                    singleton._vla_lock.locked(),
                )
            )
            return orig_evict_vla_unlocked()

        def _spy_evict_grounding_unlocked():
            lock_state_observations.append(
                (
                    threading.current_thread().name,
                    singleton._engine_lock.locked(),
                    singleton._vla_lock.locked(),
                )
            )
            return orig_evict_grounding_unlocked()

        class _FakeGroundingEngine:
            _instance = None

            def unload(self) -> None:
                time.sleep(0.002)

        def _make_vla_instance(**_kwargs):
            inst = MagicMock(name="ConcurrentSpotlessVLA")
            inst.unload.side_effect = lambda: time.sleep(0.002)
            time.sleep(0.002)
            return inst

        mock_vla_cls.from_pretrained.side_effect = _make_vla_instance

        fake_zelyr_engine_mod = ModuleType("zelyr.core.vision_engine.zelyr_engine")
        fake_zelyr_engine_mod.ZelyrGroundingEngine = _FakeGroundingEngine  # type: ignore[attr-defined]

        all_mods = dict(fake_mods)
        all_mods["zelyr.core.vision_engine.zelyr_engine"] = fake_zelyr_engine_mod

        errors: list[BaseException] = []
        dual_residency_violations = 0
        start_barrier = threading.Barrier(4)

        def _worker_grounding() -> None:
            try:
                start_barrier.wait(timeout=3.0)
                for _ in range(12):
                    eng = singleton.get_grounding_engine()
                    self.assertIsNotNone(eng)
                    if singleton._engine_instance is not None and singleton._vla_instance is not None:
                        nonlocal dual_residency_violations
                        dual_residency_violations += 1
            except BaseException as exc:
                errors.append(exc)

        def _worker_vla() -> None:
            try:
                start_barrier.wait(timeout=3.0)
                for _ in range(12):
                    vla = singleton.get_vla_engine(device="cpu")
                    self.assertIsNotNone(vla)
                    if singleton._engine_instance is not None and singleton._vla_instance is not None:
                        nonlocal dual_residency_violations
                        dual_residency_violations += 1
            except BaseException as exc:
                errors.append(exc)

        def _worker_evict_grounding() -> None:
            try:
                start_barrier.wait(timeout=3.0)
                for _ in range(12):
                    singleton.evict_grounding_engine()
                    time.sleep(0.001)
            except BaseException as exc:
                errors.append(exc)

        def _worker_evict_vla() -> None:
            try:
                start_barrier.wait(timeout=3.0)
                for _ in range(12):
                    singleton.evict_vla_engine()
                    time.sleep(0.001)
            except BaseException as exc:
                errors.append(exc)

        with (
            patch.dict(sys.modules, all_mods),
            patch.object(singleton, "_evict_ollama_vram"),
            patch.object(singleton, "_evict_vla_engine_unlocked", side_effect=_spy_evict_vla_unlocked),
            patch.object(
                singleton,
                "_evict_grounding_engine_unlocked",
                side_effect=_spy_evict_grounding_unlocked,
            ),
        ):
            threads = [
                threading.Thread(target=_worker_grounding, name="t-grounding"),
                threading.Thread(target=_worker_vla, name="t-vla"),
                threading.Thread(target=_worker_evict_grounding, name="t-evict-grounding"),
                threading.Thread(target=_worker_evict_vla, name="t-evict-vla"),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5.0)
                self.assertFalse(t.is_alive(), f"Deadlock detected: thread {t.name} hung!")

        self.assertEqual(errors, [])
        self.assertEqual(dual_residency_violations, 0)
        self.assertTrue(len(lock_state_observations) > 0)
        for thread_name, eng_locked, vla_locked in lock_state_observations:
            if thread_name in ("t-grounding", "t-vla"):
                self.assertTrue(eng_locked, f"{thread_name} did not hold _engine_lock")
                self.assertTrue(vla_locked, f"{thread_name} did not hold _vla_lock")

    # ------------------------------------------------------------------
    # 13. Fail-Closed Real Input Authorization, Ring 0 Enforcement & dry_run_arm Forwarding
    # ------------------------------------------------------------------
    async def test_real_input_authorization_ring_zero_governor_and_dry_run_forwarding(
        self,
    ) -> None:
        from zelyr.services.governor import GovernorService
        from zelyr.tool_registry import TaskLocalToolScope, activate_task_local_scope
        import zelyr.core.plugins as plugins_mod

        (
            fake_mods,
            _,
            _,
            mock_runner_cls,
            _,
            _,
            mock_arm_cls,
        ) = self._make_mock_spotless_stack()

        tool = InteractWithUITool()
        self.assertEqual(tool.REQUIRED_RING, 0)
        self.assertEqual(tool.required_ring, 0)
        self.assertTrue(tool.requires_approval)

        registry = ToolRegistry()
        registry.register_tool("interact_with_ui", tool)

        with (
            patch.dict(sys.modules, fake_mods),
            patch.object(singleton, "_evict_ollama_vram"),
        ):
            # 13a. Tool registration in ToolRegistry must NOT downgrade ring 0 to ring 1
            # even when zelyr.core.plugins._rings is empty.
            with patch.dict(plugins_mod._rings, {}, clear=True):
                denied_ring1 = await registry.execute_async(
                    "interact_with_ui",
                    caller_ring=1,
                    allowed_tools=["interact_with_ui"],
                    instruction="Click Confirm",
                )
            self.assertFalse(denied_ring1.success)
            self.assertEqual(denied_ring1.error_type, "AUTHORIZATION_DENIED")
            mock_runner_cls.assert_not_called()
            mock_arm_cls.assert_not_called()

            # 13b. Fail-closed when caller_ring is absent, allowed_tools is absent, or both are absent
            # (both via ToolRegistry and via trusted_ui_execution_context).
            missing_ring_reg = await registry.execute_async(
                "interact_with_ui",
                allowed_tools=["interact_with_ui"],
                instruction="Click Confirm",
            )
            self.assertFalse(missing_ring_reg.success)
            self.assertEqual(missing_ring_reg.error_type, "AUTHORIZATION_DENIED")

            missing_allowed_reg = await registry.execute_async(
                "interact_with_ui",
                caller_ring=0,
                instruction="Click Confirm",
            )
            self.assertFalse(missing_allowed_reg.success)
            self.assertEqual(missing_allowed_reg.error_type, "AUTHORIZATION_DENIED")

            missing_both_reg = await registry.execute_async(
                "interact_with_ui",
                instruction="Click Confirm",
            )
            self.assertFalse(missing_both_reg.success)
            self.assertEqual(missing_both_reg.error_type, "AUTHORIZATION_DENIED")

            with trusted_ui_execution_context(allowed_tools=["interact_with_ui"]):
                missing_ring_ctx = await tool.execute({"instruction": "Click Confirm"})
            self.assertFalse(missing_ring_ctx.success)
            self.assertEqual(missing_ring_ctx.error_type, "AUTHORIZATION_DENIED")

            with trusted_ui_execution_context(caller_ring=0):
                missing_allowed_ctx = await tool.execute({"instruction": "Click Confirm"})
            self.assertFalse(missing_allowed_ctx.success)
            self.assertEqual(missing_allowed_ctx.error_type, "AUTHORIZATION_DENIED")

            with trusted_ui_execution_context():
                missing_both_ctx = await tool.execute({"instruction": "Click Confirm"})
            self.assertFalse(missing_both_ctx.success)
            self.assertEqual(missing_both_ctx.error_type, "AUTHORIZATION_DENIED")
            mock_arm_cls.assert_not_called()

            # 13c. Direct invocation outside ToolRegistry and trusted_ui_execution_context
            # fails closed even when Governor strict mode is disabled or no Governor exists.
            with patch.object(GovernorService, "_instance", None):
                direct_unauthenticated = await tool.execute({"instruction": "Click Confirm"})
            self.assertFalse(direct_unauthenticated.success)
            self.assertEqual(direct_unauthenticated.error_type, "AUTHORIZATION_DENIED")
            mock_arm_cls.assert_not_called()

            # 13d. Untrusted model-generated tool arguments supplying caller_ring=0 / allowed_tools
            # are ALWAYS rejected, even inside an otherwise valid trusted_ui_execution_context.
            for smuggled_params in (
                {
                    "instruction": "Click Confirm",
                    "caller_ring": 0,
                    "allowed_tools": ["interact_with_ui"],
                },
                {"instruction": "Click Confirm", "caller_ring": 0},
                {"instruction": "Click Confirm", "allowed_tools": ["interact_with_ui"]},
            ):
                direct_smuggled = await tool.execute(smuggled_params)
                self.assertFalse(direct_smuggled.success)
                self.assertEqual(direct_smuggled.error_type, "AUTHORIZATION_DENIED")

                with trusted_ui_execution_context(
                    caller_ring=0, allowed_tools=["interact_with_ui"]
                ):
                    ctx_smuggled = await tool.execute(smuggled_params)
                self.assertFalse(ctx_smuggled.success)
                self.assertEqual(ctx_smuggled.error_type, "AUTHORIZATION_DENIED")
            mock_arm_cls.assert_not_called()

            # 13e. TaskLocalToolScope isolation blocks execution
            scope = TaskLocalToolScope(
                tool_names=("interact_with_ui",), tool_rings={"interact_with_ui": 0}
            )
            with (
                activate_task_local_scope(scope),
                trusted_ui_execution_context(caller_ring=0, allowed_tools=["interact_with_ui"]),
            ):
                scoped_res = await tool.execute({"instruction": "Click Confirm"})
            self.assertFalse(scoped_res.success)
            self.assertEqual(scoped_res.error_type, "AUTHORIZATION_DENIED")
            mock_arm_cls.assert_not_called()

            # 13f. Governor strict-mode denial blocks execution before LinuxVirtualArm
            mock_gov = MagicMock()
            mock_gov.is_strict_mode.return_value = True
            mock_gov.request_approval = AsyncMock(return_value=False)
            with (
                patch.object(GovernorService, "instance", return_value=mock_gov),
                trusted_ui_execution_context(caller_ring=0, allowed_tools=["interact_with_ui"]),
            ):
                gov_denied = await tool.execute({"instruction": "Click Confirm"})
            self.assertFalse(gov_denied.success)
            self.assertEqual(gov_denied.error_type, "AUTHORIZATION_DENIED")
            self.assertIn("Governor", gov_denied.error)
            mock_arm_cls.assert_not_called()

            # 13g. Normal authorized ToolRegistry execution succeeds and forwards dry_run_arm=True
            ok_res = await registry.execute_async(
                "interact_with_ui",
                caller_ring=0,
                allowed_tools=["interact_with_ui"],
                instruction="Click Confirm",
                move_only=True,
                dry_run_arm=True,
            )
            self.assertTrue(ok_res.success)
            mock_arm_cls.assert_called_once_with(dry_run=True)

    # ------------------------------------------------------------------
    # 14. Lightweight Availability Check When Package Is Absent
    # ------------------------------------------------------------------
    async def test_absent_package_lightweight_find_spec_without_importing_submodules(
        self,
    ) -> None:
        cleaned_modules = {
            k: v for k, v in sys.modules.items() if not k.startswith("spotless_arms")
        }
        with (
            patch.dict(sys.modules, cleaned_modules, clear=True),
            patch("importlib.util.find_spec", return_value=None) as mock_find_spec,
            patch("importlib.import_module") as mock_import_module,
        ):
            self.assertFalse(singleton.is_spotless_arms_available())
            mock_find_spec.assert_called_once_with("spotless_arms")
            mock_import_module.assert_not_called()

    # ------------------------------------------------------------------
    # 15. Active VLA Operations Protected from Concurrent Eviction & Grounding Init
    # ------------------------------------------------------------------
    async def test_active_vla_operations_protected_from_concurrent_eviction_and_grounding_init(
        self,
    ) -> None:
        import threading

        (
            fake_mods,
            _,
            mock_vla_instance,
            _,
            mock_runner_instance,
            _,
            _,
        ) = self._make_mock_spotless_stack()

        class _FakeGroundingEngine:
            _instance = None

            def unload(self) -> None:
                pass

        fake_zelyr_engine_mod = ModuleType("zelyr.core.vision_engine.zelyr_engine")
        fake_zelyr_engine_mod.ZelyrGroundingEngine = _FakeGroundingEngine  # type: ignore[attr-defined]
        all_mods = dict(fake_mods)
        all_mods["zelyr.core.vision_engine.zelyr_engine"] = fake_zelyr_engine_mod

        sample_telemetry = list(mock_runner_instance.run.return_value)

        with (
            patch.dict(sys.modules, all_mods),
            patch.object(singleton, "_evict_ollama_vram"),
        ):
            # 15a. Active VLA operation while another thread requests evict_vla_engine()
            in_flight_a = threading.Event()
            release_a = threading.Event()
            evict_done_a = threading.Event()
            unloaded_while_active_a = False

            def _blocking_run_a(**_kwargs):
                in_flight_a.set()
                release_a.wait(timeout=3.0)
                if singleton._vla_instance is None or mock_vla_instance.unload.called:
                    nonlocal unloaded_while_active_a
                    unloaded_while_active_a = True
                return sample_telemetry

            mock_runner_instance.run.side_effect = _blocking_run_a

            t_vla_a = threading.Thread(
                target=lambda: singleton.run_reflex_action(
                    "Click Target A", move_only=True, dry_run_arm=True
                ),
                name="vla-active-a",
            )
            t_evict_a = threading.Thread(
                target=lambda: (singleton.evict_vla_engine(), evict_done_a.set()),
                name="vla-evict-a",
            )

            t_vla_a.start()
            self.assertTrue(in_flight_a.wait(timeout=3.0))
            self.assertEqual(singleton._vla_active_count, 1)

            t_evict_a.start()
            # Give eviction thread a moment to block on _vla_cond
            self.assertFalse(evict_done_a.wait(timeout=0.05))
            self.assertIs(singleton._vla_instance, mock_vla_instance)
            mock_vla_instance.unload.assert_not_called()

            # Release active VLA operation; eviction completes immediately afterward
            release_a.set()
            t_vla_a.join(timeout=3.0)
            t_evict_a.join(timeout=3.0)
            self.assertFalse(t_vla_a.is_alive())
            self.assertFalse(t_evict_a.is_alive())
            self.assertFalse(unloaded_while_active_a)
            self.assertEqual(singleton._vla_active_count, 0)
            self.assertIsNone(singleton._vla_instance)
            mock_vla_instance.unload.assert_called_once()

            # 15b. Active VLA operation while another thread requests get_grounding_engine()
            mock_vla_instance.unload.reset_mock()
            in_flight_b = threading.Event()
            release_b = threading.Event()
            grounding_done_b = threading.Event()
            unloaded_while_active_b = False

            def _blocking_run_b(**_kwargs):
                in_flight_b.set()
                release_b.wait(timeout=3.0)
                if (
                    singleton._vla_instance is None
                    or mock_vla_instance.unload.called
                    or singleton._engine_instance is not None
                ):
                    nonlocal unloaded_while_active_b
                    unloaded_while_active_b = True
                return sample_telemetry

            mock_runner_instance.run.side_effect = _blocking_run_b

            t_vla_b = threading.Thread(
                target=lambda: singleton.run_reflex_action(
                    "Click Target B", move_only=True, dry_run_arm=True
                ),
                name="vla-active-b",
            )
            t_ground_b = threading.Thread(
                target=lambda: (singleton.get_grounding_engine(), grounding_done_b.set()),
                name="grounding-init-b",
            )

            t_vla_b.start()
            self.assertTrue(in_flight_b.wait(timeout=3.0))
            t_ground_b.start()
            self.assertFalse(grounding_done_b.wait(timeout=0.05))
            self.assertIs(singleton._vla_instance, mock_vla_instance)
            self.assertIsNone(singleton._engine_instance)
            mock_vla_instance.unload.assert_not_called()

            release_b.set()
            t_vla_b.join(timeout=3.0)
            t_ground_b.join(timeout=3.0)
            self.assertFalse(t_vla_b.is_alive())
            self.assertFalse(t_ground_b.is_alive())
            self.assertFalse(unloaded_while_active_b)
            self.assertIsNone(singleton._vla_instance)
            self.assertIsNotNone(singleton._engine_instance)
            mock_vla_instance.unload.assert_called_once()

            # 15c. Multiple concurrent VLA operations defer eviction until ALL finish
            singleton.evict_grounding_engine()
            mock_vla_instance.unload.reset_mock()
            both_in_flight = threading.Barrier(3)
            release_c1 = threading.Event()
            release_c2 = threading.Event()
            evict_done_c = threading.Event()
            gate_lock = threading.Lock()
            call_idx = 0

            def _concurrent_run_c(**_kwargs):
                nonlocal call_idx
                with gate_lock:
                    call_idx += 1
                    my_idx = call_idx
                both_in_flight.wait(timeout=3.0)
                if my_idx == 1:
                    release_c1.wait(timeout=3.0)
                else:
                    release_c2.wait(timeout=3.0)
                return sample_telemetry

            mock_runner_instance.run.side_effect = _concurrent_run_c

            t_c1 = threading.Thread(
                target=lambda: singleton.run_reflex_action(
                    "Concurrent 1", move_only=True, dry_run_arm=True
                ),
                name="vla-c1",
            )
            t_c2 = threading.Thread(
                target=lambda: singleton.run_reflex_action(
                    "Concurrent 2", move_only=True, dry_run_arm=True
                ),
                name="vla-c2",
            )
            t_c1.start()
            t_c2.start()
            both_in_flight.wait(timeout=3.0)
            self.assertEqual(singleton._vla_active_count, 2)

            t_evict_c = threading.Thread(
                target=lambda: (singleton.evict_vla_engine(), evict_done_c.set()),
                name="vla-evict-c",
            )
            t_evict_c.start()
            self.assertFalse(evict_done_c.wait(timeout=0.05))

            # First VLA operation completes; second is still active -> engine MUST remain resident
            release_c1.set()
            t_c1.join(timeout=3.0)
            self.assertFalse(evict_done_c.wait(timeout=0.05))
            self.assertEqual(singleton._vla_active_count, 1)
            self.assertIs(singleton._vla_instance, mock_vla_instance)
            mock_vla_instance.unload.assert_not_called()

            # Second VLA operation completes -> eviction now proceeds
            release_c2.set()
            t_c2.join(timeout=3.0)
            t_evict_c.join(timeout=3.0)
            self.assertTrue(evict_done_c.is_set())
            self.assertEqual(singleton._vla_active_count, 0)
            self.assertIsNone(singleton._vla_instance)
            mock_vla_instance.unload.assert_called_once()

            # 15d. Lease cleanup after runtime exception or cancellation
            mock_vla_instance.unload.reset_mock()
            mock_runner_instance.run.side_effect = RuntimeError("runner crashed mid-tick")
            with self.assertRaises(RuntimeError):
                singleton.run_reflex_action("Crash test", move_only=True, dry_run_arm=True)
            self.assertEqual(singleton._vla_active_count, 0)
            singleton.evict_vla_engine()
            self.assertIsNone(singleton._vla_instance)
            mock_vla_instance.unload.assert_called_once()


if __name__ == "__main__":
    unittest.main()
