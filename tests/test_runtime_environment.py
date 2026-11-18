"""Tests for Runtime Environment Introspection and Capability Manifest."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.core.instructions import InstructionManager
from zelyr.engine.budget_manager import ContextBudgetEngine
from zelyr.engine.environment import RuntimeEnvironmentManifest


class TestInferenceLocalityDetection:
    """Verify classification of inference endpoints as local_daemon vs remote_provider."""

    @pytest.mark.parametrize(
        "endpoint",
        [
            "http://localhost:11434",
            "http://127.0.0.1:11434",
            "http://0.0.0.0:8000",
            "http://[::1]:11434",
            "localhost:11434",
            "127.0.0.1:8080",
            "/var/run/ollama.sock",
            "unix:///tmp/ollama.sock",
        ],
    )
    def test_loopback_and_local_urls_return_local_daemon(self, endpoint):
        assert RuntimeEnvironmentManifest.detect_inference_locality(endpoint) == "local_daemon"

    @pytest.mark.parametrize(
        "endpoint",
        [
            "https://api.openai.com/v1",
            "https://api.anthropic.com/v1",
            "https://api.deepseek.com",
            "http://remote-cluster.internal:11434",
            "https://remote.ollama.ai",
        ],
    )
    def test_external_and_remote_urls_return_remote_provider(self, endpoint):
        assert RuntimeEnvironmentManifest.detect_inference_locality(endpoint) == "remote_provider"

    def test_config_object_with_ollama_base_url(self):
        cfg_local = MagicMock(ollama_base_url="http://127.0.0.1:11434", model="qwen3:8b")
        assert RuntimeEnvironmentManifest.detect_inference_locality(cfg_local) == "local_daemon"

        cfg_remote = MagicMock(ollama_base_url="https://api.groq.com/openai/v1", model="llama-3.3-70b")
        assert RuntimeEnvironmentManifest.detect_inference_locality(cfg_remote) == "remote_provider"

    def test_dict_config(self):
        assert (
            RuntimeEnvironmentManifest.detect_inference_locality(
                {"ollama_base_url": "http://localhost:11434"}
            )
            == "local_daemon"
        )
        assert (
            RuntimeEnvironmentManifest.detect_inference_locality(
                {"base_url": "https://api.mistral.ai/v1"}
            )
            == "remote_provider"
        )

    def test_none_config_falls_back_to_local_default(self):
        # Default Zelyr configuration uses 127.0.0.1:11434
        assert RuntimeEnvironmentManifest.detect_inference_locality(None) == "local_daemon"


class TestDesktopStateDetection:
    """Verify detection of Hyprland Wayland, generic Wayland, and headless environments."""

    def test_hyprland_socket_present_returns_hyprland_wayland(self, monkeypatch):
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
        with patch("zelyr.engine.hyprland_ipc.HyprlandIPCClient.is_available", return_value=True):
            assert RuntimeEnvironmentManifest.detect_desktop_state() == "hyprland_wayland"

    def test_hyprland_socket_absent_with_wayland_display_returns_generic_wayland(self, monkeypatch):
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
        with patch("zelyr.engine.hyprland_ipc.HyprlandIPCClient.is_available", return_value=False):
            assert RuntimeEnvironmentManifest.detect_desktop_state() == "generic_wayland"

    def test_hyprland_socket_absent_without_wayland_display_returns_headless(self, monkeypatch):
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        with patch("zelyr.engine.hyprland_ipc.HyprlandIPCClient.is_available", return_value=False):
            assert RuntimeEnvironmentManifest.detect_desktop_state() == "headless_or_unsupported"


class TestCapabilityReportingAndDegradation:
    """Verify active actuators and truthful degradation reporting."""

    def test_active_actuators_when_hyprland_and_browser_active(self):
        with patch.object(RuntimeEnvironmentManifest, "detect_desktop_state", return_value="hyprland_wayland"):
            with patch.object(RuntimeEnvironmentManifest, "_is_browser_active", return_value=True):
                actuators = RuntimeEnvironmentManifest.get_active_actuators()
                assert actuators["shell"] is True
                assert actuators["filesystem"] is True
                assert actuators["desktop_window"] is True
                assert actuators["browser"] is True
                assert actuators["memory"] is True

    def test_desktop_window_degraded_when_hyprland_unavailable(self):
        with patch.object(RuntimeEnvironmentManifest, "detect_desktop_state", return_value="headless_or_unsupported"):
            with patch.object(RuntimeEnvironmentManifest, "_is_browser_active", return_value=False):
                actuators = RuntimeEnvironmentManifest.get_active_actuators()
                assert actuators["desktop_window"] is False
                assert actuators["browser"] is False
                assert actuators["shell"] is True
                assert actuators["filesystem"] is True
                assert actuators["memory"] is True

                manifest = RuntimeEnvironmentManifest.build_manifest_block()
                assert "Active Actuators: shell, filesystem, memory" in manifest
                assert "Degraded/Unavailable: desktop_window, browser" in manifest

    def test_all_actuators_active_reports_degraded_none(self):
        with patch.object(RuntimeEnvironmentManifest, "detect_desktop_state", return_value="hyprland_wayland"):
            with patch.object(RuntimeEnvironmentManifest, "_is_browser_active", return_value=True):
                manifest = RuntimeEnvironmentManifest.build_manifest_block()
                assert "Degraded/Unavailable: none" in manifest


class TestTokenEconomyAndFormatting:
    """Verify strict token bounds and absence of sensitive data leakage."""

    def test_manifest_token_economy_under_120_tokens(self):
        manifest = RuntimeEnvironmentManifest.build_manifest_block()
        tokens = NativeOrchestrator.count_tokens(manifest)
        assert tokens <= 120, f"Manifest exceeds 120 tokens: {tokens}"
        assert len(manifest) <= 480, f"Manifest exceeds 480 characters: {len(manifest)}"

    def test_sensitive_information_protection(self):
        # Inject sensitive environment variables to verify none leak into the manifest
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "sk-secret-test-key-12345",
                "AWS_SECRET_ACCESS_KEY": "super-secret-aws-key",
                "USER_PASSWORD": "supersecretpassword",
            },
        ):
            manifest = RuntimeEnvironmentManifest.build_manifest_block()
            assert "sk-secret-test-key-12345" not in manifest
            assert "super-secret-aws-key" not in manifest
            assert "supersecretpassword" not in manifest
            assert "OPENAI_API_KEY" not in manifest
            assert "AWS_SECRET_ACCESS_KEY" not in manifest

            # Check that lines conform to expected manifest schema
            lines = [line.strip() for line in manifest.splitlines() if line.strip()]
            assert lines[0] == "[RUNTIME ENVIRONMENT & CAPABILITY MANIFEST]"
            assert lines[1].startswith("- Inference:")
            assert lines[2].startswith("- Target Host:")
            assert lines[3].startswith("- Desktop Session:")
            assert lines[4].startswith("- Active Actuators:")
            assert lines[5].startswith("- Degraded/Unavailable:")
            assert lines[6] == "- Constraints: Non-interactive execution, DEVNULL stdin, no interactive sudo."


class TestContextBudgetingAndAnchorImmunity:
    """Verify manifest survives sliding-window turn eviction under high token pressure."""

    def test_manifest_survives_extreme_context_pruning(self):
        engine = ContextBudgetEngine()
        manager = InstructionManager()
        compiled_instructions = manager.get_compiled_instructions()

        assert "[RUNTIME ENVIRONMENT & CAPABILITY MANIFEST]" in compiled_instructions

        sys_msg = {"role": "system", "content": compiled_instructions}
        root_user = {"role": "user", "content": "Initial root task instruction."}

        # Create massive conversation history that exceeds target budget
        bulky_messages = [sys_msg, root_user]
        for i in range(10):
            bulky_messages.append({"role": "assistant", "content": f"Step {i} " + ("bulky context text " * 50)})
            bulky_messages.append({"role": "user", "content": f"User query {i} " + ("more query context " * 50)})

        active_user = {"role": "user", "content": "Latest user turn needing execution."}
        bulky_messages.append(active_user)

        # Apply aggressive context compression (e.g. max 1000 tokens)
        pruned_messages = engine.apply_context_budget(
            bulky_messages,
            max_context_tokens=1000,
            reserve_tokens=200,
        )

        # Inviolate root anchor check: System message with manifest MUST survive
        assert len(pruned_messages) >= 2
        assert pruned_messages[0]["role"] == "system"
        assert "[RUNTIME ENVIRONMENT & CAPABILITY MANIFEST]" in pruned_messages[0]["content"]
        assert "- Constraints: Non-interactive execution" in pruned_messages[0]["content"]


class TestOrchestratorIntegration:
    """Verify NativeOrchestrator builds system prompt with truthful manifest."""

    def test_native_orchestrator_build_system_prompt_includes_manifest(self):
        orch = NativeOrchestrator()
        prompt = orch._build_system_prompt()
        assert "[RUNTIME ENVIRONMENT & CAPABILITY MANIFEST]" in prompt
        assert "- Inference:" in prompt
        assert "- Target Host:" in prompt
        assert "- Desktop Session:" in prompt
        assert "- Active Actuators:" in prompt
        assert "- Degraded/Unavailable:" in prompt
