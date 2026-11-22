"""Tests for Runtime Environment Introspection and Capability Manifest."""

from __future__ import annotations

import json
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

    def test_orchestrator_programmatic_manifest_and_planning_context(self):
        from zelyr.engine.environment import CapabilityStatus
        from zelyr.tool_registry import ToolRegistry

        reg = ToolRegistry()
        for tid in reg.get_builtin_tools():
            reg.get_tool(tid)

        orch = NativeOrchestrator(
            tool_registry=reg,
            caller_ring=1,
            allowed_tools=["echo", "file_read"],
        )
        manifest = orch.get_runtime_manifest(probe_services=False, refresh=True)
        assert manifest["schema_version"] == "1.4"
        assert "system" in manifest
        assert "ai_infrastructure" in manifest
        assert "tools" in manifest
        assert "integrations" in manifest
        assert manifest["tools"]["registry_known"] is True
        assert manifest["tools"]["authorization_scoped"] is True
        assert "echo" in manifest["tools"]["authorized"]
        assert "shell" not in manifest["tools"]["authorized"]

        # Unprobed endpoint reports UNKNOWN with source="configured", never fabricated AVAILABLE
        assert manifest["ai_infrastructure"]["endpoint_status"] == CapabilityStatus.UNKNOWN.value
        assert manifest["ai_infrastructure"]["verified"] is False
        assert manifest["ai_infrastructure"]["source"] == "configured"

        planning_block = orch.get_planning_environment_context()
        assert "[RUNTIME ENVIRONMENT & CAPABILITY MANIFEST]" in planning_block
        assert NativeOrchestrator.count_tokens(planning_block) <= 120

        refreshed = orch.refresh_runtime_manifest(probe_services=False)
        assert refreshed["schema_version"] == "1.4"


class TestStructuredManifestSectionsAndStatusSemantics:
    """Verify AVAILABLE, UNAVAILABLE, DEGRADED, and UNKNOWN semantics across all sections."""

    def test_system_section_and_psutil_fallback_to_proc_meminfo(self, monkeypatch):
        from zelyr.engine.environment import CapabilityStatus

        monkeypatch.setenv("DISPLAY", ":0")
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        with (
            patch.object(
                RuntimeEnvironmentManifest,
                "detect_desktop_state",
                return_value="headless_or_unsupported",
            ),
            patch.dict("sys.modules", {"psutil": None}),
        ):
            sys_info = RuntimeEnvironmentManifest.collect_system_info()
            assert sys_info["os_name"]
            assert sys_info["architecture"]
            assert sys_info["python_version"]
            assert sys_info["cpu"]["status"] == CapabilityStatus.AVAILABLE.value
            assert sys_info["display"]["session_type"] == "x11"
            assert sys_info["display"]["status"] == CapabilityStatus.DEGRADED.value
            assert sys_info["display"]["headless"] is False
            assert sys_info["memory"]["status"] in (
                CapabilityStatus.AVAILABLE.value,
                CapabilityStatus.UNKNOWN.value,
            )

    def test_ai_infrastructure_available_degraded_unavailable_and_timeout(self):
        import io
        import socket
        import urllib.error
        from zelyr.engine.environment import CapabilityStatus

        RuntimeEnvironmentManifest.clear_cache()
        cfg = {"ollama_base_url": "http://127.0.0.1:11434", "ollama_model": "qwen2.5-coder:7b"}

        # 1. Endpoint online + configured model present -> AVAILABLE
        class _RespOK:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"models": [{"name": "qwen2.5-coder:7b"}]}'

        with patch("urllib.request.urlopen", return_value=_RespOK()):
            ai_ok = RuntimeEnvironmentManifest.collect_ai_infrastructure(
                config=cfg, probe_services=True, refresh=True
            )
            assert ai_ok["endpoint_status"] == CapabilityStatus.AVAILABLE.value
            assert ai_ok["verified"] is True
            assert ai_ok["source"] == "detected"
            assert ai_ok["model_installed"] is True
            assert ai_ok["context_limit"]["max_tokens"] == 32768
            assert ai_ok["context_limit"]["source"] == "catalog_heuristic"
            assert ai_ok["capabilities"]["tool_calling"]["source"] == "assumed"
            assert ai_ok["capabilities"]["tool_calling"]["verified"] is False

        # 2. Endpoint online + configured model missing -> DEGRADED
        class _RespMissingModel:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"models": [{"name": "llama3:8b"}]}'

        with patch("urllib.request.urlopen", return_value=_RespMissingModel()):
            ai_deg = RuntimeEnvironmentManifest.collect_ai_infrastructure(
                config=cfg, probe_services=True, refresh=True
            )
            assert ai_deg["endpoint_status"] == CapabilityStatus.DEGRADED.value
            assert ai_deg["verified"] is True
            assert ai_deg["model_installed"] is False
            assert "not found in installed tags" in ai_deg["details"]

        # 3. Endpoint timeout -> UNAVAILABLE with bounded timeout message
        with patch("urllib.request.urlopen", side_effect=socket.timeout("timed out")):
            ai_timeout = RuntimeEnvironmentManifest.collect_ai_infrastructure(
                config=cfg, probe_services=True, probe_timeout=0.25, refresh=True
            )
            assert ai_timeout["endpoint_status"] == CapabilityStatus.UNAVAILABLE.value
            assert ai_timeout["verified"] is True
            assert "timed out after 0.25s" in ai_timeout["details"]

        # 4. Endpoint connection refused -> UNAVAILABLE
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError(ConnectionRefusedError("Connection refused")),
        ):
            ai_down = RuntimeEnvironmentManifest.collect_ai_infrastructure(
                config=cfg, probe_services=True, refresh=True
            )
            assert ai_down["endpoint_status"] == CapabilityStatus.UNAVAILABLE.value
            assert ai_down["verified"] is True
            assert "unreachable" in ai_down["details"].lower()

        # 5. HTTP 404/401 -> DEGRADED
        http_404 = urllib.error.HTTPError(
            "http://127.0.0.1:11434/api/tags", 404, "Not Found", {}, io.BytesIO(b"")
        )
        with patch("urllib.request.urlopen", side_effect=http_404):
            ai_404 = RuntimeEnvironmentManifest.collect_ai_infrastructure(
                config=cfg, probe_services=True, refresh=True
            )
            assert ai_404["endpoint_status"] == CapabilityStatus.DEGRADED.value

    def test_integrations_missing_optional_deps_and_degraded_states(self, monkeypatch):
        from zelyr.engine.environment import CapabilityStatus

        # Case A: Wayland display active, VLM available, Spotless-Arms absent, git present without gh
        with (
            patch.object(
                RuntimeEnvironmentManifest,
                "detect_desktop_state",
                return_value="generic_wayland",
            ),
            patch.object(
                RuntimeEnvironmentManifest,
                "_is_browser_active",
                return_value=False,
            ),
            patch(
                "zelyr.core.vision_engine.singleton.is_spotless_arms_available",
                return_value=False,
            ),
            patch(
                "zelyr.engine.environment._is_module_importable",
                side_effect=lambda mod: mod in ("torch", "transformers", "playwright"),
            ),
            patch(
                "zelyr.engine.environment.shutil.which",
                side_effect=lambda b: f"/usr/bin/{b}" if b in ("git", "grim", "wl-copy", "wl-paste") else None,
            ),
        ):
            integrations = RuntimeEnvironmentManifest.collect_integrations()
            assert integrations["desktop_compositor"]["status"] == CapabilityStatus.DEGRADED.value
            assert integrations["browser"]["status"] == CapabilityStatus.DEGRADED.value
            assert integrations["vision_vla"]["status"] == CapabilityStatus.DEGRADED.value
            assert integrations["vision_vla"]["backend"] == "zelyr_grounding_vlm"
            assert integrations["github"]["status"] == CapabilityStatus.DEGRADED.value
            assert integrations["github"]["git_installed"] is True
            assert integrations["github"]["gh_cli_installed"] is False
            assert integrations["clipboard"]["status"] == CapabilityStatus.AVAILABLE.value
            assert integrations["media_mpris"]["status"] == CapabilityStatus.UNAVAILABLE.value

        # Case B: Transitive ModuleNotFoundError inside is_spotless_arms_available() is caught safely
        with (
            patch.object(
                RuntimeEnvironmentManifest,
                "detect_desktop_state",
                return_value="hyprland_wayland",
            ),
            patch(
                "zelyr.core.vision_engine.singleton.is_spotless_arms_available",
                side_effect=ModuleNotFoundError("No module named 'broken_subdep'", name="broken_subdep"),
            ),
            patch("zelyr.engine.environment._is_module_importable", return_value=False),
        ):
            vision_info = RuntimeEnvironmentManifest._inspect_vision_runtime()
            assert vision_info["status"] == CapabilityStatus.UNAVAILABLE.value
            assert any("broken_subdep" in lim for lim in vision_info["limitations"])

    def test_tool_not_claimed_available_merely_because_module_imports(self):
        """A registered tool class with missing runtime dependencies or failed self-check is UNAVAILABLE."""
        from zelyr.engine.environment import CapabilityStatus

        class _ImportedToolWithMissingDeps:
            required_ring = 1
            required_binaries = ["nonexistent_binary_for_zelyr_test_999"]
            required_modules = ["nonexistent_python_mod_for_zelyr_test_999"]

        class _ImportedToolFailingSelfCheck:
            required_ring = 1

            @staticmethod
            def is_available() -> bool:
                return False

        class _Registry:
            def __init__(self, tools_map):
                self._tools = dict(tools_map)
                self._core_registry = MagicMock(list_tools=lambda: dict(self._tools))

            def list_tools(self):
                return dict(self._tools)

            @staticmethod
            def _get_tool_ring(_tid, tool):
                return getattr(tool, "required_ring", 1)

        reg = _Registry(
            {
                "custom_bin_tool": _ImportedToolWithMissingDeps(),
                "custom_check_tool": _ImportedToolFailingSelfCheck(),
                "manage_desktop_window": _ImportedToolFailingSelfCheck(),
                "interact_with_browser": _ImportedToolFailingSelfCheck(),
            }
        )

        with (
            patch.object(
                RuntimeEnvironmentManifest,
                "detect_desktop_state",
                return_value="headless_or_unsupported",
            ),
            patch.object(RuntimeEnvironmentManifest, "_is_browser_active", return_value=False),
        ):
            summary = RuntimeEnvironmentManifest.get_tool_capability_summary(
                tool_registry=reg, caller_ring=1
            )
            assert summary["registered"] == [
                "custom_bin_tool",
                "custom_check_tool",
                "interact_with_browser",
                "manage_desktop_window",
            ]
            assert summary["available"] == []
            assert summary["authorized"] == []
            assert summary["unavailable"] == [
                "custom_bin_tool",
                "custom_check_tool",
                "interact_with_browser",
                "manage_desktop_window",
            ]
            details = summary["details"]
            assert details["custom_bin_tool"]["status"] == CapabilityStatus.UNAVAILABLE.value
            assert "nonexistent_binary_for_zelyr_test_999" in details["custom_bin_tool"]["missing_dependencies"]
            assert "nonexistent_python_mod_for_zelyr_test_999" in details["custom_bin_tool"]["missing_dependencies"]
            assert details["custom_check_tool"]["status"] == CapabilityStatus.UNAVAILABLE.value
            assert "tool_runtime_check" in details["custom_check_tool"]["missing_dependencies"]

    def test_tool_registration_changes_automatically_invalidate_manifest_cache(self):
        RuntimeEnvironmentManifest.clear_cache()

        class _MutableRegistry:
            def __init__(self):
                self.tools = {"manage_workspace_file": MagicMock(required_ring=1)}
                self._core_registry = MagicMock(list_tools=lambda: dict(self.tools))

            def list_tools(self):
                return dict(self.tools)

            @staticmethod
            def _get_tool_ring(_tid, tool):
                return getattr(tool, "required_ring", 1)

        reg = _MutableRegistry()
        m1 = RuntimeEnvironmentManifest.collect_manifest(tool_registry=reg, caller_ring=1)
        assert m1["tools"]["registered"] == ["manage_workspace_file"]

        # Mutate registered tools without calling clear_cache() — cache key must detect the change
        reg.tools["execute_command"] = MagicMock(required_ring=0)
        m2 = RuntimeEnvironmentManifest.collect_manifest(tool_registry=reg, caller_ring=1)
        assert m2["tools"]["registered"] == ["execute_command", "manage_workspace_file"]
        assert "execute_command" in m2["tools"]["unauthorized"]
        assert m2["tools"]["details"]["execute_command"]["required_ring"] == 0
        assert m2["tools"]["details"]["execute_command"]["execution_mode"] == "external_subprocess"

    def test_secret_redaction_across_manifest_and_diagnostic_report(self, monkeypatch):
        monkeypatch.setenv("GH_TOKEN", "ghp_SuperSecretGitHubToken987654321")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-UltraSecretOpenAIKey123456789")
        monkeypatch.setenv("CUSTOM_SERVICE_SECRET", "my-custom-private-secret-token-val")

        cfg = {
            "ollama_base_url": "https://admin:url_password_123@llm.example.com:11434/v1?api_key=query_secret_456",
            "ollama_model": "qwen2.5-coder:7b",
        }
        manifest = RuntimeEnvironmentManifest.collect_manifest(config=cfg, refresh=True)
        report = RuntimeEnvironmentManifest.format_diagnostic_report(manifest)
        serialized = json.dumps(manifest)

        # Boolean presence is reported for GitHub token, but no secret value ever appears
        assert manifest["integrations"]["github"]["token_configured"] is True
        assert manifest["ai_infrastructure"]["endpoint"] == "https://llm.example.com:11434/v1"

        for secret in (
            "ghp_SuperSecretGitHubToken987654321",
            "sk-proj-UltraSecretOpenAIKey123456789",
            "my-custom-private-secret-token-val",
            "url_password_123",
            "query_secret_456",
        ):
            assert secret not in serialized
            assert secret not in report

    def test_cli_and_repl_manifest_diagnostic_commands(self, capsys):
        import io
        from rich.console import Console
        from zelyr.cli.main import _build_parser, cmd_env, cmd_status
        import zelyr.cli.repl as repl_mod
        from zelyr.cli.repl import InlineRepl

        parser = _build_parser()
        env_args = parser.parse_args(["env", "--json", "--no-probe", "--caller-ring", "1"])
        cmd_env(env_args)
        out_json = capsys.readouterr().out
        parsed = json.loads(out_json)
        assert parsed["schema_version"] == "1.4"
        assert parsed["tools"]["registry_known"] is True
        assert parsed["tools"]["authorization_scoped"] is True

        compact_args = parser.parse_args(["env", "--compact", "--no-probe"])
        cmd_env(compact_args)
        out_compact = capsys.readouterr().out
        assert "[RUNTIME ENVIRONMENT & CAPABILITY MANIFEST]" in out_compact

        status_manifest_args = parser.parse_args(["status", "--manifest", "--no-probe"])
        cmd_status(status_manifest_args)
        out_report = capsys.readouterr().out
        assert "Zelyr Runtime Environment Manifest (v1.4)" in out_report
        assert "[System]" in out_report
        assert "[AI Infrastructure]" in out_report
        assert "[Tools]" in out_report
        assert "[Integrations]" in out_report

        # REPL /env command
        repl = InlineRepl()
        buf = io.StringIO()
        orig_console = repl_mod.console
        try:
            repl_mod.console = Console(file=buf, force_terminal=False, color_system=None)
            assert repl.handle_command("/env --no-probe") is True
            assert "Zelyr Runtime Environment Manifest" in buf.getvalue()
        finally:
            repl_mod.console = orig_console
