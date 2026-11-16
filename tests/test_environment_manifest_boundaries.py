"""Focused boundary tests for RuntimeEnvironmentManifest and InstructionManager."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

from zelyr.core.instructions import InstructionManager
from zelyr.engine.environment import RuntimeEnvironmentManifest


class _DummyTool:
    def __init__(self, required_ring: int = 1) -> None:
        self.required_ring = required_ring


class _DummyCoreRegistry:
    def __init__(self, tools: dict) -> None:
        self._tools = dict(tools)

    def list_tools(self) -> dict:
        return dict(self._tools)


class _DummyRegistry:
    def __init__(self, registered: dict, disabled: set[str]) -> None:
        self._core_registry = _DummyCoreRegistry(registered)
        self._disabled = set(disabled)

    def list_tools(self) -> dict:
        return {k: v for k, v in self._core_registry.list_tools().items() if k not in self._disabled}

    @staticmethod
    def _get_tool_ring(_tool_id: str, tool: object) -> int:
        return int(getattr(tool, "required_ring", 1))


def test_inference_locality_local_remote_and_unknown() -> None:
    assert RuntimeEnvironmentManifest.detect_inference_locality("http://127.0.0.1:11434") == "local_daemon"
    assert RuntimeEnvironmentManifest.locality_category("local_daemon") == "local"

    assert RuntimeEnvironmentManifest.detect_inference_locality("http://100.100.33.5:11434") == "remote_provider"
    assert RuntimeEnvironmentManifest.locality_category("remote_provider") == "remote"

    for bad_cfg in ("", "   ", "://bad-url", "http://[::1"):
        loc = RuntimeEnvironmentManifest.detect_inference_locality(bad_cfg)
        assert loc == "unknown"
        assert RuntimeEnvironmentManifest.locality_category(loc) == "unknown"

    with patch("zelyr.config.get_config", side_effect=RuntimeError("no config")):
        assert RuntimeEnvironmentManifest.detect_inference_locality(None) == "unknown"


def test_registered_available_and_authorized_tools_not_conflated() -> None:
    reg = _DummyRegistry(
        registered={
            "execute_command": _DummyTool(required_ring=0),
            "manage_workspace_file": _DummyTool(required_ring=1),
            "manage_desktop_window": _DummyTool(required_ring=1),
            "craft_tool": _DummyTool(required_ring=1),
            "manage_memory": _DummyTool(required_ring=1),
        },
        disabled={"craft_tool"},
    )

    with (
        patch.object(RuntimeEnvironmentManifest, "detect_desktop_state", return_value="headless_or_unsupported"),
        patch.object(RuntimeEnvironmentManifest, "_is_browser_active", return_value=False),
        patch.object(RuntimeEnvironmentManifest, "_is_shell_runtime_available", return_value=True),
        patch.object(RuntimeEnvironmentManifest, "_is_filesystem_runtime_available", return_value=True),
        patch.object(RuntimeEnvironmentManifest, "_is_memory_runtime_available", return_value=True),
    ):
        summary = RuntimeEnvironmentManifest.get_tool_capability_summary(
            tool_registry=reg,
            caller_ring=1,
            allowed_tools={"execute_command", "manage_workspace_file", "manage_desktop_window", "craft_tool"},
        )
        assert summary["registry_known"] is True
        assert summary["authorization_scoped"] is True
        # Registered includes disabled craft_tool and unavailable manage_desktop_window
        assert summary["registered"] == [
            "craft_tool",
            "execute_command",
            "manage_desktop_window",
            "manage_memory",
            "manage_workspace_file",
        ]
        # Available excludes disabled (craft_tool) and unsubstantiated desktop (manage_desktop_window)
        assert summary["available"] == ["execute_command", "manage_memory", "manage_workspace_file"]
        # Authorized enforces caller_ring=1 (blocks ring-0 execute_command) and allowed_tools (excludes manage_memory)
        assert summary["authorized"] == ["manage_workspace_file"]

        actuators = RuntimeEnvironmentManifest.get_active_actuators(
            tool_registry=reg,
            caller_ring=1,
            allowed_tools={"execute_command", "manage_workspace_file"},
        )
        assert actuators["filesystem"] is True
        assert actuators["shell"] is False
        assert actuators["desktop_window"] is False
        assert actuators["browser"] is False
        assert actuators["memory"] is False


def test_secrets_excluded_and_instructions_preserve_runtime_authority(tmp_path: Path) -> None:
    cfg = {
        "ollama_base_url": "https://user:supersecret@remote-node.example:11434/v1?api_key=tok123",
        "ollama_model": "qwen2.5:3b",
    }
    with (
        patch.dict(os.environ, {"OPENAI_API_KEY": "sk-secret-value", "ZELYR_API_TOKEN": "tok-hidden"}),
        patch.object(RuntimeEnvironmentManifest, "detect_desktop_state", return_value="headless_or_unsupported"),
        patch.object(RuntimeEnvironmentManifest, "_is_browser_active", return_value=False),
    ):
        collected = RuntimeEnvironmentManifest.collect_manifest(config=cfg)
        assert collected["locality"] == "remote"
        assert collected["tool_execution_locality"] == "local"
        assert collected["tools"]["registry_known"] is False

        serialized = json.dumps(collected)
        for secret in ("supersecret", "tok123", "api_key", "sk-secret-value", "tok-hidden"):
            assert secret not in serialized

        compiled = InstructionManager(global_path=tmp_path / "inst.md", cwd=tmp_path).get_compiled_instructions(config=cfg)
        assert "[RUNTIME ENVIRONMENT & CAPABILITY MANIFEST]" in compiled
        assert "remote_provider (qwen2.5:3b) | Tool Exec: local" in compiled
        assert "Degraded/Unavailable: desktop_window, browser" in compiled
        assert "[RUNTIME AUTHORITY & CAPABILITY BOUNDARIES]" in compiled
        assert "Inference locality (local or remote endpoint) is separate from tool execution" in compiled
        assert "Never represent an unavailable or unauthorized tool call as successfully executed" in compiled
        for secret in ("supersecret", "tok123", "sk-secret-value", "tok-hidden"):
            assert secret not in compiled
