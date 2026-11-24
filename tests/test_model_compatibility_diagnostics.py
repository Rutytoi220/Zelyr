"""Tests for model and provider compatibility diagnostics."""

from __future__ import annotations

import io
import json
import socket
import sys
import urllib.error
from typing import Any, Dict

import pytest

from zelyr.config import ZelyrConfig, set_config
from zelyr.engine.model_compatibility import (
    ModelCompatibilityDiagnostics,
    ModelCompatibilityReport,
    check_model_compatibility,
    format_model_compatibility_report,
)


def test_ollama_detected_capabilities_and_context_limits() -> None:
    calls: list[tuple[str, str, Any]] = []

    def fake_http(
        method: str,
        url: str,
        *,
        body: Dict[str, Any] | None = None,
        timeout: float = 2.0,
    ) -> tuple[int, Dict[str, Any]]:
        calls.append((method, url, body))
        if url.endswith("/api/tags"):
            return 200, {"models": [{"name": "qwen3:8b"}]}
        if url.endswith("/api/show"):
            return 200, {
                "capabilities": ["completion", "tools", "thinking"],
                "details": {"family": "qwen3", "parameter_size": "8B"},
                "model_info": {"qwen3.context_length": 32768},
            }
        raise AssertionError(f"Unexpected URL: {url}")

    cfg = ZelyrConfig(ollama_base_url="http://127.0.0.1:11434", ollama_model="qwen3:8b", ollama_num_ctx=8192)
    report = check_model_compatibility(config=cfg, workflow="planned_graph", http_request_fn=fake_http)

    assert isinstance(report, ModelCompatibilityReport)
    assert [c[0] for c in calls] == ["GET", "POST"]
    assert report.provider_type == "ollama"
    assert report.overall_status == "AVAILABLE"
    assert report.model_installed is True
    assert report.workflow_compatible is True
    assert report.model_capability_confidence == "medium"

    assert report.capabilities["provider_availability"]["status"] == "AVAILABLE"
    assert report.capabilities["native_tool_calling"]["status"] == "AVAILABLE"
    assert report.capabilities["native_tool_calling"]["source"] == "detected"
    assert report.capabilities["tool_argument_formatting"]["status"] == "AVAILABLE"
    assert report.capabilities["thinking_metadata"]["status"] == "AVAILABLE"
    assert report.capabilities["context_limits"]["status"] == "AVAILABLE"
    assert report.capabilities["context_limits"]["detected_context_length"] == 32768
    assert report.capabilities["streaming_behavior"]["status"] == "AVAILABLE"
    # Multi-turn and structured output remain honestly UNKNOWN without protocol execution evidence.
    assert report.capabilities["multi_turn_tool_calling"]["status"] == "UNKNOWN"
    assert report.capabilities["structured_output_support"]["status"] == "UNKNOWN"


def test_no_probe_never_guesses_capabilities_from_model_name() -> None:
    def fail_if_called(*args: Any, **kwargs: Any) -> tuple[int, Dict[str, Any]]:
        raise AssertionError("HTTP probe must not be called when probe=False")

    for model_name in ("qwen3:32b", "gpt-4o", "claude-3-7-sonnet"):
        report = ModelCompatibilityDiagnostics.diagnose(
            config=ZelyrConfig(ollama_base_url="http://127.0.0.1:11434"),
            model=model_name,
            probe=False,
            workflow="planned_graph",
            http_request_fn=fail_if_called,
        )
        assert report.overall_status == "UNKNOWN"
        assert report.model_installed is None
        assert report.workflow_compatible is None
        assert report.model_capability_confidence == "unknown"
        for cap_name, cap in report.capabilities.items():
            assert cap["status"] == "UNKNOWN", f"{model_name} {cap_name} guessed non-unknown status"
            assert cap["verified"] is False


def test_missing_model_and_missing_tools_and_context_overflow() -> None:
    def missing_model_http(
        method: str,
        url: str,
        *,
        body: Dict[str, Any] | None = None,
        timeout: float = 2.0,
    ) -> tuple[int, Dict[str, Any]]:
        if url.endswith("/api/tags"):
            return 200, {"models": [{"name": "llama3:8b"}]}
        raise AssertionError("Should not query /api/show when model is not installed")

    cfg = ZelyrConfig(ollama_base_url="http://127.0.0.1:11434", ollama_model="qwen3:8b")
    missing_report = check_model_compatibility(config=cfg, http_request_fn=missing_model_http)
    assert missing_report.model_installed is False
    assert missing_report.overall_status == "DEGRADED"
    assert missing_report.workflow_compatible is False
    assert any("not listed" in g for g in missing_report.blocking_gaps)

    def no_tools_small_ctx_http(
        method: str,
        url: str,
        *,
        body: Dict[str, Any] | None = None,
        timeout: float = 2.0,
    ) -> tuple[int, Dict[str, Any]]:
        if url.endswith("/api/tags"):
            return 200, {"models": [{"name": "tiny-model:latest"}]}
        return 200, {
            "capabilities": ["completion"],
            "model_info": {"llama.context_length": 4096},
        }

    degraded_cfg = ZelyrConfig(
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="tiny-model",
        ollama_num_ctx=16384,
    )
    degraded_report = check_model_compatibility(config=degraded_cfg, http_request_fn=no_tools_small_ctx_http)
    assert degraded_report.model_installed is True
    assert degraded_report.capabilities["native_tool_calling"]["status"] == "UNAVAILABLE"
    assert degraded_report.capabilities["tool_argument_formatting"]["status"] == "DEGRADED"
    assert degraded_report.capabilities["context_limits"]["status"] == "DEGRADED"
    assert degraded_report.overall_status == "DEGRADED"
    assert degraded_report.workflow_compatible is True
    assert any("FallbackToolParser" in w for w in degraded_report.warnings)
    assert any("num_ctx=16384" in w and "4096" in w for w in degraded_report.warnings)


def test_unreachable_timeout_http_error_and_secret_scrubbing() -> None:
    def timeout_http(*args: Any, **kwargs: Any) -> tuple[int, Dict[str, Any]]:
        raise TimeoutError("timed out")

    cfg = ZelyrConfig(ollama_base_url="http://127.0.0.1:11434")
    timeout_report = check_model_compatibility(config=cfg, http_request_fn=timeout_http, timeout=0.2)
    assert timeout_report.overall_status == "UNAVAILABLE"
    assert timeout_report.workflow_compatible is False
    assert timeout_report.capabilities["provider_availability"]["status"] == "UNAVAILABLE"
    assert timeout_report.capabilities["streaming_behavior"]["status"] == "UNAVAILABLE"

    def leaky_error_http(*args: Any, **kwargs: Any) -> tuple[int, Dict[str, Any]]:
        raise RuntimeError("connect failed to http://alice:supersecret@127.0.0.1:11434 Bearer tok_secret123 sk-1234567890ab")

    leaky_report = check_model_compatibility(config=cfg, http_request_fn=leaky_error_http)
    assert leaky_report.overall_status == "UNAVAILABLE"
    rendered = format_model_compatibility_report(leaky_report)
    assert "supersecret" not in rendered
    assert "tok_secret123" not in rendered
    assert "sk-1234567890ab" not in rendered
    assert "[REDACTED]" in rendered

    def http_401(*args: Any, **kwargs: Any) -> tuple[int, Dict[str, Any]]:
        raise urllib.error.HTTPError(args[1], 401, "Unauthorized", {}, io.BytesIO(b""))

    auth_report = check_model_compatibility(config=cfg, http_request_fn=http_401)
    assert auth_report.overall_status == "DEGRADED"
    assert auth_report.workflow_compatible is False
    assert auth_report.capabilities["provider_availability"]["status"] == "DEGRADED"


def test_openai_compatible_protocol_observations_and_vision_workflow() -> None:
    def v1_http(method: str, url: str, *, body: Dict[str, Any] | None = None, timeout: float = 2.0) -> tuple[int, Dict[str, Any]]:
        assert url.endswith("/v1/models")
        return 200, {"data": [{"id": "local-coder"}]}

    v1_cfg = ZelyrConfig(ollama_base_url="http://127.0.0.1:4000/v1", ollama_model="local-coder")
    base_v1 = check_model_compatibility(config=v1_cfg, workflow="planned_graph", http_request_fn=v1_http)
    assert base_v1.provider_type == "openai_compatible"
    assert base_v1.model_installed is True
    assert base_v1.capabilities["streaming_behavior"]["status"] == "AVAILABLE"
    assert base_v1.capabilities["native_tool_calling"]["status"] == "UNKNOWN"
    assert base_v1.workflow_compatible is None

    verified_v1 = check_model_compatibility(
        config=v1_cfg,
        workflow="planned_graph",
        http_request_fn=v1_http,
        protocol_observations={
            "native_tool_calling": True,
            "tool_argument_formatting": True,
            "multi_turn_tool_call": True,
            "thinking_metadata": True,
            "streaming": True,
            "structured_output": True,
        },
    )
    assert verified_v1.overall_status == "AVAILABLE"
    assert verified_v1.workflow_compatible is True
    # Context window length is still unknown for OpenAI /v1/models without explicit observation.
    assert verified_v1.model_capability_confidence == "medium"
    assert verified_v1.capabilities["context_limits"]["status"] == "UNKNOWN"

    full_verified_v1 = check_model_compatibility(
        config=v1_cfg,
        workflow="planned_graph",
        http_request_fn=v1_http,
        protocol_observations={
            "native_tool_calling": True,
            "tool_argument_formatting": True,
            "multi_turn_tool_call": True,
            "thinking_metadata": True,
            "streaming": True,
            "structured_output": True,
            "context_length": 32768,
        },
    )
    assert full_verified_v1.model_capability_confidence == "high"
    assert full_verified_v1.capabilities["context_limits"]["status"] == "AVAILABLE"
    assert full_verified_v1.capabilities["multi_turn_tool_calling"]["source"] == "protocol_probe"
    assert full_verified_v1.capabilities["structured_output_support"]["source"] == "protocol_probe"

    broken_json = check_model_compatibility(
        config=v1_cfg,
        workflow="planned_graph",
        http_request_fn=v1_http,
        protocol_observations={"structured_output": False},
    )
    assert broken_json.overall_status == "DEGRADED"
    assert broken_json.workflow_compatible is False
    assert any("structured JSON output" in gap for gap in broken_json.blocking_gaps)

    def no_vision_http(method: str, url: str, *, body: Dict[str, Any] | None = None, timeout: float = 2.0) -> tuple[int, Dict[str, Any]]:
        if url.endswith("/api/tags"):
            return 200, {"models": [{"name": "text-only:latest"}]}
        return 200, {"capabilities": ["completion", "tools"]}

    vis_report = check_model_compatibility(
        config=ZelyrConfig(ollama_base_url="http://127.0.0.1:11434", ollama_model="text-only"),
        workflow="vision",
        http_request_fn=no_vision_http,
    )
    assert vis_report.overall_status == "DEGRADED"
    assert vis_report.workflow_compatible is False
    assert any("vision" in g.lower() for g in vis_report.blocking_gaps)


def test_cli_model_check_env_integration_and_top_level_exports(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import zelyr
    from zelyr.cli.main import main as cli_main

    assert zelyr.check_model_compatibility is check_model_compatibility
    assert zelyr.ModelCompatibilityDiagnostics is ModelCompatibilityDiagnostics
    assert zelyr.ModelCompatibilityReport is ModelCompatibilityReport

    monkeypatch.setenv("HOME", str(tmp_path))
    set_config(ZelyrConfig(ollama_base_url="http://127.0.0.1:11434", ollama_model="qwen3:8b"))

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    closed_port = sock.getsockname()[1]
    sock.close()

    try:
        monkeypatch.setattr(sys, "argv", ["zelyr", "model", "check", "--no-probe", "--json"])
        cli_main()
        out = json.loads(capsys.readouterr().out)
        assert out["overall_status"] == "UNKNOWN"
        assert out["workflow_compatible"] is None
        assert out["capabilities"]["native_tool_calling"]["status"] == "UNKNOWN"

        monkeypatch.setattr(sys, "argv", ["zelyr", "model", "check", "--no-probe", "--strict", "--json"])
        with pytest.raises(SystemExit) as exc:
            cli_main()
        assert exc.value.code == 1
        capsys.readouterr()

        monkeypatch.setattr(
            sys,
            "argv",
            ["zelyr", "model", "check", "--base-url", f"http://127.0.0.1:{closed_port}", "--timeout", "0.2", "--json"],
        )
        with pytest.raises(SystemExit) as unreach_exc:
            cli_main()
        assert unreach_exc.value.code == 1
        unreach_out = json.loads(capsys.readouterr().out)
        assert unreach_out["overall_status"] == "UNAVAILABLE"
        assert unreach_out["workflow_compatible"] is False

        monkeypatch.setattr(sys, "argv", ["zelyr", "model", "check", "--base-url", "ftp://bad.example.com"])
        with pytest.raises(SystemExit) as url_exc:
            cli_main()
        assert url_exc.value.code == 1
        assert "scheme" in capsys.readouterr().err.lower()

        monkeypatch.setattr(sys, "argv", ["zelyr", "env", "--no-probe", "--model-check", "--json"])
        cli_main()
        env_out = json.loads(capsys.readouterr().out)
        assert "model_compatibility" in env_out
        assert env_out["model_compatibility"]["overall_status"] == "UNKNOWN"
    finally:
        set_config(ZelyrConfig())
