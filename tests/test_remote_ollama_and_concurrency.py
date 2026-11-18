"""Tests for remote Ollama endpoint configuration, bounded keep_alive, and concurrent isolation."""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from zelyr.agents.native_orchestrator import NativeOrchestrator
import zelyr.agents.native_orchestrator as nat_orch
from zelyr.cli.main import _build_parser, _execute_headless_run
from zelyr.config import ZelyrConfig, get_config


class _FakeStreamResponse:
    def __init__(self, lines, status_code=200):
        self.status_code = status_code
        self._lines = lines
        self.text = ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    async def aread(self):
        return b""

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakePostResponse:
    def __init__(self, status_code=200):
        self.status_code = status_code


class TestRemoteOllamaConfig:
    def test_defaults_remain_local_and_bounded(self, monkeypatch):
        monkeypatch.delenv("ZELYR_OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("AXIOM_OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("ZELYR_OLLAMA_KEEP_ALIVE", raising=False)
        cfg = ZelyrConfig()
        assert cfg.ollama_base_url == "http://127.0.0.1:11434"
        assert cfg.get_effective_ollama_base_url() == "http://127.0.0.1:11434"
        assert cfg.ollama_keep_alive == "5m"
        assert cfg.get_effective_keep_alive() == "5m"

    def test_base_url_precedence_and_normalization(self, monkeypatch):
        cfg = ZelyrConfig(ollama_base_url="http://192.0.2.50:11434/")
        assert cfg.get_effective_ollama_base_url() == "http://192.0.2.50:11434"

        monkeypatch.setenv("OLLAMA_HOST", "192.0.2.60:11434/")
        assert cfg.get_effective_ollama_base_url() == "http://192.0.2.60:11434"

        monkeypatch.setenv("ZELYR_OLLAMA_BASE_URL", "http://192.0.2.70:11434/")
        assert cfg.get_effective_ollama_base_url() == "http://192.0.2.70:11434"

        assert (
            cfg.get_effective_ollama_base_url(caller_override="http://192.0.2.80:11434/")
            == "http://192.0.2.80:11434"
        )

    def test_keep_alive_precedence_honors_zero(self, monkeypatch):
        cfg = ZelyrConfig(ollama_keep_alive="2m")
        assert cfg.get_effective_keep_alive() == "2m"
        assert cfg.get_effective_keep_alive(caller_override=0) == 0

        monkeypatch.setenv("ZELYR_OLLAMA_KEEP_ALIVE", "90s")
        assert cfg.get_effective_keep_alive() == "90s"
        assert cfg.get_effective_keep_alive(caller_override="10s") == "10s"

    def test_read_timeout_precedence_and_fallback(self, monkeypatch):
        monkeypatch.delenv("ZELYR_OLLAMA_READ_TIMEOUT", raising=False)
        cfg = ZelyrConfig()
        assert cfg.ollama_read_timeout == 300.0
        assert cfg.get_effective_read_timeout() == 300.0

        cfg_slow = ZelyrConfig(ollama_read_timeout=1200.0)
        assert cfg_slow.get_effective_read_timeout() == 1200.0

        monkeypatch.setenv("ZELYR_OLLAMA_READ_TIMEOUT", "1500")
        assert cfg_slow.get_effective_read_timeout() == 1500.0
        assert cfg_slow.get_effective_read_timeout(caller_override=900) == 900.0

        monkeypatch.setenv("ZELYR_OLLAMA_READ_TIMEOUT", "invalid")
        assert cfg_slow.get_effective_read_timeout() == 1200.0

    def test_cli_run_accepts_base_url_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "hello", "--base-url", "http://192.0.2.10:11434"])
        assert args.base_url == "http://192.0.2.10:11434"


class TestRemoteHttpDispatchAndConcurrencyIsolation:
    @pytest.mark.asyncio
    async def test_orchestrator_uses_instance_config_base_url_and_keep_alive(self, monkeypatch):
        monkeypatch.delenv("ZELYR_OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        captured_posts = []
        captured_streams = []
        captured_timeouts = []

        class MockAsyncClient:
            def __init__(self, *args, **kwargs):
                captured_timeouts.append(kwargs.get("timeout"))

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return False

            async def post(self, url, json=None, **kwargs):
                captured_posts.append((url, json))
                return _FakePostResponse(200)

            def stream(self, method, url, json=None, **kwargs):
                captured_streams.append((method, url, json))
                lines = [
                    '{"message":{"role":"assistant","content":"Remote response ready."},"done":true,"done_reason":"stop"}'
                ]
                return _FakeStreamResponse(lines, status_code=200)

        remote_cfg = ZelyrConfig(
            ollama_base_url="http://192.0.2.10:11434/",
            ollama_num_ctx=4096,
            ollama_keep_alive="90s",
            ollama_read_timeout=1200.0,
        )
        orch = NativeOrchestrator(config=remote_cfg)
        assert orch.max_context_tokens == 4096

        payload = {
            "model": "qwen2.5:3b",
            "messages": [{"role": "user", "content": "Say hello"}],
            "tools": [],
        }

        with patch("httpx.AsyncClient", MockAsyncClient):
            chunks = [c async for c in orch.generate_stream(payload)]

        assert any("Remote response ready." in str(c) for c in chunks)
        assert captured_posts, "Expected prewarm /api/generate POST call"
        assert captured_posts[0][0] == "http://192.0.2.10:11434/api/generate"
        assert captured_posts[0][1]["options"]["num_ctx"] == 4096
        assert captured_posts[0][1]["keep_alive"] == "90s"

        assert captured_streams, "Expected /api/chat stream POST call"
        assert captured_streams[0][1] == "http://192.0.2.10:11434/api/chat"
        assert captured_streams[0][2]["options"]["num_ctx"] == 4096
        assert captured_streams[0][2]["keep_alive"] == "90s"
        assert any(getattr(t, "read", None) == 1200.0 for t in captured_timeouts)

    @pytest.mark.asyncio
    async def test_read_timeout_error_surfaces_exception_class_name(self):
        import httpx

        class TimeoutAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return False

            async def post(self, url, json=None, **kwargs):
                return _FakePostResponse(200)

            def stream(self, method, url, json=None, **kwargs):
                raise httpx.ReadTimeout("")

        orch = NativeOrchestrator(config=ZelyrConfig(ollama_base_url="http://192.0.2.10:11434"))
        with patch("httpx.AsyncClient", TimeoutAsyncClient):
            chunks = [
                c async for c in orch.generate_stream(
                    {"model": "qwen2.5:3b", "messages": [{"role": "user", "content": "hi"}], "tools": []}
                )
            ]
        joined = "".join(c.get("choices", [{}])[0].get("delta", {}).get("content", "") for c in chunks)
        assert "ReadTimeout" in joined

    @pytest.mark.asyncio
    async def test_concurrent_headless_runs_isolate_failure_ledgers_and_globals(self, monkeypatch):
        monkeypatch.setenv("ZELYR_TESTING", "1")
        monkeypatch.setenv("ZELYR_AUTOPILOT", "1")
        orig_global_model = get_config().ollama_model
        orig_module_execute = nat_orch.execute_tool

        with tempfile.TemporaryDirectory() as tmpdir:
            ws_a = Path(tmpdir) / "ws_a"
            ws_b = Path(tmpdir) / "ws_b"
            ws_a.mkdir()
            ws_b.mkdir()
            file_a = ws_a / "alpha.txt"
            file_b = ws_b / "beta.txt"

            turn_counts = {"model-alpha": 0, "model-beta": 0}
            seen_urls = {}

            class MockAsyncClient:
                def __init__(self, *args, **kwargs):
                    pass

                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    return False

                async def post(self, url, json=None, **kwargs):
                    return _FakePostResponse(200)

                def stream(self, method, url, json=None, **kwargs):
                    model = json["model"]
                    seen_urls.setdefault(model, []).append(url)
                    turn_counts[model] += 1
                    idx = turn_counts[model]
                    if model == "model-alpha":
                        if idx == 1:
                            tc = {
                                "message": {
                                    "role": "assistant",
                                    "content": "",
                                    "tool_calls": [
                                        {
                                            "id": "call_a1",
                                            "type": "function",
                                            "function": {
                                                "name": "manage_workspace_file",
                                                "arguments": {
                                                    "action": "write",
                                                    "path": str(file_a),
                                                    "content": "ALPHA_OK",
                                                },
                                            },
                                        }
                                    ],
                                },
                                "done": False,
                            }
                            return _FakeStreamResponse(
                                [json_dumps(tc), '{"message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}']
                            )
                        return _FakeStreamResponse(
                            ['{"message":{"role":"assistant","content":"Alpha done."},"done":true,"done_reason":"stop"}']
                        )
                    else:
                        if idx == 1:
                            tc = {
                                "message": {
                                    "role": "assistant",
                                    "content": "",
                                    "tool_calls": [
                                        {
                                            "id": "call_b1",
                                            "type": "function",
                                            "function": {
                                                "name": "manage_workspace_file",
                                                "arguments": {
                                                    "action": "write",
                                                    "path": str(file_b),
                                                    "content": "WRONG_TOKEN",
                                                },
                                            },
                                        }
                                    ],
                                },
                                "done": False,
                            }
                            return _FakeStreamResponse(
                                [json_dumps(tc), '{"message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}']
                            )
                        return _FakeStreamResponse(
                            ['{"message":{"role":"assistant","content":"Beta done."},"done":true,"done_reason":"stop"}']
                        )

            def json_dumps(obj):
                return json.dumps(obj)

            cfg_a = ZelyrConfig(ollama_base_url="http://192.0.2.11:11434", ollama_num_ctx=4096)
            cfg_b = ZelyrConfig(ollama_base_url="http://192.0.2.12:11434", ollama_num_ctx=4096)
            orch_a = NativeOrchestrator(max_steps=4, config=cfg_a)
            orch_a.max_replanning_attempts = 1
            orch_b = NativeOrchestrator(max_steps=1, config=cfg_b)
            orch_b.max_replanning_attempts = 0

            with patch("httpx.AsyncClient", MockAsyncClient):
                code_a, code_b = await asyncio.gather(
                    _execute_headless_run(
                        prompt=f"Write ALPHA_OK to {file_a}",
                        model_override="model-alpha",
                        output_json=True,
                        orchestrator=orch_a,
                        verify_spec=f"file_contains:{file_a}:ALPHA_OK",
                    ),
                    _execute_headless_run(
                        prompt=f"Write WRONG_TOKEN to {file_b}",
                        model_override="model-beta",
                        output_json=True,
                        orchestrator=orch_b,
                        verify_spec=f"file_contains:{file_b}:REQUIRED_MISSING_TOKEN",
                    ),
                )

            assert code_a == 0
            assert code_b == 2
            assert file_a.read_text(encoding="utf-8") == "ALPHA_OK"
            assert file_b.read_text(encoding="utf-8") == "WRONG_TOKEN"
            assert seen_urls["model-alpha"] == [
                "http://192.0.2.11:11434/api/chat",
                "http://192.0.2.11:11434/api/chat",
            ]
            assert seen_urls["model-beta"][0] == "http://192.0.2.12:11434/api/chat"

            assert orch_a.current_ledger is not None
            assert orch_b.current_ledger is not None
            assert not orch_a.current_ledger.has_unresolved_failures()
            assert all(not r["verification_failed"] for r in orch_a.current_ledger.to_dict()["records"])
            assert orch_b.current_ledger.has_unresolved_failures()
            assert any(r["verification_failed"] for r in orch_b.current_ledger.to_dict()["records"])

            assert get_config().ollama_model == orig_global_model
            assert nat_orch.execute_tool is orig_module_execute
