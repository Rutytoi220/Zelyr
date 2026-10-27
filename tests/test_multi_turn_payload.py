"""Test suite for authoritative context configuration and proven multi-turn Ollama protocol compliance."""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from typing import Any, AsyncGenerator, Dict, List
from unittest.mock import patch

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.config import ZelyrConfig, get_config, get_effective_num_ctx, set_config, validate_and_clamp_num_ctx


class TestMultiTurnPayload(unittest.IsolatedAsyncioTestCase):
    """Verifies authoritative context precedence, Turn 2 serialization, and thinking preservation."""

    def setUp(self) -> None:
        self.original_env = os.environ.get("ZELYR_NUM_CTX")
        if "ZELYR_NUM_CTX" in os.environ:
            del os.environ["ZELYR_NUM_CTX"]
        self.original_config = get_config()
        self.test_config = ZelyrConfig()
        set_config(self.test_config)

    def tearDown(self) -> None:
        if self.original_env is not None:
            os.environ["ZELYR_NUM_CTX"] = self.original_env
        elif "ZELYR_NUM_CTX" in os.environ:
            del os.environ["ZELYR_NUM_CTX"]
        set_config(self.original_config)

    def test_context_precedence_hierarchy(self) -> None:
        """Verify strict 4-tier context hierarchy and clamping validation."""
        # Tier 4: Default system fallback is 8192
        self.assertEqual(get_effective_num_ctx(), 8192)
        orch_default = NativeOrchestrator()
        self.assertEqual(orch_default.max_context_tokens, 8192)

        # Tier 3: Config setting in ZelyrConfig
        self.test_config.ollama_num_ctx = 4096
        self.assertEqual(get_effective_num_ctx(), 4096)
        self.assertEqual(orch_default.max_context_tokens, 4096)

        # Tier 2: Environment variable ZELYR_NUM_CTX overrides config
        os.environ["ZELYR_NUM_CTX"] = "16384"
        self.assertEqual(get_effective_num_ctx(), 16384)
        self.assertEqual(orch_default.max_context_tokens, 16384)

        # Tier 1: Explicit caller override overrides environment and config
        self.assertEqual(get_effective_num_ctx(caller_override=32768), 32768)

        # Explicit constructor argument takes precedence on orchestrator instance
        orch_explicit = NativeOrchestrator(max_context_tokens=12288)
        self.assertEqual(orch_explicit.max_context_tokens, 12288)

        # Validation and clamping
        self.assertEqual(validate_and_clamp_num_ctx(None), 8192)
        self.assertEqual(validate_and_clamp_num_ctx(500), 8192)  # < 1024
        self.assertEqual(validate_and_clamp_num_ctx(100000), 8192)  # > 65536
        self.assertEqual(validate_and_clamp_num_ctx("not_a_number"), 8192)
        self.assertEqual(validate_and_clamp_num_ctx(True), 8192)  # Boolean rejected
        self.assertEqual(validate_and_clamp_num_ctx(8192), 8192)
        self.assertEqual(validate_and_clamp_num_ctx("16384"), 16384)

    async def test_proven_turn2_reproduction_and_fix(self) -> None:
        """Prove that Turn 2 outgoing payload formats function.arguments as a native dict, not str."""
        captured_payloads: List[Dict[str, Any]] = []

        # Turn 1: Model emits tool call where arguments arrives as a dict (Ollama style)
        # Turn 2: Model finishes with final answer
        turn_counter = 0

        async def mock_stream_provider(payload: dict) -> AsyncGenerator[dict, None]:
            nonlocal turn_counter
            captured_payloads.append(json.loads(json.dumps(payload)))
            turn_counter += 1

            if turn_counter == 1:
                # Simulate Ollama streaming tool call
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_file_1",
                                "type": "function",
                                "function": {
                                    "name": "mock_read",
                                    "arguments": {"path": "crash.log"},
                                }
                            }]
                        }
                    }]
                }
            else:
                yield {
                    "choices": [{
                        "delta": {
                            "content": "Analysis complete: file exists."
                        }
                    }]
                }

        async def mock_tool_executor(name: str, args: dict) -> str:
            self.assertEqual(name, "mock_read")
            self.assertEqual(args, {"path": "crash.log"})
            return json.dumps({"status": "ok", "content": "Sample crash trace"})

        orchestrator = NativeOrchestrator(
            stream_provider=mock_stream_provider,
            tool_executor=mock_tool_executor,
            max_depth=3,
        )

        initial_payload = {
            "model": "qwen3:8b",
            "messages": [{"role": "user", "content": "Read crash.log"}],
        }

        output_chunks = []
        async for chunk in orchestrator.generate_stream(initial_payload):
            output_chunks.append(chunk)

        self.assertEqual(len(captured_payloads), 2)

        # Inspect Turn 2 payload
        turn2_payload = captured_payloads[1]
        messages = turn2_payload["messages"]

        # Turn 2 must include: System prompt (0), User query (1), Assistant tool call (2), Tool result (3)
        self.assertGreaterEqual(len(messages), 4)

        assistant_msg = next(m for m in messages if m.get("role") == "assistant")
        self.assertIn("tool_calls", assistant_msg)
        self.assertEqual(len(assistant_msg["tool_calls"]), 1)

        tc = assistant_msg["tool_calls"][0]
        # PROOF: function.arguments MUST be a native Python dict, NOT a stringified dict
        self.assertIsInstance(tc["function"]["arguments"], dict)
        self.assertEqual(tc["function"]["arguments"], {"path": "crash.log"})
        self.assertNotIsInstance(tc["function"]["arguments"], str)
        self.assertEqual(tc["id"], "call_file_1")
        self.assertEqual(assistant_msg["content"], "")

        # Verify tool result message
        tool_msg = next(m for m in messages if m.get("role") == "tool")
        self.assertEqual(tool_msg["tool_call_id"], "call_file_1")
        self.assertEqual(tool_msg["name"], "mock_read")

        # Verify num_ctx is set authoritatively to 8192
        self.assertEqual(turn2_payload["options"]["num_ctx"], 8192)

    async def test_thinking_metadata_preservation(self) -> None:
        """Verify thinking tokens (<think>...</think>) are preserved in Turn 2 assistant message."""
        captured_payloads: List[Dict[str, Any]] = []
        turn_counter = 0

        async def mock_stream_provider(payload: dict) -> AsyncGenerator[dict, None]:
            nonlocal turn_counter
            captured_payloads.append(json.loads(json.dumps(payload)))
            turn_counter += 1

            if turn_counter == 1:
                # Emit thinking tags first, then tool call
                yield {
                    "choices": [{
                        "delta": {
                            "reasoning_content": "Inspecting system diagnostics..."
                        }
                    }]
                }
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_diag_1",
                                "type": "function",
                                "function": {
                                    "name": "check_diag",
                                    "arguments": {"subsystem": "gpu"},
                                }
                            }]
                        }
                    }]
                }
            else:
                yield {
                    "choices": [{
                        "delta": {
                            "content": "GPU status is nominal."
                        }
                    }]
                }

        async def mock_tool_executor(name: str, args: dict) -> str:
            return json.dumps({"status": "nominal"})

        orchestrator = NativeOrchestrator(
            stream_provider=mock_stream_provider,
            tool_executor=mock_tool_executor,
            max_depth=3,
        )

        initial_payload = {
            "model": "qwen3:8b",
            "messages": [{"role": "user", "content": "Check GPU health"}],
        }

        async for _ in orchestrator.generate_stream(initial_payload):
            pass

        self.assertEqual(len(captured_payloads), 2)
        turn2_payload = captured_payloads[1]
        assistant_msg = next(m for m in turn2_payload["messages"] if m.get("role") == "assistant")

        # Verify thinking tags are preserved
        self.assertIn("<think>", assistant_msg["content"])
        self.assertIn("Inspecting system diagnostics...", assistant_msg["content"])
        self.assertIn("</think>", assistant_msg["content"])

        # Tool calls arguments must still be a dict
        self.assertIsInstance(assistant_msg["tool_calls"][0]["function"]["arguments"], dict)

    async def test_mock_ollama_multiturn_react_e2e(self) -> None:
        """Full 2-turn ReAct test ensuring caller overrides and tool execution complete seamlessly."""
        captured_num_ctx_values = []

        async def mock_stream_provider(payload: dict) -> AsyncGenerator[dict, None]:
            captured_num_ctx_values.append(payload["options"]["num_ctx"])
            messages = payload["messages"]

            if len(messages) <= 2:
                # Turn 1
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_2turn",
                                "type": "function",
                                "function": {
                                    "name": "echo_tool",
                                    "arguments": "{\"text\": \"hello\"}",  # Stringified JSON
                                }
                            }]
                        }
                    }]
                }
            else:
                # Turn 2
                yield {
                    "choices": [{
                        "delta": {
                            "content": "Done: tool returned success."
                        }
                    }]
                }

        async def mock_tool_executor(name: str, args: dict) -> str:
            return f"Echo: {args.get('text')}"

        orchestrator = NativeOrchestrator(
            stream_provider=mock_stream_provider,
            tool_executor=mock_tool_executor,
            max_depth=3,
        )

        # Pass caller override in options
        payload = {
            "model": "qwen3:8b",
            "messages": [{"role": "user", "content": "Run echo"}],
            "options": {"num_ctx": 12288},
        }

        final_text = []
        async for chunk in orchestrator.generate_stream(payload):
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if "content" in delta and delta["content"]:
                final_text.append(delta["content"])

        # Both turns must have respected caller override num_ctx 12288
        self.assertEqual(captured_num_ctx_values, [12288, 12288])
        self.assertIn("Done: tool returned success.", "".join(final_text))


if __name__ == "__main__":
    unittest.main()
