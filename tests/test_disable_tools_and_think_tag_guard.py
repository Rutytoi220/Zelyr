"""Regression tests for disable_tools fallback suppression and orphan </think> prevention."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.engine.execution_ledger import TurnExecutionLedger


class TestDisableToolsAndThinkTagGuard(unittest.IsolatedAsyncioTestCase):
    async def test_disable_tools_ignores_fallback_text_tool_calls_after_verification(self) -> None:
        executed_calls: List[tuple[str, Dict[str, Any]]] = []
        turn_tools: List[int] = []

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "alpha.txt"

            async def mock_stream(payload: dict):
                turn = len(turn_tools)
                turn_tools.append(len(payload.get("tools") or []))
                if turn == 0:
                    yield {
                        "choices": [{
                            "delta": {
                                "tool_calls": [{
                                    "index": 0,
                                    "id": "call_write_1",
                                    "type": "function",
                                    "function": {
                                        "name": "manage_workspace_file",
                                        "arguments": {"action": "write", "path": str(target), "content": "DONE"},
                                    },
                                }]
                            }
                        }]
                    }
                else:
                    # Even with tools=[], a small model may still emit JSON tool syntax in text.
                    yield {
                        "choices": [{
                            "delta": {
                                "content": json.dumps({
                                    "name": "manage_workspace_file",
                                    "arguments": {"action": "write", "path": str(target), "content": "DONE"},
                                })
                            }
                        }]
                    }

            async def mock_exec(name: str, args: dict) -> str:
                executed_calls.append((name, dict(args)))
                target.write_text(str(args["content"]), encoding="utf-8")
                return json.dumps({"status": "ok", "action": "write", "path": str(target)})

            ledger = TurnExecutionLedger()
            orch = NativeOrchestrator(
                tool_schemas=[{
                    "type": "function",
                    "function": {
                        "name": "manage_workspace_file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string"},
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                        },
                    },
                }],
                stream_provider=mock_stream,
                tool_executor=mock_exec,
            )

            payload = {
                "model": "qwen3:4b",
                "messages": [{"role": "user", "content": f"Create {target} containing DONE"}],
                "acceptance_criteria": f"file_contains:{target}:DONE",
            }
            async for _ in orch.generate_stream(payload, ledger=ledger):
                pass

            self.assertEqual(len(executed_calls), 1)
            self.assertEqual(len(turn_tools), 2)
            self.assertEqual(turn_tools[1], 0)
            self.assertEqual(ledger.total_actions(), 1)
            self.assertEqual(ledger.failed_actions(), [])
            self.assertEqual(ledger.unresolved_failures(), [])

    async def test_disable_tools_stays_sticky_when_rejected_call_triggers_false_success_gate(self) -> None:
        tools_per_turn: List[int] = []

        async def mock_stream(payload: dict):
            turn = len(tools_per_turn)
            tools_per_turn.append(len(payload.get("tools") or []))
            if turn == 0:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_step_0",
                                "type": "function",
                                "function": {"name": "step_tool", "arguments": {"n": 0}},
                            }]
                        }
                    }]
                }
            elif turn == 1:
                # Duplicate call triggers redundant_action rejection with a remaining failure in ledger.
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_step_1",
                                "type": "function",
                                "function": {"name": "step_tool", "arguments": {"n": 0}},
                            }]
                        }
                    }]
                }
            elif turn == 2:
                # False-success phrasing on unresolved failure invokes recursive retry.
                yield {"choices": [{"delta": {"content": "Finished without tools."}}]}
            else:
                yield {"choices": [{"delta": {"content": "Unable to proceed because tools are disabled."}}]}

        async def mock_exec(name: str, args: dict) -> str:
            return json.dumps({"status": "ok"})

        orch = NativeOrchestrator(
            tool_schemas=[{"type": "function", "function": {"name": "step_tool", "parameters": {"type": "object"}}}],
            stream_provider=mock_stream,
            tool_executor=mock_exec,
        )
        orch.max_steps = 4

        payload = {"model": "qwen3:4b", "messages": [{"role": "user", "content": "Say done"}]}
        async for _ in orch.generate_stream(payload, disable_tools=True):
            pass

        self.assertGreaterEqual(len(tools_per_turn), 3)
        self.assertTrue(all(c == 0 for c in tools_per_turn))

    async def test_no_orphan_closing_think_tag_yielded_before_tool_call(self) -> None:
        captured_payloads: List[dict] = []

        async def mock_stream(payload: dict):
            turn = len(captured_payloads)
            captured_payloads.append(json.loads(json.dumps(payload)))
            if turn == 0:
                yield {"choices": [{"delta": {"reasoning_content": "Planning tool action..."}}]}
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_think_1",
                                "type": "function",
                                "function": {"name": "check_tool", "arguments": {}},
                            }]
                        }
                    }]
                }
                yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}
            else:
                yield {"choices": [{"delta": {"content": "Completed cleanly."}}]}

        async def mock_exec(name: str, args: dict) -> str:
            return json.dumps({"status": "ok"})

        orch = NativeOrchestrator(
            tool_schemas=[{"type": "function", "function": {"name": "check_tool", "parameters": {"type": "object"}}}],
            stream_provider=mock_stream,
            tool_executor=mock_exec,
        )
        payload = {"model": "qwen3:4b", "messages": [{"role": "user", "content": "Run check_tool"}]}
        streamed_pieces: List[str] = []
        async for chunk in orch.generate_stream(payload):
            for ch in chunk.get("choices", []):
                txt = ch.get("delta", {}).get("content", "")
                if txt:
                    streamed_pieces.append(txt)

        streamed_text = "".join(streamed_pieces)
        self.assertNotIn("</think>", streamed_text)
        self.assertNotIn("<think>", streamed_text)
        self.assertIn("Completed cleanly.", streamed_text)

        assistant_msg = next(m for m in captured_payloads[1]["messages"] if m.get("role") == "assistant")
        self.assertIn("<think>Planning tool action...</think>", assistant_msg["content"])


if __name__ == "__main__":
    unittest.main()
