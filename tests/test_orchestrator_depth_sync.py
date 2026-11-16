"""Tests keeping NativeOrchestrator max_steps and max_depth synchronized."""

import json
import unittest
from typing import Any, Dict, List

from zelyr.agents.native_orchestrator import NativeOrchestrator


class TestOrchestratorMaxStepsDepthSync(unittest.IsolatedAsyncioTestCase):
    async def test_mutating_max_steps_after_init_updates_effective_max_depth(self) -> None:
        turns_seen: List[int] = []
        executed_calls: List[tuple[str, Dict[str, Any]]] = []

        async def mock_stream(payload: dict):
            turn_idx = len(turns_seen)
            turns_seen.append(turn_idx)
            if turn_idx < 4:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": f"call_read_{turn_idx}",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": {"action": "read", "path": f"note_{turn_idx}.txt"},
                                },
                            }]
                        }
                    }]
                }
            elif turn_idx == 4:
                # Model stops with text at depth=4 before performing the requested write.
                yield {"choices": [{"delta": {"content": "I inspected note files."}}]}
            elif turn_idx == 5:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_write_5",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": {
                                        "action": "write",
                                        "path": "note.txt",
                                        "content": "hello",
                                    },
                                },
                            }]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Updated note.txt to hello."}}]}

        async def mock_exec(name: str, args: dict) -> str:
            executed_calls.append((name, dict(args)))
            if args.get("action") == "read":
                return json.dumps({"status": "ok", "action": "read", "path": args["path"], "content": "old\n"})
            return json.dumps({"status": "ok", "action": "write", "path": "note.txt"})

        orch = NativeOrchestrator(
            tool_schemas=[
                {
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
                }
            ],
            stream_provider=mock_stream,
            tool_executor=mock_exec,
        )
        # Simulates callers (CLI / live runner) overriding max_steps after construction.
        orch.max_steps = 8

        self.assertEqual(orch.max_depth, 8)

        payload = {
            "model": "qwen2.5:7b",
            "messages": [{"role": "user", "content": "Update note.txt to contain hello"}],
        }
        chunks = []
        async for chunk in orch.generate_stream(payload):
            chunks.append(chunk)

        self.assertIn(
            ("manage_workspace_file", {"action": "write", "path": "note.txt", "content": "hello"}),
            executed_calls,
        )
        self.assertGreaterEqual(len(turns_seen), 6)


if __name__ == "__main__":
    unittest.main()
