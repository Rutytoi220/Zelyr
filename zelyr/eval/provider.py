"""Deterministic mock model provider for reproducible, zero-network evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Union


@dataclass
class MockTurn:
    """Represents the pre-scripted output chunks for a single model request turn."""
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    exception: Optional[Exception] = None

    @classmethod
    def text(cls, content: str, chunk_size: Optional[int] = None) -> MockTurn:
        """Create a mock turn that streams text content."""
        if not content:
            return cls(chunks=[{"choices": [{"delta": {"content": ""}}]}])
        
        if chunk_size and chunk_size > 0:
            chunks = []
            for i in range(0, len(content), chunk_size):
                chunks.append({"choices": [{"delta": {"content": content[i : i + chunk_size]}}]})
            return cls(chunks=chunks)

        return cls(chunks=[{"choices": [{"delta": {"content": content}}]}])

    @classmethod
    def tool_call(
        cls,
        name: str,
        arguments: Union[Dict[str, Any], str],
        call_id: str = "call_mock_1",
        thought: Optional[str] = None,
    ) -> MockTurn:
        """Create a mock turn that triggers a tool call (with optional preceding reasoning)."""
        chunks = []
        if thought:
            chunks.append({"choices": [{"delta": {"content": f"<think>{thought}</think>"}}]})

        args_str = arguments if isinstance(arguments, str) else json.dumps(arguments)
        tc_delta = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": args_str,
                                },
                            }
                        ]
                    }
                }
            ]
        }
        chunks.append(tc_delta)
        return cls(chunks=chunks)

    @classmethod
    def refusal(cls, reason: str = "I cannot fulfill this request.") -> MockTurn:
        """Create a mock turn indicating conversational refusal."""
        return cls(chunks=[{"choices": [{"delta": {"content": reason}}]}])

    @classmethod
    def error(cls, exception: Exception) -> MockTurn:
        """Create a mock turn that raises an exception upon invocation."""
        return cls(chunks=[], exception=exception)

    @classmethod
    def raw_chunks(cls, chunks: List[Dict[str, Any]]) -> MockTurn:
        """Create a mock turn from explicit raw chunks."""
        return cls(chunks=list(chunks))


class DeterministicModelProvider:
    """Mock model provider fulfilling stream_chat() without external network/Ollama access."""

    def __init__(
        self,
        turns: Optional[List[Union[MockTurn, Dict[str, Any], str]]] = None,
        default_response: Optional[str] = None,
    ):
        self.turns: List[MockTurn] = []
        if turns:
            for t in turns:
                if isinstance(t, MockTurn):
                    self.turns.append(t)
                elif isinstance(t, str):
                    self.turns.append(MockTurn.text(t))
                elif isinstance(t, dict):
                    # Could be raw chunk or dict
                    self.turns.append(MockTurn.raw_chunks([t]))
                else:
                    raise TypeError(f"Unsupported turn specification: {type(t)}")

        self.default_response = default_response
        self.request_count: int = 0
        self.recorded_payloads: List[Dict[str, Any]] = []
        self._total_prompt_chars: int = 0
        self._total_completion_chars: int = 0

    @property
    def estimated_tokens(self) -> Dict[str, int]:
        """Estimated token counts using a standard 4 chars/token heuristic.
        
        Note: These are explicitly labeled as estimated_tokens, never masqueraded
        as measured GPU/Ollama tokens.
        """
        p_tok = self._total_prompt_chars // 4
        c_tok = self._total_completion_chars // 4
        return {
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "total_tokens": p_tok + c_tok,
        }

    async def stream_chat(self, payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream chunks for the current turn to NativeOrchestrator."""
        self.recorded_payloads.append(payload)
        current_idx = self.request_count
        self.request_count += 1

        # Count prompt characters for token estimation
        try:
            for msg in payload.get("messages", []):
                content = msg.get("content", "")
                if isinstance(content, str):
                    self._total_prompt_chars += len(content)
        except Exception:
            pass

        if current_idx < len(self.turns):
            turn = self.turns[current_idx]
            if turn.exception is not None:
                raise turn.exception
            for chunk in turn.chunks:
                # Track completion chars
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                c = delta.get("content", "")
                if c:
                    self._total_completion_chars += len(c)
                yield chunk
        else:
            fallback = (
                self.default_response
                if self.default_response is not None
                else f"[DeterministicModelProvider: Completed {self.request_count} requests, no further turns configured.]"
            )
            self._total_completion_chars += len(fallback)
            yield {"choices": [{"delta": {"content": fallback}}]}
