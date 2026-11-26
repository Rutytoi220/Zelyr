"""Adaptive Context Window Budgeting and Historical Tool Compaction Engine.

Provides hierarchical context management for long-running multi-turn sessions
and deep ReAct agent loops:
1. Model Context Catalog: Discovers effective context limits across model families.
2. Stage 1 Compaction: Summarizes verbose historical tool outputs older than recent turns.
3. Stage 2 Turn Pruning: Sliding window turn eviction with semantic preamble summarization.
4. Fast Path: Sub-millisecond zero-allocation pass-through for conversations <50% capacity.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Standard heuristic: 1 token ≈ 1.3 words or ~4 characters.
_TOKENS_PER_WORD = 1.3

# ---------------------------------------------------------------------------
# Model Context Window Catalog
# ---------------------------------------------------------------------------

MODEL_CONTEXT_CATALOG: Dict[str, int] = {
    # Qwen family
    "qwen2.5-coder": 32768,
    "qwen2.5": 32768,
    "qwen2": 32768,
    "qwen3": 32768,
    "qwen": 32768,
    # Llama 3 family
    "llama-3.3": 131072,
    "llama-3.2": 131072,
    "llama-3.1": 131072,
    "llama3.3": 131072,
    "llama3.2": 131072,
    "llama3.1": 131072,
    "llama-3": 8192,
    "llama3": 8192,
    "llama-2": 4096,
    "llama2": 4096,
    # Mistral family
    "mistral-nemo": 128000,
    "mistral-large": 128000,
    "mistral-small": 32768,
    "codestral": 32768,
    "mixtral": 32768,
    "mistral": 32768,
    # DeepSeek family
    "deepseek-r1": 65536,
    "deepseek-coder": 65536,
    "deepseek-v3": 65536,
    "deepseek": 65536,
    # OpenAI family
    "gpt-4o-mini": 128000,
    "gpt-4o": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16384,
    "o1-preview": 128000,
    "o1-mini": 128000,
    "o1": 128000,
    "o3-mini": 128000,
    # Anthropic family
    "claude-3-5-sonnet": 200000,
    "claude-3-5-haiku": 200000,
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
    "claude-3": 200000,
    # Google family
    "gemini-2.0-flash": 1048576,
    "gemini-1.5-pro": 2097152,
    "gemini-1.5-flash": 1048576,
    "gemini-pro": 32768,
    # Microsoft / Others
    "phi-4": 16384,
    "phi-3.5": 128000,
    "phi-3": 128000,
    "phi-2": 2048,
    "gemma-2": 8192,
    "gemma2": 8192,
    "gemma": 8192,
}


def get_model_context_limit(model_name: Optional[str], default_limit: int = 8192) -> int:
    """Resolve effective context window limit for a model name from catalog."""
    if not model_name:
        return default_limit

    clean = model_name.lower().strip()
    if "/" in clean:
        clean = clean.split("/")[-1]
    base_name = clean.split(":")[0]

    # 1. Exact match
    if clean in MODEL_CONTEXT_CATALOG:
        return MODEL_CONTEXT_CATALOG[clean]
    if base_name in MODEL_CONTEXT_CATALOG:
        return MODEL_CONTEXT_CATALOG[base_name]

    # 2. Substring search
    for key, limit in MODEL_CONTEXT_CATALOG.items():
        if key in clean or clean in key:
            return limit

    return default_limit


def estimate_tokens(text: str) -> int:
    """Estimate token count from text using standard word-and-character heuristic."""
    if not text:
        return 0
    words = len(text.split())
    return max(len(text) // 4, int(words * _TOKENS_PER_WORD), 1)


def count_message_tokens(msg: Dict[str, Any]) -> int:
    """Estimate token footprint of a chat completion message dictionary."""
    content = msg.get("content", "")
    tokens = 0
    if isinstance(content, str):
        tokens = estimate_tokens(content)
    elif isinstance(content, list):
        tokens = sum(estimate_tokens(c.get("text", "")) for c in content if isinstance(c, dict))
    elif isinstance(content, dict):
        tokens = estimate_tokens(json.dumps(content, default=str))
    else:
        tokens = estimate_tokens(str(content))

    if "tool_calls" in msg and msg["tool_calls"]:
        tokens += estimate_tokens(json.dumps(msg["tool_calls"], default=str))

    return tokens + 4


def estimate_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    """Estimate total tokens across a list of chat messages using word/char heuristic."""
    return sum(count_message_tokens(m) for m in messages)


def estimate_tools_tokens(tools: Optional[List[Dict[str, Any]]]) -> int:
    """Estimate token footprint of serialized tool schema definitions."""
    if not tools:
        return 0
    return estimate_tokens(json.dumps(tools, default=str)) + (8 * len(tools))


class ContextBudgetLimitError(ValueError):
    """Structured error raised when required context exceeds the effective token budget."""

    def __init__(
        self,
        required_tokens: int,
        target_budget: int,
        max_context_tokens: int,
        reserve_tokens: int,
        message: Optional[str] = None,
    ) -> None:
        self.error_type = "CONTEXT_BUDGET_EXCEEDED"
        self.required_tokens = required_tokens
        self.target_budget = target_budget
        self.max_context_tokens = max_context_tokens
        self.reserve_tokens = reserve_tokens
        msg = message or (
            f"Required prompt context (~{required_tokens} estimated tokens) exceeds effective budget "
            f"({target_budget} tokens; max={max_context_tokens}, reserve={reserve_tokens})."
        )
        super().__init__(msg)

    def to_dict(self) -> Dict[str, Any]:
        """Return structured limit error dictionary."""
        return {
            "error_type": self.error_type,
            "error": str(self),
            "required_tokens": self.required_tokens,
            "target_budget": self.target_budget,
            "max_context_tokens": self.max_context_tokens,
            "reserve_tokens": self.reserve_tokens,
            "token_counting_mode": "heuristic_estimate",
        }


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to fit within max_tokens worth of words."""
    if estimate_tokens(text) <= max_tokens:
        return text
    words = text.split()
    keep = max(1, int(max_tokens / _TOKENS_PER_WORD))
    return " ".join(words[:keep]) + "..."


def is_tool_result_message(msg: Dict[str, Any]) -> bool:
    """Detect whether a message represents a tool execution output."""
    role = msg.get("role", "")
    if role == "tool":
        return True

    # Fallback checks for text-formatted observations
    content = msg.get("content", "")
    if isinstance(content, str):
        c_stripped = content.strip()
        if c_stripped.startswith("Tool [") or c_stripped.startswith("Tool output:"):
            return True
        if c_stripped.startswith("Observation:") or c_stripped.startswith("[Tool Result]"):
            return True
        if c_stripped.startswith("Recent tool results:"):
            return True
        if "Tool [" in c_stripped and "] output:" in c_stripped:
            return True

    return False


def _group_into_atomic_turns(messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Group messages into atomic ReAct turn blocks.

    An atomic turn block is an indivisible unit for context pruning:
    - System message: [sys]
    - User message with immediate non-tool assistant response: [user, assistant]
    - User message alone: [user]
    - Assistant message with tool_calls + all corresponding tool responses:
      [assistant, tool_1, tool_2, ...]
    - Standalone assistant response: [assistant]
    - Standalone tool message: [tool]
    """
    if not messages:
        return []

    blocks: List[List[Dict[str, Any]]] = []
    i = 0
    n = len(messages)
    while i < n:
        msg = messages[i]
        role = msg.get("role")
        if role == "assistant":
            block = [msg]
            j = i + 1
            while j < n and messages[j].get("role") == "tool":
                block.append(messages[j])
                j += 1
            blocks.append(block)
            i = j
        elif role == "user":
            # If immediately followed by an assistant message that has NO tool calls
            # and is NOT followed by tool messages, pair them into an atomic turn block
            if (
                i + 1 < n
                and messages[i + 1].get("role") == "assistant"
                and not messages[i + 1].get("tool_calls")
                and not (i + 2 < n and messages[i + 2].get("role") == "tool")
            ):
                blocks.append([msg, messages[i + 1]])
                i += 2
            else:
                blocks.append([msg])
                i += 1
        else:
            blocks.append([msg])
            i += 1

    return blocks


group_into_atomic_turns = _group_into_atomic_turns


def _sanitize_wire_invariants(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sanitize message sequence to guarantee strict ReAct wire-format compliance.

    Invariants enforced:
    1. Every 'tool' message must be preceded by an 'assistant' message
       (or another 'tool' message in the same multi-tool response batch) that initiated tool calls.
       Orphaned 'tool' messages without an assistant caller are discarded.
    2. Any 'assistant' message with 'tool_calls' followed by non-tool messages must have
       matching tool responses for each declared tool_call_id; missing responses are synthesized.
    3. Tool messages have valid tool_call_id alignment with their caller.
    """
    if not messages:
        return []

    sanitized: List[Dict[str, Any]] = []
    i = 0
    n = len(messages)

    while i < n:
        msg = messages[i]
        role = msg.get("role")

        if role == "tool":
            # Check if immediately preceded by assistant with tool_calls, or valid tool sequence
            if not sanitized:
                logger.warning("Dropping orphaned tool message at index 0: %s", msg.get("name"))
                i += 1
                continue

            prev = sanitized[-1]
            prev_role = prev.get("role")
            if prev_role not in ("assistant", "tool"):
                logger.warning("Dropping orphaned tool message preceded by %s: %s", prev_role, msg.get("name"))
                i += 1
                continue

            sanitized.append(msg)
            i += 1

        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                sanitized.append(msg)
                i += 1
                continue

            # Assistant has tool_calls
            sanitized.append(msg)
            i += 1

            # If this is the last message in the sequence, it's an in-flight tool call
            if i >= n:
                break

            # Collect subsequent tool messages
            tool_msgs: List[Dict[str, Any]] = []
            while i < n and messages[i].get("role") == "tool":
                tool_msgs.append(messages[i])
                i += 1

            declared_call_ids = [tc.get("id") for tc in tool_calls if isinstance(tc, dict) and tc.get("id")]
            present_call_ids = {tm.get("tool_call_id") for tm in tool_msgs if tm.get("tool_call_id")}

            # Align IDs if tool messages exist but are missing IDs
            if declared_call_ids and tool_msgs:
                for idx, tm in enumerate(tool_msgs):
                    if not tm.get("tool_call_id") and idx < len(declared_call_ids):
                        tm_copy = dict(tm)
                        tm_copy["tool_call_id"] = declared_call_ids[idx]
                        tool_msgs[idx] = tm_copy
                        present_call_ids.add(declared_call_ids[idx])

            # If there are declared call IDs that were not answered, and we have non-tool messages following:
            missing_ids = [cid for cid in declared_call_ids if cid not in present_call_ids]
            if missing_ids and (i < n or tool_msgs):
                for cid in missing_ids:
                    matching_tc = next((tc for tc in tool_calls if tc.get("id") == cid), {})
                    fn_name = (
                        matching_tc.get("function", {}).get("name")
                        if isinstance(matching_tc.get("function"), dict)
                        else matching_tc.get("name", "tool")
                    )
                    tool_msgs.append({
                        "role": "tool",
                        "tool_call_id": cid,
                        "name": fn_name,
                        "content": json.dumps({"status": "evicted", "output": "[Observation omitted for context budgeting]"}),
                    })

            sanitized.extend(tool_msgs)

        else:
            sanitized.append(msg)
            i += 1

    return sanitized


sanitize_wire_invariants = _sanitize_wire_invariants


# ---------------------------------------------------------------------------
# Context Budget Engine
# ---------------------------------------------------------------------------

class ContextBudgetEngine:
    """Adaptive Context Window Budgeting and Historical Tool Compaction Engine.

    Manages context window budgets proactively:
    1. Fast-path: immediate bypass when message history is <50% of budget.
    2. Stage 1: compacts verbose historical tool outputs older than recent turns.
    3. Stage 2: sliding-window turn eviction with concise semantic summary anchor.
    4. Guarantees root system directives and immediate user query remain inviolate.
    """

    _group_into_atomic_turns = staticmethod(_group_into_atomic_turns)
    _sanitize_wire_invariants = staticmethod(_sanitize_wire_invariants)

    def __init__(
        self,
        default_model: str = "qwen2.5-coder:7b",
        default_max_tokens: int = 8192,
        reserve_ratio: float = 0.20,
        min_reserve_tokens: int = 1000,
        max_reserve_tokens: int = 4096,
        tool_output_token_limit: int = 250,
        recent_turns_to_preserve: int = 2,
    ):
        self.default_model = default_model
        self.default_max_tokens = default_max_tokens
        self.reserve_ratio = reserve_ratio
        self.min_reserve_tokens = min_reserve_tokens
        self.max_reserve_tokens = max_reserve_tokens
        self.tool_output_token_limit = tool_output_token_limit
        self.recent_turns_to_preserve = recent_turns_to_preserve

    def _resolve_config_default_limit(self, config: Optional[Any] = None) -> Optional[int]:
        """Resolve explicit context limit from configuration when supplied."""
        if config is None:
            return None
        if isinstance(config, dict):
            raw = config.get("ollama_num_ctx") or config.get("num_ctx")
        elif hasattr(config, "get_effective_num_ctx"):
            try:
                raw = config.get_effective_num_ctx()
            except Exception:
                raw = getattr(config, "ollama_num_ctx", None)
        else:
            raw = getattr(config, "ollama_num_ctx", None)
        if raw is None or isinstance(raw, bool):
            return None
        try:
            from zelyr.config import validate_and_clamp_num_ctx

            return validate_and_clamp_num_ctx(raw)
        except Exception:
            try:
                return int(raw)
            except Exception:
                return None

    def get_model_limit(self, model_name: Optional[str], config: Optional[Any] = None) -> int:
        """Query explicit config limit, model catalog, or global config fallback."""
        cfg_limit = self._resolve_config_default_limit(config)
        if cfg_limit is not None:
            return cfg_limit

        fallback_default = self.default_max_tokens
        try:
            from zelyr.config import get_config

            global_cfg = get_config()
            raw_global = getattr(global_cfg, "ollama_num_ctx", None)
            if isinstance(raw_global, int) and raw_global > 0:
                fallback_default = raw_global
        except Exception:
            pass

        return get_model_context_limit(model_name or self.default_model, default_limit=fallback_default)

    def calculate_budget(
        self,
        model_name: Optional[str] = None,
        max_context_tokens: Optional[int] = None,
        reserve_tokens: Optional[int] = None,
        config: Optional[Any] = None,
    ) -> Tuple[int, int]:
        """Compute effective context limit and response generation reserve."""
        limit = max_context_tokens if (max_context_tokens is not None and max_context_tokens > 0) else self.get_model_limit(model_name, config=config)
        if reserve_tokens is not None:
            reserve = reserve_tokens
        else:
            calc_reserve = int(limit * self.reserve_ratio)
            reserve = max(self.min_reserve_tokens, min(calc_reserve, self.max_reserve_tokens))
            if reserve >= limit:
                reserve = max(100, int(limit * 0.25))

        return limit, reserve

    def get_context_headroom(
        self,
        messages: List[Dict[str, Any]],
        model_name: Optional[str] = None,
        max_context_tokens: Optional[int] = None,
        reserve_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Calculate estimated token usage, tool overhead, reserve, and available headroom."""
        limit, reserve = self.calculate_budget(model_name, max_context_tokens, reserve_tokens, config=config)
        tools_tokens = estimate_tools_tokens(tools)
        target_budget = max(0, limit - reserve - tools_tokens)
        current_tokens = estimate_messages_tokens(messages)
        headroom = max(0, target_budget - current_tokens)
        utilization = (current_tokens / target_budget) if target_budget > 0 else 1.0
        over = current_tokens > target_budget

        status: Dict[str, Any] = {
            "max_context_tokens": limit,
            "reserve_tokens": reserve,
            "target_budget": target_budget,
            "current_tokens": current_tokens,
            "headroom_tokens": headroom,
            "utilization": utilization,
            "is_over_budget": over,
            "token_counting_mode": "heuristic_estimate",
            "estimated_current_tokens": current_tokens,
            "estimated_tools_tokens": tools_tokens,
            "estimated_headroom_tokens": headroom,
        }
        if over:
            status["limit_error"] = ContextBudgetLimitError(
                required_tokens=current_tokens,
                target_budget=target_budget,
                max_context_tokens=limit,
                reserve_tokens=reserve,
            ).to_dict()
        return status

    def calculate_budget_status(
        self,
        messages: List[Dict[str, Any]],
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        max_context_tokens: Optional[int] = None,
        reserve_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Alias for get_context_headroom to compute utilization and token budget metrics."""
        limit = max_tokens or max_context_tokens
        return self.get_context_headroom(
            messages,
            model_name=model_name,
            max_context_tokens=limit,
            reserve_tokens=reserve_tokens,
            tools=tools,
            config=config,
        )

    def compact_tool_output(
        self,
        text: str,
        token_limit: int = 250,
        keep_head_chars: Optional[int] = None,
        keep_tail_chars: Optional[int] = None,
    ) -> str:
        """Compact a verbose tool output into a structured excerpt with truncation indicator."""
        if len(text) <= 1500 and estimate_tokens(text) <= token_limit:
            return text

        if keep_head_chars is None:
            keep_head_chars = 400 if token_limit <= 200 else 500
        if keep_tail_chars is None:
            keep_tail_chars = 150 if token_limit <= 200 else 500

        # Check if text is framed tool observation
        if text.startswith("[TOOL OBSERVATION:"):
            modified_observation = False

            def _excerpt_section(section_pattern: str, current_text: str) -> Tuple[str, bool]:
                match = re.search(section_pattern, current_text)
                if not match:
                    return current_text, False
                body = match.group(2)
                if len(body) > 1000 or estimate_tokens(body) > token_limit:
                    head = body[:keep_head_chars]
                    tail = body[-keep_tail_chars:] if keep_tail_chars > 0 else ""
                    omitted = len(body) - len(head) - len(tail)
                    if omitted > 0:
                        compacted_body = (
                            f"{head}\n\n[... output truncated: {omitted} chars omitted ...]\n"
                            f"... [OUTPUT TRUNCATED FOR CONTEXT BUDGET] ...\n\n{tail}"
                        )
                        return current_text[: match.start(2)] + compacted_body + current_text[match.end(2) :], True
                return current_text, False

            # 1. Check for [Output]: section (strip bulky write/patch echoes before optional head/tail slice)
            out_pattern = (
                r"(\[Output\]:\n)([\s\S]*?)"
                r"(\n\n\[(?:Informational Diagnostic Stream|Machine State Verified|Outcome Verification Failed|"
                r"Error Diagnostic|Error Output|Output Trace|Tool Error)|$)"
            )
            out_section = re.search(out_pattern, text)
            if out_section:
                body = out_section.group(2)
                if body.lstrip().startswith("{"):
                    try:
                        parsed_out = json.loads(body)
                        if isinstance(parsed_out, dict) and any(
                            k in parsed_out for k in ("written_content", "patched_content", "backup")
                        ):
                            had_echo = any(k in parsed_out for k in ("written_content", "patched_content"))
                            parsed_out.pop("written_content", None)
                            parsed_out.pop("patched_content", None)
                            parsed_out.pop("backup", None)
                            if had_echo:
                                parsed_out["content_omitted"] = True
                            stripped_json = json.dumps(parsed_out, ensure_ascii=False)
                            text = text[: out_section.start(2)] + stripped_json + text[out_section.end(2) :]
                            modified_observation = True
                    except Exception:
                        pass
                text, sec1_compacted = _excerpt_section(out_pattern, text)
                modified_observation = modified_observation or sec1_compacted

            # 2 & 3. Check oversized [Output Trace], [Error Output (stderr)], or [Informational Diagnostic Stream]
            for sec_pat in (
                r"(\[Output Trace\]:\n)([\s\S]*?)(\n\n\[(?:Tool Error)|$)",
                r"(\[Error Output \(stderr\)\]:\n)([\s\S]*?)(\n\n\[(?:Output Trace|Tool Error)|$)",
                r"(\[Informational Diagnostic Stream \(stderr\)\]:\n)([\s\S]*?)(\n\(Note:|\n\n\[|$)",
            ):
                text, sec_compacted = _excerpt_section(sec_pat, text)
                modified_observation = modified_observation or sec_compacted

            if modified_observation:
                return text

        # Compact structured planner observation blocks while preserving error, uncertainty & refs
        if text.startswith("[OBSERVATION "):
            kept_lines: List[str] = []
            for ln in text.splitlines():
                if ln.startswith("- Result Preview:") and len(ln) > max(120, keep_head_chars):
                    kept_lines.append(ln[: max(80, keep_head_chars // 2)] + "... [TRUNCATED FOR CONTEXT BUDGET]")
                elif ln.startswith("- Reported:") and len(ln) > 260:
                    kept_lines.append(ln[:220] + "... [TRUNCATED]")
                else:
                    kept_lines.append(ln)
            candidate = "\n".join(kept_lines)
            if len(candidate) <= len(text):
                return candidate

        # Check if text is JSON string with status/result structure
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                modified = False
                for key in ("output", "stdout", "content", "result", "data", "error", "stderr"):
                    if key in parsed and isinstance(parsed[key], str):
                        val = parsed[key]
                        if len(val) > 1500 or estimate_tokens(val) > token_limit:
                            head = val[:keep_head_chars]
                            tail = val[-keep_tail_chars:] if keep_tail_chars > 0 else ""
                            omitted = len(val) - len(head) - len(tail)
                            if omitted > 0:
                                parsed[key] = (
                                    f"{head}\n\n[... output truncated: {omitted} chars omitted ...]\n"
                                    f"... [TRUNCATED FOR CONTEXT BUDGET] ...\n\n{tail}"
                                )
                                modified = True
                if modified:
                    return json.dumps(parsed, indent=2)

                str_keys = [k for k, v in parsed.items() if isinstance(v, str) and len(v) > 200]
                if str_keys:
                    longest_k = max(str_keys, key=lambda k: len(parsed[k]))
                    val = parsed[longest_k]
                    head = val[:keep_head_chars]
                    tail = val[-keep_tail_chars:] if keep_tail_chars > 0 else ""
                    omitted = len(val) - len(head) - len(tail)
                    if omitted > 0:
                        parsed[longest_k] = (
                            f"{head}\n\n[... output truncated: {omitted} chars omitted ...]\n"
                            f"... [TRUNCATED FOR CONTEXT BUDGET] ...\n\n{tail}"
                        )
                    return json.dumps(parsed, indent=2)
        except Exception:
            pass

        # Text excerpting
        head = text[:keep_head_chars]
        tail = text[-keep_tail_chars:] if keep_tail_chars > 0 else ""
        omitted = len(text) - len(head) - len(tail)
        if omitted <= 0:
            return text

        return (
            f"{head}\n\n[... output truncated: {omitted} chars omitted ...]\n"
            f"... [TRUNCATED FOR CONTEXT BUDGET] ...\n\n{tail}"
        )

    def stage1_compact_tool_results(
        self,
        messages: List[Dict[str, Any]],
        recent_turns_to_preserve: int = 2,
        tool_output_token_limit: int = 250,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Stage 1: Compact historical tool results older than recent turns.

        Protects:
        1. Tool outputs occurring in the last `recent_turns_to_preserve` user turns.
        2. The last `recent_turns_to_preserve` tool messages overall (to protect recent ReAct steps).
        """
        if not messages:
            return messages, 0

        # Find user turn boundaries
        user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        cutoff_user_idx = None
        if len(user_indices) >= recent_turns_to_preserve:
            cutoff_user_idx = user_indices[-recent_turns_to_preserve]

        # Find tool message indices to protect the latest tool interactions in single-turn loops
        tool_indices = [i for i, m in enumerate(messages) if is_tool_result_message(m)]
        protected_tool_indices = set(tool_indices[-recent_turns_to_preserve:]) if tool_indices else set()

        compacted: List[Dict[str, Any]] = []
        compacted_count = 0

        for idx, msg in enumerate(messages):
            is_tool = is_tool_result_message(msg)
            if cutoff_user_idx is not None:
                # Multi-turn conversation: messages before recent user turn boundary are historical
                is_historical = (idx < cutoff_user_idx)
            else:
                # Single-turn ReAct loop: protect the last N tool steps, compact earlier ones
                is_historical = (idx not in protected_tool_indices)

            is_eligible = is_tool and is_historical

            content = msg.get("content", "")
            if is_eligible and isinstance(content, str) and (len(content) > 1500 or estimate_tokens(content) > tool_output_token_limit):
                new_content = self.compact_tool_output(content, token_limit=tool_output_token_limit)
                new_msg = dict(msg)
                new_msg["content"] = new_content
                compacted.append(new_msg)
                compacted_count += 1
            else:
                compacted.append(msg)

        return compacted, compacted_count

    @staticmethod
    def _strip_thinking_tags(text: str) -> str:
        """Strip closed or truncated <think>...</think> blocks from historical messages."""
        stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        stripped = re.sub(r"<think>.*$", "", stripped, flags=re.DOTALL)
        return stripped.strip()

    def stage1_5_compact_thinking(
        self,
        messages: List[Dict[str, Any]],
        target_budget: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Stage 1.5: Compact bulky <think> blocks before Stage 2 evicts whole turns.

        Compacts oldest assistant messages first so recent reasoning stays intact when
        sufficient, and strips the remaining <think> block if doing so keeps the
        historical tool observations inside target_budget.
        """
        if not messages:
            return messages, 0

        assistant_indices = [
            i for i, m in enumerate(messages)
            if m.get("role") == "assistant"
            and isinstance(m.get("content"), str)
            and "<think>" in m["content"]
        ]
        if not assistant_indices:
            return messages, 0

        compacted = [dict(m) for m in messages]
        stripped_count = 0

        strip_order = assistant_indices[:-1] + assistant_indices[-1:]
        for idx in strip_order:
            msg = compacted[idx]
            orig_content = msg.get("content", "")
            cleaned = self._strip_thinking_tags(orig_content)
            if cleaned != orig_content.strip():
                if not cleaned and not msg.get("tool_calls"):
                    cleaned = "[Reasoning compacted]"
                msg["content"] = cleaned
                stripped_count += 1
                if estimate_messages_tokens(compacted) <= target_budget:
                    break

        return compacted, stripped_count

    def summarize_turns(self, turns: List[Dict[str, Any]], max_summary_tokens: int = 300) -> str:
        """Extract key intents, tool actions, and outcomes from evicted turns."""
        if not turns:
            return "No prior turns to summarize."

        user_queries: List[str] = []
        tool_actions: List[str] = []
        recent_steps: List[str] = []
        assistant_notes: List[str] = []
        failure_evidence: List[str] = []

        def _format_step_descriptor(fn_name: str, raw_args: Any) -> str:
            args_dict: Dict[str, Any] = {}
            if isinstance(raw_args, dict):
                args_dict = raw_args
            elif isinstance(raw_args, str) and raw_args.strip().startswith("{"):
                try:
                    parsed_a = json.loads(raw_args)
                    if isinstance(parsed_a, dict):
                        args_dict = parsed_a
                except Exception:
                    pass
            if not args_dict:
                return fn_name
            bits: List[str] = []
            if args_dict.get("action"):
                bits.append(str(args_dict["action"]))
            if args_dict.get("x") is not None and args_dict.get("y") is not None:
                bits.append(f"({args_dict['x']},{args_dict['y']})")
            for key_k in ("text", "key", "path", "target", "window"):
                v = args_dict.get(key_k)
                if v not in (None, "") and (key_k != "target" or not args_dict.get("window")):
                    sv = str(v)
                    if len(sv) > 24:
                        sv = sv[:21] + "..."
                    bits.append(f"{key_k}={sv}")
            if args_dict.get("command"):
                cmd_s = str(args_dict["command"]).strip()
                if len(cmd_s) > 28:
                    cmd_s = cmd_s[:25] + "..."
                bits.append(cmd_s)
            return f"{fn_name}({' '.join(bits[:4])})" if bits else fn_name

        for msg in turns:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
            content_str = str(content).strip()

            if role == "user":
                clean_q = re.sub(r"\[Active Window:[^\]]*\]\s*", "", content_str).strip()
                if clean_q:
                    snip = clean_q[:90] + ("..." if len(clean_q) > 90 else "")
                    user_queries.append(snip)
            elif role == "assistant":
                for tc in msg.get("tool_calls") or []:
                    fn_obj = tc.get("function", {}) if isinstance(tc, dict) else {}
                    fn = fn_obj.get("name") or (tc.get("name") if isinstance(tc, dict) else None)
                    if fn:
                        tool_actions.append(fn)
                        recent_steps.append(_format_step_descriptor(str(fn), fn_obj.get("arguments")))
                if content_str and len(content_str) > 15:
                    clean_c = re.sub(r"<think>.*?</think>", "", content_str, flags=re.DOTALL).strip()
                    if clean_c:
                        snip = clean_c[:100] + ("..." if len(clean_c) > 100 else "")
                        assistant_notes.append(snip)
            elif role == "tool":
                fn = msg.get("name") or "tool"
                if msg.get("name"):
                    tool_actions.append(fn)
                # Preserve critical failure / verification-failure evidence instead of silently losing it
                if any(
                    marker in content_str
                    for marker in (
                        "[Error Diagnostic:",
                        "[Outcome Verification Failed]:",
                        "VERIFICATION_FAILED",
                        "\"status\": \"error\"",
                        "\"verification_failed\": true",
                    )
                ):
                    err_lines = [
                        ln.strip()
                        for ln in content_str.splitlines()
                        if any(
                            k in ln
                            for k in (
                                "[Error Diagnostic:",
                                "- Failure Reason:",
                                "[Outcome Verification Failed]:",
                                "- Failed Criterion:",
                                "- Verifier Reason:",
                            )
                        )
                    ]
                    if err_lines:
                        failure_evidence.append(f"{fn}: {' / '.join(err_lines[:3])[:260]}")
                    else:
                        failure_evidence.append(f"{fn}: {content_str[:160]}")

        parts: List[str] = []
        if user_queries:
            parts.append(f"User asked: {'; '.join(user_queries[:4])}")
        if tool_actions:
            deduped = list(dict.fromkeys(tool_actions))
            parts.append(f"Tools executed: {', '.join(deduped[:6])}")
        if recent_steps:
            parts.append(f"Recent steps: {' -> '.join(recent_steps[-6:])}")
        if failure_evidence:
            parts.append(f"Preserved failures/verification errors: {'; '.join(failure_evidence[-3:])}")
        if assistant_notes:
            parts.append(f"Key outcomes: {'; '.join(assistant_notes[-2:])}")

        summary = " | ".join(parts) if parts else "Prior conversational context compacted."
        return _truncate_to_tokens(summary, max_summary_tokens)

    _SUMMARY_PREFIX = "[Previous conversation summary:"

    @classmethod
    def _merge_conversation_summaries(
        cls,
        prior_summary_msgs: List[Dict[str, Any]],
        new_summary: str,
        max_summary_tokens: int = 250,
    ) -> str:
        """Consolidate prior Stage 2 summaries with newly evicted summary clauses without stacking duplicates."""
        all_bodies: List[str] = []
        for pm in prior_summary_msgs:
            raw_c = str(pm.get("content", "")).strip()
            if raw_c.startswith(cls._SUMMARY_PREFIX):
                body = raw_c[len(cls._SUMMARY_PREFIX) :].strip()
                if body.endswith("]"):
                    body = body[:-1].strip()
                if body:
                    all_bodies.append(body)
        if new_summary:
            all_bodies.append(new_summary.strip())

        user_asked: List[str] = []
        tools_exec: List[str] = []
        recent_steps_merged: List[str] = []
        failures: List[str] = []
        outcomes: List[str] = []
        other_clauses: List[str] = []

        for body in all_bodies:
            for clause in [c.strip() for c in body.split(" | ") if c.strip()]:
                if clause.startswith("User asked:"):
                    val = clause[len("User asked:") :].strip()
                    if val and val not in user_asked:
                        user_asked.append(val)
                elif clause.startswith("Tools executed:"):
                    for t in clause[len("Tools executed:") :].split(","):
                        ts = t.strip()
                        if ts and ts not in tools_exec:
                            tools_exec.append(ts)
                elif clause.startswith("Recent steps:"):
                    for st_item in clause[len("Recent steps:") :].split(" -> "):
                        ss = st_item.strip()
                        if ss:
                            recent_steps_merged.append(ss)
                elif clause.startswith("Preserved failures/verification errors:"):
                    for f_item in clause[len("Preserved failures/verification errors:") :].split(";"):
                        fs = f_item.strip()
                        if fs and fs not in failures:
                            failures.append(fs)
                elif clause.startswith("Key outcomes:"):
                    for o_item in clause[len("Key outcomes:") :].split(";"):
                        os_ = o_item.strip()
                        if os_ and os_ not in outcomes:
                            outcomes.append(os_)
                elif clause not in other_clauses:
                    other_clauses.append(clause)

        merged_parts: List[str] = []
        if user_asked:
            merged_parts.append(f"User asked: {'; '.join(user_asked[:4])}")
        if tools_exec:
            merged_parts.append(f"Tools executed: {', '.join(tools_exec[:8])}")
        if recent_steps_merged:
            merged_parts.append(f"Recent steps: {' -> '.join(recent_steps_merged[-6:])}")
        if failures:
            merged_parts.append(f"Preserved failures/verification errors: {'; '.join(failures[-4:])}")
        if outcomes:
            merged_parts.append(f"Key outcomes: {'; '.join(outcomes[-2:])}")
        for oc in other_clauses:
            if oc not in merged_parts:
                merged_parts.append(oc)

        merged = " | ".join(merged_parts) if merged_parts else new_summary or "Prior conversational context compacted."
        return _truncate_to_tokens(merged, max_summary_tokens)

    def stage2_prune_turns(
        self,
        messages: List[Dict[str, Any]],
        target_budget: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Stage 2: Sliding-window turn eviction with atomic ReAct block preservation and semantic summarization.

        Inviolate Anchors:
        - System directives and Runtime Environment Manifest (leading messages with role=='system' containing [RUNTIME ENVIRONMENT & CAPABILITY MANIFEST]).
        - Root user goal (first user message) is strictly preserved as the task foundation.
        - Active user turn (from the latest user message to the end of the conversation) is strictly preserved.
        - ReAct step blocks (assistant with tool_calls + matching tool outputs) are indivisible.
        """
        if not messages:
            return messages, 0

        # Current total token check
        total_tokens = sum(count_message_tokens(m) for m in messages)
        if total_tokens <= target_budget:
            return messages, 0

        # Separate leading system messages into core directives and prior consolidated summaries
        sys_idx = 0
        while sys_idx < len(messages) and messages[sys_idx].get("role") == "system":
            sys_idx += 1

        leading_sys = messages[:sys_idx]
        core_system_messages = [
            m for m in leading_sys
            if not str(m.get("content", "")).lstrip().startswith(self._SUMMARY_PREFIX)
        ]
        prior_summary_messages = [
            m for m in leading_sys
            if str(m.get("content", "")).lstrip().startswith(self._SUMMARY_PREFIX)
        ]
        conv_messages = messages[sys_idx:]

        if not conv_messages:
            return core_system_messages + prior_summary_messages[-1:], 0

        user_indices = [i for i, m in enumerate(conv_messages) if m.get("role") == "user"]

        if not user_indices:
            if len(conv_messages) == 1:
                return core_system_messages + prior_summary_messages[-1:] + conv_messages, 0
            root_user_msg = conv_messages[0]
            active_turn_msgs = [conv_messages[-1]]
            middle_blocks = self._group_into_atomic_turns(conv_messages[1:-1])
        elif len(user_indices) == 1:
            first_idx = user_indices[0]
            root_user_msg = conv_messages[first_idx]
            subsequent_msgs = conv_messages[first_idx + 1:]

            if not subsequent_msgs:
                return core_system_messages + prior_summary_messages[-1:] + [root_user_msg], 0

            react_blocks = self._group_into_atomic_turns(subsequent_msgs)
            active_turn_msgs = react_blocks[-1]
            middle_blocks = react_blocks[:-1]
        else:
            first_user_idx = user_indices[0]
            last_user_idx = user_indices[-1]
            root_user_msg = conv_messages[first_user_idx]
            active_turn_msgs = conv_messages[last_user_idx:]
            middle_blocks = self._group_into_atomic_turns(conv_messages[first_user_idx + 1 : last_user_idx])

        if not middle_blocks:
            return self._sanitize_wire_invariants(
                core_system_messages + prior_summary_messages[-1:] + [root_user_msg] + active_turn_msgs
            ), 0

        sys_tokens = sum(count_message_tokens(m) for m in core_system_messages)
        root_tokens = count_message_tokens(root_user_msg)
        active_tokens = sum(count_message_tokens(m) for m in active_turn_msgs)
        summary_reserve = 250 if prior_summary_messages else 150
        avail_for_middle = target_budget - sys_tokens - root_tokens - active_tokens - summary_reserve

        kept_middle_blocks: List[List[Dict[str, Any]]] = []
        evicted_middle_blocks: List[List[Dict[str, Any]]] = []
        accum_tokens = 0

        for b in reversed(middle_blocks):
            b_tokens = sum(count_message_tokens(m) for m in b)
            if accum_tokens + b_tokens <= avail_for_middle:
                kept_middle_blocks.insert(0, b)
                accum_tokens += b_tokens
            else:
                evicted_middle_blocks.insert(0, b)

        evicted_msgs = [m for b in evicted_middle_blocks for m in b]
        pruned_conv = [root_user_msg]
        for b in kept_middle_blocks:
            pruned_conv.extend(b)
        pruned_conv.extend(active_turn_msgs)

        if evicted_msgs:
            summary_text = self.summarize_turns(evicted_msgs, max_summary_tokens=250)
            if prior_summary_messages:
                summary_text = self._merge_conversation_summaries(
                    prior_summary_messages, summary_text, max_summary_tokens=250
                )
            summary_msg = {
                "role": "system",
                "content": f"{self._SUMMARY_PREFIX} {summary_text}]",
            }
            assembled = core_system_messages + [summary_msg] + pruned_conv
            return self._sanitize_wire_invariants(assembled), len(evicted_msgs)

        if len(prior_summary_messages) > 1:
            consolidated_text = self._merge_conversation_summaries(
                prior_summary_messages, "", max_summary_tokens=250
            )
            prior_anchor = [{"role": "system", "content": f"{self._SUMMARY_PREFIX} {consolidated_text}]"}]
        else:
            prior_anchor = prior_summary_messages
        return self._sanitize_wire_invariants(core_system_messages + prior_anchor + pruned_conv), 0

    def apply_context_budget(
        self,
        messages: List[Dict[str, Any]],
        model_name: Optional[str] = None,
        max_context_tokens: Optional[int] = None,
        reserve_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Any] = None,
        raise_on_overflow: bool = False,
    ) -> List[Dict[str, Any]]:
        """Apply hierarchical adaptive context budgeting across stages."""
        if not messages:
            return messages

        effective_max, effective_reserve = self.calculate_budget(
            model_name=model_name,
            max_context_tokens=max_context_tokens,
            reserve_tokens=reserve_tokens,
            config=config,
        )
        tools_tokens = estimate_tools_tokens(tools)
        target_budget = effective_max - effective_reserve - tools_tokens
        if target_budget <= 0:
            required = estimate_messages_tokens(messages)
            if raise_on_overflow:
                raise ContextBudgetLimitError(required, max(0, target_budget), effective_max, effective_reserve)
            return messages

        # 0. Zero-Overhead Fast Path:
        # If conversation is small (< 6 messages) and characters are well under 50% of budget capacity,
        # return immediately with zero deep iteration or memory allocation.
        fast_threshold = int(target_budget * 0.50)
        rough_chars = sum(len(m.get("content", "")) for m in messages if isinstance(m.get("content"), str))
        if len(messages) <= 6 and (rough_chars // 3) < fast_threshold:
            return messages

        initial_tokens = estimate_messages_tokens(messages)

        # 1. Stage 1 — Historical Tool Result Compaction
        stage1_msgs, compacted_tools = self.stage1_compact_tool_results(
            messages,
            recent_turns_to_preserve=self.recent_turns_to_preserve,
            tool_output_token_limit=self.tool_output_token_limit,
        )

        post_s1_tokens = estimate_messages_tokens(stage1_msgs)

        # If Stage 1 brought token footprint within target budget, return cleanly
        if post_s1_tokens <= target_budget:
            logger.debug(
                "Context budget: Stage 1 sufficient. initial=%d, final=%d, compacted_tools=%d, pruned_turns=0, max=%d",
                initial_tokens,
                post_s1_tokens,
                compacted_tools,
                effective_max,
            )
            return stage1_msgs

        # 1.5. Stage 1.5 — Historical <think> Reasoning Compaction before evicting whole turns
        stage1_5_msgs, compacted_thinks = self.stage1_5_compact_thinking(
            stage1_msgs,
            target_budget=target_budget,
        )
        post_s1_5_tokens = estimate_messages_tokens(stage1_5_msgs)
        if post_s1_5_tokens <= target_budget:
            logger.debug(
                "Context budget: Stage 1.5 sufficient. initial=%d, final=%d, compacted_tools=%d, compacted_thinks=%d, pruned_turns=0, max=%d",
                initial_tokens,
                post_s1_5_tokens,
                compacted_tools,
                compacted_thinks,
                effective_max,
            )
            return stage1_5_msgs

        # 2. Stage 2 — Sliding-Window Turn Pruning with Semantic Summarization
        final_msgs, pruned_turns = self.stage2_prune_turns(
            stage1_5_msgs,
            target_budget=target_budget,
        )
        final_tokens = estimate_messages_tokens(final_msgs)

        if final_tokens > target_budget:
            if raise_on_overflow:
                raise ContextBudgetLimitError(
                    required_tokens=final_tokens,
                    target_budget=target_budget,
                    max_context_tokens=effective_max,
                    reserve_tokens=effective_reserve,
                )
            logger.warning(
                "Context budget overflow after Stage 2 (%d > %d estimated tokens; tools=%d).",
                final_tokens,
                target_budget,
                tools_tokens,
            )

        logger.debug(
            "Context budget: Stage 2 applied. initial=%d, post_s1=%d, final=%d, compacted_tools=%d, pruned_turns=%d, max=%d",
            initial_tokens,
            post_s1_tokens,
            final_tokens,
            compacted_tools,
            pruned_turns,
            effective_max,
        )
        return final_msgs

    def enforce_context_budget(
        self,
        messages: List[Dict[str, Any]],
        model_name: Optional[str] = None,
        max_context_tokens: Optional[int] = None,
        reserve_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Apply context budgeting and fail explicitly with ContextBudgetLimitError if required anchors still exceed budget."""
        return self.apply_context_budget(
            messages,
            model_name=model_name,
            max_context_tokens=max_context_tokens,
            reserve_tokens=reserve_tokens,
            tools=tools,
            config=config,
            raise_on_overflow=True,
        )


# Facade aliases
AdaptiveContextBudgeter = ContextBudgetEngine
BudgetEngine = ContextBudgetEngine

__all__ = [
    "ContextBudgetEngine",
    "ContextBudgetLimitError",
    "AdaptiveContextBudgeter",
    "BudgetEngine",
    "MODEL_CONTEXT_CATALOG",
    "get_model_context_limit",
    "estimate_tokens",
    "count_message_tokens",
    "estimate_messages_tokens",
    "estimate_tools_tokens",
    "is_tool_result_message",
    "group_into_atomic_turns",
    "_group_into_atomic_turns",
    "sanitize_wire_invariants",
    "_sanitize_wire_invariants",
]
