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
    """Estimate total tokens across a list of chat messages."""
    return sum(count_message_tokens(m) for m in messages)


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

    def get_model_limit(self, model_name: Optional[str]) -> int:
        """Query catalog for model context limit or use default."""
        return get_model_context_limit(model_name or self.default_model, default_limit=self.default_max_tokens)

    def calculate_budget(
        self,
        model_name: Optional[str] = None,
        max_context_tokens: Optional[int] = None,
        reserve_tokens: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Compute effective context limit and response generation reserve."""
        limit = max_context_tokens or self.get_model_limit(model_name)
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
    ) -> Dict[str, Any]:
        """Calculate token usage, limit, reserve, and available headroom."""
        limit, reserve = self.calculate_budget(model_name, max_context_tokens, reserve_tokens)
        target_budget = limit - reserve
        current_tokens = estimate_messages_tokens(messages)
        headroom = max(0, target_budget - current_tokens)
        utilization = (current_tokens / target_budget) if target_budget > 0 else 1.0

        return {
            "max_context_tokens": limit,
            "reserve_tokens": reserve,
            "target_budget": target_budget,
            "current_tokens": current_tokens,
            "headroom_tokens": headroom,
            "utilization": utilization,
            "is_over_budget": current_tokens > target_budget,
        }

    def compact_tool_output(
        self,
        text: str,
        token_limit: int = 250,
        keep_head_chars: int = 400,
        keep_tail_chars: int = 150,
    ) -> str:
        """Compact a verbose tool output into a structured excerpt with truncation indicator."""
        if estimate_tokens(text) <= token_limit:
            return text

        # Check if text is JSON string with status/result structure
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                modified = False
                for key in ("output", "stdout", "content", "result", "data"):
                    if key in parsed and isinstance(parsed[key], str):
                        val = parsed[key]
                        if estimate_tokens(val) > token_limit:
                            head = val[:keep_head_chars]
                            tail = val[-keep_tail_chars:]
                            omitted = len(val) - len(head) - len(tail)
                            if omitted > 0:
                                parsed[key] = f"{head}\n\n[... output truncated: {omitted} chars omitted ...]\n\n{tail}"
                                modified = True
                if modified:
                    return json.dumps(parsed)
        except Exception:
            pass

        # Text excerpting
        head = text[:keep_head_chars]
        tail = text[-keep_tail_chars:]
        omitted = len(text) - len(head) - len(tail)
        if omitted <= 0:
            return text

        return f"{head}\n\n[... output truncated: {omitted} chars omitted ...]\n\n{tail}"

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
            if is_eligible and isinstance(content, str) and estimate_tokens(content) > tool_output_token_limit:
                new_content = self.compact_tool_output(content, token_limit=tool_output_token_limit)
                new_msg = dict(msg)
                new_msg["content"] = new_content
                compacted.append(new_msg)
                compacted_count += 1
            else:
                compacted.append(msg)

        return compacted, compacted_count

    def summarize_turns(self, turns: List[Dict[str, Any]], max_summary_tokens: int = 300) -> str:
        """Extract key intents, tool actions, and outcomes from evicted turns."""
        if not turns:
            return "No prior turns to summarize."

        user_queries: List[str] = []
        tool_actions: List[str] = []
        assistant_notes: List[str] = []

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
                    fn = tc.get("function", {}).get("name") or tc.get("name")
                    if fn:
                        tool_actions.append(fn)
                if content_str and len(content_str) > 15:
                    clean_c = re.sub(r"<think>.*?</think>", "", content_str, flags=re.DOTALL).strip()
                    if clean_c:
                        snip = clean_c[:100] + ("..." if len(clean_c) > 100 else "")
                        assistant_notes.append(snip)
            elif role == "tool":
                fn = msg.get("name")
                if fn:
                    tool_actions.append(fn)

        parts: List[str] = []
        if user_queries:
            parts.append(f"User asked: {'; '.join(user_queries[:4])}")
        if tool_actions:
            deduped = list(dict.fromkeys(tool_actions))
            parts.append(f"Tools executed: {', '.join(deduped[:6])}")
        if assistant_notes:
            parts.append(f"Key outcomes: {'; '.join(assistant_notes[-2:])}")

        summary = " | ".join(parts) if parts else "Prior conversational context compacted."
        return _truncate_to_tokens(summary, max_summary_tokens)

    def stage2_prune_turns(
        self,
        messages: List[Dict[str, Any]],
        target_budget: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Stage 2: Sliding-window turn eviction with semantic summarization.

        Inviolate Anchors:
        - System directives (messages at start with role=='system').
        - Initial user turn (conv_messages[0]) preserved as the initial task anchor when possible.
        - Latest user turn (conv_messages[-1]) strictly preserved as the immediate user query.
        """
        if not messages:
            return messages, 0

        system_messages = [m for m in messages if m.get("role") == "system"]
        conv_messages = [m for m in messages if m.get("role") != "system"]

        if not conv_messages:
            return messages, 0

        # Current total token check
        total_tokens = sum(count_message_tokens(m) for m in messages)
        if total_tokens <= target_budget:
            return messages, 0

        # Latest turn is inviolate anchor
        latest_turn = conv_messages[-1]
        if len(conv_messages) == 1:
            return system_messages + [latest_turn], 0

        # Initial turn is preserved as context anchor if possible
        initial_turn = conv_messages[0]
        middle_turns = conv_messages[1:-1]

        evicted: List[Dict[str, Any]] = []
        kept_middle: List[Dict[str, Any]] = []

        # If there are middle turns, evict oldest middle turns first
        if middle_turns:
            # We want to keep as many recent middle turns as fit within budget
            # Available budget for middle turns after reserving system, initial turn, latest turn, and summary (~150 tokens)
            sys_tokens = sum(count_message_tokens(m) for m in system_messages)
            init_tokens = count_message_tokens(initial_turn)
            latest_tokens = count_message_tokens(latest_turn)
            summary_reserve = 150

            avail_for_middle = target_budget - sys_tokens - init_tokens - latest_tokens - summary_reserve

            # Select recent middle turns from back to front
            accum_tokens = 0
            for msg in reversed(middle_turns):
                t_count = count_message_tokens(msg)
                if accum_tokens + t_count <= avail_for_middle:
                    kept_middle.insert(0, msg)
                    accum_tokens += t_count
                else:
                    evicted.insert(0, msg)
        else:
            # Only initial_turn and latest_turn exist.
            # If initial_turn + latest_turn > target_budget, evict initial_turn into summary
            sys_tokens = sum(count_message_tokens(m) for m in system_messages)
            init_tokens = count_message_tokens(initial_turn)
            latest_tokens = count_message_tokens(latest_turn)
            if sys_tokens + init_tokens + latest_tokens > target_budget:
                evicted.append(initial_turn)
                initial_turn = None

        pruned_conv: List[Dict[str, Any]] = []
        if initial_turn is not None:
            pruned_conv.append(initial_turn)
        pruned_conv.extend(kept_middle)
        pruned_conv.append(latest_turn)

        # Inject semantic summary anchor if any turns were evicted
        if evicted:
            summary_text = self.summarize_turns(evicted, max_summary_tokens=250)
            summary_msg = {
                "role": "system",
                "content": f"[Previous conversation summary: {summary_text}]",
            }
            # Place immediately after the root system prompt(s)
            return system_messages + [summary_msg] + pruned_conv, len(evicted)

        return system_messages + pruned_conv, 0

    def apply_context_budget(
        self,
        messages: List[Dict[str, Any]],
        model_name: Optional[str] = None,
        max_context_tokens: Optional[int] = None,
        reserve_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Apply hierarchical adaptive context budgeting across stages."""
        if not messages:
            return messages

        effective_max, effective_reserve = self.calculate_budget(
            model_name=model_name,
            max_context_tokens=max_context_tokens,
            reserve_tokens=reserve_tokens,
        )
        target_budget = effective_max - effective_reserve

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

        # 2. Stage 2 — Sliding-Window Turn Pruning with Semantic Summarization
        final_msgs, pruned_turns = self.stage2_prune_turns(
            stage1_msgs,
            target_budget=target_budget,
        )
        final_tokens = estimate_messages_tokens(final_msgs)

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


# Facade alias
AdaptiveContextBudgeter = ContextBudgetEngine

__all__ = [
    "ContextBudgetEngine",
    "AdaptiveContextBudgeter",
    "MODEL_CONTEXT_CATALOG",
    "get_model_context_limit",
    "estimate_tokens",
    "count_message_tokens",
    "estimate_messages_tokens",
    "is_tool_result_message",
]
