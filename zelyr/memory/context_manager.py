"""Token-aware context management for Zelyr.

Prevents context overflow by tracking approximate token usage,
summarizing old conversation turns, and assembling prompts within
a configurable token budget.
"""
import json
import logging
from typing import Any, Callable, Dict, List, Optional
logger = logging.getLogger(__name__)
_TOKENS_PER_WORD = 1.3

def estimate_tokens(text: str) -> int:
    """Estimate token count from text using word-count heuristic."""
    if not text:
        return 0
    return int(len(text.split()) * _TOKENS_PER_WORD)

def estimate_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    """Estimate total tokens across a list of chat messages."""
    total = 0
    for msg in messages:
        if '_cached_tokens' in msg:
            total += msg['_cached_tokens']
            continue
        content = msg.get('content', '')
        tokens = 0
        if isinstance(content, str):
            tokens = estimate_tokens(content)
        elif isinstance(content, dict):
            tokens = estimate_tokens(json.dumps(content, default=str))
        msg['_cached_tokens'] = tokens
        total += tokens
    return total

def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens worth of words."""
    if estimate_tokens(text) <= max_tokens:
        return text
    words = text.split()
    keep = int(max_tokens / _TOKENS_PER_WORD)
    if keep < 1:
        keep = 1
    return ' '.join(words[:keep]) + '...'

class ContextManager:
    """Manages LLM context windows to prevent overflow.

    Tracks token usage and applies these strategies in order:
    1. Always keep the system prompt.
    2. Keep the most recent N conversation turns.
    3. Summarize older turns into a compact digest.
    4. Include retrieved memories if budget allows.
    5. Never exceed the configured token limit.

    Args:
        max_tokens: Hard token budget for the full prompt.  Defaults to 6144
            which fits comfortably within most 8k-context Ollama models while
            leaving room for the response.
        reserve_tokens: Tokens reserved for the LLM response.  Subtracted
            from *max_tokens* to get the available input budget.
        summary_threshold: When the conversation history exceeds this many
            turns, older turns are summarized.
        summary_max_tokens: Target token count for the summarized digest.
        summarize_fn: Optional callback ``(old_turns: list[dict]) -> str``
            that produces a summary string.  When *None*, old turns are
            simply truncated rather than summarized.
    """

    def __init__(self, max_tokens: int=6144, reserve_tokens: int=1024, summary_threshold: int=10, summary_max_tokens: int=512, summarize_fn: Optional[Callable[[List[Dict[str, Any]]], str]]=None):
        """Auto-generated docstring.

Args:
    max_tokens: Argument.
    reserve_tokens: Argument.
    summary_threshold: Argument.
    summary_max_tokens: Argument.
    summarize_fn: Argument.

Returns:
    Return value.
"""
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self.summary_threshold = summary_threshold
        self.summary_max_tokens = summary_max_tokens
        self.summarize_fn = summarize_fn

    @property
    def input_budget(self) -> int:
        """Available tokens for input messages (max_tokens minus reserve)."""
        return self.max_tokens - self.reserve_tokens

    def build_system_prompt(self, task: str, tool_schemas: List[Dict[str, Any]], intent: str) -> List[Dict[str, Any]]:
        import json
        from zelyr.agents.prompts import _build_react_prompt, _build_tool_guidelines
        
        prompt = _build_react_prompt(intent)
        prompt += "\n\n" + _build_tool_guidelines()
        
        if tool_schemas:
            prompt += "\n\nAvailable Tools:\n" + json.dumps(tool_schemas, indent=2)
            
        return [{"role": "system", "content": prompt}]

    def build_context_window(self, system_messages: List[Dict[str, Any]], chat_history: List[Dict[str, Any]], current_task: str, retrieved_memories: Optional[List[Dict[str, Any]]]=None, observations: Optional[List[Dict[str, Any]]]=None) -> List[Dict[str, Any]]:
        """Assemble a prompt list that fits within the token budget.

        Priority order:
        1. System messages (always included).
        2. Current user task (always included).
        3. Recent chat history (most recent turns first).
        4. Summarized older turns.
        5. Retrieved memories.
        6. Recent observations.

        Returns a list of dicts suitable for ``OllamaClient.chat_with_tools``.
        """
        budget = self.input_budget
        result: List[Dict[str, Any]] = []
        system_tokens = 0
        for msg in system_messages:
            tokens = estimate_messages_tokens([msg])
            system_tokens += tokens
            result.append(msg)
        remaining = budget - system_tokens
        task_msg = {'role': 'user', 'content': current_task}
        task_tokens = estimate_messages_tokens([task_msg])
        remaining -= task_tokens
        if remaining < 0:
            task_msg['content'] = _truncate_to_tokens(current_task, max(256, remaining + task_tokens))
            task_tokens = estimate_messages_tokens([task_msg])
            remaining = budget - system_tokens - task_tokens
        result.append(task_msg)
        sections: List[tuple[int, List[Dict[str, Any]]]] = []
        if observations:
            recent_obs = observations[-3:] if len(observations) > 3 else observations
            obs_content = json.dumps(recent_obs, default=str, indent=1)
            obs_msg = {'role': 'system', 'content': f'Recent tool results:\n{obs_content}'}
            obs_tokens = estimate_messages_tokens([obs_msg])
            sections.append((obs_tokens, [obs_msg]))
        recent_turns = self._get_recent_turns(chat_history)
        if recent_turns:
            hist_tokens = estimate_messages_tokens(recent_turns)
            sections.append((hist_tokens, recent_turns))
        older_turns = self._get_older_turns(chat_history)
        summarized = self._summarize_turns(older_turns)
        if summarized:
            sum_tokens = estimate_messages_tokens([summarized])
            sections.append((sum_tokens, [summarized]))
        if retrieved_memories:
            mem_content = json.dumps(retrieved_memories, default=str, indent=1)
            mem_msg = {'role': 'system', 'content': f'Relevant memories:\n{mem_content}'}
            mem_tokens = estimate_messages_tokens([mem_msg])
            sections.append((mem_tokens, [mem_msg]))
        for tokens, msgs in sections:
            if tokens <= remaining:
                result.extend(msgs)
                remaining -= tokens
            elif remaining > 256:
                truncated = self._truncate_section(msgs, remaining)
                if truncated:
                    result.extend(truncated)
                    remaining -= estimate_messages_tokens(truncated)
        return result

    def should_summarize(self, chat_history: List[Dict[str, Any]]) -> bool:
        """Return True if the history exceeds the summarization threshold."""
        return len(chat_history) > self.summary_threshold

    def get_turns_for_summary(self, chat_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return the older turns that should be summarized."""
        return self._get_older_turns(chat_history)

    def _get_recent_turns(self, chat_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return the most recent turns that fit in half the remaining budget."""
        if not chat_history:
            return []
        recent = chat_history[-12:]
        return recent

    def _get_older_turns(self, chat_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return turns older than the recent window."""
        if len(chat_history) <= 12:
            return []
        return chat_history[:-12]

    def _summarize_turns(self, turns: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Summarize older turns into a single system message."""
        if not turns:
            return None
        if self.summarize_fn is not None:
            try:
                summary_text = self.summarize_fn(turns)
                if summary_text:
                    return {'role': 'system', 'content': f'Summary of earlier conversation:\n{summary_text}'}
            except Exception as exc:
                logger.warning('Summary function failed, using fallback: %s', exc)
        parts = []
        for t in turns:
            role = t.get('role', '')
            content = t.get('content', '')
            if not isinstance(content, str) or not content:
                continue
            if role == 'user':
                parts.append(f'User asked: {content[:200]}')
            elif role == 'assistant':
                parts.append(f'Assistant said: {content[:200]}')
        if not parts:
            return None
        digest = ' | '.join(parts[-8:])
        digest = _truncate_to_tokens(digest, self.summary_max_tokens)
        return {'role': 'system', 'content': f'Earlier conversation:\n{digest}'}

    def _truncate_section(self, msgs: List[Dict[str, Any]], max_tokens: int) -> Optional[List[Dict[str, Any]]]:
        """Truncate a section of messages to fit within max_tokens."""
        if not msgs:
            return None
        result = []
        remaining = max_tokens
        for msg in msgs:
            tokens = estimate_messages_tokens([msg])
            if tokens <= remaining:
                result.append(msg)
                remaining -= tokens
            else:
                content = msg.get('content', '')
                if isinstance(content, str):
                    truncated_content = _truncate_to_tokens(content, remaining)
                    if truncated_content:
                        result.append({**msg, 'content': truncated_content})
                break
        return result if result else None
