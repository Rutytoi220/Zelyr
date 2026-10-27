"""Zelyr Contextual Memory Hydration Engine.

Implements relevance-based retrieval, composite ranking (BM25 + recency + utility),
and hard token-budgeted prompt hydration for NativeOrchestrator.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from zelyr.db.memory import MemoryStore, get_db_path

logger = logging.getLogger(__name__)


def format_contextual_memory(memories: List[Dict[str, Any]]) -> str:
    """Format matching memories into the standard compact Markdown block."""
    if not memories:
        return ""

    lines = []
    for m in memories:
        cat = m.get("category", "preference")
        key = m.get("key")
        content = m.get("content", "")
        if key:
            lines.append(f"- ({cat}) {key}: {content}")
        else:
            lines.append(f"- ({cat}) {content}")

    return "[Contextual Memory]:\n<recalled_memory>\n" + "\n".join(lines) + "\n</recalled_memory>"


def hydrate_context_memory(
    query: str,
    db_path: Optional[str] = None,
    max_tokens: int = 350,
    store: Optional[MemoryStore] = None,
) -> str:
    """Retrieve relevant memories and format into a token-budgeted markdown block.

    Returns empty string if query is empty or no memories match.
    """
    query_str = (query or "").strip()
    if not query_str:
        return ""

    mem_store = store or MemoryStore(db_path=db_path or get_db_path())
    relevant = mem_store.get_relevant_memories(query_str, max_tokens=max_tokens)
    return format_contextual_memory(relevant)
