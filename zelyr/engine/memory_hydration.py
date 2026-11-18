"""Proactive FTS5 Contextual Memory Retrieval Engine for Zelyr.

Implements relevance-based retrieval, composite ranking (BM25 + exponential recency decay + utility boost),
token-budgeted prompt hydration (<= 350 tokens ceiling), and access tracking.
"""

from __future__ import annotations

import math
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from zelyr.db.connection import get_db_connection, get_default_db_path, managed_transaction
from zelyr.db.memory import init_db

# Common English stop words to filter out during query keyword extraction
STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further",
    "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's",
    "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves", "tell", "please", "show", "give", "find",
}


def _count_tokens(text: str) -> int:
    """Deterministic token estimation helper (~4 chars per token)."""
    try:
        from zelyr.agents.native_orchestrator import NativeOrchestrator
        return NativeOrchestrator.count_tokens(text)
    except Exception:
        words = len(text.split())
        return max(len(text) // 4, int(words * 1.3), 1)


class MemoryHydrator:
    """Proactive contextual memory retrieval engine with FTS5 composite scoring and token budgeting."""

    def __init__(self, db_path: Optional[str | Path] = None, max_tokens: int = 350) -> None:
        self.db_path = str(db_path or get_default_db_path("memory.db"))
        self.max_tokens = max_tokens
        init_db(self.db_path)

    def _get_conn(self) -> sqlite3.Connection:
        """Centralized connection factory access."""
        return get_db_connection(self.db_path)

    def extract_query_keywords(self, text: str) -> str:
        """Strip active window headers, punctuation, and common stop words.

        Returns sanitized tokens suitable for SQLite FTS5 MATCH queries.
        """
        if not text or not isinstance(text, str):
            return ""

        # 1. Strip active window headers
        cleaned = re.sub(r"\[Active Window:[^\]]*\]\s*", "", text).strip()
        if not cleaned:
            return ""

        # 2. Extract alphanumeric tokens allowing internal dots, hyphens, and underscores (e.g. user.editor, 192.168.1.1)
        raw_tokens = re.findall(r"[\w\.-]+", cleaned)
        filtered_tokens = []
        for tok in raw_tokens:
            tok_stripped = tok.strip(".-_")
            if not tok_stripped:
                continue
            lower_tok = tok_stripped.lower()
            if lower_tok not in STOP_WORDS and len(tok_stripped) >= 2:
                filtered_tokens.append(tok_stripped)

        return " ".join(filtered_tokens)

    def get_relevant_memories(
        self,
        query: str,
        limit: int = 10,
        category: Optional[str] = None,
        update_access: bool = True,
    ) -> List[Dict[str, Any]]:
        """Retrieve active memories scored by composite relevance within the token ceiling.

        Composite Score:
            normalized_bm25 * recency * utility_boost * confidence

        Where:
            - normalized_bm25: Normalized positive FTS5 BM25 relevance score [0.0, 1.0].
            - recency: exp(-Delta t / 604800) (half-life of 7 days).
            - utility_boost: 1.0 + min(access_count * 0.05, 0.5) (up to 50% boost).
            - confidence: confidence weight of the stored memory.
        """
        keywords = self.extract_query_keywords(query)
        if not keywords:
            return []

        tokens = keywords.split()
        clean_tokens = [t.replace('"', '""') for t in tokens if t]
        if not clean_tokens:
            return []

        and_fts = " ".join(f'"{t}"' for t in clean_tokens)
        or_fts = " OR ".join(f'"{t}"' for t in clean_tokens)

        with self._get_conn() as conn:
            def _execute_fts(match_expr: str) -> List[Dict[str, Any]]:
                sql = [
                    "SELECT m.id, m.category, m.key, m.content, m.confidence, m.is_active, ",
                    "m.superseded_by, m.created_at, m.updated_at, m.access_count, memories_fts.rank as bm25_rank ",
                    "FROM memories_fts ",
                    "JOIN memories m ON m.id = memories_fts.rowid ",
                    "WHERE memories_fts MATCH ? AND m.is_active = 1 ",
                ]
                params: List[Any] = [match_expr]
                if category:
                    sql.append("AND m.category = ? ")
                    params.append(category)

                sql.append("ORDER BY memories_fts.rank ASC LIMIT ?")
                params.append(max(limit * 3, 30))

                cursor = conn.execute("".join(sql), tuple(params))
                return [dict(r) for r in cursor.fetchall()]

            candidates: List[Dict[str, Any]] = []
            try:
                candidates = _execute_fts(and_fts)
                if not candidates and len(clean_tokens) > 1:
                    candidates = _execute_fts(or_fts)
            except sqlite3.OperationalError:
                # Fallback to sanitized word query if syntax issues arise
                try:
                    words = [re.sub(r"[^\w]", "", t) for t in clean_tokens]
                    valid_words = [w for w in words if w]
                    if valid_words:
                        fallback_expr = " OR ".join(f'"{w}"' for w in valid_words)
                        candidates = _execute_fts(fallback_expr)
                except Exception:
                    candidates = []

            if not candidates:
                return []

            # Compute composite relevance score
            now = time.time()
            raw_ranks = [abs(float(c.get("bm25_rank", 1.0))) for c in candidates]
            max_rank = max(raw_ranks) if raw_ranks else 1.0
            if max_rank <= 0.0:
                max_rank = 1.0

            scored_candidates: List[Dict[str, Any]] = []
            for c in candidates:
                raw_bm25 = abs(float(c.get("bm25_rank", 1.0)))
                # Normalize BM25 rank to positive [0.0, 1.0] where higher magnitude is better
                normalized_bm25 = raw_bm25 / max_rank

                # Exponential recency decay: exp(-Delta t / 604800) with 7 days half-life
                updated_at = float(c.get("updated_at", now))
                delta_t = max(0.0, now - updated_at)
                recency = math.exp(-delta_t / 604800.0)

                # Utility boost: 1.0 + min(access_count * 0.05, 0.5)
                access_count = int(c.get("access_count", 0))
                utility_boost = 1.0 + min(access_count * 0.05, 0.5)

                confidence = float(c.get("confidence", 1.0))

                composite_score = normalized_bm25 * recency * utility_boost * confidence
                c_copy = dict(c)
                c_copy["composite_score"] = composite_score
                scored_candidates.append(c_copy)

            # Sort descending by composite score
            scored_candidates.sort(key=lambda x: x["composite_score"], reverse=True)

            # Evaluate token consumption against max_tokens ceiling
            header = "[Contextual Memory]:\n<recalled_memory>\n"
            footer = "\n</recalled_memory>"

            selected: List[Dict[str, Any]] = []
            current_lines: List[str] = []

            for candidate in scored_candidates:
                cat = candidate.get("category", "preference")
                k = candidate.get("key")
                cnt = candidate.get("content", "")
                if k:
                    line = f"- ({cat}) {k}: {cnt}"
                else:
                    line = f"- ({cat}) {cnt}"

                test_block = header + "\n".join(current_lines + [line]) + footer
                if _count_tokens(test_block) <= self.max_tokens:
                    selected.append(candidate)
                    current_lines.append(line)
                    if len(selected) >= limit:
                        break
                else:
                    # Token budget reached
                    break

            # Update access counts and updated_at for injected candidates
            if update_access and selected:
                try:
                    with managed_transaction(conn):
                        for s in selected:
                            conn.execute(
                                "UPDATE memories SET access_count = access_count + 1, updated_at = ? WHERE id = ?",
                                (now, s["id"]),
                            )
                            s["access_count"] = int(s.get("access_count", 0)) + 1
                            s["updated_at"] = now
                except Exception:
                    pass

            return selected

    def format_recalled_memory_block(self, memories: List[Dict[str, Any]]) -> str:
        """Format recalled memories into the standard bounded markdown block.

        Returns empty string if memories is empty (zero token injection).
        """
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
    db_path: Optional[str | Path] = None,
    max_tokens: int = 350,
) -> str:
    """Convenience helper to retrieve and format contextual memories."""
    hydrator = MemoryHydrator(db_path=db_path, max_tokens=max_tokens)
    memories = hydrator.get_relevant_memories(query)
    return hydrator.format_recalled_memory_block(memories)


__all__ = [
    "MemoryHydrator",
    "hydrate_context_memory",
    "STOP_WORDS",
]
