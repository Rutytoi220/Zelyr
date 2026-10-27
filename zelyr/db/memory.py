"""Zelyr Persistent Memory & Session Storage.

SQLite-backed persistent conversational history and structured FTS5 memory store for Zelyr.
Provides concurrency-hardened (WAL mode, busy_timeout=5000) persistent storage with
key versioning, category segmentation, and full-text search.
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from zelyr.compat import get_env, repository_memory_db
from zelyr.config import get_data_dir

VALID_CATEGORIES: Set[str] = {
    "preference",
    "environment",
    "homelab",
    "workflow",
    "capability",
}

MEMORIES_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL CHECK(category IN ('preference', 'environment', 'homelab', 'workflow', 'capability')),
    key TEXT,
    content TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    is_active INTEGER DEFAULT 1,
    superseded_by INTEGER REFERENCES memories(id),
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    access_count INTEGER DEFAULT 0,
    embedding BLOB
);
"""

MEMORIES_INDEXES_DDL = """
CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(key);
CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(is_active);
"""

MEMORIES_FTS_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    key,
    content,
    category,
    content='memories',
    content_rowid='id'
);
"""

MEMORIES_TRIGGERS_DDL = """
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
  INSERT INTO memories_fts(rowid, key, content, category) VALUES (new.id, new.key, new.content, new.category);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
  INSERT INTO memories_fts(memories_fts, rowid, key, content, category) VALUES('delete', old.id, old.key, old.content, old.category);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
  INSERT INTO memories_fts(memories_fts, rowid, key, content, category) VALUES('delete', old.id, old.key, old.content, old.category);
  INSERT INTO memories_fts(rowid, key, content, category) VALUES (new.id, new.key, new.content, new.category);
END;
"""


def get_db_path() -> str:
    """Return the absolute path to the SQLite memory database.

    ``ZELYR_MEMORY_DB`` is preferred, with ``AXIOM_MEMORY_DB`` retained as a
    compatibility fallback. Existing ``axiom_memory.db`` files are reused
    in place; they are never copied or deleted.
    """
    env_path = get_env("MEMORY_DB")
    if env_path:
        return str(Path(env_path).expanduser().resolve())

    repo_root = Path(__file__).resolve().parent.parent.parent
    parts = repo_root.parts
    if "site-packages" in parts or "dist-packages" in parts:
        user_data = get_data_dir()
        user_data.mkdir(parents=True, exist_ok=True)
        legacy_db = user_data / "axiom_memory.db"
        new_db = user_data / "zelyr_memory.db"
        if not new_db.exists() and legacy_db.exists():
            return str(legacy_db)
        return str(new_db)

    return str(repository_memory_db(repo_root))


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Create and return a SQLite connection configured for WAL mode, high busy timeout,
    and foreign key constraints.
    """
    path = db_path or get_db_path()
    conn = sqlite3.connect(path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize the SQLite memory tables, FTS5 virtual table, and triggers if they do not exist."""
    path = db_path or get_db_path()
    with get_connection(path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );
        """)
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(content, session_id UNINDEXED, role UNINDEXED);
        """)

        # Migration check for existing memories table
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories'")
        has_memories = cursor.fetchone() is not None

        if has_memories:
            col_info = conn.execute("PRAGMA table_info(memories)").fetchall()
            col_names = {row["name"] for row in col_info}
            if "category" not in col_names:
                # Migrate legacy memories table
                conn.execute("ALTER TABLE memories RENAME TO memories_legacy_backup;")
                conn.execute(MEMORIES_TABLE_DDL)
                conn.execute("""
                    INSERT INTO memories (id, category, content, confidence, is_active, created_at, updated_at, access_count, embedding)
                    SELECT id, 'preference', content, 1.0, 1,
                           COALESCE(CAST(strftime('%s', created_at) AS REAL), CAST(strftime('%s', 'now') AS REAL)),
                           COALESCE(CAST(strftime('%s', created_at) AS REAL), CAST(strftime('%s', 'now') AS REAL)),
                           0, embedding
                    FROM memories_legacy_backup;
                """)
                conn.execute("DROP TABLE memories_legacy_backup;")
        else:
            conn.execute(MEMORIES_TABLE_DDL)

        # Indexes
        conn.executescript(MEMORIES_INDEXES_DDL)

        # FTS5 Virtual Table & Triggers
        conn.execute(MEMORIES_FTS_DDL)
        conn.executescript(MEMORIES_TRIGGERS_DDL)

        # Synchronize FTS5 index if needed
        try:
            conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild');")
        except sqlite3.OperationalError:
            pass


class MemoryStore:
    """Thread-safe persistent memory store backed by SQLite + FTS5 with versioning and WAL concurrency."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or get_db_path()
        init_db(self.db_path)
        self._write_lock = threading.Lock()

    def _get_conn(self) -> sqlite3.Connection:
        return get_connection(self.db_path)

    def store(
        self,
        content: str,
        category: str = "preference",
        key: Optional[str] = None,
        confidence: float = 1.0,
    ) -> int:
        """Store a structured memory with category validation and automatic key-based superseding.

        If key is provided and an active memory with that key already exists:
        atomically set existing row is_active = 0, superseded_by = new_row_id.
        """
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category '{category}'. Must be one of: {sorted(VALID_CATEGORIES)}"
            )

        content = str(content).strip()
        if not content:
            raise ValueError("Memory content cannot be empty.")

        now = time.time()
        with self._write_lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO memories (category, key, content, confidence, is_active, superseded_by, created_at, updated_at, access_count)
                    VALUES (?, ?, ?, ?, 1, NULL, ?, ?, 0)
                    """,
                    (category, key, content, float(confidence), now, now),
                )
                new_id = cursor.lastrowid
                if new_id is None:
                    raise RuntimeError("Failed to retrieve lastrowid after inserting memory.")

                # If key was provided, atomically supersede any prior active memory with same key
                if key:
                    conn.execute(
                        """
                        UPDATE memories
                        SET is_active = 0, superseded_by = ?, updated_at = ?
                        WHERE key = ? AND is_active = 1 AND id != ?
                        """,
                        (new_id, now, key, new_id),
                    )
                conn.commit()
                return new_id

    def _infer_fact_key(self, content: str, category: str) -> Optional[str]:
        """Infer an appropriate unique key for contradiction resolution and superseding."""
        text = content.strip()
        lower = text.lower()

        # 1. Check for explicit key prefix e.g. "user.shell: zsh" or "theme = tokyo-night"
        prefix_match = re.match(r"^([a-zA-Z0-9_\.-]{2,30})\s*[:=]\s*(.+)$", text)
        if prefix_match:
            k = prefix_match.group(1).lower()
            if not k.startswith("http") and not k.startswith("note"):
                return k

        # 2. Check canonical domain key patterns
        domain_patterns = [
            (r"\b(shell|zsh|bash|fish|nushell)\b", "user.shell"),
            (r"\b(editor|vim|neovim|nvim|emacs|vscode|sublime|nano)\b", "user.editor"),
            (r"\b(theme|tokyo[\s_-]?night|catppuccin|dracula|gruvbox|solarized|nord)\b", "user.theme"),
            (r"\b(browser|firefox|chrome|chromium|brave|safari)\b", "user.browser"),
            (r"\b(git\s+rebase|git\s+merge|git\s+commit|git\s+branch|git\s+workflow)\b", "workflow.git"),
            (r"\b(python|rust|golang|nodejs|typescript|javascript)\b.*\b(prefer|version|standard)\b", "user.language"),
            (r"\b(postgres|postgresql|mysql|sqlite|redis|mongodb)\b", "homelab.db"),
            (r"\b(docker|podman|container|compose)\b", "homelab.container"),
            (r"\b(server\s+ip|ip\s+address|host\b|port\s+\d+)\b", "environment.server"),
            (r"\b(os|distro|ubuntu|debian|arch[\s_-]?linux|fedora|nixos|macos)\b", "environment.os"),
        ]

        for pat, matched_key in domain_patterns:
            if re.search(pat, lower):
                return matched_key

        # 3. Match against existing active keys in the same category
        try:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT key, content FROM memories WHERE category = ? AND is_active = 1 AND key IS NOT NULL",
                    (category,),
                ).fetchall()
                for row in rows:
                    existing_key = row["key"]
                    existing_content = row["content"].lower()
                    existing_tokens = set(re.findall(r"\w{4,}", existing_content))
                    new_tokens = set(re.findall(r"\w{4,}", lower))
                    overlap = existing_tokens.intersection(new_tokens)
                    if len(overlap) >= 2:
                        return existing_key
        except Exception:
            pass

        return None

    def store_fact(
        self,
        content: str,
        category: str = "preference",
        key: Optional[str] = None,
        confidence: float = 1.0,
    ) -> int:
        """Store a fact with key inference and automatic contradiction/superseding resolution.

        If key is None, infers an appropriate key based on common entity patterns
        or existing active memories in the same category.
        """
        content = str(content).strip()
        if not content:
            raise ValueError("Memory content cannot be empty.")

        if category not in VALID_CATEGORIES:
            category = "preference"

        if not key:
            key = self._infer_fact_key(content, category)

        return self.store(content=content, category=category, key=key, confidence=confidence)

    def get(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a memory by ID (active or inactive) and increment its access count."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                SELECT id, category, key, content, confidence, is_active, superseded_by, created_at, updated_at, access_count
                FROM memories
                WHERE id = ?
                """,
                (memory_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            data = dict(row)
            # Increment access count
            try:
                with self._write_lock:
                    conn.execute(
                        "UPDATE memories SET access_count = access_count + 1 WHERE id = ?",
                        (memory_id,),
                    )
                    conn.commit()
                data["access_count"] += 1
            except Exception:
                pass
            return data

    def get_by_key(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve the currently active memory for a given key."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                SELECT id, category, key, content, confidence, is_active, superseded_by, created_at, updated_at, access_count
                FROM memories
                WHERE key = ? AND is_active = 1
                ORDER BY id DESC LIMIT 1
                """,
                (key,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            data = dict(row)
            try:
                with self._write_lock:
                    conn.execute(
                        "UPDATE memories SET access_count = access_count + 1 WHERE id = ?",
                        (data["id"],),
                    )
                    conn.commit()
                data["access_count"] += 1
            except Exception:
                pass
            return data

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 10,
        include_inactive: bool = False,
    ) -> List[Dict[str, Any]]:
        """Search memories using FTS5 MATCH with BM25 ranking on active rows.

        Supports querying across content, key, and category fields.
        """
        query_str = (query or "").strip()
        if not query_str:
            return []

        tokens = re.findall(r"[\w\.-]+", query_str)
        if not tokens:
            return []

        clean_tokens = [t.replace('"', '""') for t in tokens if t]
        # Build AND query
        and_fts = " ".join(f'"{t}"' for t in clean_tokens)
        # Build OR query for fallback
        or_fts = " OR ".join(f'"{t}"' for t in clean_tokens)

        with self._get_conn() as conn:
            def _execute_fts(match_expr: str) -> List[Dict[str, Any]]:
                sql = [
                    "SELECT m.id, m.category, m.key, m.content, m.confidence, m.is_active, "
                    "m.superseded_by, m.created_at, m.updated_at, m.access_count, memories_fts.rank as bm25_rank "
                    "FROM memories_fts "
                    "JOIN memories m ON m.id = memories_fts.rowid "
                    "WHERE memories_fts MATCH ?"
                ]
                params: List[Any] = [match_expr]

                if not include_inactive:
                    sql.append("AND m.is_active = 1")
                if category:
                    sql.append("AND m.category = ?")
                    params.append(category)

                sql.append("ORDER BY memories_fts.rank ASC LIMIT ?")
                params.append(limit)

                c = conn.execute(" ".join(sql), tuple(params))
                return [dict(r) for r in c.fetchall()]

            try:
                results = _execute_fts(and_fts)
                if not results and len(clean_tokens) > 1:
                    results = _execute_fts(or_fts)
                return results
            except sqlite3.OperationalError:
                # If syntax error in match expression, try safe sanitized word search
                try:
                    words = [re.sub(r"[^\w]", "", t) for t in clean_tokens]
                    valid_words = [w for w in words if w]
                    if valid_words:
                        fallback_expr = " OR ".join(f'"{w}"' for w in valid_words)
                        return _execute_fts(fallback_expr)
                except Exception:
                    pass
                return []

    def get_relevant_memories(
        self,
        query: str,
        max_tokens: int = 350,
        category: Optional[str] = None,
        limit: int = 15,
        update_access: bool = True,
    ) -> List[Dict[str, Any]]:
        """Retrieve active memories scored by composite relevance within a hard token budget.

        Composite Score:
            normalized_bm25 * recency_decay * (1.0 + min(access_count * 0.05, 0.5)) * confidence

        Where:
            - normalized_bm25 is normalized positive BM25 relevance score from FTS5.
            - recency_decay is a smooth decay factor: 1.0 / (1.0 + 0.05 * age_in_days).
            - utility_boost is (1.0 + min(access_count * 0.05, 0.5)).
            - confidence is the memory confidence weight.
        """
        query_str = (query or "").strip()
        if not query_str:
            return []

        candidates = self.search(
            query=query_str,
            category=category,
            limit=max(limit * 2, 30),
            include_inactive=False,
        )
        if not candidates:
            return []

        now = time.time()
        raw_ranks = [abs(float(c.get("bm25_rank", 1.0))) for c in candidates]
        max_rank = max(raw_ranks) if raw_ranks else 1.0
        if max_rank <= 0.0:
            max_rank = 1.0

        scored_candidates = []
        for c in candidates:
            raw_bm25 = abs(float(c.get("bm25_rank", 1.0)))
            normalized_bm25 = raw_bm25 / max_rank

            updated_at = float(c.get("updated_at", now))
            age_days = max(0.0, (now - updated_at) / 86400.0)
            recency_decay = 1.0 / (1.0 + 0.05 * age_days)

            access_count = int(c.get("access_count", 0))
            utility_boost = 1.0 + min(access_count * 0.05, 0.5)

            confidence = float(c.get("confidence", 1.0))

            composite_score = normalized_bm25 * recency_decay * utility_boost * confidence
            c_copy = dict(c)
            c_copy["composite_score"] = composite_score
            scored_candidates.append(c_copy)

        scored_candidates.sort(key=lambda x: x["composite_score"], reverse=True)

        try:
            from zelyr.agents.native_orchestrator import NativeOrchestrator
            count_fn = NativeOrchestrator.count_tokens
        except Exception:
            def count_fn(text: str) -> int:
                words = len(text.split())
                return max(len(text) // 4, int(words * 1.3), 1)

        header = "[Contextual Memory]:\n<recalled_memory>\n"
        footer = "\n</recalled_memory>"

        selected: List[Dict[str, Any]] = []
        current_lines: List[str] = []

        for candidate in scored_candidates:
            cat = candidate.get("category", "preference")
            key = candidate.get("key")
            content = candidate.get("content", "")
            if key:
                line = f"- ({cat}) {key}: {content}"
            else:
                line = f"- ({cat}) {content}"

            test_block = header + "\n".join(current_lines + [line]) + footer
            if count_fn(test_block) <= max_tokens:
                selected.append(candidate)
                current_lines.append(line)
            else:
                break

        if update_access and selected:
            try:
                with self._write_lock:
                    with self._get_conn() as conn:
                        for s in selected:
                            conn.execute(
                                "UPDATE memories SET access_count = access_count + 1 WHERE id = ?",
                                (s["id"],),
                            )
                            s["access_count"] += 1
                        conn.commit()
            except Exception:
                pass

        return selected

    def update(
        self,
        memory_id: int,
        content: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> bool:
        """Update content and/or confidence of an existing memory."""
        if content is None and confidence is None:
            # Nothing to update, check existence
            with self._get_conn() as conn:
                r = conn.execute("SELECT 1 FROM memories WHERE id = ?", (memory_id,)).fetchone()
                return r is not None

        clauses = ["updated_at = ?"]
        now = time.time()
        params: List[Any] = [now]

        if content is not None:
            cleaned = str(content).strip()
            if not cleaned:
                raise ValueError("Updated memory content cannot be empty.")
            clauses.append("content = ?")
            params.append(cleaned)

        if confidence is not None:
            clauses.append("confidence = ?")
            params.append(float(confidence))

        params.append(memory_id)

        with self._write_lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    f"UPDATE memories SET {', '.join(clauses)} WHERE id = ?",
                    tuple(params),
                )
                conn.commit()
                return cursor.rowcount > 0

    def delete(self, memory_id: int, hard_delete: bool = False) -> bool:
        """Delete or soft-delete a memory by ID.

        If hard_delete=False (default): soft-deletes by setting is_active = 0.
        If hard_delete=True: permanently removes the record and FTS index entry.
        """
        with self._write_lock:
            with self._get_conn() as conn:
                if hard_delete:
                    cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                    conn.commit()
                    return cursor.rowcount > 0
                else:
                    # Soft delete: only succeed if it was currently active
                    cursor = conn.execute(
                        "UPDATE memories SET is_active = 0, updated_at = ? WHERE id = ? AND is_active = 1",
                        (time.time(), memory_id),
                    )
                    conn.commit()
                    return cursor.rowcount > 0

    def purge_inactive(self, older_than_seconds: Optional[float] = None) -> int:
        """Permanently delete inactive (soft-deleted) memories.

        If older_than_seconds is provided, only memories whose updated_at < (now - older_than_seconds)
        are purged.
        """
        now = time.time()
        with self._write_lock:
            with self._get_conn() as conn:
                if older_than_seconds is not None:
                    cutoff = now - float(older_than_seconds)
                    cursor = conn.execute(
                        "DELETE FROM memories WHERE is_active = 0 AND updated_at < ?",
                        (cutoff,),
                    )
                else:
                    cursor = conn.execute("DELETE FROM memories WHERE is_active = 0")
                conn.commit()
                return cursor.rowcount


# ---------------------------------------------------------------------------
# Existing Chat Session History Functions (Preserved for 100% Backwards Compatibility)
# ---------------------------------------------------------------------------

def create_session(
    session_id: str,
    title: str = "New Session",
    db_path: Optional[str] = None,
) -> None:
    """Insert a new session record into the database."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT OR IGNORE INTO sessions (id, title, created_at)
            VALUES (?, ?, ?)
            """,
            (session_id, title, now),
        )


def get_latest_session_id(db_path: Optional[str] = None) -> Optional[str]:
    """Retrieve the most recently created session ID."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT id FROM sessions ORDER BY created_at DESC, rowid DESC LIMIT 1"
        )
        row = cursor.fetchone()
        return row["id"] if row else None


def add_message(
    session_id: str,
    role: str,
    content: str,
    db_path: Optional[str] = None,
) -> None:
    """Add a message entry to the session history."""
    init_db(db_path)
    create_session(session_id, db_path=db_path)
    with get_connection(db_path) as conn:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT INTO messages (session_id, role, content, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, role, content, now),
        )
        conn.execute(
            """
            INSERT INTO messages_fts (content, session_id, role)
            VALUES (?, ?, ?)
            """,
            (content, session_id, role),
        )


def get_session_messages(
    session_id: str,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve all messages for a given session in chronological order."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT id, session_id, role, content, timestamp
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        )
        return [dict(row) for row in cursor.fetchall()]


def get_all_sessions(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all stored sessions."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT id, title, created_at FROM sessions ORDER BY created_at DESC"
        )
        return [dict(row) for row in cursor.fetchall()]


def search_historical_context(
    query: str,
    session_id: str,
    limit: int = 5,
    db_path: Optional[str] = None,
) -> List[str]:
    """Search for relevant past messages using SQLite FTS5."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        try:
            # Simple sanitization for FTS5 syntax
            safe_query = query.replace('"', '""')
            cursor = conn.execute(
                """
                SELECT role, content FROM messages_fts
                WHERE messages_fts MATCH ? AND session_id = ?
                ORDER BY rank LIMIT ?
                """,
                (f'"{safe_query}"', session_id, limit),
            )
            results = []
            for row in cursor.fetchall():
                results.append(f"[{row['role']}]: {row['content']}")
            return results
        except sqlite3.OperationalError:
            # Fallback if query syntax is invalid
            return []


def store_fact(
    content: str,
    category: str = "preference",
    key: Optional[str] = None,
    confidence: float = 1.0,
    db_path: Optional[str] = None,
) -> int:
    """Convenience top-level wrapper to store a fact via MemoryStore."""
    store = MemoryStore(db_path=db_path)
    return store.store_fact(content=content, category=category, key=key, confidence=confidence)
