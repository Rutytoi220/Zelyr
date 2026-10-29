from zelyr.compat import data_dir as _zelyr_data_dir
import asyncio
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import uuid

import aiosqlite
from zelyr.memory.db import MemoryDatabaseManager

logger = logging.getLogger(__name__)


class SessionManager:
    """Thread-safe SQLite persistent store for multi-turn chat sessions and message history."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self._mem_conn = None
        if db_path is None:
            data_dir = _zelyr_data_dir()
            data_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(data_dir / "sessions.db")
        else:
            self.db_path = str(db_path)
            if self.db_path != ":memory:":
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self.db_path == ":memory:":
            if self._mem_conn is None:
                self._mem_conn = sqlite3.connect(":memory:", timeout=30.0)
                self._mem_conn.row_factory = sqlite3.Row
                self._mem_conn.execute("PRAGMA foreign_keys=ON;")
            return self._mem_conn

        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self) -> None:
        """Initialize sessions and session_messages tables with schema migrations."""
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at REAL,
                    updated_at REAL,
                    model_name TEXT,
                    message_history TEXT
                )
                """
            )
            # Schema migration: ensure all columns exist
            cols = {row["name"] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
            if "updated_at" not in cols:
                conn.execute("ALTER TABLE sessions ADD COLUMN updated_at REAL")
            if "model_name" not in cols:
                conn.execute("ALTER TABLE sessions ADD COLUMN model_name TEXT")
            if "message_history" not in cols:
                conn.execute("ALTER TABLE sessions ADD COLUMN message_history TEXT")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    tool_calls TEXT,
                    timestamp REAL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_messages_sid ON session_messages(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC)")

            # Migrate any legacy message_history into session_messages
            try:
                cur = conn.execute(
                    "SELECT session_id, created_at, message_history FROM sessions "
                    "WHERE message_history IS NOT NULL AND message_history != '' AND message_history != '[]'"
                )
                for s_row in cur.fetchall():
                    s_id = s_row["session_id"]
                    cnt = conn.execute(
                        "SELECT COUNT(*) as c FROM session_messages WHERE session_id = ?",
                        (s_id,),
                    ).fetchone()["c"]
                    if cnt == 0:
                        leg_msgs = json.loads(s_row["message_history"])
                        for m in leg_msgs:
                            tc_val = json.dumps(m["tool_calls"]) if "tool_calls" in m and m["tool_calls"] is not None else None
                            conn.execute(
                                """
                                INSERT INTO session_messages (session_id, role, content, tool_calls, timestamp)
                                VALUES (?, ?, ?, ?, ?)
                                """,
                                (
                                    s_id,
                                    m.get("role", "user"),
                                    m.get("content", ""),
                                    tc_val,
                                    m.get("timestamp") or s_row["created_at"] or time.time(),
                                ),
                            )
            except Exception as exc:
                logger.debug("Failed migrating legacy message_history: %s", exc)

    def create_session(
        self,
        session_id: Optional[str] = None,
        title: str = "New Session",
        model_name: Optional[str] = None,
    ) -> str:
        """Create a new session record and return its unique identifier."""
        sid = session_id or f"session_{uuid.uuid4().hex[:12]}"
        now = time.time()
        initial_history = json.dumps([])
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO sessions (session_id, title, created_at, updated_at, model_name, message_history)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (sid, title, now, now, model_name, initial_history),
            )
        return sid

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a session record by its identifier, including turn count."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                SELECT session_id, title, created_at, updated_at, model_name, message_history
                FROM sessions WHERE session_id = ?
                """,
                (session_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            # Calculate message count from session_messages or fallback
            msg_cursor = conn.execute(
                "SELECT COUNT(*) as count FROM session_messages WHERE session_id = ?",
                (session_id,),
            )
            count = msg_cursor.fetchone()["count"]

            # Fallback to message_history if session_messages has 0
            if count == 0 and row["message_history"]:
                try:
                    legacy_msgs = json.loads(row["message_history"])
                    count = len(legacy_msgs)
                except Exception:
                    count = 0

            return {
                "session_id": row["session_id"],
                "title": row["title"] or "New Session",
                "created_at": row["created_at"],
                "updated_at": row["updated_at"] or row["created_at"],
                "model_name": row["model_name"] or "",
                "message_count": count,
            }

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return a list of recent sessions ordered by latest activity."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                SELECT s.session_id, s.title, s.created_at, s.updated_at, s.model_name, s.message_history,
                       (SELECT COUNT(*) FROM session_messages m WHERE m.session_id = s.session_id) as msg_count
                FROM sessions s
                ORDER BY COALESCE(s.updated_at, s.created_at) DESC
                LIMIT ?
                """,
                (limit,),
            )
            results = []
            for row in cursor.fetchall():
                count = row["msg_count"]
                if count == 0 and row["message_history"]:
                    try:
                        count = len(json.loads(row["message_history"]))
                    except Exception:
                        count = 0
                results.append(
                    {
                        "session_id": row["session_id"],
                        "title": row["title"] or "New Session",
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"] or row["created_at"],
                        "model_name": row["model_name"] or "",
                        "message_count": count,
                    }
                )
            return results

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Append a message turn to the persistent history."""
        now = timestamp if timestamp is not None else time.time()
        tc_json = json.dumps(tool_calls) if tool_calls is not None else None

        # Auto-create session if missing
        self.create_session(session_id=session_id)

        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO session_messages (session_id, role, content, tool_calls, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, role, content, tc_json, now),
            )
            # Update session's updated_at timestamp
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )

            # Synchronize message_history for backwards compatibility
            msg_obj: Dict[str, Any] = {"role": role, "content": content}
            if tool_calls is not None:
                msg_obj["tool_calls"] = tool_calls
            try:
                cur = conn.execute("SELECT message_history FROM sessions WHERE session_id = ?", (session_id,))
                row = cur.fetchone()
                hist = json.loads(row["message_history"]) if row and row["message_history"] else []
                hist.append(msg_obj)
                conn.execute(
                    "UPDATE sessions SET message_history = ? WHERE session_id = ?",
                    (json.dumps(hist), session_id),
                )
            except Exception as exc:
                logger.debug("Could not update legacy message_history: %s", exc)

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve chronological message turns for a given session."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                SELECT role, content, tool_calls, timestamp
                FROM session_messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            )
            rows = cursor.fetchall()
            if rows:
                messages = []
                for r in rows:
                    msg: Dict[str, Any] = {
                        "role": r["role"],
                        "content": r["content"],
                        "tool_calls": None,
                        "timestamp": r["timestamp"],
                    }
                    if r["tool_calls"]:
                        try:
                            msg["tool_calls"] = json.loads(r["tool_calls"])
                        except Exception:
                            msg["tool_calls"] = None
                    messages.append(msg)
                return messages

            # Fallback to legacy message_history
            cur = conn.execute("SELECT message_history FROM sessions WHERE session_id = ?", (session_id,))
            row = cur.fetchone()
            if row and row["message_history"]:
                try:
                    return json.loads(row["message_history"])
                except Exception:
                    pass
            return []

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its associated messages."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM session_messages WHERE session_id = ?", (session_id,))
            cur = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            return cur.rowcount > 0

    def clear_session(self, session_id: str) -> bool:
        """Clear all messages in a session while preserving the session container."""
        now = time.time()
        with self._get_conn() as conn:
            conn.execute("DELETE FROM session_messages WHERE session_id = ?", (session_id,))
            cur = conn.execute(
                "UPDATE sessions SET message_history = '[]', updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            return cur.rowcount > 0

    def count_tokens(self, messages_or_session_id: Union[str, List[Dict[str, Any]]]) -> int:
        """Estimate token count for a message list or session ID."""
        if isinstance(messages_or_session_id, str):
            messages = self.get_messages(messages_or_session_id)
        else:
            messages = messages_or_session_id

        total_words = 0
        for m in messages:
            content = str(m.get("content", ""))
            total_words += len(content.split())
            if m.get("tool_calls"):
                total_words += len(str(m["tool_calls"]).split())
        return int(total_words * 1.3)


class SessionDatabase:
    """Async SQLite database for tracking conversational sessions."""
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            data_dir = _zelyr_data_dir()
            data_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(data_dir / "sessions.db")
        else:
            self.db_path = db_path
            
        self._db = None

    async def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            self._db_mgr = await MemoryDatabaseManager.get_instance(self.db_path)
            self._db = await self._db_mgr.get_connection()
        return self._db

    async def initialize(self) -> None:
        """Create the sessions table if it doesn't exist."""
        try:
            db = await self._conn()
            async with db.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at REAL,
                    message_history TEXT
                )
                """
            ):
                pass
            await db.commit()
        except Exception as e:
            logger.error(f"Failed to initialize SessionDatabase: {e}")

    async def create_session(self, title: str = "New Session") -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())
        created_at = time.time()
        initial_history = json.dumps([])
        
        try:
            db = await self._conn()
            async with db.execute(
                "INSERT INTO sessions (session_id, title, created_at, message_history) VALUES (?, ?, ?, ?)",
                (session_id, title, created_at, initial_history)
            ):
                pass
            await db.commit()
            return session_id
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return session_id

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a session by its ID."""
        try:
            db = await self._conn()
            async with db.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    return {
                        "session_id": row["session_id"],
                        "title": row["title"],
                        "created_at": row["created_at"],
                        "message_history": json.loads(row["message_history"])
                    }
                return None
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None

    async def get_recent_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get a list of recent sessions, ordered by newest first."""
        try:
            db = await self._conn()
            async with db.execute("SELECT session_id, title, created_at FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,)) as cursor:
                rows = await cursor.fetchall()
                return [
                    {
                        "session_id": row["session_id"],
                        "title": row["title"],
                        "created_at": row["created_at"]
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Failed to get recent sessions: {e}")
            return []

    async def append_message(self, session_id: str, message: Dict[str, Any]) -> None:
        """Append a new message to the session's history."""
        try:
            session = await self.get_session(session_id)
            if not session:
                logger.warning(f"Cannot append message to non-existent session {session_id}")
                return
            
            history = session["message_history"]
            history.append(message)
            
            db = await self._conn()
            async with db.execute(
                "UPDATE sessions SET message_history = ? WHERE session_id = ?",
                (json.dumps(history), session_id)
            ):
                pass
            await db.commit()
        except Exception as e:
            logger.error(f"Failed to append message to session {session_id}: {e}")

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session by its ID."""
        try:
            db = await self._conn()
            async with db.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,)) as cursor:
                pass
            await db.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False


__all__ = ["SessionManager", "SessionDatabase"]
