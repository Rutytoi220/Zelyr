from zelyr.compat import data_dir as _zelyr_data_dir
import asyncio
import json
import logging
import uuid
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import aiosqlite
from zelyr.memory.db import MemoryDatabaseManager

logger = logging.getLogger(__name__)

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
