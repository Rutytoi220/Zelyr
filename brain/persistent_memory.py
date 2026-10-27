"""Persistent memory for Zelyr — SQLite-backed action storage.

Stores all executed actions to disk so patterns can be detected
across sessions. Loaded on startup, written after each action.
"""

import os
import sqlite3
import time
from pathlib import Path
from typing import List, Dict, Optional
from utils.logger import get_logger
from utils.config import get_config
from zelyr.compat import resolve_file

logger = get_logger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PersistentMemory:
    """SQLite-backed persistent storage for action history."""

    def __init__(self, db_path: Optional[str] = None):
        cfg = get_config() or {}
        raw = db_path or cfg.get('storage', {}).get('db_path', 'data/zelyr.db')
        # Resolve relative paths against project root and reuse legacy AXIOM data.
        resolved = Path(raw) if os.path.isabs(raw) else Path(_PROJECT_ROOT) / raw
        if not os.path.isabs(raw):
            resolved = resolve_file(resolved, Path(_PROJECT_ROOT) / 'data' / 'axiom.db')
        self.db_path = str(resolved)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        """Create tables and indices if they don't exist."""
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute('''
            CREATE TABLE IF NOT EXISTS actions (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                action    TEXT    NOT NULL,
                params    TEXT    DEFAULT '',
                ok        INTEGER NOT NULL DEFAULT 1,
                message   TEXT    DEFAULT '',
                timestamp REAL    NOT NULL
            )
        ''')
        self._conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_actions_ts ON actions(timestamp)'
        )
        self._conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_actions_name ON actions(action)'
        )
        self._conn.commit()
        logger.info("Persistent memory ready: %s", self.db_path)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_action(self, action: str, params: str, ok: bool, message: str) -> None:
        """Persist a single action record."""
        if self._conn is None:
            return
        try:
            self._conn.execute(
                'INSERT INTO actions (action, params, ok, message, timestamp) '
                'VALUES (?, ?, ?, ?, ?)',
                (action, params, int(ok), message, time.time()),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            logger.error("Failed to persist action: %s", e)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_recent(self, n: int = 20) -> List[Dict]:
        """Return the *n* most recent actions (newest first)."""
        if self._conn is None:
            return []
        try:
            cur = self._conn.execute(
                'SELECT action, params, ok, message, timestamp '
                'FROM actions ORDER BY timestamp DESC LIMIT ?', (n,)
            )
            return [dict(r) for r in cur.fetchall()]
        except sqlite3.Error as e:
            logger.error("Query failed: %s", e)
            return []

    def get_actions_since(self, since_ts: float) -> List[Dict]:
        """Return all actions since a Unix timestamp (oldest first)."""
        if self._conn is None:
            return []
        try:
            cur = self._conn.execute(
                'SELECT action, params, ok, message, timestamp '
                'FROM actions WHERE timestamp >= ? ORDER BY timestamp',
                (since_ts,),
            )
            return [dict(r) for r in cur.fetchall()]
        except sqlite3.Error as e:
            logger.error("Query failed: %s", e)
            return []

    def get_action_count(self, action: str, since_ts: float) -> int:
        """Count occurrences of *action* since a timestamp."""
        if self._conn is None:
            return 0
        try:
            cur = self._conn.execute(
                'SELECT COUNT(*) FROM actions WHERE action = ? AND timestamp >= ?',
                (action, since_ts),
            )
            return cur.fetchone()[0]
        except sqlite3.Error as e:
            logger.error("Count query failed: %s", e)
            return 0

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# ------------------------------------------------------------------
# Global singleton
# ------------------------------------------------------------------

_instance: Optional[PersistentMemory] = None


def get_persistent_memory() -> PersistentMemory:
    """Get or create the global PersistentMemory instance."""
    global _instance
    if _instance is None:
        _instance = PersistentMemory()
    return _instance
