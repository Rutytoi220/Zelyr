import aiosqlite
import logging
from pathlib import Path
from typing import Optional
import asyncio

logger = logging.getLogger(__name__)

class MemoryDatabaseManager:
    """Unified SQLite connection manager for the Semantic Vector Memory.
    Enforces WAL mode, NORMAL synchronous, and strict concurrency safety.
    """
    _instance = None
    _lock = asyncio.Lock()

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._shared_conn: Optional[aiosqlite.Connection] = None

    @classmethod
    async def get_instance(cls, db_path: str | Path) -> 'MemoryDatabaseManager':
        async with cls._lock:
            if cls._instance is None:
                cls._instance = MemoryDatabaseManager(db_path)
            return cls._instance

    async def get_connection(self) -> aiosqlite.Connection:
        """Returns the shared robust aiosqlite connection."""
        async with self._lock:
            if self._shared_conn is None:
                self._shared_conn = await aiosqlite.connect(self.db_path, timeout=10.0) # 10s busy timeout
                self._shared_conn.row_factory = aiosqlite.Row
                
                # Enforce WAL journaling for crash resilience and concurrency
                await self._shared_conn.execute("PRAGMA journal_mode = WAL;")
                await self._shared_conn.execute("PRAGMA synchronous = NORMAL;")
                await self._shared_conn.execute("PRAGMA cache_size = -10000;")
                await self._shared_conn.execute("PRAGMA busy_timeout = 10000;")
                
                await self._shared_conn.commit()
            return self._shared_conn
            
    async def close(self):
        async with self._lock:
            if self._shared_conn:
                await self._shared_conn.close()
                self._shared_conn = None
            MemoryDatabaseManager._instance = None
