"""Async SQLite-backed memory store for Zelyr."""
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import shutil
import sqlite3
import aiosqlite
logger = logging.getLogger(__name__)
from zelyr.memory.protocol import MemoryBackend
from zelyr.memory.semantic import SemanticIndex
import sys
import os

from zelyr.compat import resolve_file

if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # Running in a PyInstaller bundle
    base_dir = Path(sys._MEIPASS) / 'zelyr' / 'memory'
else:
    # Running in normal Python environment
    base_dir = Path(__file__).parent

_SCHEMA = (base_dir / 'schema.sql').read_text(encoding='utf-8')

class MemoryStore(MemoryBackend):
    """Async SQLite-backed memory store implementing the MemoryBackend protocol.

    Provides key-value storage, conversations, summaries, semantic search,
    event logging, and agent session tracking.
    """

    def __init__(self, db_path: Optional[str]=None, semantic_index: Optional[SemanticIndex]=None):
        """Auto-generated docstring.

Args:
    db_path: Argument.
    semantic_index: Argument.

Returns:
    Return value.
"""
        if db_path is None:
            db_path = str(resolve_file(Path.cwd() / "zelyr.db", Path.cwd() / "axiom.db"))
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None
        self._initialized = False
        self._semantic = semantic_index or SemanticIndex()
        import asyncio
        self._lock = asyncio.Lock()

    @property
    def semantic(self) -> SemanticIndex:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return self._semantic

    async def initialize(self) -> None:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        if self._initialized:
            return
        try:
            await self._init_db_connection()
        except sqlite3.DatabaseError as e:
            if self._db:
                await self._db.close()
                self._db = None
            backup_path = f'{self.db_path}.corrupted.{int(time.time())}'
            logger.error(f'Database corruption detected ({e}). Backing up to {backup_path} and resetting.')
            try:
                shutil.move(self.db_path, backup_path)
            except OSError:
                pass
            await self._init_db_connection()

    async def _init_db_connection(self) -> None:
        """Initialize database connection using MemoryDatabaseManager."""
        from zelyr.memory.db import MemoryDatabaseManager
        self._db_mgr = await MemoryDatabaseManager.get_instance(self.db_path)
        self._db = await self._db_mgr.get_connection()
        
        async with self._db.execute('PRAGMA user_version') as cursor:
            row = await cursor.fetchone()
            current_version = row[0] if row else 0
        if current_version == 0:
            await self._db.executescript(_SCHEMA)
            async with self._db.execute('PRAGMA user_version = 2'): pass
            await self._db.commit()
        elif current_version == 1:
            try:
                async with self._db.execute('ALTER TABLE memories ADD COLUMN retrieval_count INTEGER DEFAULT 0'): pass
                async with self._db.execute('ALTER TABLE memories ADD COLUMN confidence_weight REAL DEFAULT 1.0'): pass
            except Exception:
                pass
            async with self._db.execute('PRAGMA user_version = 2'): pass
            await self._db.commit()
        if current_version > 2:
            raise sqlite3.DatabaseError(f'Database version {current_version} is higher than supported version 2. Please upgrade Zelyr.')
        self._initialized = True
        import asyncio
        asyncio.create_task(self._migrate_legacy_embeddings())

    async def _migrate_legacy_embeddings(self) -> None:
        """Migrate existing SQLite embeddings to Qdrant Local Store on startup."""
        if not hasattr(self._semantic, '_vector_store') or not self._semantic._vector_store:
            return
        vector_store = self._semantic._vector_store
        try:
            import asyncio
            count = await asyncio.to_thread(vector_store.count)
            if count > 0:
                return
            db = self._conn()
            async with db.execute('SELECT id, owner_id, owner_type, embedding_json, model FROM embeddings') as cursor:
                rows = await cursor.fetchall()
                if not rows:
                    return
                logger.info(f'Starting migration of {len(list(rows))} legacy semantic embeddings to Qdrant...')
                for row in rows:
                    try:
                        embedding = json.loads(row['embedding_json'])
                        await asyncio.to_thread(vector_store.upsert, row['owner_id'], row['owner_type'], embedding, {'model': row['model']})
                    except Exception as e:
                        logger.warning(f"Failed to migrate embedding {row['id']}: {e}")
                logger.info('Successfully completed semantic embedding migration.')
        except Exception as e:
            logger.error(f'Migration to Qdrant failed: {e}')

    async def close(self) -> None:
        """Close database connection safely."""
        if hasattr(self, '_db_mgr') and self._db_mgr:
            await self._db_mgr.close()
        elif self._db:
            await self._db.close()
        self._db = None
        self._initialized = False
        import asyncio
        self._lock = asyncio.Lock()

    def _conn(self) -> aiosqlite.Connection:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        if not self._initialized or self._db is None:
            raise RuntimeError('MemoryStore not initialized')
        return self._db

    async def set(self, key: str, value: Any, tags: Optional[List[str]]=None, ttl: Optional[float]=None) -> None:
        """Auto-generated docstring.

Args:
    key: Argument.
    value: Argument.
    tags: Argument.
    ttl: Argument.

Returns:
    Return value.
"""
        db = self._conn()
        now = time.time()
        tags_json = json.dumps(tags or [])
        value_json = json.dumps(value)
        async with db.execute('SELECT created_at FROM memories WHERE key = ?', (key,)) as cursor:
            row = await cursor.fetchone()
            created_at = float(row['created_at']) if row else now
            async with db.execute('INSERT OR REPLACE INTO memories\n            (key, value_json, tags_json, created_at, updated_at, ttl_seconds)\n            VALUES (?, ?, ?, ?, ?, ?)', (key, value_json, tags_json, created_at, now, ttl)): pass
            await db.commit()

    async def get(self, key: str) -> Any | None:
        """Auto-generated docstring.

Args:
    key: Argument.

Returns:
    Return value.
"""
        db = self._conn()
        now = time.time()
        async with db.execute('SELECT value_json FROM memories\n            WHERE key = ? AND (ttl_seconds IS NULL OR created_at + ttl_seconds > ?)', (key, now)) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            async with db.execute('UPDATE memories SET retrieval_count = retrieval_count + 1 WHERE key = ?', (key,)): pass
            await db.commit()
            return json.loads(row['value_json'])

    async def delete(self, key: str) -> bool:
        """Auto-generated docstring.

Args:
    key: Argument.

Returns:
    Return value.
"""
        db = self._conn()
        async with db.execute('DELETE FROM memories WHERE key = ?', (key,)) as cursor:
            await db.commit()
            return cursor.rowcount > 0

    async def search(self, tags: List[str]) -> List[Dict[str, Any]]:
        """Auto-generated docstring.

Args:
    tags: Argument.

Returns:
    Return value.
"""
        if not tags:
            return []
        db = self._conn()
        now = time.time()
        async with db.execute('SELECT key, value_json, tags_json, created_at, updated_at\n            FROM memories WHERE ttl_seconds IS NULL OR created_at + ttl_seconds > ?', (now,)) as cursor:
            rows = await cursor.fetchall()
            results: List[Dict[str, Any]] = []
            for row in rows:
                stored_tags = json.loads(row['tags_json'])
                if all((tag in stored_tags for tag in tags)):
                    results.append({'key': row['key'], 'value': json.loads(row['value_json']), 'tags': stored_tags, 'created_at': row['created_at'], 'updated_at': row['updated_at']})
            return results

    async def search_by_tags(self, tags: List[str]) -> List[Dict[str, Any]]:
        """Auto-generated docstring.

Args:
    tags: Argument.

Returns:
    Return value.
"""
        return await self.search(tags)

    async def expire_ttl(self) -> int:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        db = self._conn()
        now = time.time()
        async with db.execute('DELETE FROM memories WHERE ttl_seconds IS NOT NULL AND created_at + ttl_seconds < ?', (now,)) as cursor:
            await db.commit()
            return cursor.rowcount

    async def create_conversation(self, title: str='') -> str:
        """Auto-generated docstring.

Args:
    title: Argument.

Returns:
    Return value.
"""
        db = self._conn()
        conv_id = uuid.uuid4().hex
        async with db.execute('INSERT INTO conversations (id, title, updated_at) VALUES (?, ?, ?)', (conv_id, title, time.time())):
            pass
        await db.commit()
        return conv_id

    async def add_message(self, conversation_id: str, role: str, content: str, metadata: Optional[Dict]=None) -> str:
        """Auto-generated docstring.

Args:
    conversation_id: Argument.
    role: Argument.
    content: Argument.
    metadata: Argument.

Returns:
    Return value.
"""
        db = self._conn()
        msg_id = uuid.uuid4().hex
        async with db.execute('INSERT INTO messages (id, conversation_id, role, content, metadata_json)\n            VALUES (?, ?, ?, ?, ?)', (msg_id, conversation_id, role, content, json.dumps(metadata) if metadata else None)):
            pass
        async with db.execute('UPDATE conversations SET updated_at = ? WHERE id = ?', (time.time(), conversation_id)):
            pass
        await db.commit()
        return msg_id

    async def get_messages(self, conversation_id: str, limit: int=100) -> List[Dict[str, Any]]:
        """Auto-generated docstring.

Args:
    conversation_id: Argument.
    limit: Argument.

Returns:
    Return value.
"""
        db = self._conn()
        async with db.execute('SELECT id, role, content, timestamp, metadata_json\n            FROM messages WHERE conversation_id = ?\n            ORDER BY rowid ASC LIMIT ?', (conversation_id, limit)) as cursor:
            rows = await cursor.fetchall()
            return [{'id': row['id'], 'role': row['role'], 'content': row['content'], 'timestamp': row['timestamp'], 'metadata': json.loads(row['metadata_json']) if row['metadata_json'] else None} for row in rows]

    async def list_conversations(self, limit: int=50) -> List[Dict[str, Any]]:
        """Auto-generated docstring.

Args:
    limit: Argument.

Returns:
    Return value.
"""
        db = self._conn()
        async with db.execute('SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC LIMIT ?', (limit,)) as cursor:
            rows = await cursor.fetchall()
            return [{'id': row['id'], 'title': row['title'], 'created_at': row['created_at'], 'updated_at': row['updated_at']} for row in rows]

    async def save_summary(self, conversation_id: str, summary: str, msg_start: int, msg_end: int) -> None:
        """Auto-generated docstring.

Args:
    conversation_id: Argument.
    summary: Argument.
    msg_start: Argument.
    msg_end: Argument.

Returns:
    Return value.
"""
        db = self._conn()
        async with db.execute('INSERT INTO summaries (conversation_id, summary, msg_start, msg_end)\n            VALUES (?, ?, ?, ?)', (conversation_id, summary, msg_start, msg_end)):
            pass
        await db.commit()

    async def get_summaries(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Auto-generated docstring.

Args:
    conversation_id: Argument.

Returns:
    Return value.
"""
        db = self._conn()
        async with db.execute('SELECT id, summary, msg_start, msg_end, created_at\n            FROM summaries WHERE conversation_id = ?\n            ORDER BY msg_start ASC', (conversation_id,)) as cursor:
            rows = await cursor.fetchall()
            return [{'id': row['id'], 'summary': row['summary'], 'msg_start': row['msg_start'], 'msg_end': row['msg_end'], 'created_at': row['created_at']} for row in rows]

    async def store_embedding(self, owner_id: str, owner_type: str, embedding: List[float], model: str='') -> None:
        """Auto-generated docstring.

Args:
    owner_id: Argument.
    owner_type: Argument.
    embedding: Argument.
    model: Argument.

Returns:
    Return value.
"""
        await self._semantic.store(self._conn(), owner_id, owner_type, embedding, model)

    async def search_similar(self, embedding: List[float], owner_type: str='', top_k: int=5) -> List[Dict[str, Any]]:
        """Auto-generated docstring.

Args:
    embedding: Argument.
    owner_type: Argument.
    top_k: Argument.

Returns:
    Return value.
"""
        return await self._semantic.search(self._conn(), embedding, owner_type, top_k)

    async def log_event(self, event_name: str, payload: Any=None, source: Optional[str]=None) -> int:
        """Auto-generated docstring.

Args:
    event_name: Argument.
    payload: Argument.
    source: Argument.

Returns:
    Return value.
"""
        db = self._conn()
        async with db.execute('INSERT INTO events (event_name, payload_json, source, timestamp) VALUES (?, ?, ?, ?)', (event_name, json.dumps(payload) if payload is not None else None, source, time.time())) as cursor:
            await db.commit()
            return int(cursor.lastrowid) if cursor.lastrowid is not None else 0

    async def get_events(self, event_name: Optional[str]=None, source: Optional[str]=None, limit: int=100) -> List[Dict[str, Any]]:
        """Auto-generated docstring.

Args:
    event_name: Argument.
    source: Argument.
    limit: Argument.

Returns:
    Return value.
"""
        db = self._conn()
        query = 'SELECT id, event_name, payload_json, source, timestamp FROM events WHERE 1=1'
        params: List[Any] = []
        if event_name is not None:
            query += ' AND event_name = ?'
            params.append(event_name)
        if source is not None:
            query += ' AND source = ?'
            params.append(source)
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        async with db.execute(query, tuple(params)) as cursor:
            rows = await cursor.fetchall()
            return [{'id': row['id'], 'event_name': row['event_name'], 'payload': json.loads(row['payload_json']) if row['payload_json'] else None, 'source': row['source'], 'timestamp': row['timestamp']} for row in rows]

    async def create_agent_session(self, agent_name: str, task: str) -> int:
        """Auto-generated docstring.

Args:
    agent_name: Argument.
    task: Argument.

Returns:
    Return value.
"""
        db = self._conn()
        async with db.execute("INSERT INTO agent_sessions (agent_name, task, started_at, status) VALUES (?, ?, ?, 'running')", (agent_name, task, time.time())) as cursor:
            await db.commit()
            return int(cursor.lastrowid) if cursor.lastrowid is not None else 0

    async def log_tool_call(self, session_id: int, tool_name: str, params: Any, result: Any, duration_ms: int, success: bool, error_message: Optional[str]=None) -> int:
        """Auto-generated docstring.

Args:
    session_id: Argument.
    tool_name: Argument.
    params: Argument.
    result: Argument.
    duration_ms: Argument.
    success: Argument.
    error_message: Argument.

Returns:
    Return value.
"""
        db = self._conn()
        async with db.execute('INSERT INTO tool_calls\n            (session_id, tool_name, params_json, result_json,\n             duration_ms, success, error_message, timestamp)\n            VALUES (?, ?, ?, ?, ?, ?, ?, ?)', (session_id, tool_name, json.dumps(params), json.dumps(result), duration_ms, success, error_message, time.time())) as cursor:
            await db.commit()
            return int(cursor.lastrowid) if cursor.lastrowid is not None else 0

    async def get_session_tool_calls(self, session_id: int) -> List[Dict[str, Any]]:
        """Auto-generated docstring.

Args:
    session_id: Argument.

Returns:
    Return value.
"""
        db = self._conn()
        async with db.execute('SELECT tool_name, params_json, result_json, duration_ms, success, error_message\n            FROM tool_calls WHERE session_id = ? ORDER BY timestamp', (session_id,)) as cursor:
            rows = await cursor.fetchall()
            return [{'tool_name': row['tool_name'], 'params': json.loads(row['params_json']) if row['params_json'] else {}, 'result': json.loads(row['result_json']) if row['result_json'] else None, 'duration_ms': row['duration_ms'], 'success': bool(row['success']), 'error_message': row['error_message']} for row in rows]

    async def complete_agent_session(self, session_id: int, result: Any, success: bool=True) -> None:
        """Auto-generated docstring.

Args:
    session_id: Argument.
    result: Argument.
    success: Argument.

Returns:
    Return value.
"""
        db = self._conn()
        status = 'completed' if success else 'failed'
        async with db.execute('UPDATE agent_sessions SET completed_at = ?, result_json = ?, status = ? WHERE id = ?', (time.time(), json.dumps(result), status, session_id)):
            pass
        await db.commit()
