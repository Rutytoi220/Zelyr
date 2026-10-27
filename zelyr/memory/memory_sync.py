"""Synchronous adapter over the async MemoryStore for Zelyr.

Provides a sync interface for consumers that cannot use async/await
(OrchestratorAgent, Engine, CLI). Delegates all work to MemoryStore.
"""
from typing import Any, Dict, List, Optional
from zelyr.core.async_bridge import run_sync as _run
from zelyr.memory.memory_async import MemoryStore
import logging
import threading
import uuid
logger = logging.getLogger(__name__)

class SyncMemoryStore:
    """Sync adapter wrapping the async MemoryStore.
    
    Also provides conversation tracking and semantic search for the CLI.
    """

    def __init__(self, db_path: Optional[str]=None, embedding_provider: Optional[Any]=None):
        """Auto-generated docstring.

Args:
    db_path: Argument.
    embedding_provider: Argument.

Returns:
    Return value.
"""
        self._store = MemoryStore(db_path)
        self._current_conversation: Optional[str] = None
        self._lock = threading.RLock()
        self._embedding_provider = embedding_provider
        _run(self._store.initialize())

    @property
    def store(self) -> MemoryStore:
        """Access the underlying async store."""
        return self._store

    @property
    def _conn(self):
        """Direct connection access for backward compatibility."""
        return self._store._conn()

    def close(self) -> None:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        _run(self._store.close())

    def set(self, key: str, value: Any, tags: Optional[List[str]]=None, ttl: Optional[float]=None) -> None:
        """Auto-generated docstring.

Args:
    key: Argument.
    value: Argument.
    tags: Argument.
    ttl: Argument.

Returns:
    Return value.
"""
        _run(self._store.set(key, value, tags, ttl))

    def get(self, key: str) -> Any | None:
        """Auto-generated docstring.

Args:
    key: Argument.

Returns:
    Return value.
"""
        return _run(self._store.get(key))

    def delete(self, key: str) -> bool:
        """Auto-generated docstring.

Args:
    key: Argument.

Returns:
    Return value.
"""
        return _run(self._store.delete(key))

    def search(self, tags: List[str]) -> List[Dict[str, Any]]:
        """Auto-generated docstring.

Args:
    tags: Argument.

Returns:
    Return value.
"""
        return _run(self._store.search(tags))

    def expire_ttl(self) -> int:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return _run(self._store.expire_ttl())

    def log_event(self, event_name: str, payload: Any=None, source: Optional[str]=None, data: Any=None) -> int:
        """Auto-generated docstring.

Args:
    event_name: Argument.
    payload: Argument.
    source: Argument.
    data: Argument.

Returns:
    Return value.
"""
        return _run(self._store.log_event(event_name, payload or data, source))

    def get_events(self, event_name: Optional[str]=None, source: Optional[str]=None, limit: int=100) -> List[Dict[str, Any]]:
        """Auto-generated docstring.

Args:
    event_name: Argument.
    source: Argument.
    limit: Argument.

Returns:
    Return value.
"""
        return _run(self._store.get_events(event_name, source, limit))

    def list_keys(self) -> List[str]:
        """List all non-expired memory keys."""
        return _run(self._list_keys_async())

    async def _list_keys_async(self) -> List[str]:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        import time as _time
        db = self._store._conn()
        now = _time.time()
        cursor = await db.execute('SELECT key FROM memories WHERE ttl_seconds IS NULL OR created_at + ttl_seconds > ?', (now,))
        rows = await cursor.fetchall()
        return [row['key'] for row in rows]

    def expire(self, key: str | None = None, ttl_seconds: float | None = None) -> int:
        """Backward-compatible expire method.

        Called with no args: cleans up expired entries.
        Called with key + ttl_seconds: sets TTL on that key.
        """
        if key is None or ttl_seconds is None:
            return self.expire_ttl()
        assert key is not None and ttl_seconds is not None
        return _run(self._set_ttl(key, ttl_seconds))

    async def _set_ttl(self, key: str, ttl: float) -> int:
        """Auto-generated docstring.

Args:
    key: Argument.
    ttl: Argument.

Returns:
    Return value.
"""
        import time as _time
        db = self._store._conn()
        now = _time.time()
        await db.execute('UPDATE memories SET ttl_seconds = ?, created_at = ?, updated_at = ? WHERE key = ?', (ttl, now, now, key))
        await db.commit()
        return 0

    def create_agent_session(self, agent_name: str, task: str) -> int:
        """Auto-generated docstring.

Args:
    agent_name: Argument.
    task: Argument.

Returns:
    Return value.
"""
        return _run(self._store.create_agent_session(agent_name, task))

    def log_tool_call(self, session_id: int, tool_name: str, params: Any, result: Any, duration_ms: int, success: bool, error_message: Optional[str]=None) -> int:
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
        return _run(self._store.log_tool_call(session_id, tool_name, params, result, duration_ms, success, error_message))

    def get_session_tool_calls(self, session_id: int) -> List[Dict[str, Any]]:
        """Auto-generated docstring.

Args:
    session_id: Argument.

Returns:
    Return value.
"""
        return _run(self._store.get_session_tool_calls(session_id))

    def complete_agent_session(self, session_id: int, result: Any, success: bool=True) -> None:
        """Auto-generated docstring.

Args:
    session_id: Argument.
    result: Argument.
    success: Argument.

Returns:
    Return value.
"""
        _run(self._store.complete_agent_session(session_id, result, success))

    def create_conversation(self, title: str='') -> str:
        """Auto-generated docstring.

Args:
    title: Argument.

Returns:
    Return value.
"""
        conversation_id = _run(self._store.create_conversation(title))
        with self._lock:
            self._current_conversation = conversation_id
        return conversation_id

    def set_conversation(self, conversation_id: str) -> None:
        """Auto-generated docstring.

Args:
    conversation_id: Argument.

Returns:
    Return value.
"""
        with self._lock:
            self._current_conversation = conversation_id

    def get_conversation(self) -> Optional[str]:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        with self._lock:
            return self._current_conversation

    def add_message(self, role: str, content: str, metadata: Optional[Dict]=None) -> str:
        """Auto-generated docstring.

Args:
    role: Argument.
    content: Argument.
    metadata: Argument.

Returns:
    Return value.
"""
        with self._lock:
            conversation_id = self._current_conversation
        if not conversation_id:
            raise RuntimeError('No active conversation')
        message_id = _run(self._store.add_message(conversation_id, role, content, metadata))
        if self._embedding_provider is not None:
            self._try_embed_message(message_id, content)
        return message_id

    def _try_embed_message(self, message_id: str, content: str) -> None:
        """Best-effort embedding of a stored message for semantic search."""
        if self._embedding_provider is None:
            return
        try:
            embedding = self._embedding_provider.embed(content)
        except Exception as exc:
            logger.warning('Failed to compute embedding for message %s: %s', message_id, exc)
            return
        if not embedding:
            return
        try:
            _run(self._store.store_embedding(message_id, 'message', embedding))
        except Exception as exc:
            logger.warning('Failed to store embedding for message %s: %s', message_id, exc)

    def get_conversation_history(self, limit: int=100) -> List[Dict[str, Any]]:
        """Auto-generated docstring.

Args:
    limit: Argument.

Returns:
    Return value.
"""
        with self._lock:
            conversation_id = self._current_conversation
        if not conversation_id:
            return []
        return _run(self._store.get_messages(conversation_id, limit))

    def restore_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Auto-generated docstring.

Args:
    conversation_id: Argument.

Returns:
    Return value.
"""
        self.set_conversation(conversation_id)
        return self.get_conversation_history(limit=1000)

    def search_relevant(self, query: str, limit: int=10) -> List[Dict[str, Any]]:
        """Search messages in current conversation by keyword scoring."""
        history = self.get_conversation_history(limit=200)
        terms = [t.lower() for t in query.split() if len(t) >= 3]
        if not terms:
            return []
        scored = []
        for msg in history:
            text = msg.get('content', '').lower()
            score = sum((1 for term in terms if term in text))
            if score:
                scored.append((score, msg))
        return [r for _, r in sorted(scored, key=lambda x: x[0], reverse=True)[:limit]]

    def search_semantic(self, query: str, limit: int=10) -> List[Dict[str, Any]]:
        """Search messages in the current conversation by embedding similarity."""
        if self._embedding_provider is None:
            return []
        try:
            query_embedding = self._embedding_provider.embed(query)
        except Exception as exc:
            logger.warning('Semantic search embedding failed: %s', exc)
            return []
        if not query_embedding:
            return []
        matches = _run(self._store.search_similar(query_embedding, owner_type='message', top_k=limit))
        if not matches:
            return []
        history_by_id = {msg['id']: msg for msg in self.get_conversation_history(limit=1000)}
        results = []
        for match in matches:
            msg = history_by_id.get(match['owner_id'])
            if msg is not None:
                results.append({**msg, 'similarity': match['similarity']})
        return results

    def save_tool_execution(self, tool_name: str, input_data: Any, output_data: Any, context_id: Optional[str]=None) -> str:
        """Auto-generated docstring.

Args:
    tool_name: Argument.
    input_data: Argument.
    output_data: Argument.
    context_id: Argument.

Returns:
    Return value.
"""
        result_id = str(uuid.uuid4())
        status = 'success' if not isinstance(output_data, dict) or output_data.get('success', True) else 'error'
        _run(self._store.log_tool_call(0, tool_name, input_data, output_data, 0, status == 'success'))
        return result_id
