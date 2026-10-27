"""Protocol defining the memory backend interface for Zelyr."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class MemoryBackend(ABC):
    """Abstract interface for pluggable memory backends.

    Implementations must be async. Sync consumers can wrap with asyncio.run().
    To swap backends, implement this protocol and pass to agents/engine.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Set up connection and schema. Must be called before any other method."""

    @abstractmethod
    async def close(self) -> None:
        """Release resources."""

    # -- Key-value store --

    @abstractmethod
    async def set(
        self, key: str, value: Any, tags: Optional[List[str]] = None, ttl: Optional[float] = None
    ) -> None:
        """Store a value with optional tags and time-to-live in seconds."""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Retrieve a value by key. Returns None if missing or expired."""

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a key. Returns True if it existed."""

    @abstractmethod
    async def search_by_tags(self, tags: List[str]) -> List[Dict[str, Any]]:
        """Find all entries matching ALL given tags (AND logic)."""

    @abstractmethod
    async def expire_ttl(self) -> int:
        """Remove expired entries. Returns count removed."""

    # -- Conversations --

    @abstractmethod
    async def create_conversation(self, title: str = "") -> str:
        """Create a new conversation. Returns its ID."""

    @abstractmethod
    async def add_message(
        self, conversation_id: str, role: str, content: str, metadata: Optional[Dict] = None
    ) -> str:
        """Append a message to a conversation. Returns the message ID."""

    @abstractmethod
    async def get_messages(self, conversation_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get messages for a conversation in chronological order."""

    @abstractmethod
    async def list_conversations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent conversations with metadata."""

    # -- Summaries --

    @abstractmethod
    async def save_summary(
        self, conversation_id: str, summary: str, msg_start: int, msg_end: int
    ) -> None:
        """Save a conversation summary covering a range of messages."""

    @abstractmethod
    async def get_summaries(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all summaries for a conversation."""

    # -- Semantic search --

    @abstractmethod
    async def store_embedding(
        self, owner_id: str, owner_type: str, embedding: List[float], model: str = ""
    ) -> None:
        """Store an embedding vector linked to a content object."""

    @abstractmethod
    async def search_similar(
        self, embedding: List[float], owner_type: str = "", top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find most similar entries by cosine similarity."""

    # -- Events --

    @abstractmethod
    async def log_event(
        self, event_name: str, payload: Any = None, source: Optional[str] = None
    ) -> int:
        """Log an event. Returns the event ID."""

    @abstractmethod
    async def get_events(
        self,
        event_name: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve events with optional filters."""

    # -- Agent sessions --

    @abstractmethod
    async def create_agent_session(self, agent_name: str, task: str) -> int:
        """Start tracking an agent session. Returns session ID."""

    @abstractmethod
    async def log_tool_call(
        self,
        session_id: int,
        tool_name: str,
        params: Any,
        result: Any,
        duration_ms: int,
        success: bool,
        error_message: Optional[str] = None,
    ) -> int:
        """Record a tool call within a session."""

    @abstractmethod
    async def get_session_tool_calls(self, session_id: int) -> List[Dict[str, Any]]:
        """Get all tool calls for a session."""

    @abstractmethod
    async def complete_agent_session(
        self, session_id: int, result: Any, success: bool = True
    ) -> None:
        """Mark a session as completed."""
