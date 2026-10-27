"""Event system for Zelyr - core pub/sub mechanism.

Synchronous, thread-friendly event bus with fnmatch wildcard pattern
matching and meta-event emission.  This is the single canonical EventBus
for the Zelyr framework.
"""
from typing import Any, Callable, Dict, List, Set
from dataclasses import dataclass, field
from datetime import datetime
from fnmatch import fnmatch
import logging
import uuid
import time
logger = logging.getLogger(__name__)

@dataclass
class Event:
    """Base event class for all system events."""
    event_type: str
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ttl: int = 5
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {'event_type': self.event_type, 'source': self.source, 'timestamp': self.timestamp.isoformat(), 'event_id': self.event_id, 'correlation_id': self.correlation_id, 'ttl': self.ttl, 'data': self.data, 'metadata': self.metadata}

class EventBus:
    """Central event pub/sub system.

    Supports exact event type matching and fnmatch wildcard patterns
    (e.g. ``agent.*``, ``*.started``, ``*``).  A ``bus.published``
    meta-event is emitted after every publish so observability tools can
    monitor the bus without intercepting every event type.
    """

    def __init__(self) -> None:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        self._subscribers: Dict[str, List[Callable]] = {}
        self._event_history: List[Event] = []
        self._max_history: int = 1000
        self._published_events: Set[str] = set()
        self._in_meta_event: bool = False
        self._debounce_cache: Dict[str, float] = {}
        self._debounce_ttl: float = 2.0

    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to an event type or pattern."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """Unsubscribe from an event type."""
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
            except ValueError:
                pass
            if not self._subscribers[event_type]:
                del self._subscribers[event_type]

    def publish(self, event: Event) -> None:
        """Publish an event to all matching subscribers.

        Exact matches and fnmatch wildcard patterns (``agent.*``,
        ``*.error``, ``*``) are both evaluated.  A ``bus.published``
        meta-event is emitted after the primary handlers complete.
        """
        if event.ttl <= 0:
            logger.warning(f"[EventBus] Dropped packet {event.event_type} to prevent Event Storm loop (ttl <= 0).")
            return
            
        # Debouncing
        event_hash = f"{event.event_type}:{event.source}:{hash(str(event.data))}"
        now = time.time()
        last_seen = self._debounce_cache.get(event_hash, 0)
        
        if now - last_seen < self._debounce_ttl:
            logger.debug(f"[EventBus] Debounced duplicate event: {event.event_type}")
            return
            
        self._debounce_cache[event_hash] = now
        
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        self._published_events.add(event.event_type)
        handlers = self._matching_handlers(event.event_type)
        
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        for handler in handlers:
            try:
                if loop:
                    if asyncio.iscoroutinefunction(handler):
                        loop.create_task(handler(event))
                    else:
                        loop.run_in_executor(None, handler, event)
                else:
                    handler(event)
            except Exception as exc:
                logger.error('Handler %s raised for event %s: %s', getattr(handler, '__name__', handler), event.event_type, exc, exc_info=True)
        self._emit_meta_event(event.event_type, event.data)

    def publish_sync(self, event_name: str, data: Any=None) -> None:
        """Convenience publisher for string-based event names.

        Creates an :class:`Event` from *event_name* and *data*, then
        publishes it.  This is the entry point used by
        ``SimpleBaseAgent._emit()`` and other sync callers.
        """
        payload = data if isinstance(data, dict) else {'data': data}
        event = Event(event_type=event_name, source='agent', data=payload)
        self.publish(event)

    def get_history(self, event_type: str | None = None, limit: int=100) -> List[Event]:
        """Return recent events, optionally filtered by type."""
        if event_type:
            return [e for e in self._event_history if e.event_type == event_type][-limit:]
        return self._event_history[-limit:]

    def get_subscriptions(self) -> Dict[str, int]:
        """Return subscription counts keyed by pattern."""
        return {pattern: len(handlers) for pattern, handlers in self._subscribers.items()}

    def get_published_events(self) -> Set[str]:
        """Return the set of distinct event types that have been published."""
        return self._published_events.copy()

    def subscribers(self, event_type: str) -> List[Callable]:
        """Return a snapshot of handlers subscribed to *event_type*."""
        return list(self._subscribers.get(event_type, []))

    def clear(self) -> None:
        """Remove all subscriptions, published-event tracking, and history."""
        self._subscribers.clear()
        self._published_events.clear()
        self._event_history.clear()

    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()

    def _matching_handlers(self, event_type: str) -> List[Callable]:
        """Return all handlers whose subscription pattern matches *event_type*."""
        matching: List[Callable] = []
        for pattern, handlers in self._subscribers.items():
            if pattern == event_type:
                matching.extend(handlers)
            elif '*' in pattern and fnmatch(event_type, pattern):
                matching.extend(handlers)
        return matching

    def _emit_meta_event(self, event_type: str, data: Any) -> None:
        """Emit a ``bus.published`` meta-event (re-entrancy guarded)."""
        if self._in_meta_event:
            return
        meta_handlers = self._matching_handlers('bus.published')
        if not meta_handlers:
            return
        meta_event = Event(event_type='bus.published', source='EventBus', data={'event': event_type, 'original_payload': data, 'total_published': len(self._published_events)})
        self._in_meta_event = True
        try:
            for handler in meta_handlers:
                try:
                    handler(meta_event)
                except Exception as exc:
                    logger.error('Meta-event handler error: %s', exc, exc_info=True)
        finally:
            self._in_meta_event = False
