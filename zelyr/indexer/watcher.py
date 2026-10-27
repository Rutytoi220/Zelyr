"""EventBus watcher to trigger AST Graph Index incremental updates."""
import logging
from typing import Any, Dict
from zelyr.core.events import EventBus, Event
from zelyr.indexer.graph_engine import CodeGraphIndex
logger = logging.getLogger(__name__)

class GraphWatcher:
    """Subscribes to EventBus to incrementally update the CodeGraphIndex."""

    def __init__(self, bus: EventBus, index: CodeGraphIndex):
        """Auto-generated docstring.

Args:
    bus: Argument.
    index: Argument.

Returns:
    Return value.
"""
        self._bus = bus
        self._index = index
        self._bus.subscribe('transaction.committed', self._on_transaction_end)
        self._bus.subscribe('transaction.rolled_back', self._on_transaction_end)
        logger.debug('GraphWatcher attached to EventBus.')

    def _on_transaction_end(self, event: Event) -> None:
        """Handle both commit and rollback events to update modified files."""
        files_modified = event.data.get('files_modified', [])
        if not files_modified:
            return
        updated_count = 0
        for filepath in files_modified:
            if str(filepath).endswith('.py'):
                self._index.update_file(filepath)
                updated_count += 1
        if updated_count > 0:
            logger.debug(f'CodeGraphIndex incrementally updated {updated_count} files after {event.event_type}')
