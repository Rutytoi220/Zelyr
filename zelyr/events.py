"""Backward-compatible re-export of the canonical EventBus.

``zelyr.core.events`` is the single source of truth.  This module exists
so that existing ``from zelyr.events import EventBus, Event`` imports
continue to work without changes.
"""

from zelyr.core.events import EventBus, Event

__all__ = ["EventBus", "Event"]
