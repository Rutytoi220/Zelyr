"""Zelyr Engine - Simple synchronous orchestration."""

import time
from typing import Optional
from zelyr.events import EventBus
from zelyr.registry import Registry


class Engine:
    """Simple synchronous Zelyr engine with event bus and registry."""
    
    def __init__(self, bus=None, registry=None, memory=None):
        """Initialize the engine.
        
        Args:
            bus: Optional EventBus instance. Creates new one if not provided.
            registry: Optional Registry instance. Creates new one if not provided.
            memory: Optional MemoryStore instance. Creates new one if not provided.
        """
        self.running = False
        self.started_at = None
        self.bus = bus if bus is not None else EventBus()
        self.registry = registry if registry is not None else Registry()
        from zelyr.memory import SyncMemoryStore

        self.memory = memory if memory is not None else SyncMemoryStore(":memory:")
    
    def start(self) -> None:
        """Start the engine.
        
        Sets running flag and publishes engine.started event.
        """
        self.running = True
        self.started_at = time.time()
        self.bus.publish_sync("engine.started", {"started_at": self.started_at})
        self.memory.log_event("engine.started", {"started_at": self.started_at})
    
    def stop(self) -> None:
        """Stop the engine.
        
        Clears running flag and publishes engine.stopped event.
        """
        self.running = False
        stopped_at = time.time()
        self.bus.publish_sync("engine.stopped", {"stopped_at": stopped_at})
        self.memory.log_event("engine.stopped", {"stopped_at": stopped_at})
    
    def status(self) -> dict:
        """Get engine status.
        
        Returns:
            Dictionary with engine status and metrics
        """
        return {
            "running": self.running,
            "started_at": self.started_at,
            "event_count": len(self.memory.get_events())
        }
