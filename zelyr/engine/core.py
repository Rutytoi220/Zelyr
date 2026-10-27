"""Core engine for Zelyr orchestration framework."""

import asyncio
import signal
import logging
import inspect
from typing import Dict, Any, Optional, Callable, List

logger = logging.getLogger(__name__)


class CoreEngine:
    """
    Core orchestration engine for Zelyr.
    
    Manages the main event loop, execution context, lifecycle events,
    and graceful signal handling for SIGINT/SIGTERM.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the CoreEngine.
        
        Args:
            config: Optional configuration dictionary for engine settings.
                   Defaults to empty dict if not provided.
        
        No side effects on import - engine must be explicitly started.
        """
        self.config = config or {}
        self._context: Dict[str, Any] = {}
        self._running = False
        self._event_listeners: Dict[str, List[Callable]] = {}
        self._signal_handlers: Dict[int, Any] = {}
    
    @property
    def context(self) -> Dict[str, Any]:
        """
        Get the execution context dictionary.
        
        This dict is accessible to all modules and shared across the engine.
        Modifications persist across calls.
        
        Returns:
            Dictionary for storing execution context data.
        """
        return self._context
    
    async def start(self) -> None:
        """
        Start the engine.
        
        Sets up signal handlers and emits the "engine.started" event.
        Can be called only once - subsequent calls are safe but logged as warnings.
        
        Raises:
            Any exceptions from event subscribers are logged but not propagated.
        """
        if self._running:
            logger.warning("Engine is already running")
            return
        
        self._running = True
        self._setup_signal_handlers()
        await self._emit_event("engine.started")
        logger.info("Engine started successfully")
    
    async def stop(self) -> None:
        """
        Stop the engine gracefully.
        
        Emits the "engine.stopped" event and cleans up signal handlers.
        Can be called safely even if engine is not running.
        """
        if not self._running:
            logger.warning("Engine is not running")
            return
        
        self._running = False
        await self._emit_event("engine.stopped")
        self._cleanup_signal_handlers()
        logger.info("Engine stopped")
    
    async def shutdown(self) -> None:
        """
        Shutdown the engine and cleanup all resources.
        
        Calls stop() to perform graceful shutdown.
        """
        await self.stop()
    
    def subscribe(self, event: str, callback: Callable) -> None:
        """
        Subscribe to an engine event.
        
        Args:
            event: Event name (e.g., "engine.started", "engine.stopped")
            callback: Callable to invoke when event is emitted.
                     Can be async or sync function.
        
        Multiple subscribers can listen to the same event.
        """
        if event not in self._event_listeners:
            self._event_listeners[event] = []
        self._event_listeners[event].append(callback)
    
    def _setup_signal_handlers(self) -> None:
        """
        Setup signal handlers for graceful shutdown.
        
        Registers handlers for SIGINT and SIGTERM to trigger
        graceful shutdown. Safe on platforms that don't support
        loop.add_signal_handler (will be skipped silently).
        """
        try:
            loop = asyncio.get_event_loop()
            
            def signal_handler(signum: int) -> None:
                """Handle signals by triggering graceful shutdown."""
                logger.info(f"Received signal {signum}, initiating shutdown...")
                if self._running:
                    # Schedule stop as a task to allow current operations to finish
                    asyncio.create_task(self.stop())
            
            for sig in (signal.SIGINT, signal.SIGTERM):
                handler = loop.add_signal_handler(sig, signal_handler, sig)
                self._signal_handlers[sig] = handler
            
            logger.debug("Signal handlers setup complete")
        except (NotImplementedError, RuntimeError) as e:
            # Signal handling not available in current context
            # (e.g., Windows, thread-based event loop, no running loop)
            logger.debug(f"Signal handler setup not available: {e}")
    
    def _cleanup_signal_handlers(self) -> None:
        """
        Remove and cleanup all registered signal handlers.
        
        Safe to call even if handlers were never setup.
        """
        try:
            loop = asyncio.get_event_loop()
            for sig in list(self._signal_handlers.keys()):
                loop.remove_signal_handler(sig)
                del self._signal_handlers[sig]
            logger.debug("Signal handlers cleaned up")
        except (NotImplementedError, RuntimeError) as e:
            logger.debug(f"Signal handler cleanup not available: {e}")
    
    async def _emit_event(self, event: str) -> None:
        """
        Emit an event to all subscribers.
        
        Args:
            event: Event name to emit
        
        Handles both sync and async callbacks.
        Exceptions in callbacks are logged but don't propagate.
        """
        if event not in self._event_listeners:
            return
        
        for callback in self._event_listeners[event]:
            try:
                if inspect.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Error in event handler for '{event}': {e}", exc_info=True)
