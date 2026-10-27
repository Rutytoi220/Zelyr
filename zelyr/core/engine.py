"""Zelyr Engine - Core orchestration and event handling."""

from typing import Optional, Any, Dict
import logging
import time
from zelyr.core.events import EventBus, Event
from zelyr.core.registry import Registry
from zelyr.core.context import ExecutionContext
from zelyr.core.recorder import FlightRecorder
import os
from zelyr.perception.watcher import ProactiveWatcher, SystemHealthWatchdog
from zelyr.perception.audio_queue import AudioDaemon

logger = logging.getLogger(__name__)


class Engine:
    """Main Zelyr orchestration engine."""
    
    def __init__(self, bus=None, registry=None, memory=None):
        """Auto-generated docstring.

        Args:
            bus: Argument description.
            registry: Argument description.
            memory: Argument description.

        Returns:
            Return description.

        Raises:
            Exception: If something fails.
        """
        self.event_bus = bus or EventBus()
        self.registry = registry or Registry()
        self.context: Optional[ExecutionContext] = None
        self._running = False
        self.started_at = None
        
        self.recorder = FlightRecorder(bus=self.event_bus)
        
        # Setup memory (use MemoryStore if not provided)
        if memory is None:
            from zelyr.memory import SyncMemoryStore

            self.memory = SyncMemoryStore(":memory:")
        else:
            self.memory = memory
            
        # Proactive Kernel
        workspaces = [os.getcwd()]
        self.proactive_watcher = ProactiveWatcher(self.event_bus, workspaces)
        self.os_watcher = SystemHealthWatchdog(self.event_bus)
        
        # Audio Daemon
        self.audio_daemon = AudioDaemon()
        
        self._setup_internal_handlers()
    
    def _setup_internal_handlers(self) -> None:
        """Setup internal event handlers."""
        self._last_activity = time.time()
        self.event_bus.subscribe("*", self._update_activity)
        self.event_bus.subscribe("error.handler", self._handle_error)
        self.event_bus.subscribe("system.shutdown", self._handle_shutdown)
        self.event_bus.subscribe("llm.response.completed", self._handle_llm_response)
        self.event_bus.subscribe("system.anomaly", self._handle_system_anomaly)
        self.event_bus.subscribe("system.idle.rem_sleep", self._handle_rem_sleep)

    def _update_activity(self, event: Event) -> None:
        """Update last activity timestamp."""
        self._last_activity = time.time()
        
    async def _idle_monitor(self) -> None:
        """Monitor for deep idle state."""
        import asyncio
        import time
        in_idle = False
        while self._running:
            now = time.time()
            if now - self._last_activity > 60:
                if not in_idle:
                    in_idle = True
                    logger.info("Deep Idle state entered. Suspending active background tasks.")
                    self.event_bus.publish_sync("system.deep_idle.enter")
            else:
                if in_idle:
                    in_idle = False
                    logger.info("Deep Idle state exited. Restoring active background tasks.")
                    self.event_bus.publish_sync("system.deep_idle.exit")
            await asyncio.sleep(10)

    def _handle_system_anomaly(self, event: Event) -> None:
        """Handle system anomalies by consulting SmartRouter and sending to TTS."""
        anomaly_type = event.data.get("type", "unknown")
        details = event.data.get("details", "")
        
        try:
            from zelyr.llm.universal_client import UniversalLLMClient
            from zelyr.engine.router import SmartRouter, IntentCategory
            
            # Use isolated instance so we don't interfere with active chats
            llm_client = UniversalLLMClient()
            router = SmartRouter(llm_client=llm_client, event_bus=self.event_bus)
            
            prompt = f"System anomaly detected: {details}. Provide a 1-sentence recommended fix. Do not explain, just give the command or action to fix it."
            messages = [{"role": "user", "content": prompt}]
            
            logger.info(f"Routing anomaly '{anomaly_type}' to SmartRouter SYSTEM intent...")
            response = router.chat(messages, model=router.model_tiers[IntentCategory.SYSTEM])
            
            if response:
                logger.warning(f"OSWatcher Fix Recommended: {response}")
                self._log_and_notify("system.anomaly.handled", {"status": "success", "response": response})
            
        except Exception as e:
            logger.error(f"Failed to handle system anomaly: {e}")
            
    def _handle_rem_sleep(self, event: Event) -> None:
        """Trigger Deep Memory Consolidation."""
        if getattr(self.orchestrator, '_power_throttled', False):
            logger.info("DeepMemoryConsolidation paused due to power throttling.")
            return
        try:
            from zelyr.memory.rem_sleep import DeepMemoryConsolidation
            import asyncio
            
            consolidation = DeepMemoryConsolidation(self.event_bus)
            
            graph_nodes = []
            keys = []
            if hasattr(self.memory, 'list_keys') and hasattr(self.memory, 'get'):
                keys = self.memory.list_keys()
                for k in keys:
                    v = self.memory.get(k)
                    if isinstance(v, dict):
                        v['id'] = k
                        graph_nodes.append(v)
            else:
                graph_nodes = [{"id": "mock_1", "text": "Duplicate string A", "created_at": time.time() - 86400 * 40},
                               {"id": "mock_2", "text": "DUPLICATE string a", "created_at": time.time()}]
                
            async def run_rem():
                consolidated = await consolidation.trigger_rem_sleep(graph_nodes)
                if hasattr(self.memory, 'set'):
                    for node in consolidated:
                        self.memory.set(node['id'], node)
                if hasattr(self.memory, 'delete'):
                    surviving_ids = {n['id'] for n in consolidated}
                    for k in keys:
                        if k not in surviving_ids:
                            self.memory.delete(k)
                            
            try:
                loop = asyncio.get_running_loop()
                asyncio.run_coroutine_threadsafe(run_rem(), loop)
            except RuntimeError:
                asyncio.run(run_rem())
        except Exception as e:
            logger.error(f"Failed to run REM Sleep consolidation: {e}")

    def _handle_llm_response(self, event: Event) -> None:
        """Push LLM responses to TTS queue."""
        response_text = event.data.get("response", event.data.get("text", ""))
        if response_text:
            self.audio_daemon.send_tts(response_text)
    
    def _handle_error(self, event: Event) -> None:
        """Handle error events."""
        logger.error(f"Error in handler: {event.data.get('error')}")
    
    def _handle_shutdown(self, event: Event) -> None:
        """Handle shutdown events."""
        self._running = False
        logger.info("Engine shutdown initiated")
    
    def initialize(self) -> None:
        """Initialize the engine."""
        logger.info("Initializing Zelyr Engine")
        self._running = True
        self.started_at = time.time()
        
        self.recorder.start()
        
        # Start proactive watcher (respects config.proactive_kernel internally)
        import multiprocessing
        import platform
        if multiprocessing.current_process().name == 'MainProcess':
            self.proactive_watcher.start()
            
            if platform.system() == "Linux":
                self.os_watcher.start()
            else:
                logger.info("OSWatcher disabled on non-Linux hosts (Graceful Degradation).")
                
            # Start audio daemon
            self.audio_daemon.start()
            
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._idle_monitor())
        except RuntimeError:
            pass
        
        # Log engine started event
        self.memory.log_event("engine.started", data={"started_at": self.started_at}, source="Engine")
        
        # Publish initialization event
        init_event = Event(
            event_type="system.initialized",
            source="Engine",
            data={"status": "ready"}
        )
        self.event_bus.publish(init_event)
    
    def create_context(self, user_input: str = "") -> ExecutionContext:
        """Create a new execution context."""
        self.context = ExecutionContext(user_input=user_input)
        return self.context
    
    def process(self, input_text: str) -> Any:
        """Process input through the engine."""
        if not self._running:
            raise RuntimeError("Engine is not running")
        
        # Create context
        self.context = self.create_context(input_text)
        
        # Publish input event
        input_event = Event(
            event_type="input.received",
            source="Engine",
            data={"input": input_text, "context_id": self.context.context_id}
        )
        self.event_bus.publish(input_event)
        
        return {
            "context_id": self.context.context_id,
            "status": "processing"
        }
    
    def get_current_context(self) -> Optional[ExecutionContext]:
        """Get current execution context."""
        return self.context
    
    def shutdown(self) -> None:
        """Shutdown the engine."""
        # Log engine stopped event
        self.memory.log_event("engine.stopped", data={"stopped_at": time.time()}, source="Engine")
        
        shutdown_event = Event(
            event_type="system.shutdown",
            source="Engine"
        )
        self.event_bus.publish(shutdown_event)
        
        self.os_watcher.stop()
        self.recorder.stop()
    
    def is_running(self) -> bool:
        """Check if engine is running."""
        return self._running

    def status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            "running": self._running,
            "started_at": self.started_at,
            "event_count": len(self.memory.get_events())
        }

