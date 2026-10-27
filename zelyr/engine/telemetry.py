"""Hardware Telemetry Daemon for Zelyr v2.

Monitors system resources (RAM, CPU) and Ollama VRAM usage via its `/api/ps`
endpoint. Emits a `hardware.resource_warning` event onto the EventBus if
available memory drops below a critical threshold (default 15%).
"""
import asyncio
import logging
import json
import urllib.request
import urllib.error
import psutil
from typing import Dict, Any, Optional
from zelyr.core.events import EventBus, Event
logger = logging.getLogger(__name__)

class HardwareTelemetryDaemon:
    """Background daemon that monitors hardware load and emits warnings."""

    def __init__(self, event_bus: EventBus, ollama_url: str='http://127.0.0.1:11434'):
        """Auto-generated docstring.

Args:
    event_bus: Argument.
    ollama_url: Argument.

Returns:
    Return value.
"""
        self.event_bus = event_bus
        self.ollama_url = ollama_url
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._memory_threshold_percent = 15.0
        self._poll_interval_seconds = 3.0
        self.latest_state: Dict[str, Any] = {'ram_available_percent': 100.0, 'ollama_vram_bytes': 0, 'warning': False}

    def start(self) -> None:
        """Start the telemetry daemon."""
        if self._running:
            return
        self._running = True
        try:
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self._monitor_loop())
            logger.info('HardwareTelemetryDaemon started.')
        except RuntimeError:
            logger.debug('No running asyncio loop. Telemetry daemon will not start in synchronous environments.')

    def stop(self) -> None:
        """Stop the telemetry daemon."""
        self._running = False
        if self._task and (not self._task.done()):
            self._task.cancel()
        logger.info('HardwareTelemetryDaemon stopped.')

    async def _monitor_loop(self) -> None:
        """Main monitoring loop running in the background."""
        while self._running:
            try:
                mem = psutil.virtual_memory()
                ram_available_percent = mem.available / mem.total * 100.0
                ollama_vram = await self._get_ollama_vram_usage()
                is_warning = ram_available_percent < self._memory_threshold_percent
                self.latest_state = {'ram_available_percent': ram_available_percent, 'ollama_vram_bytes': ollama_vram, 'warning': is_warning}
                if is_warning:
                    logger.warning(f'Hardware Resource Warning: Available RAM at {ram_available_percent:.1f}%')
                    self.event_bus.publish(Event(event_type='hardware.resource_warning', source='HardwareTelemetryDaemon', data=self.latest_state.copy()))
            except Exception as e:
                logger.debug(f'Telemetry polling error: {e}')
            await asyncio.sleep(self._poll_interval_seconds)

    async def _get_ollama_vram_usage(self) -> int:
        """Fetch VRAM footprint from Ollama's /api/ps endpoint."""

        def fetch():
            """Auto-generated docstring.


Returns:
    Return value.
"""
            try:
                req = urllib.request.Request(f"{self.ollama_url.rstrip('/')}/api/ps")
                with urllib.request.urlopen(req, timeout=1.0) as resp:
                    data = json.loads(resp.read().decode())
                    total_vram = 0
                    for model in data.get('models', []):
                        total_vram += model.get('size_vram', 0)
                    return total_vram
            except Exception:
                return 0
        return await asyncio.to_thread(fetch)

import structlog
import uuid
from contextvars import ContextVar

# Context variable to hold the current correlation ID
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")

def setup_structured_logging() -> None:
    """Configure structlog for JSON output and context variables."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

class StructuredTracer:
    """Structured OpenTelemetry-style tracer using structlog.
    
    Assigns correlation UUIDs to trace lifecycles across the daemon.
    """
    
    def __init__(self, logger_name: str = "zelyr.tracer"):
        self.logger = structlog.get_logger(logger_name)
        
    def start_trace(self, operation: str, **kwargs) -> str:
        """Begin a new trace with a unique correlation ID."""
        trace_id = str(uuid.uuid4())
        _correlation_id.set(trace_id)
        structlog.contextvars.bind_contextvars(correlation_id=trace_id)
        self.logger.info(f"Trace started: {operation}", operation=operation, **kwargs)
        return trace_id
        
    def log_step(self, step_name: str, **kwargs):
        """Log a step in the current trace."""
        self.logger.info(step_name, **kwargs)
        
    def end_trace(self, operation: str, **kwargs):
        """End the current trace."""
        self.logger.info(f"Trace ended: {operation}", operation=operation, **kwargs)
        structlog.contextvars.clear_contextvars()
