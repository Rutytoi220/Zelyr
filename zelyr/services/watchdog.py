import logging
import asyncio
from zelyr.engine.repair_loop import trigger_repair

logger = logging.getLogger(__name__)

class PluginCrashHandler(logging.Handler):
    def __init__(self, registry, event_bus, loop):
        super().__init__()
        self.registry = registry
        self.event_bus = event_bus
        self.loop = loop

    def emit(self, record):
        traceback_str = self.format(record)
        logger.info("[Watchdog] Intercepted plugin crash. Triggering auto-heal...")
        asyncio.run_coroutine_threadsafe(
            trigger_repair(traceback_str, self.registry, self.event_bus), 
            self.loop
        )

class WatchdogService:
    def __init__(self, registry, event_bus):
        self.registry = registry
        self.event_bus = event_bus
        self.handler = None
        self._loop = None

    def start(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self.handler = PluginCrashHandler(self.registry, self.event_bus, loop)
        crash_logger = logging.getLogger("zelyr.plugin_crash")
        crash_logger.setLevel(logging.ERROR)
        crash_logger.addHandler(self.handler)
        logger.info("WatchdogService started listening for plugin crashes.")

    def stop(self):
        if self.handler:
            crash_logger = logging.getLogger("zelyr.plugin_crash")
            crash_logger.removeHandler(self.handler)
            self.handler = None
        logger.info("WatchdogService stopped.")
