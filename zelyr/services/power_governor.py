"""Power-Aware Cognitive Throttling.

Polls Linux battery states via psutil. Emits EventBus signals to autonomously
downshift Zelyr's cognitive model (e.g. from an 8B parameter model to a 1.5B model)
when unplugged and battery is critical (<30%). Restores max performance on AC power.
"""
import logging
import asyncio
from typing import Optional
from zelyr.core.events import EventBus

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

class PowerStateService:
    """Manages cognitive throttling based on hardware power states."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._running = False
        self._loop = None
        self._current_state = "normal"  # "normal" or "eco"
        
    def start(self):
        if not PSUTIL_AVAILABLE:
            logger.warning("PowerStateService: 'psutil' not available. Cognitive Throttling disabled.")
            return
            
        self._running = True
        self._loop = asyncio.get_event_loop()
        self._loop.create_task(self._poll_power())
        logger.info("PowerStateService: Monitoring hardware power draw.")
        
    def stop(self):
        self._running = False
        
    async def _poll_power(self):
        while self._running:
            try:
                battery = psutil.sensors_battery()
                if battery:
                    self._evaluate_state(battery.percent, battery.power_plugged)
                await asyncio.sleep(10) # Poll every 10 seconds
            except Exception as e:
                logger.error(f"PowerStateService: Error reading battery sensor - {e}")
                await asyncio.sleep(60)
                
    def _evaluate_state(self, percent: float, plugged: bool):
        if not plugged and percent < 30.0:
            if self._current_state != "eco":
                self._current_state = "eco"
                logger.warning(f"PowerStateService: Battery critical ({percent}%). Downshifting to ECO Mode.")
                self.event_bus.publish_sync("power.state.critical", {
                    "percent": percent,
                    "action": "downshift",
                    "target_model": "llama3.2:1b"
                })
        elif plugged or percent > 50.0:
            if self._current_state != "normal":
                self._current_state = "normal"
                logger.info(f"PowerStateService: Power sufficient ({percent}%). Restoring MAX PERFORMANCE.")
                self.event_bus.publish_sync("power.state.normal", {
                    "percent": percent,
                    "action": "upshift",
                    "target_model": "default"
                })
                
    def mock_battery_event(self, percent: float, plugged: bool):
        """Allows testing without physical hardware triggers."""
        logger.debug(f"PowerStateService: Simulating battery event (Percent={percent}, Plugged={plugged})")
        self._evaluate_state(percent, plugged)
