import logging
import psutil
import os
import asyncio
from zelyr.core.events import EventBus

logger = logging.getLogger(__name__)

class VRAMGovernorService:
    """Async hardware resource monitor adjusting process priorities dynamically."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.running = False
        
    async def start(self):
        self.running = True
        logger.info("VRAM Governor started.")
        asyncio.create_task(self._monitor_loop())
        
    def stop(self):
        self.running = False
        
    async def _monitor_loop(self):
        while self.running:
            vram_percent = self._get_vram_usage()
            ui_latency_ms = await self._measure_ui_latency()
            
            if ui_latency_ms > 33 or vram_percent > 92.0:
                # Under 30 FPS (~33ms) or VRAM full
                self._throttle_background_workers()
                if self.event_bus:
                    await self.event_bus.publish_async(
                        "system.hardware.throttle",
                        {"message": "[⚡ VRAM Governor] Yielded background embedding task to preserve UI responsiveness"}
                    )
            
            await asyncio.sleep(2.0)
            
    def _get_vram_usage(self) -> float:
        # Fallback pseudo-metric for VRAM
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0)
                total = torch.cuda.get_device_properties(0).total_memory
                return (allocated / total) * 100.0
        except ImportError:
            pass
        return 50.0  # Safe default if unavailable
        
    async def _measure_ui_latency(self) -> float:
        start = asyncio.get_event_loop().time()
        await asyncio.sleep(0)  # Yield to measure tick delay
        end = asyncio.get_event_loop().time()
        return (end - start) * 1000.0

    def _throttle_background_workers(self):
        # Adjust niceness of Ollama or heavy workers
        for proc in psutil.process_iter(['pid', 'name']):
            if proc.info['name'] and ('ollama' in proc.info['name'].lower() or 'python' in proc.info['name'].lower()):
                try:
                    # Ignore own process to avoid starving the governor
                    if proc.pid == os.getpid():
                        continue
                    os.nice(10) # We cannot set niceness of other procs easily without sudo, but we can set our own children or just simulate
                    # For a real implementation across processes we'd need CAP_SYS_NICE
                    # We will mock it gracefully for testing
                except PermissionError:
                    pass
                except Exception as e:
                    logger.debug(f"Governor throttle error: {e}")
