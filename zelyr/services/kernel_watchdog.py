import asyncio
import json
import logging
import subprocess
from collections import deque
from typing import Optional
from zelyr.core.events import EventBus

logger = logging.getLogger(__name__)

class KernelWatchdogService:
    """Streams journalctl to detect critical OS incidents in real-time."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._running = False
        self._log_buffer = deque(maxlen=50)
        self._process: Optional[asyncio.subprocess.Process] = None
        
    async def start(self):
        if self._running:
            return
        self._running = True
        logger.info("Starting Kernel Watchdog Service...")
        asyncio.create_task(self._watch_loop())
        
    async def stop(self):
        self._running = False
        if self._process:
            self._process.terminate()
            
    async def _watch_loop(self):
        """Asynchronously stream journald logs."""
        try:
            # --priority=0..3 captures emerg, alert, crit, err
            cmd = ["journalctl", "-f", "-o", "json", "--priority=0..3"]
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            while self._running and self._process.stdout:
                line = await self._process.stdout.readline()
                if not line:
                    break
                    
                try:
                    entry = json.loads(line.decode('utf-8'))
                    self._log_buffer.append(entry)
                    self._analyze_entry(entry)
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            logger.error(f"KernelWatchdog failed: {e}")
            self._running = False
            
    def _analyze_entry(self, entry: dict):
        message = entry.get("MESSAGE", "")
        unit = entry.get("_SYSTEMD_UNIT", "")
        
        is_critical = False
        reason = ""
        
        # 1. Systemd unit failures
        if unit and ("failed" in message.lower() or "crashed" in message.lower()):
            is_critical = True
            reason = f"Service failure detected in {unit}"
            
        # 2. OOM killer
        elif "out of memory" in message.lower() or "oom-killer" in message.lower():
            is_critical = True
            reason = "OOM Killer invoked"
            
        # 3. Segfaults / kernel panics
        elif "segfault" in message.lower() or "kernel panic" in message.lower():
            is_critical = True
            reason = "Kernel Panic / Segfault detected"
            
        if is_critical:
            logger.warning(f"Watchdog detected anomaly: {reason}. Message: {message}")
            
            # Emit incident event to trigger self-healing
            self.event_bus.publish_sync(
                "os.incident.detected",
                data={
                    "reason": reason,
                    "message": message,
                    "unit": unit,
                    "context": list(self._log_buffer)
                }
            )
