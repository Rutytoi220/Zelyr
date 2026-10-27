import asyncio
import logging
import subprocess
import time
import urllib.request
import urllib.error
from PySide6.QtCore import QObject, Signal, QThread

logger = logging.getLogger(__name__)

class OllamaHealthMonitor(QObject):
    """Background HTTP health monitor for local Ollama service."""
    
    # Emitted when status changes: (is_online, latency_ms)
    status_changed = Signal(bool, float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._is_online = False
        self._polling_interval = 2.0
        self._task = None
        self._rapid_polling = False
        self._first_poll = True
        
    def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        
    def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

    def trigger_rapid_polling(self):
        self._rapid_polling = True
            
    async def _poll_loop(self):
        while self._running:
            start_time = time.perf_counter()
            online = await asyncio.to_thread(self._ping)
            latency = (time.perf_counter() - start_time) * 1000
            
            # If state changed, or we just recovered during rapid polling, or it's the first poll
            if self._first_poll or online != self._is_online or (online and self._rapid_polling):
                self._is_online = online
                self._first_poll = False
                self.status_changed.emit(online, latency)
                if online:
                    self._rapid_polling = False
                    
            sleep_time = 0.5 if self._rapid_polling else self._polling_interval
            await asyncio.sleep(sleep_time)

    def _ping(self) -> bool:
        try:
            req = urllib.request.Request("http://127.0.0.1:11434/api/version", method="GET")
            with urllib.request.urlopen(req, timeout=1.0) as response:
                return response.status == 200
        except Exception:
            return False

    @staticmethod
    def spawn_ollama_service() -> bool:
        """Spawn the Ollama daemon non-blockingly."""
        import os
        import platform
        
        # 1. Try systemctl user service first (common on Bazzite/Linux)
        try:
            res = subprocess.run(["systemctl", "--user", "is-enabled", "ollama"], capture_output=True, text=True)
            if "enabled" in res.stdout or "disabled" in res.stdout:
                # Service exists, try to start it
                subprocess.run(["systemctl", "--user", "start", "ollama"], check=False)
                return True
        except FileNotFoundError:
            pass
            
        # 2. Try raw subprocess spawn
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return True
        except Exception as e:
            logger.error(f"Failed to spawn ollama serve: {e}")
            return False
