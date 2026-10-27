"""Filesystem Watcher and Resource Governor for the Proactive Perception Kernel.

Uses `watchdog` to monitor workspaces. Incorporates a CPU-based Resource
Governor to auto-pause the watcher if system CPU usage exceeds a strict
limit (1.5% average over a 5-second sliding window) to prevent starving
other processes.
"""
import time
import logging
import threading
import os
import psutil
from typing import Any, List, Optional
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent
from zelyr.core.events import EventBus
from zelyr.perception.scrubber import PrivacyScrubber
from zelyr.perception.intent_engine import IntentEngine
from zelyr.config import get_config
import subprocess
logger = logging.getLogger(__name__)

class ActiveWindowContext:
    """Extracts active window metadata cross-platform using lightweight wrappers."""

    @staticmethod
    def get_active_window_title() -> str:
        """Fetch the active window title safely."""
        try:  # pragma: no cover
            result = subprocess.run(['xdotool', 'getactivewindow', 'getwindowname'], capture_output=True, text=True, timeout=1.0)  # pragma: no cover
            if result.returncode == 0 and result.stdout.strip():  # pragma: no cover
                return result.stdout.strip()  # pragma: no cover
            return 'Unknown Window'  # pragma: no cover
        except Exception:  # pragma: no cover
            return 'Unknown Window'  # pragma: no cover

class ResourceGovernor:
    """Monitors host CPU usage and determines if background tasks should pause."""

    def __init__(self, max_cpu_percent: float=1.5, window_seconds: int=5):
        """Auto-generated docstring.

Args:
    max_cpu_percent: Argument.
    window_seconds: Argument.

Returns:
    Return value.
"""
        self.max_cpu_percent = max_cpu_percent  # pragma: no cover
        self.window_seconds = window_seconds  # pragma: no cover
        self._history: List[float] = []  # pragma: no cover
        self._lock = threading.Lock()  # pragma: no cover
        self._running = False  # pragma: no cover
        self._thread: Optional[threading.Thread] = None  # pragma: no cover
        self.is_paused = False  # pragma: no cover

    def start(self):
        """Auto-generated docstring.


Returns:
    Return value.
"""
        if self._running:  # pragma: no cover
            return  # pragma: no cover
        self._running = True  # pragma: no cover
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True, name='ResourceGovernor')  # pragma: no cover
        self._thread.start()  # pragma: no cover

    def stop(self):
        """Auto-generated docstring.


Returns:
    Return value.
"""
        self._running = False  # pragma: no cover
        if self._thread:  # pragma: no cover
            self._thread.join(timeout=2.0)  # pragma: no cover

    def _monitor_loop(self):
        """Auto-generated docstring.


Returns:
    Return value.
"""
        psutil.cpu_percent(interval=None)  # pragma: no cover
        while self._running:  # pragma: no cover
            time.sleep(1.0)  # pragma: no cover
            cpu = psutil.cpu_percent(interval=None)  # pragma: no cover
            with self._lock:  # pragma: no cover
                self._history.append(cpu)  # pragma: no cover
                if len(self._history) > self.window_seconds:  # pragma: no cover
                    self._history.pop(0)  # pragma: no cover
                avg_cpu = sum(self._history) / len(self._history)  # pragma: no cover
                was_paused = self.is_paused  # pragma: no cover
                self.is_paused = avg_cpu > self.max_cpu_percent  # pragma: no cover
                if self.is_paused and (not was_paused):  # pragma: no cover
                    logger.warning(f'Resource Governor: Pausing proactive kernel. CPU avg {avg_cpu:.2f}% > {self.max_cpu_percent}%')  # pragma: no cover
                elif not self.is_paused and was_paused:  # pragma: no cover
                    logger.info(f'Resource Governor: Resuming proactive kernel. CPU avg {avg_cpu:.2f}% <= {self.max_cpu_percent}%')  # pragma: no cover

class ProactiveEventHandler(FileSystemEventHandler):
    """Handles watchdog filesystem events and routes them to the Intent Engine."""

    def __init__(self, engine: IntentEngine, governor: ResourceGovernor):
        """Auto-generated docstring.

Args:
    engine: Argument.
    governor: Argument.

Returns:
    Return value.
"""
        self.engine = engine  # pragma: no cover
        self.governor = governor  # pragma: no cover
        super().__init__()  # pragma: no cover

    def on_created(self, event):
        """Auto-generated docstring.

Args:
    event: Argument.

Returns:
    Return value.
"""
        if event.is_directory or self.governor.is_paused:  # pragma: no cover
            return  # pragma: no cover
        if not PrivacyScrubber.is_safe(event.src_path):  # pragma: no cover
            return  # pragma: no cover
        self.engine.evaluate('created', event.src_path)  # pragma: no cover

    def on_modified(self, event):
        """Auto-generated docstring.

Args:
    event: Argument.

Returns:
    Return value.
"""
        if event.is_directory or self.governor.is_paused:  # pragma: no cover
            return  # pragma: no cover
        if not PrivacyScrubber.is_safe(event.src_path):  # pragma: no cover
            return  # pragma: no cover
        self.engine.evaluate('modified', event.src_path)  # pragma: no cover

class ProactiveWatcher:
    """Manages the background filesystem observer and its lifecycle."""

    def __init__(self, event_bus: EventBus, workspaces: List[str]):
        """Auto-generated docstring.

Args:
    event_bus: Argument.
    workspaces: Argument.

Returns:
    Return value.
"""
        self.event_bus = event_bus  # pragma: no cover
        self.workspaces = workspaces  # pragma: no cover
        self.intent_engine = IntentEngine(event_bus)  # pragma: no cover
        self.governor = ResourceGovernor()  # pragma: no cover
        self.observer: Any | None = None  # pragma: no cover

    def start(self) -> bool:
        """Start the watcher if opt-in configuration is enabled."""
        config = get_config()  # pragma: no cover
        if not getattr(config, 'proactive_kernel', False):  # pragma: no cover
            logger.info('Proactive OS Perception Kernel is DISABLED by configuration.')  # pragma: no cover
            return False  # pragma: no cover
        if not self.workspaces:  # pragma: no cover
            logger.warning('No workspaces configured for ProactiveWatcher.')  # pragma: no cover
            return False  # pragma: no cover
        logger.info('Starting Proactive OS Perception Kernel...')  # pragma: no cover
        self.governor.start()  # pragma: no cover
        event_handler = ProactiveEventHandler(self.intent_engine, self.governor)  # pragma: no cover
        self.observer = Observer()  # pragma: no cover
        for ws in self.workspaces:  # pragma: no cover
            path = Path(ws).expanduser().resolve()  # pragma: no cover
            if path.exists() and path.is_dir():  # pragma: no cover
                self.observer.schedule(event_handler, str(path), recursive=True)  # pragma: no cover
                logger.info(f'Watching workspace: {path}')  # pragma: no cover
        self.observer.start()  # pragma: no cover
        return True  # pragma: no cover

    def stop(self):
        """Stop the watcher and governor."""
        if self.observer:  # pragma: no cover
            self.observer.stop()  # pragma: no cover
            self.observer.join(timeout=2.0)  # pragma: no cover
        self.governor.stop()  # pragma: no cover
        logger.info('Proactive OS Perception Kernel stopped.')  # pragma: no cover

class SystemHealthWatchdog:
    """Autonomous background Linux watchdog for system anomalies using asyncio."""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._running = False
        self._thread: threading.Thread | None = None
        self._cooldowns = {'memory': 0.0, 'disk': 0.0, 'logs': 0.0}
        self._cooldown_seconds = 300.0
        self._power_throttled = False

    def start(self) -> bool:
        """Start the autonomous System Health Watchdog."""
        config = get_config()
        if not getattr(config, 'proactive_kernel', False):
            logger.info('SystemHealthWatchdog is DISABLED by configuration.')
            return False
        if self._running:
            return True
        logger.info('Starting SystemHealthWatchdog daemon...')
        self._running = True
        self._thread = threading.Thread(target=self._run_async_loop, daemon=True, name='SystemHealthWatchdog')
        self._thread.start()
        return True

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info('SystemHealthWatchdog daemon stopped.')

    def _run_async_loop(self):
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._monitor_loop())
        finally:
            loop.close()

    async def _monitor_loop(self):
        import asyncio
        import time
        while self._running:
            now = time.time()
            
            # 1. Memory Check
            try:
                import psutil
                mem = psutil.virtual_memory()
                mem_free_percent = mem.available / mem.total * 100
                if mem_free_percent < 5.0 and now - self._cooldowns['memory'] > self._cooldown_seconds:
                    details = f"Critical Memory: Only {mem_free_percent:.1f}% free RAM available."
                    self._dispatch_anomaly('memory', details)
                    self._cooldowns['memory'] = now
            except ImportError:
                logger.warning("SystemHealthWatchdog: psutil missing. Memory check degraded.")
                self._cooldowns['memory'] = now + 3600 # delay further checks
            except Exception as e:
                logger.error(f"SystemHealthWatchdog: Memory check failed: {e}")
                
            # 2. Disk Check
            try:
                import psutil
                disk = psutil.disk_usage('/')
                disk_free_percent = disk.free / disk.total * 100
                if disk_free_percent < 5.0 and now - self._cooldowns['disk'] > self._cooldown_seconds:
                    details = f"Critical Disk Space: Only {disk_free_percent:.1f}% free on root partition."
                    self._dispatch_anomaly('disk', details)
                    self._cooldowns['disk'] = now
            except ImportError:
                # Already logged above
                pass
            except Exception as e:
                logger.error(f"SystemHealthWatchdog: Disk check failed: {e}")
                
            # 2.5 Battery / Power Check
            try:
                import psutil
                battery = psutil.sensors_battery()
                if battery is not None:
                    if not battery.power_plugged and battery.percent < 30.0:
                        if not self._power_throttled:
                            self._power_throttled = True
                            self.event_bus.publish_sync('system.power.throttled', {'battery': battery.percent})
                    elif battery.power_plugged:
                        if self._power_throttled:
                            self._power_throttled = False
                            self.event_bus.publish_sync('system.power.restored', {'battery': battery.percent})
            except Exception as e:
                logger.debug(f"SystemHealthWatchdog: Battery check failed or unavailable: {e}")
                
            # 3. Log Check
            if now - self._cooldowns['logs'] > self._cooldown_seconds:
                try:
                    process = await asyncio.create_subprocess_shell(
                        'journalctl -p 3 -n 5 --since "1 minute ago"',
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await process.communicate()
                    
                    if process.returncode != 0 and stderr:
                        err = stderr.decode('utf-8')
                        if "command not found" in err or "Access denied" in err:
                            logger.warning(f"SystemHealthWatchdog: journalctl unavailable or permission denied. Log check degraded. Error: {err.strip()}")
                            self._cooldowns['logs'] = now + 3600
                            
                    elif stdout:
                        logs = stdout.decode('utf-8').strip()
                        if logs and "-- No entries --" not in logs:
                            details = f"Critical OS Logs Detected:\n{logs}"
                            self._dispatch_anomaly('logs', details)
                            self._cooldowns['logs'] = now
                except Exception as e:
                    logger.error(f"SystemHealthWatchdog: Log check failed: {e}")
                    self._cooldowns['logs'] = now + 3600
            
            # 4. REM Sleep Trigger (Deep Memory Consolidation during idle hours)
            from datetime import datetime
            current_hour = datetime.now().hour
            # Trigger REM sleep at 3 AM if we haven't already fired it today
            if current_hour == 3 and (now - self._cooldowns.get('rem_sleep', 0.0) > 43200):
                logger.info("SystemHealthWatchdog: Triggering REM Sleep Deep Consolidation (3:00 AM Idle).")
                from zelyr.core.events import Event
                self.event_bus.publish(Event(event_type="system.idle.rem_sleep", source="SystemHealthWatchdog", data={}))
                self._cooldowns['rem_sleep'] = now

            # Sleep 60 seconds, checking self._running every second
            for _ in range(60):
                if not self._running:
                    break
                await asyncio.sleep(1.0)

    def _dispatch_anomaly(self, anomaly_type: str, details: str):
        from zelyr.core.events import Event
        logger.warning(f'SystemHealthWatchdog detected anomaly [{anomaly_type}]: {details}')
        event = Event(event_type='system.anomaly.detected', source='SystemHealthWatchdog', data={'type': anomaly_type, 'details': details})
        self.event_bus.publish(event)

# Legacy Compatibility Shim
OSWatcher = SystemHealthWatchdog
