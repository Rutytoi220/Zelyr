"""Proactive System Health Watchdog."""
import time
import logging
import threading
import psutil
import os
from typing import Optional, Callable

from zelyr.gui.notifications import DesktopNotifier

logger = logging.getLogger(__name__)

class SystemHealthWatchdog:
    """Monitors system metrics and triggers diagnostics when thresholds are breached."""

    def __init__(self, submit_task_callback: Optional[Callable[[str], None]] = None):
        self._running = False
        self._thread = None
        self._submit_task = submit_task_callback
        
        self.poll_interval = 15.0  # Seconds
        
        # Thresholds
        self.cpu_threshold = 95.0
        self.ram_threshold = 90.0
        self.disk_gb_threshold = 5.0
        
        # Debounce (prevent spam)
        self._last_alert_time = 0
        self._alert_cooldown = 300  # 5 minutes

    def start(self):
        """Starts the watchdog background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True, name="SystemWatchdog")
        self._thread.start()
        logger.info("SystemHealthWatchdog started.")

    def stop(self):
        """Stops the watchdog."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _monitor_loop(self):
        while self._running:
            try:
                self._check_metrics()
            except Exception as e:
                logger.error(f"Watchdog error: {e}")
            time.sleep(self.poll_interval)

    def _check_metrics(self):
        if time.time() - self._last_alert_time < self._alert_cooldown:
            return

        alerts = []
        
        # CPU
        cpu = psutil.cpu_percent(interval=1.0)
        if cpu > self.cpu_threshold:
            alerts.append(f"CPU usage is at {cpu}%")

        # RAM
        ram = psutil.virtual_memory()
        if ram.percent > self.ram_threshold:
            alerts.append(f"RAM usage is at {ram.percent}%")

        # Disk /
        try:
            if not os.path.exists('/run/ostree-booted'):
                root_disk = psutil.disk_usage('/')
                root_free_gb = root_disk.free / (1024**3)
                if root_free_gb < self.disk_gb_threshold:
                    alerts.append(f"Root disk '/' has only {root_free_gb:.1f}GB free")
        except Exception:
            pass
            
        # Disk /var/home or /home
        try:
            home_path = '/var/home' if os.path.exists('/var/home') else '/home'
            home_disk = psutil.disk_usage(home_path)
            home_free_gb = home_disk.free / (1024**3)
            if home_free_gb < self.disk_gb_threshold:
                alerts.append(f"Home disk '{home_path}' has only {home_free_gb:.1f}GB free")
        except Exception:
            pass

        if alerts:
            self._trigger_alert(alerts)

    def _trigger_alert(self, alerts: list):
        self._last_alert_time = time.time()
        
        msg = ", ".join(alerts)
        logger.warning(f"System Alert: {msg}")
        
        DesktopNotifier.notify(
            title="[Zelyr System Alert] High Resource Usage",
            body=f"{msg}. Running autonomous diagnosis...",
            icon="dialog-warning"
        )
        
        if self._submit_task:
            prompt = f"System alert detected: {msg}. Please run diagnostic commands (like 'ps aux --sort=-%mem | head -n 5' or 'df -h') to identify the issue and prepare a remediation plan."
            # We must call it safely if it expects main thread
            self._submit_task(prompt)
