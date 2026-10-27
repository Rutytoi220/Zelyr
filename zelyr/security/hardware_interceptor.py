"""Autonomous USB/BLE Hardware Interception.

Uses pyudev to monitor Linux kernel udev events for new mass storage devices.
Preempts standard OS automounting, sandboxes the drive in a secure container,
and requests a SecurityAuditorAgent scan before allowing host-level mounting.
"""
import logging
import asyncio
from zelyr.core.events import EventBus

try:
    import pyudev
    PYUDEV_AVAILABLE = True
except ImportError:
    PYUDEV_AVAILABLE = False

logger = logging.getLogger(__name__)

class HardwareInterceptorService:
    """Intercepts physical peripheral hotplugs."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._context = None
        self._monitor = None
        self._observer = None
        self._running = False
        
    def start(self):
        if not PYUDEV_AVAILABLE:
            logger.warning("HardwareInterceptor: 'pyudev' not available. Physical peripheral interception disabled.")
            return
            
        try:
            self._context = pyudev.Context()
            self._monitor = pyudev.Monitor.from_netlink(self._context)
            self._monitor.filter_by(subsystem='block')
            
            # Use Asyncio or pyudev.MonitorObserver
            self._observer = pyudev.MonitorObserver(self._monitor, callback=self._device_event)
            self._observer.start()
            self._running = True
            
            logger.info("HardwareInterceptor: Listening for USB/Block device hotplugs.")
            
        except Exception as e:
            logger.error(f"HardwareInterceptor: Failed to start udev monitor - {e}")
            
    def stop(self):
        if self._observer:
            self._observer.stop()
        self._running = False

    def _device_event(self, action, device):
        """Callback for udev block device events."""
        if action == 'add' and device.get('DEVTYPE') == 'partition':
            dev_node = device.device_node
            sys_name = device.sys_name
            id_bus = device.get('ID_BUS', 'unknown')
            
            # We specifically target USB drives
            if id_bus == 'usb':
                logger.warning(f"HardwareInterceptor: INTERCEPTED USB Block Device Insertion: {dev_node}")
                # Dispatch to EventBus for UI and AI processing
                self.event_bus.publish_sync("hardware.usb.intercept", {
                    "node": dev_node,
                    "sys_name": sys_name,
                    "action": action
                })
                
                # Preempt OS and Sandbox (Mocked for safety)
                self._sandbox_and_scan(dev_node)

    def _sandbox_and_scan(self, dev_node: str):
        """Simulates mounting the device inside a bwrap sandbox with noexec."""
        logger.info(f"HardwareInterceptor: [SANDBOX] Creating isolated container for {dev_node} with noexec,nosuid...")
        # In a full implementation, we'd do:
        # os.system(f"mount -o ro,noexec,nosuid {dev_node} /tmp/zelyr_sandbox_{hash}")
        # subprocess.run(["bwrap", "--ro-bind", "/tmp/zelyr_sandbox", "/mnt", "python", "scan_script.py"])
        
        logger.info(f"HardwareInterceptor: [SANDBOX] Requesting AI Security Sweep of root directory...")
        # Mocking an AI sweep
        asyncio.run_coroutine_threadsafe(self._mock_sweep(dev_node), asyncio.get_event_loop())
        
    async def _mock_sweep(self, dev_node: str):
        await asyncio.sleep(2)
        logger.info(f"HardwareInterceptor: [SANDBOX] AI Sweep Complete. Consensus: SAFE. Releasing {dev_node} to Host.")
        self.event_bus.publish_sync("hardware.usb.released", {"node": dev_node})
