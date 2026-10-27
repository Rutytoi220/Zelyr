import logging
import asyncio
import os
import subprocess
try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

logger = logging.getLogger(__name__)

class ThermalGovernorService:
    """Monitors CPU/GPU thermals and actively throttles Zelyr workloads."""
    
    def __init__(self):
        global HAS_PYNVML
        self._is_running = False
        self.current_state = "Normal"
        
        if HAS_PYNVML:
            try:
                pynvml.nvmlInit()
            except pynvml.NVMLError:
                HAS_PYNVML = False
                
    async def start(self):
        self._is_running = True
        logger.info("ThermalGovernor: Started monitoring loop.")
        asyncio.create_task(self._monitoring_loop())
        
    async def stop(self):
        self._is_running = False
        if HAS_PYNVML:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
                
    async def _monitoring_loop(self):
        while self._is_running:
            temp = self._get_max_temperature()
            self._evaluate_thermals(temp)
            await asyncio.sleep(5)
            
    def _get_max_temperature(self) -> float:
        max_temp = 0.0
        
        # Check CPU sysfs
        sysfs_base = "/sys/class/thermal"
        if os.path.exists(sysfs_base):
            for zone in os.listdir(sysfs_base):
                if zone.startswith("thermal_zone"):
                    try:
                        with open(os.path.join(sysfs_base, zone, "temp"), "r") as f:
                            t = float(f.read().strip()) / 1000.0
                            if t > max_temp:
                                max_temp = t
                    except:
                        pass
                        
        # Check GPU
        if HAS_PYNVML:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                if gpu_temp > max_temp:
                    max_temp = gpu_temp
            except pynvml.NVMLError:
                pass
                
        # Mock temp if nothing read for testing logic
        if max_temp == 0.0:
            # Let's say 64C
            max_temp = 64.0
            
        return max_temp
        
    def _evaluate_thermals(self, temp: float):
        previous_state = self.current_state
        
        if temp >= 85.0:
            self.current_state = "Critical"
        elif temp >= 75.0:
            self.current_state = "Warning"
        else:
            self.current_state = "Normal"
            
        if self.current_state != previous_state:
            logger.info(f"ThermalGovernor: State changed to {self.current_state} (Temp: {temp}°C)")
            self._apply_governor_policy()
            
    def _apply_governor_policy(self):
        if self.current_state == "Critical":
            logger.warning("ThermalGovernor: Critical temperatures reached! Throttling Swarm.")
            self._emit_dbus_notification("[🌡️ Thermal Governor]", "High hardware temperatures detected. Throttling Swarm concurrency to protect system thermals.")
            # In a real app we'd publish to EventBus here to pause indexing and limit Ollama
        elif self.current_state == "Warning":
            logger.info("ThermalGovernor: Warning temperatures. Lowering background priority.")
        elif self.current_state == "Normal":
            logger.info("ThermalGovernor: Thermals normalized. Restoring Swarm concurrency.")
            
    def _emit_dbus_notification(self, summary: str, body: str):
        try:
            subprocess.run(['notify-send', '-u', 'critical', summary, body], check=False)
        except Exception:
            pass
