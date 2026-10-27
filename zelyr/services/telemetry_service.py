from zelyr.compat import config_dir
import asyncio
import logging
import time
import psutil
from typing import Dict, Any

from zelyr.core.events import EventBus, Event

logger = logging.getLogger("zelyr.telemetry")

class TelemetryService:
    """Samples and broadcasts health metrics non-blockingly."""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._running = False
        self._task = None
        self._last_tick = time.time()
        
        # Capability Detection
        import shutil
        self._has_nvidia = shutil.which("nvidia-smi") is not None
        self._has_rocm = shutil.which("rocm-smi") is not None

    def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._tick_loop())
        logger.info("TelemetryService started.")

    def stop(self):
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("TelemetryService stopped.")

    async def _tick_loop(self):
        while self._running:
            try:
                now = time.time()
                latency = (now - self._last_tick) * 1000
                self._last_tick = now

                # Basic OS Metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory()
                ram_mb = mem.used / (1024 * 1024)
                ram_percent = mem.percent

                # Hardware Telemetry (GPU Agnostic)
                vram_percent = 0.0
                vram_mb = 0.0
                gpu_name = "Integrated / CPU-Only"
                
                import subprocess
                try:
                    if self._has_nvidia:
                        gpu_name = "Nvidia Discrete"
                        proc = subprocess.run(
                            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
                            capture_output=True, text=True, timeout=1.0
                        )
                        if proc.returncode == 0 and proc.stdout.strip():
                            used, total = map(float, proc.stdout.strip().split(",")[0:2])
                            vram_mb = used
                            vram_percent = (used / total) * 100 if total > 0 else 0
                    
                    elif self._has_rocm:
                        gpu_name = "AMD Discrete (ROCm)"
                        # rocm-smi output parsing (simplified)
                        proc = subprocess.run(
                            ["rocm-smi", "--showmeminfo", "vram", "--json"],
                            capture_output=True, text=True, timeout=1.0
                        )
                        if proc.returncode == 0 and proc.stdout.strip():
                            import json
                            data = json.loads(proc.stdout)
                            # rocm-smi returns e.g. {"card0": {"VRAM Total Memory (B)": "...", "VRAM Total Used Memory (B)": "..."}}
                            card = list(data.keys())[0]
                            if card != "system":
                                total_b = float(data[card].get("VRAM Total Memory (B)", 0))
                                used_b = float(data[card].get("VRAM Total Used Memory (B)", 0))
                                if total_b > 0:
                                    vram_mb = used_b / (1024 * 1024)
                                    vram_percent = (used_b / total_b) * 100
                
                except subprocess.TimeoutExpired:
                    logger.warning(f"GPU Telemetry timeout: {gpu_name}")
                except Exception as e:
                    logger.debug(f"GPU Telemetry error: {e}")

                # Event Loop Metrics
                active_tasks = len(asyncio.all_tasks())

                # MCP Connections (Hack: try to reach the CLI or global MCP manager)
                # For a pure background service, we just emit what we know locally.
                # A proper implementation might query the manager, but here we emit standard metrics.
                mcp_servers = 0
                from zelyr.config import get_config
                cfg = get_config()
                try:
                    import json
                    from pathlib import Path
                    mcp_path = config_dir() / "mcp.json"
                    if mcp_path.exists():
                        mcp_cfg = json.loads(mcp_path.read_text())
                        mcp_servers = len(mcp_cfg.get("mcpServers", {}))
                except Exception:
                    pass

                payload = {
                    "cpu_percent": cpu_percent,
                    "ram_mb": ram_mb,
                    "ram_percent": ram_percent,
                    "vram_mb": vram_mb,
                    "vram_percent": vram_percent,
                    "gpu_name": gpu_name,
                    "active_tasks": active_tasks,
                    "loop_latency_ms": latency,
                    "mcp_servers_configured": mcp_servers
                }

                self.event_bus.publish(Event(
                    event_type="system.telemetry.tick",
                    source="TelemetryService",
                    data=payload
                ))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Telemetry tick error: {e}")

            await asyncio.sleep(2.0)
