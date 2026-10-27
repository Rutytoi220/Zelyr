"""Zelyr Swarm Router.

Intercepts the active generation request and routes it to remote nodes
if the local node is thermally throttled or otherwise constrained.
"""

import logging
from typing import Any, Optional

from zelyr.core.events import EventBus
from zelyr.config import get_config

logger = logging.getLogger(__name__)

class SwarmRouter:
    """Singleton that manages network offloading of LLM inference."""
    
    _instance = None
    
    @classmethod
    def instance(cls, event_bus: Optional[EventBus] = None):
        if cls._instance is None:
            if event_bus is None:
                # If we don't have an event bus, we just run passively (no dynamic triggering)
                cls._instance = cls(None)
            else:
                cls._instance = cls(event_bus)
        return cls._instance

    def __init__(self, event_bus: Optional[EventBus]):
        if SwarmRouter._instance is not None:
            raise RuntimeError("SwarmRouter is a singleton. Use .instance().")
            
        self.bus = event_bus
        self._is_throttled = False
        self._active_endpoint = None
        
        if self.bus:
            self.bus.subscribe("system.throttle", self._on_throttle)
            
        logger.info("SwarmRouter initialized.")

    def _on_throttle(self, event: Any) -> None:
        """Handle thermal governor throttle signals."""
        data = getattr(event, "data", {})
        active = data.get("active", False)
        
        if self._is_throttled == active:
            return
            
        self._is_throttled = active
        self._recalculate_route()

    def _recalculate_route(self) -> None:
        config = get_config()
        
        previous_endpoint = self._active_endpoint
        
        if not config.swarm_enabled or not config.remote_endpoints:
            self._active_endpoint = None
        else:
            if config.offload_strategy == 'thermal_trigger' and self._is_throttled:
                # Naive routing: just pick the first available endpoint
                self._active_endpoint = config.remote_endpoints[0]
            elif config.offload_strategy == 'manual':
                self._active_endpoint = config.remote_endpoints[0]
            else:
                self._active_endpoint = None
                
        if previous_endpoint != self._active_endpoint:
            if self._active_endpoint:
                logger.info(f"SwarmRouter: Offloading inference to {self._active_endpoint}")
            else:
                logger.info("SwarmRouter: Resuming local inference")
                
            if self.bus:
                self.bus.publish_sync("swarm.status.changed", {
                    "active": self._active_endpoint is not None,
                    "endpoint": self._active_endpoint
                })

    def get_ollama_base_url(self) -> str:
        """Dynamically return the correct Ollama API base URL."""
        # Ensure we're up to date with config if someone changed it manually without a throttle event
        self._recalculate_route()
        
        if self._active_endpoint:
            endpoint = self._active_endpoint
            if not endpoint.startswith("http"):
                endpoint = f"http://{endpoint}"
            # Don't add port if user already specified one
            if ":" not in endpoint.replace("http://", "").replace("https://", ""):
                endpoint = f"{endpoint}:11434"
            return endpoint
            
        return "http://127.0.0.1:11434"
