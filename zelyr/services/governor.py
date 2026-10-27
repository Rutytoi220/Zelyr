import asyncio
import logging
from typing import Dict

logger = logging.getLogger("zelyr.services.governor")

class GovernorService:
    _instance = None
    
    @classmethod
    def instance(cls, event_bus=None):
        if cls._instance is None:
            cls._instance = cls(event_bus)
        return cls._instance

    def __init__(self, event_bus=None):
        self.event_bus = event_bus
        self._strict_mode = False
        self._pending_approvals: Dict[str, asyncio.Future] = {}
        
        if self.event_bus:
            self.event_bus.subscribe("governor.approval_response", self._on_approval_response)
            self.event_bus.subscribe("governor.set_strict_mode", self._on_set_strict_mode)
            
    def is_strict_mode(self) -> bool:
        return self._strict_mode
        
    def _on_set_strict_mode(self, event):
        data = getattr(event, 'data', getattr(event, 'payload', {}))
        self._strict_mode = data.get("enabled", False)
        logger.info(f"[Governor] Strict Mode {'enabled' if self._strict_mode else 'disabled'}.")
        
    def _on_approval_response(self, event):
        data = getattr(event, 'data', getattr(event, 'payload', {}))
        tool_name = data.get("tool_name")
        approved = data.get("approved", False)
        
        future = self._pending_approvals.get(tool_name)
        if future and not future.done():
            # Resolve future safely in event loop if possible
            try:
                future.get_loop().call_soon_threadsafe(future.set_result, approved)
            except Exception:
                future.set_result(approved)
        
    async def request_approval(self, tool_name: str, arguments: dict) -> bool:
        if not self._strict_mode:
            return True
            
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending_approvals[tool_name] = future
        
        if self.event_bus:
            from zelyr.core.events import Event
            logger.info(f"[Governor] Requesting approval for {tool_name}")
            try:
                self.event_bus.publish(Event(
                    event_type="governor.approval_requested",
                    source="GovernorService",
                    data={"tool_name": tool_name, "arguments": arguments}
                ))
            except Exception as e:
                logger.error(f"[Governor] Failed to emit event: {e}")
                return False
                
        try:
            approved = await future
        finally:
            self._pending_approvals.pop(tool_name, None)
            
        return approved
