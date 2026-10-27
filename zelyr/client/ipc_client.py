import asyncio
import json
import logging
import websockets
from typing import Callable, Optional

logger = logging.getLogger(__name__)

class ZelyrDaemonClient:
    def __init__(self, uri="ws://127.0.0.1:8000"):
        self.uri = uri
        self.ws = None
        self.on_event: Optional[Callable[[dict], None]] = None
        self.on_connect: Optional[Callable[[], None]] = None
        self.on_disconnect: Optional[Callable[[], None]] = None
        self._listen_task = None
        self._connected = False

    async def connect(self):
        try:
            self.ws = await websockets.connect(self.uri, extra_headers={"X-Protection-Ring": "0"})
            self._connected = True
            logger.info(f"Connected to daemon at {self.uri}")
            if self.on_connect:
                self.on_connect()
            self._listen_task = asyncio.create_task(self._listen())

            # Phase 2: Send state sync handshake
            try:
                await self.ws.send(json.dumps({"action": "sync_state"}))
            except Exception as e:
                logger.error(f"Failed to send sync_state: {e}")

            return True
        except Exception as e:
            logger.error(f"Failed to connect to daemon: {e}")
            self._connected = False
            if self.on_disconnect:
                self.on_disconnect()
            return False

    async def disconnect(self):
        self._connected = False
        if self.ws:
            await self.ws.close()
            self.ws = None
        if self._listen_task:
            self._listen_task.cancel()
            self._listen_task = None
        if self.on_disconnect:
            self.on_disconnect()

    @property
    def is_connected(self):
        return self._connected

    async def _listen(self):
        try:
            async for message in self.ws:
                try:
                    data = json.loads(message)
                    if data.get("type") in ("event", "response"):
                        if self.on_event:
                            self.on_event(data)
                except json.JSONDecodeError:
                    logger.warning("Received invalid JSON from daemon")
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Daemon connection closed")
        finally:
            await self.disconnect()

    async def request_tools(self):
        if not self.ws or not self._connected:
            return False
        try:
            await self.ws.send(json.dumps({"action": "get_tools"}))
            return True
        except Exception as e:
            logger.error(f"Failed to request tools: {e}")
            return False

    async def toggle_tool(self, tool_id: str, enabled: bool):
        if not self.ws or not self._connected:
            return False
        try:
            await self.ws.send(json.dumps({
                "action": "toggle_tool",
                "tool_id": tool_id,
                "enabled": enabled
            }))
            return True
        except Exception as e:
            logger.error(f"Failed to toggle tool: {e}")
            return False

    async def submit_task(self, prompt: str):
        if not self.ws or not self._connected:
            logger.error("Cannot submit task, not connected to daemon.")
            return False

        try:
            payload = {
                "action": "submit_task",
                "prompt": prompt
            }
            await self.ws.send(json.dumps(payload))
            return True
        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            return False

    async def submit_task_and_stream(self, prompt: str):
        """Submit task and stream response to stdout, blocking until complete."""
        import sys
        if not await self.connect():
            return False

        loop_complete = asyncio.Event()

        def handle_event(data):
            if data.get("type") == "event":
                event_type = data.get("event", {}).get("type")
                if event_type == "telemetry.token":
                    token = data.get("event", {}).get("data", {}).get("token", "")
                    sys.stdout.write(token)
                    sys.stdout.flush()
                elif event_type == "telemetry.update":
                    msg = data.get("event", {}).get("data", {}).get("message", "")
                    sys.stdout.write(f"\n{msg}\n")
                    sys.stdout.flush()
                elif event_type == "orchestrator.completed":
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    loop_complete.set()
                elif event_type == "orchestrator.error":
                    sys.stdout.write(f"\nError: {data.get('event', {}).get('data', {}).get('error', '')}\n")
                    sys.stdout.flush()
                    loop_complete.set()

        self.on_event = handle_event

        success = await self.submit_task(prompt)
        if success:
            await loop_complete.wait()

        await self.disconnect()
        return success

# Legacy AXIOM API compatibility aliases
AxiomDaemonClient = ZelyrDaemonClient
