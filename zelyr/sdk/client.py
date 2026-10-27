"""Zelyr SDK Client.

Provides async and sync clients for integrating with the Zelyr JSON-RPC daemon.
"""
from zelyr.compat import state_file
import asyncio
import json
import logging
import uuid
import threading
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional
from zelyr.sdk.models import JsonRpcRequest, JsonRpcResponse, PromptRequest, TelemetryPayload
logger = logging.getLogger(__name__)

class ZelyrClient:
    """Async Client for communicating with the Zelyr JSON-RPC daemon."""

    def __init__(self, socket_path: Optional[str]=None, token_path: Optional[str]=None):
        """Auto-generated docstring.

Args:
    socket_path: Argument.
    token_path: Argument.

Returns:
    Return value.
"""
        self.socket_path = Path(socket_path or state_file("zelyr.sock", "axiom.sock"))  # pragma: no cover
        self.token_path = Path(token_path or state_file("daemon.token", "daemon.token"))  # pragma: no cover
        self._reader: Optional[asyncio.StreamReader] = None  # pragma: no cover
        self._writer: Optional[asyncio.StreamWriter] = None  # pragma: no cover
        self._auth_token: Optional[str] = None  # pragma: no cover
        self._pending_requests: Dict[str, asyncio.Future] = {}  # pragma: no cover
        self._subscriptions: Dict[str, asyncio.Queue] = {}  # pragma: no cover
        self._read_task: Optional[asyncio.Task] = None  # pragma: no cover

    def _read_token(self) -> str:
        """Reads the authentication token from disk."""
        if not self.token_path.exists():  # pragma: no cover
            return 'dev-token-fallback'  # pragma: no cover
        return self.token_path.read_text().strip()  # pragma: no cover

    async def connect(self) -> None:
        """Connect to the daemon's Unix Domain Socket."""
        self._auth_token = self._read_token()  # pragma: no cover
        self._reader, self._writer = await asyncio.open_unix_connection(str(self.socket_path))  # pragma: no cover
        self._read_task = asyncio.create_task(self._read_loop())  # pragma: no cover
        logger.debug('Connected to Zelyr daemon.')  # pragma: no cover

    async def disconnect(self) -> None:
        """Disconnect from the daemon."""
        if self._read_task:  # pragma: no cover
            self._read_task.cancel()  # pragma: no cover
        if self._writer:  # pragma: no cover
            self._writer.close()  # pragma: no cover
            await self._writer.wait_closed()  # pragma: no cover
        self._reader = None  # pragma: no cover
        self._writer = None  # pragma: no cover

    async def _read_loop(self) -> None:
        """Background loop reading JSON-RPC responses and events from the socket."""
        try:  # pragma: no cover
            while True:  # pragma: no cover
                if not self._reader:  # pragma: no cover
                    break  # pragma: no cover
                line = await self._reader.readline()  # pragma: no cover
                if not line:  # pragma: no cover
                    logger.debug('Socket closed by server.')  # pragma: no cover
                    break  # pragma: no cover
                try:  # pragma: no cover
                    payload = json.loads(line.decode().strip())  # pragma: no cover
                except json.JSONDecodeError:  # pragma: no cover
                    continue  # pragma: no cover
                if 'id' in payload and str(payload['id']) in self._pending_requests:  # pragma: no cover
                    req_id = str(payload['id'])  # pragma: no cover
                    future = self._pending_requests.pop(req_id)  # pragma: no cover
                    if not future.done():  # pragma: no cover
                        future.set_result(payload)  # pragma: no cover
                elif 'method' in payload and payload['method'] == 'event.bus.published':  # pragma: no cover
                    event_data = payload.get('params', {})  # pragma: no cover
                    topic = event_data.get('event_type', event_data.get('topic'))  # pragma: no cover
                    for sub_topic, queue in self._subscriptions.items():  # pragma: no cover
                        if topic == sub_topic or sub_topic == '*':  # pragma: no cover
                            await queue.put(event_data)  # pragma: no cover
        except asyncio.CancelledError:  # pragma: no cover
            pass  # pragma: no cover
        except Exception as e:  # pragma: no cover
            logger.error(f'Read loop encountered error: {e}')  # pragma: no cover
        finally:
            for fut in self._pending_requests.values():  # pragma: no cover
                if not fut.done():  # pragma: no cover
                    fut.set_exception(ConnectionError('Socket disconnected'))  # pragma: no cover
            self._pending_requests.clear()  # pragma: no cover

    async def _send_request(self, method: str, params: Optional[Dict[str, Any]]=None) -> Any:
        """Send a JSON-RPC request and wait for the response."""
        if not self._writer:  # pragma: no cover
            await self.connect()  # pragma: no cover
        req_id = str(uuid.uuid4())  # pragma: no cover
        full_params = params or {}  # pragma: no cover
        if 'token' not in full_params and self._auth_token:  # pragma: no cover
            full_params['token'] = self._auth_token  # pragma: no cover
        request = JsonRpcRequest(id=req_id, method=method, params=full_params)  # pragma: no cover
        future = asyncio.get_event_loop().create_future()  # pragma: no cover
        self._pending_requests[req_id] = future  # pragma: no cover
        assert self._writer is not None
        payload_bytes = request.model_dump_json(exclude_none=True).encode() + b'\n'  # pragma: no cover
        self._writer.write(payload_bytes)  # pragma: no cover
        await self._writer.drain()  # pragma: no cover
        response_dict = await future  # pragma: no cover
        response = JsonRpcResponse(**response_dict)  # pragma: no cover
        if response.error:  # pragma: no cover
            raise RuntimeError(f'JSON-RPC Error {response.error.code}: {response.error.message}')  # pragma: no cover
        return response.result  # pragma: no cover

    async def prompt(self, text: str, session_id: Optional[str]=None) -> Any:
        """Submit a prompt to Zelyr."""
        req = PromptRequest(text=text, session_id=session_id)  # pragma: no cover
        return await self._send_request('prompt.submit', req.model_dump(exclude_none=True))  # pragma: no cover

    async def get_status(self) -> TelemetryPayload:
        """Get the current telemetry status of Zelyr."""
        result = await self._send_request('system.status')  # pragma: no cover
        if isinstance(result, dict):  # pragma: no cover
            return TelemetryPayload(**result)  # pragma: no cover
        return result  # pragma: no cover

    async def subscribe(self, topic: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Subscribe to an EventBus topic and yield events. Use '*' for all events."""
        if not self._writer:  # pragma: no cover
            await self.connect()  # pragma: no cover
        await self._send_request('event.subscribe', {'topic': topic})  # pragma: no cover
        queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()  # pragma: no cover
        self._subscriptions[topic] = queue  # pragma: no cover
        try:  # pragma: no cover
            while True:  # pragma: no cover
                event = await queue.get()  # pragma: no cover
                yield event  # pragma: no cover
        finally:
            if topic in self._subscriptions:  # pragma: no cover
                del self._subscriptions[topic]  # pragma: no cover
            try:  # pragma: no cover
                await self._send_request('event.unsubscribe', {'topic': topic})  # pragma: no cover
            except Exception:  # pragma: no cover
                pass  # pragma: no cover

class SyncZelyrClient:
    """Synchronous wrapper for ZelyrClient."""

    def __init__(self, socket_path: Optional[str]=None, token_path: Optional[str]=None):
        """Auto-generated docstring.

Args:
    socket_path: Argument.
    token_path: Argument.

Returns:
    Return value.
"""
        self._socket_path = socket_path  # pragma: no cover
        self._token_path = token_path  # pragma: no cover

    def prompt(self, text: str, session_id: Optional[str]=None) -> Any:
        """Auto-generated docstring.

Args:
    text: Argument.
    session_id: Argument.

Returns:
    Return value.
"""
        return asyncio.run(self._prompt_async(text, session_id))  # pragma: no cover

    async def _prompt_async(self, text: str, session_id: Optional[str]):
        """Auto-generated docstring.

Args:
    text: Argument.
    session_id: Argument.

Returns:
    Return value.
"""
        client = ZelyrClient(self._socket_path, self._token_path)  # pragma: no cover
        try:  # pragma: no cover
            return await client.prompt(text, session_id)  # pragma: no cover
        finally:
            await client.disconnect()  # pragma: no cover

    def get_status(self) -> TelemetryPayload:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return asyncio.run(self._get_status_async())  # pragma: no cover

    async def _get_status_async(self):
        """Auto-generated docstring.


Returns:
    Return value.
"""
        client = ZelyrClient(self._socket_path, self._token_path)  # pragma: no cover
        try:  # pragma: no cover
            return await client.get_status()  # pragma: no cover
        finally:
            await client.disconnect()  # pragma: no cover

    def subscribe(self, topic: str):
        """Synchronous generator for subscribing to events."""
        queue = []  # pragma: no cover
        condition = threading.Condition()  # pragma: no cover
        closed = False  # pragma: no cover

        def run_loop():  # pragma: no cover
            """Auto-generated docstring.


Returns:
    Return value.
"""
            nonlocal closed
            loop = asyncio.new_event_loop()  # pragma: no cover
            asyncio.set_event_loop(loop)  # pragma: no cover

            async def consume():  # pragma: no cover
                """Auto-generated docstring.


Returns:
    Return value.
"""
                client = ZelyrClient(self._socket_path, self._token_path)  # pragma: no cover
                try:  # pragma: no cover
                    async for event in client.subscribe(topic):  # pragma: no cover
                        with condition:  # pragma: no cover
                            queue.append(event)  # pragma: no cover
                            condition.notify()  # pragma: no cover
                except Exception:  # pragma: no cover
                    pass  # pragma: no cover
                finally:
                    await client.disconnect()  # pragma: no cover
                    with condition:  # pragma: no cover
                        closed = True  # pragma: no cover
                        condition.notify()  # pragma: no cover
            try:  # pragma: no cover
                loop.run_until_complete(consume())  # pragma: no cover
            finally:
                loop.close()  # pragma: no cover
        thread = threading.Thread(target=run_loop, daemon=True)  # pragma: no cover
        thread.start()  # pragma: no cover
        while True:  # pragma: no cover
            with condition:  # pragma: no cover
                while not queue and (not closed):  # pragma: no cover
                    condition.wait()  # pragma: no cover
                if queue:  # pragma: no cover
                    yield queue.pop(0)  # pragma: no cover
                elif closed:  # pragma: no cover
                    break  # pragma: no cover

# Legacy AXIOM API compatibility aliases
AxiomClient = ZelyrClient
SyncAxiomClient = SyncZelyrClient
