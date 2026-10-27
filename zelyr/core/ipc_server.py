"""Headless Daemon Mode and JSON-RPC 2.0 IPC Server for Zelyr.

Provides WebSockets and Unix Domain Sockets for external communication.
"""
from zelyr.compat import (
    RPC_PREFIX,
    canonical_rpc_method,
    rpc_method_prefix,
    state_dir,
    state_file,
)
import asyncio
import json
import logging
import os
import secrets
import webbrowser
from pathlib import Path
from typing import Any, Dict, Set, Callable
import websockets
import websockets.server
import websockets.exceptions
logger = logging.getLogger(__name__)

class JSONRPCError(Exception):
    """Auto-generated docstring.

"""

    def __init__(self, code: int, message: str, data: Any=None):
        """Auto-generated docstring.

Args:
    code: Argument.
    message: Argument.
    data: Argument.

Returns:
    Return value.
"""
        self.code = code
        self.message = message
        self.data = data
        super().__init__(message)

class ZelyrDaemon:
    """IPC Server orchestrating WS and UDS communication."""

    def __init__(self, cli: Any):
        """Auto-generated docstring.

Args:
    cli: Argument.

Returns:
    Return value.
"""
        self.cli = cli
        self.engine = cli.engine
        self.zelyr_dir = state_dir()
        self.zelyr_dir.mkdir(parents=True, exist_ok=True)
        self.token_path = state_file('daemon.token')
        self.sock_path = state_file('zelyr.sock', 'axiom.sock')
        self.token = self._load_or_generate_token()
        self.authenticated_clients: Set[Any] = set()
        self.subscribers: Dict[Any, Set[str]] = {}
        self._client_protocols: Dict[Any, str] = {}
        self._ws_server: Any | None = None
        self._uds_server: asyncio.Server | None = None
        self._http_server: Any | None = None
        self._http_task: asyncio.Task[Any] | None = None
        self._is_running = False
        if hasattr(self.engine.event_bus, 'subscribe'):
            self.engine.event_bus.subscribe('*', self._on_event)

    def _load_or_generate_token(self) -> str:
        """Load the session token, or generate a new one if it doesn't exist."""
        if not self.token_path.exists():
            token = secrets.token_hex(32)
            self.token_path.write_text(token, encoding='utf-8')
            os.chmod(self.token_path, 384)
            return token
        return self.token_path.read_text(encoding='utf-8').strip()

    async def start(self) -> None:
        """Start the WS and UDS servers."""
        if self._is_running:
            return
        self._is_running = True
        logger.info(f'Starting ZelyrDaemon. Token stored at {self.token_path}')
        self._ws_server = await websockets.serve(self._handle_ws_client, '127.0.0.1', 8765)
        logger.info('WebSocket server listening on ws://127.0.0.1:8765')
        if os.name != 'nt':
            if self.sock_path.exists():
                try:
                    self.sock_path.unlink()
                except OSError:
                    pass
            self._uds_server = await asyncio.start_unix_server(self._handle_uds_client, path=str(self.sock_path))
            os.chmod(self.sock_path, 384)
            logger.info(f'UDS server listening on {self.sock_path}')
        import uvicorn
        from zelyr.client.gateway import create_app
        app = create_app(self)
        config = uvicorn.Config(app=app, host='127.0.0.1', port=49103, log_level='error')
        self._http_server = uvicorn.Server(config)
        self._http_task = asyncio.create_task(self._http_server.serve())
        logger.info('HTTP/WS Gateway listening on http://127.0.0.1:49103')
        asyncio.create_task(self._spawn_browser())

    async def _spawn_browser(self) -> None:
        """Wait briefly for the HTTP server to bind, then spawn the browser."""
        await asyncio.sleep(0.5)
        target_url = f'http://127.0.0.1:49103/?token={self.token}'
        try:
            await asyncio.to_thread(webbrowser.open_new_tab, target_url)
            logger.info('Triggered OS default browser for GUI Gateway.')
        except Exception as e:
            logger.error(f'Failed to spawn default browser: {e}')

    async def stop(self) -> None:
        """Stop all servers."""
        self._is_running = False
        if self._ws_server:
            self._ws_server.close()
            await self._ws_server.wait_closed()
        if self._uds_server:
            self._uds_server.close()
            await self._uds_server.wait_closed()
            if self.sock_path.exists():
                try:
                    self.sock_path.unlink()
                except OSError:
                    pass
        if self._http_server:
            self._http_server.should_exit = True
            if self._http_task:
                try:
                    await asyncio.wait_for(self._http_task, timeout=5.0)
                except asyncio.TimeoutError:
                    pass
        logger.info('ZelyrDaemon stopped')

    def _on_event(self, event: Any) -> None:
        """Forward events to subscribed clients."""
        if not self._is_running or not self.subscribers:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(self._dispatch_event, event)
        except RuntimeError:
            pass

    def _dispatch_event(self, event: Any) -> None:
        """Send JSON-RPC notification to subscribers."""
        name = getattr(event, 'name', getattr(event, 'event_type', 'unknown'))
        payload = getattr(event, 'payload', getattr(event, 'data', None))
        for ws, topics in self.subscribers.items():
            if '*' in topics or name in topics:
                prefix = self._client_protocols.get(ws, RPC_PREFIX)
                msg = {
                    'jsonrpc': '2.0',
                    'method': f'{prefix}event',
                    'params': {'event_type': name, 'payload': payload},
                }
                asyncio.create_task(self._send_safely(ws, json.dumps(msg, default=str)))

    async def _send_safely(self, ws, msg: str) -> None:
        """Auto-generated docstring.

Args:
    ws: Argument.
    msg: Argument.

Returns:
    Return value.
"""
        try:
            await ws.send(msg)
        except Exception:
            pass

    async def _handle_ws_client(self, websocket) -> None:
        """Handle incoming WS connections."""
        is_authenticated = False
        try:
            async for message in websocket:
                req = {}
                try:
                    req = json.loads(message)
                except Exception:
                    pass
                method = req.get('method', '') if isinstance(req, dict) else ''
                if method:
                    self._client_protocols[websocket] = rpc_method_prefix(method)
                if canonical_rpc_method(method) == 'zelyr.subscribe' and is_authenticated:
                    event_type = req.get('params', {}).get('event_type', '*')
                    if websocket not in self.subscribers:
                        self.subscribers[websocket] = set()
                    self.subscribers[websocket].add(event_type)
                    response = self._make_response(req.get('id'), 'Subscribed')
                    await websocket.send(json.dumps(response, default=str))
                    continue
                response = await self._process_rpc_message(message, is_authenticated, is_uds=False)
                if not is_authenticated and response.get('result') == 'Authenticated':
                    is_authenticated = True
                    self.authenticated_clients.add(websocket)
                if response:
                    await websocket.send(json.dumps(response, default=str))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.authenticated_clients.discard(websocket)
            self.subscribers.pop(websocket, None)
            self._client_protocols.pop(websocket, None)

    async def _handle_uds_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle incoming UDS connections."""
        is_authenticated = True
        try:
            while True:
                data = await reader.readline()
                if not data:
                    break
                message = data.decode('utf-8').strip()
                if not message:
                    continue
                response = await self._process_rpc_message(message, is_authenticated, is_uds=True)
                if response:
                    writer.write(json.dumps(response, default=str).encode('utf-8') + b'\n')
                    await writer.drain()
                if response and response.get('result') == 'Daemon stopping':
                    asyncio.create_task(self.stop())
                    break
        except Exception as e:
            logger.error(f'UDS connection error: {e}')
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _process_rpc_message(self, message: str, is_authenticated: bool, is_uds: bool) -> Dict[str, Any]:
        """Parse and process a single JSON-RPC message."""
        req = {}
        try:
            req = json.loads(message)
            if req.get('jsonrpc') != '2.0' or 'method' not in req:
                raise JSONRPCError(-32600, 'Invalid Request')
            method = canonical_rpc_method(str(req['method']))
            params = req.get('params', {})
            msg_id = req.get('id')
            if method == 'zelyr.authenticate':
                if params.get('token') != self.token:
                    raise JSONRPCError(-32000, 'Invalid token')
                return self._make_response(msg_id, 'Authenticated')
            if not is_authenticated:
                raise JSONRPCError(-32001, 'Not authenticated')
            if method == 'zelyr.prompt':
                text = params.get('text', '')
                result = await asyncio.to_thread(self._handle_prompt, text)
                return self._make_response(msg_id, result)
            elif method == 'zelyr.status':
                status = {'engine_running': self.engine.is_running(), 'llm_available': self.cli.ollama.is_available(), 'agents': len(self.engine.registry.list_agents()), 'tools': len(self.engine.registry.list_tools())}
                return self._make_response(msg_id, status)
            elif method == 'zelyr.subscribe':
                raise JSONRPCError(-32601, 'Subscribe not supported over UDS')
            elif method == 'zelyr.stop':
                return self._make_response(msg_id, 'Daemon stopping')
            else:
                raise JSONRPCError(-32601, 'Method not found')
        except JSONRPCError as e:
            return self._make_error(req.get('id'), e.code, e.message, e.data)
        except json.JSONDecodeError:
            return self._make_error(None, -32700, 'Parse error')
        except Exception as e:
            logger.error(f'Internal RPC error: {e}', exc_info=True)
            return self._make_error(req.get('id'), -32603, 'Internal error', str(e))

    def _handle_prompt(self, text: str) -> Any:
        """Run a prompt via the orchestrator synchronously."""
        response = self.cli.orchestrator.run(text)
        if response.success:
            return response.output
        else:
            raise JSONRPCError(-32002, 'Prompt execution failed', response.error)

    def _make_response(self, msg_id: Any, result: Any) -> Dict[str, Any]:
        """Auto-generated docstring.

Args:
    msg_id: Argument.
    result: Argument.

Returns:
    Return value.
"""
        if msg_id is None:
            return {}
        return {'jsonrpc': '2.0', 'result': result, 'id': msg_id}

    def _make_error(self, msg_id: Any, code: int, message: str, data: Any=None) -> Dict[str, Any]:
        """Auto-generated docstring.

Args:
    msg_id: Argument.
    code: Argument.
    message: Argument.
    data: Argument.

Returns:
    Return value.
"""
        err = {'code': code, 'message': message}
        if data:
            err['data'] = data
        return {'jsonrpc': '2.0', 'error': err, 'id': msg_id}

# Legacy AXIOM API compatibility aliases
AxiomDaemon = ZelyrDaemon
