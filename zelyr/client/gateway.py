from zelyr.compat import RPC_PREFIX, rpc_method_prefix, state_dir
import asyncio
import json
import logging
import psutil
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, Header, Query, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
logger = logging.getLogger(__name__)

class MacroStopRequest(BaseModel):
    """Auto-generated docstring.

"""
    name: str

def create_app(daemon) -> FastAPI:
    """Auto-generated docstring.

Args:
    daemon: Argument.

Returns:
    Return value.
"""
    app = FastAPI(title='Zelyr Local Gateway')
    app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
    app.state.daemon = daemon
    gui_dir = state_dir() / 'gui'
    gui_dir.mkdir(parents=True, exist_ok=True)
    index_file = gui_dir / 'index.html'
    if not index_file.exists() or index_file.stat().st_size == 0:
        fallback_html = '<!DOCTYPE html>\n<html lang="en">\n<head>\n    <meta charset="UTF-8">\n    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n    <title>Zelyr Gateway</title>\n    <style>\n        body {\n            margin: 0;\n            padding: 0;\n            font-family: \'Inter\', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;\n            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);\n            color: #f8fafc;\n            display: flex;\n            align-items: center;\n            justify-content: center;\n            height: 100vh;\n        }\n        .container {\n            background: rgba(255, 255, 255, 0.05);\n            backdrop-filter: blur(16px);\n            -webkit-backdrop-filter: blur(16px);\n            border: 1px solid rgba(255, 255, 255, 0.1);\n            border-radius: 24px;\n            padding: 48px;\n            max-width: 600px;\n            text-align: center;\n            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);\n            animation: float 6s ease-in-out infinite;\n        }\n        @keyframes float {\n            0% { transform: translateY(0px); }\n            50% { transform: translateY(-10px); }\n            100% { transform: translateY(0px); }\n        }\n        h1 {\n            font-size: 2.5rem;\n            margin-bottom: 16px;\n            background: linear-gradient(to right, #38bdf8, #818cf8);\n            -webkit-background-clip: text;\n            -webkit-text-fill-color: transparent;\n        }\n        p {\n            font-size: 1.125rem;\n            line-height: 1.6;\n            color: #94a3b8;\n            margin-bottom: 32px;\n        }\n        .code-block {\n            background: rgba(0, 0, 0, 0.4);\n            border-radius: 8px;\n            padding: 16px;\n            font-family: monospace;\n            color: #34d399;\n            text-align: left;\n            overflow-x: auto;\n            border: 1px solid rgba(255, 255, 255, 0.05);\n        }\n    </style>\n</head>\n<body>\n    <div class="container">\n        <h1>Zelyr GUI Sandbox</h1>\n        <p>The backend daemon is running securely, but no frontend assets were found.</p>\n        <p>To deploy the React GUI payload, build your frontend and copy it to the asset directory:</p>\n        <div class="code-block">\nnpm run build<br>\ncp -r dist/* ~/.zelyr/gui/\n        </div>\n    </div>\n</body>\n</html>'
        index_file.write_text(fallback_html, encoding='utf-8')

    async def verify_token(request: Request, authorization: Optional[str]=Header(None), token: Optional[str]=Query(None)):
        """Auto-generated docstring.

Args:
    request: Argument.
    authorization: Argument.
    token: Argument.

Returns:
    Return value.
"""
        expected = daemon.token
        if authorization and authorization.startswith('Bearer '):
            provided = authorization[7:]
        elif token:
            provided = token
        else:
            raise HTTPException(status_code=401, detail='Unauthorized')
        if provided != expected:
            raise HTTPException(status_code=401, detail='Invalid token')
        return provided

    @app.get('/api/status')
    async def get_status(token: Optional[str]=Query(None)):
        """Auto-generated docstring.

Args:
    token: Argument.

Returns:
    Return value.
"""
        engine = daemon.engine
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.1)
        engine_running = False
        if hasattr(engine, 'is_running'):
            engine_running = engine.is_running()
        elif hasattr(engine, '_running'):
            engine_running = engine._running
        agents = len(engine.registry.list_agents()) if hasattr(engine, 'registry') and hasattr(engine.registry, 'list_agents') else 0
        tools = len(engine.registry.list_tools()) if hasattr(engine, 'registry') and hasattr(engine.registry, 'list_tools') else 0
        plugins = len(engine.registry.list_plugins()) if hasattr(engine, 'registry') and hasattr(engine.registry, 'list_plugins') else 0
        model_name = 'Unknown'
        if hasattr(daemon, 'ollama') and hasattr(daemon.ollama, 'config'):
            model_name = daemon.ollama.config.model
        return {'engine_running': engine_running, 'ram_percent': mem.percent, 'cpu_percent': cpu, 'agents': agents, 'tools': tools, 'plugins': plugins, 'model': model_name}

    @app.post('/api/macros/start')
    async def start_macro(token: str=Depends(verify_token)):
        """Auto-generated docstring.

Args:
    token: Argument.

Returns:
    Return value.
"""
        plugin = daemon.engine.registry.get_plugin('automation')
        if not plugin:
            raise HTTPException(status_code=404, detail='Automation plugin not found')
        plugin.start_recording()
        return {'status': 'recording_started'}

    @app.post('/api/macros/stop')
    async def stop_macro(req: MacroStopRequest, token: str=Depends(verify_token)):
        """Auto-generated docstring.

Args:
    req: Argument.
    token: Argument.

Returns:
    Return value.
"""
        plugin = daemon.engine.registry.get_plugin('automation')
        if not plugin:
            raise HTTPException(status_code=404, detail='Automation plugin not found')
        macro_id = plugin.stop_recording(req.name)
        if not macro_id:
            raise HTTPException(status_code=400, detail='No steps recorded')
        return {'status': 'recording_stopped', 'macro_id': macro_id}

    @app.get('/api/macros')
    async def get_macros(token: Optional[str]=Query(None)):
        """Auto-generated docstring.

Args:
    token: Argument.

Returns:
    Return value.
"""
        plugin = daemon.engine.registry.get_plugin('automation')
        if not plugin:
            raise HTTPException(status_code=404, detail='Automation plugin not found')
        return {'macros': plugin.list_macros()}

    @app.post('/api/macros/{macro_id}/execute')
    async def execute_macro(macro_id: str, token: str=Depends(verify_token)):
        """Auto-generated docstring.

Args:
    macro_id: Argument.
    token: Argument.

Returns:
    Return value.
"""
        plugin = daemon.engine.registry.get_plugin('automation')
        if not plugin:
            raise HTTPException(status_code=404, detail='Automation plugin not found')
        success = plugin.execute_macro(macro_id)
        if not success:
            raise HTTPException(status_code=400, detail='Failed to execute macro')
        return {'status': 'executing'}

    @app.delete('/api/macros/{macro_id}')
    async def delete_macro(macro_id: str, token: str=Depends(verify_token)):
        """Auto-generated docstring.

Args:
    macro_id: Argument.
    token: Argument.

Returns:
    Return value.
"""
        plugin = daemon.engine.registry.get_plugin('automation')
        if not plugin:
            raise HTTPException(status_code=404, detail='Automation plugin not found')
        success = plugin.delete_macro(macro_id)
        if not success:
            raise HTTPException(status_code=404, detail='Macro not found')
        return {'status': 'deleted'}

    @app.websocket('/ws/events')
    async def websocket_endpoint(websocket: WebSocket, token: Optional[str]=Query(None)):
        """Auto-generated docstring.

Args:
    websocket: Argument.
    token: Argument.

Returns:
    Return value.
"""
        client_host = websocket.client.host if websocket.client else ''
        is_local = client_host in ('127.0.0.1', 'localhost', '::1')
        expected = daemon.token
        if not is_local and token != expected:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        queue: asyncio.Queue[str] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        protocol = {'prefix': RPC_PREFIX}

        async def receive_control_messages() -> None:
            """Learn whether a client speaks the legacy or canonical dialect."""
            while True:
                try:
                    raw = await websocket.receive_text()
                except (WebSocketDisconnect, RuntimeError):
                    return
                try:
                    request = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    continue
                if isinstance(request, dict):
                    method = str(request.get('method', ''))
                    if method.startswith(('zelyr.', 'axiom.')):
                        protocol['prefix'] = rpc_method_prefix(method)

        receiver = asyncio.create_task(receive_control_messages())

        def event_handler(event):
            """Auto-generated docstring.

Args:
    event: Argument.

Returns:
    Return value.
"""
            name = getattr(event, 'name', getattr(event, 'event_type', 'unknown'))
            payload = getattr(event, 'payload', getattr(event, 'data', None))
            msg = {
                'jsonrpc': '2.0',
                'method': f"{protocol['prefix']}event",
                'params': {'event_type': name, 'payload': payload},
            }
            try:
                loop.call_soon_threadsafe(queue.put_nowait, json.dumps(msg, default=str))
            except Exception:
                pass
        bus = getattr(daemon.engine, 'event_bus', None)
        if bus and hasattr(bus, 'subscribe'):
            bus.subscribe('*', event_handler)
        try:
            while True:
                msg = await queue.get()
                await websocket.send_text(msg)
        except WebSocketDisconnect:
            pass
        finally:
            receiver.cancel()
            try:
                await receiver
            except asyncio.CancelledError:
                pass
            if bus and hasattr(bus, 'unsubscribe'):
                bus.unsubscribe('*', event_handler)
    app.mount('/', StaticFiles(directory=str(gui_dir), html=True), name='gui')
    return app
