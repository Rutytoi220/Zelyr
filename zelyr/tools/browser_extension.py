"""Zelyr Browser Extension Local WebSocket Automation Bridge.

Provides zero-friction local browser automation via a WebSocket daemon on ws://127.0.0.1:41144.
Compatible with Zen Browser (Firefox/Gecko) and Chromium browsers running the Zelyr WebExtension.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Set

import websockets
try:
    from websockets.asyncio.server import Server as WebSocketServer, serve as ws_serve
except ImportError:
    from websockets.server import WebSocketServer, serve as ws_serve  # type: ignore[no-redef]

logger = logging.getLogger("zelyr.browser_bridge")

DEFAULT_BRIDGE_HOST = "127.0.0.1"
DEFAULT_BRIDGE_PORT = 41144


class BrowserResult(dict):
    """Dictionary subclass providing attribute-style access for tool results."""

    @property
    def success(self) -> bool:
        return bool(self.get("success", False))

    @property
    def error(self) -> Optional[str]:
        return self.get("error")

    @property
    def remedy_hint(self) -> Optional[str]:
        return self.get("remedy_hint")

    @property
    def allowed_actions(self) -> List[str]:
        return self.get("allowed_actions") or []

    @property
    def output(self) -> Any:
        return (
            self.get("summary")
            or self.get("elements")
            or self.get("result")
            or self.get("tabs")
            or self.get("tab")
            or self.get("content")
            or ({"tab_id": self.get("tab_id"), "title": self.get("title"), "url": self.get("url")} if "tab_id" in self else None)
            or ({"closed_tab_id": self.get("closed_tab_id"), "title": self.get("title")} if "closed_tab_id" in self else None)
            or ({"clicked_id": self.get("clicked_id"), "tag": self.get("tag"), "text": self.get("text")} if "clicked_id" in self else None)
            or ({"element_id": self.get("element_id"), "element_type": self.get("element_type"), "value": self.get("value"), "text": self.get("text"), "submitted": self.get("submitted")} if "element_id" in self and "element_type" in self else None)
            or ({"element_id": self.get("element_id"), "text": self.get("text"), "submitted": self.get("submitted")} if "element_id" in self and "submitted" in self else None)
            or ({"screenshot_path": self.get("screenshot_path"), "format": self.get("format", "png"), "length": self.get("length")} if "screenshot_path" in self else None)
            or ({"format": self.get("format"), "data_url_prefix": self.get("data_url_prefix"), "length": self.get("length")} if "data_url_prefix" in self else None)
            or ({"scrollY": self.get("scrollY"), "innerHeight": self.get("innerHeight"), "scrollHeight": self.get("scrollHeight")} if "scrollY" in self else None)
            or ({"pinned": self.get("pinned"), "tab_id": self.get("tab_id")} if "pinned" in self else None)
            or self
        )


def _is_client_open(client: Any) -> bool:
    """Returns True if client WebSocket connection is open."""
    if client is None:
        return False
    state = getattr(client, "state", None)
    if state is not None:
        name = getattr(state, "name", str(state))
        return name == "OPEN"
    return bool(getattr(client, "open", True) and not getattr(client, "closed", False))


class BrowserExtensionBridge:
    """Manages WebSocket connections from browser extensions and routes commands."""

    _instance: Optional[BrowserExtensionBridge] = None

    def __init__(self, host: str = DEFAULT_BRIDGE_HOST, port: int = DEFAULT_BRIDGE_PORT):
        self.host = host
        self.port = port
        self._client: Optional[Any] = None
        self.active_clients: Set[Any] = set()
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self.server: Optional[WebSocketServer] = None
        self._lock: Optional[asyncio.Lock] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._last_activity: float = time.monotonic()

    @property
    def pending_requests(self) -> Dict[str, asyncio.Future]:
        return self._pending_requests

    @pending_requests.setter
    def pending_requests(self, val: Dict[str, asyncio.Future]) -> None:
        self._pending_requests = val

    def _get_lock(self) -> asyncio.Lock:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if self._lock is None:
            self._lock = asyncio.Lock()
        elif loop is not None and getattr(self._lock, "_loop", None) not in (None, loop):
            self._lock = asyncio.Lock()
        return self._lock

    @classmethod
    def get_instance(cls, host: str = DEFAULT_BRIDGE_HOST, port: int = DEFAULT_BRIDGE_PORT) -> BrowserExtensionBridge:
        if cls._instance is None:
            cls._instance = cls(host=host, port=port)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    def is_connected(self) -> bool:
        """Returns True if at least one browser extension is actively connected."""
        if self._client is not None:
            if _is_client_open(self._client):
                return True
            self._client = None

        self.active_clients = {
            c for c in self.active_clients
            if _is_client_open(c)
        }
        if self.active_clients and self._client is None:
            self._client = next(iter(self.active_clients))
        return len(self.active_clients) > 0

    @property
    def client_count(self) -> int:
        return len(self.active_clients)

    async def _listen(self, websocket: Any, *args: Any) -> None:
        """Handles incoming WebSocket connection from a browser extension."""
        # When a new client connects: if self._client is already set and open, close it cleanly first.
        if self._client is not None and self._client != websocket:
            try:
                if _is_client_open(self._client):
                    await self._client.close()
            except Exception:
                pass
        self._client = websocket
        self.active_clients.add(websocket)
        remote = getattr(websocket, "remote_address", "client")
        logger.info("[Zelyr Extension Bridge] Browser extension connected: %s", remote)

        try:
            async for raw_message in websocket:
                self._last_activity = time.monotonic()
                try:
                    data = json.loads(raw_message)

                    # Handle heartbeat ping frames from browser extension
                    msg_type = data.get("type")
                    msg_action = data.get("action")
                    if msg_type == "ping" or msg_action == "ping":
                        logger.debug("Heartbeat ping received from extension.")
                        pong_payload = {"type": "pong", "id": data.get("id"), "action": "pong", "success": True}
                        await websocket.send(json.dumps(pong_payload))
                        continue

                    # Silently consume or ignore pong replies from extension
                    if msg_type == "pong" or msg_action == "pong":
                        logger.debug("Heartbeat pong received from extension.")
                        continue

                    msg_id = data.get("id")
                    if msg_id and msg_id in self._pending_requests:
                        future = self._pending_requests.pop(msg_id)
                        if not future.done():
                            future.set_result(data)
                except json.JSONDecodeError:
                    logger.warning("[Zelyr Extension Bridge] Received invalid JSON payload: %s", raw_message[:150])
        except (websockets.ConnectionClosed, ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            is_active = (self._client == websocket)
            if is_active:
                self._client = None
            self.active_clients.discard(websocket)
            if is_active or len(self.active_clients) == 0:
                for req_id, fut in list(self._pending_requests.items()):
                    if not fut.done():
                        fut.set_exception(ConnectionResetError("Browser extension disconnected mid-command."))
                self._pending_requests.clear()
            logger.info("[Zelyr Extension Bridge] Browser extension disconnected: %s", remote)

    _handle_client = _listen

    async def _heartbeat_loop(self) -> None:
        """Periodically sends ping frames to keep connection alive if idle for > 15s."""
        try:
            while True:
                await asyncio.sleep(5.0)
                if self.is_connected() and self._client is not None:
                    idle_time = time.monotonic() - self._last_activity
                    if idle_time >= 15.0:
                        try:
                            ping_payload = {"id": "heartbeat", "action": "ping"}
                            await self._client.send(json.dumps(ping_payload))
                            self._last_activity = time.monotonic()
                        except Exception as exc:
                            logger.debug("[Zelyr Extension Bridge] Heartbeat ping failed: %s", exc)
        except asyncio.CancelledError:
            pass

    async def ping(self) -> bool:
        """Sends an immediate heartbeat ping to the connected browser extension."""
        if not self.is_connected() or self._client is None:
            return False
        try:
            ping_payload = {"id": "heartbeat", "action": "ping"}
            await self._client.send(json.dumps(ping_payload))
            self._last_activity = time.monotonic()
            return True
        except Exception:
            return False

    async def start_server(self) -> None:
        """Starts the WebSocket server on 127.0.0.1:41144 with socket reuse."""
        async with self._get_lock():
            if self.server is not None:
                return
            try:
                self.server = await ws_serve(
                    self._listen,
                    self.host,
                    self.port,
                    reuse_address=True,
                )
                logger.info("[Zelyr Extension Bridge] Daemon listening on ws://%s:%d", self.host, self.port)
                self._last_activity = time.monotonic()
                if self._heartbeat_task is None or self._heartbeat_task.done():
                    self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            except OSError as exc:
                logger.warning("[Zelyr Extension Bridge] Failed to bind ws://%s:%d: %s", self.host, self.port, exc)

    async def stop_server(self) -> None:
        """Closes all client connections and shuts down the server."""
        async with self._get_lock():
            if self._heartbeat_task is not None and not self._heartbeat_task.done():
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
                self._heartbeat_task = None

            for fut in list(self._pending_requests.values()):
                if not fut.done():
                    fut.set_exception(ConnectionResetError("Browser extension disconnected mid-command."))
            self._pending_requests.clear()

            if self._client is not None:
                try:
                    await self._client.close()
                except Exception:
                    pass
                self._client = None

            for client in list(self.active_clients):
                try:
                    await client.close()
                except Exception:
                    pass
            self.active_clients.clear()

            if self.server is not None:
                try:
                    self.server.close()
                    await self.server.wait_closed()
                except Exception as exc:
                    logger.debug("[Zelyr Extension Bridge] Error closing server: %s", exc)
                finally:
                    self.server = None
            logger.info("[Zelyr Extension Bridge] Daemon shut down.")

    async def ensure_server(self) -> None:
        """Ensures the daemon is listening."""
        if self.server is None:
            await self.start_server()

    async def send_command(self, action: str, timeout: float = 8.5, **kwargs: Any) -> BrowserResult:
        """Dispatches an action to the connected extension and waits for response."""
        await self.ensure_server()

        if not self.is_connected():
            return BrowserResult({
                "success": False,
                "error": (
                    f"No browser extension connected on ws://{self.host}:{self.port}. "
                    "Ensure the Zelyr extension is loaded in Zen Browser (about:debugging) or Chromium. "
                    "Do not ask for --remote-debugging-port flags."
                ),
                "port": self.port,
                "action": action,
            })

        msg_id = f"ext_{uuid.uuid4().hex[:10]}"
        kwargs.pop("action", None)
        payload = {
            "id": msg_id,
            "action": action,
            **kwargs,
        }

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending_requests[msg_id] = future

        client = self._client or next(iter(self.active_clients))
        try:
            await client.send(json.dumps(payload))
            self._last_activity = time.monotonic()
        except (websockets.ConnectionClosed, ConnectionResetError, BrokenPipeError) as exc:
            self._pending_requests.pop(msg_id, None)
            raise ConnectionResetError("Browser extension disconnected mid-command.") from exc
        except Exception as exc:
            self._pending_requests.pop(msg_id, None)
            return BrowserResult({
                "success": False,
                "error": f"Failed to send '{action}' to browser extension: {exc}",
                "action": action,
            })

        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            return BrowserResult(response)
        except asyncio.TimeoutError:
            self._pending_requests.pop(msg_id, None)
            return BrowserResult({
                "success": False,
                "error": f"Command '{action}' timed out after {timeout}s awaiting extension response",
                "action": action,
            })


def get_bridge() -> BrowserExtensionBridge:
    """Returns the singleton BrowserExtensionBridge instance."""
    return BrowserExtensionBridge.get_instance()


async def interact_with_browser(
    action: str,
    query: str = "",
    selector: str = "",
    text: str = "",
    url: str = "",
    tab_id: Any = None,
    element_id: Optional[int] = None,
    submit: bool = False,
    timeout: float = 8.5,
    **kwargs: Any,
) -> BrowserResult:
    """Interacts with browser via local WebExtension WebSocket bridge.

    Actions:
      - navigate_url: Navigates current tab to URL in-place without opening a new tab.
      - capture_tab_screenshot: Captures visible viewport as PNG.
      - get_page_snapshot: Inspects active tab, numbers interactive elements, and returns compact summary.
      - click_element: Clicks element by numeric snapshot ID (element_id).
      - fill_element: Types text into element by numeric ID, or selects dropdown/checkbox, optionally submitting form.
      - list_tabs: Query list of open tabs with IDs, titles, URLs, active state.
      - switch_tab: Matches tab titles or URLs against query and switches focus.
      - open_tab: Opens a new tab with given URL.
      - close_tab: Closes the tab matching tab_id or query.
      - get_active_tab: Returns details of currently focused tab.
      - click: Evaluates document.querySelector(selector).click() in active tab.
      - type: Dispatches synthetic input/typing to selector in active tab.
      - get_dom: Returns text and structure content of active tab.
    """
    bridge = get_bridge()
    cmd_args: Dict[str, Any] = dict(kwargs)
    if query:
        cmd_args["query"] = query
    if selector:
        cmd_args["selector"] = selector
    if text:
        cmd_args["text"] = text
    if url:
        cmd_args["url"] = url
    if tab_id is not None and tab_id != "":
        cmd_args["tab_id"] = tab_id
    if element_id is not None and element_id != "":
        cmd_args["element_id"] = element_id
    if submit:
        cmd_args["submit"] = submit

    return await bridge.send_command(action, timeout=timeout, **cmd_args)


def __getattr__(name: str) -> Any:
    if name == "InteractWithBrowserTool":
        from zelyr.tools.browser_cdp import InteractWithBrowserTool
        return InteractWithBrowserTool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
