"""Zelyr Swarm WebSocket Client.

Manages a persistent async WebSocket connection to a remote Zelyr Node.
Runs on a background QThread and emits Qt signals back to the main GUI thread.
"""
import asyncio
import json
import logging
import threading
from typing import Optional

from PySide6.QtCore import QObject, Signal, QThread

logger = logging.getLogger(__name__)


class SwarmWorker(QObject):
    """Runs in a background QThread — manages the async WebSocket lifecycle."""

    connected = Signal()
    disconnected = Signal()
    error = Signal(str)
    token_received = Signal(str)
    response_complete = Signal(str)

    def __init__(self):
        super().__init__()
        self._ws = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._url: Optional[str] = None
        self._is_connected = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Public API (called from GUI thread — thread-safe)                   #
    # ------------------------------------------------------------------ #
    def start_connection(self, url: str):
        self._url = url
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._connect(), self._loop)

    def disconnect(self):
        if self._loop and self._loop.is_running() and self._ws:
            asyncio.run_coroutine_threadsafe(self._disconnect(), self._loop)

    def send_prompt(self, prompt: str):
        if self._loop and self._loop.is_running() and self._is_connected and self._ws:
            asyncio.run_coroutine_threadsafe(self._send(prompt), self._loop)
            return True
        return False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    # ------------------------------------------------------------------ #
    # QThread entry point                                                  #
    # ------------------------------------------------------------------ #
    def run_event_loop(self):
        """Called from QThread.started. Spins the asyncio event loop forever."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        if self._url:
            self._loop.run_until_complete(self._connect())
        self._loop.run_forever()

    # ------------------------------------------------------------------ #
    # Async internals                                                      #
    # ------------------------------------------------------------------ #
    async def _connect(self):
        try:
            import websockets
            logger.info(f"SwarmClient: connecting to {self._url}")
            self._ws = await websockets.connect(self._url, ping_interval=20)
            self._is_connected = True
            self.connected.emit()
            logger.info("SwarmClient: connection established")
            await self._recv_loop()
        except Exception as e:
            self._is_connected = False
            self._ws = None
            logger.error(f"SwarmClient: connection failed: {e}")
            self.error.emit(str(e))

    async def _recv_loop(self):
        """Receive messages from the remote Node until the connection closes."""
        try:
            while True:
                raw = await self._ws.recv()
                try:
                    data = json.loads(raw)
                    status = data.get("status")
                    if status == "complete":
                        self.response_complete.emit(data.get("response", ""))
                    elif status == "error":
                        self.error.emit(data.get("message", "Unknown node error"))
                    # "processing" status → no-op, we show loading state in UI
                except json.JSONDecodeError:
                    # Raw text fallback
                    self.token_received.emit(raw)
        except Exception:
            self._is_connected = False
            self._ws = None
            self.disconnected.emit()

    async def _send(self, prompt: str):
        try:
            if self._ws:
                await self._ws.send(json.dumps({"prompt": prompt}))
        except Exception as e:
            logger.error(f"SwarmClient: send failed: {e}")
            self.error.emit(str(e))

    async def _disconnect(self):
        if self._ws:
            await self._ws.close()
        self._is_connected = False
        self._ws = None
        self.disconnected.emit()


class SwarmClient(QObject):
    """High-level controller that owns the QThread + SwarmWorker.

    Exposes simple Qt signals that the MainWindow can connect to directly.
    """

    # Re-exported signals for MainWindow convenience
    connected = Signal()
    disconnected = Signal()
    connection_error = Signal(str)
    response_complete = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = QThread(self)
        self._worker = SwarmWorker()
        self._worker.moveToThread(self._thread)

        # Wire worker → client signals
        self._worker.connected.connect(self.connected)
        self._worker.disconnected.connect(self.disconnected)
        self._worker.error.connect(self.connection_error)
        self._worker.response_complete.connect(self.response_complete)

        # Start the thread's asyncio event loop
        self._thread.started.connect(self._worker.run_event_loop)
        self._thread.start()

    @property
    def is_connected(self) -> bool:
        return self._worker.is_connected

    def connect_to_node(self, host: str):
        """Connect to ws://<host>/ws/swarm (adds schema/path if missing)."""
        host = host.strip().rstrip("/")
        if not host.startswith("ws://") and not host.startswith("wss://"):
            host = f"ws://{host}"
        url = f"{host}/ws/swarm"
        self._worker.start_connection(url)

    def disconnect_from_node(self):
        self._worker.disconnect()

    def send_prompt(self, prompt: str) -> bool:
        """Returns True if sent over swarm, False if swarm is offline."""
        return self._worker.send_prompt(prompt)

    def shutdown(self):
        self._worker.disconnect()
        self._thread.quit()
        self._thread.wait(3000)
