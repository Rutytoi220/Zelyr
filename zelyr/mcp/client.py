import json
import logging
import subprocess
import threading
import uuid
import time
from typing import Dict, Any, List, Optional

logger = logging.getLogger("zelyr.mcp.client")

class MCPStdioClient:
    """A lightweight JSON-RPC client for MCP servers over stdio."""

    def __init__(self, name: str, command: str, args: List[str]):
        self.name = name
        self.command = command
        self.args = args
        self.process: Optional[subprocess.Popen] = None
        self._request_id = 0

        # Threading for response futures
        self._pending_requests: Dict[int, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._running = False
        self._reader_thread: Optional[threading.Thread] = None

    def _next_id(self) -> int:
        with self._lock:
            self._request_id += 1
            return self._request_id

    def connect(self) -> bool:
        """Spawn the process and initialize the protocol."""
        try:
            self.process = subprocess.Popen(
                [self.command] + self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            self._running = True

            # Start the background reader thread
            self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
            self._reader_thread.start()

            # Perform MCP Handshake
            init_req = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "zelyr", "version": "11.0"}
                }
            }

            # Send initialize and wait for response
            resp = self.send_request(init_req)
            if "error" in resp:
                logger.error(f"[{self.name}] Initialization failed: {resp['error']}")
                return False

            # Send initialized notification
            notif = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            self.send_notification(notif)
            logger.info(f"[{self.name}] MCP server initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"[{self.name}] Failed to connect MCP server: {e}")
            return False

    def disconnect(self):
        self._running = False
        if self.process:
            self.process.terminate()
            self.process = None

    def _read_loop(self):
        """Continuously read from the process stdout."""
        if not self.process or not self.process.stdout:
            return

        while self._running:
            try:
                line = self.process.stdout.readline()
                if not line:
                    time.sleep(0.01)
                    continue

                payload = json.loads(line)
                req_id = payload.get("id")

                if req_id is not None:
                    with self._lock:
                        if req_id in self._pending_requests:
                            self._pending_requests[req_id] = payload
                else:
                    # It's a notification
                    method = payload.get("method")
                    logger.debug(f"[{self.name}] Received notification: {method}")

            except json.JSONDecodeError:
                continue
            except Exception as e:
                if self._running:
                    logger.debug(f"[{self.name}] Read loop exception: {e}")

    def send_notification(self, payload: dict):
        if not self.process or not self.process.stdin:
            raise RuntimeError("MCP process not running")

        raw = json.dumps(payload) + "\n"
        self.process.stdin.write(raw)
        self.process.stdin.flush()

    def send_request(self, payload: dict, timeout: float = 30.0) -> Dict[str, Any]:
        """Send a JSON-RPC request and synchronously block for the response."""
        if not self.process or not self.process.stdin:
            raise RuntimeError("MCP process not running")

        req_id = payload.get("id")
        if req_id is None:
            req_id = self._next_id()
            payload["id"] = req_id

        with self._lock:
            self._pending_requests[req_id] = None  # type: ignore

        # Write to stdin
        raw = json.dumps(payload) + "\n"
        self.process.stdin.write(raw)
        self.process.stdin.flush()

        # Poll for response
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self._lock:
                resp = self._pending_requests.get(req_id)
                if resp is not None:
                    del self._pending_requests[req_id]
                    return resp
            time.sleep(0.05)

        with self._lock:
            if req_id in self._pending_requests:
                del self._pending_requests[req_id]

        return {"error": {"message": f"Timeout waiting for response to request {req_id}"}}

    def list_tools(self) -> List[Dict[str, Any]]:
        """Fetch available tools from the server."""
        req = {
            "jsonrpc": "2.0",
            "method": "tools/list"
        }
        resp = self.send_request(req)
        if "error" in resp:
            logger.error(f"[{self.name}] Failed to list tools: {resp['error']}")
            return []

        return resp.get("result", {}).get("tools", [])

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool on the server."""
        req = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        }
        resp = self.send_request(req, timeout=60.0)

        if "error" in resp:
            return {"success": False, "error": resp["error"].get("message", str(resp["error"]))}

        content = resp.get("result", {}).get("content", [])
        text = " ".join([c.get("text", "") for c in content if c.get("type") == "text"])
        return {"success": True, "output": text}
