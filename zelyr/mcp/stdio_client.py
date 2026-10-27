import asyncio
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("zelyr.mcp.stdio_client")

class MCPStdioClientAsync:
    """Async JSON-RPC 2.0 client for MCP servers over stdio."""

    def __init__(self, name: str, command: str, args: List[str]):
        self.name = name
        self.command = command
        self.args = args
        self.process: Optional[asyncio.subprocess.Process] = None
        self._request_id = 0
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._running = False
        self._read_task: Optional[asyncio.Task] = None

        # Tools caching
        self.capabilities: Dict = {}
        self.server_info: Dict = {}

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def connect(self) -> bool:
        """Spawn the process and initialize the protocol."""
        try:
            self.process = await asyncio.create_subprocess_exec(
                self.command, *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            self._running = True

            # Start background reader task
            self._read_task = asyncio.create_task(self._read_loop())

            # Perform MCP Handshake
            init_req = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "zelyr", "version": "11.5"}
                }
            }

            resp = await self.send_request(init_req, timeout=10.0)
            if "error" in resp:
                logger.error(f"[{self.name}] Initialization failed: {resp['error']}")
                await self.disconnect()
                return False

            self.capabilities = resp.get("result", {}).get("capabilities", {})
            self.server_info = resp.get("result", {}).get("serverInfo", {})

            # Send initialized notification
            notif = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            await self.send_notification(notif)
            logger.info(f"[{self.name}] MCP server initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"[{self.name}] Failed to connect MCP server: {e}")
            await self.disconnect()
            return False

    async def disconnect(self):
        """Gracefully disconnect and cleanup."""
        self._running = False
        if self.process:
            try:
                if self.process.returncode is None:
                    self.process.terminate()
                    try:
                        await asyncio.wait_for(self.process.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        self.process.kill()
            except ProcessLookupError:
                pass
            self.process = None

        if self._read_task and not self._read_task.done():
            self._read_task.cancel()

        # Cancel all pending requests
        for fut in self._pending_requests.values():
            if not fut.done():
                fut.set_exception(RuntimeError("MCP server disconnected"))
        self._pending_requests.clear()

    async def _read_loop(self):
        """Continuously read from stdout asynchronously."""
        if not self.process or not self.process.stdout:
            return

        buffer = bytearray()
        while self._running:
            try:
                line = await self.process.stdout.readline()
                if not line:
                    logger.warning(f"[{self.name}] EOF from MCP server stdout.")
                    break

                payload = json.loads(line.decode('utf-8'))
                req_id = payload.get("id")

                if req_id is not None:
                    fut = self._pending_requests.get(req_id)
                    if fut and not fut.done():
                        fut.set_result(payload)
                else:
                    # Notification
                    method = payload.get("method")
                    logger.debug(f"[{self.name}] Received notification: {method}")

            except json.JSONDecodeError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[{self.name}] Read loop error: {e}")

        # Clean up on exit
        await self.disconnect()

    async def send_notification(self, payload: dict):
        if not self.process or not self.process.stdin or not self._running:
            raise RuntimeError("MCP process not running")

        raw = json.dumps(payload) + "\n"
        self.process.stdin.write(raw.encode('utf-8'))
        await self.process.stdin.drain()

    async def send_request(self, payload: dict, timeout: float = 30.0) -> Dict[str, Any]:
        """Send a JSON-RPC request and wait for the response."""
        if not self.process or not self.process.stdin or not self._running:
            raise RuntimeError("MCP process not running")

        req_id = payload.get("id")
        if req_id is None:
            req_id = self._next_id()
            payload["id"] = req_id

        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._pending_requests[req_id] = fut

        raw = json.dumps(payload) + "\n"
        self.process.stdin.write(raw.encode('utf-8'))
        await self.process.stdin.drain()

        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            if req_id in self._pending_requests:
                del self._pending_requests[req_id]
            return {"error": {"message": f"Timeout waiting for response to request {req_id}"}}
        except Exception as e:
            if req_id in self._pending_requests:
                del self._pending_requests[req_id]
            return {"error": {"message": str(e)}}

    async def list_tools(self) -> List[Dict[str, Any]]:
        """Fetch available tools from the server."""
        req = {
            "jsonrpc": "2.0",
            "method": "tools/list"
        }
        resp = await self.send_request(req)
        if "error" in resp:
            logger.error(f"[{self.name}] Failed to list tools: {resp['error']}")
            return []

        return resp.get("result", {}).get("tools", [])

    async def call_tool(self, name: str, arguments: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        """Execute a tool on the server."""
        req = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        }
        resp = await self.send_request(req, timeout=timeout)

        if "error" in resp:
            err_msg = resp["error"].get("message", str(resp["error"]))
            return {"success": False, "error": err_msg}

        content = resp.get("result", {}).get("content", [])
        is_error = resp.get("result", {}).get("isError", False)

        text = " ".join([c.get("text", "") for c in content if c.get("type") == "text"])

        if is_error:
            return {"success": False, "error": text}

        return {"success": True, "output": text}
