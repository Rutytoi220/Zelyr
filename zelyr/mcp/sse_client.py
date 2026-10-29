"""Model Context Protocol (MCP) Remote Server-Sent Events (SSE) Client Transport.

Provides asynchronous HTTP/SSE transport for communicating with remote MCP servers,
including dynamic endpoint negotiation, message correlation, automatic reconnection
with exponential backoff, and tool bridging into Zelyr's ToolRegistry.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.parse
from typing import Any, Dict, List, Optional

import httpx

from zelyr.tools.core import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger("zelyr.mcp.sse_client")


class MCPSSEBridgedTool(BaseTool):
    """Wrapper that translates a BaseTool execution into an MCP tools/call request
    over an MCPSSEClient transport.
    """

    def __init__(
        self,
        server_name: str,
        client: MCPSSEClient,
        mcp_tool_def: Dict[str, Any],
        timeout: float = 30.0,
    ) -> None:
        self._server_name = server_name
        self._client = client
        self._remote_tool_name = mcp_tool_def.get("name", "")
        self._timeout = timeout

        tool_id = f"mcp__{server_name}__{self._remote_tool_name}"
        desc = f"[{server_name} MCP] " + mcp_tool_def.get("description", "")

        super().__init__(tool_id=tool_id, name=tool_id, description=desc)

        input_schema = mcp_tool_def.get("inputSchema", {})
        props = input_schema.get("properties", {})
        req = input_schema.get("required", [])

        for param_name, meta in props.items():
            self.add_parameter(
                ToolParameter(
                    name=param_name,
                    type=meta.get("type", "string"),
                    description=meta.get("description", ""),
                    required=param_name in req,
                    default=meta.get("default"),
                )
            )

    async def execute(self, params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> ToolResult:
        """Execute the remote MCP tool via HTTP/SSE."""
        args = dict(params or {})
        args.update(kwargs)
        try:
            resp = await asyncio.wait_for(
                self._client.call_tool(self._remote_tool_name, args, timeout=self._timeout),
                timeout=self._timeout,
            )
            if not resp.get("success", False):
                return ToolResult(
                    success=False,
                    error=resp.get("error", "Unknown MCP SSE Error"),
                )
            return ToolResult(
                success=True,
                output=resp.get("output", ""),
            )
        except asyncio.TimeoutError:
            err_msg = f"MCP SSE Tool {self.tool_id} timed out after {self._timeout}s"
            logger.error(err_msg)
            return ToolResult(success=False, error=err_msg)
        except Exception as exc:
            logger.error(f"Failed to execute MCP SSE tool {self.tool_id}: {exc}")
            return ToolResult(success=False, error=str(exc))


class MCPSSEClient:
    """Async Client for Model Context Protocol (MCP) using HTTP Server-Sent Events (SSE).

    Protocol flow:
    1. Client performs HTTP GET to SSE stream endpoint (e.g. /sse).
    2. Server sends "endpoint" event with relative or absolute POST URL for client messages.
    3. Client performs JSON-RPC 2.0 handshake ('initialize') and sends 'notifications/initialized'.
    4. Outbound JSON-RPC requests are sent via HTTP POST to the resolved endpoint.
    5. Server responses and notifications stream back via SSE 'message' events.
    6. Responses are correlated by message 'id' through asyncio.Future objects.
    7. Transient network drops trigger automatic reconnection with exponential backoff.
    """

    def __init__(
        self,
        name: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        auto_reconnect: bool = True,
        initial_reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 15.0,
    ) -> None:
        self.name = name
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout
        self._transport = transport
        self.auto_reconnect = auto_reconnect
        self.initial_reconnect_delay = initial_reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay

        self.post_url: Optional[str] = None
        self.capabilities: Dict[str, Any] = {}
        self.server_info: Dict[str, Any] = {}

        self._request_id = 0
        self._pending_requests: Dict[int, asyncio.Future[Dict[str, Any]]] = {}
        self._running = False
        self._connected = False
        self._endpoint_received = asyncio.Event()

        self._http_client: Optional[httpx.AsyncClient] = None
        self._listen_task: Optional[asyncio.Task] = None
        self._reconnect_count = 0
        self._lock = asyncio.Lock()

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def connect(self, timeout: Optional[float] = None) -> bool:
        """Connect to the remote SSE stream, resolve the POST endpoint, and initialize MCP."""
        connect_timeout = timeout or self.timeout
        async with self._lock:
            if self._connected:
                return True

            self._running = True
            if self._http_client is None or self._http_client.is_closed:
                self._http_client = httpx.AsyncClient(
                    headers=self.headers,
                    timeout=httpx.Timeout(self.timeout),
                    transport=self._transport,
                )

            self._endpoint_received.clear()
            self._listen_task = asyncio.create_task(self._read_loop())

            try:
                # Wait for initial "endpoint" SSE event
                await asyncio.wait_for(self._endpoint_received.wait(), timeout=connect_timeout)
            except asyncio.TimeoutError:
                logger.error(f"[{self.name}] Timed out waiting for MCP SSE 'endpoint' event.")
                await self.close()
                return False
            except Exception as e:
                logger.error(f"[{self.name}] Error during MCP SSE handshake: {e}")
                await self.close()
                return False

            # Perform MCP Handshake: initialize
            init_payload = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "zelyr", "version": "1.2.0"},
                },
            }

            try:
                resp = await self.send_request(init_payload, timeout=connect_timeout)
                if "error" in resp:
                    logger.error(f"[{self.name}] MCP initialization failed: {resp['error']}")
                    await self.close()
                    return False

                result = resp.get("result", {})
                self.capabilities = result.get("capabilities", {})
                self.server_info = result.get("serverInfo", {})

                # Send initialized notification
                await self.send_notification("notifications/initialized")
                self._connected = True
                logger.info(
                    f"[{self.name}] Connected to MCP SSE server at {self.url} (post endpoint: {self.post_url})"
                )
                return True
            except Exception as e:
                logger.error(f"[{self.name}] Failed to complete initialize handshake: {e}")
                await self.close()
                return False

    async def _read_loop(self) -> None:
        """Background loop continuously reading Server-Sent Events with auto-reconnection."""
        reconnect_delay = self.initial_reconnect_delay

        while self._running:
            try:
                if self._http_client is None or self._http_client.is_closed:
                    self._http_client = httpx.AsyncClient(
                        headers=self.headers,
                        timeout=httpx.Timeout(self.timeout),
                        transport=self._transport,
                    )

                headers = {"Accept": "text/event-stream"}
                headers.update(self.headers)

                async with self._http_client.stream("GET", self.url, headers=headers) as response:
                    if response.status_code >= 400:
                        logger.warning(
                            f"[{self.name}] SSE connection returned status {response.status_code}"
                        )
                        raise httpx.HTTPStatusError(
                            f"HTTP {response.status_code}",
                            request=response.request,
                            response=response,
                        )

                    # Reconnection or first connect successful
                    reconnect_delay = self.initial_reconnect_delay

                    current_event = "message"
                    current_data: List[str] = []

                    async for line in response.aiter_lines():
                        if not self._running:
                            break

                        line = line.rstrip("\r\n")
                        if not line:
                            # Empty line dispatches the buffered event
                            if current_data:
                                event_payload = "\n".join(current_data)
                                self._dispatch_sse_event(current_event, event_payload)
                                current_event = "message"
                                current_data = []
                            continue

                        if line.startswith(":"):
                            # Comment / keep-alive ping, safe to ignore
                            continue

                        if ":" in line:
                            field, _, val = line.partition(":")
                            field = field.strip()
                            val = val.lstrip(" ")
                            if field == "event":
                                current_event = val
                            elif field == "data":
                                current_data.append(val)

                    # If stream ended while event buffer had data, dispatch it
                    if current_data:
                        event_payload = "\n".join(current_data)
                        self._dispatch_sse_event(current_event, event_payload)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                if not self._running:
                    break
                logger.warning(f"[{self.name}] SSE stream disconnected: {exc}")

            if not self._running:
                break

            if not self.auto_reconnect:
                logger.info(f"[{self.name}] Auto-reconnect is disabled. Stopping SSE read loop.")
                break

            # Reconnection backoff
            self._reconnect_count += 1
            logger.info(
                f"[{self.name}] Reconnecting in {reconnect_delay:.2f}s (attempt {self._reconnect_count})..."
            )
            try:
                await asyncio.sleep(reconnect_delay)
            except asyncio.CancelledError:
                break

            reconnect_delay = min(reconnect_delay * 2.0, self.max_reconnect_delay)

    def _dispatch_sse_event(self, event: str, data: str) -> None:
        """Handle dispatched SSE event."""
        if event == "endpoint":
            endpoint = data.strip()
            self.post_url = urllib.parse.urljoin(self.url, endpoint)
            self._endpoint_received.set()
            logger.debug(f"[{self.name}] Resolved MCP POST endpoint: {self.post_url}")
        elif event == "message":
            try:
                payload = json.loads(data)
            except json.JSONDecodeError as exc:
                logger.warning(f"[{self.name}] Invalid JSON in SSE message: {exc}")
                return

            req_id = payload.get("id")
            if req_id is not None:
                fut = self._pending_requests.get(req_id)
                if fut and not fut.done():
                    fut.set_result(payload)
            else:
                method = payload.get("method")
                logger.debug(f"[{self.name}] Received server notification: {method}")
        elif event in ("ping", "keepalive"):
            logger.debug(f"[{self.name}] Received keepalive event: {event}")
        else:
            logger.debug(f"[{self.name}] Unhandled SSE event '{event}': {data[:100]}")

    async def send_request(
        self,
        method_or_payload: str | Dict[str, Any],
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Send a JSON-RPC request to the resolved POST endpoint and await response."""
        req_timeout = timeout or self.timeout

        if not self._running:
            raise RuntimeError(f"MCP SSE client '{self.name}' is not running")

        # Wait until endpoint is resolved
        if not self._endpoint_received.is_set():
            await asyncio.wait_for(self._endpoint_received.wait(), timeout=req_timeout)

        if not self.post_url:
            raise RuntimeError(f"MCP SSE client '{self.name}' has no resolved POST endpoint")

        if isinstance(method_or_payload, dict):
            payload = dict(method_or_payload)
            req_id = payload.get("id")
            if req_id is None:
                req_id = self._next_id()
                payload["id"] = req_id
            if "jsonrpc" not in payload:
                payload["jsonrpc"] = "2.0"
        else:
            req_id = self._next_id()
            payload = {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": method_or_payload,
                "params": params if params is not None else {},
            }

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Dict[str, Any]] = loop.create_future()
        self._pending_requests[req_id] = fut

        try:
            if self._http_client is None or self._http_client.is_closed:
                raise RuntimeError("HTTP client is closed")

            post_resp = await self._http_client.post(
                self.post_url,
                json=payload,
                timeout=req_timeout,
            )

            if post_resp.status_code >= 400:
                self._pending_requests.pop(req_id, None)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32603,
                        "message": f"HTTP {post_resp.status_code}: {post_resp.text}",
                    },
                }

            # If response returned directly in body
            content_type = post_resp.headers.get("content-type", "")
            if "application/json" in content_type:
                try:
                    direct_json = post_resp.json()
                    if isinstance(direct_json, dict) and direct_json.get("id") == req_id:
                        if not fut.done():
                            fut.set_result(direct_json)
                except Exception:
                    pass

            # Await the response from SSE stream or body
            return await asyncio.wait_for(fut, timeout=req_timeout)

        except asyncio.TimeoutError:
            self._pending_requests.pop(req_id, None)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32603,
                    "message": f"Timeout waiting for response to request {req_id}",
                },
            }
        except Exception as exc:
            self._pending_requests.pop(req_id, None)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32603,
                    "message": str(exc),
                },
            }

    async def send_notification(
        self,
        method_or_payload: str | Dict[str, Any],
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send a JSON-RPC notification to the resolved POST endpoint."""
        if not self._running:
            raise RuntimeError(f"MCP SSE client '{self.name}' is not running")

        if not self._endpoint_received.is_set():
            await asyncio.wait_for(self._endpoint_received.wait(), timeout=self.timeout)

        if not self.post_url:
            raise RuntimeError(f"MCP SSE client '{self.name}' has no resolved POST endpoint")

        if isinstance(method_or_payload, dict):
            payload = dict(method_or_payload)
            payload.pop("id", None)
            if "jsonrpc" not in payload:
                payload["jsonrpc"] = "2.0"
        else:
            payload = {
                "jsonrpc": "2.0",
                "method": method_or_payload,
                "params": params if params is not None else {},
            }

        if self._http_client and not self._http_client.is_closed:
            await self._http_client.post(self.post_url, json=payload, timeout=self.timeout)

    async def list_tools(self) -> List[Dict[str, Any]]:
        """Fetch available tools from the remote MCP server."""
        resp = await self.send_request("tools/list")
        if "error" in resp:
            logger.error(f"[{self.name}] Failed to list tools: {resp['error']}")
            return []
        return resp.get("result", {}).get("tools", [])

    async def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """Execute a tool on the remote server."""
        req_params = {
            "name": name,
            "arguments": arguments,
        }
        resp = await self.send_request("tools/call", req_params, timeout=timeout)

        if "error" in resp:
            err = resp["error"]
            err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            return {"success": False, "error": err_msg}

        result = resp.get("result", {})
        is_error = result.get("isError", False)
        content = result.get("content", [])

        texts: List[str] = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                texts.append(c.get("text", ""))
            elif isinstance(c, str):
                texts.append(c)

        output = " ".join(texts)
        if is_error:
            return {"success": False, "error": output or "Remote MCP tool error"}

        return {"success": True, "output": output}

    async def register_tools(self, registry: Any) -> List[str]:
        """Fetch remote tools and register each into the provided ToolRegistry
        with namespacing prefix 'mcp__<server_name>__<tool_name>'.
        """
        tools = await self.list_tools()
        registered_ids: List[str] = []

        for tool_def in tools:
            bridged_tool = MCPSSEBridgedTool(
                server_name=self.name,
                client=self,
                mcp_tool_def=tool_def,
                timeout=self.timeout,
            )
            tool_id = bridged_tool.tool_id

            if hasattr(registry, "register_tool"):
                registry.register_tool(tool_id, bridged_tool)
            elif hasattr(registry, "register"):
                try:
                    registry.register(bridged_tool)
                except Exception:
                    registry.register_tool(tool_id, bridged_tool)
            elif hasattr(registry, "add_tool"):
                registry.add_tool(bridged_tool)

            registered_ids.append(tool_id)
            logger.info(f"Registered MCP SSE tool '{tool_id}'")

        return registered_ids

    def unregister_tools(self, registry: Any, tool_ids: Optional[List[str]] = None) -> None:
        """Unregister tools from the registry."""
        ids_to_remove = tool_ids or []
        for tool_id in ids_to_remove:
            if hasattr(registry, "unregister_tool"):
                registry.unregister_tool(tool_id)
            elif hasattr(registry, "unregister"):
                registry.unregister(tool_id)

    async def close(self) -> None:
        """Terminate the client, cancel background tasks, and close HTTP sessions."""
        self._running = False
        self._connected = False
        self._endpoint_received.clear()

        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
            try:
                await self._listen_task
            except (asyncio.CancelledError, Exception):
                pass
            self._listen_task = None

        # Fail pending requests
        for fut in list(self._pending_requests.values()):
            if not fut.done():
                fut.set_exception(RuntimeError(f"MCP SSE client '{self.name}' closed"))
        self._pending_requests.clear()

        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    async def disconnect(self) -> None:
        """Alias for close()."""
        await self.close()
