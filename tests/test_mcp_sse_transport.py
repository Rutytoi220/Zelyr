"""Tests for Model Context Protocol (MCP) Remote Server-Sent Events (SSE) Transport."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional
import pytest
import httpx

pytestmark = pytest.mark.integration

from zelyr.mcp.sse_client import MCPSSEBridgedTool, MCPSSEClient
from zelyr.mcp.manager import MCPManager
from zelyr.tool_registry import ToolRegistry
from zelyr.tools.core import ToolResult


class MockMCPServer:
    """Mock remote MCP server communicating via HTTP SSE and POST endpoints."""

    def __init__(
        self,
        sse_url: str = "http://testserver/sse",
        post_endpoint: str = "/messages",
        respond_in_post_body: bool = False,
    ) -> None:
        self.sse_url = sse_url
        self.post_endpoint = post_endpoint
        self.respond_in_post_body = respond_in_post_body
        self.sse_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.post_requests: List[Dict[str, Any]] = []
        self.connect_count = 0
        self.disconnect_on_connect_count: Optional[int] = None

        self.tools = [
            {
                "name": "remote_calculator",
                "description": "Calculates math expressions",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "expr": {
                            "type": "string",
                            "description": "The math expression to evaluate",
                        }
                    },
                    "required": ["expr"],
                },
            }
        ]

    def emit_event(self, event: str, data: str) -> None:
        payload = f"event: {event}\ndata: {data}\n\n".encode("utf-8")
        self.sse_queue.put_nowait(payload)

    def emit_comment(self, comment: str) -> None:
        self.sse_queue.put_nowait(f": {comment}\n\n".encode("utf-8"))

    async def _sse_stream(self):
        self.connect_count += 1
        # Always emit initial endpoint event upon connection
        self.emit_event("endpoint", self.post_endpoint)

        if self.disconnect_on_connect_count == self.connect_count:
            # Yield initial endpoint, then raise transport disconnect
            chunk = await self.sse_queue.get()
            yield chunk
            raise httpx.TransportError("Simulated sudden connection drop")

        while True:
            chunk = await self.sse_queue.get()
            if chunk == b"__CLOSE__":
                break
            yield chunk

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)

        if request.method == "GET" and url_str.endswith("/sse"):
            return httpx.Response(
                200,
                content=self._sse_stream(),
                headers={"content-type": "text/event-stream"},
            )

        if request.method == "POST" and self.post_endpoint in url_str:
            body = json.loads(request.read().decode("utf-8"))
            self.post_requests.append(body)
            req_id = body.get("id")
            method = body.get("method")

            if method == "initialize":
                resp_payload = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "mock-remote-server", "version": "1.0.0"},
                    },
                }
                if self.respond_in_post_body:
                    return httpx.Response(200, json=resp_payload)
                self.emit_event("message", json.dumps(resp_payload))
                return httpx.Response(202, text="Accepted")

            if method == "notifications/initialized":
                return httpx.Response(200, text="OK")

            if method == "tools/list":
                resp_payload = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"tools": self.tools},
                }
                if self.respond_in_post_body:
                    return httpx.Response(200, json=resp_payload)
                self.emit_event("message", json.dumps(resp_payload))
                return httpx.Response(202, text="Accepted")

            if method == "tools/call":
                params = body.get("params", {})
                tool_name = params.get("name")
                arguments = params.get("arguments", {})

                if tool_name == "remote_calculator":
                    expr = arguments.get("expr", "")
                    if expr == "trigger_rpc_error":
                        resp_payload = {
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "error": {"code": -32603, "message": "Internal computation failure"},
                        }
                    elif expr == "trigger_tool_error":
                        resp_payload = {
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": {
                                "content": [{"type": "text", "text": "Division by zero error"}],
                                "isError": True,
                            },
                        }
                    else:
                        resp_payload = {
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": {
                                "content": [{"type": "text", "text": f"Computed: {expr} = 42"}],
                                "isError": False,
                            },
                        }

                    if self.respond_in_post_body:
                        return httpx.Response(200, json=resp_payload)
                    self.emit_event("message", json.dumps(resp_payload))
                    return httpx.Response(202, text="Accepted")

            return httpx.Response(404, text=f"Unknown method {method}")

        return httpx.Response(404, text="Not Found")


@pytest.fixture
def mock_server() -> MockMCPServer:
    return MockMCPServer()


@pytest.fixture
def transport(mock_server: MockMCPServer) -> httpx.MockTransport:
    return httpx.MockTransport(mock_server.handle_request)


@pytest.mark.asyncio
async def test_handshake_and_endpoint_resolution(mock_server, transport):
    """Phase 3.1: Test initial endpoint discovery, initialize handshake, and post URL resolution."""
    client = MCPSSEClient(
        name="test_srv",
        url="http://testserver/sse",
        transport=transport,
    )

    try:
        connected = await client.connect()
        assert connected is True
        assert client._connected is True
        assert client.post_url == "http://testserver/messages"
        assert client.server_info.get("name") == "mock-remote-server"
        assert client.server_info.get("version") == "1.0.0"

        # Verify sent messages: initialize and notifications/initialized
        methods = [r.get("method") for r in mock_server.post_requests]
        assert "initialize" in methods
        assert "notifications/initialized" in methods
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_tool_listing_and_schema_registration(mock_server, transport):
    """Phase 3.2: Verify tools/list, tool namespacing (mcp__), and ToolRegistry bridging."""
    client = MCPSSEClient(
        name="finance",
        url="http://testserver/sse",
        transport=transport,
    )
    registry = ToolRegistry()

    try:
        connected = await client.connect()
        assert connected is True

        tool_ids = await client.register_tools(registry)
        expected_id = "mcp__finance__remote_calculator"
        assert expected_id in tool_ids
        assert expected_id in registry

        tool = registry.get_tool(expected_id)
        assert tool is not None
        assert isinstance(tool, MCPSSEBridgedTool)
        assert tool.tool_id == expected_id
        assert "[finance MCP]" in tool.description

        schema = tool.schema
        assert "expr" in schema.get("properties", {})
        assert "expr" in schema.get("required", [])
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_tool_execution_and_response_correlation(mock_server, transport):
    """Phase 3.3: Execute registered remote tool, verify POST payload and response correlation."""
    client = MCPSSEClient(
        name="math",
        url="http://testserver/sse",
        transport=transport,
    )
    registry = ToolRegistry()

    try:
        await client.connect()
        await client.register_tools(registry)

        tool_id = "mcp__math__remote_calculator"
        result: ToolResult = await registry.execute_async(tool_id, expr="10 * 4 + 2")

        assert result.success is True
        assert "Computed: 10 * 4 + 2 = 42" in str(result.output)

        # Verify request parameters
        call_req = [r for r in mock_server.post_requests if r.get("method") == "tools/call"]
        assert len(call_req) == 1
        assert call_req[0]["params"]["name"] == "remote_calculator"
        assert call_req[0]["params"]["arguments"] == {"expr": "10 * 4 + 2"}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_error_handling_and_reconnection(mock_server, transport):
    """Phase 3.4: Test JSON-RPC error translation, tool error reporting, and transient stream reconnection."""
    client = MCPSSEClient(
        name="robust_srv",
        url="http://testserver/sse",
        transport=transport,
        initial_reconnect_delay=0.01,
        max_reconnect_delay=0.05,
    )
    registry = ToolRegistry()

    try:
        await client.connect()
        await client.register_tools(registry)
        tool_id = "mcp__robust_srv__remote_calculator"

        # 1. Test JSON-RPC server error (-32603)
        res_rpc_err: ToolResult = await registry.execute_async(tool_id, expr="trigger_rpc_error")
        assert res_rpc_err.success is False
        assert "Internal computation failure" in str(res_rpc_err.error)

        # 2. Test tool execution error (isError = True)
        res_tool_err: ToolResult = await registry.execute_async(tool_id, expr="trigger_tool_error")
        assert res_tool_err.success is False
        assert "Division by zero error" in str(res_tool_err.error)

        # 3. Simulate sudden stream drop and automatic reconnection
        # Configure mock server to drop on 2nd connection
        mock_server.disconnect_on_connect_count = 2
        # Put close sentinel into current queue to force stream completion
        mock_server.sse_queue.put_nowait(b"__CLOSE__")

        # Give client's read loop a moment to detect and reconnect with backoff
        await asyncio.sleep(0.15)
        assert client._reconnect_count >= 1

        # Verify tool executions still work seamlessly post-reconnect
        res_after_reconnect: ToolResult = await registry.execute_async(tool_id, expr="7 * 6")
        assert res_after_reconnect.success is True
        assert "Computed: 7 * 6 = 42" in str(res_after_reconnect.output)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_server_ping_and_keepalive_handling(mock_server, transport):
    """Verify that SSE keepalive comments and ping events are safely consumed."""
    client = MCPSSEClient(
        name="ping_srv",
        url="http://testserver/sse",
        transport=transport,
    )

    try:
        await client.connect()
        # Emit comments and ping events
        mock_server.emit_comment("keepalive heart-beat")
        mock_server.emit_event("ping", "{}")
        mock_server.emit_event("keepalive", "")

        # Verify client is still running and responsive
        tools = await client.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "remote_calculator"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_resource_cleanup_and_task_termination(mock_server, transport):
    """Phase 3.5: Verify close() cleanly terminates background coroutines without task leaks."""
    client = MCPSSEClient(
        name="cleanup_srv",
        url="http://testserver/sse",
        transport=transport,
    )

    await client.connect()
    assert client._running is True
    assert client._listen_task is not None
    assert not client._listen_task.done()

    listen_task = client._listen_task
    await client.close()
    assert client._running is False
    assert listen_task.done()
    assert client._http_client is None

    # Invoking send_request on closed client must raise RuntimeError
    with pytest.raises(RuntimeError, match="is not running"):
        await client.send_request("tools/list")


@pytest.mark.asyncio
async def test_mcp_manager_sse_integration(mock_server, transport, tmp_path):
    """Verify MCPManager loads, starts, monitors, and unregisters SSE servers from config."""
    registry = ToolRegistry()
    manager = MCPManager(registry=registry)
    manager.config_path = tmp_path / "mcp.json"
    manager.config_path.write_text(json.dumps({"mcpServers": {}}))

    try:
        # Add SSE server configuration dynamically
        success = await manager.add_sse_server_config(
            name="cloud_agent",
            url="http://testserver/sse",
            headers={"Authorization": "Bearer secret-token"},
            transport=transport,
        )
        assert success is True

        status = manager.get_status()
        assert status["bridged_tools_count"] == 1
        servers = status["connected_servers"]
        assert len(servers) == 1
        assert servers[0]["name"] == "cloud_agent"
        assert servers[0]["type"] == "SSE"
        assert servers[0]["status"] == "ONLINE"
        assert servers[0]["url"] == "http://testserver/sse"

        # Verify tool is available in registry
        expected_tool_id = "mcp__cloud_agent__remote_calculator"
        assert expected_tool_id in registry

        # Stop server and verify cleanup
        stopped = await manager.stop_server("cloud_agent")
        assert stopped is True
        assert expected_tool_id not in registry
        assert manager.get_status()["bridged_tools_count"] == 0
    finally:
        await manager.stop()
