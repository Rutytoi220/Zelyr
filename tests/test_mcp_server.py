"""Focused unit and integration tests for Zelyr Model Context Protocol (MCP) Server Gateway."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List

import anyio
import pytest
import uvicorn
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
import mcp.types as types

from zelyr.db.memory import MemoryStore
from zelyr.mcp.server import MCPServer, ZelyrMCPServerGateway
from zelyr.tool_registry import ToolRegistry
from zelyr.tools import EchoTool
from zelyr.tools.core import BaseTool, ToolParameter, ToolResult

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


# ── Test Helpers & Fixtures ──────────────────────────────────────────────────


class MockCustomFailingTool(BaseTool):
    """Tool that deliberately raises an unhandled exception or returns an error."""

    def __init__(self):
        super().__init__(
            tool_id="mock_failing_tool",
            name="mock_failing_tool",
            description="Fails deliberately for error containment verification.",
        )
        self.add_parameter(ToolParameter(name="action", type="string", description="Action to perform", required=True))

    def execute(self, action: str) -> ToolResult:
        if action == "raise":
            raise RuntimeError("Deliberate unhandled explosion inside tool execution")
        return ToolResult(success=False, error="Deliberate structured tool failure")


async def create_connected_session(gateway: ZelyrMCPServerGateway):
    """Helper connecting an in-memory ClientSession directly to the gateway server."""
    server_to_client_send, server_to_client_recv = anyio.create_memory_object_stream[Any](20)
    client_to_server_send, client_to_server_recv = anyio.create_memory_object_stream[Any](20)

    tg = anyio.create_task_group()
    await tg.__aenter__()

    tg.start_soon(
        gateway.server.run,
        client_to_server_recv,
        server_to_client_send,
        gateway.server.create_initialization_options(),
    )

    client = ClientSession(server_to_client_recv, client_to_server_send)
    await client.__aenter__()
    return client, tg


async def close_connected_session(client: ClientSession, tg: anyio.abc.TaskGroup):
    """Cleanly teardown the connected session and its task group."""
    try:
        await client.__aexit__(None, None, None)
    except Exception:
        pass
    tg.cancel_scope.cancel()
    try:
        await tg.__aexit__(None, None, None)
    except Exception:
        pass


# ── Test Suite ───────────────────────────────────────────────────────────────


class TestMCPServer:
    """Complete test suite verifying MCPServer gateway functionality."""

    @pytest.mark.asyncio
    async def test_handshake_negotiation(self):
        """1. Handshake negotiation: verify initialize returns protocol version and advertised capabilities."""
        reg = ToolRegistry()
        server = MCPServer(registry=reg, server_name="zelyr", server_version="1.2.0")
        client, tg = await create_connected_session(server)
        try:
            init_res = await client.initialize()
            assert init_res.serverInfo.name == "zelyr"
            assert init_res.serverInfo.version == "1.2.0"
            assert init_res.protocolVersion is not None
            # Advertised capabilities
            assert init_res.capabilities.tools is not None
            assert init_res.capabilities.resources is not None
        finally:
            await close_connected_session(client, tg)

    @pytest.mark.asyncio
    async def test_initialized_notification_and_ping(self):
        """2. Client confirmation and keepalive: verify initialized notification and ping."""
        server = MCPServer()
        client, tg = await create_connected_session(server)
        try:
            await client.initialize()
            ping_res = await client.send_ping()
            assert ping_res is not None
        finally:
            await close_connected_session(client, tg)

    @pytest.mark.asyncio
    async def test_tool_enumeration(self):
        """3. Tool enumeration: verify tools/list returns active tools with valid JSON Schema."""
        reg = ToolRegistry()
        server = MCPServer(registry=reg)
        client, tg = await create_connected_session(server)
        try:
            await client.initialize()
            tools_resp = await client.list_tools()
            tools_by_name = {t.name: t for t in tools_resp.tools}

            assert "echo" in tools_by_name
            echo_tool = tools_by_name["echo"]
            assert echo_tool.description == "Echoes the input back to verify tool calls are working."
            schema = echo_tool.inputSchema
            assert schema.get("type") == "object"
            assert "text" in schema.get("properties", {})
            assert "text" in schema.get("required", [])
        finally:
            await close_connected_session(client, tg)

    @pytest.mark.asyncio
    async def test_tool_execution_builtin_echo(self):
        """4. Tool execution: verify tools/call for built-in echo executes cleanly."""
        reg = ToolRegistry()
        server = MCPServer(registry=reg)
        client, tg = await create_connected_session(server)
        try:
            await client.initialize()
            call_res = await client.call_tool("echo", {"text": "zelyr-mcp-verified-echo"})

            assert call_res.isError is False
            assert len(call_res.content) == 1
            assert call_res.content[0].type == "text"
            assert call_res.content[0].text == "zelyr-mcp-verified-echo"
        finally:
            await close_connected_session(client, tg)

    @pytest.mark.asyncio
    async def test_tool_failure_containment(self):
        """5. Tool failure containment: verify errors return structured isError: true without crashing."""
        reg = ToolRegistry()
        reg.register(MockCustomFailingTool())
        server = MCPServer(registry=reg)
        client, tg = await create_connected_session(server)
        try:
            await client.initialize()

            # Non-existent tool
            res_unknown = await client.call_tool("nonexistent_tool_id", {})
            assert res_unknown.isError is True
            assert "ToolNotFound" in res_unknown.content[0].text

            # Missing required arguments for echo
            res_missing_args = await client.call_tool("echo", {})
            assert res_missing_args.isError is True
            assert "required" in res_missing_args.content[0].text.lower()

            # Structured tool failure
            res_tool_err = await client.call_tool("mock_failing_tool", {"action": "fail"})
            assert res_tool_err.isError is True
            assert "Deliberate structured tool failure" in res_tool_err.content[0].text

            # Unhandled exception inside tool
            res_exc = await client.call_tool("mock_failing_tool", {"action": "raise"})
            assert res_exc.isError is True
            assert "Deliberate unhandled explosion" in res_exc.content[0].text

            # Confirm server loop stayed completely functional for subsequent calls
            res_healthy = await client.call_tool("echo", {"text": "recovered-and-healthy"})
            assert res_healthy.isError is False
            assert res_healthy.content[0].text == "recovered-and-healthy"
        finally:
            await close_connected_session(client, tg)

    def test_malformed_frames_error_codes(self):
        """6. Malformed frames: verify non-JSON and invalid frames return appropriate error codes."""
        cmd = [sys.executable, "-m", "zelyr.cli.main", "mcp", "serve", "--transport", "stdio"]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        try:
            # 1. Unknown JSON-RPC method -> code -32601 / -32600
            unknown_rpc = {
                "jsonrpc": "2.0",
                "id": 42,
                "method": "unrecognized/method",
                "params": {},
            }
            proc.stdin.write(json.dumps(unknown_rpc) + "\n")
            proc.stdin.flush()

            line_resp = proc.stdout.readline()
            data_resp = json.loads(line_resp.strip())
            assert data_resp.get("id") == 42
            assert "error" in data_resp
            assert data_resp["error"]["code"] in (-32600, -32601, -32602) or "error" in data_resp["error"]["message"].lower()

            # 2. Malformed non-JSON frame -> code -32700 (Parse Error)
            proc.stdin.write("MALFORMED_NON_JSON_RAW_STRING\n")
            proc.stdin.flush()

            line_parse_err = proc.stdout.readline()
            data_parse_err = json.loads(line_parse_err.strip())
            if "error" in data_parse_err:
                assert data_parse_err["error"]["code"] == -32700 or "parse" in data_parse_err["error"]["message"].lower()
            else:
                assert data_parse_err.get("method") == "notifications/message"

            proc.stdin.close()
            proc.wait(timeout=5.0)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2.0)

    @pytest.mark.asyncio
    async def test_resource_access_and_queries(self):
        """7. Resource access: verify resources/list and resources/read query memory store."""
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            store = MemoryStore(tmp.name)
            id1 = store.store("Developer prefers dark mode theme", category="preference", key="theme.mode")
            id2 = store.store("Linux kernel 6.8 on x86_64 architecture", category="environment", key="sys.kernel")

            server = MCPServer(memory_store=store, expose_memory=True)
            client, tg = await create_connected_session(server)
            try:
                await client.initialize()

                # 1. Listing resources
                res_list = await client.list_resources()
                uris = [str(r.uri) for r in res_list.resources]
                assert "zelyr://memory/active" in uris
                assert "zelyr://memory/recent" in uris
                assert "zelyr://memory/category/preference" in uris

                # 2. Reading active memories
                read_active = await client.read_resource("zelyr://memory/active")
                assert len(read_active.contents) == 1
                active_memories = json.loads(read_active.contents[0].text)
                assert len(active_memories) == 2

                # 3. Reading recent memories
                read_recent = await client.read_resource("zelyr://memory/recent")
                recent_memories = json.loads(read_recent.contents[0].text)
                assert len(recent_memories) == 2

                # 4. Reading by category
                read_cat = await client.read_resource("zelyr://memory/category/preference")
                cat_memories = json.loads(read_cat.contents[0].text)
                assert len(cat_memories) == 1
                assert cat_memories[0]["key"] == "theme.mode"

                # 5. Search query
                read_search = await client.read_resource("zelyr://memory/search?q=Linux")
                search_memories = json.loads(read_search.contents[0].text)
                assert len(search_memories) >= 1
                assert any("Linux" in m["content"] for m in search_memories)

                # 6. Invalid category
                with pytest.raises(Exception):
                    await client.read_resource("zelyr://memory/category/nonexistent_category_foo")

                # 7. Disallowed URI scheme
                with pytest.raises(Exception):
                    await client.read_resource("http://malicious.endpoint/steal")
            finally:
                await close_connected_session(client, tg)

    def test_stdio_stream_isolation(self):
        """8. Stdio stream isolation: verify stdout contains strictly valid JSON-RPC frames."""
        cmd = [sys.executable, "-m", "zelyr.cli.main", "mcp", "serve", "--transport", "stdio"]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        try:
            # 1. Send initialize
            init_req = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test-stdio-runner", "version": "1.0"},
                },
            }
            proc.stdin.write(json.dumps(init_req) + "\n")
            proc.stdin.flush()

            line_init = proc.stdout.readline()
            assert line_init, "Expected JSON-RPC line on stdout"
            # Strict stdout purity assertion: no escape codes, no banners
            assert not line_init.startswith("\x1b")
            data_init = json.loads(line_init.strip())
            assert data_init.get("jsonrpc") == "2.0"
            assert data_init.get("id") == 1
            assert data_init.get("result", {}).get("serverInfo", {}).get("name") == "zelyr"

            # 2. Send ping
            ping_req = {"jsonrpc": "2.0", "id": 2, "method": "ping"}
            proc.stdin.write(json.dumps(ping_req) + "\n")
            proc.stdin.flush()

            line_ping = proc.stdout.readline()
            data_ping = json.loads(line_ping.strip())
            assert data_ping.get("id") == 2
            assert "result" in data_ping

            # 3. Call built-in echo tool over stdio
            call_req = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "echo",
                    "arguments": {"text": "stdio-purity-echo-test"},
                },
            }
            proc.stdin.write(json.dumps(call_req) + "\n")
            proc.stdin.flush()

            line_call = proc.stdout.readline()
            data_call = json.loads(line_call.strip())
            assert data_call.get("id") == 3
            result = data_call.get("result", {})
            assert result.get("isError") is False
            assert result.get("content", [{}])[0].get("text") == "stdio-purity-echo-test"

            # 4. Clean shutdown on stdin close
            proc.stdin.close()
            exit_code = proc.wait(timeout=5.0)
            assert exit_code == 0
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2.0)

    def test_sse_transport_lifecycle(self):
        """9. SSE Transport: verify remote HTTP client connects over SSE, lists tools, and executes calls."""
        port = 19989
        server = MCPServer()
        app = server.create_asgi_app(transport="sse")

        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        uv_server = uvicorn.Server(config)

        server_thread = threading.Thread(target=uv_server.run, daemon=True)
        server_thread.start()
        time.sleep(0.5)

        async def run_sse_client():
            async with sse_client(f"http://127.0.0.1:{port}/sse") as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    init_res = await session.initialize()
                    assert init_res.serverInfo.name == "zelyr"

                    tools = await session.list_tools()
                    assert any(t.name == "echo" for t in tools.tools)

                    res = await session.call_tool("echo", {"text": "sse-echo-verified"})
                    assert res.isError is False
                    assert res.content[0].text == "sse-echo-verified"

        try:
            anyio.run(run_sse_client)
        finally:
            uv_server.should_exit = True
            server_thread.join(timeout=3.0)
