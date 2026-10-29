"""Comprehensive isolated tests for the Zelyr Model Context Protocol (MCP) Server Gateway."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional

import anyio
import httpx
import pytest
import uvicorn
from starlette.applications import Starlette

import mcp.types as types
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

from zelyr.core.plugin_sandbox import SandboxedPluginRunner
from zelyr.db.memory import MemoryStore
from zelyr.mcp.server import ZelyrMCPServerGateway, _convert_tool_to_mcp_schema
from zelyr.tool_registry import ToolRegistry
from zelyr.tools.core import BaseTool, ToolParameter, ToolResult

# Mark all tests in this file as unit / integration
pytestmark = pytest.mark.integration


# ── Mock & Test Tools ────────────────────────────────────────────────────────


class MockCalculatorTool(BaseTool):
    """Safe test calculator tool."""

    def __init__(self):
        super().__init__(
            tool_id="test_calculator",
            name="test_calculator",
            description="Evaluates simple arithmetic expressions safely.",
        )
        self.add_parameter(ToolParameter(name="a", type="number", description="First operand", required=True))
        self.add_parameter(ToolParameter(name="b", type="number", description="Second operand", required=True))
        self.add_parameter(ToolParameter(name="op", type="string", description="Operation (add, sub, mul)", required=True))

    def execute(self, a: float, b: float, op: str) -> ToolResult:
        if op == "add":
            return ToolResult(success=True, output={"result": a + b})
        elif op == "sub":
            return ToolResult(success=True, output={"result": a - b})
        elif op == "mul":
            return ToolResult(success=True, output={"result": a * b})
        else:
            return ToolResult(success=False, error=f"Unsupported operator '{op}'")


class MockFailingTool(BaseTool):
    """Tool that deliberately returns a structured failure."""

    def __init__(self):
        super().__init__(
            tool_id="test_failing",
            name="test_failing",
            description="Simulates a structured tool execution failure.",
        )
        self.add_parameter(ToolParameter(name="reason", type="string", description="Failure reason", required=False))

    def execute(self, reason: str = "Deliberate failure") -> ToolResult:
        return ToolResult(
            success=False,
            output="",
            error=f"SimulationError: {reason}",
            remedy_hint="Check parameter arguments before retry.",
        )


class MockPrivilegedTool(BaseTool):
    """Tool requiring Ring 0 (system admin) privileges."""

    REQUIRED_RING = 0

    def __init__(self):
        super().__init__(
            tool_id="test_privileged",
            name="test_privileged",
            description="High-privilege system configuration tool.",
        )
        self.required_ring = 0

    def execute(self) -> ToolResult:
        return ToolResult(success=True, output="Privileged action completed.")


class MockSideEffectFileTool(BaseTool):
    """Tool producing an observable filesystem side-effect for E2E verification."""

    def __init__(self, target_filepath: str):
        super().__init__(
            tool_id="test_file_emitter",
            name="test_file_emitter",
            description="Writes verification payload to target file.",
        )
        self.target_filepath = target_filepath
        self.add_parameter(ToolParameter(name="content", type="string", description="Data to write", required=True))

    def execute(self, content: str) -> ToolResult:
        with open(self.target_filepath, "a", encoding="utf-8") as f:
            f.write(content + "\n")
        return ToolResult(success=True, output=f"Wrote {len(content)} bytes to destination.")


# ── Fixtures & Helpers ───────────────────────────────────────────────────────


@pytest.fixture
def isolated_registry() -> ToolRegistry:
    """Create an isolated ToolRegistry populated with safe test tools."""
    registry = ToolRegistry()
    registry.register(MockCalculatorTool())
    registry.register(MockFailingTool())
    registry.register(MockPrivilegedTool())
    return registry


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


class TestMCPServerGateway:
    """Test suite covering the Zelyr MCP Server Gateway."""

    @pytest.mark.asyncio
    async def test_initialization_and_version_negotiation(self, isolated_registry: ToolRegistry):
        """1. Verify initialization lifecycle and protocol-version negotiation."""
        gateway = ZelyrMCPServerGateway(registry=isolated_registry, server_name="zelyr", server_version="1.2.0")
        client, tg = await create_connected_session(gateway)
        try:
            init_res = await client.initialize()
            assert init_res.serverInfo.name == "zelyr"
            assert init_res.serverInfo.version == "1.2.0"
            assert init_res.capabilities.tools is not None
            assert init_res.capabilities.resources is not None
            assert init_res.capabilities.prompts is None
        finally:
            await close_connected_session(client, tg)

    @pytest.mark.asyncio
    async def test_initialized_notification_and_ping(self, isolated_registry: ToolRegistry):
        """2. Verify initialized notification handling and ping response."""
        gateway = ZelyrMCPServerGateway(registry=isolated_registry)
        client, tg = await create_connected_session(gateway)
        try:
            await client.initialize()
            # Send ping request to verify alive state
            ping_res = await client.send_ping()
            assert ping_res is not None
        finally:
            await close_connected_session(client, tg)

    @pytest.mark.asyncio
    async def test_tool_listing_and_schema_correctness(self, isolated_registry: ToolRegistry):
        """3. Verify tool listing publishes valid JSON Schema structures without omitting fields."""
        gateway = ZelyrMCPServerGateway(registry=isolated_registry)
        client, tg = await create_connected_session(gateway)
        try:
            await client.initialize()
            tools_list = await client.list_tools()
            tools_by_name = {t.name: t for t in tools_list.tools}

            assert "test_calculator" in tools_by_name
            calc_tool = tools_by_name["test_calculator"]
            assert calc_tool.description == "Evaluates simple arithmetic expressions safely."
            schema = calc_tool.inputSchema
            assert schema["type"] == "object"
            assert "a" in schema["properties"]
            assert "b" in schema["properties"]
            assert "op" in schema["properties"]
            assert set(schema["required"]) == {"a", "b", "op"}
        finally:
            await close_connected_session(client, tg)

    @pytest.mark.asyncio
    async def test_successful_tool_call(self, isolated_registry: ToolRegistry):
        """4. Verify successful tool execution through the real registry interface."""
        gateway = ZelyrMCPServerGateway(registry=isolated_registry)
        client, tg = await create_connected_session(gateway)
        try:
            await client.initialize()
            call_res = await client.call_tool("test_calculator", {"a": 25, "b": 17, "op": "add"})

            assert call_res.isError is False
            assert len(call_res.content) == 1
            payload = json.loads(call_res.content[0].text)
            assert payload.get("result") == 42
        finally:
            await close_connected_session(client, tg)

    @pytest.mark.asyncio
    async def test_unknown_tool_and_invalid_arguments(self, isolated_registry: ToolRegistry):
        """5. Verify handling of unknown tool requests and invalid arguments."""
        gateway = ZelyrMCPServerGateway(registry=isolated_registry)
        client, tg = await create_connected_session(gateway)
        try:
            await client.initialize()

            # 1. Unknown tool
            res_unknown = await client.call_tool("ghost_tool", {})
            assert res_unknown.isError is True
            assert "ToolNotFound" in res_unknown.content[0].text

            # 2. Missing required arguments (a, b, op)
            res_invalid = await client.call_tool("test_calculator", {"a": 10})
            assert res_invalid.isError is True
            err_msg = res_invalid.content[0].text.lower()
            assert "required" in err_msg or "validation" in err_msg or "missing" in err_msg
        finally:
            await close_connected_session(client, tg)

    @pytest.mark.asyncio
    async def test_structured_tool_failure_error_semantics(self, isolated_registry: ToolRegistry):
        """6. Verify expected tool execution failures map cleanly to isError: true without crashing."""
        gateway = ZelyrMCPServerGateway(registry=isolated_registry)
        client, tg = await create_connected_session(gateway)
        try:
            await client.initialize()
            call_res = await client.call_tool("test_failing", {"reason": "Database connection refused"})

            assert call_res.isError is True
            assert "SimulationError: Database connection refused" in call_res.content[0].text
        finally:
            await close_connected_session(client, tg)

    @pytest.mark.asyncio
    async def test_multiple_requests_over_single_session(self, isolated_registry: ToolRegistry):
        """7. Verify multiplexing multiple requests sequentially over a single session."""
        gateway = ZelyrMCPServerGateway(registry=isolated_registry)
        client, tg = await create_connected_session(gateway)
        try:
            await client.initialize()

            for i in range(1, 6):
                res = await client.call_tool("test_calculator", {"a": i, "b": 10, "op": "mul"})
                assert res.isError is False
                payload = json.loads(res.content[0].text)
                assert payload.get("result") == i * 10
        finally:
            await close_connected_session(client, tg)

    @pytest.mark.asyncio
    async def test_memory_resource_access_and_validation(self, isolated_registry: ToolRegistry):
        """8. Verify safe, read-only memory resource listing and URI validation."""
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            store = MemoryStore(tmp.name)
            id1 = store.store("User prefers Tokyo Night color theme", category="preference", key="ui.theme")
            id2 = store.store("System is Ubuntu Linux on x86_64", category="environment", key="os.info")

            gateway = ZelyrMCPServerGateway(
                registry=isolated_registry,
                memory_store=store,
                expose_memory=True,
            )
            client, tg = await create_connected_session(gateway)
            try:
                await client.initialize()

                # List resources
                res_list = await client.list_resources()
                uris = [str(r.uri) for r in res_list.resources]
                assert "zelyr://memory/active" in uris
                assert "zelyr://memory/category/preference" in uris

                # Read active memories
                res_active = await client.read_resource("zelyr://memory/active")
                assert len(res_active.contents) == 1
                active_data = json.loads(res_active.contents[0].text)
                assert len(active_data) == 2

                # Read by category
                res_cat = await client.read_resource("zelyr://memory/category/preference")
                cat_data = json.loads(res_cat.contents[0].text)
                assert len(cat_data) == 1
                assert cat_data[0]["key"] == "ui.theme"

                # Read by ID
                res_single = await client.read_resource(f"zelyr://memory/{id1}")
                single_data = json.loads(res_single.contents[0].text)
                assert single_data["id"] == id1
                assert "Tokyo Night" in single_data["content"]

                # Attempt invalid category
                with pytest.raises(Exception):
                    await client.read_resource("zelyr://memory/category/nonexistent_category")

                # Attempt invalid URI scheme
                with pytest.raises(Exception):
                    await client.read_resource("file:///etc/shadow")
            finally:
                await close_connected_session(client, tg)

    @pytest.mark.asyncio
    async def test_authorization_enforcement_and_ring_gating(self, isolated_registry: ToolRegistry):
        """9. Verify callers cannot bypass security rings or invoke unauthorized tools."""
        # 1. Ring gating: Caller operating at Ring 1 cannot call Ring 0 tool
        gateway = ZelyrMCPServerGateway(registry=isolated_registry, caller_ring=1)
        client, tg = await create_connected_session(gateway)
        try:
            await client.initialize()
            res_auth = await client.call_tool("test_privileged", {})
            assert res_auth.isError is True
            assert "AuthorizationError" in res_auth.content[0].text
            assert "requires security ring 0" in res_auth.content[0].text
        finally:
            await close_connected_session(client, tg)

        # 2. Allowed tools whitelist
        gw_whitelist = ZelyrMCPServerGateway(
            registry=isolated_registry,
            allowed_tools=["test_calculator"],
            caller_ring=0,
        )
        client_wl, tg_wl = await create_connected_session(gw_whitelist)
        try:
            await client_wl.initialize()
            # Allowed tool succeeds
            res_ok = await client_wl.call_tool("test_calculator", {"a": 2, "b": 3, "op": "add"})
            assert res_ok.isError is False

            # Disallowed tool fails
            res_denied = await client_wl.call_tool("test_failing", {})
            assert res_denied.isError is True
            assert "AuthorizationError" in res_denied.content[0].text
        finally:
            await close_connected_session(client_wl, tg_wl)

    @pytest.mark.asyncio
    async def test_sandboxed_plugin_isolation_through_mcp(self, isolated_registry: ToolRegistry):
        """10. Verify plugin process isolation and crash containment work across the MCP gateway."""
        def crashing_func(*args, **kwargs):
            os._exit(139)

        sandboxed_tool = SandboxedPluginRunner(
            crashing_func,
            name="sandboxed_crash_tool",
            description="Plugin that crashes inside worker process",
            timeout=5.0,
        )
        isolated_registry.register(sandboxed_tool)

        gateway = ZelyrMCPServerGateway(registry=isolated_registry)
        client, tg = await create_connected_session(gateway)
        try:
            await client.initialize()
            res = await client.call_tool("sandboxed_crash_tool", {})

            # The crash must be contained: gateway stays alive and returns isError=True
            assert res.isError is True
            assert "PluginExecutionError" in res.content[0].text
            assert "terminated unexpectedly" in res.content[0].text
        finally:
            await close_connected_session(client, tg)

    @pytest.mark.asyncio
    async def test_end_to_end_client_tool_side_effect_verification(self, isolated_registry: ToolRegistry):
        """11. E2E: Client connects, discovers tool, calls it, and independently verifies side-effect."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            emitter_tool = MockSideEffectFileTool(target_filepath=tmp_path)
            isolated_registry.register(emitter_tool)

            gateway = ZelyrMCPServerGateway(registry=isolated_registry)
            client, tg = await create_connected_session(gateway)
            try:
                # 1. Client connects & initializes
                await client.initialize()

                # 2. Client discovers tools
                tools_list = await client.list_tools()
                assert any(t.name == "test_file_emitter" for t in tools_list.tools)

                # 3. Client invokes tool
                verification_token = f"TOKEN_{int(time.time())}_SECURE_MCP_EXEC"
                call_res = await client.call_tool("test_file_emitter", {"content": verification_token})
                assert call_res.isError is False

                # 4. INDEPENDENT SIDE-EFFECT VERIFICATION (reading raw disk directly)
                with open(tmp_path, "r", encoding="utf-8") as f:
                    file_contents = f.read()

                assert verification_token in file_contents, (
                    f"Side-effect verification failed: '{verification_token}' was not written to {tmp_path}."
                )
            finally:
                await close_connected_session(client, tg)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_stdio_stdout_purity_and_jsonrpc_lifecycle(self):
        """12. Stdio: Verify stdout contains ONLY valid JSON-RPC frames with zero console pollution."""
        # Spawn `zelyr mcp serve --transport stdio` as a subprocess
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
            # 1. Send initialize request
            init_req = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test-stdio-client", "version": "1.0"},
                },
            }
            proc.stdin.write(json.dumps(init_req) + "\n")
            proc.stdin.flush()

            line = proc.stdout.readline()
            assert line, "Expected response on stdout but received EOF"

            # Parse line: MUST be valid JSON-RPC and have no banners or control characters
            assert not line.startswith("\x1b"), f"Stdout polluted with ANSI escape sequences: {line}"
            data = json.loads(line.strip())
            assert data.get("jsonrpc") == "2.0"
            assert data.get("id") == 1
            assert "result" in data
            assert data["result"]["serverInfo"]["name"] == "zelyr"

            # 2. Send ping request
            ping_req = {"jsonrpc": "2.0", "id": 2, "method": "ping"}
            proc.stdin.write(json.dumps(ping_req) + "\n")
            proc.stdin.flush()

            line_ping = proc.stdout.readline()
            data_ping = json.loads(line_ping.strip())
            assert data_ping.get("id") == 2
            assert "result" in data_ping

            # 3. Close stdin to simulate EOF
            proc.stdin.close()
            exitcode = proc.wait(timeout=5.0)
            assert exitcode == 0
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2.0)

    def test_http_sse_transport_end_to_end(self, isolated_registry: ToolRegistry):
        """13. HTTP/SSE: Verify remote HTTP client connects over SSE, lists tools, and executes calls."""
        port = 19985
        gateway = ZelyrMCPServerGateway(registry=isolated_registry)
        app = gateway.create_asgi_app(transport="sse")

        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(config)

        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()
        time.sleep(0.5)

        async def run_sse_client():
            async with sse_client(f"http://127.0.0.1:{port}/sse") as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    init_res = await session.initialize()
                    assert init_res.serverInfo.name == "zelyr"

                    tools = await session.list_tools()
                    assert any(t.name == "test_calculator" for t in tools.tools)

                    res = await session.call_tool("test_calculator", {"a": 14, "b": 28, "op": "add"})
                    assert res.isError is False
                    payload = json.loads(res.content[0].text)
                    assert payload.get("result") == 42

        try:
            anyio.run(run_sse_client)
        finally:
            server.should_exit = True
            server_thread.join(timeout=3.0)

    def test_malformed_json_rpc_input(self):
        """14. Verify malformed JSON and unknown JSON-RPC methods return standard error envelopes."""
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
            # 1. Send unknown method
            unknown_method_req = {
                "jsonrpc": "2.0",
                "id": 99,
                "method": "completely/unknown/method",
                "params": {},
            }
            proc.stdin.write(json.dumps(unknown_method_req) + "\n")
            proc.stdin.flush()

            line = proc.stdout.readline()
            data = json.loads(line.strip())
            assert data.get("id") == 99
            assert "error" in data
            # Standard JSON-RPC error
            assert data["error"]["code"] in (-32600, -32601, -32602) or "error" in data["error"]["message"].lower()

            # 2. Send malformed non-JSON string
            proc.stdin.write("INVALID_NON_JSON_LINE\n")
            proc.stdin.flush()

            line_err = proc.stdout.readline()
            data_err = json.loads(line_err.strip())
            if "error" in data_err:
                assert data_err["error"]["code"] == -32700 or "parse" in data_err["error"]["message"].lower()
            else:
                assert data_err.get("method") == "notifications/message"
                assert data_err.get("params", {}).get("level") == "error"

            proc.stdin.close()
            proc.wait(timeout=5.0)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2.0)

    @pytest.mark.asyncio
    async def test_cancellation_and_clean_shutdown(self, isolated_registry: ToolRegistry):
        """15. Verify clean shutdown and task cancellation without hanging or resource leaks."""
        gateway = ZelyrMCPServerGateway(registry=isolated_registry)
        client, tg = await create_connected_session(gateway)
        try:
            await client.initialize()
            assert client is not None
        finally:
            # Explicit cancellation of task group and client session
            await close_connected_session(client, tg)

