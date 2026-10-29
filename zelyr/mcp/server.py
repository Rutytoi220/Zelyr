"""Model Context Protocol (MCP) Server Gateway for Zelyr.

Exposes Zelyr's native tool registry and read-only memory resources to external
AI harnesses and client applications via standard stdio and HTTP/SSE transports.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Set, Union
from urllib.parse import urlparse

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
import mcp.server.stdio as mcp_stdio
from mcp.server.models import InitializationOptions

from zelyr.tool_registry import ToolRegistry
from zelyr.tools.core import BaseTool, ToolResult

logger = logging.getLogger("zelyr.mcp.server")

# Standard supported memory categories
VALID_MEMORY_CATEGORIES: Set[str] = {
    "capability",
    "environment",
    "homelab",
    "preference",
    "workflow",
}


def _convert_tool_to_mcp_schema(tool_id: str, tool: BaseTool) -> types.Tool:
    """Convert a Zelyr BaseTool instance into an official MCP Tool definition."""
    name = str(tool.name or tool_id or tool.tool_id).strip()
    description = str(tool.description or f"Zelyr tool: {name}").strip()

    # Resolve schema from BaseTool
    input_schema: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    if hasattr(tool, "schema") and isinstance(tool.schema, dict) and tool.schema:
        raw_schema = dict(tool.schema)
        schema_type = raw_schema.get("type", "object")
        props = raw_schema.get("properties", {})
        required = raw_schema.get("required", [])

        if not isinstance(props, dict):
            props = {}
        if not isinstance(required, list):
            required = []

        input_schema = {
            "type": schema_type if schema_type == "object" else "object",
            "properties": props,
            "required": required,
        }
        if "additionalProperties" in raw_schema:
            input_schema["additionalProperties"] = raw_schema["additionalProperties"]
    elif hasattr(tool, "parameters") and isinstance(tool.parameters, list) and tool.parameters:
        props = {}
        required = []
        for p in tool.parameters:
            p_name = getattr(p, "name", "")
            if not p_name:
                continue
            props[p_name] = {
                "type": getattr(p, "type", "string"),
                "description": getattr(p, "description", ""),
            }
            if getattr(p, "required", False):
                required.append(p_name)
        input_schema = {
            "type": "object",
            "properties": props,
            "required": required,
        }
    else:
        # Fallback inspection of execute signature
        exec_fn = getattr(tool, "execute", None)
        if callable(exec_fn):
            try:
                sig = inspect.signature(exec_fn)
                props = {}
                required = []
                for param_name, param in sig.parameters.items():
                    if param_name in ("self", "cls", "params", "kwargs", "arguments"):
                        continue
                    param_type = "string"
                    if param.annotation == int:
                        param_type = "integer"
                    elif param.annotation == float:
                        param_type = "number"
                    elif param.annotation == bool:
                        param_type = "boolean"
                    elif param.annotation in (dict, Dict):
                        param_type = "object"
                    elif param.annotation in (list, List):
                        param_type = "array"

                    props[param_name] = {"type": param_type}
                    if param.default is inspect.Parameter.empty:
                        required.append(param_name)
                input_schema = {
                    "type": "object",
                    "properties": props,
                    "required": required,
                }
            except Exception:
                pass

    return types.Tool(
        name=name,
        description=description,
        inputSchema=input_schema,
    )


class ZelyrMCPServerGateway:
    """Supervised MCP Server Gateway exposing Zelyr tools and memory."""

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        memory_store: Optional[Any] = None,
        caller_ring: int = 1,
        allowed_tools: Optional[List[str]] = None,
        expose_memory: bool = True,
        server_name: str = "zelyr",
        server_version: str = "1.2.0",
    ) -> None:
        self.registry = registry or ToolRegistry()
        self.memory_store = memory_store
        self.caller_ring = int(caller_ring)
        self.allowed_tools = set(allowed_tools) if allowed_tools is not None else None
        self.expose_memory = bool(expose_memory)
        self.server_name = server_name
        self.server_version = server_version

        # Low-level protocol server
        self.server = Server(server_name, version=server_version)
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register MCP protocol handlers for tools and resources."""

        # ── 1. Tool Discovery: tools/list ────────────────────────────────────
        @self.server.list_tools()
        async def list_tools() -> List[types.Tool]:
            tools_dict = dict(self.registry.list_tools())
            if hasattr(self.registry, "get_builtin_tools"):
                try:
                    for builtin_id in self.registry.get_builtin_tools():
                        if builtin_id not in tools_dict:
                            t = self.registry.get_tool(builtin_id)
                            if t is not None:
                                tools_dict[builtin_id] = t
                except Exception as exc:
                    logger.debug("Could not enumerate builtin tools: %s", exc)

            result: List[types.Tool] = []

            for tool_id, tool_inst in tools_dict.items():
                if self.allowed_tools is not None and tool_id not in self.allowed_tools:
                    continue
                try:
                    mcp_tool = _convert_tool_to_mcp_schema(tool_id, tool_inst)
                    result.append(mcp_tool)
                except Exception as exc:
                    logger.warning("Failed to convert tool '%s' to MCP schema: %s", tool_id, exc)

            return result

        # ── 2. Tool Execution: tools/call ────────────────────────────────────
        @self.server.call_tool()
        async def call_tool(
            name: str, arguments: Optional[Dict[str, Any]] = None
        ) -> types.CallToolResult:
            args = arguments or {}

            # Check if tool is allowed
            if self.allowed_tools is not None and name not in self.allowed_tools:
                return types.CallToolResult(
                    content=[
                        types.TextContent(
                            type="text",
                            text=f"AuthorizationError: Tool '{name}' is not in allowed tools list.",
                        )
                    ],
                    isError=True,
                )

            tool_inst = self.registry.get_tool(name)
            if tool_inst is None:
                return types.CallToolResult(
                    content=[
                        types.TextContent(
                            type="text",
                            text=f"ToolNotFound: Tool '{name}' is not registered in Zelyr.",
                        )
                    ],
                    isError=True,
                )

            # Security ring check
            required_ring = self.registry._get_tool_ring(name, tool_inst)
            if self.caller_ring > required_ring:
                return types.CallToolResult(
                    content=[
                        types.TextContent(
                            type="text",
                            text=(
                                f"AuthorizationError: Tool '{name}' requires security ring {required_ring}, "
                                f"but caller operates at ring {self.caller_ring}."
                            ),
                        )
                    ],
                    isError=True,
                )

            # Argument validation against required schema fields
            mcp_tool = _convert_tool_to_mcp_schema(name, tool_inst)
            required_fields = mcp_tool.inputSchema.get("required", [])
            missing_fields = [f for f in required_fields if f not in args]
            if missing_fields:
                return types.CallToolResult(
                    content=[
                        types.TextContent(
                            type="text",
                            text=(
                                f"InvalidArguments: Missing required argument(s) for tool '{name}': "
                                f"{', '.join(missing_fields)}"
                            ),
                        )
                    ],
                    isError=True,
                )

            # Execute tool through Zelyr registry (preserving process isolation for sandboxed plugins)
            try:
                result: ToolResult = await self.registry.execute_async(
                    name,
                    caller_ring=self.caller_ring,
                    allowed_tools=self.allowed_tools,
                    **args,
                )
            except Exception as exc:
                logger.error("Unhandled exception executing tool '%s': %s", name, exc, exc_info=True)
                return types.CallToolResult(
                    content=[
                        types.TextContent(
                            type="text",
                            text=f"ToolExecutionError: Unhandled exception during tool invocation: {exc}",
                        )
                    ],
                    isError=True,
                )

            # Format tool outcome into MCP CallToolResult envelope
            if result.success:
                if isinstance(result.output, (dict, list)):
                    output_text = json.dumps(result.output, indent=2, ensure_ascii=False)
                elif result.output is None:
                    output_text = ""
                else:
                    output_text = str(result.output)

                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=output_text)],
                    isError=False,
                )
            else:
                err_text = result.error or str(result.output) or "Tool execution failed."
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=err_text)],
                    isError=True,
                )

        # ── 3. Read-Only Memory Resources (Optional) ─────────────────────────
        if self.expose_memory:
            self._register_resource_handlers()

    def _register_resource_handlers(self) -> None:
        """Register safe read-only memory resource endpoints."""

        @self.server.list_resources()
        async def list_resources() -> List[types.Resource]:
            resources: List[types.Resource] = [
                types.Resource(
                    uri="zelyr://memory/active",  # type: ignore[arg-type]
                    name="Active Memories",
                    description="List of active persistent memory entries",
                    mimeType="application/json",
                ),
                types.Resource(
                    uri="zelyr://memory/recent",  # type: ignore[arg-type]
                    name="Recent Memories",
                    description="List of recently accessed or modified memory entries",
                    mimeType="application/json",
                ),
            ]
            for cat in sorted(VALID_MEMORY_CATEGORIES):
                resources.append(
                    types.Resource(
                        uri=f"zelyr://memory/category/{cat}",  # type: ignore[arg-type]
                        name=f"Memories: {cat}",
                        description=f"Active memories under category '{cat}'",
                        mimeType="application/json",
                    )
                )
            return resources

        @self.server.read_resource()
        async def read_resource(uri: Any) -> List[ReadResourceContents]:
            uri_str = str(uri).strip()
            parsed = urlparse(uri_str)

            if parsed.scheme != "zelyr":
                raise ValueError(f"Unsupported URI scheme: '{parsed.scheme}'. Only 'zelyr://' is permitted.")

            path = parsed.path.lstrip("/")
            netloc = parsed.netloc

            if netloc != "memory":
                raise ValueError(f"Unknown resource namespace '{netloc}'. Only 'zelyr://memory/...' is permitted.")

            if self.memory_store is None:
                try:
                    from zelyr.db.memory import MemoryStore

                    self.memory_store = MemoryStore()
                except Exception as exc:
                    raise RuntimeError(f"Memory storage subsystem is unavailable: {exc}") from exc

            # Route: zelyr://memory/active
            if path == "active" or not path:
                memories = self._query_active_memories(limit=100)
                payload = json.dumps(memories, indent=2, ensure_ascii=False)
                return [ReadResourceContents(content=payload, mime_type="application/json")]

            # Route: zelyr://memory/recent
            if path == "recent":
                memories = self._query_active_memories(limit=50)
                payload = json.dumps(memories, indent=2, ensure_ascii=False)
                return [ReadResourceContents(content=payload, mime_type="application/json")]

            # Route: zelyr://memory/search?q={query} or zelyr://memory/search/{query}
            if path == "search" or path.startswith("search/"):
                search_query = ""
                if path.startswith("search/"):
                    search_query = path[len("search/") :].strip()
                elif parsed.query:
                    import urllib.parse
                    qs = urllib.parse.parse_qs(parsed.query)
                    search_query = qs.get("q", [""])[0]

                results = self.memory_store.search(search_query, limit=20) if search_query else []
                safe_memories = [
                    {
                        "id": m.get("id"),
                        "category": m.get("category"),
                        "key": m.get("key"),
                        "content": m.get("content"),
                        "confidence": m.get("confidence"),
                        "created_at": m.get("created_at"),
                        "updated_at": m.get("updated_at"),
                    }
                    for m in results
                ]
                payload = json.dumps(safe_memories, indent=2, ensure_ascii=False)
                return [ReadResourceContents(content=payload, mime_type="application/json")]

            # Route: zelyr://memory/category/{category}
            if path.startswith("category/"):
                cat = path[len("category/") :].strip()
                if cat not in VALID_MEMORY_CATEGORIES:
                    raise ValueError(
                        f"Invalid memory category '{cat}'. Must be one of: {sorted(VALID_MEMORY_CATEGORIES)}"
                    )
                memories = self._query_active_memories(category=cat, limit=100)
                payload = json.dumps(memories, indent=2, ensure_ascii=False)
                return [ReadResourceContents(content=payload, mime_type="application/json")]

            # Route: zelyr://memory/{memory_id}
            if path.isdigit():
                mem_id = int(path)
                mem = self.memory_store.get(mem_id)
                if not mem or not mem.get("is_active", 1):
                    raise ValueError(f"Active memory record #{mem_id} not found.")
                safe_mem = {
                    "id": mem.get("id"),
                    "category": mem.get("category"),
                    "key": mem.get("key"),
                    "content": mem.get("content"),
                    "confidence": mem.get("confidence"),
                    "created_at": mem.get("created_at"),
                    "updated_at": mem.get("updated_at"),
                }
                payload = json.dumps(safe_mem, indent=2, ensure_ascii=False)
                return [ReadResourceContents(content=payload, mime_type="application/json")]

            raise ValueError(f"Invalid or unrecognized resource path: '{path}'")

    def _query_active_memories(
        self, category: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Safely fetch sanitized active memory records without exposing credentials or internal state."""
        if not self.memory_store:
            return []

        try:
            with self.memory_store._get_conn() as conn:
                if category:
                    cursor = conn.execute(
                        """
                        SELECT id, category, key, content, confidence, created_at, updated_at
                        FROM memories
                        WHERE is_active = 1 AND category = ?
                        ORDER BY updated_at DESC LIMIT ?
                        """,
                        (category, limit),
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT id, category, key, content, confidence, created_at, updated_at
                        FROM memories
                        WHERE is_active = 1
                        ORDER BY updated_at DESC LIMIT ?
                        """,
                        (limit,),
                    )

                results = []
                for row in cursor.fetchall():
                    results.append(
                        {
                            "id": row["id"],
                            "category": row["category"],
                            "key": row["key"],
                            "content": row["content"],
                            "confidence": row["confidence"],
                            "created_at": row["created_at"],
                            "updated_at": row["updated_at"],
                        }
                    )
                return results
        except Exception as exc:
            logger.warning("Failed to query active memories: %s", exc)
            return []

    # ── Transports ───────────────────────────────────────────────────────────

    async def run_stdio_async(self) -> None:
        """Run the MCP server over standard input/output with guaranteed stdout purity."""
        # Divert all logging to stderr so stdout remains reserved exclusively for JSON-RPC messages
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s] %(message)s")
        )
        root_logger.addHandler(stderr_handler)

        async with mcp_stdio.stdio_server() as (read_stream, write_stream):
            init_opts = self.server.create_initialization_options()
            await self.server.run(read_stream, write_stream, init_opts)

    def create_asgi_app(
        self,
        transport: str = "http",
        allowed_hosts: Optional[List[str]] = None,
    ) -> Any:
        """Create a Starlette ASGI application hosting SSE and Streamable HTTP endpoints."""
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse, Response
        from starlette.routing import Mount, Route
        from mcp.server.sse import SseServerTransport
        from mcp.server.streamable_http import StreamableHTTPServerTransport

        sse_transport = SseServerTransport("/messages/")

        async def handle_sse(request: Request) -> Response:
            async with sse_transport.connect_sse(
                request.scope, request.receive, request._send
            ) as (read_stream, write_stream):
                init_opts = self.server.create_initialization_options()
                await self.server.run(read_stream, write_stream, init_opts)
            return Response()

        async def handle_health(_request: Request) -> Response:
            return JSONResponse(
                {
                    "status": "online",
                    "server": self.server_name,
                    "version": self.server_version,
                    "transport": transport,
                    "tools_count": len(self.registry),
                }
            )

        routes = [
            Route("/health", endpoint=handle_health, methods=["GET"]),
            Route("/sse", endpoint=handle_sse, methods=["GET"]),
            Mount("/messages/", app=sse_transport.handle_post_message),
        ]

        app = Starlette(debug=False, routes=routes)
        return app

    async def run_http_async(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        transport: str = "http",
        log_level: str = "info",
    ) -> None:
        """Run the MCP server over HTTP/SSE with Uvicorn."""
        import uvicorn

        app = self.create_asgi_app(transport=transport)
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level=log_level,
            access_log=False,
        )
        server = uvicorn.Server(config)
        await server.serve()


def create_mcp_server(
    registry: Optional[ToolRegistry] = None,
    caller_ring: int = 1,
    allowed_tools: Optional[List[str]] = None,
    expose_memory: bool = True,
) -> ZelyrMCPServerGateway:
    """Convenience factory function creating a configured ZelyrMCPServerGateway."""
    return ZelyrMCPServerGateway(
        registry=registry,
        caller_ring=caller_ring,
        allowed_tools=allowed_tools,
        expose_memory=expose_memory,
    )


# Standard MCP Server alias
MCPServer = ZelyrMCPServerGateway

__all__ = [
    "ZelyrMCPServerGateway",
    "MCPServer",
    "create_mcp_server",
    "VALID_MEMORY_CATEGORIES",
]
