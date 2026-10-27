import logging
import asyncio
from typing import Dict, Any, List, Optional
from zelyr.tools.core import BaseTool, ToolResult, ToolParameter
from zelyr.mcp.stdio_client import MCPStdioClientAsync

logger = logging.getLogger("zelyr.mcp.adapter")

class MCPBridgedToolAsync(BaseTool):
    """Wrapper that translates a BaseTool execution into an MCP tools/call request.
    Enforces a strict 30-second non-blocking Event Loop firewall.
    """

    def __init__(self, server_name: str, client: MCPStdioClientAsync, mcp_tool_def: dict):
        self._server_name = server_name
        self._client = client
        self._mcp_tool_name = mcp_tool_def.get("name", "")

        tool_id = f"{server_name}_{self._mcp_tool_name}"
        desc = f"[{server_name} MCP] " + mcp_tool_def.get("description", "")

        super().__init__(tool_id=tool_id, name=tool_id, description=desc)

        # Translate JSON Schema to ToolParameter
        input_schema = mcp_tool_def.get("inputSchema", {})
        props = input_schema.get("properties", {})
        req = input_schema.get("required", [])

        for k, v in props.items():
            self.add_parameter(ToolParameter(
                name=k,
                type=v.get("type", "string"),
                description=v.get("description", ""),
                required=k in req
            ))

    async def execute(self, **kwargs) -> ToolResult:
        """Call the MCP tool remotely with strict timeout gating."""
        try:
            # Enforce 30-second Event Loop Firewall
            resp = await asyncio.wait_for(
                self._client.call_tool(self._mcp_tool_name, kwargs, timeout=30.0),
                timeout=30.0
            )

            if not resp.get("success", False):
                return ToolResult(success=False, error=resp.get("error", "Unknown MCP Error"))

            return ToolResult(success=True, output=resp.get("output", ""))

        except asyncio.TimeoutError:
            error_msg = f"Event Loop Firewall: MCP Tool {self.tool_id} timed out after 30 seconds."
            logger.error(error_msg)
            return ToolResult(success=False, error=error_msg)
        except Exception as e:
            logger.error(f"Failed to execute MCP tool {self.tool_id}: {e}")
            return ToolResult(success=False, error=str(e))
