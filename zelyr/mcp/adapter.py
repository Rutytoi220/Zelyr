import asyncio
import copy
import logging
from typing import Dict, Any, List, Optional
from zelyr.tools.core import BaseTool, ToolResult, ToolParameter
from zelyr.mcp.stdio_client import MCPStdioClientAsync

logger = logging.getLogger("zelyr.mcp.adapter")

class MCPBridgedToolAsync(BaseTool):
    """Wrapper that translates a BaseTool execution into an MCP tools/call request.
    Enforces a strict 30-second non-blocking Event Loop firewall.
    """

    is_external: bool = True

    def __init__(self, server_name: str, client: MCPStdioClientAsync, mcp_tool_def: dict):
        from zelyr.tools.external import sanitize_external_text, sanitize_source_endpoint

        self._server_name = server_name
        self._client = client
        self._mcp_tool_name = mcp_tool_def.get("name", "")
        self.is_external = True
        self.source_id = f"mcp:{server_name}"
        self.source_type = "mcp_stdio"
        self.external_tool_name = self._mcp_tool_name
        self.source_endpoint = sanitize_source_endpoint(getattr(client, "command", server_name))
        self.required_ring = 1

        tool_id = f"{server_name}_{self._mcp_tool_name}"
        desc = f"[{server_name} MCP] " + sanitize_external_text(mcp_tool_def.get("description", ""))

        super().__init__(tool_id=tool_id, name=tool_id, description=desc, requires_bridge=True)

        # Translate JSON Schema to ToolParameter while preserving full schema constraints
        input_schema = mcp_tool_def.get("inputSchema", {})
        self._preserved_input_schema: Optional[Dict[str, Any]] = (
            copy.deepcopy(input_schema)
            if isinstance(input_schema, dict) and input_schema
            else None
        )
        props = input_schema.get("properties", {}) if isinstance(input_schema, dict) else {}
        req = input_schema.get("required", []) if isinstance(input_schema, dict) else []

        if isinstance(props, dict):
            req_set = set(req) if isinstance(req, list) else set()
            for k, v in props.items():
                if not isinstance(v, dict):
                    continue
                self.add_parameter(ToolParameter(
                    name=k,
                    type=v.get("type", "string"),
                    description=sanitize_external_text(v.get("description", ""), max_len=200),
                    required=k in req_set,
                    default=v.get("default"),
                ))

    @property
    def schema(self) -> Dict[str, Any]:
        if self._preserved_input_schema is not None:
            return copy.deepcopy(self._preserved_input_schema)
        return super().schema

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
