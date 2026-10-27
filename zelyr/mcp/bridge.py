from zelyr.compat import config_dir as _zelyr_config_dir
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

from zelyr.tools.base import BaseTool, ToolResult
from zelyr.mcp.client import MCPStdioClient

logger = logging.getLogger("zelyr.mcp.bridge")

class MCPBridgedTool(BaseTool):
    """Wrapper that translates a BaseTool execution into an MCP tools/call request."""

    def __init__(self, server_name: str, client: MCPStdioClient, mcp_tool_def: dict):
        self._server_name = server_name
        self._client = client
        self._mcp_tool_name = mcp_tool_def.get("name", "")

        tool_id = f"{server_name}_{self._mcp_tool_name}"
        desc = f"[{server_name} MCP] " + mcp_tool_def.get("description", "")

        # Translate JSON Schema to our tool schema
        input_schema = mcp_tool_def.get("inputSchema", {})
        props = input_schema.get("properties", {})
        req = input_schema.get("required", [])

        # Note: In the older Zelyr system, parameters are sometimes stored directly or via `schema`.
        # I'll create a structured parameters list mapping
        self.schema_params = []
        for k, v in props.items():
            self.schema_params.append({
                "name": k,
                "type": v.get("type", "string"),
                "description": v.get("description", ""),
                "required": k in req
            })

        super().__init__(tool_id=tool_id, name=tool_id, description=desc)

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info["parameters"] = self.schema_params
        return info

    def execute(self, **kwargs) -> ToolResult:
        """Call the MCP tool remotely."""
        super().execute(**kwargs)

        resp = self._client.call_tool(self._mcp_tool_name, kwargs)
        if not resp.get("success", False):
            return ToolResult(success=False, error=resp.get("error", "Unknown MCP Error"))

        return ToolResult(success=True, output=resp.get("output", ""))


class MCPBridgeManager:
    """Manages external MCP clients and registers their tools with the Zelyr Registry."""

    def __init__(self, registry):
        self.registry = registry
        self.clients: Dict[str, MCPStdioClient] = {}
        self.bridged_tools: List[str] = []

        config_dir = _zelyr_config_dir()
        config_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = config_dir / "mcp_servers.json"

        self._ensure_config()

    def _ensure_config(self):
        if not self.config_path.exists() or self.config_path.stat().st_size == 0:
            self.config_path.write_text(json.dumps({"mcpServers": {}}, indent=2))

    def start(self):
        """Load servers from configuration and bridge their tools."""
        try:
            content = self.config_path.read_text().strip()
            if not content:
                config = {"mcpServers": {}}
                self.config_path.write_text(json.dumps(config, indent=2))
            else:
                config = json.loads(content)

            servers = config.get("mcpServers", {})
            for name, cfg in servers.items():
                if "command" in cfg:
                    self.add_server(name, cfg["command"], cfg.get("args", []))
        except json.JSONDecodeError:
            logger.warning("MCP config is invalid JSON. Resetting to default.")
            self.config_path.write_text(json.dumps({"mcpServers": {}}, indent=2))
        except Exception as e:
            logger.error(f"Failed to load MCP servers config: {e}")

    def add_server(self, name: str, command: str, args: List[str]) -> bool:
        """Connect to an MCP server and register its tools dynamically."""
        if name in self.clients:
            logger.warning(f"MCP Server {name} is already connected.")
            return True

        client = MCPStdioClient(name, command, args)
        if client.connect():
            self.clients[name] = client
            tools = client.list_tools()
            for t in tools:
                self._bridge_tool(name, client, t)
            return True
        return False

    def _bridge_tool(self, server_name: str, client: MCPStdioClient, tool_def: dict):
        bridged_tool = MCPBridgedTool(server_name, client, tool_def)

        # Register directly into the Zelyr tool registry
        if hasattr(self.registry, "register_tool"):
            self.registry.register_tool(bridged_tool.tool_id, bridged_tool)
        elif hasattr(self.registry, "add_tool"):
            self.registry.add_tool(bridged_tool)

        self.bridged_tools.append(bridged_tool.tool_id)
        logger.info(f"Bridged MCP Tool: {bridged_tool.tool_id}")

    def get_status(self) -> Dict[str, Any]:
        """Return the current bridge status."""
        server_info = []
        for name, client in self.clients.items():
            status = "ONLINE" if client._running else "OFFLINE"
            server_info.append({
                "name": name,
                "type": "STDIO",
                "status": status
            })

        return {
            "connected_servers": server_info,
            "bridged_tools_count": len(self.bridged_tools)
        }

    def stop(self):
        """Disconnect all clients."""
        for client in self.clients.values():
            client.disconnect()
        self.clients.clear()
