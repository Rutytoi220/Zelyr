from zelyr.compat import config_dir as _zelyr_config_dir
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional

from zelyr.mcp.stdio_client import MCPStdioClientAsync
from zelyr.mcp.adapter import MCPBridgedToolAsync

logger = logging.getLogger("zelyr.mcp.manager")

class MCPManager:
    """Manages dynamic MCP servers and registers tools with Zelyr at runtime."""

    def __init__(self, registry, event_bus=None):
        self.registry = registry
        self.event_bus = event_bus
        self.clients: Dict[str, MCPStdioClientAsync] = {}
        self.bridged_tools: Dict[str, List[str]] = {}

        config_dir = _zelyr_config_dir()
        config_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = config_dir / "mcp.json"

        self._ensure_config()

    def _ensure_config(self):
        if not self.config_path.exists() or self.config_path.stat().st_size == 0:
            self.config_path.write_text(json.dumps({"mcpServers": {}}, indent=2))

    async def start_all(self):
        """Load and start all configured servers."""
        try:
            content = self.config_path.read_text().strip()
            if not content:
                config = {"mcpServers": {}}
                self.config_path.write_text(json.dumps(config, indent=2))
            else:
                config = json.loads(content)

            servers = config.get("mcpServers", {})
            for name, cfg in servers.items():
                transport = cfg.get("transport", "")
                if transport == "sse" or ("url" in cfg and "command" not in cfg):
                    await self.start_sse_server(name, cfg["url"], headers=cfg.get("headers"))
                elif transport == "stdio" or "command" in cfg:
                    await self.start_server(name, cfg["command"], cfg.get("args", []))
        except json.JSONDecodeError:
            logger.warning("MCP config is invalid JSON. Resetting to default.")
            self.config_path.write_text(json.dumps({"mcpServers": {}}, indent=2))
        except Exception as e:
            logger.error(f"Failed to load MCP servers config: {e}")

    async def start_sse_server(
        self,
        name: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        transport: Optional[Any] = None,
    ) -> bool:
        """Start a remote MCP SSE client and register its tools dynamically."""
        if name in self.clients:
            logger.warning(f"MCP Server {name} is already running.")
            return True

        from zelyr.mcp.sse_client import MCPSSEClient

        client = MCPSSEClient(name=name, url=url, headers=headers, transport=transport)
        success = await client.connect()
        if success:
            self.clients[name] = client
            self.bridged_tools[name] = []

            tool_ids = await client.register_tools(self.registry)
            self.bridged_tools[name] = tool_ids

            if self.event_bus:
                from zelyr.core.events import Event
                try:
                    self.event_bus.publish(
                        Event(
                            event_type="mcp.server_started",
                            source="MCPManager",
                            data={"server": name, "transport": "sse", "tools_count": len(tool_ids)},
                        )
                    )
                except Exception:
                    pass
            return True
        return False

    async def start_server(self, name: str, command: str, args: List[str]) -> bool:
        """Start a specific MCP server and register its tools dynamically."""
        if name in self.clients:
            logger.warning(f"MCP Server {name} is already running.")
            return True

        client = MCPStdioClientAsync(name, command, args)
        success = await client.connect()
        if success:
            self.clients[name] = client
            self.bridged_tools[name] = []

            tools = await client.list_tools()
            for t in tools:
                self._bridge_tool(name, client, t)

            if self.event_bus:
                from zelyr.core.events import Event
                try:
                    self.event_bus.publish(Event(event_type="mcp.server_started", source="MCPManager", data={"server": name, "tools_count": len(tools)}))
                except Exception:
                    pass
            return True
        return False

    async def stop_server(self, name: str) -> bool:
        """Stop an MCP server and unregister its tools."""
        if name not in self.clients:
            return False

        client = self.clients[name]
        if hasattr(client, "close"):
            await client.close()
        elif hasattr(client, "disconnect"):
            await client.disconnect()

        # Unregister tools
        for tool_id in self.bridged_tools.get(name, []):
            if hasattr(self.registry, "unregister_tool"):
                self.registry.unregister_tool(tool_id)
            elif hasattr(self.registry, "unregister"):
                self.registry.unregister(tool_id)

        del self.clients[name]
        del self.bridged_tools[name]

        if self.event_bus:
            from zelyr.core.events import Event
            try:
                self.event_bus.publish(Event(event_type="mcp.server_stopped", source="MCPManager", data={"server": name}))
            except Exception:
                pass
        return True

    async def reload_all(self):
        """Stop all running servers and reload from config."""
        servers_to_stop = list(self.clients.keys())
        for name in servers_to_stop:
            await self.stop_server(name)

        await self.start_all()

    def _bridge_tool(self, server_name: str, client: MCPStdioClientAsync, tool_def: dict):
        bridged_tool = MCPBridgedToolAsync(server_name, client, tool_def)

        # Register directly into the Zelyr tool registry
        if hasattr(self.registry, "register_tool"):
            self.registry.register_tool(bridged_tool.tool_id, bridged_tool)
        elif hasattr(self.registry, "add_tool"):
            self.registry.add_tool(bridged_tool)

        self.bridged_tools[server_name].append(bridged_tool.tool_id)
        logger.info(f"Bridged MCP Tool: {bridged_tool.tool_id}")

    def get_status(self) -> Dict[str, Any]:
        """Return the current bridge status."""
        server_info = []
        for name, client in self.clients.items():
            is_running = getattr(client, "_running", False)
            status = "ONLINE" if is_running else "OFFLINE"
            if hasattr(client, "url"):
                server_info.append({
                    "name": name,
                    "type": "SSE",
                    "status": status,
                    "url": client.url,
                    "tools_count": len(self.bridged_tools.get(name, []))
                })
            else:
                server_info.append({
                    "name": name,
                    "type": "STDIO",
                    "status": status,
                    "command": getattr(client, "command", ""),
                    "args": getattr(client, "args", []),
                    "tools_count": len(self.bridged_tools.get(name, []))
                })

        return {
            "connected_servers": server_info,
            "bridged_tools_count": sum(len(t) for t in self.bridged_tools.values())
        }

    async def add_sse_server_config(
        self,
        name: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        transport: Optional[Any] = None,
    ) -> bool:
        """Save a new SSE server to mcp.json and start it dynamically."""
        content = self.config_path.read_text().strip()
        config = json.loads(content) if content else {"mcpServers": {}}

        server_entry: Dict[str, Any] = {
            "transport": "sse",
            "url": url,
        }
        if headers:
            server_entry["headers"] = headers

        config.setdefault("mcpServers", {})[name] = server_entry
        self.config_path.write_text(json.dumps(config, indent=2))
        success = await self.start_sse_server(name, url, headers=headers, transport=transport)
        if self.event_bus:
            from zelyr.core.events import Event
            self.event_bus.publish(
                Event(
                    event_type="mcp.servers.updated",
                    source="MCPManager",
                    data=self.get_status(),
                )
            )
        return success

    async def add_server_config(self, name: str, command: str, args: List[str]):
        """Save a new server to mcp.json and start it dynamically."""
        content = self.config_path.read_text().strip()
        config = json.loads(content) if content else {"mcpServers": {}}

        config.setdefault("mcpServers", {})[name] = {
            "command": command,
            "args": args
        }
        self.config_path.write_text(json.dumps(config, indent=2))
        await self.start_server(name, command, args)
        if self.event_bus:
            from zelyr.core.events import Event
            self.event_bus.publish(Event(event_type="mcp.servers.updated", source="MCPManager", data=self.get_status()))

    async def remove_server_config(self, name: str):
        """Remove a server from mcp.json and stop it dynamically."""
        content = self.config_path.read_text().strip()
        config = json.loads(content) if content else {"mcpServers": {}}

        if name in config.setdefault("mcpServers", {}):
            del config["mcpServers"][name]
            self.config_path.write_text(json.dumps(config, indent=2))

        await self.stop_server(name)
        if self.event_bus:
            from zelyr.core.events import Event
            self.event_bus.publish(Event(event_type="mcp.servers.updated", source="MCPManager", data=self.get_status()))

    async def broadcast_status(self):
        if self.event_bus:
            from zelyr.core.events import Event
            self.event_bus.publish(Event(event_type="mcp.servers.updated", source="MCPManager", data=self.get_status()))

    async def stop(self):
        """Disconnect all clients."""
        servers_to_stop = list(self.clients.keys())
        for name in servers_to_stop:
            await self.stop_server(name)
