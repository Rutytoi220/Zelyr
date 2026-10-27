import os
import httpx
from typing import Dict, Any, Optional
import logging
from zelyr.tools.core import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)

class FileTeleportTool(BaseTool):
    """Tool to teleport files (push/pull) across the Swarm mesh."""

    def __init__(self):
        super().__init__(
            tool_id="file_teleport",
            name="file_teleport",
            description="Push or pull files across the Swarm mesh using Tailscale IPs."
        )
        self.add_parameter(ToolParameter("host", "string", "The remote node IP address (Tailscale 100.x.x.x or LAN IP)."))
        self.add_parameter(ToolParameter("action", "string", "Action to perform: 'push' or 'pull'."))
        self.add_parameter(ToolParameter("local_path", "string", "Path to the local file to push, or destination to save pulled file."))
        self.add_parameter(ToolParameter("remote_filename", "string", "Name of the file in the remote sandbox (~/.zelyr/swarm_workspace/)."))

    async def execute(self, host: str, action: str, local_path: str, remote_filename: str) -> ToolResult:
        if action not in ("push", "pull"):
            return ToolResult(success=False, error="Action must be 'push' or 'pull'.")
            
        base_url = f"http://{host}:8000"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                if action == "push":
                    local_file = os.path.expanduser(local_path)
                    if not os.path.exists(local_file):
                        return ToolResult(success=False, error=f"Local file not found: {local_file}")
                        
                    with open(local_file, "rb") as f:
                        files = {"file": (os.path.basename(local_file), f, "application/octet-stream")}
                        response = await client.post(f"{base_url}/teleport/push", params={"filename": remote_filename}, files=files)
                        
                    if response.status_code == 200:
                        return ToolResult(success=True, output=response.json().get("message", "File pushed successfully."))
                    else:
                        return ToolResult(success=False, error=f"HTTP {response.status_code}: {response.text}")
                        
                elif action == "pull":
                    local_file = os.path.expanduser(local_path)
                    response = await client.get(f"{base_url}/teleport/pull", params={"filename": remote_filename})
                    
                    if response.status_code == 200:
                        # Stream to file
                        import aiofiles
                        async with aiofiles.open(local_file, "wb") as f:
                            async for chunk in response.aiter_bytes():
                                await f.write(chunk)
                        return ToolResult(success=True, output=f"File pulled successfully to {local_file}.")
                    else:
                        return ToolResult(success=False, error=f"HTTP {response.status_code}: {response.text}")
                        
        except httpx.TimeoutException:
            return ToolResult(success=False, error=f"Timeout communicating with {host}. The node might be offline or unreachable.")
        except httpx.RequestError as e:
            return ToolResult(success=False, error=f"Network error communicating with {host}: {e}")
        except Exception as e:
            logger.error(f"Teleport error: {e}")
            return ToolResult(success=False, error=f"Error communicating with {host}: {e}")
