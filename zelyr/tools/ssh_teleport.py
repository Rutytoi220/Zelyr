import subprocess
import logging
from typing import Any, Dict
from pydantic import BaseModel, Field
from zelyr.tools.core import BaseTool

logger = logging.getLogger(__name__)

class SSHTeleportSchema(BaseModel):
    target: str = Field(..., description="Target server SSH connection string, e.g., 'user@192.168.1.50'")
    command: str = Field(..., description="The shell command to execute remotely.")
    timeout: int = Field(5, description="Connection timeout in seconds.")

class SSHTeleportTool(BaseTool):
    """Zero-Trust SSH Teleportation Tool for executing tasks on remote nodes."""
    
    name = "ssh_teleport"
    description = "Execute arbitrary shell scripts securely across remote SSH hosts."
    schema = SSHTeleportSchema
    is_async = False
    
    def __call__(self, target: str, command: str, timeout: int = 5) -> Dict[str, Any]:
        logger.info(f"Initiating SSH Teleport to {target}...")
        
        try:
            # We enforce BatchMode to prevent interactive password prompts locking the agent
            cmd_args = [
                "ssh",
                "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"ConnectTimeout={timeout}",
                target,
                command
            ]
            
            res = subprocess.run(cmd_args, capture_output=True, text=True, timeout=timeout + 5)
            
            if res.returncode == 0:
                logger.info(f"SSH command succeeded on {target}")
                return {"status": "SUCCESS", "output": res.stdout}
            else:
                logger.error(f"SSH command failed on {target}: {res.stderr}")
                return {"status": "FAILED", "error": res.stderr}
                
        except subprocess.TimeoutExpired:
            return {"status": "FAILED", "error": f"SSH connection to {target} timed out."}
        except Exception as e:
            return {"status": "FAILED", "error": f"SSH Execution Error: {e}"}
