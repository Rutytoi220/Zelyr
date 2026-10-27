import os
import subprocess
import logging
from typing import Dict, Any

logger = logging.getLogger("zelyr.security")

class BwrapSandbox:
    def __init__(self, workspace_dir: str = "~/.local/share/zelyr/workspace"):
        self.workspace_dir = os.path.expanduser(workspace_dir)
        os.makedirs(self.workspace_dir, exist_ok=True)
    
    def execute(self, command: str, timeout: float = 60.0, allow_net: bool = False) -> Dict[str, Any]:
        """Execute a command securely inside a rootless bwrap sandbox."""
        logger.info(f"Sandboxing execution: {command}")
        
        bwrap_cmd = [
            "bwrap",
            "--ro-bind", "/usr", "/usr",
            "--ro-bind", "/bin", "/bin",
            "--ro-bind", "/lib", "/lib",
        ]
        
        # Some systems use /lib64
        if os.path.exists("/lib64"):
            bwrap_cmd.extend(["--ro-bind", "/lib64", "/lib64"])
            
        # Bind essential config for networking/DNS if needed, read-only
        if os.path.exists("/etc/resolv.conf"):
            bwrap_cmd.extend(["--ro-bind", "/etc/resolv.conf", "/etc/resolv.conf"])
        if os.path.exists("/etc/pki"):
            bwrap_cmd.extend(["--ro-bind", "/etc/pki", "/etc/pki"])
        if os.path.exists("/etc/ssl"):
            bwrap_cmd.extend(["--ro-bind", "/etc/ssl", "/etc/ssl"])

        # Create isolated tmp and bind workspace
        bwrap_cmd.extend([
            "--tmpfs", "/tmp",
            "--bind", self.workspace_dir, self.workspace_dir,
            "--dir", self.workspace_dir,
            "--chdir", self.workspace_dir,
            "--die-with-parent"
        ])
        
        if not allow_net:
            bwrap_cmd.append("--unshare-net")
            
        # Final command injection using bash
        bwrap_cmd.extend(["--", "/bin/bash", "-c", command])
        
        try:
            result = subprocess.run(bwrap_cmd, capture_output=True, text=True, timeout=timeout)
            
            # Intercept read-only filesystem errors to provide clean feedback
            if result.returncode != 0 and "Read-only file system" in result.stderr:
                logger.warning(f"Sandbox intercepted unauthorized write: {command}")
                return {
                    "exit_code": 1,
                    "stdout": result.stdout,
                    "stderr": "Permission Denied: Command attempted to modify a sandboxed file system outside of the workspace.",
                }
                
            return {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "exit_code": 124,
                "stdout": "",
                "stderr": f"Execution timed out after {timeout} seconds",
            }
        except FileNotFoundError:
            logger.error("bwrap is not installed on the system.")
            return {
                "exit_code": 127,
                "stdout": "",
                "stderr": "bwrap is not installed on the system. Cannot execute.",
            }
        except Exception as e:
            return {
                "exit_code": 1,
                "stdout": "",
                "stderr": str(e),
            }
