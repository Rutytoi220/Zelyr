import os
import shlex
import shutil
import logging
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

@dataclass
class SandboxPolicy:
    """Defines the execution boundaries for an untrusted payload."""
    workspace_dir: str = "/tmp/zelyr_sandbox"
    require_network: bool = False
    allow_home_read: bool = False
    bind_paths: List[str] = field(default_factory=list)
    timeout_seconds: int = 300

class SandboxRunner:
    """A clean abstraction for safely executing commands within a bwrap boundary."""
    
    def __init__(self, policy: SandboxPolicy):
        self.policy = policy
        self._has_bwrap = shutil.which("bwrap") is not None

    def _build_command(self, command: str, args: List[str]) -> List[str]:
        os.makedirs(self.policy.workspace_dir, exist_ok=True)
        if not self._has_bwrap:
            logger.warning("DEGRADED_SECURITY: bwrap is missing. Falling back to native execution.")
            try:
                from zelyr.core.events import EventBus
                # Optionally warn via bus if initialized somewhere else
            except ImportError:
                pass
            return [command] + args

        bwrap_cmd = [
            "bwrap",
            "--ro-bind", "/", "/",
            "--dev", "/dev",
            "--proc", "/proc",
            "--die-with-parent",
            "--chdir", self.policy.workspace_dir,
            "--bind", self.policy.workspace_dir, self.policy.workspace_dir
        ]

        if not self.policy.require_network:
            bwrap_cmd.append("--unshare-net")
            
        for path in self.policy.bind_paths:
            # Blindly bind mounting provided paths; caller must validate them.
            bwrap_cmd.extend(["--bind", path, path])

        if self.policy.allow_home_read:
            home = str(os.path.expanduser("~"))
            bwrap_cmd.extend(["--ro-bind", home, home])

        bwrap_cmd.extend(["--", command] + args)
        return bwrap_cmd

    async def run_async(self, command: str, *args: str) -> Tuple[int, str, str]:
        """
        Executes the command asynchronously under the defined SandboxPolicy.
        Returns (exit_code, stdout, stderr).
        """
        full_args = list(args)
        cmd_list = self._build_command(command, full_args)
        
        # If running inside bwrap, we execute it. 
        # If it's bash -c "...", we format it correctly.
        
        # If we have a single string shell command, we must format it for shell
        if len(args) == 0 and " " in command and not self._has_bwrap:
            # It's a raw shell string being executed un-sandboxed
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.policy.workspace_dir
            )
        else:
            proc = await asyncio.create_subprocess_exec(
                *cmd_list,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.policy.workspace_dir if not self._has_bwrap else None
            )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), 
                timeout=self.policy.timeout_seconds
            )
            return proc.returncode, stdout_bytes.decode(errors='replace'), stderr_bytes.decode(errors='replace')
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except (OSError, ProcessLookupError):
                pass
            return -1, "", f"Execution timed out after {self.policy.timeout_seconds} seconds."

    def run_bash_script(self, script_body: str) -> str:
        """Helper backward-compatibility wrapper used for converting string commands."""
        # For legacy `wrap_bash_command` compatibility if still needed
        cmd_list = self._build_command("bash", ["-c", script_body])
        return " ".join(shlex.quote(a) for a in cmd_list)

# Backward compatibility wrapper for old references
def wrap_bash_command(command: str, require_network: bool = False, workspace_dir: str = "/tmp/zelyr_sandbox") -> str:
    policy = SandboxPolicy(workspace_dir=workspace_dir, require_network=require_network)
    runner = SandboxRunner(policy)
    return runner.run_bash_script(command)
