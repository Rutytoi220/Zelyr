import asyncio
import logging
from zelyr.tools.core import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)

class SystemAdminTool(BaseTool):
    """
    A tool that allows the Swarm Node LLM to securely execute arbitrary bash commands 
    on the host Linux machine, with timeout protections and full stdout/stderr capture.
    """

    def __init__(self):
        super().__init__(
            tool_id="system_admin",
            name="SystemAdminTool",
            description=(
                "You are running as an admin user on a Linux system. Use this tool to execute bash commands, "
                "install software (always use non-interactive flags like -y or DEBIAN_FRONTEND=noninteractive), "
                "or read files. You will receive the terminal output. Use this to administer the system."
            )
        )
        self.add_parameter(ToolParameter(
            name="bash_command",
            type="string",
            description="The exact bash command to execute in the shell."
        ))

    async def execute(self, bash_command: str) -> ToolResult:
        logger.info(f"SystemAdminTool executing: {bash_command}")
        
        from zelyr.core.sandbox import SandboxRunner, SandboxPolicy
        policy = SandboxPolicy(require_network=True, timeout_seconds=60)
        runner = SandboxRunner(policy)
        
        try:
            returncode, out_str, err_str = await runner.run_async("bash", "-c", bash_command)
            
            if returncode == 0:
                output_msg = out_str if out_str else "(Command succeeded with no output)"
                if err_str:
                    output_msg += f"\n\nStandard Error output:\n{err_str}"
                return ToolResult(success=True, output=output_msg)
            else:
                err_msg = f"Command failed with exit code {returncode}."
                if out_str:
                    err_msg += f"\n\nStandard Output:\n{out_str}"
                if err_str:
                    err_msg += f"\n\nStandard Error:\n{err_str}"
                return ToolResult(success=False, error=err_msg)
                
        except asyncio.TimeoutError:
            logger.warning(f"SystemAdminTool execution timed out after 60s: {bash_command}")
            return ToolResult(
                success=False, 
                error="Execution timed out after 60 seconds. The command likely hung waiting for interactive input or took too long to complete."
            )
        except Exception as e:
            logger.error(f"SystemAdminTool execution error: {e}")
            return ToolResult(success=False, error=f"Execution error: {str(e)}")
