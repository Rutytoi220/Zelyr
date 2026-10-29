import logging
import asyncio
from mcp.server.fastmcp import FastMCP
from zelyr.engine.memory_tx import TransactionalMemoryManager
from zelyr.engine.consensus import SwarmConsensusEngine
from zelyr.core.engine import Engine

from zelyr.mcp.server import MCPServer, ZelyrMCPServerGateway

logger = logging.getLogger(__name__)

mcp = FastMCP("zelyr")

@mcp.tool()
def query_zelyr_memory(query: str, top_k: int = 3) -> str:
    """Queries TransactionalMemoryManager and returns relevant local RAG context."""
    logger.info(f"MCP Tool 'query_zelyr_memory' invoked with query: {query}")
    try:
        manager = TransactionalMemoryManager()
        # Ensure we run this in an async context properly if it requires one
        results = manager.engine.query_memory_sync(query, top_k=top_k)
        if not results:
            return "No relevant context found."
            
        formatted = "Found relevant context:\n\n"
        for i, res in enumerate(results):
            score = res.get('score', 0)
            text = res.get('payload', {}).get('text', '')
            if text:
                formatted += f"--- Result {i+1} (Score: {score:.2f}) ---\n{text}\n\n"
        return formatted
    except Exception as e:
        return f"Error querying memory: {str(e)}"

@mcp.tool()
def trigger_swarm_verification(code_snippet: str, language: str) -> str:
    """Submits code to SwarmConsensusEngine for syntax validation and returns the report."""
    logger.info(f"MCP Tool 'trigger_swarm_verification' invoked for {language}")
    if language.lower() != "python":
        return "Only python verification is supported via py_compile currently."
        
    try:
        consensus = SwarmConsensusEngine()
        is_valid, error = consensus._verify_syntax(code_snippet)
        if is_valid:
            return "Swarm Consensus: Verification PASSED. Code is syntactically valid."
        else:
            return f"Swarm Consensus: Verification FAILED.\nError Trace:\n{error}"
    except Exception as e:
        return f"Error during verification: {str(e)}"

@mcp.tool()
def inspect_security_audit(limit: int = 10) -> str:
    """Retrieves recent tool execution logs from AuditLedger."""
    logger.info(f"MCP Tool 'inspect_security_audit' invoked with limit: {limit}")
    try:
        from zelyr.engine.audit_ledger import AuditLedger
        ledger = AuditLedger()
        logs = ledger.get_recent_logs(limit)
        if not logs:
            return "No recent audit logs found."
            
        formatted = "Recent Security Audit Logs:\n\n"
        for log in logs:
            formatted += f"[{log.get('timestamp')}] Tool: {log.get('tool_name')} | Status: {log.get('status')}\n"
        return formatted
    except Exception as e:
        # Fallback if AuditLedger isn't perfectly matched to the above signature
        return f"Could not retrieve audit logs. Error: {str(e)}"

def run_mcp_server(transport: str = "stdio", **kwargs):
    """Start the MCP server using stdio or specified transport."""
    logger.info("Starting Zelyr MCP Server over %s...", transport)
    try:
        import anyio
        from zelyr.mcp.server import ZelyrMCPServerGateway

        gateway = ZelyrMCPServerGateway(**kwargs)
        if transport == "stdio":
            anyio.run(gateway.run_stdio_async)
        else:
            anyio.run(
                gateway.run_http_async,
                kwargs.get("host", "127.0.0.1"),
                kwargs.get("port", 8000),
                transport,
            )
    except Exception as exc:
        logger.warning("Falling back to legacy FastMCP runner: %s", exc)
        mcp.run()

if __name__ == "__main__":
    run_mcp_server()

# Legacy AXIOM API compatibility alias
query_axiom_memory = query_zelyr_memory

__all__ = [
    "run_mcp_server",
    "MCPServer",
    "ZelyrMCPServerGateway",
    "mcp",
    "query_zelyr_memory",
    "trigger_swarm_verification",
    "inspect_security_audit",
]
