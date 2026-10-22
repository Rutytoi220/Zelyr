import logging
import asyncio
import subprocess
from axiom.core.events import EventBus
from axiom.engine.audit_ledger import AuditLedger
from axiom.engine.memory_tx import TransactionalMemoryManager
from axiom.llm.universal_client import UniversalLLMClient
from axiom.agents.swarm.base_swarm import BaseSubagent

logger = logging.getLogger(__name__)

class HealerAgent(BaseSubagent):
    """Specialized agent with limited shell access to diagnose and heal the OS."""
    
    def __init__(self, llm_client=None):
        super().__init__(
            name="HealerAgent",
            description="Diagnoses OS crashes and issues safe remediation commands.",
            topic="os.incident.detected",
            llm_client=llm_client
        )
        self.memory = TransactionalMemoryManager()
        
    def prescribe_and_execute(self, reason: str, unit: str, context: list) -> str:
        """Use ReAct loop to diagnose and prescribe a fix."""
        logger.info(f"HealerAgent investigating {unit}: {reason}")
        
        # 1. Diagnose (check memory for similar issues)
        search_query = f"Resolved incidents for {unit} {reason}"
        mem_results = self.memory.engine.query_memory_sync(search_query, top_k=2)
        
        mem_context = ""
        if mem_results:
            mem_context = "Previous resolutions:\n" + "\n".join(r.get('payload', {}).get('text', '') for r in mem_results)
            
        # 2. Prescribe (ask LLM for a command)
        prompt = f"""
        OS Incident Detected: {reason}
        Unit: {unit}
        Context (last log entries): {context[-5:] if len(context) > 5 else context}
        {mem_context}
        
        You are the AXIOM Healer Agent. Prescribe a SINGLE terminal command to fix this (e.g. systemctl --user restart {unit}).
        Return ONLY the raw shell command to execute, no markdown, no explanation.
        """
        
        response = self.llm_client.generate([{"role": "user", "content": prompt}])
        cmd = response.get("content", "").strip()
        
        # Cleanup markdown if LLM misbehaved
        if cmd.startswith("```"):
            cmd = cmd.strip("`").replace("bash", "").replace("shell", "").strip()
            
        if not cmd:
            return "No command prescribed."
            
        # 3. Execute and Verify
        logger.info(f"HealerAgent executing prescribed fix: {cmd}")
        try:
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            if res.returncode == 0:
                return f"SUCCESS: {cmd} executed cleanly. Output: {res.stdout}"
            else:
                return f"FAILED: {cmd} returned {res.returncode}. Error: {res.stderr}"
        except Exception as e:
            return f"EXECUTION ERROR: {e}"


class AutonomousSelfHealer:
    """Listens for OS incidents and triggers the HealerAgent."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.ledger = AuditLedger()
        self.llm = UniversalLLMClient()
        self.event_bus.subscribe("os.incident.detected", self._on_incident)
        
    def _on_incident(self, event) -> None:
        """Handle detected OS incidents asynchronously."""
        data = event.data
        asyncio.create_task(self._remediate(data))
        
    async def _remediate(self, data: dict):
        reason = data.get("reason", "Unknown Incident")
        unit = data.get("unit", "unknown.service")
        message = data.get("message", "")
        
        logger.info(f"AutonomousSelfHealer initiating remediation for {unit}")
        
        # Instantiate agent and prescribe fix
        agent = HealerAgent(self.llm)
        
        # Run sync method in thread
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, agent.prescribe_and_execute, reason, unit, data.get("context", []))
        
        # Log to ledger
        status = "SUCCESS" if "SUCCESS" in result else "FAILED"
        self.ledger.log_execution(
            "AutonomousSelfHealer",
            "heal_unit",
            {"unit": unit, "reason": reason, "result": result},
            "HIGH",
            status,
        )
        
        # Emit Desktop notification
        try:
            summary = "🛡️ AXIOM Self-Healer"
            body = f"Autonomously responded to {unit} crash.\nStatus: {status}"
            subprocess.run(['notify-send', '-u', 'normal', summary, body], check=False)
        except Exception as e:
            logger.error(f"Failed to emit notify-send: {e}")
            
        # Update Health Radar via event bus
        self.event_bus.publish_sync(
            "os.incident.healed",
            data={
                "unit": unit,
                "reason": reason,
                "status": status,
                "details": result
            }
        )
