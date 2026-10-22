import logging
import asyncio
from axiom.agents.swarm.supervisor import SwarmSupervisor
from axiom.engine.audit_ledger import AuditLedger
from axiom.core.events import EventBus

logger = logging.getLogger(__name__)

class CronSwarmEngine:
    """Manages headless SwarmSupervisor execution for scheduled cron jobs."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.ledger = AuditLedger()
        
    async def execute_job(self, job_name: str, prompt: str, model_tier: str = "tier_2_local"):
        """Execute a cron job autonomously and log it."""
        logger.info(f"CronSwarmEngine: Starting job '{job_name}'")
        
        # Publish start telemetry
        self.event_bus.publish_sync(
            "telemetry.update",
            data={"message": f"[⏱️ Cron Swarm Started: {job_name}]"}
        )
        
        # Instantiate a detached supervisor
        from axiom.llm.universal_client import UniversalLLMClient
        # Assume model_tier overrides could be applied to UniversalLLMClient, but for now we just use it
        llm = UniversalLLMClient()
        supervisor = SwarmSupervisor(llm_client=llm, event_bus=self.event_bus)
        
        try:
            # We treat the cron prompt as a normal user prompt and let supervisor handle it
            # In a fully headless way, we can use the analyze_task & synthesize logic or orchestrator.
            # Here we just route it through the supervisor's synthesis to execute it.
            
            # Usually cron prompts don't have existing 'results', so we just ask the LLM to do it
            # For a real implementation, we could spawn the orchestrator, but for isolated cron:
            response = supervisor.synthesize_results(prompt, {})
            
            # Log successful completion
            self.ledger.log_execution(
                "CronSwarmEngine",
                "execute_cron_job",
                {"job_name": job_name, "response_preview": (response or "")[:100]},
                "MEDIUM",
                "SUCCESS",
            )
            
            self.event_bus.publish_sync(
                "telemetry.update",
                data={"message": f"[✅ Cron Swarm Finished: {job_name}]"}
            )
            
        except Exception as e:
            logger.error(f"CronSwarmEngine: Job '{job_name}' failed: {e}")
            self.ledger.log_execution(
                "CronSwarmEngine",
                "execute_cron_job",
                {"job_name": job_name, "error": str(e)},
                "MEDIUM",
                "FAILED",
            )
            self.event_bus.publish_sync(
                "telemetry.update",
                data={"message": f"[❌ Cron Swarm Failed: {job_name}]"}
            )
