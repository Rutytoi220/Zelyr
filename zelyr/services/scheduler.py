import logging
import asyncio
import time
from datetime import datetime
from croniter import croniter
from zelyr.memory.schedules import ScheduleDatabase
from zelyr.agents.orchestrator_agent import OrchestratorAgent
from zelyr.tool_registry import ToolRegistry
from zelyr.llm.universal_client import UniversalLLMClient
from zelyr.gui.notifications import DesktopNotifier
from zelyr.tools.plugin_loader import load_plugins

logger = logging.getLogger(__name__)

class TemporalService:
    """Async background daemon for the Zelyr Temporal Engine."""
    
    def __init__(self, event_bus=None):
        self.db = ScheduleDatabase()
        self.event_bus = event_bus
        self._running = False
        self._task = None
        
    def start(self, loop=None):
        if self._running:
            return
        self._running = True
        logger.info("[TemporalService] Starting semantic scheduler daemon.")
        loop = loop or asyncio.get_event_loop()
        
        if loop.is_running():
            asyncio.create_task(self.db.initialize())
            self._task = asyncio.create_task(self._daemon_loop())
        else:
            loop.run_until_complete(self.db.initialize())
            self._task = loop.create_task(self._daemon_loop())
            
    def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("[TemporalService] Stopped.")
        
    async def _daemon_loop(self):
        while self._running:
            try:
                now = time.time()
                schedules = await self.db.get_schedules()
                
                for sched in schedules:
                    if not sched["is_active"]:
                        continue
                        
                    cron_expr = sched["cron_expression"]
                    last_run = sched["last_run"]
                    
                    try:
                        # Fallback base time if it has never run
                        base_time = last_run if last_run > 0 else now - 60
                        cron = croniter(cron_expr, base_time)
                        next_run = cron.get_next(float)
                        
                        if now >= next_run:
                            logger.info(f"[TemporalService] Triggering scheduled task: {sched['user_prompt']}")
                            await self.db.update_last_run(sched["id"], now)
                            self._spawn_orchestrator(sched["user_prompt"])
                            
                    except Exception as e:
                        logger.error(f"[TemporalService] Invalid cron expression '{cron_expr}' for task {sched['id']}: {e}")
                        
            except Exception as e:
                logger.error(f"[TemporalService] Daemon loop error: {e}")
                
            await asyncio.sleep(60)
            
    def _spawn_orchestrator(self, prompt: str):
        DesktopNotifier.notify(
            title="[Zelyr Temporal Engine]",
            body=f"Running scheduled task: {prompt[:40]}...",
            icon="appointment-new"
        )
        
        if self.event_bus:
            self.event_bus.publish_sync("temporal.task.started", {"prompt": prompt})
            
        def _run_agent():
            try:
                llm = UniversalLLMClient()
                registry = ToolRegistry()
                load_plugins(registry)
                
                agent = OrchestratorAgent(registry=registry, bus=self.event_bus, llm=llm)
                logger.info(f"[TemporalService] Orchestrator started for prompt: {prompt}")
                
                result = agent.run(task=prompt, use_tools=True)
                
                if self.event_bus:
                    self.event_bus.publish_sync("temporal.task.completed", {"prompt": prompt, "success": result.success})
                
                logger.info(f"[TemporalService] Orchestrator finished: {result.success}")
            except Exception as e:
                logger.error(f"[TemporalService] Orchestrator execution failed: {e}")
                
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _run_agent)
