from zelyr.compat import data_file
import asyncio
import logging
from datetime import datetime, timezone
import aiosqlite
from typing import Optional
from zelyr.core.events import EventBus

logger = logging.getLogger(__name__)

class CronService:
    """Non-blocking background loop to execute scheduled tasks."""
    
    def __init__(self, event_bus: EventBus, db_path: Optional[str] = None):
        self._bus = event_bus
        if db_path is None:
            db_file = data_file("zelyr.db", "axiom.db")
            db_file.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(db_file)
        else:
            self.db_path = db_path
            
        self._running = False
        
    async def start(self):
        self._running = True
        logger.info("CronService started. Polling every 60 seconds.")
        while self._running:
            try:
                await self._check_tasks()
            except Exception as e:
                logger.error(f"Error in CronService polling loop: {e}")
            await asyncio.sleep(60)
            
    def stop(self):
        self._running = False
        logger.info("CronService stopped.")
        
    async def _check_tasks(self):
        # Fetch pending tasks whose execution time is <= now
        now_iso = datetime.now(timezone.utc).isoformat()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT id, prompt FROM scheduled_tasks WHERE status = 'pending' AND execution_time <= ?",
                    (now_iso,)
                ) as cursor:
                    tasks = await cursor.fetchall()
                    
                for task in tasks:
                    task_id = task["id"]
                    prompt = task["prompt"]
                    
                    logger.info(f"CronService executing scheduled task {task_id}: {prompt}")
                    
                    # Mark as completed
                    await db.execute(
                        "UPDATE scheduled_tasks SET status = 'completed' WHERE id = ?",
                        (task_id,)
                    )
                    await db.commit()
                    
                    # Inject into EventBus
                    self._bus.publish_sync("orchestrator.trigger", {
                        "prompt": prompt,
                        "source": "cron"
                    })
        except Exception as e:
            logger.error(f"CronService database error: {e}")
