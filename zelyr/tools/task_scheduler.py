from zelyr.compat import data_file
import uuid
import datetime
import logging
import re
import aiosqlite
from typing import Dict, Any
from zelyr.tools.core import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)

class TaskSchedulerTool(BaseTool):
    """Tool to schedule an autonomous task for a later time."""

    def __init__(self, db_path: str = None):
        super().__init__(
            tool_id="task_scheduler",
            name="task_scheduler",
            description="Schedule a task to run at a specific future time. Supports natural language like 'in 1 hour', 'in 30 minutes', or a cron expression."
        )
        if db_path is None:
            self.db_path = str(data_file("zelyr.db", "axiom.db"))
        else:
            self.db_path = db_path
            
        self.add_parameter(ToolParameter("schedule", "string", "Time delay (e.g., 'in 1 hour', 'in 15 minutes', 'in 30 seconds') or a cron string."))
        self.add_parameter(ToolParameter("prompt", "string", "The task description to execute."))

    def _parse_schedule(self, schedule: str) -> datetime.datetime:
        now = datetime.datetime.now(datetime.timezone.utc)
        
        # Simple "in X unit" parsing
        match = re.match(r'^in\s+(\d+)\s+(second|minute|hour|day)s?$', schedule.strip().lower())
        if match:
            val = int(match.group(1))
            unit = match.group(2)
            if unit == 'second':
                return now + datetime.timedelta(seconds=val)
            elif unit == 'minute':
                return now + datetime.timedelta(minutes=val)
            elif unit == 'hour':
                return now + datetime.timedelta(hours=val)
            elif unit == 'day':
                return now + datetime.timedelta(days=val)
                
        # If it's a cron string, we just evaluate the next run
        try:
            from croniter import croniter
            if croniter.is_valid(schedule):
                iter = croniter(schedule, now)
                return iter.get_next(datetime.datetime)
        except ImportError:
            pass
            
        # Fallback if unparseable
        raise ValueError(f"Could not parse schedule expression: {schedule}")

    async def execute(self, schedule: str, prompt: str) -> ToolResult:
        if not prompt:
            return ToolResult(success=False, error="Task prompt is required.")
            
        task_id = str(uuid.uuid4())
        
        try:
            execution_time = self._parse_schedule(schedule)
            execution_iso = execution_time.isoformat()
            
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT INTO scheduled_tasks (id, execution_time, prompt, status) VALUES (?, ?, ?, ?)",
                    (task_id, execution_iso, prompt, "pending")
                )
                await db.commit()
            return ToolResult(success=True, output=f"Task scheduled successfully. ID: {task_id}, Execution Time: {execution_iso}")
        except Exception as e:
            logger.error(f"Failed to schedule task: {e}")
            return ToolResult(success=False, error=str(e))
