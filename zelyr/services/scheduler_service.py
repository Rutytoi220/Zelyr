"""Autonomous Background Scheduler Service.

Integrates RoutineEngine with the Zelyr daemon and event bus,
supporting both cron and interval recurring workflows.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from zelyr.core.events import EventBus
from zelyr.core.routine import RoutineEngine

logger = logging.getLogger(__name__)


class BackgroundSchedulerService:
    """Manages recurring cron/interval workflows via RoutineEngine."""

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        db_path: Optional[Union[str, Path]] = None,
    ):
        self._bus = event_bus
        self.routine_engine = RoutineEngine(
            db_path=db_path,
            event_bus=event_bus,
            check_interval_seconds=2.0,
        )

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Starts the scheduler engine."""
        self.routine_engine.start(loop=loop)
        logger.info("BackgroundSchedulerService started.")

    async def stop_async(self) -> None:
        """Stops the scheduler engine asynchronously."""
        await self.routine_engine.stop()
        logger.info("BackgroundSchedulerService stopped.")

    def stop(self) -> None:
        """Stops the scheduler engine."""
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.create_task(self.routine_engine.stop())
                return
        except RuntimeError:
            pass
        asyncio.run(self.routine_engine.stop())
        logger.info("BackgroundSchedulerService stopped.")

    def add_job(
        self,
        name: str,
        prompt: str,
        trigger_type: str,
        schedule: str,
        save: bool = True,
    ) -> str:
        """Adds a job to the scheduler."""
        routine_id = self.routine_engine.add_routine(
            name_or_prompt=name,
            schedule_expr=schedule,
            schedule_type=trigger_type,
            action_type="prompt",
            action_payload=prompt,
            description=f"Job: {name}",
        )
        logger.info("Added job '%s' with %s schedule: %s", name, trigger_type, schedule)
        return routine_id

    def remove_job(self, job_id: str) -> bool:
        """Removes a job by ID or name."""
        return self.routine_engine.delete_routine(job_id)

    def get_jobs(self) -> List[Dict[str, Any]]:
        """Returns list of active/configured jobs."""
        routines = self.routine_engine.list_routines(include_disabled=True)
        jobs = []
        for r in routines:
            jobs.append(
                {
                    "id": r["id"],
                    "name": r["name"],
                    "prompt": (
                        r["action_payload"].get("prompt")
                        if isinstance(r["action_payload"], dict)
                        else str(r["action_payload"])
                    ),
                    "trigger_type": r["schedule_type"],
                    "schedule": r["schedule_expr"],
                    "enabled": bool(r.get("enabled", 1)),
                    "next_run_time": (
                        str(r["next_run_at"]) if r["enabled"] and r["next_run_at"] else "Paused"
                    ),
                    "last_run": r.get("last_run_at"),
                }
            )
        return jobs

    def pause_job(self, job_id: str) -> bool:
        """Pause a job."""
        return self.routine_engine.db.sync_toggle_routine(job_id, False)

    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job."""
        return self.routine_engine.db.sync_toggle_routine(job_id, True)
