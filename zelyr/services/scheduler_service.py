"""Autonomous Background Scheduler Service."""
import json
import logging
import os
from typing import Optional, Callable
import time
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from zelyr.gui.notifications import DesktopNotifier
from zelyr.core.events import EventBus
from zelyr.compat import config_file

logger = logging.getLogger(__name__)

class BackgroundSchedulerService:
    """Manages recurring cron/interval workflows."""

    def __init__(self, event_bus: EventBus):
        self.scheduler = BackgroundScheduler()
        self._bus = event_bus
        self.config_file = str(config_file("cron.json"))
        self._job_schedules = {}
        
    def start(self):
        """Starts the scheduler and loads persisted jobs."""
        if not self.scheduler.running:
            self.scheduler.start()
            self._load_jobs()
            logger.info("BackgroundSchedulerService started.")

    def stop(self):
        """Stops the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("BackgroundSchedulerService stopped.")

    def _execute_job(self, name: str, prompt: str):
        """The actual function called when a job triggers."""
        logger.info(f"Executing scheduled job: {name}")
        DesktopNotifier.notify(
            title=f"[Zelyr Job] {name}",
            body="Task started...",
            icon="document-open-recent"
        )
        
        # Fire off the Cron Swarm Engine asynchronously
        from zelyr.services.cron_swarm import CronSwarmEngine
        import asyncio
        engine = CronSwarmEngine(self._bus)
        
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(engine.execute_job(name, prompt))
        else:
            asyncio.run(engine.execute_job(name, prompt))

    def _save_jobs(self):
        """Save jobs to JSON."""
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        jobs_data = []
        for job in self.scheduler.get_jobs():
            trigger_type = 'cron' if isinstance(job.trigger, CronTrigger) else 'interval'
            schedule = self._job_schedules.get(job.id)
            
            if not schedule:
                if trigger_type == 'interval':
                    schedule = str(int(job.trigger.interval.total_seconds()))
                else:
                    schedule = "0 8 * * *" # Fallback

            jobs_data.append({
                "id": job.id,
                "name": job.name,
                "prompt": job.args[1],
                "trigger_type": trigger_type,
                "schedule": schedule,
                "paused": not job.next_run_time
            })
            
        with open(self.config_file, 'w') as f:
            json.dump(jobs_data, f, indent=4)

    def _load_jobs(self):
        """Load jobs from JSON."""
        if not os.path.exists(self.config_file):
            return
            
        try:
            with open(self.config_file, 'r') as f:
                jobs_data = json.load(f)
                
            jobs_list = jobs_data.get("jobs", [])
            for j in jobs_list:
                # Handle old schema or new schema
                trigger_type = j.get("trigger_type", "cron")
                schedule = j.get("cron", j.get("schedule", "0 8 * * *"))
                
                self.add_job(
                    name=j["name"],
                    prompt=j["prompt"],
                    trigger_type=trigger_type,
                    schedule=schedule,
                    save=False
                )
                if j.get("paused"):
                    self.pause_job(j.get("id", j["name"]))
                    
        except Exception as e:
            logger.error(f"Failed to load scheduled jobs: {e}")

    def add_job(self, name: str, prompt: str, trigger_type: str, schedule: str, save: bool = True):
        """
        Adds a job to the scheduler.
        """
        job_id = name.replace(" ", "_").lower()
        
        if self.scheduler.get_job(job_id):
            self.scheduler.remove_job(job_id)
            
        if trigger_type == 'cron':
            trigger = CronTrigger.from_crontab(schedule)
        elif trigger_type == 'interval':
            trigger = IntervalTrigger(seconds=int(schedule))
        else:
            raise ValueError(f"Unknown trigger_type: {trigger_type}")

        job = self.scheduler.add_job(
            func=self._execute_job,
            trigger=trigger,
            args=[name, prompt],
            id=job_id,
            name=name
        )
        self._job_schedules[job_id] = schedule
        
        logger.info(f"Added job '{name}' with {trigger_type} schedule: {schedule}")
        
        if save:
            self._save_jobs()

    def remove_job(self, job_id: str):
        if self.scheduler.get_job(job_id):
            self.scheduler.remove_job(job_id)
            self._save_jobs()

    def get_jobs(self) -> list:
        """Returns list of active jobs."""
        jobs = []
        for job in self.scheduler.get_jobs():
            trigger_type = 'cron' if isinstance(job.trigger, CronTrigger) else 'interval'
            schedule = self._job_schedules.get(job.id, str(job.trigger))
            jobs.append({
                "id": job.id,
                "name": job.name,
                "prompt": job.args[1],
                "trigger_type": trigger_type,
                "schedule": schedule,
                "next_run_time": str(job.next_run_time) if job.next_run_time else "Paused"
            })
        return jobs

    def pause_job(self, job_id: str):
        self.scheduler.pause_job(job_id)
        self._save_jobs()

    def resume_job(self, job_id: str):
        self.scheduler.resume_job(job_id)
        self._save_jobs()
