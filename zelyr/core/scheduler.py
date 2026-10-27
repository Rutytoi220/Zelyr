from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from zelyr.core.ipc import SYSTEM_BUS
from zelyr.core.pipeline import run_pipeline
from zelyr.db.memory import get_db_path
from datetime import datetime
import httpx
import uuid
import json

db_url = f"sqlite:///{get_db_path()}"
jobstores = {
    'default': SQLAlchemyJobStore(url=db_url)
}
job_defaults = {
    'misfire_grace_time': 3600
}

scheduler = AsyncIOScheduler(jobstores=jobstores, job_defaults=job_defaults)
_is_started = False

def _ensure_started():
    global _is_started
    if not _is_started:
        scheduler.start()
        _is_started = True

async def execute_scheduled_task(payload_type: str, payload: str | list):
    try:
        if payload_type == "pipeline":
            # payload is an array of pipeline stages
            result = await run_pipeline(payload, "")
            res_str = json.dumps(result, indent=2)
            await SYSTEM_BUS.put(f"[Cron] Pipeline executed. Result:\n{res_str}")
            
        elif payload_type == "agent_prompt":
            # payload is a prompt instruction string
            async with httpx.AsyncClient(timeout=300.0) as client:
                req_payload = {
                    "model": "laguna-xs-2.1",
                    "messages": [
                        {"role": "system", "content": "You are a scheduled background agent. Execute the prompt and return only the requested output."},
                        {"role": "user", "content": payload}
                    ],
                    "stream": False
                }
                resp = await client.post(
                    "http://127.0.0.1:11434/v1/chat/completions",
                    json=req_payload,
                    headers={"X-Protection-Ring": "3"}
                )
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices", [])
                
                content = "Error: Agent returned no response."
                if choices:
                    msg = choices[0].get("message", {})
                    content = msg.get("content", "")
                    if not content:
                        content = msg.get("reasoning_content", "")
                    content = content.strip()
                
                await SYSTEM_BUS.put(f"[Cron] Agent executed. Result:\n{content}")
    except Exception as e:
        await SYSTEM_BUS.put(f"[Cron] Task failed: {e}")

def schedule_job(cron_expression: str, payload_type: str, payload: str | list) -> str:
    _ensure_started()
    job_id = f"cron_{uuid.uuid4().hex[:8]}"
    
    # Check if it's an ISO timestamp
    is_date = False
    try:
        dt = datetime.fromisoformat(cron_expression.replace("Z", "+00:00"))
        trigger = DateTrigger(run_date=dt)
        is_date = True
    except ValueError:
        trigger = CronTrigger.from_crontab(cron_expression)

    scheduler.add_job(
        execute_scheduled_task,
        trigger=trigger,
        args=[payload_type, payload],
        id=job_id,
        name=f"{payload_type} task"
    )
    
    return job_id

def get_active_jobs():
    _ensure_started()
    return scheduler.get_jobs()
