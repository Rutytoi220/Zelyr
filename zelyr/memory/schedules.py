from zelyr.compat import data_dir as _zelyr_data_dir
import asyncio
import logging
import uuid
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import aiosqlite
from zelyr.memory.db import MemoryDatabaseManager

logger = logging.getLogger(__name__)

class ScheduleDatabase:
    """Async SQLite database for tracking scheduled AI tasks."""
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            data_dir = _zelyr_data_dir()
            data_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(data_dir / "schedules.db")
        else:
            self.db_path = db_path
            
        self._db = None

    async def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            self._db_mgr = await MemoryDatabaseManager.get_instance(self.db_path)
            self._db = await self._db_mgr.get_connection()
        return self._db

    async def initialize(self) -> None:
        """Create the schedules table if it doesn't exist."""
        try:
            db = await self._conn()
            async with db.execute(
                """
                CREATE TABLE IF NOT EXISTS schedules (
                    id TEXT PRIMARY KEY,
                    user_prompt TEXT,
                    cron_expression TEXT,
                    last_run REAL,
                    is_active INTEGER
                )
                """
            ):
                pass
            await db.commit()
        except Exception as e:
            logger.error(f"Failed to initialize ScheduleDatabase: {e}")

    async def add_schedule(self, user_prompt: str, cron_expression: str) -> str:
        """Create a new schedule and return its ID."""
        schedule_id = str(uuid.uuid4())
        
        try:
            db = await self._conn()
            async with db.execute(
                "INSERT INTO schedules (id, user_prompt, cron_expression, last_run, is_active) VALUES (?, ?, ?, ?, ?)",
                (schedule_id, user_prompt, cron_expression, 0.0, 1)
            ):
                pass
            await db.commit()
            return schedule_id
        except Exception as e:
            logger.error(f"Failed to create schedule: {e}")
            return schedule_id

    async def get_schedules(self) -> List[Dict[str, Any]]:
        """Retrieve all schedules."""
        try:
            db = await self._conn()
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM schedules") as cursor:
                rows = await cursor.fetchall()
                return [
                    {
                        "id": row["id"],
                        "user_prompt": row["user_prompt"],
                        "cron_expression": row["cron_expression"],
                        "last_run": row["last_run"],
                        "is_active": bool(row["is_active"])
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Failed to get schedules: {e}")
            return []

    async def update_last_run(self, schedule_id: str, timestamp: float) -> None:
        """Update the last_run timestamp for a schedule."""
        try:
            db = await self._conn()
            async with db.execute(
                "UPDATE schedules SET last_run = ? WHERE id = ?",
                (timestamp, schedule_id)
            ):
                pass
            await db.commit()
        except Exception as e:
            logger.error(f"Failed to update last_run for {schedule_id}: {e}")

    async def toggle_schedule(self, schedule_id: str, is_active: bool) -> None:
        """Toggle a schedule active or inactive."""
        try:
            db = await self._conn()
            async with db.execute(
                "UPDATE schedules SET is_active = ? WHERE id = ?",
                (1 if is_active else 0, schedule_id)
            ):
                pass
            await db.commit()
        except Exception as e:
            logger.error(f"Failed to toggle schedule {schedule_id}: {e}")

    async def delete_schedule(self, schedule_id: str) -> None:
        """Delete a schedule."""
        try:
            db = await self._conn()
            async with db.execute(
                "DELETE FROM schedules WHERE id = ?",
                (schedule_id,)
            ):
                pass
            await db.commit()
        except Exception as e:
            logger.error(f"Failed to delete schedule {schedule_id}: {e}")
