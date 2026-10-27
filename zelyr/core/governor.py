"""Thermal Governor.

Listens to hardware telemetry and emits throttle events to protect system stability
when temperatures or VRAM usage exceed configured limits.
"""
from zelyr.compat import get_env, state_file

import logging
from typing import Any

from zelyr.core.events import EventBus

logger = logging.getLogger(__name__)

class ThermalGovernor:
    """Singleton that monitors telemetry and emits throttling signals."""

    _instance = None

    @classmethod
    def instance(cls, event_bus: EventBus = None):
        if cls._instance is None:
            if event_bus is None:
                raise ValueError("event_bus must be provided on first initialization.")
            cls._instance = cls(event_bus)
        return cls._instance

    def __init__(self, event_bus: EventBus):
        if ThermalGovernor._instance is not None:
            raise RuntimeError("ThermalGovernor is a singleton. Use .instance().")
            
        self.bus = event_bus
        self.bus.subscribe("telemetry.update", self._on_telemetry)
        
        # Thresholds
        self.max_cpu_temp_avg = 88.0
        self.max_cpu_temp_peak = 95.0
        self.hysteresis_temp = 78.0
        self.max_gpu_temp = 90.0
        self.max_vram_usage = 95.0
        
        # State
        self.is_throttled = False
        self._violation_count = 0
        
        logger.info("ThermalGovernor initialized.")

    def _on_telemetry(self, event: Any) -> None:
        """Evaluate hardware telemetry and trigger limits if breached."""
        import os
        if get_env("DISABLE_THROTTLE", "0") == "1":
            return
            
        data = getattr(event, "data", {})
        
        cpu_temp = data.get("cpu_temp", -1.0)
        cpu_temp_max = data.get("cpu_temp_max", -1.0)
        gpu_temp = data.get("gpu_temp", -1.0)
        vram = data.get("vram", -1.0)
        
        breach_reasons = []
        
        is_cpu_violation = (cpu_temp > self.max_cpu_temp_avg) or (cpu_temp_max > self.max_cpu_temp_peak)
        
        if is_cpu_violation:
            breach_reasons.append(f"CPU Temp Critical (Avg: {cpu_temp:.1f}C, Peak: {cpu_temp_max:.1f}C)")
            
        if gpu_temp > self.max_gpu_temp:
            breach_reasons.append(f"GPU Temp Critical ({gpu_temp:.1f}C)")
            
        if vram > self.max_vram_usage:
            breach_reasons.append(f"VRAM Capacity Critical ({vram:.1f}%)")

        if breach_reasons:
            self._violation_count += 1
            if self._violation_count > 3:
                if not self.is_throttled:
                    logger.warning(f"ThermalGovernor: Throttling engaged! {', '.join(breach_reasons)}")
                    self.is_throttled = True
                    self.bus.publish_sync("system.throttle", {"active": True, "reasons": breach_reasons})
        else:
            self._violation_count = 0
            if self.is_throttled:
                # Apply hysteresis: only disengage when CPU avg drops below 78.0C
                if cpu_temp < self.hysteresis_temp:
                    logger.info(f"ThermalGovernor: Operating parameters nominal (CPU Avg: {cpu_temp:.1f}C). Throttling disengaged.")
                    self.is_throttled = False
                    self.bus.publish_sync("system.throttle", {"active": False, "reasons": []})

import json
import time
import uuid
import asyncio
import aiosqlite
from pathlib import Path

class ApprovalQueue:
    """SQLite-backed queue for pending high-risk tool actions."""
    
    def __init__(self, db_path: str = None):
        if not db_path:
            db_path = str(state_file("approval_queue.db", "approval_queue.db"))
        self.db_path = db_path
        self._db_initialized = False
        self._use_fallback = False
        self._fallback_queue = {}

    async def _ensure_init(self):
        if self._use_fallback or self._db_initialized:
            return
        try:
            await self._init_db()
            self._db_initialized = True
        except Exception as e:
            logger.error(f"ApprovalQueue initialization failed: {e}. Falling back to memory.")
            self._use_fallback = True

    async def _init_db(self):
        async with aiosqlite.connect(self.db_path, timeout=5.0) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS pending_actions (
                    id TEXT PRIMARY KEY,
                    tool_name TEXT,
                    arguments TEXT,
                    status TEXT,
                    timestamp REAL
                )
            ''')
            await db.commit()

    async def _execute_with_retry(self, operation):
        if self._use_fallback:
            return None
        
        last_exception = None
        for attempt in range(3):
            try:
                return await operation()
            except Exception as e:
                last_exception = e
                await asyncio.sleep(0.1 * (2 ** attempt))
                
        logger.error(f"SQLite operation failed after 3 retries: {last_exception}. Switching to memory fallback.")
        self._use_fallback = True
        return None

    async def enqueue(self, tool_name: str, arguments: dict) -> str:
        await self._ensure_init()
        action_id = str(uuid.uuid4())
        
        if self._use_fallback:
            self._fallback_queue[action_id] = {
                "id": action_id, "tool_name": tool_name, 
                "arguments": arguments, "status": "pending", 
                "timestamp": time.time()
            }
            return action_id

        async def _do_insert():
            async with aiosqlite.connect(self.db_path, timeout=5.0) as db:
                await db.execute(
                    "INSERT INTO pending_actions (id, tool_name, arguments, status, timestamp) VALUES (?, ?, ?, ?, ?)",
                    (action_id, tool_name, json.dumps(arguments), "pending", time.time())
                )
                await db.commit()
                return action_id

        result = await self._execute_with_retry(_do_insert)
        if result is None and self._use_fallback:
            # retry failed and fell back
            self._fallback_queue[action_id] = {
                "id": action_id, "tool_name": tool_name, 
                "arguments": arguments, "status": "pending", 
                "timestamp": time.time()
            }
            return action_id
        return result

    async def list_pending(self) -> list:
        await self._ensure_init()
        if self._use_fallback:
            return [v for v in self._fallback_queue.values() if v["status"] == "pending"]

        async def _do_list():
            async with aiosqlite.connect(self.db_path, timeout=5.0) as db:
                async with db.execute("SELECT id, tool_name, arguments, timestamp FROM pending_actions WHERE status = 'pending'") as cursor:
                    rows = await cursor.fetchall()
                    return [{"id": r[0], "tool_name": r[1], "arguments": json.loads(r[2]), "timestamp": r[3]} for r in rows]
                    
        result = await self._execute_with_retry(_do_list)
        if result is None and self._use_fallback:
            return [v for v in self._fallback_queue.values() if v["status"] == "pending"]
        return result or []

    async def resolve(self, action_id: str, approved: bool):
        await self._ensure_init()
        status = "approved" if approved else "denied"
        
        if self._use_fallback:
            if action_id in self._fallback_queue:
                self._fallback_queue[action_id]["status"] = status
            return

        async def _do_resolve():
            async with aiosqlite.connect(self.db_path, timeout=5.0) as db:
                await db.execute("UPDATE pending_actions SET status = ? WHERE id = ?", (status, action_id))
                await db.commit()
                return True

        await self._execute_with_retry(_do_resolve)

class ActionGovernor:
    """Evaluates whether an action requires explicit user approval."""
    
    HIGH_RISK_TOOLS = {"shell", "bash", "execute_command", "file_write", "desktop_automation", "system_reboot"}

    def __init__(self, queue: ApprovalQueue = None):
        self.queue = queue or ApprovalQueue()

    async def check_action(self, tool_name: str, arguments: dict, auth_mode: str, user_present: bool = False) -> tuple[bool, str]:
        """
        Check if an action can proceed immediately.
        Returns (is_approved, action_id_if_queued_or_reason).
        """
        if auth_mode == "AUTOPILOT":
            return True, "Auto-approved by policy"
            
        is_high_risk = tool_name in self.HIGH_RISK_TOOLS
        
        if auth_mode == "STRICT" or is_high_risk:
            if user_present:
                # We assume the caller handles synchronous GUI/CLI prompts if user_present is True
                # Allow it to proceed to the actual tool execution where GUI prompting happens
                return True, "Delegating to interactive tool prompt"
            else:
                # User is away. Do not hang. Queue it and yield.
                action_id = await self.queue.enqueue(tool_name, arguments)
                return False, f"Queued for approval (ID: {action_id})"
                
        # BASIC mode + non-high-risk tool -> allow
        return True, "Allowed by policy"
