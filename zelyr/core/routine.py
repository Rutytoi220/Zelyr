"""Autonomous background routine scheduling and execution engine for Zelyr.

Provides persistent SQLite-backed routine management, cron and interval
schedule evaluation, execution isolation with timeouts, and run history tracking.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, Union

import aiosqlite
import croniter

from zelyr.compat import data_file

logger = logging.getLogger(__name__)


# ── Data Models ─────────────────────────────────────────────────────────────


@dataclass
class Routine:
    """Definition of an autonomous background routine."""

    id: str
    name: str
    description: str
    schedule_type: str  # "cron" or "interval"
    schedule_expr: str  # e.g. "*/15 * * * *" or "300"
    action_type: str  # "prompt", "tool", or "maintenance"
    action_payload: Any  # Prompt text, tool config dict, or hook name
    enabled: bool = True
    timeout_seconds: float = 120.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_run_at: Optional[float] = None
    next_run_at: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert routine to a serializable dictionary."""
        d = asdict(self)
        if isinstance(d.get("action_payload"), (dict, list)):
            d["action_payload_json"] = json.dumps(d["action_payload"])
        else:
            d["action_payload_json"] = str(d["action_payload"])
        if isinstance(d.get("metadata"), (dict, list)):
            d["metadata_json"] = json.dumps(d["metadata"])
        else:
            d["metadata_json"] = "{}"
        return d

    @classmethod
    def from_row(cls, row: Union[sqlite3.Row, aiosqlite.Row, Dict[str, Any]]) -> Routine:
        """Construct Routine instance from a SQLite row or dict."""
        r = dict(row)
        action_payload_raw = r.get("action_payload", "")
        try:
            action_payload = json.loads(action_payload_raw)
        except (json.JSONDecodeError, TypeError):
            action_payload = action_payload_raw

        metadata_raw = r.get("metadata", "{}")
        try:
            metadata = json.loads(metadata_raw) if metadata_raw else {}
        except (json.JSONDecodeError, TypeError):
            metadata = {}

        return cls(
            id=str(r["id"]),
            name=str(r["name"]),
            description=str(r.get("description", "") or ""),
            schedule_type=str(r["schedule_type"]),
            schedule_expr=str(r["schedule_expr"]),
            action_type=str(r["action_type"]),
            action_payload=action_payload,
            enabled=bool(r.get("enabled", 1)),
            timeout_seconds=float(r.get("timeout_seconds", 120.0)),
            metadata=metadata,
            last_run_at=float(r["last_run_at"]) if r.get("last_run_at") is not None else None,
            next_run_at=float(r["next_run_at"]) if r.get("next_run_at") is not None else None,
            created_at=float(r.get("created_at", time.time())),
            updated_at=float(r.get("updated_at", time.time())),
        )


@dataclass
class RoutineRun:
    """Historical execution record of a single routine run."""

    id: str
    routine_id: str
    routine_name: str
    started_at: float
    completed_at: float
    duration_seconds: float
    status: str  # "success", "failure", "timeout"
    output: str = ""
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Union[sqlite3.Row, aiosqlite.Row, Dict[str, Any]]) -> RoutineRun:
        r = dict(row)
        return cls(
            id=str(r["id"]),
            routine_id=str(r["routine_id"]),
            routine_name=str(r["routine_name"]),
            started_at=float(r["started_at"]),
            completed_at=float(r["completed_at"]),
            duration_seconds=float(r["duration_seconds"]),
            status=str(r["status"]),
            output=str(r.get("output", "") or ""),
            error=str(r.get("error", "") or ""),
        )


# ── Schedule Calculation Helper ─────────────────────────────────────────────


def parse_schedule_expression(
    expr: str, schedule_type: Optional[str] = None
) -> Tuple[str, str]:
    """Validate and normalize a schedule expression.

    Returns:
        (schedule_type, normalized_expr)
    """
    cleaned = expr.strip()
    if not cleaned:
        raise ValueError("Schedule expression cannot be empty.")

    # Check explicit type or auto-detect
    if schedule_type == "interval" or (schedule_type is None and (cleaned.isdigit() or _is_float(cleaned))):
        val = float(cleaned)
        if val <= 0:
            raise ValueError("Interval schedule must be greater than 0 seconds.")
        # If integer, store as int string, else float string
        return "interval", str(int(val) if val.is_integer() else val)

    # Assume cron
    try:
        from croniter import croniter

        if not croniter.is_valid(cleaned):
            raise ValueError(f"Invalid cron expression: '{cleaned}'")
        return "cron", cleaned
    except ImportError:
        # If croniter is missing and caller requested cron
        if schedule_type == "cron":
            raise RuntimeError("croniter package is required for cron-based routines.")
        # Try fallback if it looks like an interval
        if _is_float(cleaned):
            val = float(cleaned)
            return "interval", str(int(val) if val.is_integer() else val)
        raise ValueError(f"Unparseable schedule expression: '{cleaned}'")


def _is_float(val: str) -> bool:
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


def calculate_next_run(
    schedule_type: str,
    schedule_expr: str,
    base_time: Optional[float] = None,
    now_time: Optional[float] = None,
) -> float:
    """Calculate the next timestamp (epoch float) when a routine is due."""
    now = time.time() if now_time is None else now_time
    base = base_time if (base_time is not None and base_time > 0) else now

    if schedule_type == "interval":
        interval_sec = float(schedule_expr)
        if interval_sec <= 0:
            raise ValueError("Interval must be positive.")
        # If routine has never run, base was 'now', next is now + interval
        # If routine ran at base, next is base + interval
        next_run = base + interval_sec
        if next_run <= now:
            # If past due, step forward from now
            next_run = now + interval_sec
        return next_run

    if schedule_type == "cron":
        try:
            from croniter import croniter

            cron = croniter(schedule_expr, base)
            next_run = float(cron.get_next(float))
            # Ensure next_run is strictly in the future relative to base
            if next_run <= base:
                next_run = float(cron.get_next(float))
            return next_run
        except ImportError:
            raise RuntimeError("croniter dependency required for cron calculation.")

    raise ValueError(f"Unsupported schedule type: '{schedule_type}'")


# ── SQLite Persistence Layer ────────────────────────────────────────────────


class RoutineDatabase:
    """SQLite-backed persistent store for routines and execution history."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        if db_path is None:
            self.db_path = str(data_file("routines.db"))
        else:
            self.db_path = str(db_path)
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _init_sqlite_sync(self) -> None:
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA busy_timeout = 10000;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS routines (
                    id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT DEFAULT '',
                    schedule_type TEXT NOT NULL,
                    schedule_expr TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    action_payload TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    timeout_seconds REAL NOT NULL DEFAULT 120.0,
                    metadata TEXT DEFAULT '{}',
                    last_run_at REAL,
                    next_run_at REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS routine_runs (
                    id TEXT PRIMARY KEY,
                    routine_id TEXT NOT NULL,
                    routine_name TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    completed_at REAL NOT NULL,
                    duration_seconds REAL NOT NULL,
                    status TEXT NOT NULL,
                    output TEXT DEFAULT '',
                    error TEXT DEFAULT '',
                    FOREIGN KEY(routine_id) REFERENCES routines(id) ON DELETE CASCADE
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_routines_enabled_next ON routines(enabled, next_run_at);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_routine_runs_routine ON routine_runs(routine_id, started_at DESC);"
            )
            conn.commit()

    async def initialize(self) -> None:
        """Create tables and indices asynchronously."""
        await asyncio.to_thread(self._init_sqlite_sync)

    async def add_routine(self, routine: Routine) -> Routine:
        """Insert or replace a routine in the database."""
        await self.initialize()

        payload_str = (
            json.dumps(routine.action_payload)
            if isinstance(routine.action_payload, (dict, list))
            else str(routine.action_payload)
        )
        metadata_str = (
            json.dumps(routine.metadata)
            if isinstance(routine.metadata, (dict, list))
            else "{}"
        )

        async with aiosqlite.connect(self.db_path, timeout=10.0) as db:
            await db.execute("PRAGMA busy_timeout = 10000;")
            await db.execute(
                """
                INSERT OR REPLACE INTO routines (
                    id, name, description, schedule_type, schedule_expr,
                    action_type, action_payload, enabled, timeout_seconds,
                    metadata, last_run_at, next_run_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    routine.id,
                    routine.name,
                    routine.description,
                    routine.schedule_type,
                    routine.schedule_expr,
                    routine.action_type,
                    payload_str,
                    1 if routine.enabled else 0,
                    routine.timeout_seconds,
                    metadata_str,
                    routine.last_run_at,
                    routine.next_run_at,
                    routine.created_at,
                    routine.updated_at,
                ),
            )
            await db.commit()
        return routine

    async def get_routine(self, name_or_id: str) -> Optional[Routine]:
        """Fetch a single routine by exact name, exact ID, or ID prefix."""
        await self.initialize()
        async with aiosqlite.connect(self.db_path, timeout=10.0) as db:
            db.row_factory = aiosqlite.Row
            # Try exact name first
            async with db.execute(
                "SELECT * FROM routines WHERE name = ? OR id = ?",
                (name_or_id, name_or_id),
            ) as cur:
                row = await cur.fetchone()
                if row:
                    return Routine.from_row(row)

            # Try prefix match on ID
            async with db.execute(
                "SELECT * FROM routines WHERE id LIKE ? LIMIT 1",
                (f"{name_or_id}%",),
            ) as cur:
                row = await cur.fetchone()
                if row:
                    return Routine.from_row(row)
        return None

    async def list_routines(self, include_disabled: bool = True) -> List[Routine]:
        """List all routines, sorted by name."""
        await self.initialize()
        query = "SELECT * FROM routines"
        params: Tuple[Any, ...] = ()
        if not include_disabled:
            query += " WHERE enabled = 1"
        query += " ORDER BY name ASC"

        async with aiosqlite.connect(self.db_path, timeout=10.0) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cur:
                rows = await cur.fetchall()
                return [Routine.from_row(r) for r in rows]

    async def list_due_routines(self, now: float) -> List[Routine]:
        """Fetch all enabled routines whose next_run_at <= now or next_run_at is NULL."""
        await self.initialize()
        async with aiosqlite.connect(self.db_path, timeout=10.0) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT * FROM routines
                WHERE enabled = 1 AND (next_run_at IS NULL OR next_run_at <= ?)
                ORDER BY next_run_at ASC
                """,
                (now,),
            ) as cur:
                rows = await cur.fetchall()
                return [Routine.from_row(r) for r in rows]

    async def update_run_timestamps(
        self, routine_id: str, last_run_at: float, next_run_at: float
    ) -> None:
        """Update last_run_at and next_run_at timestamps for a routine."""
        await self.initialize()
        async with aiosqlite.connect(self.db_path, timeout=10.0) as db:
            await db.execute("PRAGMA busy_timeout = 10000;")
            await db.execute(
                """
                UPDATE routines
                SET last_run_at = ?, next_run_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (last_run_at, next_run_at, time.time(), routine_id),
            )
            await db.commit()

    async def toggle_routine(self, name_or_id: str, enabled: bool) -> bool:
        """Enable or disable a routine."""
        r = await self.get_routine(name_or_id)
        if not r:
            return False
        async with aiosqlite.connect(self.db_path, timeout=10.0) as db:
            await db.execute(
                "UPDATE routines SET enabled = ?, updated_at = ? WHERE id = ?",
                (1 if enabled else 0, time.time(), r.id),
            )
            await db.commit()
        return True

    async def delete_routine(self, name_or_id: str) -> bool:
        """Delete a routine and its history."""
        r = await self.get_routine(name_or_id)
        if not r:
            return False
        async with aiosqlite.connect(self.db_path, timeout=10.0) as db:
            await db.execute("PRAGMA busy_timeout = 10000;")
            await db.execute("DELETE FROM routine_runs WHERE routine_id = ?", (r.id,))
            await db.execute("DELETE FROM routines WHERE id = ?", (r.id,))
            await db.commit()
        return True

    async def record_run(self, run: RoutineRun) -> None:
        """Record an execution run entry."""
        await self.initialize()
        async with aiosqlite.connect(self.db_path, timeout=10.0) as db:
            await db.execute("PRAGMA busy_timeout = 10000;")
            await db.execute(
                """
                INSERT INTO routine_runs (
                    id, routine_id, routine_name, started_at, completed_at,
                    duration_seconds, status, output, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.routine_id,
                    run.routine_name,
                    run.started_at,
                    run.completed_at,
                    run.duration_seconds,
                    run.status,
                    run.output,
                    run.error,
                ),
            )
            await db.commit()

    async def get_history(
        self, name_or_id: Optional[str] = None, limit: int = 20
    ) -> List[RoutineRun]:
        """Fetch recent execution runs."""
        await self.initialize()
        query = "SELECT * FROM routine_runs"
        params: List[Any] = []
        if name_or_id:
            r = await self.get_routine(name_or_id)
            target_id = r.id if r else name_or_id
            query += " WHERE routine_id = ? OR routine_name = ?"
            params.extend([target_id, name_or_id])

        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)

        async with aiosqlite.connect(self.db_path, timeout=10.0) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, tuple(params)) as cur:
                rows = await cur.fetchall()
                return [RoutineRun.from_row(row) for row in rows]

    # Synchronous helpers for quick CLI queries without loop setup
    def sync_add_routine(self, routine: Routine) -> Routine:
        self._init_sqlite_sync()
        payload_str = (
            json.dumps(routine.action_payload)
            if isinstance(routine.action_payload, (dict, list))
            else str(routine.action_payload)
        )
        metadata_str = (
            json.dumps(routine.metadata)
            if isinstance(routine.metadata, (dict, list))
            else "{}"
        )
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO routines (
                    id, name, description, schedule_type, schedule_expr,
                    action_type, action_payload, enabled, timeout_seconds,
                    metadata, last_run_at, next_run_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    routine.id,
                    routine.name,
                    routine.description,
                    routine.schedule_type,
                    routine.schedule_expr,
                    routine.action_type,
                    payload_str,
                    1 if routine.enabled else 0,
                    routine.timeout_seconds,
                    metadata_str,
                    routine.last_run_at,
                    routine.next_run_at,
                    routine.created_at,
                    routine.updated_at,
                ),
            )
            conn.commit()
        return routine

    def sync_delete_routine(self, name_or_id: str) -> bool:
        self._init_sqlite_sync()
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id FROM routines WHERE name = ? OR id = ? OR id LIKE ?",
                (name_or_id, name_or_id, f"{name_or_id}%"),
            )
            row = cur.fetchone()
            if not row:
                return False
            rid = row[0]
            cur.execute("DELETE FROM routine_runs WHERE routine_id = ?", (rid,))
            cur.execute("DELETE FROM routines WHERE id = ?", (rid,))
            conn.commit()
            return True

    def sync_toggle_routine(self, name_or_id: str, enabled: bool) -> bool:
        self._init_sqlite_sync()
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id FROM routines WHERE name = ? OR id = ? OR id LIKE ?",
                (name_or_id, name_or_id, f"{name_or_id}%"),
            )
            row = cur.fetchone()
            if not row:
                return False
            rid = row[0]
            cur.execute(
                "UPDATE routines SET enabled = ?, updated_at = ? WHERE id = ?",
                (1 if enabled else 0, time.time(), rid),
            )
            conn.commit()
            return True

    def sync_list_routines(self, include_disabled: bool = True) -> List[Dict[str, Any]]:
        self._init_sqlite_sync()
        with sqlite3.connect(self.db_path, timeout=5.0) as conn:
            conn.row_factory = sqlite3.Row
            q = "SELECT * FROM routines"
            if not include_disabled:
                q += " WHERE enabled = 1"
            q += " ORDER BY name ASC"
            return [dict(r) for r in conn.execute(q).fetchall()]

    def sync_get_history(self, name_or_id: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        self._init_sqlite_sync()
        with sqlite3.connect(self.db_path, timeout=5.0) as conn:
            conn.row_factory = sqlite3.Row
            q = "SELECT * FROM routine_runs"
            p: List[Any] = []
            if name_or_id:
                q += " WHERE routine_id = ? OR routine_name = ?"
                p.extend([name_or_id, name_or_id])
            q += " ORDER BY started_at DESC LIMIT ?"
            p.append(limit)
            return [dict(r) for r in conn.execute(q, p).fetchall()]


# ── Routine Engine ──────────────────────────────────────────────────────────


class RoutineEngine:
    """Core autonomous routine scheduling and execution engine.

    Evaluates scheduled routines, dispatches actions asynchronously, enforces
    execution timeouts, contains exceptions, and records full run histories.
    """

    def __init__(
        self,
        cli: Any = None,
        db_path: Optional[Union[str, Path]] = None,
        event_bus: Any = None,
        orchestrator: Any = None,
        check_interval_seconds: float = 2.0,
    ):
        self.cli = cli
        self.db = RoutineDatabase(db_path)
        self.check_interval_seconds = check_interval_seconds

        # Event bus resolution (standalone or via CLI)
        if event_bus is not None:
            self.event_bus = event_bus
        elif cli and hasattr(cli, "engine") and hasattr(cli.engine, "event_bus"):
            self.event_bus = cli.engine.event_bus
        else:
            self.event_bus = None

        # Orchestrator resolution
        if orchestrator is not None:
            self.orchestrator = orchestrator
        elif cli and hasattr(cli, "orchestrator"):
            self.orchestrator = cli.orchestrator
        else:
            self.orchestrator = None

        self._is_running = False
        self._task: Optional[asyncio.Task[None]] = None
        self._running_routines: set[str] = set()
        self._active_tasks: set[asyncio.Task[Any]] = set()

        # Pluggable action and maintenance handlers
        self._action_handlers: Dict[str, Callable[[Routine], Coroutine[Any, Any, Any]]] = {}
        self._maintenance_hooks: Dict[str, Callable[[], Union[Any, Coroutine[Any, Any, Any]]]] = {}

    def register_action_handler(
        self,
        action_type: str,
        handler: Callable[[Routine], Coroutine[Any, Any, Any]],
    ) -> None:
        """Register a custom async handler for an action_type."""
        self._action_handlers[action_type] = handler

    def register_maintenance_hook(
        self,
        hook_name: str,
        hook: Callable[[], Union[Any, Coroutine[Any, Any, Any]]],
    ) -> None:
        """Register a maintenance hook callable."""
        self._maintenance_hooks[hook_name] = hook

    def _emit_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        """Safely emit an event on the event bus."""
        if not self.event_bus:
            return
        try:
            if hasattr(self.event_bus, "publish_sync"):
                self.event_bus.publish_sync(event_name, payload)
            elif hasattr(self.event_bus, "publish"):
                self.event_bus.publish(event_name, payload)
            elif hasattr(self.event_bus, "emit"):
                self.event_bus.emit(event_name, payload)
        except Exception as e:
            logger.debug("Failed to emit event %s: %s", event_name, e)

    async def parse_schedule_to_cron(self, prompt: str) -> str:
        """Use LLM or heuristic to translate natural language into a cron expression."""
        if self.cli and hasattr(self.cli, "ollama") and self.cli.ollama:
            sys_prompt = (
                "You are a cron expression generator. Translate the user's natural language schedule "
                "into a standard 5-field cron expression. Output ONLY the cron expression, nothing else."
            )
            try:
                response = await asyncio.to_thread(
                    self.cli.ollama.generate, prompt, system_prompt=sys_prompt
                )
                cron_expr = response.strip().replace("`", "").strip()
                import croniter

                if croniter.croniter.is_valid(cron_expr):
                    return cron_expr
            except Exception as e:
                logger.debug("LLM cron parse failed: %s", e)

        # Basic heuristic fallback
        low = prompt.lower()
        if "every minute" in low or "every 1 minute" in low:
            return "* * * * *"
        if "every hour" in low or "hourly" in low:
            return "0 * * * *"
        if "midnight" in low or "daily" in low or "every day" in low:
            return "0 0 * * *"
        return "0 0 * * *"

    async def add_routine_async(
        self,
        name: str,
        schedule_expr: str,
        schedule_type: Optional[str] = None,
        action_type: str = "prompt",
        action_payload: Any = None,
        timeout_seconds: float = 120.0,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ) -> Routine:
        """Add and persist a new routine."""
        stype, norm_expr = parse_schedule_expression(schedule_expr, schedule_type)
        now = time.time()
        next_run = calculate_next_run(stype, norm_expr, base_time=now, now_time=now)

        payload = action_payload if action_payload is not None else name
        routine = Routine(
            id=str(uuid.uuid4()),
            name=name.strip(),
            description=description,
            schedule_type=stype,
            schedule_expr=norm_expr,
            action_type=action_type,
            action_payload=payload,
            enabled=enabled,
            timeout_seconds=timeout_seconds,
            metadata=metadata or {},
            last_run_at=None,
            next_run_at=next_run if enabled else None,
            created_at=now,
            updated_at=now,
        )
        saved = await self.db.add_routine(routine)
        self._emit_event("routine.added", {"routine_id": saved.id, "name": saved.name})
        return saved

    def add_routine(
        self,
        name_or_prompt: str,
        schedule_expr: Optional[str] = None,
        *,
        schedule_type: Optional[str] = None,
        action_type: str = "prompt",
        action_payload: Any = None,
        timeout_seconds: float = 120.0,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ) -> str:
        """Add routine (synchronous API for backwards compatibility with CLI)."""
        # Backwards compatible signature: add_routine(prompt, cron_expr)
        if schedule_expr is not None and action_payload is None:
            name = name_or_prompt.strip()
            payload = name_or_prompt
        else:
            name = name_or_prompt.strip()
            payload = action_payload if action_payload is not None else name_or_prompt
            if schedule_expr is None:
                schedule_expr = "0 0 * * *"

        stype, norm_expr = parse_schedule_expression(schedule_expr, schedule_type)
        now = time.time()
        next_run = calculate_next_run(stype, norm_expr, base_time=now, now_time=now)

        routine = Routine(
            id=str(uuid.uuid4()),
            name=name.strip(),
            description=description,
            schedule_type=stype,
            schedule_expr=norm_expr,
            action_type=action_type,
            action_payload=payload,
            enabled=enabled,
            timeout_seconds=timeout_seconds,
            metadata=metadata or {},
            last_run_at=None,
            next_run_at=next_run if enabled else None,
            created_at=now,
            updated_at=now,
        )
        self.db.sync_add_routine(routine)
        self._emit_event("routine.added", {"routine_id": routine.id, "name": routine.name})
        return routine.id

    def list_routines(self, include_disabled: bool = True) -> List[Dict[str, Any]]:
        """List routines synchronously (backwards compatibility)."""
        return self.db.sync_list_routines(include_disabled=include_disabled)

    async def list_routines_async(self, include_disabled: bool = True) -> List[Routine]:
        """List routines asynchronously."""
        return await self.db.list_routines(include_disabled=include_disabled)

    def delete_routine(self, routine_id_or_name: str) -> bool:
        """Delete routine synchronously."""
        res = self.db.sync_delete_routine(routine_id_or_name)
        if res:
            self._emit_event("routine.deleted", {"identifier": routine_id_or_name})
        return res

    async def delete_routine_async(self, routine_id_or_name: str) -> bool:
        """Delete routine asynchronously."""
        res = await self.db.delete_routine(routine_id_or_name)
        if res:
            self._emit_event("routine.deleted", {"identifier": routine_id_or_name})
        return res

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Start the autonomous routine evaluation loop."""
        if self._is_running:
            return
        self._is_running = True

        try:
            active_loop = loop or asyncio.get_running_loop()
            self._task = active_loop.create_task(self._loop())
        except RuntimeError:
            logger.debug("RoutineEngine.start called outside of running loop.")

        logger.info("RoutineEngine started (evaluating every %.1fs)", self.check_interval_seconds)

    async def stop(self) -> None:
        """Stop the routine evaluation loop and cancel active runs cleanly."""
        self._is_running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        # Cancel any active running routines
        for t in list(self._active_tasks):
            if not t.done():
                t.cancel()

        if self._active_tasks:
            await asyncio.gather(*self._active_tasks, return_exceptions=True)
            self._active_tasks.clear()

        self._running_routines.clear()
        logger.info("RoutineEngine stopped cleanly.")

    async def reload(self) -> None:
        """Reload routines and notify daemon/clients."""
        logger.debug("RoutineEngine reloaded routines.")
        self._emit_event("routine.reloaded", {})

    async def _loop(self) -> None:
        """Periodic background evaluation loop."""
        await self.db.initialize()

        while self._is_running:
            try:
                now = time.time()
                due_routines = await self.db.list_due_routines(now)

                for r in due_routines:
                    if r.id in self._running_routines:
                        # Prevent duplicate concurrent executions of the same routine
                        continue

                    # Immediately mark as running to prevent double-dispatch in rapid ticks
                    self._running_routines.add(r.id)

                    # Compute and store next_run_at immediately so duplicate scans skip it
                    next_run = calculate_next_run(r.schedule_type, r.schedule_expr, base_time=now, now_time=now)
                    await self.db.update_run_timestamps(r.id, last_run_at=now, next_run_at=next_run)

                    # Dispatch execution in background task
                    task = asyncio.create_task(self._run_routine_guarded(r))
                    self._active_tasks.add(task)
                    task.add_done_callback(self._active_tasks.discard)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("RoutineEngine evaluation loop error: %s", e)

            try:
                await asyncio.sleep(self.check_interval_seconds)
            except asyncio.CancelledError:
                break

    async def execute_routine_once(self, routine: Routine) -> RoutineRun:
        """Execute a routine immediately (manual trigger) and record history."""
        return await self._run_routine_guarded(routine, is_manual=True)

    async def _run_routine_guarded(
        self, routine: Routine, is_manual: bool = False
    ) -> RoutineRun:
        """Execute a routine with timeout enforcement and error containment."""
        start_time = time.time()
        run_id = str(uuid.uuid4())

        self._emit_event(
            "routine.started",
            {
                "routine_id": routine.id,
                "name": routine.name,
                "manual": is_manual,
                "action_type": routine.action_type,
            },
        )

        status = "success"
        output = ""
        error = ""

        try:
            # Wrap execution with timeout
            output = await asyncio.wait_for(
                self._dispatch_action(routine),
                timeout=routine.timeout_seconds,
            )
            status = "success"
        except asyncio.TimeoutError:
            status = "timeout"
            error = f"Routine '{routine.name}' timed out after {routine.timeout_seconds} seconds"
            logger.warning("[Routine] %s", error)
            self._emit_event(
                "routine.timeout",
                {"routine_id": routine.id, "name": routine.name, "timeout": routine.timeout_seconds},
            )
        except Exception as exc:
            status = "failure"
            error = str(exc)
            logger.error("[Routine] Error running '%s': %s", routine.name, exc)
            self._emit_event(
                "routine.failed",
                {"routine_id": routine.id, "name": routine.name, "error": error},
            )
        finally:
            end_time = time.time()
            duration = max(0.001, end_time - start_time)

            run = RoutineRun(
                id=run_id,
                routine_id=routine.id,
                routine_name=routine.name,
                started_at=start_time,
                completed_at=end_time,
                duration_seconds=round(duration, 4),
                status=status,
                output=str(output) if output else "",
                error=error,
            )

            try:
                await self.db.record_run(run)
                # Ensure timestamps are up to date
                next_run = calculate_next_run(
                    routine.schedule_type,
                    routine.schedule_expr,
                    base_time=start_time,
                    now_time=end_time,
                )
                await self.db.update_run_timestamps(
                    routine.id, last_run_at=start_time, next_run_at=next_run
                )
            except Exception as dberr:
                logger.error("Failed to record routine run history: %s", dberr)

            self._running_routines.discard(routine.id)

            if status == "success":
                self._emit_event(
                    "routine.completed",
                    {
                        "routine_id": routine.id,
                        "name": routine.name,
                        "duration": round(duration, 3),
                        "output": str(output)[:300],
                    },
                )

        return run

    async def _dispatch_action(self, routine: Routine) -> str:
        """Route routine action to registered handlers, tools, or orchestrator."""
        # 1. Custom registered handler
        if routine.action_type in self._action_handlers:
            handler = self._action_handlers[routine.action_type]
            res = await handler(routine)
            return str(res) if res is not None else ""

        # 2. Tool action
        if routine.action_type == "tool":
            return await self._dispatch_tool_action(routine)

        # 3. Maintenance action
        if routine.action_type == "maintenance":
            return await self._dispatch_maintenance_action(routine)

        # 4. Default Prompt action
        return await self._dispatch_prompt_action(routine)

    async def _dispatch_tool_action(self, routine: Routine) -> str:
        """Execute a tool via ToolRegistry."""
        from zelyr.tool_registry import ToolRegistry

        registry = ToolRegistry()
        payload = routine.action_payload

        if isinstance(payload, dict):
            tool_id = payload.get("tool") or payload.get("tool_id", "")
            args = payload.get("args") or payload.get("arguments") or {}
        else:
            tool_id = str(payload).strip()
            args = {}

        if not tool_id:
            raise ValueError("Tool action missing 'tool' name in payload.")

        # Execute in thread to keep asyncio event loop responsive
        def _exec() -> Any:
            return registry.execute(tool_id, **args)

        result = await asyncio.to_thread(_exec)
        if not getattr(result, "success", True):
            err_msg = getattr(result, "error", "Tool execution failed")
            raise RuntimeError(f"Tool '{tool_id}' error: {err_msg}")

        return str(getattr(result, "output", result))

    async def _dispatch_maintenance_action(self, routine: Routine) -> str:
        """Execute a registered maintenance hook."""
        payload = routine.action_payload
        hook_name = payload if isinstance(payload, str) else str(payload.get("hook", payload.get("task", "")))

        if hook_name in self._maintenance_hooks:
            hook = self._maintenance_hooks[hook_name]
            if asyncio.iscoroutinefunction(hook):
                res = await hook()
            else:
                res = await asyncio.to_thread(hook)
            return str(res) if res is not None else f"Maintenance hook '{hook_name}' finished."

        if hook_name == "memory_compaction":
            from zelyr.memory.compactor import REMCompactor

            compactor = REMCompactor()
            res = await asyncio.to_thread(compactor.compact_all)
            return f"Memory compaction completed: {res}"

        if hook_name == "health_check":
            import psutil

            return f"System Health OK - CPU: {psutil.cpu_percent()}%, RAM: {psutil.virtual_memory().percent}%"

        raise ValueError(f"Unknown maintenance hook: '{hook_name}'")

    async def _dispatch_prompt_action(self, routine: Routine) -> str:
        """Execute an agent prompt via orchestrator."""
        payload = routine.action_payload
        prompt = payload if isinstance(payload, str) else str(payload.get("prompt", str(payload)))

        if not prompt.strip():
            raise ValueError("Routine prompt cannot be empty.")

        # If orchestrator is available, run prompt
        if self.orchestrator:

            def _run_orch() -> str:
                # Mock input for headless background execution
                import builtins

                orig_input = builtins.input

                def _mock_input(*args: Any, **kwargs: Any) -> str:
                    raise RuntimeError("Routine requires interactive user attention")

                builtins.input = _mock_input
                try:
                    res = self.orchestrator.run(
                        prompt,
                        use_tools=True,
                        session_id=f"routine_{routine.id[:8]}",
                    )
                    if hasattr(res, "success") and not res.success:
                        raise RuntimeError(getattr(res, "error", "Orchestrator failed"))
                    return str(getattr(res, "output", res))
                finally:
                    builtins.input = orig_input

            return await asyncio.to_thread(_run_orch)

        # Standalone daemon mode without LLM orchestrator loaded: emit trigger
        self._emit_event(
            "orchestrator.trigger",
            {"prompt": prompt, "source": "routine", "routine_id": routine.id},
        )
        return f"Triggered orchestrator for prompt: {prompt[:60]}..."
