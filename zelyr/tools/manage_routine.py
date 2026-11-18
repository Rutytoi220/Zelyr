"""Autonomous Routine Actuator & Daemon Temporal Task Execution.

Canonical tool for scheduling, managing, and triggering autonomous background routines,
interval/cron schedule evaluation, and SQLite run history tracking.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from zelyr.core.routine import (
    Routine,
    RoutineDatabase,
    RoutineEngine,
    RoutineRun,
    parse_schedule_expression,
)
from zelyr.tools.core import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)

REQUIRED_RING = 1
TIER = 1
TUI_HINT = "⏰ Managing scheduled routine..."

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "manage_routine",
        "description": "Create, list, run, toggle, delete, and inspect history of autonomous scheduled routines.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "run", "toggle", "delete", "history"],
                    "description": "Routine management action to execute.",
                },
                "name": {
                    "type": "string",
                    "description": "Unique identifier name for the routine.",
                },
                "schedule": {
                    "type": "string",
                    "description": "Schedule specification (e.g. 'every 3600s', '300', or cron expression '0 8 * * *').",
                },
                "routine_type": {
                    "type": "string",
                    "enum": ["command", "prompt", "maintenance"],
                    "description": "Type of action the routine executes (default: command).",
                    "default": "command",
                },
                "action_payload": {
                    "type": "string",
                    "description": "Action payload to execute (e.g. shell command, prompt text, or maintenance task).",
                },
                "description": {
                    "type": "string",
                    "description": "Optional human-readable description of the routine.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of history records to return (default: 10).",
                    "default": 10,
                },
            },
            "required": ["action"],
        },
    },
}


def _error_response(error_msg: str, status_code: str = "ROUTINE_ERROR") -> str:
    """Format structured error payload."""
    payload = {
        "status": "error",
        "error_type": status_code,
        "error": error_msg,
        "remedy_hint": "Routine operation failed. Verify schedule syntax (interval in seconds or valid cron expression) and routine name.",
        "allowed_actions": ["manage_routine", "execute_command"],
    }
    return json.dumps(payload)


async def execute(action: str, **kwargs: Any) -> str:
    """Execute routine management actions and return structured JSON string."""
    clean_action = (action or "").strip().lower()
    db_path = kwargs.get("db_path")
    db = RoutineDatabase(db_path=db_path)
    engine = RoutineEngine(db_path=db_path)

    try:
        # 1. Action: CREATE / ADD
        if clean_action in ("create", "add"):
            name = (kwargs.get("name") or "").strip()
            if not name:
                return _error_response("Routine creation requires a non-empty 'name'.")

            schedule = (kwargs.get("schedule") or kwargs.get("cron") or kwargs.get("interval") or "").strip()
            if not schedule:
                return _error_response(f"Routine '{name}' creation requires a valid 'schedule'.")

            try:
                stype, norm_expr = parse_schedule_expression(schedule)
            except ValueError as ve:
                return _error_response(f"Invalid schedule specification '{schedule}': {ve}")
            except Exception as exc:
                return _error_response(f"Schedule parsing error: {exc}")

            routine_type = (kwargs.get("routine_type") or "command").strip().lower()
            if routine_type not in ("command", "prompt", "maintenance", "tool"):
                routine_type = "command"

            action_payload = kwargs.get("action_payload")
            if action_payload is None:
                action_payload = kwargs.get("command") or kwargs.get("payload") or kwargs.get("instruction") or name

            description = kwargs.get("description", "")
            timeout_seconds = float(kwargs.get("timeout_seconds", 120.0))
            enabled = bool(kwargs.get("enabled", True))

            saved = await engine.add_routine_async(
                name=name,
                schedule_expr=norm_expr,
                schedule_type=stype,
                action_type=routine_type,
                action_payload=action_payload,
                timeout_seconds=timeout_seconds,
                description=description,
                enabled=enabled,
            )

            res = {
                "status": "success",
                "action": "create",
                "name": saved.name,
                "schedule": saved.schedule_expr,
                "schedule_type": saved.schedule_type,
                "routine_type": saved.action_type,
                "next_run": saved.next_run_at,
                "enabled": saved.enabled,
            }
            return json.dumps(res)

        # 2. Action: LIST / LS
        elif clean_action in ("list", "ls"):
            include_disabled = bool(kwargs.get("include_disabled", True))
            routines = await db.list_routines(include_disabled=include_disabled)
            items = []
            for r in routines:
                items.append({
                    "name": r.name,
                    "schedule": r.schedule_expr,
                    "schedule_type": r.schedule_type,
                    "action_type": r.action_type,
                    "enabled": r.enabled,
                    "last_run": r.last_run_at,
                    "next_run": r.next_run_at,
                    "description": r.description,
                })
            res = {
                "status": "success",
                "action": "list",
                "count": len(items),
                "routines": items,
            }
            return json.dumps(res)

        # 3. Action: RUN (manual single-shot trigger)
        elif clean_action == "run":
            name = (kwargs.get("name") or "").strip()
            if not name:
                return _error_response("Action 'run' requires a routine 'name'.")

            r = await db.get_routine(name)
            if not r:
                return _error_response(f"Routine '{name}' not found.")

            run_res = await engine.execute_routine_once(r)
            res = {
                "status": "success" if run_res.status == "success" else "error",
                "action": "run",
                "name": r.name,
                "run_id": run_res.id,
                "execution_status": run_res.status,
                "duration_seconds": run_res.duration_seconds,
                "output": run_res.output,
                "error": run_res.error,
            }
            return json.dumps(res)

        # 4. Action: TOGGLE
        elif clean_action == "toggle":
            name = (kwargs.get("name") or "").strip()
            if not name:
                return _error_response("Action 'toggle' requires a routine 'name'.")

            r = await db.get_routine(name)
            if not r:
                return _error_response(f"Routine '{name}' not found.")

            target_enabled = kwargs.get("enabled")
            if target_enabled is None:
                new_state = not r.enabled
            else:
                new_state = bool(target_enabled)

            await db.toggle_routine(r.id, new_state)
            res = {
                "status": "success",
                "action": "toggle",
                "name": r.name,
                "enabled": new_state,
            }
            return json.dumps(res)

        # 5. Action: DELETE / REMOVE
        elif clean_action in ("delete", "remove", "rm"):
            name = (kwargs.get("name") or "").strip()
            if not name:
                return _error_response("Action 'delete' requires a routine 'name'.")

            ok = await db.delete_routine(name)
            if not ok:
                return _error_response(f"Routine '{name}' not found.")

            res = {
                "status": "success",
                "action": "delete",
                "name": name,
            }
            return json.dumps(res)

        # 6. Action: HISTORY
        elif clean_action == "history":
            name = (kwargs.get("name") or "").strip() or None
            limit = int(kwargs.get("limit", 10))
            history_runs = await db.get_history(name_or_id=name, limit=limit)
            items = []
            for h in history_runs:
                items.append({
                    "id": h.id,
                    "routine_name": h.routine_name,
                    "status": h.status,
                    "started_at": h.started_at,
                    "completed_at": h.completed_at,
                    "duration_seconds": h.duration_seconds,
                    "output": h.output,
                    "error": h.error,
                })
            res = {
                "status": "success",
                "action": "history",
                "name": name,
                "count": len(items),
                "history": items,
            }
            return json.dumps(res)

        else:
            return _error_response(
                f"Unknown routine action '{clean_action}'. Must be one of: create, list, run, toggle, delete, history."
            )

    except Exception as exc:
        logger.error("Error executing manage_routine action '%s': %s", clean_action, exc)
        return _error_response(str(exc))
    finally:
        try:
            await engine.checkpoint()
        except Exception:
            pass


class ManageRoutineTool(BaseTool):
    """Canonical ManageRoutineTool registered in ToolRegistry."""

    REQUIRED_RING: int = 1
    TIER: int = 1
    TUI_HINT: str = "⏰ Managing scheduled routine..."

    def __init__(self, db_path: Optional[Union[str, Path]] = None, **kwargs: Any) -> None:
        super().__init__(
            tool_id="manage_routine",
            name="manage_routine",
            description="Create, list, run, toggle, delete, and inspect history of autonomous scheduled routines.",
            tier=1,
        )
        self.db_path = db_path
        self._custom_schema = TOOL_SCHEMA["function"]["parameters"]
        self.add_parameter(
            ToolParameter(
                "action",
                "string",
                "Routine action to execute (create, list, run, toggle, delete, history)",
                required=True,
            )
        )
        self.add_parameter(ToolParameter("name", "string", "Routine identifier name", required=False))
        self.add_parameter(ToolParameter("schedule", "string", "Schedule expression (interval or cron)", required=False))
        self.add_parameter(ToolParameter("routine_type", "string", "Routine action type", required=False, default="command"))
        self.add_parameter(ToolParameter("action_payload", "string", "Action payload to execute", required=False))
        self.add_parameter(ToolParameter("description", "string", "Description of routine", required=False))
        self.add_parameter(ToolParameter("limit", "integer", "History limit", required=False, default=10))

    @property
    def schema(self) -> Dict[str, Any]:
        return self._custom_schema

    def execute(self, params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> ToolResult:
        """Synchronous execution wrapper."""
        merged = dict(params or {})
        merged.update(kwargs)
        if self.db_path and "db_path" not in merged:
            merged["db_path"] = self.db_path

        try:
            loop = asyncio.get_running_loop()

            async def _inner():
                raw = await self.execute_async(**merged)
                return raw

            return loop.create_task(_inner())
        except RuntimeError:
            from zelyr.core.async_bridge import run_sync

            return run_sync(self.execute_async(**merged))

    async def execute_async(self, params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> ToolResult:
        """Asynchronous execution entry point returning ToolResult."""
        merged = dict(params or {})
        merged.update(kwargs)
        if self.db_path and "db_path" not in merged:
            merged["db_path"] = self.db_path

        action = merged.pop("action", "list")
        raw_json_str = await execute(action=action, **merged)
        try:
            data = json.loads(raw_json_str)
            if data.get("status") == "success":
                return ToolResult(
                    success=True,
                    output=raw_json_str,
                    metadata=data,
                )
            else:
                return ToolResult(
                    success=False,
                    output=raw_json_str,
                    error=data.get("error", "Routine operation failed"),
                    remedy_hint=data.get("remedy_hint"),
                    allowed_actions=data.get("allowed_actions", ["manage_routine", "execute_command"]),
                )
        except Exception:
            return ToolResult(success=True, output=raw_json_str)


__all__ = [
    "ManageRoutineTool",
    "execute",
    "REQUIRED_RING",
    "TIER",
    "TUI_HINT",
    "TOOL_SCHEMA",
]
