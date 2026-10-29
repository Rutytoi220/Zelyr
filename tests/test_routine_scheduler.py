"""Comprehensive test suite for Autonomous Background Routine Scheduling and Execution.

Covers:
1. Schedule evaluation (cron and interval next due time calculations, edge cases).
2. Routine persistence and lifecycle operations across simulated process restarts.
3. Execution containment under unhandled exceptions (failure recorded in history, loop survives).
4. Execution containment under timeout deadlines (timeout recorded in history, loop survives).
5. Successful routine execution updating timestamps, next runs, and history.
6. Concurrency & duplicate execution prevention within the same interval.
7. Dynamic reload of routine configurations.
8. CLI subcommands (list, add, run, remove, history).
9. Startup invariance & lazy-import performance boundaries (zero premature imports).
10. Backward compatibility with legacy RoutineEngine callers and BackgroundSchedulerService.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import pytest

from zelyr.core.events import EventBus
from zelyr.core.routine import (
    Routine,
    RoutineDatabase,
    RoutineEngine,
    RoutineRun,
    calculate_next_run,
    parse_schedule_expression,
)
from zelyr.services.scheduler_service import BackgroundSchedulerService


class TestRoutineScheduler:
    """Test suite for autonomous background routines."""

    def test_schedule_expression_parsing_interval(self):
        """Interval expressions should parse correctly as positive seconds."""
        stype, expr = parse_schedule_expression("300", "interval")
        assert stype == "interval"
        assert expr == "300"

        # Auto-detect numeric string as interval
        stype2, expr2 = parse_schedule_expression("60")
        assert stype2 == "interval"
        assert expr2 == "60"

        # Float interval
        stype3, expr3 = parse_schedule_expression("1.5")
        assert stype3 == "interval"
        assert expr3 == "1.5"

        # Invalid intervals
        with pytest.raises(ValueError, match="greater than 0"):
            parse_schedule_expression("0", "interval")
        with pytest.raises(ValueError, match="greater than 0"):
            parse_schedule_expression("-10", "interval")

    def test_schedule_expression_parsing_cron(self):
        """Standard 5-field cron expressions should be validated."""
        stype, expr = parse_schedule_expression("0 8 * * *", "cron")
        assert stype == "cron"
        assert expr == "0 8 * * *"

        stype2, expr2 = parse_schedule_expression("*/15 * * * *")
        assert stype2 == "cron"
        assert expr2 == "*/15 * * * *"

        with pytest.raises(ValueError, match="Invalid cron expression"):
            parse_schedule_expression("not a valid cron expression", "cron")

    def test_calculate_next_run_interval(self):
        """Interval next run should step forward deterministically."""
        base = 1000.0
        now = 1000.0
        next_t = calculate_next_run("interval", "60", base_time=base, now_time=now)
        assert next_t == 1060.0

        # When overdue, next run steps forward from now
        overdue_base = 500.0
        now = 1000.0
        next_overdue = calculate_next_run("interval", "60", base_time=overdue_base, now_time=now)
        assert next_overdue == 1060.0

    def test_calculate_next_run_cron(self):
        """Cron next run should compute next alignment in the future."""
        # 2026-10-29 12:00:00 UTC
        base_t = 1793275200.0
        next_t = calculate_next_run("cron", "0 * * * *", base_time=base_t, now_time=base_t)
        assert next_t > base_t
        # Exactly 1 hour later: 3600s
        assert next_t == base_t + 3600.0

    @pytest.mark.asyncio
    async def test_database_persistence_and_lifecycle(self, tmp_path: Path):
        """Routines must persist across database re-initialization (simulated restart)."""
        db_file = tmp_path / "test_routines.db"
        db = RoutineDatabase(db_file)
        await db.initialize()

        # 1. Add routine
        r1 = Routine(
            id="routine-1",
            name="nightly-backup",
            description="Daily data backup",
            schedule_type="cron",
            schedule_expr="0 2 * * *",
            action_type="maintenance",
            action_payload={"task": "backup"},
            enabled=True,
            timeout_seconds=60.0,
        )
        await db.add_routine(r1)

        # 2. Add second routine
        r2 = Routine(
            id="routine-2",
            name="health-probe",
            description="Periodic ping",
            schedule_type="interval",
            schedule_expr="30",
            action_type="tool",
            action_payload={"tool": "echo", "args": {"text": "ping"}},
            enabled=False,
            timeout_seconds=10.0,
        )
        await db.add_routine(r2)

        # 3. Simulate process restart by opening fresh DB connection
        db_restarted = RoutineDatabase(db_file)
        routines = await db_restarted.list_routines(include_disabled=True)
        assert len(routines) == 2

        fetched = await db_restarted.get_routine("nightly-backup")
        assert fetched is not None
        assert fetched.id == "routine-1"
        assert fetched.schedule_type == "cron"
        assert fetched.schedule_expr == "0 2 * * *"
        assert fetched.action_type == "maintenance"
        assert fetched.action_payload == {"task": "backup"}
        assert fetched.enabled is True

        # Toggle enabled state
        await db_restarted.toggle_routine("routine-1", False)
        updated = await db_restarted.get_routine("routine-1")
        assert updated.enabled is False

        # Delete routine
        deleted = await db_restarted.delete_routine("routine-2")
        assert deleted is True
        remaining = await db_restarted.list_routines(include_disabled=True)
        assert len(remaining) == 1
        assert remaining[0].id == "routine-1"

    @pytest.mark.asyncio
    async def test_successful_routine_execution(self, tmp_path: Path):
        """Executing a valid routine records success, duration, and output in history."""
        db_file = tmp_path / "test_routines.db"
        bus = EventBus()
        events_received = []
        bus.subscribe("*", lambda e: events_received.append(e))

        engine = RoutineEngine(db_path=db_file, event_bus=bus, check_interval_seconds=0.1)
        await engine.db.initialize()

        # Register custom action handler that produces deterministic output
        async def mock_handler(r: Routine) -> str:
            return f"Processed action: {r.name}"

        engine.register_action_handler("custom_action", mock_handler)

        routine = await engine.add_routine_async(
            name="test-success-task",
            schedule_expr="10",
            schedule_type="interval",
            action_type="custom_action",
            action_payload="test_payload",
            timeout_seconds=5.0,
        )

        run = await engine.execute_routine_once(routine)
        assert run.status == "success"
        assert "Processed action: test-success-task" in run.output
        assert run.duration_seconds > 0.0
        assert run.error == ""

        # Check DB history
        history = await engine.db.get_history(routine.id)
        assert len(history) == 1
        assert history[0].status == "success"
        assert history[0].routine_name == "test-success-task"
        assert "Processed action" in history[0].output

        # Verify last_run_at and next_run_at were updated
        refreshed = await engine.db.get_routine(routine.id)
        assert refreshed.last_run_at is not None
        assert refreshed.next_run_at is not None
        assert refreshed.next_run_at > refreshed.last_run_at

        # Check events
        event_names = [getattr(e, "name", getattr(e, "event_type", "")) for e in events_received]
        assert "routine.started" in event_names
        assert "routine.completed" in event_names

    @pytest.mark.asyncio
    async def test_execution_containment_on_exception(self, tmp_path: Path):
        """Routines raising unhandled exceptions are caught and recorded as failure."""
        db_file = tmp_path / "test_routines.db"
        engine = RoutineEngine(db_path=db_file, check_interval_seconds=0.1)
        await engine.db.initialize()

        async def failing_handler(r: Routine):
            raise ZeroDivisionError("Simulated calculation explosion")

        engine.register_action_handler("failing_action", failing_handler)

        routine = await engine.add_routine_async(
            name="exploding-task",
            schedule_expr="10",
            schedule_type="interval",
            action_type="failing_action",
            action_payload="boom",
            timeout_seconds=5.0,
        )

        run = await engine.execute_routine_once(routine)
        assert run.status == "failure"
        assert "Simulated calculation explosion" in run.error

        # Ensure history captures failure
        history = await engine.db.get_history(routine.id)
        assert len(history) == 1
        assert history[0].status == "failure"
        assert "Simulated calculation explosion" in history[0].error

    @pytest.mark.asyncio
    async def test_execution_containment_on_timeout(self, tmp_path: Path):
        """Routines exceeding timeout_seconds are aborted and recorded as timeout."""
        db_file = tmp_path / "test_routines.db"
        engine = RoutineEngine(db_path=db_file, check_interval_seconds=0.1)
        await engine.db.initialize()

        async def hung_handler(r: Routine):
            # Sleep longer than the 0.15s timeout
            await asyncio.sleep(2.0)
            return "Should never return"

        engine.register_action_handler("hung_action", hung_handler)

        routine = await engine.add_routine_async(
            name="hung-task",
            schedule_expr="10",
            schedule_type="interval",
            action_type="hung_action",
            action_payload="hang",
            timeout_seconds=0.15,
        )

        run = await engine.execute_routine_once(routine)
        assert run.status == "timeout"
        assert "timed out after 0.15" in run.error

        # Verify DB history
        history = await engine.db.get_history(routine.id)
        assert len(history) == 1
        assert history[0].status == "timeout"
        assert "timed out" in history[0].error

    @pytest.mark.asyncio
    async def test_tool_action_dispatch(self, tmp_path: Path):
        """Routine with action_type='tool' executes via ToolRegistry."""
        db_file = tmp_path / "test_routines.db"
        engine = RoutineEngine(db_path=db_file)
        await engine.db.initialize()

        routine = await engine.add_routine_async(
            name="echo-routine",
            schedule_expr="60",
            schedule_type="interval",
            action_type="tool",
            action_payload={"tool": "echo", "args": {"text": "hello_from_routine"}},
        )

        run = await engine.execute_routine_once(routine)
        assert run.status == "success"
        assert "hello_from_routine" in run.output

    @pytest.mark.asyncio
    async def test_maintenance_action_dispatch(self, tmp_path: Path):
        """Routine with action_type='maintenance' executes registered hook."""
        db_file = tmp_path / "test_routines.db"
        engine = RoutineEngine(db_path=db_file)
        await engine.db.initialize()

        called = []

        def my_hook():
            called.append(True)
            return "Hook execution OK"

        engine.register_maintenance_hook("custom_hook", my_hook)

        routine = await engine.add_routine_async(
            name="maintenance-routine",
            schedule_expr="60",
            schedule_type="interval",
            action_type="maintenance",
            action_payload="custom_hook",
        )

        run = await engine.execute_routine_once(routine)
        assert run.status == "success"
        assert "Hook execution OK" in run.output
        assert len(called) == 1

    @pytest.mark.asyncio
    async def test_autonomous_scheduler_loop_and_concurrency_guard(self, tmp_path: Path):
        """Scheduler loop dispatches due routines and guards against duplicate concurrent runs."""
        db_file = tmp_path / "test_routines.db"
        engine = RoutineEngine(db_path=db_file, check_interval_seconds=0.05)
        await engine.db.initialize()

        exec_count = 0
        in_flight = 0
        max_concurrent = 0

        async def slow_handler(r: Routine):
            nonlocal exec_count, in_flight, max_concurrent
            in_flight += 1
            max_concurrent = max(max_concurrent, in_flight)
            exec_count += 1
            await asyncio.sleep(0.12)
            in_flight -= 1
            return "slow done"

        engine.register_action_handler("slow_action", slow_handler)

        # Add routine due immediately
        now = time.time()
        routine = Routine(
            id="concurrency-test",
            name="concurrency-test",
            description="",
            schedule_type="interval",
            schedule_expr="1",
            action_type="slow_action",
            action_payload="payload",
            enabled=True,
            timeout_seconds=2.0,
            next_run_at=now - 5,  # due immediately
        )
        await engine.db.add_routine(routine)

        # Start background loop
        engine.start()
        # Let it run for 0.35 seconds
        await asyncio.sleep(0.35)
        await engine.stop()

        # At any single moment, concurrency for the same routine must never exceed 1
        assert max_concurrent == 1
        assert exec_count >= 1

    @pytest.mark.asyncio
    async def test_dynamic_reloading(self, tmp_path: Path):
        """Calling reload() triggers routine.reloaded event without restart."""
        db_file = tmp_path / "test_routines.db"
        bus = EventBus()
        reloads = []
        bus.subscribe("routine.reloaded", lambda e: reloads.append(e))

        engine = RoutineEngine(db_path=db_file, event_bus=bus)
        await engine.reload()
        assert len(reloads) == 1

    @pytest.mark.asyncio
    async def test_background_scheduler_service_wrapper(self, tmp_path: Path):
        """BackgroundSchedulerService wraps RoutineEngine cleanly without apscheduler."""
        db_file = tmp_path / "test_routines.db"
        bus = EventBus()
        svc = BackgroundSchedulerService(event_bus=bus, db_path=db_file)

        rid = svc.add_job("test-job", "Perform task", "interval", "60")
        assert rid is not None

        jobs = svc.get_jobs()
        assert len(jobs) == 1
        assert jobs[0]["name"] == "test-job"
        assert jobs[0]["schedule"] == "60"

        # Pause and resume
        svc.pause_job(rid)
        paused_jobs = svc.get_jobs()
        assert paused_jobs[0]["enabled"] is False

        svc.resume_job(rid)
        resumed_jobs = svc.get_jobs()
        assert resumed_jobs[0]["enabled"] is True

        # Remove
        svc.remove_job(rid)
        remaining = svc.get_jobs()
        assert len(remaining) == 0

        # Start and stop cleanly
        svc.start()
        await svc.stop_async()

    def test_startup_invariance_and_lazy_loading(self):
        """zelyr --version and zelyr --help must NOT eagerly import routine engine."""
        code = (
            "import sys\n"
            "from zelyr.cli.main import main\n"
            "sys.argv = ['zelyr', '--version']\n"
            "try:\n"
            "    main()\n"
            "except SystemExit:\n"
            "    pass\n"
            "assert 'zelyr.core.routine' not in sys.modules, 'Routine engine loaded prematurely!'\n"
            "assert 'croniter' not in sys.modules, 'croniter loaded prematurely!'\n"
        )
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert res.returncode == 0, f"Startup invariance violation:\n{res.stdout}\n{res.stderr}"

    def test_cli_routine_lifecycle_e2e(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Verify CLI subcommands (add, list, run, history, remove) end-to-end."""
        test_db = str(tmp_path / "cli_routines.db")
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["ZELYR_TESTING"] = "1"
        env["COLUMNS"] = "120"
        env.pop("COV_CORE_SOURCE", None)
        env.pop("COVERAGE_PROCESS_START", None)

        # 1. Add routine via CLI
        add_cmd = [
            sys.executable,
            "-m",
            "zelyr.cli.main",
            "routine",
            "add",
            "cli-test-routine",
            "--schedule",
            "120",
            "--type",
            "interval",
            "--action",
            "tool",
            "--target",
            '{"tool": "echo", "args": {"text": "cli_echo_pass"}}',
            "--description",
            "CLI E2E Test Routine",
        ]
        res_add = subprocess.run(add_cmd, capture_output=True, text=True, env=env)
        assert res_add.returncode == 0, res_add.stderr
        assert "Added routine" in res_add.stdout
        assert "cli-test-routine" in res_add.stdout

        # 2. List routines via CLI
        list_cmd = [sys.executable, "-m", "zelyr.cli.main", "routine", "list"]
        res_list = subprocess.run(list_cmd, capture_output=True, text=True, env=env)
        assert res_list.returncode == 0, res_list.stderr
        assert "cli-test-routine" in res_list.stdout

        # 3. Manually run routine via CLI
        run_cmd = [sys.executable, "-m", "zelyr.cli.main", "routine", "run", "cli-test-routine"]
        res_run = subprocess.run(run_cmd, capture_output=True, text=True, env=env)
        assert res_run.returncode == 0, res_run.stderr
        assert "Execution succeeded" in res_run.stdout or "completed" in res_run.stdout

        # 4. View history via CLI
        hist_cmd = [sys.executable, "-m", "zelyr.cli.main", "routine", "history", "cli-test-routine"]
        res_hist = subprocess.run(hist_cmd, capture_output=True, text=True, env=env)
        assert res_hist.returncode == 0, res_hist.stderr
        assert "SUCCESS" in res_hist.stdout or "cli-test-routine" in res_hist.stdout

        # 5. Remove routine via CLI
        rm_cmd = [sys.executable, "-m", "zelyr.cli.main", "routine", "remove", "cli-test-routine"]
        res_rm = subprocess.run(rm_cmd, capture_output=True, text=True, env=env)
        assert res_rm.returncode == 0, res_rm.stderr
        assert "removed successfully" in res_rm.stdout

        # 6. Verify empty list
        res_list_after = subprocess.run(list_cmd, capture_output=True, text=True, env=env)
        assert res_list_after.returncode == 0
        assert "No routines configured" in res_list_after.stdout or "cli-test-routine" not in res_list_after.stdout

    def test_legacy_routine_engine_cli_compatibility(self, tmp_path: Path):
        """RoutineEngine initialized with legacy cli parameter works transparently."""
        class MockEngine:
            event_bus = EventBus()

        class MockCLI:
            engine = MockEngine()
            orchestrator = None

        cli = MockCLI()
        db_file = tmp_path / "legacy_routines.db"
        engine = RoutineEngine(cli=cli, db_path=db_file)

        # Legacy add_routine(prompt, cron_expr)
        rid = engine.add_routine("Backup knowledge base", "0 3 * * *")
        assert rid is not None

        routines = engine.list_routines()
        assert len(routines) == 1
        assert routines[0]["schedule_expr"] == "0 3 * * *"

        # Legacy delete
        del_ok = engine.delete_routine(rid)
        assert del_ok is True
        assert len(engine.list_routines()) == 0
