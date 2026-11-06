"""Tests for autonomous routine actuator (manage_routine) and routine engine components."""

import json
from pathlib import Path
import pytest

from zelyr.engine.error_classifier import ErrorClassifier
from zelyr.tools.manage_routine import ManageRoutineTool, execute


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "routines_test.db"


@pytest.mark.asyncio
async def test_schedule_syntax_validation(tmp_db_path: Path):
    db_file = str(tmp_db_path)

    bad_interval_res = await execute(
        action="create",
        name="bad_interval",
        schedule="-5",
        db_path=db_file,
    )
    bad_interval = json.loads(bad_interval_res)
    assert bad_interval["status"] == "error"
    assert "must be greater than 0" in bad_interval["error"]

    bad_cron_res = await execute(
        action="create",
        name="bad_cron",
        schedule="99 99 99 99 99",
        db_path=db_file,
    )
    bad_cron = json.loads(bad_cron_res)
    assert bad_cron["status"] == "error"
    assert "Invalid cron expression" in bad_cron["error"]

    missing_sched_res = await execute(
        action="create",
        name="no_sched",
        schedule="",
        db_path=db_file,
    )
    missing_sched = json.loads(missing_sched_res)
    assert missing_sched["status"] == "error"
    assert "requires a valid 'schedule'" in missing_sched["error"]

    good_cron_res = await execute(
        action="create",
        name="morning_report",
        schedule="0 8 * * *",
        routine_type="prompt",
        action_payload="Generate morning status report",
        db_path=db_file,
    )
    good_cron = json.loads(good_cron_res)
    assert good_cron["status"] == "success"
    assert good_cron["schedule"] == "0 8 * * *"
    assert good_cron["schedule_type"] == "cron"


def test_error_classifier_routine_error():
    classified = ErrorClassifier.classify(
        tool_name="manage_routine",
        error="Invalid cron expression: 'invalid_syntax_cron'",
    )
    assert classified.error_type == "ROUTINE_ERROR"
    assert "Routine operation" in classified.summary
    assert "Verify schedule syntax" in classified.remedy_hint
    assert "manage_routine" in classified.allowed_actions
    assert "execute_command" in classified.allowed_actions
