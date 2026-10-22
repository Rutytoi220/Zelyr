"""Tests for the TUI footer's optional cron-scheduler probe.

``axiom.core.scheduler`` imports ``apscheduler`` and the removed
``axiom.core.pipeline`` module at module scope.  The TUI only needs the number
of active jobs for its footer, so the probe must report availability rather
than raising into the render loop or claiming that scheduling is active.
"""

from __future__ import annotations

import logging
import sys
import types


def test_scheduler_probe_reports_active_jobs_when_available(monkeypatch):
    import axiom.cli.tui as tui

    fake_scheduler = types.ModuleType("axiom.core.scheduler")
    fake_scheduler.get_active_jobs = lambda: [object(), object(), object()]
    monkeypatch.setitem(sys.modules, "axiom.core.scheduler", fake_scheduler)
    monkeypatch.setattr(tui, "_scheduler_warning_emitted", False)

    assert tui.get_scheduler_job_count() == (3, True)


def test_scheduler_probe_degrades_and_warns_once_when_absent(monkeypatch, caplog):
    import axiom.cli.tui as tui

    # ``None`` in ``sys.modules`` forces an ImportError for the module.
    monkeypatch.setitem(sys.modules, "axiom.core.scheduler", None)
    monkeypatch.setattr(tui, "_scheduler_warning_emitted", False)

    with caplog.at_level(logging.WARNING, logger=tui.__name__):
        assert tui.get_scheduler_job_count() == (0, False)
        assert tui.get_scheduler_job_count() == (0, False)

    warnings = [
        record
        for record in caplog.records
        if "Cron scheduler unavailable" in record.getMessage()
    ]
    assert len(warnings) == 1
