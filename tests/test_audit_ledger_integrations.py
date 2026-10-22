"""Regression tests for the AuditLedger integrations repaired after the cleanup.

The cleanup removed ``axiom.api.audit_ledger`` and left two callers importing
it.  The live replacement is :class:`axiom.engine.audit_ledger.AuditLedger`,
whose ``log_execution`` signature takes five arguments.  These tests exercise
the real SQLite persistence and the two repaired call sites.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest

from axiom.engine.audit_ledger import AuditLedger


def test_audit_ledger_persists_records(tmp_path):
    ledger = AuditLedger(tmp_path / "audit.db")

    row_id = ledger.log_execution("agent", "tool", {"arg": 1}, "LOW", "SUCCESS")

    assert row_id > 0
    rows = ledger.get_recent_logs(10)
    assert len(rows) == 1
    assert rows[0]["agent_name"] == "agent"
    assert rows[0]["tool_name"] == "tool"
    assert rows[0]["risk_level"] == "LOW"
    assert rows[0]["status"] == "SUCCESS"
    assert '"arg": 1' in rows[0]["arguments"]


@pytest.mark.asyncio
async def test_cron_swarm_logs_successful_job(monkeypatch, tmp_path):
    from axiom.core.events import EventBus
    from axiom.services import cron_swarm

    class FakeSupervisor:
        def __init__(self, *args, **kwargs):
            pass

        def synthesize_results(self, prompt, results):
            return "job completed"

    class FakeLLM:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(cron_swarm, "SwarmSupervisor", FakeSupervisor)
    monkeypatch.setattr(
        "axiom.llm.universal_client.UniversalLLMClient", FakeLLM
    )

    engine = cron_swarm.CronSwarmEngine(EventBus())
    engine.ledger = AuditLedger(tmp_path / "cron.db")

    await engine.execute_job("nightly_report", "summarize the day")

    rows = engine.ledger.get_recent_logs(10)
    assert len(rows) == 1
    assert rows[0]["tool_name"] == "execute_cron_job"
    assert rows[0]["status"] == "SUCCESS"
    assert "nightly_report" in rows[0]["arguments"]
    assert "job completed" in rows[0]["arguments"]


@pytest.mark.asyncio
async def test_cron_swarm_logs_failed_job(monkeypatch, tmp_path):
    from axiom.core.events import EventBus
    from axiom.services import cron_swarm

    class FailingSupervisor:
        def __init__(self, *args, **kwargs):
            pass

        def synthesize_results(self, prompt, results):
            raise RuntimeError("synthesis exploded")

    class FakeLLM:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(cron_swarm, "SwarmSupervisor", FailingSupervisor)
    monkeypatch.setattr(
        "axiom.llm.universal_client.UniversalLLMClient", FakeLLM
    )

    engine = cron_swarm.CronSwarmEngine(EventBus())
    engine.ledger = AuditLedger(tmp_path / "cron.db")

    await engine.execute_job("broken_job", "do something")

    rows = engine.ledger.get_recent_logs(10)
    assert len(rows) == 1
    assert rows[0]["tool_name"] == "execute_cron_job"
    assert rows[0]["status"] == "FAILED"
    assert "synthesis exploded" in rows[0]["arguments"]


def test_mcp_audit_tool_reads_engine_ledger(monkeypatch):
    """``inspect_security_audit`` must use the live AuditLedger implementation.

    The MCP SDK is an optional, currently uninstalled dependency, so it is
    replaced with a minimal decorator stub; the ledger path under test is real.
    """

    class FakeFastMCP:
        def __init__(self, name):
            self.name = name

        def tool(self):
            def decorator(func):
                return func

            return decorator

        def run(self):
            raise AssertionError("server must not run during this test")

    fake_mcp = types.ModuleType("mcp")
    fake_server = types.ModuleType("mcp.server")
    fake_fastmcp = types.ModuleType("mcp.server.fastmcp")
    fake_fastmcp.FastMCP = FakeFastMCP
    fake_mcp.server = fake_server
    fake_server.fastmcp = fake_fastmcp

    monkeypatch.setitem(sys.modules, "mcp", fake_mcp)
    monkeypatch.setitem(sys.modules, "mcp.server", fake_server)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fake_fastmcp)
    monkeypatch.delitem(sys.modules, "axiom.server.mcp_server", raising=False)

    module = importlib.import_module("axiom.server.mcp_server")

    AuditLedger().log_execution("agent", "my_audited_tool", {}, "LOW", "SUCCESS")

    report = module.inspect_security_audit(5)
    assert "my_audited_tool" in report
    assert "SUCCESS" in report
