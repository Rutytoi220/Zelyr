"""Integration tests for Swarm Triad (Supervisor, Coder, TestRunner) Orchestration.

Verifies:
1. Triad delegation & cross-agent blackboard communication.
2. WorkspaceTransactionManager atomic rollback & blackboard purging on failure.
3. Distributed consensus voting with 2/3 quorum enforcement and commit hook.
4. Subagent timeout containment and supervisor recovery.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from zelyr.agents.base import AgentResult
from zelyr.agents.swarm.coder_agent import CoderAgent
from zelyr.agents.swarm.consensus import ConsensusEngine, ProposalState
from zelyr.agents.swarm.dispatcher import SwarmDispatcher
from zelyr.agents.swarm.supervisor import SwarmSupervisor, SwarmTask
from zelyr.agents.swarm.test_runner_agent import TestRunnerAgent
from zelyr.core.events import Event, EventBus
from zelyr.core.transaction import WorkspaceTransactionManager
from zelyr.memory.blackboard import BlackboardStore


@pytest.mark.asyncio
async def test_swarm_triad_delegation(tmp_path: Path):
    """Supervisor analyzes prompt, delegates to Coder and TestRunner, and synthesizes output."""
    bus = EventBus()
    bb = BlackboardStore()
    session_id = "triad_delegation_session"

    # Mock LLM for supervisor
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = [
        # 1. analyze_task call: returns decomposition
        {
            "content": (
                '{"is_swarm_required": true, "tasks": ['
                '{"agent_type": "CoderAgent", "task_description": "Implement fibonacci function"},'
                '{"agent_type": "TestRunnerAgent", "task_description": "Run unit tests on fibonacci"}'
                ']}'
            )
        },
        # 2. synthesize_results call: returns final response
        {"content": "Fibonacci implementation verified with 100% test pass rate."},
    ]

    supervisor = SwarmSupervisor(llm_client=mock_llm, event_bus=bus)
    dispatcher = SwarmDispatcher(
        event_bus=bus,
        tool_registry=MagicMock(),
        llm_client=mock_llm,
        session_id=session_id,
        blackboard=bb,
    )

    # 1. Supervisor decomposes prompt
    tasks = supervisor.analyze_task("Create and test a fibonacci utility")
    assert len(tasks) == 2
    assert tasks[0].agent_type == "CoderAgent"
    assert tasks[1].agent_type == "TestRunnerAgent"

    # Custom run methods for this triad session exercising blackboard coordination
    async def coder_run(desc: str, context=None):
        bb.write(session_id, "CoderAgent", "draft_code", "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)")
        return AgentResult(success=True, output="Generated fibonacci in draft_code")

    async def test_runner_run(desc: str, context=None):
        code = bb.read(session_id, "CoderAgent", "draft_code")
        assert code is not None, "TestRunnerAgent should be able to read Coder's artifact"
        bb.write(session_id, "TestRunnerAgent", "test_report", "All 5 unit tests passed.")
        return AgentResult(success=True, output="Verified draft_code successfully")

    dispatcher.agents["CoderAgent"].run = coder_run
    dispatcher.agents["TestRunnerAgent"].run = test_runner_run

    # 2. Dispatch tasks concurrently
    results = await dispatcher.dispatch(tasks)
    assert "CoderAgent" in results
    assert "TestRunnerAgent" in results
    assert "Generated fibonacci" in results["CoderAgent"]
    assert "Verified draft_code" in results["TestRunnerAgent"]

    # 3. Verify blackboard artifacts exist across both agent namespaces
    assert bb.read(session_id, "CoderAgent", "draft_code").startswith("def fib")
    assert bb.read(session_id, "TestRunnerAgent", "test_report") == "All 5 unit tests passed."

    # 4. Supervisor synthesizes results
    final_output = supervisor.synthesize_results("Create and test a fibonacci utility", results)
    assert "Fibonacci implementation verified" in final_output


@pytest.mark.asyncio
async def test_blackboard_transaction_rollback_on_subagent_failure(tmp_path: Path):
    """Subagent mutation failure triggers transaction rollback and complete blackboard purge."""
    bus = EventBus()
    bb = BlackboardStore()
    session_id = "txn_rollback_session"
    target_file = tmp_path / "module.py"
    target_file.write_text("original_version = 1\n", encoding="utf-8")

    # Agent populates blackboard artifacts
    bb.write(session_id, "CoderAgent", "patch_ast", {"change": "add_function"})
    bb.write(session_id, "TestRunnerAgent", "precheck", "ok")
    assert len(bb.list_keys(session_id, "CoderAgent")) == 1

    txn = WorkspaceTransactionManager(
        bus=bus,
        transaction_id="txn_triad_001",
        staging_root=tmp_path / "staging",
        blackboard=bb,
        session_id=session_id,
        verbose=False,
    )

    txn.begin()
    txn.snapshot(target_file)

    # Mutate target file
    target_file.write_text("broken_version = 2\n", encoding="utf-8")

    # Simulate subagent failure triggering transaction rollback
    txn.rollback()

    # Verify filesystem restored
    assert target_file.read_text(encoding="utf-8") == "original_version = 1\n"

    # Verify blackboard is completely purged with 0 remaining artifacts
    assert bb.read(session_id, "CoderAgent", "patch_ast") is None
    assert bb.read(session_id, "TestRunnerAgent", "precheck") is None
    assert len(bb.dump_session(session_id)) == 0


def test_consensus_voting_protocol_quorum():
    """Consensus voting protocol with 2/3 quorum: approval vs rejection thresholds."""
    bus = EventBus()
    bb = BlackboardStore()
    memory_store = MagicMock()
    session_id = "quorum_session"

    engine = ConsensusEngine(
        event_bus=bus,
        blackboard=bb,
        memory_store=memory_store,
        session_id=session_id,
        required_quorum=2 / 3,
        total_voters=3,
    )

    # Case A: 2 of 3 vote APPROVED -> Quorum (66.7%) met -> APPROVED
    bb.write(session_id, "CoderAgent", "draft_patch", "def safe_patch(): pass")
    proposal_a_id = "prop_approved_001"
    engine._proposals[proposal_a_id] = {
        "id": proposal_a_id,
        "agent": "CoderAgent",
        "tool": "write_file",
        "arguments": {"path": "safe.py"},
        "state": ProposalState.PROPOSED,
        "timestamp": 0.0,
        "votes": {},
        "total_voters": 3,
        "required_quorum": 2 / 3,
    }

    # Vote 1: CoderAgent approves (1/3 < 2/3 -> still PROPOSED)
    bus.publish(Event("swarm.vote", source="Voter1", data={"proposal_id": proposal_a_id, "vote": "APPROVED", "voter": "CoderAgent"}))
    assert engine._proposals[proposal_a_id]["state"] == ProposalState.PROPOSED

    # Vote 2: TestRunnerAgent approves (2/3 >= 2/3 -> APPROVED)
    bus.publish(Event("swarm.vote", source="Voter2", data={"proposal_id": proposal_a_id, "vote": "APPROVED", "voter": "TestRunnerAgent"}))
    assert engine._proposals[proposal_a_id]["state"] == ProposalState.APPROVED
    # Consensus hook committed artifact to memory store
    memory_store.add_message.assert_called_once()
    assert "Swarm Consensus Commit" in memory_store.add_message.call_args[0][1]

    # Case B: 2 of 3 vote REJECTED -> Quorum impossible -> REJECTED
    memory_store.reset_mock()
    proposal_b_id = "prop_rejected_002"
    engine._proposals[proposal_b_id] = {
        "id": proposal_b_id,
        "agent": "CoderAgent",
        "tool": "shell",
        "arguments": {"command": "rm -rf /"},
        "state": ProposalState.PROPOSED,
        "timestamp": 0.0,
        "votes": {},
        "total_voters": 3,
        "required_quorum": 2 / 3,
    }

    # Vote 1: TestRunnerAgent rejects
    bus.publish(Event("swarm.vote", source="Voter3", data={"proposal_id": proposal_b_id, "vote": "REJECTED", "voter": "TestRunnerAgent"}))
    assert engine._proposals[proposal_b_id]["state"] == ProposalState.PROPOSED

    # Vote 2: Supervisor rejects (2 rejections out of 3 -> max approvals 1/3 < 2/3 -> REJECTED)
    bus.publish(Event("swarm.vote", source="Voter4", data={"proposal_id": proposal_b_id, "vote": "REJECTED", "voter": "Supervisor"}))
    assert engine._proposals[proposal_b_id]["state"] == ProposalState.REJECTED
    # Ensure memory store was NOT called
    memory_store.add_message.assert_not_called()


@pytest.mark.asyncio
async def test_subagent_timeout_containment_and_supervisor_recovery():
    """Hanging subagent is contained via timeout; supervisor recovers gracefully."""
    bus = EventBus()
    supervisor = SwarmSupervisor(llm_client=None, event_bus=bus)
    dispatcher = SwarmDispatcher(
        event_bus=bus,
        tool_registry=MagicMock(),
        llm_client=None,
        session_id="timeout_containment_session",
    )

    async def hanging_coder(desc: str, context=None):
        await asyncio.sleep(5.0)
        return AgentResult(success=True, output="Should not be reached")

    async def responsive_tester(desc: str, context=None):
        return AgentResult(success=True, output="Pre-flight checks passed")

    dispatcher.agents["CoderAgent"].run = hanging_coder
    dispatcher.agents["TestRunnerAgent"].run = responsive_tester

    tasks = [
        SwarmTask(agent_type="CoderAgent", task_description="Long running code synthesis"),
        SwarmTask(agent_type="TestRunnerAgent", task_description="Fast status check"),
    ]

    # Dispatch with 0.1s timeout
    results = await dispatcher.dispatch(tasks, timeout=0.1)

    assert "CoderAgent" in results
    assert "TestRunnerAgent" in results
    assert "TimeoutError" in results["CoderAgent"]
    assert "Pre-flight checks passed" in results["TestRunnerAgent"]

    # Supervisor recovers and synthesizes partial results without crashing
    synthesis = supervisor.synthesize_results("Perform maintenance", results)
    assert "TimeoutError" in synthesis
    assert "Pre-flight checks passed" in synthesis
