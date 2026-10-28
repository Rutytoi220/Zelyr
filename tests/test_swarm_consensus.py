"""Comprehensive test suite for Swarm Multi-Agent Consensus Quorum and Blackboard Rollback.

Verifies:
1. >= 2/3 quorum approval allows execution to proceed.
2. Quorum rejection cleanly halts execution and triggers blackboard rollback.
3. Agent timeout or unhandled exception is counted as a non-vote/rejection and handled gracefully without deadlock.
4. Split-decision ties fail safely with informative diagnostic logging.
5. Blackboard transactional isolation: rolled-back keys return to previous state or are removed.
"""
from __future__ import annotations

import logging
import threading
import time
from unittest.mock import MagicMock

import pytest

from zelyr.agents.swarm.base_swarm import BaseSubagent
from zelyr.agents.swarm.consensus import ConsensusEngine, ProposalState
from zelyr.core.events import Event, EventBus
from zelyr.memory.blackboard import BlackboardStore


# ---------------------------------------------------------------------------
# Test 1: Quorum Approval (>= 2/3) allows execution to proceed
# ---------------------------------------------------------------------------

def test_quorum_approval_allows_execution_to_proceed():
    """Verify >= 2/3 quorum approval allows execution to proceed and commits artifacts."""
    bus = EventBus()
    bb = BlackboardStore()
    memory_store = MagicMock()
    session_id = "test_session_approval"

    engine = ConsensusEngine(
        event_bus=bus,
        blackboard=bb,
        memory_store=memory_store,
        session_id=session_id,
        required_quorum=2 / 3,
        total_voters=3,
    )

    # Pre-populate blackboard with proposing agent's artifact
    bb.write(session_id, "CoderAgent", "draft_patch", "def test_ok(): pass")

    events_received = []
    bus.subscribe("swarm.consensus.approved", lambda e: events_received.append(e))

    def cast_approvals():
        time.sleep(0.05)
        # 2 out of 3 votes are APPROVED (66.7% >= 66.7%)
        bus.publish(Event(
            event_type="swarm.vote",
            source="TestRunner_1",
            data={"proposal_id": pid, "vote": "APPROVED", "voter": "TestRunner_1"},
        ))
        bus.publish(Event(
            event_type="swarm.vote",
            source="TestRunner_2",
            data={"proposal_id": pid, "vote": "APPROVED", "voter": "TestRunner_2"},
        ))

    # Create proposal non-blocking to grab proposal ID
    pid = engine.create_proposal(
        agent_name="CoderAgent",
        tool_name="write_file",
        arguments={"path": "main.py", "content": "code"},
        total_voters=3,
        required_quorum=2 / 3,
    )

    vote_thread = threading.Thread(target=cast_approvals)
    vote_thread.start()

    # Poll for consensus resolution
    start = time.time()
    while time.time() - start < 2.0 and engine._proposals[pid]["state"] == ProposalState.PROPOSED:
        time.sleep(0.02)
    vote_thread.join()

    proposal = engine._proposals[pid]
    assert proposal["state"] == ProposalState.APPROVED
    # Verify commit hook was called
    memory_store.add_message.assert_called_once()
    assert "draft_patch" in memory_store.add_message.call_args[0][1]

    # Verify event publication
    assert len(events_received) == 1
    assert events_received[0].data["approved"] is True
    assert events_received[0].data["proposal_id"] == pid


# ---------------------------------------------------------------------------
# Test 2: Quorum rejection cleanly halts execution and triggers rollback
# ---------------------------------------------------------------------------

def test_quorum_rejection_halts_execution_and_triggers_blackboard_rollback():
    """Verify quorum rejection cleanly halts execution and triggers blackboard rollback."""
    bus = EventBus()
    bb = BlackboardStore()
    session_id = "test_session_rejection"

    engine = ConsensusEngine(
        event_bus=bus,
        blackboard=bb,
        session_id=session_id,
        required_quorum=2 / 3,
        total_voters=3,
    )

    # Initial blackboard state before proposal
    bb.write(session_id, "CoderAgent", "stable_key", "original_value")

    events_received = []
    bus.subscribe("swarm.consensus.rejected", lambda e: events_received.append(e))

    # Propose mutating tool (captures pre-proposal snapshot)
    pid = engine.create_proposal(
        agent_name="CoderAgent",
        tool_name="delete_file",
        arguments={"path": "critical.py"},
        total_voters=3,
        required_quorum=2 / 3,
    )

    # Speculatively mutate existing key and add new keys / namespaces during proposal
    bb.write(session_id, "CoderAgent", "stable_key", "MUTATED_VALUE")
    bb.write(session_id, "CoderAgent", "speculative_scratch", "to_be_discarded")
    bb.write(session_id, "RogueAgent", "leak", "sensitive_token")

    # 2 out of 3 voters REJECT the proposal
    bus.publish(Event(
        event_type="swarm.vote",
        source="SafetyVoter_1",
        data={"proposal_id": pid, "vote": "REJECTED", "voter": "SafetyVoter_1"},
    ))
    bus.publish(Event(
        event_type="swarm.vote",
        source="SafetyVoter_2",
        data={"proposal_id": pid, "vote": "REJECTED", "voter": "SafetyVoter_2"},
    ))

    proposal = engine._proposals[pid]
    assert proposal["state"] == ProposalState.REJECTED

    # Verify blackboard state is restored to pre-proposal snapshot
    assert bb.read(session_id, "CoderAgent", "stable_key") == "original_value"
    assert bb.read(session_id, "CoderAgent", "speculative_scratch") is None
    assert bb.read(session_id, "RogueAgent", "leak") is None
    assert "RogueAgent" not in bb.list_agents(session_id)

    # Verify rejection event was emitted
    assert len(events_received) == 1
    assert events_received[0].data["approved"] is False
    assert events_received[0].data["proposal_id"] == pid


# ---------------------------------------------------------------------------
# Test 3: Agent timeout and unhandled exception counted as non-vote
# ---------------------------------------------------------------------------

def test_agent_timeout_and_unhandled_exception_counted_as_non_vote_without_deadlock():
    """Verify timeout / unhandled exception votes are treated as non-approval without deadlock."""
    bus = EventBus()
    bb = BlackboardStore()
    session_id = "test_session_timeout_exception"

    engine = ConsensusEngine(
        event_bus=bus,
        blackboard=bb,
        session_id=session_id,
        required_quorum=2 / 3,
        total_voters=3,
    )

    pid = engine.create_proposal(
        agent_name="CoderAgent",
        tool_name="shell",
        arguments={"command": "rm -rf /tmp/data"},
        total_voters=3,
        required_quorum=2 / 3,
    )

    # Agent 1 approves
    bus.publish(Event(
        event_type="swarm.vote",
        source="Voter_1",
        data={"proposal_id": pid, "vote": "APPROVED", "voter": "Voter_1"},
    ))
    # Agent 2 encounters an unhandled exception / error
    bus.publish(Event(
        event_type="swarm.vote",
        source="Voter_2",
        data={"proposal_id": pid, "vote": "ERROR", "voter": "Voter_2", "error": "Crash"},
    ))
    # Agent 3 times out and posts TIMEOUT
    bus.publish(Event(
        event_type="swarm.vote",
        source="Voter_3",
        data={"proposal_id": pid, "vote": "TIMEOUT", "voter": "Voter_3"},
    ))

    # Only 1 approval out of 3 = 33.3% < 66.7% -> Rejected immediately
    assert engine._proposals[pid]["state"] == ProposalState.REJECTED


def test_silent_timeout_auto_rejects_without_deadlock():
    """Verify propose() auto-rejects cleanly on silent timeout without hanging."""
    bus = EventBus()
    bb = BlackboardStore()
    session_id = "test_session_silent_timeout"

    engine = ConsensusEngine(
        event_bus=bus,
        blackboard=bb,
        session_id=session_id,
        required_quorum=2 / 3,
        total_voters=3,
    )

    bb.write(session_id, "CoderAgent", "draft", "temp_code")

    start = time.time()
    # Fast 0.15s timeout
    result = engine.propose(
        agent_name="CoderAgent",
        tool_name="patch_file",
        arguments={"path": "app.py"},
        timeout=0.15,
        total_voters=3,
        required_quorum=2 / 3,
    )
    elapsed = time.time() - start

    assert result is False
    assert 0.10 <= elapsed < 1.0  # Completed swiftly without deadlocking
    # Blackboard rolled back
    assert bb.read(session_id, "CoderAgent", "draft") == "temp_code"


# ---------------------------------------------------------------------------
# Test 4: Split-decision ties fail safely with informative diagnostic logging
# ---------------------------------------------------------------------------

def test_split_decision_tie_fails_safely_with_diagnostic_logging(caplog):
    """Verify split votes (1 approve, 1 reject out of 2) fail safely with diagnostic log."""
    bus = EventBus()
    bb = BlackboardStore()
    session_id = "test_session_split"

    engine = ConsensusEngine(
        event_bus=bus,
        blackboard=bb,
        session_id=session_id,
        required_quorum=2 / 3,
        total_voters=2,
    )

    bb.write(session_id, "CoderAgent", "state_key", "before_split")

    events_received = []
    bus.subscribe("swarm.consensus.rejected", lambda e: events_received.append(e))

    pid = engine.create_proposal(
        agent_name="CoderAgent",
        tool_name="write_file",
        arguments={"path": "config.yaml"},
        total_voters=2,
        required_quorum=2 / 3,
    )

    # Mutate blackboard during proposal
    bb.write(session_id, "CoderAgent", "state_key", "after_split")

    with caplog.at_level(logging.WARNING):
        # 1 APPROVED, 1 REJECTED (Split decision tie)
        bus.publish(Event(
            event_type="swarm.vote",
            source="AgentA",
            data={"proposal_id": pid, "vote": "APPROVED", "voter": "AgentA"},
        ))
        bus.publish(Event(
            event_type="swarm.vote",
            source="AgentB",
            data={"proposal_id": pid, "vote": "REJECTED", "voter": "AgentB"},
        ))

    proposal = engine._proposals[pid]
    assert proposal["state"] == ProposalState.REJECTED

    # Check diagnostic logging
    warning_records = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    split_warnings = [m for m in warning_records if "split vote / tie" in m]
    assert len(split_warnings) >= 1
    assert "Safely rejecting and rolling back" in split_warnings[0]

    # Blackboard rolled back
    assert bb.read(session_id, "CoderAgent", "state_key") == "before_split"

    # Event emitted with is_split=True
    assert len(events_received) == 1
    assert events_received[0].data["is_split"] is True
    assert events_received[0].data["reason"] == "split_decision"


# ---------------------------------------------------------------------------
# Test 5: Blackboard Transactional Isolation and Rollback
# ---------------------------------------------------------------------------

def test_blackboard_transactional_isolation_and_savepoints():
    """Verify BlackboardStore create_snapshot, restore_snapshot, and savepoint transactions."""
    bb = BlackboardStore()
    session_id = "test_txn_isolation"

    # Initial state
    bb.write(session_id, "agent_1", "config", {"debug": True, "workers": 4})
    bb.write(session_id, "agent_1", "version", 1)

    # 1. Direct snapshot test
    snapshot = bb.create_snapshot(session_id)

    # Mutate values and add keys
    bb.write(session_id, "agent_1", "config", {"debug": False, "workers": 1})
    bb.write(session_id, "agent_1", "version", 2)
    bb.write(session_id, "agent_1", "scratch", "unwanted")
    bb.write(session_id, "agent_2", "temp", "rogue")

    # Restore snapshot
    bb.restore_snapshot(session_id, snapshot)

    assert bb.read(session_id, "agent_1", "config") == {"debug": True, "workers": 4}
    assert bb.read(session_id, "agent_1", "version") == 1
    assert bb.read(session_id, "agent_1", "scratch") is None
    assert bb.read(session_id, "agent_2", "temp") is None
    assert "agent_2" not in bb.list_agents(session_id)

    # 2. Transaction savepoint stack (begin_transaction / rollback_transaction)
    t1 = bb.begin_transaction(session_id)
    bb.write(session_id, "agent_1", "version", 10)
    bb.write(session_id, "agent_1", "new_key", "t1_val")

    # Nested transaction t2
    t2 = bb.begin_transaction(session_id)
    bb.write(session_id, "agent_1", "version", 20)
    bb.write(session_id, "agent_1", "inner_key", "t2_val")

    # Roll back t2
    rolled_back = bb.rollback_transaction(session_id, t2)
    assert rolled_back is True
    assert bb.read(session_id, "agent_1", "version") == 10
    assert bb.read(session_id, "agent_1", "inner_key") is None
    assert bb.read(session_id, "agent_1", "new_key") == "t1_val"

    # Roll back t1
    rolled_back = bb.rollback_transaction(session_id, t1)
    assert rolled_back is True
    assert bb.read(session_id, "agent_1", "version") == 1
    assert bb.read(session_id, "agent_1", "new_key") is None

    # 3. Commit transaction
    t3 = bb.begin_transaction(session_id)
    bb.write(session_id, "agent_1", "committed_key", "persisted")
    assert bb.commit_transaction(session_id, t3) is True
    # Committed key remains intact
    assert bb.read(session_id, "agent_1", "committed_key") == "persisted"


# ---------------------------------------------------------------------------
# Test 6: BaseSubagent & Mutating Tool Execution Safety
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_base_subagent_blocks_mutating_tool_on_rejected_consensus():
    """Verify BaseSubagent prevents mutating tool execution when consensus rejects."""
    bus = EventBus()
    bb = BlackboardStore()
    session_id = "test_subagent_blocks"

    mock_registry = MagicMock()
    mock_tool = MagicMock(return_value="executed_successfully")
    mock_registry.get_tool.return_value = mock_tool

    subagent = BaseSubagent(
        name="CoderSubagent",
        description="Coder",
        topic="swarm.code",
        event_bus=bus,
        tool_registry=mock_registry,
        blackboard=bb,
        session_id=session_id,
    )
    # Configure consensus to require 2 voters
    subagent.consensus.total_voters = 2

    # Reject proposal automatically upon publication
    def auto_reject(event: Event):
        pid = event.data.get("proposal_id")
        bus.publish(Event(
            event_type="swarm.vote",
            source="SafetyAgent",
            data={"proposal_id": pid, "vote": "REJECTED", "voter": "SafetyAgent"},
        ))

    bus.subscribe("swarm.proposal", auto_reject)

    result = await subagent._execute_tool_safely(
        tool_name="write_file",
        arguments={"path": "bad.py", "content": "rm -rf"},
        timeout=1.0,
    )

    assert "Execution blocked: Swarm consensus rejected the proposal." in result.get("error", "")
    mock_tool.assert_not_called()


# ---------------------------------------------------------------------------
# Test 7: Early rejection when quorum is mathematically impossible
# ---------------------------------------------------------------------------

def test_early_rejection_when_quorum_mathematically_impossible():
    """Verify proposal fails immediately when non-approvals prevent reaching quorum."""
    bus = EventBus()
    bb = BlackboardStore()
    session_id = "test_early_rejection"

    # 5 voters, required quorum 0.8 (requires 4 approvals)
    engine = ConsensusEngine(
        event_bus=bus,
        blackboard=bb,
        session_id=session_id,
        required_quorum=0.8,
        total_voters=5,
    )

    pid = engine.create_proposal(
        agent_name="CoderAgent",
        tool_name="delete_file",
        arguments={"path": "data.db"},
        total_voters=5,
        required_quorum=0.8,
    )

    # Voter 1 rejects
    bus.publish(Event("swarm.vote", "Voter_1", data={"proposal_id": pid, "vote": "REJECTED", "voter": "Voter_1"}))
    assert engine._proposals[pid]["state"] == ProposalState.PROPOSED

    # Voter 2 rejects (Now max possible approvals = 3/5 = 0.60 < 0.80)
    bus.publish(Event("swarm.vote", "Voter_2", data={"proposal_id": pid, "vote": "REJECTED", "voter": "Voter_2"}))

    # Early rejection triggers without waiting for voters 3, 4, 5!
    assert engine._proposals[pid]["state"] == ProposalState.REJECTED
