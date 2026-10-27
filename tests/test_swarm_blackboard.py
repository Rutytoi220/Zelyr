import pytest
pytest.importorskip("zelyr.agents.swarm")
"""Tests for Swarm Blackboard Distributed Memory Namespace."""

import threading
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile

from zelyr.memory.blackboard import BlackboardStore, BlackboardNamespace
from zelyr.agents.swarm import BaseSubagent
from zelyr.agents.swarm.consensus import ConsensusEngine, ProposalState
from zelyr.core.events import EventBus, Event
from zelyr.core.transaction import WorkspaceTransactionManager


# ---------------------------------------------------------------------------
# BlackboardStore unit tests
# ---------------------------------------------------------------------------

def test_blackboard_write_and_read():
    """Basic write/read round-trip."""
    bb = BlackboardStore()
    bb.write("session_1", "coder_agent", "draft_ast", {"node": "FunctionDef"})
    result = bb.read("session_1", "coder_agent", "draft_ast")
    assert result == {"node": "FunctionDef"}


def test_blackboard_default_on_missing_key():
    """Reading a non-existent key returns the default."""
    bb = BlackboardStore()
    assert bb.read("session_x", "agent_y", "missing_key", default="none") == "none"


def test_blackboard_namespace_isolation():
    """Two agents in the same session cannot see each other's artifacts."""
    bb = BlackboardStore()
    bb.write("session_1", "coder_agent", "secret", "my_code")
    bb.write("session_1", "test_runner", "secret", "my_tests")

    assert bb.read("session_1", "coder_agent", "secret") == "my_code"
    assert bb.read("session_1", "test_runner", "secret") == "my_tests"


def test_blackboard_session_isolation():
    """Two concurrent sessions do not contaminate each other."""
    bb = BlackboardStore()
    bb.write("session_A", "coder_agent", "draft", "code for A")
    bb.write("session_B", "coder_agent", "draft", "code for B")

    assert bb.read("session_A", "coder_agent", "draft") == "code for A"
    assert bb.read("session_B", "coder_agent", "draft") == "code for B"


def test_blackboard_list_keys():
    """list_keys returns only the agent's own keys."""
    bb = BlackboardStore()
    bb.write("s1", "coder", "file_a", "...")
    bb.write("s1", "coder", "file_b", "...")
    bb.write("s1", "tester", "log", "...")

    keys = bb.list_keys("s1", "coder")
    assert set(keys) == {"file_a", "file_b"}


def test_blackboard_purge_session():
    """purge_session removes all artifacts and frees all namespaces."""
    bb = BlackboardStore()
    bb.write("s1", "coder", "draft", "code")
    bb.write("s1", "tester", "report", "tests")
    bb.write("s2", "coder", "draft", "other code")

    freed = bb.purge_session("s1")
    assert freed == 2  # 2 artifacts purged

    # s1 should now be empty
    assert bb.read("s1", "coder", "draft") is None
    assert bb.read("s1", "tester", "report") is None

    # s2 should remain untouched
    assert bb.read("s2", "coder", "draft") == "other code"


def test_blackboard_uri_format():
    """URI helper produces the correct blackboard:// path."""
    bb = BlackboardStore()
    uri = bb.uri("session_102", "coder_agent", "draft_ast")
    assert uri == "blackboard://session_102/coder_agent/draft_ast"


def test_blackboard_dump_session():
    """dump_session returns a full snapshot of all agent data."""
    bb = BlackboardStore()
    bb.write("s1", "coder", "draft", "some code")
    bb.write("s1", "tester", "log", "pass!")

    dump = bb.dump_session("s1")
    assert dump["coder"]["draft"] == "some code"
    assert dump["tester"]["log"] == "pass!"


# ---------------------------------------------------------------------------
# Thread-safety tests
# ---------------------------------------------------------------------------

def test_blackboard_concurrent_writes_are_isolated():
    """Multiple threads writing to separate namespaces never conflict."""
    bb = BlackboardStore()
    errors = []

    def writer(agent_id: str):
        for i in range(50):
            try:
                bb.write("shared_session", agent_id, f"key_{i}", f"value_{i}")
            except Exception as e:
                errors.append(e)

    threads = [threading.Thread(target=writer, args=(f"agent_{t}",)) for t in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    # Each agent should have 50 entries
    for t in range(5):
        keys = bb.list_keys("shared_session", f"agent_{t}")
        assert len(keys) == 50


# ---------------------------------------------------------------------------
# BaseSubagent Blackboard integration tests
# ---------------------------------------------------------------------------

def test_subagent_write_to_blackboard():
    """Sub-agent writes to Blackboard, readable afterwards."""
    bb = BlackboardStore()
    bus = EventBus()
    agent = BaseSubagent(
        name="coder_agent",
        description="Test coder",
        topic="swarm.task.code",
        event_bus=bus,
        blackboard=bb,
        session_id="test_session"
    )

    agent._write_to_blackboard("draft_code", "def hello(): pass")
    result = agent._read_from_blackboard("draft_code")
    assert result == "def hello(): pass"


def test_subagent_blackboard_cross_agent_read():
    """One agent can read another agent's namespace via explicit cross-agent read."""
    bb = BlackboardStore()
    bus = EventBus()

    coder = BaseSubagent("coder_agent", "Coder", "swarm.code", event_bus=bus, blackboard=bb, session_id="s1")
    tester = BaseSubagent("test_runner", "Tester", "swarm.test", event_bus=bus, blackboard=bb, session_id="s1")

    coder._write_to_blackboard("patch_content", "fix: off-by-one error")
    result = tester._read_agent_artifact("coder_agent", "patch_content")
    assert result == "fix: off-by-one error"


def test_subagent_blackboard_noop_without_blackboard():
    """Sub-agents with no Blackboard attached do not crash."""
    bus = EventBus()
    agent = BaseSubagent("coder", "Coder", "swarm.task", event_bus=bus)
    # These should be silent no-ops
    agent._write_to_blackboard("key", "value")
    assert agent._read_from_blackboard("key") is None


# ---------------------------------------------------------------------------
# ConsensusEngine commit hook tests
# ---------------------------------------------------------------------------

def test_consensus_commit_hook_called_on_approval():
    """On APPROVED vote, the commit hook persists blackboard artifacts to memory."""
    bb = BlackboardStore()
    memory_store = MagicMock()
    bus = EventBus()

    engine = ConsensusEngine(bus, blackboard=bb, memory_store=memory_store, session_id="hook_session")

    # Pre-populate the blackboard with an artifact from the proposing agent
    bb.write("hook_session", "TestAgent", "write_file_result", "# Fixed code")

    # Inject a proposal into the engine's internal state
    proposal_id = "proposal-abc"
    engine._proposals[proposal_id] = {
        "id": proposal_id,
        "agent": "TestAgent",
        "tool": "write_file",
        "arguments": {},
        "state": ProposalState.PROPOSED,
    }

    # Simulate an APPROVED vote event
    bus.publish(Event("swarm.vote", "Voter", data={
        "proposal_id": proposal_id,
        "vote": "APPROVED",
        "voter": "TestRunnerAgent"
    }))

    # The memory store's add_message should have been called once for our artifact
    memory_store.add_message.assert_called_once()
    call_args = memory_store.add_message.call_args[0]
    assert call_args[0] == "system"
    assert "Swarm Consensus Commit" in call_args[1]
    assert "write_file_result" in call_args[1]


def test_consensus_commit_hook_noop_when_not_configured():
    """Commit hook is a no-op when no blackboard is attached to ConsensusEngine."""
    bus = EventBus()
    engine = ConsensusEngine(bus)  # No blackboard/memory_store

    proposal_id = "proposal-xyz"
    engine._proposals[proposal_id] = {
        "id": proposal_id,
        "agent": "TestAgent",
        "tool": "write_file",
        "arguments": {},
        "state": ProposalState.PROPOSED,
    }

    # Should not raise an exception
    bus.publish(Event("swarm.vote", "Voter", data={
        "proposal_id": proposal_id,
        "vote": "APPROVED",
        "voter": "TestRunnerAgent"
    }))


# ---------------------------------------------------------------------------
# WorkspaceTransactionManager purge tests
# ---------------------------------------------------------------------------

def test_transaction_commit_purges_blackboard():
    """Commit triggers Blackboard session purge."""
    bb = BlackboardStore()
    bb.write("txn_session", "coder", "draft", "some code")

    with tempfile.TemporaryDirectory() as tmp:
        txn = WorkspaceTransactionManager(
            transaction_id="txn_001",
            staging_root=Path(tmp),
            blackboard=bb,
            session_id="txn_session"
        )
        txn.begin()
        txn.commit()

    # Blackboard session should now be empty
    assert bb.read("txn_session", "coder", "draft") is None


def test_transaction_rollback_purges_blackboard(tmp_path):
    """Rollback triggers Blackboard session purge."""
    bb = BlackboardStore()
    bb.write("txn_session_2", "coder", "bad_patch", "broken code")

    txn = WorkspaceTransactionManager(
        transaction_id="txn_002",
        staging_root=tmp_path,
        blackboard=bb,
        session_id="txn_session_2",
        verbose=False
    )
    txn.begin()
    txn.rollback()

    assert bb.read("txn_session_2", "coder", "bad_patch") is None


def test_transaction_without_blackboard_does_not_crash(tmp_path):
    """Transaction without a blackboard attached still works normally."""
    txn = WorkspaceTransactionManager(staging_root=tmp_path, verbose=False)
    txn.begin()
    txn.commit()  # Should not raise
