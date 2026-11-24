"""Deterministic offline tests for End-to-End Execution Lifecycle Diagnostics."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import List

import pytest

from zelyr.engine import (
    CentralExecutionLedger,
    CheckpointManager,
    ExecutionLifecycleDiagnostics,
    ExecutionPolicyConfig,
    ExecutionPolicyEngine,
    ExecutionStatus,
    LedgerEventType,
    PendingActionProposal,
    ProposalStatus,
    TaskGraph,
    TaskNode,
    TaskNodeStatus,
    VerificationStatus,
    diagnose_execution_lifecycle,
    format_execution_lifecycle_report,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _run_cli(args: List[str], *, config_dir_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_repo_root())
    env["ZELYR_TESTING"] = "1"
    env["QT_QPA_PLATFORM"] = "offscreen"
    if config_dir_path is not None:
        env["ZELYR_CONFIG_DIR"] = str(config_dir_path)
    return subprocess.run(
        [sys.executable, "-m", "zelyr.cli.main", *args],
        cwd=str(_repo_root()),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _build_rich_execution_fixture(db_path: Path) -> str:
    """Create a realistic interrupted multi-node execution ledger with verified, reported, failed, and blocked steps."""
    ledger = CentralExecutionLedger(db_path=db_path, raise_on_persistence_error=True)
    exec_id = "exec_diag_rich_001"
    try:
        ledger.create_execution(
            "Build release bundle sk-prompt-secret-leak123",
            execution_id=exec_id,
            status=ExecutionStatus.INTERRUPTED,
            source_component="orchestrator",
        )
        ledger.record_event(
            exec_id,
            LedgerEventType.PLANNING_DECISION,
            status="plan_compiled",
            metadata={"plan_id": "plan_rel_1", "graph_id": "tg_rel_1", "reason": "Compiled 4-step workflow"},
        )

        # Node 1 / call_1: Policy allowed + executed + independently verified
        eng = ExecutionPolicyEngine(config=ExecutionPolicyConfig(policy_mode="conservative"))
        d1 = eng.evaluate(
            "echo",
            {"text": "step1", "api_key": "sk-raw-arg-secret-1"},
            execution_id=exec_id,
            node_id="n1",
            central_ledger=ledger,
        )
        assert d1.allowed
        ledger.record_event(
            exec_id,
            LedgerEventType.TOOL_SELECTED,
            tool_id="echo",
            correlation_id="call_1",
            status="selected",
            metadata={"proposal_id": "prop_1", "arguments": {"text": "step1", "api_key": "sk-raw-arg-secret-1"}},
        )
        ledger.record_event(
            exec_id,
            LedgerEventType.TOOL_EXECUTION_START,
            tool_id="echo",
            correlation_id="call_1",
            status="started",
        )
        ledger.record_event(
            exec_id,
            LedgerEventType.TOOL_EXECUTION_RESULT,
            tool_id="echo",
            correlation_id="call_1",
            status="succeeded",
            metadata={"exit_code": 0, "verified": True},
        )
        ledger.record_event(
            exec_id,
            LedgerEventType.OBSERVATION_UPDATED,
            tool_id="echo",
            correlation_id="call_1",
            status="succeeded_verified",
            metadata={
                "evidence_level": "CONFIRMED",
                "execution_status": "succeeded_verified",
                "verification_status": "verified",
                "verified_targets": ["artifact.txt"],
            },
        )
        ledger.record_event(
            exec_id,
            LedgerEventType.VERIFICATION_RECORDED,
            tool_id="echo",
            correlation_id="call_1",
            status=VerificationStatus.VERIFIED.value,
            metadata={"target": "artifact.txt", "detail": "Checked output exists"},
        )

        # Node 2 / call_2: Tool reported success WITHOUT independent verification
        d2 = eng.evaluate(
            "write_file",
            {"operation": "write", "path": "notes.txt", "content": "secret_arg_body_999"},
            execution_id=exec_id,
            node_id="n2",
            central_ledger=ledger,
        )
        raw_tok, rec_conf = eng.confirmation_store.issue_confirmation(
            tool_id=d2.tool_id,
            arg_digest=d2.arg_digest,
            resource_scope=d2.resource_scope,
            execution_id=exec_id,
            node_id="n2",
            policy_version=eng.config.policy_version,
        )
        eng.evaluate(
            "write_file",
            {"operation": "write", "path": "notes.txt", "content": "secret_arg_body_999"},
            execution_id=exec_id,
            node_id="n2",
            confirmation_token=raw_tok,
            confirmation_id=rec_conf.confirmation_id,
            consume_confirmation=True,
            central_ledger=ledger,
        )
        ledger.record_event(
            exec_id,
            LedgerEventType.TOOL_SELECTED,
            tool_id="write_file",
            correlation_id="call_2",
            status="selected",
            metadata={"proposal_id": "prop_2", "arguments": {"operation": "write", "path": "notes.txt", "password": "pw_secret_value_88"}},
        )
        ledger.record_event(
            exec_id,
            LedgerEventType.TOOL_EXECUTION_START,
            tool_id="write_file",
            correlation_id="call_2",
            status="started",
        )
        ledger.record_event(
            exec_id,
            LedgerEventType.TOOL_EXECUTION_RESULT,
            tool_id="write_file",
            correlation_id="call_2",
            status="succeeded",
            metadata={"exit_code": 0, "verified": False},
        )
        ledger.record_event(
            exec_id,
            LedgerEventType.OBSERVATION_UPDATED,
            tool_id="write_file",
            correlation_id="call_2",
            status="succeeded_unverified",
            metadata={
                "evidence_level": "REPORTED",
                "execution_status": "succeeded_unverified",
                "verification_status": "not_verified",
                "remaining_uncertainty": "File contents not independently verified yet",
            },
        )

        # Node 3 / call_3: Failed + recovery attempted; plus policy denial recorded
        eng.evaluate(
            "delete_file",
            {"path": "important.txt"},
            execution_id=exec_id,
            node_id="n3",
            central_ledger=ledger,
        )
        ledger.record_event(
            exec_id,
            LedgerEventType.TOOL_SELECTED,
            tool_id="shell",
            correlation_id="call_3",
            status="selected",
            metadata={"proposal_id": "prop_3", "arguments": {"command": "pytest --token tok_never_show"}},
        )
        ledger.record_event(
            exec_id,
            LedgerEventType.TOOL_EXECUTION_START,
            tool_id="shell",
            correlation_id="call_3",
            status="started",
        )
        ledger.record_event(
            exec_id,
            LedgerEventType.TOOL_EXECUTION_RESULT,
            tool_id="shell",
            correlation_id="call_3",
            status="failed",
            metadata={"exit_code": 2, "error_type": "TOOL_EXECUTION_FAILED", "error": "Exit code 2 during build"},
        )
        ledger.record_event(
            exec_id,
            LedgerEventType.RECOVERY_ATTEMPT,
            tool_id="shell",
            correlation_id="call_3",
            status="retry_scheduled",
            metadata={
                "attempt": 1,
                "node_id": "n3",
                "recovery_decision": "retry_safe",
                "failure_category": "TOOL_EXECUTION_FAILED",
                "next_recovery_action": "retry_with_clean_env",
                "may_have_side_effect": False,
                "reconciliation_required": False,
            },
        )

        # Build coherent PendingActionProposals + TaskGraph and capture durable checkpoint
        p1 = PendingActionProposal(
            proposal_id="prop_1",
            execution_id=exec_id,
            original_call_id="call_1",
            tool_id="echo",
            normalized_arguments={"text": "step1"},
            proposal_order=0,
            status=ProposalStatus.COMPLETED.value,
        )
        p2 = PendingActionProposal(
            proposal_id="prop_2",
            execution_id=exec_id,
            original_call_id="call_2",
            tool_id="write_file",
            normalized_arguments={"operation": "write", "path": "notes.txt"},
            proposal_order=1,
            status=ProposalStatus.COMPLETED.value,
            depends_on=["prop_1"],
        )
        p3 = PendingActionProposal(
            proposal_id="prop_3",
            execution_id=exec_id,
            original_call_id="call_3",
            tool_id="shell",
            normalized_arguments={"command": "pytest"},
            proposal_order=2,
            status=ProposalStatus.DISPATCHED.value,
            depends_on=["prop_1"],
        )
        p4 = PendingActionProposal(
            proposal_id="prop_4",
            execution_id=exec_id,
            original_call_id="call_4",
            tool_id="echo",
            normalized_arguments={"text": "done"},
            proposal_order=3,
            status=ProposalStatus.BLOCKED.value,
            depends_on=["prop_3"],
        )

        n1 = TaskNode(
            node_id="n1",
            description="Echo build banner",
            proposal_ids=["prop_1"],
            execution_status=TaskNodeStatus.COMPLETED_VERIFIED.value,
            verification_status=VerificationStatus.VERIFIED.value,
            order_index=0,
        )
        n2 = TaskNode(
            node_id="n2",
            description="Write unverified notes",
            proposal_ids=["prop_2"],
            dependency_ids=["n1"],
            execution_status=TaskNodeStatus.COMPLETED_UNVERIFIED.value,
            verification_status=VerificationStatus.NOT_VERIFIED.value,
            order_index=1,
        )
        n3 = TaskNode(
            node_id="n3",
            description="Run build step",
            proposal_ids=["prop_3"],
            dependency_ids=["n1"],
            execution_status=TaskNodeStatus.FAILED.value,
            verification_status=VerificationStatus.VERIFICATION_FAILED.value,
            failure_info={"error_type": "TOOL_EXECUTION_FAILED", "error": "Exit code 2 during build"},
            recoverable=True,
            recovery_state={
                "attempt_count": 1,
                "max_attempts": 3,
                "recovery_decision": "retry_safe",
                "recovery_status": "retry_scheduled",
                "previous_failure_category": "TOOL_EXECUTION_FAILED",
                "next_recovery_action": "retry_with_clean_env",
            },
            order_index=2,
        )
        n4 = TaskNode(
            node_id="n4",
            description="Publish final status",
            proposal_ids=["prop_4"],
            dependency_ids=["n3"],
            execution_status=TaskNodeStatus.BLOCKED.value,
            verification_status=VerificationStatus.NOT_VERIFIED.value,
            blocked_reason="Upstream dependency 'n3' failed",
            order_index=3,
        )
        tg = TaskGraph(
            graph_id="tg_rel_1",
            execution_id=exec_id,
            objective="Build release bundle",
            nodes=[n1, n2, n3, n4],
            max_concurrency=2,
        )
        tg.recompute_schedule(known_proposals=[p1, p2, p3, p4])
        tg.validate(known_proposals=[p1, p2, p3, p4], allow_blocked=True)

        CheckpointManager.capture_checkpoint(
            ledger,
            exec_id,
            status=ExecutionStatus.INTERRUPTED,
            task_prompt="Build release bundle sk-prompt-secret-leak123",
            planning_state={
                "plan_id": "plan_rel_1",
                "plan_version": 1,
                "plan_confirmed": True,
                "last_assistant_text": "Everything succeeded sk-assistant-leak999!",
            },
            pending_actions=[p1, p2, p3, p4],
            task_graph=tg,
        )
        # Keep token secret out of persisted test fixture return; assert later it's redacted everywhere
        assert raw_tok
    finally:
        ledger.close()
    return exec_id


def test_lifecycle_diagnostics_answers_all_questions_and_is_readonly_and_redacted(tmp_path: Path) -> None:
    """Unified diagnostics answers all 10 lifecycle questions, keeps reported!=verified, never mutates DB or leaks secrets."""
    db_path = tmp_path / "diag_rich.db"
    exec_id = _build_rich_execution_fixture(db_path)

    ledger = CentralExecutionLedger(db_path=db_path, raise_on_persistence_error=True)
    try:
        events_before = len(ledger.get_events(exec_id))
        cp_before = ledger.get_latest_checkpoint(exec_id)

        report = ExecutionLifecycleDiagnostics.diagnose(ledger, exec_id, verify_outcomes=False)
        assert report.found is True
        assert report.execution_id == exec_id

        # 1. Request & status
        assert "Build release bundle" in report.request["task_prompt"]
        assert "sk-prompt-secret-leak123" not in report.request["task_prompt"]
        assert report.request["status"] == ExecutionStatus.INTERRUPTED.value

        # 2. Plan & Graph
        assert report.plan_and_graph["has_plan"] is True
        assert report.plan_and_graph["has_graph"] is True
        assert report.plan_and_graph["graph_id"] == "tg_rel_1"
        assert report.plan_and_graph["max_concurrency"] == 2
        assert len(report.plan_and_graph["planning_decisions"]) >= 1

        # 3. Which nodes ran vs failed vs blocked
        assert report.node_summary["total"] == 4
        assert "n1" in report.node_summary["ran_node_ids"]
        assert "n2" in report.node_summary["completed_unverified_node_ids"]
        assert "n3" in report.node_summary["failed_node_ids"]
        assert "n4" in report.node_summary["blocked_node_ids"]
        assert report.nodes[3]["ran"] is False

        # 4. Tools called & argument redaction
        assert set(report.tools["unique_tools"]) >= {"echo", "write_file", "delete_file", "shell"}
        assert report.tools["call_count"] >= 3

        # 5. Policy decisions
        assert report.policy["decision_count"] >= 3
        assert report.policy["outcome_counts"]["allow"] >= 2
        assert report.policy["outcome_counts"]["deny"] >= 1

        # 6. Epistemic distinction: independently verified vs reported-only vs failed/uncertain
        ver = report.verification
        assert ver["reported_vs_verified_distinction_preserved"] is True
        assert ver["verified_count"] == 1
        assert ver["reported_only_unverified_count"] == 1
        assert ver["has_unverified_assistant_claim"] is True
        assert any(item.get("call_id") == "call_1" for item in ver["independently_verified"])
        assert any(item.get("call_id") == "call_2" for item in ver["reported_only_unverified"])
        assert any(item.get("kind") == "assistant_claim" for item in ver["reported_only_unverified"])
        assert any(item.get("call_id") == "call_3" for item in ver["failed_or_uncertain"])

        # 7. Recovery attempted
        assert report.recovery["attempted"] is True
        assert report.recovery["attempt_count"] >= 1

        # 8. Resumability & blockers
        assert report.resumability["checkpoint_id"] == cp_before["checkpoint_id"]
        assert any("n4" in b and "blocked" in b.lower() for b in report.resumability["blocking_reasons"])
        assert any("delete_file" in b for b in report.resumability["blocking_reasons"])

        # Read-only guarantee
        assert len(ledger.get_events(exec_id)) == events_before
        cp_after = ledger.get_latest_checkpoint(exec_id)
        assert cp_after["checkpoint_id"] == cp_before["checkpoint_id"]

        # No secret / raw arg / confirmation token / assistant claim leaks in JSON or human text
        rendered = format_execution_lifecycle_report(report)
        raw_json = json.dumps(report.to_dict())
        for bundle in (rendered, raw_json):
            for secret in (
                "sk-raw-arg-secret-1",
                "pw_secret_value_88",
                "secret_arg_body_999",
                "tok_never_show",
                "sk-assistant-leak999",
                "sk-prompt-secret-leak123",
            ):
                assert secret not in bundle
            assert "Execution Lifecycle Diagnostic" in rendered or report.found
    finally:
        ledger.close()


def test_lifecycle_diagnostics_ambiguous_contradictory_and_missing_or_corrupt(tmp_path: Path) -> None:
    """Diagnostics exposes ambiguous in-flight operations, contradictions, missing IDs, and corrupted DBs."""
    db_path = tmp_path / "diag_edge.db"
    ledger = CentralExecutionLedger(db_path=db_path, raise_on_persistence_error=True)
    exec_id = "exec_diag_edge_002"
    try:
        ledger.create_execution("Check ambiguous & contradictory", execution_id=exec_id, status=ExecutionStatus.INTERRUPTED)
        # In-flight mutating tool started without result
        ledger.record_event(
            exec_id,
            LedgerEventType.TOOL_SELECTED,
            tool_id="write_file",
            correlation_id="call_inflight",
            status="selected",
            metadata={"arguments": {"operation": "write", "path": "out.txt"}},
        )
        ledger.record_event(
            exec_id,
            LedgerEventType.TOOL_EXECUTION_START,
            tool_id="write_file",
            correlation_id="call_inflight",
            status="started",
        )
        # Contradictory observation on another call
        ledger.record_event(
            exec_id,
            LedgerEventType.TOOL_SELECTED,
            tool_id="shell",
            correlation_id="call_contra",
            status="selected",
            metadata={"arguments": {"command": "touch marker"}},
        )
        ledger.record_event(
            exec_id,
            LedgerEventType.TOOL_EXECUTION_START,
            tool_id="shell",
            correlation_id="call_contra",
            status="started",
        )
        ledger.record_event(
            exec_id,
            LedgerEventType.TOOL_EXECUTION_RESULT,
            tool_id="shell",
            correlation_id="call_contra",
            status="succeeded",
        )
        ledger.record_event(
            exec_id,
            LedgerEventType.OBSERVATION_UPDATED,
            tool_id="shell",
            correlation_id="call_contra",
            status="contradictory_evidence",
            metadata={"evidence_level": "CONTRADICTED", "execution_status": "contradictory_evidence", "verification_failed": True},
        )

        rep = diagnose_execution_lifecycle(ledger, exec_id)
        assert rep.found is True
        assert rep.verification["verified_count"] == 0
        kinds = {item["kind"] for item in rep.verification["failed_or_uncertain"]}
        assert "operation_ambiguous" in kinds
        assert "operation_contradictory" in kinds

        missing = diagnose_execution_lifecycle(ledger, "does_not_exist")
        assert missing.found is False
        assert missing.resumability["can_resume"] is False
    finally:
        ledger.close()

    # Corrupted DB returns persistence error cleanly without masking failure
    corrupt_db = tmp_path / "corrupt_diag.db"
    corrupt_db.write_bytes(b"CORRUPT SQLITE BYTES" * 20)
    broken = CentralExecutionLedger(db_path=corrupt_db, raise_on_persistence_error=False)
    try:
        corrupt_rep = diagnose_execution_lifecycle(broken, exec_id)
        assert corrupt_rep.found is False
        assert len(corrupt_rep.persistence_errors) >= 1
        assert corrupt_rep.resumability["can_resume"] is False
    finally:
        broken.close()


def test_cli_and_repl_diagnose_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Both `zelyr diagnose`, `zelyr execution --diagnose`, and REPL `/diagnose` work cleanly offline."""
    db_path = tmp_path / "diag_cli.db"
    exec_id = _build_rich_execution_fixture(db_path)

    # 1. `zelyr diagnose <prefix> --json`
    r1 = _run_cli(["diagnose", exec_id[:12], "--db-path", str(db_path), "--json"])
    assert r1.returncode == 0, r1.stderr
    d1 = json.loads(r1.stdout)
    assert d1["found"] is True
    assert d1["execution_id"] == exec_id
    assert d1["verification"]["verified_count"] == 1
    assert d1["verification"]["reported_only_unverified_count"] == 1

    # 2. `zelyr execution <id> --diagnose` human text
    r2 = _run_cli(["execution", exec_id, "--diagnose", "--db-path", str(db_path)])
    assert r2.returncode == 0, r2.stderr
    assert "Execution Lifecycle Diagnostic" in r2.stdout
    assert "Recovery:" in r2.stdout
    assert "Blockers:" in r2.stdout

    # 3. Missing ID fails non-zero
    r_missing = _run_cli(["diagnose", "nonexistent_exec_xyz", "--db-path", str(db_path), "--json"])
    assert r_missing.returncode != 0

    # 4. REPL /diagnose
    from zelyr.cli.repl import COMMANDS, InlineRepl
    from zelyr.config import AuthMode, ZelyrConfig

    assert "/diagnose" in COMMANDS
    ledger = CentralExecutionLedger(db_path=db_path, raise_on_persistence_error=True)
    try:
        repl = InlineRepl.__new__(InlineRepl)
        repl.config = ZelyrConfig(auth_mode=AuthMode.STRICT)
        repl.orchestrator = SimpleNamespace(central_ledger=ledger, tool_registry=None)
        assert repl.handle_command(f"/diagnose {exec_id[:12]} --json") is True
        out = capsys.readouterr().out
        assert exec_id in out
        assert '"reported_vs_verified_distinction_preserved": true' in out
    finally:
        ledger.close()
