"""Deterministic offline tests for Execution Policy Management & Explainability."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from zelyr.config import AuthMode, ZelyrConfig
from zelyr.engine import (
    CentralExecutionLedger,
    ConfirmationStore,
    ExecutionPolicyConfig,
    ExecutionPolicyDecision,
    ExecutionPolicyEngine,
    LedgerEventType,
    PolicyDecisionOutcome,
    PolicyExplanation,
    RiskLevel,
    inspect_policy_configuration,
    preview_tool_policy,
    query_policy_history,
)
from zelyr.tool_registry import TaskLocalToolScope, ToolRegistry
from zelyr.tools.core import BaseTool, ToolParameter, ToolResult


class _InstrumentedWriteTool(BaseTool):
    """Test tool recording invocations to verify policy preview never executes."""

    def __init__(self, tool_id: str = "safe_note_write", *, read_only: bool = False) -> None:
        self._tool_id = tool_id
        self.read_only = read_only
        self.calls: List[Dict[str, Any]] = []

    @property
    def tool_id(self) -> str:
        return self._tool_id

    @property
    def description(self) -> str:
        return "Instrumented tool for preview safety verification."

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="path", type="string", description="Target file path", required=True),
            ToolParameter(name="content", type="string", description="Content payload", required=False),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.calls.append(dict(kwargs))
        return ToolResult(success=True, output="executed")


class _OfflineMockMCPClient:
    """Minimal offline MCP client for external availability preview tests."""

    def __init__(self, server_name: str, tools: List[Dict[str, Any]]) -> None:
        self.server_name = server_name
        self._tools = list(tools)
        self.connected = True
        self.invocations: List[tuple[str, Dict[str, Any]]] = []

    def is_connected(self) -> bool:
        return self.connected

    def is_available(self) -> bool:
        return self.connected

    async def list_tools(self) -> List[Dict[str, Any]]:
        return list(self._tools)

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self.invocations.append((name, dict(arguments)))
        return {"ok": True}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _run_cli(args: List[str], *, config_dir_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    import os

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


def test_structured_decision_explanations_reflect_actual_rules(tmp_path: Path) -> None:
    """Explanations derive stage, determining rule, and reason_code from the actual branch executed."""
    ledger = CentralExecutionLedger(db_path=tmp_path / "explain.db", raise_on_persistence_error=True)
    try:
        ledger.create_execution("Policy explain test", execution_id="exec_exp_1")

        # 1. Rule 1: Hard security boundary (smuggled untrusted override key)
        eng_r1 = ExecutionPolicyEngine(config=ExecutionPolicyConfig(policy_mode="autopilot"))
        d_smuggled = eng_r1.evaluate(
            "echo",
            {"text": "ok", "bypass_policy": True},
            execution_id="exec_exp_1",
            central_ledger=ledger,
        )
        exp_r1 = d_smuggled.explain()
        assert isinstance(exp_r1, PolicyExplanation)
        assert exp_r1.outcome == "deny"
        assert exp_r1.precedence_stage == "rule1:hard_security"
        assert exp_r1.determining_rule_id == "rule1:untrusted_policy_or_auth_keys"
        assert exp_r1.reason_code == "untrusted_policy_or_auth_keys"
        assert exp_r1.policy_mode == "autopilot"
        assert exp_r1.is_preview is False
        assert "normalized_arguments" not in exp_r1.to_safe_dict()

        # 2. Rule 4: Resource restriction (denied host)
        eng_r4 = ExecutionPolicyEngine(
            config=ExecutionPolicyConfig(
                policy_mode="autopilot",
                resource_restrictions={"denied_hosts": ["evil.example"]},
            )
        )
        d_host = eng_r4.evaluate(
            "fetch_web_page",
            {"url": "https://user:supersecret@evil.example/data?token=tok123"},
            execution_id="exec_exp_1",
            central_ledger=ledger,
        )
        exp_r4 = d_host.explain()
        assert exp_r4.precedence_stage == "rule4:resource_restriction"
        assert exp_r4.determining_rule_id == "rule4:denied_host"
        assert exp_r4.reason_code == "denied_host"
        serialized_r4 = json.dumps(exp_r4.to_safe_dict())
        assert "supersecret" not in serialized_r4
        assert "tok123" not in serialized_r4

        # 3. Rule 3: Explicit tool override deny vs allow
        eng_r3 = ExecutionPolicyEngine(
            config=ExecutionPolicyConfig(
                policy_mode="conservative",
                tool_overrides={"custom_op": {"outcome": "deny"}},
            )
        )
        d_override = eng_r3.evaluate("custom_op", {"x": 1})
        exp_r3 = d_override.explain()
        assert exp_r3.precedence_stage == "rule3:tool_override"
        assert exp_r3.determining_rule_id == "rule3:tool_override_outcome:deny"
        assert exp_r3.reason_code == "tool_override_deny"

        # 4. Rule 5: Risk policy default (irreversible & conservative write)
        eng_r5 = ExecutionPolicyEngine(config=ExecutionPolicyConfig(policy_mode="conservative"))
        d_irrev = eng_r5.evaluate("delete_file", {"path": "tmp.txt"})
        assert d_irrev.explain().precedence_stage == "rule5:risk_policy"
        assert d_irrev.explain().reason_code == "irreversible_action_blocked"

        # 5. Rule 6: Confirmation required -> verified with bound confirmation -> replayed
        d_req = eng_r5.evaluate("write_file", {"path": "a.txt", "content": "hi"}, workspace_root=tmp_path)
        exp_req = d_req.explain()
        assert exp_req.outcome == "require_confirmation"
        assert exp_req.precedence_stage == "rule6:confirmation_validation"
        assert exp_req.determining_rule_id == "rule6:confirmation_missing"
        assert exp_req.reason_code == "confirmation_missing"
        assert any("rule5:risk_policy:conservative:mutating->require_confirmation" == r for r in exp_req.rule_ids)

        raw_tok, rec = eng_r5.confirmation_store.issue_confirmation(
            tool_id=d_req.tool_id,
            arg_digest=d_req.arg_digest,
            resource_scope=d_req.resource_scope,
            policy_version=eng_r5.config.policy_version,
        )
        d_ok = eng_r5.evaluate(
            "write_file",
            {"path": "a.txt", "content": "hi"},
            workspace_root=tmp_path,
            confirmation_token=raw_tok,
            confirmation_id=rec.confirmation_id,
            consume_confirmation=True,
        )
        exp_ok = d_ok.explain()
        assert exp_ok.outcome == "allow"
        assert exp_ok.precedence_stage == "rule6:confirmation_validation"
        assert exp_ok.reason_code == "confirmation_consumed"
        assert raw_tok not in json.dumps(exp_ok.to_safe_dict())

        d_replay = eng_r5.evaluate(
            "write_file",
            {"path": "a.txt", "content": "hi"},
            workspace_root=tmp_path,
            confirmation_token=raw_tok,
            confirmation_id=rec.confirmation_id,
            consume_confirmation=True,
        )
        assert d_replay.explain().outcome == "deny"
        assert d_replay.explain().reason_code == "confirmation_replayed"

        # Direct manual construction without evaluation does not fabricate a precedence branch
        manual = ExecutionPolicyDecision(
            decision_id="pol_manual",
            outcome=PolicyDecisionOutcome.DENY,
            risk_level=RiskLevel.MUTATING,
            tool_id="write_file",
            normalized_arguments={},
            arg_digest="0" * 64,
            resource_scope=[],
            reason="Manually created",
        )
        assert manual.explain().precedence_stage == "unrecorded_stage"
    finally:
        ledger.close()


def test_preview_never_executes_mints_tokens_or_writes_ledger(tmp_path: Path) -> None:
    """preview_tool_policy and preview=True are strictly non-executing, non-minting, and non-logging."""
    reg = ToolRegistry()
    inst_tool = _InstrumentedWriteTool("safe_note_write", read_only=False)
    reg.register(inst_tool)

    store = ConfirmationStore()
    ledger = CentralExecutionLedger(db_path=tmp_path / "preview.db", raise_on_persistence_error=True)
    try:
        ledger.create_execution("Preview test", execution_id="exec_prev_1")
        before_events = len(ledger.query_authorization_decisions(limit=100))

        engine = ExecutionPolicyEngine(
            config=ExecutionPolicyConfig(policy_mode="conservative"),
            confirmation_store=store,
        )
        with (
            patch.object(store, "issue_confirmation", wraps=store.issue_confirmation) as spy_issue,
            patch.object(store, "verify_or_consume", wraps=store.verify_or_consume) as spy_verify,
        ):
            preview = preview_tool_policy(
                "safe_note_write",
                {"path": str(tmp_path / "note.txt"), "content": "preview payload"},
                tool_registry=reg,
                workspace_root=tmp_path,
                policy_engine=engine,
            )

            assert inst_tool.calls == []
            assert spy_issue.call_count == 0
            assert spy_verify.call_count == 0
            assert store._by_id == {}
            assert preview.decision.is_preview is True
            assert preview.explanation.is_preview is True
            assert preview.fact_status == "verified"
            assert preview.is_registered is True
            assert preview.decision.outcome == PolicyDecisionOutcome.REQUIRE_CONFIRMATION
            assert len(ledger.query_authorization_decisions(limit=100)) == before_events

            # Direct engine.evaluate(preview=True) enforces non-mutating behavior even if mutation flags passed
            called_cb = []
            direct_prev = engine.evaluate(
                "safe_note_write",
                {"path": str(tmp_path / "note.txt"), "content": "preview payload"},
                execution_id="exec_prev_1",
                central_ledger=ledger,
                record_to_ledger=True,
                consume_confirmation=True,
                confirm_callback=lambda d: called_cb.append(d) or True,
                preview=True,
            )
            assert direct_prev.is_preview is True
            assert direct_prev.outcome == PolicyDecisionOutcome.REQUIRE_CONFIRMATION
            assert called_cb == []
            assert spy_issue.call_count == 0
            assert spy_verify.call_count == 0
            assert len(ledger.query_authorization_decisions(limit=100)) == before_events
    finally:
        ledger.close()


def test_preview_validates_schema_ring_unknown_tools_and_external_availability(tmp_path: Path) -> None:
    """Preview enforces runtime authorization, unknown fail-closed semantics, and offline external detection."""
    reg = ToolRegistry()

    # Unknown tool fails closed with RiskLevel.UNKNOWN & fact_status="unknown"
    unk_prev = preview_tool_policy("nonexistent_tool_xyz", {"x": 1}, tool_registry=reg)
    assert unk_prev.is_registered is False
    assert unk_prev.fact_status == "unknown"
    assert unk_prev.auth_allowed is False
    assert unk_prev.auth_error_type == "TOOL_NOT_FOUND"
    assert unk_prev.decision.outcome == PolicyDecisionOutcome.DENY
    assert unk_prev.decision.risk_level == RiskLevel.UNKNOWN
    assert unk_prev.explanation.precedence_stage == "rule2:runtime_authorization"
    assert unk_prev.explanation.reason_code == "auth_denied:TOOL_NOT_FOUND"

    # Caller ring insufficient rejected at Rule 2 without executing shell
    ring_prev = preview_tool_policy(
        "shell",
        {"command": "echo hi"},
        tool_registry=reg,
        caller_ring=2,
        policy_config=ExecutionPolicyConfig(policy_mode="autopilot"),
    )
    assert ring_prev.auth_allowed is False
    assert ring_prev.auth_error_type == "AUTHORIZATION_DENIED"
    assert ring_prev.decision.outcome == PolicyDecisionOutcome.DENY
    assert ring_prev.explanation.precedence_stage == "rule2:runtime_authorization"

    # Malformed / missing required arguments caught by schema validator in preview
    bad_schema_prev = preview_tool_policy("echo", {}, tool_registry=reg)
    assert bad_schema_prev.auth_allowed is False
    assert bad_schema_prev.auth_error_type == "SCHEMA_VALIDATION_ERROR"
    assert bad_schema_prev.fact_status == "unverified"
    assert bad_schema_prev.decision.outcome == PolicyDecisionOutcome.DENY

    # External MCP tool offline/disconnected is reported UNVERIFIED and DENY without calling call_tool
    client = _OfflineMockMCPClient(
        "remote_ops",
        tools=[
            {
                "name": "deploy",
                "description": "Deploy service",
                "inputSchema": {
                    "type": "object",
                    "properties": {"env": {"type": "string"}},
                    "required": ["env"],
                    "additionalProperties": False,
                },
            }
        ],
    )
    from zelyr.tools.external import MCPExternalToolSource, build_external_tool_id

    reg.register_external_source(MCPExternalToolSource("remote_ops", client))
    asyncio.run(reg.discover_external_tools())
    ext_id = build_external_tool_id("remote_ops", "deploy")
    reg.authorize_external_tool(ext_id)
    client.connected = False

    ext_prev = preview_tool_policy(ext_id, {"env": "staging"}, tool_registry=reg)
    assert ext_prev.is_external is True
    assert ext_prev.external_available is False
    assert ext_prev.fact_status == "unverified"
    assert ext_prev.decision.outcome == PolicyDecisionOutcome.DENY
    assert ext_prev.explanation.auth_error_type == "TOOL_UNAVAILABLE"
    assert client.invocations == []

    # TaskLocalToolScope preview works in isolation without consulting global ToolRegistry
    scope = TaskLocalToolScope(
        tool_names=["local_calc"],
        tool_schemas=[
            {
                "type": "function",
                "function": {
                    "name": "local_calc",
                    "parameters": {
                        "type": "object",
                        "properties": {"expr": {"type": "string"}},
                        "required": ["expr"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        tool_rings={"local_calc": 1},
    )
    scope_prev = preview_tool_policy(
        "local_calc",
        {"expr": "1+1"},
        task_local_scope=scope,
        caller_ring=1,
        policy_config=ExecutionPolicyConfig(policy_mode="conservative"),
    )
    assert scope_prev.is_registered is True
    assert scope_prev.auth_allowed is True
    assert scope_prev.decision.outcome == PolicyDecisionOutcome.REQUIRE_CONFIRMATION

    # Smuggled policy bypass keys fail closed at Rule 1 / Rule 2 during preview
    smug_prev = preview_tool_policy(
        "echo",
        {"text": "hi", "bypass_policy": True},
        tool_registry=reg,
        policy_config=ExecutionPolicyConfig(policy_mode="autopilot"),
    )
    assert smug_prev.decision.outcome == PolicyDecisionOutcome.DENY
    assert smug_prev.explanation.precedence_stage == "rule1:hard_security"


def test_configuration_inspection_provenance_and_redaction(tmp_path: Path) -> None:
    """inspect_policy_configuration reports config_file vs derived vs default sources and redacts secrets."""
    # 1. File-backed ZelyrConfig
    cfg_path = tmp_path / "config.json"
    cfg = ZelyrConfig(
        auth_mode=AuthMode.STRICT,
        execution_policy_mode="balanced",
        execution_policy_tool_overrides={
            "custom_api": {"outcome": "require_confirmation", "api_key": "sk-secret-override-999"}
        },
        execution_policy_resource_restrictions={
            "denied_hosts": ["https://admin:topsecret@internal.corp/api?token=abc123"],
            "webhook_secret": "whsec_do_not_leak",
        },
        execution_policy_confirmation_ttl_seconds=180.0,
    )
    cfg.save(cfg_path)
    loaded = ZelyrConfig.load(cfg_path)

    view = inspect_policy_configuration(zelyr_config=loaded)
    assert view["policy_mode"] == "balanced"
    assert view["config_sources"]["policy_mode"] == f"config_file:{cfg_path}"
    assert view["config_sources"]["confirmation_ttl_seconds"] == f"config_file:{cfg_path}"
    assert view["config_sources"]["max_confirmation_lifetime_seconds"] == f"config_file:{cfg_path}"

    # Minimal file omits TTL fields -> omitted fields clearly show "default" source
    min_cfg_path = tmp_path / "min_config.json"
    min_cfg_path.write_text(
        json.dumps({"auth_mode": "strict", "execution_policy_mode": "balanced"}),
        encoding="utf-8",
    )
    min_view = inspect_policy_configuration(zelyr_config=ZelyrConfig.load(min_cfg_path))
    assert min_view["config_sources"]["policy_mode"] == f"config_file:{min_cfg_path}"
    assert min_view["config_sources"]["confirmation_ttl_seconds"] == "default"
    assert min_view["config_sources"]["max_confirmation_lifetime_seconds"] == "default"
    assert view["effective_risk_matrix"]["read_only"]["workspace_scoped"] == "allow"
    assert view["effective_risk_matrix"]["reversible_write"]["workspace_scoped"] == "allow"
    assert view["effective_risk_matrix"]["reversible_write"]["outside_workspace"] == "require_confirmation"
    assert view["effective_risk_matrix"]["unknown"]["workspace_scoped"] == "deny"

    dumped = json.dumps(view)
    assert "sk-secret-override-999" not in dumped
    assert "topsecret" not in dumped
    assert "abc123" not in dumped
    assert "whsec_do_not_leak" not in dumped
    assert view["tool_overrides"]["custom_api"]["api_key"] == "[REDACTED]"
    assert view["resource_restrictions"]["webhook_secret"] == "[REDACTED]"

    # 2. Derived from auth_mode when execution_policy_mode is unset
    derived_cfg = ZelyrConfig(auth_mode=AuthMode.STRICT)
    derived_view = inspect_policy_configuration(zelyr_config=derived_cfg, governor_strict_mode=False)
    assert derived_view["policy_mode"] == "strict"
    assert derived_view["config_sources"]["policy_mode"] == "derived:auth_mode:strict"

    # 3. Derived from GovernorService strict mode when auth_mode is basic
    gov_cfg = ZelyrConfig(auth_mode=AuthMode.BASIC)
    gov_view = inspect_policy_configuration(zelyr_config=gov_cfg, governor_strict_mode=True)
    assert gov_view["policy_mode"] == "strict"
    assert gov_view["governor_strict_mode"] is True
    assert gov_view["config_sources"]["policy_mode"] == "derived:governor_strict_mode"


def test_ledger_policy_history_filtering_ordering_legacy_malformed_and_redaction(tmp_path: Path) -> None:
    """Ledger queries filter strictly by AUTHORIZATION_DECISION, order newest-first, handle malformed/legacy rows, and redact."""
    db_path = tmp_path / "policy_hist.db"
    ledger = CentralExecutionLedger(db_path=db_path, raise_on_persistence_error=True)
    try:
        ledger.create_execution("History test 1", execution_id="exec_h1")
        ledger.create_execution("History test 2", execution_id="exec_h2")

        # Record non-policy events interleaved with policy events
        ledger.record_event("exec_h1", LedgerEventType.PLANNING_DECISION, status="ok")
        eng = ExecutionPolicyEngine(config=ExecutionPolicyConfig(policy_mode="conservative"))

        eng.evaluate("delete_file", {"path": "x.txt"}, execution_id="exec_h1", central_ledger=ledger)
        eng.evaluate("write_file", {"path": "y.txt", "content": "a"}, execution_id="exec_h1", central_ledger=ledger)
        eng.evaluate("echo", {"text": "hello"}, execution_id="exec_h2", central_ledger=ledger)
        ledger.record_event("exec_h2", LedgerEventType.TOOL_EXECUTION_RESULT, tool_id="echo", status="success")
        eng.evaluate("echo", {"text": "hello2"}, execution_id="exec_h2", central_ledger=ledger)

        # Verify outcome filter applies BEFORE limit (2 newest are ALLOW; 1 older is DENY)
        deny_hist = query_policy_history(ledger, limit=1, outcome="deny")
        assert deny_hist["count"] == 1
        assert deny_hist["entries"][0]["tool_id"] == "delete_file"
        assert deny_hist["entries"][0]["outcome"] == "deny"

        # Newest-first deterministic order and exclusion of TURN_START / TOOL_RESULT
        all_hist = query_policy_history(ledger, limit=20)
        assert all_hist["count"] == 4
        assert [e["tool_id"] for e in all_hist["entries"]] == ["echo", "echo", "write_file", "delete_file"]
        assert all(e["event_id"] for e in all_hist["entries"])

        # Tool & execution filters
        assert query_policy_history(ledger, execution_id="exec_h1")["count"] == 2
        assert query_policy_history(ledger, tool_id="write_file")["count"] == 1
    finally:
        ledger.close()

    # Inject legacy partial row, malformed JSON row, list-JSON row, null-JSON row, and leaky secret metadata
    with sqlite3.connect(str(db_path)) as conn:
        now = time.time() + 100.0
        conn.execute(
            """
            INSERT INTO execution_events
            (event_id, execution_id, sequence_number, timestamp, event_type, tool_id, status, is_external, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "ev_legacy",
                "exec_h1",
                900,
                now + 1,
                LedgerEventType.AUTHORIZATION_DECISION.value,
                "shell",
                "denied",
                0,
                json.dumps(
                    {
                        "policy_decision": "deny",
                        "risk_level": "privileged",
                        "normalized_arguments": {"secret_cmd": "raw_arg_secret_888"},
                        "confirmation_token": "tok_must_not_appear",
                        "resource_scope": ["url:https://alice:pw999@example.com/hook?api_key=k_secret"],
                    }
                ),
            ),
        )
        conn.execute(
            """
            INSERT INTO execution_events
            (event_id, execution_id, sequence_number, timestamp, event_type, tool_id, status, is_external, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "ev_badjson",
                "exec_h1",
                901,
                now + 2,
                LedgerEventType.AUTHORIZATION_DECISION.value,
                "custom_bad",
                "denied",
                0,
                "{not-valid-json",
            ),
        )
        conn.execute(
            """
            INSERT INTO execution_events
            (event_id, execution_id, sequence_number, timestamp, event_type, tool_id, status, is_external, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "ev_listjson",
                "exec_h1",
                902,
                now + 3,
                LedgerEventType.AUTHORIZATION_DECISION.value,
                "custom_list",
                "requires_confirmation",
                0,
                "[1, 2, 3]",
            ),
        )
        conn.execute(
            """
            INSERT INTO execution_events
            (event_id, execution_id, sequence_number, timestamp, event_type, tool_id, status, is_external, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "ev_nulljson",
                "exec_h1",
                903,
                now + 4,
                LedgerEventType.AUTHORIZATION_DECISION.value,
                "custom_null",
                "allowed",
                0,
                "null",
            ),
        )
        conn.commit()

    reopened = CentralExecutionLedger(db_path=db_path, raise_on_persistence_error=False)
    try:
        full = query_policy_history(reopened, limit=20)
        assert full["persistence_errors"] == []
        by_id = {e["event_id"]: e for e in full["entries"]}
        assert by_id["ev_legacy"]["legacy_metadata"] is True
        assert by_id["ev_legacy"]["malformed_metadata"] is False
        assert by_id["ev_badjson"]["malformed_metadata"] is True
        assert by_id["ev_listjson"]["malformed_metadata"] is True
        assert by_id["ev_nulljson"]["malformed_metadata"] is False
        assert by_id["ev_nulljson"]["outcome"] == "allow"

        full_dump = json.dumps(full)
        assert "raw_arg_secret_888" not in full_dump
        assert "tok_must_not_appear" not in full_dump
        assert "pw999" not in full_dump
        assert "k_secret" not in full_dump
    finally:
        reopened.close()


def test_corrupted_ledger_surfaces_persistence_errors_honestly(tmp_path: Path) -> None:
    """Corrupted ledger DB exposes persistence_errors in API and fails non-zero in CLI."""
    corrupt_db = tmp_path / "corrupt_ledger.db"
    corrupt_db.write_bytes(b"THIS IS NOT A VALID SQLITE DATABASE FILE" * 16)

    ledger = CentralExecutionLedger(db_path=corrupt_db, raise_on_persistence_error=False)
    try:
        hist = query_policy_history(ledger, limit=10)
        assert hist["entries"] == []
        assert len(hist["persistence_errors"]) >= 1
    finally:
        ledger.close()

    res_cli = _run_cli(["policy", "history", "--db-path", str(corrupt_db), "--json"])
    assert res_cli.returncode != 0
    parsed = json.loads(res_cli.stdout)
    assert len(parsed["persistence_errors"]) >= 1


def test_cli_policy_subcommands_end_to_end(tmp_path: Path) -> None:
    """Real CLI subprocesses for policy show, explain, and history work offline with lazy imports."""
    cfg_dir = tmp_path / "cfg_home"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = ZelyrConfig(
        auth_mode=AuthMode.STRICT,
        execution_policy_mode="balanced",
        execution_policy_tool_overrides={"my_secret_tool": {"outcome": "deny", "token": "secret_cli_tok"}},
    )
    cfg_file = cfg_dir / "config.json"
    cfg.save(cfg_file)

    # 1. zelyr policy show --json
    r_show = _run_cli(["policy", "show", "--config-path", str(cfg_file), "--json"], config_dir_path=cfg_dir)
    assert r_show.returncode == 0, r_show.stderr
    show_data = json.loads(r_show.stdout)
    assert show_data["policy_mode"] == "balanced"
    assert show_data["tool_overrides"]["my_secret_tool"]["token"] == "[REDACTED]"
    assert "secret_cli_tok" not in r_show.stdout

    # 2. zelyr policy explain echo --args '{"text":"hello"}' --json
    r_explain_allow = _run_cli(
        ["policy", "explain", "echo", "--args", '{"text": "hello"}', "--config-path", str(cfg_file), "--json"],
        config_dir_path=cfg_dir,
    )
    assert r_explain_allow.returncode == 0, r_explain_allow.stderr
    exp_allow = json.loads(r_explain_allow.stdout)
    assert exp_allow["is_preview"] is True
    assert exp_allow["fact_status"] == "verified"
    assert exp_allow["decision"]["outcome"] == "allow"
    assert exp_allow["explanation"]["risk_level"] == "read_only"

    # 3. zelyr policy explain shell under conservative mode -> require_confirmation, human output
    r_explain_human = _run_cli(
        [
            "policy",
            "explain",
            "shell",
            "--args",
            '{"command": "echo hello"}',
            "--policy-mode",
            "conservative",
            "--config-path",
            str(cfg_file),
        ],
        config_dir_path=cfg_dir,
    )
    assert r_explain_human.returncode == 0, r_explain_human.stderr
    assert "REQUIRE_CONFIRMATION" in r_explain_human.stdout
    assert "shell" in r_explain_human.stdout
    assert "confirmation_missing" in r_explain_human.stdout

    # 3b. Schema-invalid arguments fail closed under Rule 2 during CLI preview
    r_explain_bad = _run_cli(
        [
            "policy",
            "explain",
            "write_file",
            "--args",
            '{"path": "demo.txt", "content": "hi"}',
            "--json",
        ],
        config_dir_path=cfg_dir,
    )
    assert r_explain_bad.returncode == 0, r_explain_bad.stderr
    bad_exp = json.loads(r_explain_bad.stdout)
    assert bad_exp["decision"]["outcome"] == "deny"
    assert bad_exp["auth_error_type"] == "SCHEMA_VALIDATION_ERROR"

    # 4. zelyr policy history with populated DB
    db_path = tmp_path / "cli_pol_hist.db"
    ledger = CentralExecutionLedger(db_path=db_path, raise_on_persistence_error=True)
    try:
        ledger.create_execution("CLI test", execution_id="exec_cli_1")
        eng = ExecutionPolicyEngine(config=ExecutionPolicyConfig(policy_mode="conservative"))
        eng.evaluate("write_file", {"path": "x.txt", "content": "a"}, execution_id="exec_cli_1", central_ledger=ledger)
        eng.evaluate("delete_file", {"path": "x.txt"}, execution_id="exec_cli_1", central_ledger=ledger)
    finally:
        ledger.close()

    r_hist = _run_cli(
        ["policy", "history", "--db-path", str(db_path), "--outcome", "deny", "--limit", "5", "--json"],
        config_dir_path=cfg_dir,
    )
    assert r_hist.returncode == 0, r_hist.stderr
    hist_json = json.loads(r_hist.stdout)
    assert hist_json["count"] == 1
    assert hist_json["entries"][0]["tool_id"] == "delete_file"


def test_repl_policy_slash_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """REPL /policy show, explain, and history dispatch cleanly without executing tools."""
    from types import SimpleNamespace
    from zelyr.cli.repl import COMMANDS, InlineRepl

    assert "/policy" in COMMANDS

    reg = ToolRegistry()
    inst = _InstrumentedWriteTool("repl_preview_tool", read_only=False)
    reg.register(inst)
    db_path = tmp_path / "repl_pol.db"
    ledger = CentralExecutionLedger(db_path=db_path, raise_on_persistence_error=True)
    try:
        ledger.create_execution("REPL test", execution_id="exec_repl_1")
        cfg = ZelyrConfig(auth_mode=AuthMode.STRICT)
        pol_engine = ExecutionPolicyEngine(config=ExecutionPolicyConfig.from_zelyr_config(cfg))
        pol_engine.evaluate(
            "delete_file",
            {"path": "z.txt"},
            execution_id="exec_repl_1",
            central_ledger=ledger,
        )

        repl = InlineRepl.__new__(InlineRepl)
        repl.config = cfg
        repl.orchestrator = SimpleNamespace(
            policy_engine=pol_engine,
            central_ledger=ledger,
            tool_registry=reg,
        )

        assert repl.handle_command("/policy show") is True
        out_show = capsys.readouterr().out
        assert "policy_mode:" in out_show and "strict" in out_show

        assert repl.handle_command('/policy explain repl_preview_tool \'{"path":"a.txt","content":"hi"}\' --json') is True
        out_exp = capsys.readouterr().out
        assert "REQUIRE_CONFIRMATION" in out_exp or '"require_confirmation"' in out_exp
        assert inst.calls == []

        assert repl.handle_command("/policy history --outcome deny --limit 5") is True
        out_hist = capsys.readouterr().out
        assert "delete_file" in out_hist and "deny" in out_hist
    finally:
        ledger.close()
