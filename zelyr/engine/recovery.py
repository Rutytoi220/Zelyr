"""Bounded, verification-driven recovery system for failed and interrupted TaskGraph nodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import uuid

from zelyr.engine.error_classifier import classify_tool_error
from zelyr.engine.execution_ledger import (
    ActionRecord,
    ActionState,
    CentralExecutionLedger,
    LedgerEventType,
    TurnExecutionLedger,
    VerificationStatus,
    _extract_action_target,
    criterion_applies_to_target,
    parse_acceptance_criteria,
    sanitize_ledger_payload,
    sanitize_ledger_text,
)
from zelyr.engine.observation_framing import (
    EvidenceLevel,
    ObservationExecutionStatus,
    ObservationFramingEngine,
    StructuredObservation,
)
from zelyr.engine.task_graph import (
    _FORBIDDEN_GRAPH_KEYS,
    RECOVERY_STATE_VERSION,
    SUPPORTED_RECOVERY_VERSIONS,
    _VALID_RECOVERY_DECISIONS,
    _validate_node_recovery_state,
    CorruptedTaskGraphError,
    NodeConflictProfile,
    TaskGraph,
    TaskGraphConflictError,
    TaskGraphError,
    TaskGraphValidationError,
    TaskNode,
    TaskNodeStatus,
    UnsupportedTaskGraphVersionError,
    _strip_forbidden_graph_keys,
    extract_node_conflict_profile,
)

MAX_CONFIGURABLE_RECOVERY_ATTEMPTS: int = 10

_SUPPORTED_OUTCOME_CHECKS: frozenset[str] = frozenset(
    {
        "file_exists",
        "exists",
        "file_contains",
        "contains",
        "file_checksum",
        "checksum",
        "sha256",
        "file_json",
        "json",
        "process_running",
        "port_listening",
        "command_output",
    }
)

_NON_RECOVERABLE_FAILURE_CATEGORIES: frozenset[str] = frozenset(
    {
        "AUTHORIZATION_DENIED",
        "AUTHORIZATION_ERROR",
        "PROHIBITED_BY_USER",
        "CANCELLED",
        "TASK_CANCELLED",
        "CYCLE_DETECTED",
        "PERMISSION_DENIED",
    }
)

_REPLAN_FAILURE_CATEGORIES: frozenset[str] = frozenset(
    {
        "TOOL_NOT_FOUND",
        "INVALID_TOOL_NAME",
        "UNKNOWN_TOOL",
        "TOOL_UNAVAILABLE",
        "DESKTOP_UNAVAILABLE",
        "BROWSER_UNAVAILABLE",
        "PROPOSAL_EXPIRED",
    }
)

_PRE_DISPATCH_SAFE_RETRY_CATEGORIES: frozenset[str] = frozenset(
    {
        "FAILED_BEFORE_EXECUTION",
        "TRANSIENT_FAILURE",
        "TRANSIENT_PRE_DISPATCH",
        "RATE_LIMITED",
        "TEMPORARY_UNAVAILABLE",
        "CONNECTION_ERROR",
        "NETWORK_ERROR",
    }
)

_DESTRUCTIVE_ACTION_NAMES: frozenset[str] = frozenset(
    {
        "delete",
        "remove",
        "unlink",
        "rmdir",
        "rm",
        "drop",
        "truncate",
        "purge",
        "wipe",
        "kill",
        "terminate",
        "append",
    }
)

_DESTRUCTIVE_TOOL_NAMES: frozenset[str] = frozenset({"delete_file", "remove_file"})


class RecoveryDecision(str, Enum):
    """Explicit recovery classification computed strictly from evidence."""

    RETRY_SAFE = "retry_safe"
    RETRY_AFTER_REVALIDATION = "retry_after_revalidation"
    REPLAN_NODE = "replan_node"
    REQUIRE_CONFIRMATION = "require_confirmation"
    BLOCKED = "blocked"
    ALREADY_COMPLETED = "already_completed"
    UNRESOLVED = "unresolved"


_VALID_RECOVERY_DECISIONS: frozenset[str] = frozenset(d.value for d in RecoveryDecision)


class RecoveryError(TaskGraphError):
    """Base error for graph recovery policy, validation, and transition failures."""


class CorruptedRecoveryStateError(CorruptedTaskGraphError):
    """Raised when persisted node or checkpoint recovery state is malformed."""


class UnsupportedRecoveryVersionError(UnsupportedTaskGraphVersionError):
    """Raised when persisted recovery metadata uses an unsupported schema version."""


def _sanitize_recovery_text(raw: Any, *, max_bytes: int = 256) -> str:
    txt, _, _ = sanitize_ledger_text("" if raw is None else str(raw), max_bytes=max_bytes)
    return txt or ""


def validate_recovery_state_dict(
    raw: Any,
    *,
    context_label: str = "recovery_state",
) -> Dict[str, Any]:
    """Fail closed on malformed, unsafe, or unsupported-version recovery metadata."""
    return _validate_node_recovery_state(raw, context_label=context_label)


@dataclass(frozen=True)
class RecoveryPolicy:
    """Trusted application configuration governing bounded TaskGraph recovery."""

    max_attempts: int = 2
    backoff_base_seconds: float = 0.05
    backoff_max_seconds: float = 2.0
    deadline_seconds: Optional[float] = 120.0
    allow_auto_retry: bool = True
    allow_retry_after_revalidation: bool = True
    require_confirmation_for_ambiguous: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int):
            raise RecoveryError(f"RecoveryPolicy.max_attempts must be an integer, got {self.max_attempts!r}.")
        if self.max_attempts < 0 or self.max_attempts > MAX_CONFIGURABLE_RECOVERY_ATTEMPTS:
            raise RecoveryError(
                f"RecoveryPolicy.max_attempts ({self.max_attempts}) must be between 0 and {MAX_CONFIGURABLE_RECOVERY_ATTEMPTS}."
            )
        if (
            isinstance(self.backoff_base_seconds, bool)
            or not isinstance(self.backoff_base_seconds, (int, float))
            or float(self.backoff_base_seconds) < 0.0
        ):
            raise RecoveryError(f"Invalid backoff_base_seconds: {self.backoff_base_seconds!r}.")
        if (
            isinstance(self.backoff_max_seconds, bool)
            or not isinstance(self.backoff_max_seconds, (int, float))
            or float(self.backoff_max_seconds) < 0.0
        ):
            raise RecoveryError(f"Invalid backoff_max_seconds: {self.backoff_max_seconds!r}.")
        if self.deadline_seconds is not None and (
            isinstance(self.deadline_seconds, bool)
            or not isinstance(self.deadline_seconds, (int, float))
            or float(self.deadline_seconds) <= 0.0
        ):
            raise RecoveryError(f"Invalid deadline_seconds: {self.deadline_seconds!r}.")

    @classmethod
    def from_config(
        cls,
        config: Optional[Any] = None,
        *,
        max_attempts: Optional[int] = None,
        backoff_base_seconds: Optional[float] = None,
        backoff_max_seconds: Optional[float] = None,
        deadline_seconds: Optional[float] = None,
        allow_auto_retry: Optional[bool] = None,
        allow_retry_after_revalidation: Optional[bool] = None,
    ) -> "RecoveryPolicy":
        """Construct a validated RecoveryPolicy from trusted application ZelyrConfig."""
        cfg_max = getattr(config, "graph_recovery_max_attempts", 2) if config is not None else 2
        cfg_base = getattr(config, "graph_recovery_backoff_base_seconds", 0.05) if config is not None else 0.05
        cfg_bmax = getattr(config, "graph_recovery_backoff_max_seconds", 2.0) if config is not None else 2.0
        cfg_dl = getattr(config, "graph_recovery_deadline_seconds", 120.0) if config is not None else 120.0

        return cls(
            max_attempts=cfg_max if max_attempts is None else max_attempts,
            backoff_base_seconds=float(cfg_base if backoff_base_seconds is None else backoff_base_seconds),
            backoff_max_seconds=float(cfg_bmax if backoff_max_seconds is None else backoff_max_seconds),
            deadline_seconds=float(cfg_dl if deadline_seconds is None else deadline_seconds)
            if (deadline_seconds is not None or cfg_dl is not None)
            else None,
            allow_auto_retry=True if allow_auto_retry is None else bool(allow_auto_retry),
            allow_retry_after_revalidation=(
                True if allow_retry_after_revalidation is None else bool(allow_retry_after_revalidation)
            ),
        )

    def compute_backoff_seconds(self, attempt_number: int) -> float:
        """Compute bounded exponential backoff before attempt ``attempt_number`` (1-based)."""
        if float(self.backoff_base_seconds) <= 0.0 or float(self.backoff_max_seconds) <= 0.0:
            return 0.0
        exp = max(0, int(attempt_number) - 1)
        delay = float(self.backoff_base_seconds) * (2**exp)
        return min(float(self.backoff_max_seconds), delay)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecoveryAssessment:
    """Structured, evidence-driven recovery assessment for a single TaskGraph node."""

    execution_id: str
    node_id: str
    failure_category: str
    previous_status: str
    structured_observation: Optional[Dict[str, Any]] = None
    evidence_level: str = EvidenceLevel.UNKNOWN.value
    execution_started: bool = False
    may_have_side_effect: bool = True
    verified_outcome: bool = False
    recoverable: bool = False
    recovery_guidance: Dict[str, Any] = field(default_factory=dict)
    recovery_decision: str = RecoveryDecision.BLOCKED.value
    reason: str = ""
    attempt_count: int = 0
    max_attempts: int = 2
    retry_deadline: Optional[float] = None
    backoff_seconds: float = 0.0
    reconciliation_required: bool = False
    revalidation_passed: Optional[bool] = None
    verification_result: Optional[Dict[str, Any]] = None
    evidence_refs: List[Dict[str, Any]] = field(default_factory=list)
    proposed_next_action: Optional[str] = None
    affected_dependents: List[str] = field(default_factory=list)
    blocked_reason: Optional[str] = None
    call_id: Optional[str] = None
    policy_outcome: Optional[str] = None
    policy_decision: Optional[Dict[str, Any]] = None
    proposal_id: Optional[str] = None
    tool_name: Optional[str] = None
    normalized_arguments: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _get_prop_field(prop: Any, key: str, default: Any = None) -> Any:
    if isinstance(prop, dict):
        return prop.get(key, default)
    return getattr(prop, key, default)


def _build_proposal_lookup(
    proposals: Optional[Union[Dict[str, Any], Iterable[Any]]] = None,
    turn_ledger: Optional[TurnExecutionLedger] = None,
) -> Dict[str, Any]:
    prop_map: Dict[str, Any] = {}
    items: List[Any] = []
    if isinstance(proposals, dict):
        items = list(proposals.values())
    elif proposals is not None:
        items = list(proposals)
    elif turn_ledger is not None and hasattr(turn_ledger, "get_tracked_proposals"):
        items = list(turn_ledger.get_tracked_proposals())

    for item in items:
        pid = _get_prop_field(item, "proposal_id")
        cid = _get_prop_field(item, "original_call_id")
        if pid:
            prop_map[str(pid)] = item
        if cid:
            prop_map[str(cid)] = item
    return prop_map


def _resolve_node_proposal(node: TaskNode, prop_map: Dict[str, Any]) -> Optional[Any]:
    for pid in node.proposal_ids:
        if pid in prop_map:
            return prop_map[pid]
    return prop_map.get(node.node_id)


def _collect_node_criteria(
    node: TaskNode,
    proposal: Optional[Any],
    graph_acceptance_criteria: Optional[str],
    tool_name: str,
    tool_args: Dict[str, Any],
) -> List[Dict[str, Any]]:
    criteria: List[Dict[str, Any]] = []
    prop_spec = _get_prop_field(proposal, "acceptance_criteria") if proposal is not None else None
    if prop_spec:
        criteria.extend(parse_acceptance_criteria(str(prop_spec)))

    if not criteria and getattr(node, "verification_requirement", None):
        for crit in parse_acceptance_criteria(str(node.verification_requirement)):
            chk = str(crit.get("check") or "").strip().lower()
            tgt = str(crit.get("target") or "").strip()
            if chk in _SUPPORTED_OUTCOME_CHECKS and tgt:
                criteria.append(crit)

    tool_target = _extract_action_target(tool_name, tool_args) if tool_name else None
    if graph_acceptance_criteria and tool_target:
        for crit in parse_acceptance_criteria(str(graph_acceptance_criteria)):
            if criterion_applies_to_target(crit, tool_target) and crit not in criteria:
                criteria.append(crit)
    return criteria


def _is_trusted_idempotent(tool_obj: Optional[Any], tool_schema: Optional[Dict[str, Any]], args: Dict[str, Any]) -> bool:
    act = str(args.get("action") or args.get("operation") or "").strip().lower()
    if act in _DESTRUCTIVE_ACTION_NAMES:
        return False
    if tool_obj is not None and getattr(tool_obj, "idempotent", False) is True:
        return True
    if isinstance(tool_schema, dict):
        fn = tool_schema.get("function") if isinstance(tool_schema.get("function"), dict) else tool_schema
        if tool_schema.get("idempotent") is True or fn.get("idempotent") is True:
            return True
    return False


def _is_destructive_operation(tool_name: str, args: Dict[str, Any]) -> bool:
    tname = (tool_name or "").strip().lower()
    if tname in _DESTRUCTIVE_TOOL_NAMES:
        return True
    act = str(args.get("action") or args.get("operation") or "").strip().lower()
    return act in _DESTRUCTIVE_ACTION_NAMES


def build_graph_recovery_overview(graph: Optional[TaskGraph]) -> Dict[str, Any]:
    """Summarize node recovery states, blocked dependents, and pending actions for telemetry/CLI."""
    if graph is None:
        return {
            "failed_nodes": [],
            "recovering_nodes": [],
            "blocked_dependents": [],
            "total_attempts": 0,
        }

    failed_items: List[Dict[str, Any]] = []
    recovering_items: List[Dict[str, Any]] = []
    blocked_deps: List[Dict[str, Any]] = []
    total_attempts = 0
    node_map = {n.node_id: n for n in graph.nodes}

    for node in sorted(graph.nodes, key=lambda n: (n.order_index, n.node_id)):
        rs = dict(node.recovery_state) if isinstance(node.recovery_state, dict) else {}
        att = int(rs.get("attempt_count") or 0)
        total_attempts += att
        if rs:
            recovering_items.append(
                {
                    "node_id": node.node_id,
                    "execution_status": node.execution_status,
                    "attempt_count": att,
                    "max_attempts": rs.get("max_attempts"),
                    "recovery_decision": rs.get("recovery_decision"),
                    "recovery_status": rs.get("recovery_status"),
                    "previous_failure_category": rs.get("previous_failure_category"),
                    "next_recovery_action": rs.get("next_recovery_action"),
                    "may_have_side_effect": rs.get("may_have_side_effect"),
                    "reconciliation_required": rs.get("reconciliation_required"),
                    "retry_deadline": rs.get("retry_deadline"),
                }
            )
        if node.execution_status == TaskNodeStatus.FAILED.value or node.contradictory:
            failed_items.append(
                {
                    "node_id": node.node_id,
                    "execution_status": node.execution_status,
                    "error": (node.failure_info or {}).get("error") or node.blocked_reason,
                    "failure_category": rs.get("previous_failure_category")
                    or (node.failure_info or {}).get("error_type"),
                    "recovery_status": rs.get("recovery_status")
                    or ("reconciliation_required" if node.contradictory else "unassessed"),
                    "recovery_decision": rs.get("recovery_decision"),
                    "attempt_count": att,
                    "next_recovery_action": rs.get("next_recovery_action"),
                    "reconciliation_required": bool(rs.get("reconciliation_required", node.contradictory)),
                }
            )
        if node.execution_status == TaskNodeStatus.BLOCKED.value:
            blocking_prereqs = [
                dep_id
                for dep_id in node.dependency_ids
                if dep_id in node_map
                and node_map[dep_id].execution_status != TaskNodeStatus.COMPLETED_VERIFIED.value
            ]
            blocked_deps.append(
                {
                    "node_id": node.node_id,
                    "depends_on": list(node.dependency_ids),
                    "blocking_prerequisites": blocking_prereqs,
                    "reason": node.blocked_reason or "blocked",
                    "contradictory": bool(node.contradictory),
                }
            )

    return {
        "failed_nodes": failed_items,
        "recovering_nodes": recovering_items,
        "blocked_dependents": blocked_deps,
        "total_attempts": total_attempts,
    }


class RecoveryEngine:
    """Evidence-driven assessment and bounded mutation engine for TaskGraph recovery."""

    @classmethod
    def assess_node_recovery(
        cls,
        graph: TaskGraph,
        node_id: str,
        *,
        central_ledger: Optional[CentralExecutionLedger] = None,
        turn_ledger: Optional[TurnExecutionLedger] = None,
        proposals: Optional[Union[Dict[str, Any], Iterable[Any]]] = None,
        policy: Optional[RecoveryPolicy] = None,
        tool_registry: Optional[Any] = None,
        active_schemas: Optional[List[Dict[str, Any]]] = None,
        known_tools: Optional[Iterable[str]] = None,
        caller_ring: Optional[int] = None,
        allowed_tools: Optional[Iterable[str]] = None,
        task_local_scope: Optional[Any] = None,
        acceptance_criteria: Optional[str] = None,
        verifier_fn: Optional[Callable[..., Any]] = None,
        reconcile_state: bool = False,
        confirm_recovery: bool = False,
        zelyr_config: Optional[Any] = None,
        policy_engine: Optional[Any] = None,
        policy_config: Optional[Any] = None,
        workspace_root: Optional[Any] = None,
        confirmation_token: Optional[str] = None,
        confirmation_id: Optional[str] = None,
        confirm_callback: Optional[Callable[[Any], Any]] = None,
        now_ts: Optional[float] = None,
    ) -> RecoveryAssessment:
        """Analyze ledger events, observations, verifier state, dependencies, and auth to choose recovery."""
        from zelyr.engine.verifier import OutcomeVerifier
        from zelyr.tool_registry import authorize_and_validate_tool_call

        node = graph.get_node(node_id)
        if node is None:
            raise TaskGraphValidationError(f"Unknown task node '{node_id}' for recovery assessment.")

        eff_policy = policy or RecoveryPolicy()
        eff_now = float(now_ts if now_ts is not None else time.time())
        exec_id = graph.execution_id
        prop_map = _build_proposal_lookup(proposals, turn_ledger=turn_ledger)
        proposal = _resolve_node_proposal(node, prop_map)

        call_id: Optional[str] = _get_prop_field(proposal, "original_call_id")
        if not call_id and node.observation_refs:
            for obs_r in reversed(node.observation_refs):
                if isinstance(obs_r, dict) and obs_r.get("call_id"):
                    call_id = str(obs_r["call_id"])
                    break
        if not call_id and node.proposal_ids:
            call_id = str(node.proposal_ids[0])
        proposal_id = str(_get_prop_field(proposal, "proposal_id") or (node.proposal_ids[0] if node.proposal_ids else "") or "") or None
        func_name = str(_get_prop_field(proposal, "tool_id") or "").strip()
        raw_args = _get_prop_field(proposal, "normalized_arguments")
        if not isinstance(raw_args, dict):
            raw_args = _get_prop_field(proposal, "arguments") or {}
        func_args = dict(raw_args) if isinstance(raw_args, dict) else {}
        ext_src = _get_prop_field(proposal, "external_source") or node.external_source

        # 1. Inspect central execution ledger events for this node / call_id
        events = central_ledger.get_events(exec_id) if central_ledger is not None else []
        relevant_call_ids = {cid for cid in (call_id, *(r.get("call_id") for r in node.observation_refs if isinstance(r, dict))) if cid}
        node_events = []
        current_attempt_events = []
        ledger_attempts = 0
        latest_deadline: Optional[float] = None
        for ev in events:
            meta = dict(ev.metadata or {})
            ev_nid = meta.get("node_id")
            if ev.correlation_id in relevant_call_ids or ev_nid == node_id:
                node_events.append(ev)
                if ev.event_type == LedgerEventType.RECOVERY_ATTEMPT.value:
                    att_v = int(meta.get("attempt") or 0)
                    ledger_attempts = max(ledger_attempts, att_v if att_v > 0 else ledger_attempts + 1)
                    if meta.get("retry_deadline") is not None and latest_deadline is None:
                        try:
                            latest_deadline = float(meta["retry_deadline"])
                        except Exception:
                            pass
                if call_id and ev.correlation_id == call_id:
                    current_attempt_events.append(ev)

        if not func_name:
            for ev in reversed(node_events):
                if ev.tool_id:
                    func_name = str(ev.tool_id)
                    break

        # 2. Inspect structured observation & ActionRecord
        struct_obs: Optional[StructuredObservation] = None
        act_rec: Optional[ActionRecord] = None
        if turn_ledger is not None and call_id:
            if hasattr(turn_ledger, "get_structured_observation"):
                struct_obs = turn_ledger.get_structured_observation(call_id)
            if hasattr(turn_ledger, "get_record"):
                act_rec = turn_ledger.get_record(call_id)

        last_obs_ref = dict(node.observation_refs[-1]) if node.observation_refs and isinstance(node.observation_refs[-1], dict) else {}
        obs_status = (
            getattr(struct_obs, "execution_status", None)
            or last_obs_ref.get("execution_status")
            or next(
                (str((e.metadata or {}).get("execution_status")) for e in reversed(current_attempt_events) if (e.metadata or {}).get("execution_status")),
                None,
            )
        )
        ev_level = (
            getattr(struct_obs, "evidence_level", None)
            or last_obs_ref.get("evidence_level")
            or next(
                (str((e.metadata or {}).get("evidence_level")) for e in reversed(current_attempt_events) if (e.metadata or {}).get("evidence_level")),
                EvidenceLevel.UNKNOWN.value,
            )
        )

        # Determine whether execution actually began in the current attempt
        has_exec_start = any(e.event_type == LedgerEventType.TOOL_EXECUTION_START.value for e in current_attempt_events)
        if struct_obs is not None and isinstance(struct_obs.operation_attempted, dict):
            if "dispatched" in struct_obs.operation_attempted:
                has_exec_start = bool(has_exec_start or struct_obs.operation_attempted.get("dispatched"))
        if act_rec is not None and act_rec.state in (
            ActionState.DISPATCHED,
            ActionState.EXECUTED,
            ActionState.SUCCESS,
            ActionState.OBSERVED,
            ActionState.VERIFIED,
            ActionState.UNVERIFIED,
            ActionState.PARTIALLY_COMPLETED,
        ):
            has_exec_start = True
        if node.execution_status in (TaskNodeStatus.RUNNING.value, TaskNodeStatus.COMPLETED_UNVERIFIED.value):
            has_exec_start = True
        if obs_status in (
            ObservationExecutionStatus.EXECUTED_ERROR.value,
            ObservationExecutionStatus.TIMED_OUT.value,
            ObservationExecutionStatus.PARTIALLY_SUCCEEDED.value,
            ObservationExecutionStatus.SUCCEEDED.value,
            ObservationExecutionStatus.SUCCEEDED_VERIFIED.value,
            ObservationExecutionStatus.SUCCEEDED_UNVERIFIED.value,
        ):
            has_exec_start = True
        if obs_status == ObservationExecutionStatus.FAILED_BEFORE_EXECUTION.value and not any(
            e.event_type == LedgerEventType.TOOL_EXECUTION_START.value for e in current_attempt_events
        ):
            has_exec_start = False

        # Derive error category and guidance from structured observation + classifier (never prose!)
        node_fail = dict(node.failure_info or {})
        raw_err_type = (
            str((getattr(struct_obs, "error_classification", None) or {}).get("error_type") or "")
            or str(node_fail.get("error_type") or "")
            or next(
                (
                    str((e.metadata or {}).get("error_type") or "")
                    for e in reversed(current_attempt_events)
                    if (e.metadata or {}).get("error_type")
                ),
                "",
            )
        ).strip().upper()
        raw_err_msg = (
            str((getattr(struct_obs, "structured_result", None) or {}).get("error") or "")
            or str(getattr(act_rec, "error", None) or "")
            or str(node_fail.get("error") or "")
            or str(node.blocked_reason or "")
        )

        rec_guidance: Dict[str, Any] = dict(getattr(struct_obs, "recovery_guidance", None) or {})
        if not rec_guidance and struct_obs is not None:
            rec_guidance = ObservationFramingEngine.derive_recovery_guidance(struct_obs)

        is_contradictory = bool(
            node.contradictory
            or ev_level == EvidenceLevel.CONTRADICTED.value
            or obs_status == ObservationExecutionStatus.CONTRADICTORY_EVIDENCE.value
            or getattr(act_rec, "verification_failed", False)
            or any(
                e.event_type == LedgerEventType.VERIFICATION_RECORDED.value and e.status == VerificationStatus.VERIFICATION_FAILED.value
                for e in current_attempt_events
            )
        )

        if is_contradictory:
            failure_category = "CONTRADICTORY_EVIDENCE"
        elif node.execution_status == TaskNodeStatus.CANCELLED.value or obs_status == ObservationExecutionStatus.CANCELLED.value:
            failure_category = "CANCELLED"
        elif obs_status == ObservationExecutionStatus.TIMED_OUT.value or raw_err_type in ("TIMEOUT_ERROR", "COMMAND_TIMEOUT"):
            failure_category = "TIMEOUT_ERROR"
        elif raw_err_type:
            failure_category = raw_err_type
        elif obs_status == ObservationExecutionStatus.FAILED_BEFORE_EXECUTION.value:
            failure_category = "FAILED_BEFORE_EXECUTION"
        elif node.execution_status == TaskNodeStatus.COMPLETED_UNVERIFIED.value:
            failure_category = "UNVERIFIED_OUTCOME"
        elif raw_err_msg:
            cls_res = classify_tool_error(func_name, None, "", raw_err_msg)
            failure_category = cls_res.error_type or "TOOL_EXECUTION_FAILED"
        else:
            failure_category = "TOOL_EXECUTION_FAILED"

        # Existing node recovery state (preserved across restarts; never reset)
        prev_rs = dict(node.recovery_state) if isinstance(node.recovery_state, dict) else {}
        attempt_count = max(
            int(prev_rs.get("attempt_count") or 0),
            ledger_attempts,
            int(getattr(act_rec, "recovery_attempts", 0) or 0),
        )
        retry_deadline = prev_rs.get("retry_deadline")
        if retry_deadline is None:
            retry_deadline = latest_deadline
        if retry_deadline is not None:
            retry_deadline = float(retry_deadline)
        elif eff_policy.deadline_seconds is not None:
            retry_deadline = eff_now + float(eff_policy.deadline_seconds)

        # Build concise evidence references for audit/inspection
        evidence_refs: List[Dict[str, Any]] = []
        if struct_obs is not None:
            evidence_refs.append(
                {
                    "type": "structured_observation",
                    "observation_id": struct_obs.observation_id,
                    "evidence_level": struct_obs.evidence_level,
                    "execution_status": struct_obs.execution_status,
                }
            )
        for ev in node_events[-6:]:
            evidence_refs.append(
                {
                    "type": "ledger_event",
                    "event_id": ev.event_id,
                    "sequence_number": ev.sequence_number,
                    "event_type": ev.event_type,
                    "status": ev.status,
                }
            )

        affected_dependents = [d["node_id"] for d in graph.get_blocked_dependents(node_id)]

        # 3 & 4. Verification before recovery: check OutcomeVerifier first (never model prose!)
        criteria = _collect_node_criteria(node, proposal, acceptance_criteria, func_name, func_args)
        ver_summary: Optional[Dict[str, Any]] = None
        verified_outcome = False
        criteria_failed = False

        if node.execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value and node.verification_status == VerificationStatus.VERIFIED.value:
            verified_outcome = True
        elif criteria and node.execution_status != TaskNodeStatus.CANCELLED.value:
            v_fn = verifier_fn or OutcomeVerifier.verify
            v_results: List[Tuple[Dict[str, Any], Any]] = []
            for crit in criteria:
                vr = v_fn(
                    check=crit["check"],
                    target=crit["target"],
                    expected=crit.get("expected"),
                )
                v_results.append((crit, vr))
            all_passed = all(bool(getattr(vr, "success", False)) for _, vr in v_results)
            first_fail = next(((c, r) for c, r in v_results if not bool(getattr(r, "success", False))), None)
            ver_summary = {
                "checked_count": len(v_results),
                "passed": all_passed,
                "checks": [
                    {
                        "check": c["check"],
                        "target": c["target"],
                        "passed": bool(getattr(r, "success", False)),
                        "error": getattr(r, "error", None),
                    }
                    for c, r in v_results
                ],
            }
            if first_fail is not None:
                criteria_failed = True
                ver_summary["failed_target"] = first_fail[0].get("target")
                ver_summary["error"] = getattr(first_fail[1], "error", None)
            elif all_passed:
                # Contradictory nodes require explicit reconciliation flag or valid independent confirmation
                if not is_contradictory or reconcile_state:
                    verified_outcome = True
        elif (
            not is_contradictory
            and node.execution_status != TaskNodeStatus.CANCELLED.value
            and (
                (act_rec is not None and act_rec.state == ActionState.VERIFIED and not act_rec.verification_failed)
                or any(
                    e.event_type == LedgerEventType.VERIFICATION_RECORDED.value
                    and e.status == VerificationStatus.VERIFIED.value
                    for e in current_attempt_events
                )
            )
        ):
            verified_outcome = True
            ver_summary = {"passed": True, "source": "ledger_verification_recorded"}

        # Check conflict profile & read-only / destructive semantics
        reg_tool_obj = (
            tool_registry.get_tool(func_name)
            if (tool_registry is not None and func_name and callable(getattr(tool_registry, "get_tool", None)))
            else None
        )
        schema_match: Optional[Dict[str, Any]] = None
        if isinstance(active_schemas, list) and func_name:
            for s in active_schemas:
                if not isinstance(s, dict):
                    continue
                fn = s.get("function") if isinstance(s.get("function"), dict) else {}
                if str(fn.get("name") or s.get("name") or "").strip() == func_name:
                    schema_match = s
                    break

        conflict_profile: NodeConflictProfile = extract_node_conflict_profile(
            func_name,
            func_args,
            node=node,
            external_source=ext_src if isinstance(ext_src, dict) else None,
            tool_schema=schema_match,
            tool_obj=reg_tool_obj,
        )
        is_destructive = _is_destructive_operation(func_name, func_args)
        is_read_only = bool(conflict_profile.read_only and not is_destructive)
        is_idempotent = _is_trusted_idempotent(reg_tool_obj, schema_match, func_args)

        if not has_exec_start:
            may_have_side_effect = False
        elif is_read_only and not is_contradictory and failure_category not in ("TIMEOUT_ERROR", "CANCELLED"):
            may_have_side_effect = False
        elif "may_have_side_effect" in node_fail and isinstance(node_fail.get("may_have_side_effect"), bool):
            may_have_side_effect = bool(node_fail["may_have_side_effect"]) and not (is_read_only and not is_contradictory)
        elif "may_have_side_effect" in rec_guidance:
            may_have_side_effect = bool(rec_guidance["may_have_side_effect"]) and not (is_read_only and not is_contradictory)
        else:
            may_have_side_effect = not is_read_only

        # If OutcomeVerifier already confirmed expected outcome exists, never repeat the operation
        if verified_outcome:
            return RecoveryAssessment(
                execution_id=exec_id,
                node_id=node_id,
                failure_category=failure_category,
                previous_status=node.execution_status,
                structured_observation=struct_obs.to_compact_dict() if struct_obs is not None else None,
                evidence_level=EvidenceLevel.CONFIRMED.value,
                execution_started=has_exec_start,
                may_have_side_effect=False,
                verified_outcome=True,
                recoverable=False,
                recovery_guidance=rec_guidance,
                recovery_decision=RecoveryDecision.ALREADY_COMPLETED.value,
                reason="Expected node outcome independently verified via OutcomeVerifier; mark COMPLETED_VERIFIED without repeating action.",
                attempt_count=attempt_count,
                max_attempts=eff_policy.max_attempts,
                retry_deadline=retry_deadline,
                backoff_seconds=0.0,
                reconciliation_required=False,
                revalidation_passed=True,
                verification_result=ver_summary,
                evidence_refs=evidence_refs,
                proposed_next_action="promote_completed_verified",
                affected_dependents=affected_dependents,
                blocked_reason=None,
                call_id=call_id,
                proposal_id=proposal_id,
                tool_name=func_name or None,
                normalized_arguments=func_args,
            )

        # 5. Revalidate upstream dependencies
        node_map = {n.node_id: n for n in graph.nodes}
        unsatisfied_deps: List[str] = []
        for dep_id in node.dependency_ids:
            dep_node = node_map.get(dep_id)
            if dep_node is None or not TaskGraph._is_prerequisite_satisfied(dep_node, node.requires_verified_dependencies):
                unsatisfied_deps.append(dep_id)

        if unsatisfied_deps:
            dep_reason = f"Blocked by unsatisfied/unverified prerequisite(s): {', '.join(unsatisfied_deps)}."
            return RecoveryAssessment(
                execution_id=exec_id,
                node_id=node_id,
                failure_category=failure_category,
                previous_status=node.execution_status,
                structured_observation=struct_obs.to_compact_dict() if struct_obs is not None else None,
                evidence_level=str(ev_level),
                execution_started=has_exec_start,
                may_have_side_effect=may_have_side_effect,
                verified_outcome=False,
                recoverable=False,
                recovery_guidance=rec_guidance,
                recovery_decision=RecoveryDecision.BLOCKED.value,
                reason=dep_reason,
                attempt_count=attempt_count,
                max_attempts=eff_policy.max_attempts,
                retry_deadline=retry_deadline,
                backoff_seconds=0.0,
                reconciliation_required=is_contradictory or may_have_side_effect,
                revalidation_passed=False,
                verification_result=ver_summary,
                evidence_refs=evidence_refs,
                proposed_next_action="wait_or_recover_prerequisites",
                affected_dependents=affected_dependents,
                blocked_reason=dep_reason,
                call_id=call_id,
                proposal_id=proposal_id,
                tool_name=func_name or None,
                normalized_arguments=func_args,
            )

        # 6. Contradictory evidence requires reconciliation first
        if is_contradictory:
            reconcile_msg = (
                "Contradictory observation or verification mismatch detected; state reconciliation confirmed failure/absence."
                if (reconcile_state and criteria_failed)
                else "Contradictory evidence detected; requires state reconciliation before retrying or unlocking dependents."
            )
            decision = (
                RecoveryDecision.REPLAN_NODE.value
                if (reconcile_state and criteria_failed and not may_have_side_effect)
                else RecoveryDecision.REQUIRE_CONFIRMATION.value
            )
            return RecoveryAssessment(
                execution_id=exec_id,
                node_id=node_id,
                failure_category="CONTRADICTORY_EVIDENCE",
                previous_status=node.execution_status,
                structured_observation=struct_obs.to_compact_dict() if struct_obs is not None else None,
                evidence_level=EvidenceLevel.CONTRADICTED.value,
                execution_started=has_exec_start,
                may_have_side_effect=True if not is_read_only else False,
                verified_outcome=False,
                recoverable=True,
                recovery_guidance=rec_guidance,
                recovery_decision=decision,
                reason=reconcile_msg,
                attempt_count=attempt_count,
                max_attempts=eff_policy.max_attempts,
                retry_deadline=retry_deadline,
                backoff_seconds=0.0,
                reconciliation_required=True,
                revalidation_passed=False,
                verification_result=ver_summary,
                evidence_refs=evidence_refs,
                proposed_next_action="reconcile_contradiction" if decision == RecoveryDecision.REQUIRE_CONFIRMATION.value else "replan_failed_node",
                affected_dependents=affected_dependents,
                blocked_reason=reconcile_msg,
                call_id=call_id,
                proposal_id=proposal_id,
                tool_name=func_name or None,
                normalized_arguments=func_args,
            )

        # Non-recoverable / policy-denied / cancelled failures block cleanly
        if node.execution_status == TaskNodeStatus.CANCELLED.value or failure_category in _NON_RECOVERABLE_FAILURE_CATEGORIES:
            blk_msg = (
                f"Execution cancelled; node requires explicit confirmation or replanning ({raw_err_msg or 'cancelled'})."
                if failure_category in ("CANCELLED", "TASK_CANCELLED")
                else f"Non-recoverable failure ({failure_category}): {_sanitize_recovery_text(raw_err_msg or 'blocked')}"
            )
            return RecoveryAssessment(
                execution_id=exec_id,
                node_id=node_id,
                failure_category=failure_category,
                previous_status=node.execution_status,
                structured_observation=struct_obs.to_compact_dict() if struct_obs is not None else None,
                evidence_level=str(ev_level),
                execution_started=has_exec_start,
                may_have_side_effect=may_have_side_effect,
                verified_outcome=False,
                recoverable=False,
                recovery_guidance=rec_guidance,
                recovery_decision=RecoveryDecision.BLOCKED.value,
                reason=blk_msg,
                attempt_count=attempt_count,
                max_attempts=eff_policy.max_attempts,
                retry_deadline=retry_deadline,
                backoff_seconds=0.0,
                reconciliation_required=bool(has_exec_start and may_have_side_effect),
                revalidation_passed=False,
                verification_result=ver_summary,
                evidence_refs=evidence_refs,
                proposed_next_action="manual_intervention_or_replan",
                affected_dependents=affected_dependents,
                blocked_reason=blk_msg,
                call_id=call_id,
                proposal_id=proposal_id,
                tool_name=func_name or None,
                normalized_arguments=func_args,
            )

        # Live revalidation of tool registration, external provenance, authorization, & normalized arguments
        revalidation_passed = False
        normalized_args_out: Optional[Dict[str, Any]] = dict(func_args)
        reval_fail_reason: Optional[str] = None
        reval_err_type: Optional[str] = None

        if proposal is None or not func_name:
            reval_fail_reason = f"Node '{node_id}' has no valid bound tool proposal to retry."
            reval_err_type = "MISSING_PROPOSAL"
        else:
            exp_at = _get_prop_field(proposal, "expires_at")
            if exp_at is not None and float(exp_at) <= eff_now:
                reval_fail_reason = f"Proposal '{proposal_id}' has expired."
                reval_err_type = "PROPOSAL_EXPIRED"

        if reval_fail_reason is None and isinstance(ext_src, dict) and ext_src.get("is_external"):
            if tool_registry is not None and callable(getattr(tool_registry, "get_external_tool_entry", None)):
                ext_entry = tool_registry.get_external_tool_entry(func_name)
                if ext_entry is None or getattr(ext_entry, "authorized", True) is False:
                    reval_fail_reason = f"External tool '{func_name}' is no longer authorized or registered."
                    reval_err_type = "AUTHORIZATION_DENIED"
            elif task_local_scope is not None:
                allowed_sc = set(getattr(task_local_scope, "tool_names", None) or [])
                if allowed_sc and func_name not in allowed_sc:
                    reval_fail_reason = f"External tool '{func_name}' is outside active TaskLocalToolScope."
                    reval_err_type = "AUTHORIZATION_DENIED"

        if reval_fail_reason is None and (
            tool_registry is not None
            or active_schemas is not None
            or caller_ring is not None
            or allowed_tools is not None
            or task_local_scope is not None
            or known_tools is not None
        ):
            dec = authorize_and_validate_tool_call(
                func_name,
                _get_prop_field(proposal, "arguments") or func_args,
                caller_ring=caller_ring,
                allowed_tools=allowed_tools,
                active_schemas=active_schemas,
                tool_registry=tool_registry,
                known_tools=known_tools,
                task_local_scope=task_local_scope,
                require_registered=bool(tool_registry is not None or task_local_scope is not None),
                normalize_args=True,
                validate_schema=bool(active_schemas is not None or tool_registry is not None),
                argument_source="internal",
            )
            if not dec.allowed:
                reval_fail_reason = dec.detail or "Live authorization or schema revalidation denied."
                reval_err_type = dec.error_type or "AUTHORIZATION_DENIED"
            else:
                revalidation_passed = True
                normalized_args_out = dict(dec.normalized_arguments)
        elif reval_fail_reason is None:
            revalidation_passed = True

        # Fresh execution-policy evaluation on recovery (non-consuming check; never uses stale grants).
        pol_outcome_val: Optional[str] = None
        pol_dec_dict: Optional[Dict[str, Any]] = None
        if revalidation_passed and func_name:
            from zelyr.engine.execution_policy import (
                PolicyDecisionOutcome,
                build_execution_policy_engine,
            )
            from zelyr.tool_registry import resolve_tool_ring

            pol_engine = build_execution_policy_engine(
                policy_engine=policy_engine,
                policy_config=policy_config,
                zelyr_config=zelyr_config,
            )
            reg_tool_rec = (
                tool_registry.get_tool(func_name)
                if (tool_registry is not None and callable(getattr(tool_registry, "get_tool", None)))
                else None
            )
            ext_entry_rec = (
                tool_registry.get_external_tool_entry(func_name)
                if (tool_registry is not None and callable(getattr(tool_registry, "get_external_tool_entry", None)))
                else None
            )
            matched_rec_schema = next(
                (
                    s
                    for s in (active_schemas or [])
                    if isinstance(s, dict) and (s.get("function") or {}).get("name") == func_name
                ),
                None,
            )
            req_ring_rec = resolve_tool_ring(
                func_name,
                tool=reg_tool_rec,
                task_local_scope=task_local_scope,
            )
            is_ext_rec = bool(
                (ext_src and ext_src.get("is_external"))
                or ext_entry_rec is not None
                or getattr(reg_tool_rec, "is_external", False)
                or str(func_name).startswith("ext__")
            )
            ext_meta_rec = (
                dict(getattr(ext_entry_rec, "metadata", None) or {})
                if ext_entry_rec is not None
                else None
            )
            args_for_pol = dict(normalized_args_out if normalized_args_out is not None else func_args)
            pol_dec_rec = pol_engine.evaluate(
                func_name,
                _get_prop_field(proposal, "arguments") or args_for_pol,
                normalized_arguments=args_for_pol,
                schema=matched_rec_schema,
                tool_obj=reg_tool_rec,
                required_ring=req_ring_rec,
                is_registered=True,
                is_external=is_ext_rec,
                external_metadata=ext_meta_rec,
                workspace_root=workspace_root,
                node_target_resources=getattr(node, "target_resources", ()),
                execution_id=exec_id,
                node_id=node_id,
                confirmation_token=confirmation_token,
                confirmation_id=confirmation_id,
                consume_confirmation=False,
                confirm_callback=confirm_callback,
                source_surface="recovery_engine",
                central_ledger=central_ledger,
                record_to_ledger=False,
                now=eff_now,
            )
            pol_outcome_val = pol_dec_rec.outcome.value
            pol_dec_dict = pol_dec_rec.to_safe_dict()
            if pol_dec_rec.outcome == PolicyDecisionOutcome.DENY:
                revalidation_passed = False
                reval_fail_reason = pol_dec_rec.reason
                reval_err_type = "POLICY_DENIED"
            elif pol_dec_rec.outcome == PolicyDecisionOutcome.REQUIRE_CONFIRMATION and not confirm_recovery:
                return RecoveryAssessment(
                    execution_id=exec_id,
                    node_id=node_id,
                    failure_category="POLICY_CONFIRMATION_REQUIRED",
                    previous_status=node.execution_status,
                    structured_observation=struct_obs.to_compact_dict() if struct_obs is not None else None,
                    evidence_level=str(ev_level),
                    execution_started=has_exec_start,
                    may_have_side_effect=may_have_side_effect,
                    verified_outcome=False,
                    recoverable=True,
                    recovery_guidance=rec_guidance,
                    recovery_decision=RecoveryDecision.REQUIRE_CONFIRMATION.value,
                    reason=pol_dec_rec.reason,
                    attempt_count=attempt_count,
                    max_attempts=eff_policy.max_attempts,
                    retry_deadline=retry_deadline,
                    backoff_seconds=0.0,
                    reconciliation_required=bool(has_exec_start and may_have_side_effect),
                    revalidation_passed=True,
                    verification_result=ver_summary,
                    evidence_refs=evidence_refs,
                    proposed_next_action="obtain_confirmation_for_retry",
                    affected_dependents=affected_dependents,
                    blocked_reason=pol_dec_rec.reason,
                    call_id=call_id,
                    proposal_id=proposal_id,
                    tool_name=func_name or None,
                    normalized_arguments=normalized_args_out,
                    policy_outcome=pol_outcome_val,
                    policy_decision=pol_dec_dict,
                )

        if not revalidation_passed:
            is_auth_block = reval_err_type in ("AUTHORIZATION_DENIED", "AUTHORIZATION_ERROR", "PROHIBITED_BY_USER", "POLICY_DENIED")
            dec_fail = RecoveryDecision.BLOCKED.value if is_auth_block else RecoveryDecision.REPLAN_NODE.value
            return RecoveryAssessment(
                execution_id=exec_id,
                node_id=node_id,
                failure_category=reval_err_type or failure_category,
                previous_status=node.execution_status,
                structured_observation=struct_obs.to_compact_dict() if struct_obs is not None else None,
                evidence_level=str(ev_level),
                execution_started=has_exec_start,
                may_have_side_effect=may_have_side_effect,
                verified_outcome=False,
                recoverable=not is_auth_block,
                recovery_guidance=rec_guidance,
                recovery_decision=dec_fail,
                reason=reval_fail_reason or "Live revalidation failed.",
                attempt_count=attempt_count,
                max_attempts=eff_policy.max_attempts,
                retry_deadline=retry_deadline,
                backoff_seconds=0.0,
                reconciliation_required=bool(has_exec_start and may_have_side_effect),
                revalidation_passed=False,
                verification_result=ver_summary,
                evidence_refs=evidence_refs,
                proposed_next_action="resolve_authorization" if is_auth_block else "replace_or_replan_node",
                affected_dependents=affected_dependents,
                blocked_reason=reval_fail_reason,
                call_id=call_id,
                proposal_id=proposal_id,
                tool_name=func_name or None,
                normalized_arguments=normalized_args_out,
                policy_outcome=pol_outcome_val,
                policy_decision=pol_dec_dict,
            )

        if failure_category in _REPLAN_FAILURE_CATEGORIES:
            replan_msg = f"Original operation cannot repeat ({failure_category}): {_sanitize_recovery_text(raw_err_msg or 'requires replanning')}"
            return RecoveryAssessment(
                execution_id=exec_id,
                node_id=node_id,
                failure_category=failure_category,
                previous_status=node.execution_status,
                structured_observation=struct_obs.to_compact_dict() if struct_obs is not None else None,
                evidence_level=str(ev_level),
                execution_started=has_exec_start,
                may_have_side_effect=may_have_side_effect,
                verified_outcome=False,
                recoverable=True,
                recovery_guidance=rec_guidance,
                recovery_decision=RecoveryDecision.REPLAN_NODE.value,
                reason=replan_msg,
                attempt_count=attempt_count,
                max_attempts=eff_policy.max_attempts,
                retry_deadline=retry_deadline,
                backoff_seconds=0.0,
                reconciliation_required=bool(has_exec_start and may_have_side_effect),
                revalidation_passed=True,
                verification_result=ver_summary,
                evidence_refs=evidence_refs,
                proposed_next_action="replace_or_replan_node",
                affected_dependents=affected_dependents,
                blocked_reason=replan_msg,
                call_id=call_id,
                proposal_id=proposal_id,
                tool_name=func_name or None,
                normalized_arguments=normalized_args_out,
                policy_outcome=pol_outcome_val,
                policy_decision=pol_dec_dict,
            )

        # Enforce deadline and retry attempt limits before proposing any retry
        if retry_deadline is not None and eff_now > retry_deadline:
            dl_msg = "Total recovery retry deadline exceeded; blocking further retries."
            return RecoveryAssessment(
                execution_id=exec_id,
                node_id=node_id,
                failure_category=failure_category,
                previous_status=node.execution_status,
                structured_observation=struct_obs.to_compact_dict() if struct_obs is not None else None,
                evidence_level=str(ev_level),
                execution_started=has_exec_start,
                may_have_side_effect=may_have_side_effect,
                verified_outcome=False,
                recoverable=False,
                recovery_guidance=rec_guidance,
                recovery_decision=RecoveryDecision.BLOCKED.value,
                reason=dl_msg,
                attempt_count=attempt_count,
                max_attempts=eff_policy.max_attempts,
                retry_deadline=retry_deadline,
                backoff_seconds=0.0,
                reconciliation_required=bool(has_exec_start and may_have_side_effect),
                revalidation_passed=True,
                verification_result=ver_summary,
                evidence_refs=evidence_refs,
                proposed_next_action="deadline_exceeded_replan_or_abort",
                affected_dependents=affected_dependents,
                blocked_reason=dl_msg,
                call_id=call_id,
                proposal_id=proposal_id,
                tool_name=func_name or None,
                normalized_arguments=normalized_args_out,
                policy_outcome=pol_outcome_val,
                policy_decision=pol_dec_dict,
            )

        if attempt_count >= eff_policy.max_attempts:
            ex_msg = f"Maximum recovery attempts exhausted ({attempt_count}/{eff_policy.max_attempts})."
            return RecoveryAssessment(
                execution_id=exec_id,
                node_id=node_id,
                failure_category=failure_category,
                previous_status=node.execution_status,
                structured_observation=struct_obs.to_compact_dict() if struct_obs is not None else None,
                evidence_level=str(ev_level),
                execution_started=has_exec_start,
                may_have_side_effect=may_have_side_effect,
                verified_outcome=False,
                recoverable=False,
                recovery_guidance=rec_guidance,
                recovery_decision=RecoveryDecision.BLOCKED.value,
                reason=ex_msg,
                attempt_count=attempt_count,
                max_attempts=eff_policy.max_attempts,
                retry_deadline=retry_deadline,
                backoff_seconds=0.0,
                reconciliation_required=bool(has_exec_start and may_have_side_effect),
                revalidation_passed=True,
                verification_result=ver_summary,
                evidence_refs=evidence_refs,
                proposed_next_action="max_attempts_exhausted_replan_or_abort",
                affected_dependents=affected_dependents,
                blocked_reason=ex_msg,
                call_id=call_id,
                proposal_id=proposal_id,
                tool_name=func_name or None,
                normalized_arguments=normalized_args_out,
                policy_outcome=pol_outcome_val,
                policy_decision=pol_dec_dict,
            )

        backoff_s = eff_policy.compute_backoff_seconds(attempt_count + 1)

        # Ambiguous / destructive / partial side-effect guard (never retry blindly!)
        can_reconcile_cleanly = bool(
            reconcile_state
            and not is_destructive
            and criteria_failed
            and (is_read_only or is_idempotent or conflict_profile.exclusive_resource is None)
        )
        if has_exec_start and may_have_side_effect:
            if not confirm_recovery and not can_reconcile_cleanly:
                amb_msg = (
                    "Execution started and may have produced side effects (or timed out); "
                    "automatic retry blocked until state reconciliation or explicit confirmation."
                )
                return RecoveryAssessment(
                    execution_id=exec_id,
                    node_id=node_id,
                    failure_category=failure_category,
                    previous_status=node.execution_status,
                    structured_observation=struct_obs.to_compact_dict() if struct_obs is not None else None,
                    evidence_level=str(ev_level),
                    execution_started=True,
                    may_have_side_effect=True,
                    verified_outcome=False,
                    recoverable=True,
                    recovery_guidance=rec_guidance,
                    recovery_decision=RecoveryDecision.REQUIRE_CONFIRMATION.value,
                    reason=amb_msg,
                    attempt_count=attempt_count,
                    max_attempts=eff_policy.max_attempts,
                    retry_deadline=retry_deadline,
                    backoff_seconds=0.0,
                    reconciliation_required=True,
                    revalidation_passed=True,
                    verification_result=ver_summary,
                    evidence_refs=evidence_refs,
                    proposed_next_action="reconcile_or_confirm_before_retry",
                    affected_dependents=affected_dependents,
                    blocked_reason=amb_msg,
                    call_id=call_id,
                    proposal_id=proposal_id,
                    tool_name=func_name or None,
                    normalized_arguments=normalized_args_out,
                )
            # Reconciled non-destructive state or explicitly confirmed after inspection
            if not eff_policy.allow_retry_after_revalidation:
                return RecoveryAssessment(
                    execution_id=exec_id,
                    node_id=node_id,
                    failure_category=failure_category,
                    previous_status=node.execution_status,
                    structured_observation=struct_obs.to_compact_dict() if struct_obs is not None else None,
                    evidence_level=str(ev_level),
                    execution_started=True,
                    may_have_side_effect=not can_reconcile_cleanly,
                    verified_outcome=False,
                    recoverable=True,
                    recovery_guidance=rec_guidance,
                    recovery_decision=RecoveryDecision.REQUIRE_CONFIRMATION.value,
                    reason="Retry after revalidation disabled by recovery policy.",
                    attempt_count=attempt_count,
                    max_attempts=eff_policy.max_attempts,
                    retry_deadline=retry_deadline,
                    backoff_seconds=0.0,
                    reconciliation_required=not can_reconcile_cleanly,
                    revalidation_passed=True,
                    verification_result=ver_summary,
                    evidence_refs=evidence_refs,
                    proposed_next_action="manual_confirmation_required",
                    affected_dependents=affected_dependents,
                    blocked_reason="Retry after revalidation disabled by policy.",
                    call_id=call_id,
                    proposal_id=proposal_id,
                    tool_name=func_name or None,
                    normalized_arguments=normalized_args_out,
                )
            rec_reason = (
                "State reconciliation confirmed clean target state and live auth/tool/dependencies passed; retry allowed."
                if can_reconcile_cleanly
                else "Explicit confirmation provided with live auth/tool/dependency revalidation passed; retry allowed."
            )
            return RecoveryAssessment(
                execution_id=exec_id,
                node_id=node_id,
                failure_category=failure_category,
                previous_status=node.execution_status,
                structured_observation=struct_obs.to_compact_dict() if struct_obs is not None else None,
                evidence_level=str(ev_level),
                execution_started=True,
                may_have_side_effect=not can_reconcile_cleanly,
                verified_outcome=False,
                recoverable=True,
                recovery_guidance=rec_guidance,
                recovery_decision=RecoveryDecision.RETRY_AFTER_REVALIDATION.value,
                reason=rec_reason,
                attempt_count=attempt_count,
                max_attempts=eff_policy.max_attempts,
                retry_deadline=retry_deadline,
                backoff_seconds=backoff_s,
                reconciliation_required=False,
                revalidation_passed=True,
                verification_result=ver_summary,
                evidence_refs=evidence_refs,
                proposed_next_action="retry_revalidated_node",
                affected_dependents=affected_dependents,
                blocked_reason=None,
                call_id=call_id,
                proposal_id=proposal_id,
                tool_name=func_name or None,
                normalized_arguments=normalized_args_out,
                policy_outcome=pol_outcome_val,
                policy_decision=pol_dec_dict,
            )

        # Execution did not start & no side effect could occur
        if not has_exec_start and not may_have_side_effect:
            if failure_category in _PRE_DISPATCH_SAFE_RETRY_CATEGORIES or failure_category in (
                "SCHEMA_VALIDATION_ERROR",
                "ARGUMENT_NORMALIZATION_ERROR",
                "TOOL_EXECUTION_FAILED",
                "EXECUTION_FAILED",
            ):
                return RecoveryAssessment(
                    execution_id=exec_id,
                    node_id=node_id,
                    failure_category=failure_category,
                    previous_status=node.execution_status,
                    structured_observation=struct_obs.to_compact_dict() if struct_obs is not None else None,
                    evidence_level=str(ev_level),
                    execution_started=False,
                    may_have_side_effect=False,
                    verified_outcome=False,
                    recoverable=True,
                    recovery_guidance=rec_guidance,
                    recovery_decision=RecoveryDecision.RETRY_SAFE.value,
                    reason="Execution failed before dispatch with zero possible side effects and live validation passed; safe to retry.",
                    attempt_count=attempt_count,
                    max_attempts=eff_policy.max_attempts,
                    retry_deadline=retry_deadline,
                    backoff_seconds=backoff_s,
                    reconciliation_required=False,
                    revalidation_passed=True,
                    verification_result=ver_summary,
                    evidence_refs=evidence_refs,
                    proposed_next_action="retry_safe_dispatch",
                    affected_dependents=affected_dependents,
                    blocked_reason=None,
                    call_id=call_id,
                    proposal_id=proposal_id,
                    tool_name=func_name or None,
                    normalized_arguments=normalized_args_out,
                )

        # Non-mutating read-only or trusted idempotent executed error
        if (is_read_only or is_idempotent) and not is_destructive and failure_category != "TIMEOUT_ERROR":
            return RecoveryAssessment(
                execution_id=exec_id,
                node_id=node_id,
                failure_category=failure_category,
                previous_status=node.execution_status,
                structured_observation=struct_obs.to_compact_dict() if struct_obs is not None else None,
                evidence_level=str(ev_level),
                execution_started=has_exec_start,
                may_have_side_effect=False,
                verified_outcome=False,
                recoverable=True,
                recovery_guidance=rec_guidance,
                recovery_decision=RecoveryDecision.RETRY_AFTER_REVALIDATION.value,
                reason="Non-destructive/read-only operation failed cleanly and passed live revalidation; retry allowed.",
                attempt_count=attempt_count,
                max_attempts=eff_policy.max_attempts,
                retry_deadline=retry_deadline,
                backoff_seconds=backoff_s,
                reconciliation_required=False,
                revalidation_passed=True,
                verification_result=ver_summary,
                evidence_refs=evidence_refs,
                proposed_next_action="retry_revalidated_node",
                affected_dependents=affected_dependents,
                blocked_reason=None,
                call_id=call_id,
                proposal_id=proposal_id,
                tool_name=func_name or None,
                normalized_arguments=normalized_args_out,
                policy_outcome=pol_outcome_val,
                policy_decision=pol_dec_dict,
            )

        # Insufficient or ambiguous evidence defaults fail-closed to UNRESOLVED / REQUIRE_CONFIRMATION
        unres_msg = "Insufficient verified evidence to establish safe retry; manual reconciliation required."
        return RecoveryAssessment(
            execution_id=exec_id,
            node_id=node_id,
            failure_category=failure_category,
            previous_status=node.execution_status,
            structured_observation=struct_obs.to_compact_dict() if struct_obs is not None else None,
            evidence_level=str(ev_level),
            execution_started=has_exec_start,
            may_have_side_effect=may_have_side_effect,
            verified_outcome=False,
            recoverable=False,
            recovery_guidance=rec_guidance,
            recovery_decision=RecoveryDecision.UNRESOLVED.value,
            reason=unres_msg,
            attempt_count=attempt_count,
            max_attempts=eff_policy.max_attempts,
            retry_deadline=retry_deadline,
            backoff_seconds=0.0,
            reconciliation_required=True,
            revalidation_passed=revalidation_passed,
            verification_result=ver_summary,
            evidence_refs=evidence_refs,
            proposed_next_action="inspect_and_reconcile",
            affected_dependents=affected_dependents,
            blocked_reason=unres_msg,
            call_id=call_id,
            proposal_id=proposal_id,
            tool_name=func_name or None,
            normalized_arguments=normalized_args_out,
        )

    @classmethod
    def apply_recovery_decision(
        cls,
        graph: TaskGraph,
        assessment: RecoveryAssessment,
        *,
        prop_by_ref: Optional[Dict[str, Any]] = None,
        central_ledger: Optional[CentralExecutionLedger] = None,
        turn_ledger: Optional[TurnExecutionLedger] = None,
        execute_retry: bool = True,
        owner_id: Optional[str] = None,
        now_ts: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Atomically persist recovery state, emit ledger event, and transition node/retry proposal."""
        node_id = assessment.node_id
        eff_owner = (owner_id or f"recovery:{node_id}:{uuid.uuid4().hex[:8]}").strip()
        eff_now = float(now_ts if now_ts is not None else time.time())
        exec_id = graph.execution_id

        # Exclusive per-node recovery lock prevents competing parallel workers initiating duplicate recovery
        graph.begin_node_recovery(node_id, eff_owner)
        try:
            node = graph.get_node(node_id)
            if node is None:
                raise TaskGraphValidationError(f"Unknown task node '{node_id}'.")

            prev_status = node.execution_status
            prev_rs = dict(node.recovery_state) if isinstance(node.recovery_state, dict) else {}
            history = list(prev_rs.get("recovery_history") or [])
            decision = assessment.recovery_decision
            next_attempt = assessment.attempt_count
            will_retry = bool(
                execute_retry
                and decision in (RecoveryDecision.RETRY_SAFE.value, RecoveryDecision.RETRY_AFTER_REVALIDATION.value)
            )
            new_prop_dict: Optional[Dict[str, Any]] = None

            if prev_status == TaskNodeStatus.COMPLETED_VERIFIED.value and decision != RecoveryDecision.ALREADY_COMPLETED.value:
                return {
                    "node_id": node_id,
                    "decision": RecoveryDecision.ALREADY_COMPLETED.value,
                    "action": "noop_already_verified",
                    "retried": False,
                    "promoted_verified": False,
                    "new_status": prev_status,
                }

            if decision == RecoveryDecision.ALREADY_COMPLETED.value:
                # Mark verified without repeating execution
                if prop_by_ref is not None:
                    for ref_k in list(node.proposal_ids) + ([assessment.call_id] if assessment.call_id else []):
                        p_item = prop_by_ref.get(ref_k)
                        if isinstance(p_item, dict):
                            p_item["status"] = "completed"
                            p_item["blocked_reason"] = None
                        elif p_item is not None and hasattr(p_item, "status"):
                            p_item.status = "completed"
                            p_item.blocked_reason = None
                if turn_ledger is not None and assessment.call_id:
                    for vc in (assessment.verification_result or {}).get("checks") or []:
                        if isinstance(vc, dict) and vc.get("passed") and vc.get("target"):
                            turn_ledger.record_target_verified(
                                str(vc["target"]),
                                detail="Verified prior to recovery",
                                check_type=vc.get("check"),
                            )
                rec_status = "recovered_verified"
                history.append(
                    {
                        "attempt": next_attempt,
                        "timestamp": eff_now,
                        "failure_category": assessment.failure_category,
                        "decision": decision,
                        "previous_status": prev_status,
                        "new_status": TaskNodeStatus.COMPLETED_VERIFIED.value,
                        "reason": _sanitize_recovery_text(assessment.reason, max_bytes=200),
                        "verified": True,
                    }
                )
                new_rs = validate_recovery_state_dict(
                    {
                        "recovery_version": RECOVERY_STATE_VERSION,
                        "attempt_count": next_attempt,
                        "max_attempts": assessment.max_attempts,
                        "recovery_decision": decision,
                        "recovery_status": rec_status,
                        "previous_failure_category": assessment.failure_category,
                        "retry_deadline": assessment.retry_deadline,
                        "may_have_side_effect": False,
                        "execution_started": assessment.execution_started,
                        "reconciliation_required": False,
                        "verified_outcome": True,
                        "next_recovery_action": "none",
                        "decision_reason": _sanitize_recovery_text(assessment.reason, max_bytes=256),
                        "recovery_history": history[-24:],
                    }
                )
                graph.record_node_recovery_state(
                    node_id,
                    new_rs,
                    transition_status=TaskNodeStatus.COMPLETED_VERIFIED.value,
                    verification_status=VerificationStatus.VERIFIED.value,
                    blocked_reason=None,
                    clear_contradiction=True,
                    clear_failure_info=True,
                    known_proposals=prop_by_ref,
                )
                new_status = graph.get_node(node_id).execution_status
                cls._record_recovery_ledger_event(
                    central_ledger,
                    exec_id,
                    assessment,
                    previous_status=prev_status,
                    new_status=new_status,
                    attempt_number=next_attempt,
                    status_label="verified_existing_outcome",
                )
                return {
                    "node_id": node_id,
                    "decision": decision,
                    "action": "promoted_verified",
                    "retried": False,
                    "promoted_verified": True,
                    "previous_status": prev_status,
                    "new_status": new_status,
                    "unlocked_nodes": [n.node_id for n in graph.get_ready_nodes()],
                }

            if will_retry or decision == RecoveryDecision.REPLAN_NODE.value:
                try:
                    from zelyr.engine.execution_policy import get_default_confirmation_store

                    get_default_confirmation_store().revoke_node_confirmations(exec_id, node_id)
                except Exception:
                    pass

            if will_retry:
                next_attempt = assessment.attempt_count + 1
                next_after = eff_now + float(assessment.backoff_seconds or 0.0)
                base_prop = _resolve_node_proposal(node, prop_by_ref or {})
                orig_pid = str(
                    _get_prop_field(base_prop, "proposal_id")
                    or (node.proposal_ids[0] if node.proposal_ids else f"prop_{node_id}")
                ).rsplit("__retry_", 1)[0]
                orig_cid = str(
                    _get_prop_field(base_prop, "original_call_id") or assessment.call_id or f"call_{node_id}"
                ).rsplit("__retry_", 1)[0]
                new_pid = f"{orig_pid}__retry_{next_attempt}"
                new_cid = f"{orig_cid}__retry_{next_attempt}"
                # Mark superseded attempt proposal cleanly so checkpointing does not retain stale active entries
                if prop_by_ref is not None:
                    for old_k in list(node.proposal_ids) + ([assessment.call_id] if assessment.call_id else []):
                        old_p = prop_by_ref.get(old_k)
                        if isinstance(old_p, dict):
                            old_p["status"] = "cancelled"
                            old_p["blocked_reason"] = f"Superseded by recovery retry '{new_pid}'."
                        elif old_p is not None and hasattr(old_p, "status"):
                            old_p.status = "cancelled"
                            old_p.blocked_reason = f"Superseded by recovery retry '{new_pid}'."
                tool_id = str(assessment.tool_name or _get_prop_field(base_prop, "tool_id") or "")
                norm_args = dict(
                    assessment.normalized_arguments
                    if assessment.normalized_arguments is not None
                    else (_get_prop_field(base_prop, "normalized_arguments") or {})
                )
                raw_order = int(_get_prop_field(base_prop, "proposal_order", node.order_index) or node.order_index)
                raw_deps = list(_get_prop_field(base_prop, "depends_on") or [])
                raw_ext = _get_prop_field(base_prop, "external_source") or node.external_source
                acc_spec = _get_prop_field(base_prop, "acceptance_criteria")

                new_prop_dict = {
                    "proposal_id": new_pid,
                    "execution_id": exec_id,
                    "original_call_id": new_cid,
                    "tool_id": tool_id,
                    "raw_name": str(_get_prop_field(base_prop, "raw_name") or tool_id),
                    "arguments": dict(norm_args),
                    "normalized_arguments": dict(norm_args),
                    "proposal_order": raw_order,
                    "depends_on": raw_deps,
                    "external_source": dict(raw_ext) if isinstance(raw_ext, dict) else None,
                    "status": "pending",
                    "authorization_status": "revalidated",
                    "retry_of_proposal_id": _get_prop_field(base_prop, "proposal_id"),
                    "recovery_attempt": next_attempt,
                }
                if acc_spec:
                    new_prop_dict["acceptance_criteria"] = acc_spec
                if prop_by_ref is not None:
                    prop_by_ref[new_pid] = new_prop_dict
                    prop_by_ref[new_cid] = new_prop_dict
                if turn_ledger is not None:
                    if hasattr(turn_ledger, "record_recovery_attempt") and assessment.call_id:
                        # Update attempt count cleanly on original ledger record if present
                        if assessment.call_id in getattr(turn_ledger, "_records", {}):
                            turn_ledger.record_recovery_attempt(assessment.call_id)
                    if hasattr(turn_ledger, "_proposal_meta"):
                        turn_ledger._proposal_meta[new_cid] = dict(new_prop_dict)

                history.append(
                    {
                        "attempt": next_attempt,
                        "timestamp": eff_now,
                        "failure_category": assessment.failure_category,
                        "decision": decision,
                        "previous_status": prev_status,
                        "new_status": TaskNodeStatus.READY.value,
                        "reason": _sanitize_recovery_text(assessment.reason, max_bytes=200),
                        "verified": False,
                        "retry_call_id": new_cid,
                    }
                )
                new_rs = validate_recovery_state_dict(
                    {
                        "recovery_version": RECOVERY_STATE_VERSION,
                        "attempt_count": next_attempt,
                        "max_attempts": assessment.max_attempts,
                        "recovery_decision": decision,
                        "recovery_status": "scheduled_retry",
                        "previous_failure_category": assessment.failure_category,
                        "retry_deadline": assessment.retry_deadline,
                        "next_retry_after": next_after,
                        "may_have_side_effect": assessment.may_have_side_effect,
                        "execution_started": assessment.execution_started,
                        "reconciliation_required": False,
                        "verified_outcome": False,
                        "next_recovery_action": assessment.proposed_next_action or "dispatch_retry",
                        "decision_reason": _sanitize_recovery_text(assessment.reason, max_bytes=256),
                        "recovery_history": history[-24:],
                    }
                )
                graph.prepare_node_for_retry(
                    node_id,
                    new_proposal_id=new_pid,
                    recovery_state=new_rs,
                    known_proposals=prop_by_ref,
                )
                new_status = graph.get_node(node_id).execution_status
                cls._record_recovery_ledger_event(
                    central_ledger,
                    exec_id,
                    assessment,
                    previous_status=prev_status,
                    new_status=new_status,
                    attempt_number=next_attempt,
                    status_label="retry_scheduled",
                    retry_call_id=new_cid,
                )
                return {
                    "node_id": node_id,
                    "decision": decision,
                    "action": "retry_scheduled",
                    "retried": True,
                    "promoted_verified": False,
                    "attempt_number": next_attempt,
                    "backoff_seconds": assessment.backoff_seconds,
                    "retry_proposal": new_prop_dict,
                    "previous_status": prev_status,
                    "new_status": new_status,
                }

            # Non-retried recovery outcomes (REQUIRE_CONFIRMATION, REPLAN_NODE, BLOCKED, UNRESOLVED)
            if decision == RecoveryDecision.REQUIRE_CONFIRMATION.value:
                rec_status = "reconciliation_required" if assessment.reconciliation_required else "awaiting_confirmation"
            elif decision == RecoveryDecision.REPLAN_NODE.value:
                rec_status = "replan_required"
            elif attempt_count_exhausted(assessment):
                rec_status = "exhausted"
            elif assessment.retry_deadline is not None and eff_now > assessment.retry_deadline:
                rec_status = "deadline_exceeded"
            else:
                rec_status = "blocked"

            history.append(
                {
                    "attempt": next_attempt,
                    "timestamp": eff_now,
                    "failure_category": assessment.failure_category,
                    "decision": decision,
                    "previous_status": prev_status,
                    "new_status": prev_status,
                    "reason": _sanitize_recovery_text(assessment.reason, max_bytes=200),
                    "verified": False,
                }
            )
            new_rs = validate_recovery_state_dict(
                {
                    "recovery_version": RECOVERY_STATE_VERSION,
                    "attempt_count": next_attempt,
                    "max_attempts": assessment.max_attempts,
                    "recovery_decision": decision,
                    "recovery_status": rec_status,
                    "previous_failure_category": assessment.failure_category,
                    "retry_deadline": assessment.retry_deadline,
                    "may_have_side_effect": assessment.may_have_side_effect,
                    "execution_started": assessment.execution_started,
                    "reconciliation_required": assessment.reconciliation_required,
                    "verified_outcome": False,
                    "next_recovery_action": assessment.proposed_next_action,
                    "decision_reason": _sanitize_recovery_text(assessment.reason, max_bytes=256),
                    "recovery_history": history[-24:],
                }
            )
            target_transition: Optional[str] = None
            if prev_status == TaskNodeStatus.RUNNING.value:
                target_transition = (
                    TaskNodeStatus.BLOCKED.value
                    if (assessment.reconciliation_required or decision == RecoveryDecision.BLOCKED.value)
                    else TaskNodeStatus.FAILED.value
                )
            graph.record_node_recovery_state(
                node_id,
                new_rs,
                transition_status=target_transition,
                blocked_reason=assessment.blocked_reason or node.blocked_reason,
                known_proposals=prop_by_ref,
            )
            new_status = graph.get_node(node_id).execution_status
            cls._record_recovery_ledger_event(
                central_ledger,
                exec_id,
                assessment,
                previous_status=prev_status,
                new_status=new_status,
                attempt_number=next_attempt,
                status_label=rec_status,
            )
            return {
                "node_id": node_id,
                "decision": decision,
                "action": rec_status,
                "retried": False,
                "promoted_verified": False,
                "previous_status": prev_status,
                "new_status": new_status,
                "blocked_dependents": [d["node_id"] for d in graph.get_blocked_dependents(node_id)],
            }
        finally:
            graph.end_node_recovery(node_id, eff_owner)

    @classmethod
    def _record_recovery_ledger_event(
        cls,
        central_ledger: Optional[CentralExecutionLedger],
        execution_id: str,
        assessment: RecoveryAssessment,
        *,
        previous_status: str,
        new_status: str,
        attempt_number: int,
        status_label: str,
        retry_call_id: Optional[str] = None,
    ) -> None:
        if central_ledger is None or not execution_id:
            return
        ver_status = (
            "verified"
            if assessment.verified_outcome
            else ("failed" if (assessment.verification_result or {}).get("passed") is False else "not_verified")
        )
        meta: Dict[str, Any] = {
            "node_id": assessment.node_id,
            "previous_status": previous_status,
            "new_status": new_status,
            "failure_category": assessment.failure_category,
            "recovery_decision": assessment.recovery_decision,
            "attempt": int(attempt_number),
            "max_attempts": int(assessment.max_attempts),
            "verification_result": ver_status,
            "verified_outcome": bool(assessment.verified_outcome),
            "may_have_side_effect": bool(assessment.may_have_side_effect),
            "execution_started": bool(assessment.execution_started),
            "reconciliation_required": bool(assessment.reconciliation_required),
            "retry_deadline": assessment.retry_deadline,
            "reason": _sanitize_recovery_text(assessment.reason, max_bytes=256),
            "next_recovery_action": assessment.proposed_next_action,
            "evidence_refs": list(assessment.evidence_refs[:8]),
            "affected_dependents": list(assessment.affected_dependents),
        }
        if retry_call_id:
            meta["retry_call_id"] = retry_call_id
        central_ledger.record_event(
            execution_id,
            LedgerEventType.RECOVERY_ATTEMPT,
            tool_id=assessment.tool_name,
            status=status_label,
            correlation_id=assessment.call_id or assessment.node_id,
            metadata=meta,
            idempotency_key=f"{execution_id}:recovery:{assessment.node_id}:{attempt_number}:{status_label}",
        )

    @classmethod
    def recover_node_with_alternative(
        cls,
        graph: TaskGraph,
        failed_node_id: str,
        *,
        replacement_node_id: str,
        description: str,
        tool_id: str,
        arguments: Dict[str, Any],
        extra_dependencies: Sequence[str] = (),
        verification_requirement: Optional[str] = None,
        expected_outcome: str = "",
        timeout_seconds: Optional[float] = None,
        prop_by_ref: Optional[Dict[str, Any]] = None,
        central_ledger: Optional[CentralExecutionLedger] = None,
        turn_ledger: Optional[TurnExecutionLedger] = None,
        checkpoint: bool = True,
    ) -> Tuple[TaskGraph, Dict[str, Any]]:
        """Replace a failed/blocked node with an alternative operation while preserving dependencies."""
        from zelyr.engine.checkpoint import CheckpointManager

        target = graph.get_node(failed_node_id)
        if target is None:
            raise TaskGraphValidationError(f"Cannot replace unknown node '{failed_node_id}'.")
        if target.execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value:
            raise TaskGraphValidationError(
                f"Cannot replace completed verified node '{failed_node_id}'."
            )

        # Preserve every existing prerequisite of failed_node_id!
        preserved_deps = list(target.dependency_ids)
        for extra_d in extra_dependencies:
            ed = str(extra_d).strip()
            if ed and ed not in preserved_deps:
                preserved_deps.append(ed)

        new_pid = f"prop_{replacement_node_id}"
        new_cid = f"call_{replacement_node_id}"
        clean_args = _strip_forbidden_graph_keys(dict(arguments or {}))
        new_prop = {
            "proposal_id": new_pid,
            "execution_id": graph.execution_id,
            "original_call_id": new_cid,
            "tool_id": str(tool_id).strip(),
            "raw_name": str(tool_id).strip(),
            "arguments": dict(clean_args),
            "normalized_arguments": dict(clean_args),
            "proposal_order": target.order_index,
            "depends_on": [f"prop_{d}" for d in preserved_deps],
            "status": "pending",
            "authorization_status": "pending_revalidation",
            "supersedes_node_id": failed_node_id,
        }
        if prop_by_ref is not None:
            prop_by_ref[new_pid] = new_prop
            prop_by_ref[new_cid] = new_prop
        if turn_ledger is not None and hasattr(turn_ledger, "_proposal_meta"):
            turn_ledger._proposal_meta[new_cid] = dict(new_prop)

        alt_node = TaskNode(
            node_id=replacement_node_id,
            description=_sanitize_recovery_text(description, max_bytes=512),
            proposal_ids=[new_pid],
            dependency_ids=preserved_deps,
            requires_verified_dependencies=target.requires_verified_dependencies,
            order_index=target.order_index,
            expected_outcome=_sanitize_recovery_text(expected_outcome, max_bytes=512),
            verification_requirement=verification_requirement,
            timeout_seconds=timeout_seconds,
        )
        prof = extract_node_conflict_profile(tool_id, clean_args, node=alt_node)
        alt_node.conflict_resources = [k for k in prof.lock_keys if not k.startswith("op:")]
        alt_node.exclusive_resource = prof.exclusive_resource

        def _mutate(g: TaskGraph) -> None:
            g.replace_failed_node(
                failed_node_id,
                alt_node,
                rewire_dependents=True,
                known_proposals=prop_by_ref,
            )

        if central_ledger is not None:
            CheckpointManager.mutate_task_graph(
                central_ledger,
                graph.execution_id,
                graph,
                mutation_type="recovery_replace_node",
                node_id=replacement_node_id,
                mutation_fn=_mutate,
                turn_ledger=turn_ledger,
                known_proposals=prop_by_ref,
                checkpoint=checkpoint,
            )
            central_ledger.record_event(
                graph.execution_id,
                LedgerEventType.RECOVERY_ATTEMPT,
                tool_id=tool_id,
                status="alternative_node_substituted",
                correlation_id=new_cid,
                metadata={
                    "node_id": failed_node_id,
                    "replacement_node_id": replacement_node_id,
                    "previous_status": target.execution_status,
                    "new_status": graph.get_node(replacement_node_id).execution_status,
                    "recovery_decision": RecoveryDecision.REPLAN_NODE.value,
                    "attempt": int((target.recovery_state or {}).get("attempt_count") or 0),
                },
                idempotency_key=f"{graph.execution_id}:recovery_alt:{failed_node_id}:{replacement_node_id}",
            )
        else:
            _mutate(graph)
        return graph, new_prop

    @classmethod
    def add_corrective_step(
        cls,
        graph: TaskGraph,
        failed_node_id: str,
        *,
        corrective_node_id: str,
        description: str,
        tool_id: str,
        arguments: Dict[str, Any],
        verification_requirement: Optional[str] = None,
        expected_outcome: str = "",
        prop_by_ref: Optional[Dict[str, Any]] = None,
        central_ledger: Optional[CentralExecutionLedger] = None,
        turn_ledger: Optional[TurnExecutionLedger] = None,
        checkpoint: bool = True,
    ) -> Tuple[TaskGraph, Dict[str, Any]]:
        """Insert a corrective prerequisite node before retrying ``failed_node_id`` without dropping dependencies."""
        from zelyr.engine.checkpoint import CheckpointManager

        failed_n = graph.get_node(failed_node_id)
        if failed_n is None:
            raise TaskGraphValidationError(f"Cannot add corrective step for unknown node '{failed_node_id}'.")
        if failed_n.execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value:
            raise TaskGraphValidationError(
                f"Cannot modify already completed verified node '{failed_node_id}'."
            )

        new_pid = f"prop_{corrective_node_id}"
        new_cid = f"call_{corrective_node_id}"
        clean_args = _strip_forbidden_graph_keys(dict(arguments or {}))
        corr_prop = {
            "proposal_id": new_pid,
            "execution_id": graph.execution_id,
            "original_call_id": new_cid,
            "tool_id": str(tool_id).strip(),
            "raw_name": str(tool_id).strip(),
            "arguments": dict(clean_args),
            "normalized_arguments": dict(clean_args),
            "proposal_order": failed_n.order_index,
            "depends_on": [f"prop_{d}" for d in failed_n.dependency_ids],
            "status": "pending",
            "authorization_status": "pending_revalidation",
            "corrective_for_node_id": failed_node_id,
        }
        if prop_by_ref is not None:
            prop_by_ref[new_pid] = corr_prop
            prop_by_ref[new_cid] = corr_prop
        if turn_ledger is not None and hasattr(turn_ledger, "_proposal_meta"):
            turn_ledger._proposal_meta[new_cid] = dict(corr_prop)

        corr_node = TaskNode(
            node_id=corrective_node_id,
            description=_sanitize_recovery_text(description, max_bytes=512),
            proposal_ids=[new_pid],
            dependency_ids=list(failed_n.dependency_ids),
            requires_verified_dependencies=failed_n.requires_verified_dependencies,
            order_index=failed_n.order_index,
            expected_outcome=_sanitize_recovery_text(expected_outcome, max_bytes=512),
            verification_requirement=verification_requirement,
        )
        prof = extract_node_conflict_profile(tool_id, clean_args, node=corr_node)
        corr_node.conflict_resources = [k for k in prof.lock_keys if not k.startswith("op:")]
        corr_node.exclusive_resource = prof.exclusive_resource

        def _mutate(g: TaskGraph) -> None:
            g.add_corrective_node(
                failed_node_id,
                corr_node,
                make_prerequisite_of_failed=True,
                reset_failed_to_pending=True,
                known_proposals=prop_by_ref,
            )

        if central_ledger is not None:
            CheckpointManager.mutate_task_graph(
                central_ledger,
                graph.execution_id,
                graph,
                mutation_type="recovery_add_corrective_node",
                node_id=corrective_node_id,
                mutation_fn=_mutate,
                turn_ledger=turn_ledger,
                known_proposals=prop_by_ref,
                checkpoint=checkpoint,
            )
            central_ledger.record_event(
                graph.execution_id,
                LedgerEventType.RECOVERY_ATTEMPT,
                tool_id=tool_id,
                status="corrective_node_added",
                correlation_id=new_cid,
                metadata={
                    "node_id": failed_node_id,
                    "corrective_node_id": corrective_node_id,
                    "recovery_decision": RecoveryDecision.REPLAN_NODE.value,
                    "attempt": int((failed_n.recovery_state or {}).get("attempt_count") or 0),
                },
                idempotency_key=f"{graph.execution_id}:recovery_corr:{failed_node_id}:{corrective_node_id}",
            )
        else:
            _mutate(graph)
        return graph, corr_prop

    @classmethod
    def replan_affected_subgraph(
        cls,
        graph: TaskGraph,
        replan_payload: Union[str, Dict[str, Any]],
        *,
        central_ledger: Optional[CentralExecutionLedger] = None,
        turn_ledger: Optional[TurnExecutionLedger] = None,
        known_proposals: Optional[Union[Dict[str, Any], Iterable[Any]]] = None,
        workspace_root: Optional[Any] = None,
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
        tool_registry: Optional[Any] = None,
        checkpoint: bool = True,
    ) -> TaskGraph:
        """Replan only failed/blocked portions of a TaskGraph while preserving verified nodes."""
        from zelyr.planning.compiler import replan_task_graph

        return replan_task_graph(
            graph,
            replan_payload,
            central_ledger=central_ledger,
            turn_ledger=turn_ledger,
            known_proposals=known_proposals,
            workspace_root=workspace_root,
            tool_schemas=tool_schemas,
            tool_registry=tool_registry,
            checkpoint=checkpoint,
        )


def attempt_count_exhausted(assessment: RecoveryAssessment) -> bool:
    return assessment.attempt_count >= assessment.max_attempts
