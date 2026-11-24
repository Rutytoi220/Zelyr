"""Read-only end-to-end execution lifecycle diagnostics across ledger, checkpoint, graph, observation, recovery, and policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from zelyr.engine.checkpoint import CheckpointManager, ResumeAssessment
from zelyr.engine.execution_ledger import (
    CentralExecutionLedger,
    ExecutionEvent,
    ExecutionRecord,
    LedgerEventType,
    VerificationStatus,
    _redact_sensitive_text,
    sanitize_ledger_payload,
)
from zelyr.engine.execution_policy import (
    format_authorization_history_entry,
    query_policy_history,
)
from zelyr.engine.recovery import build_graph_recovery_overview
from zelyr.engine.task_graph import TaskGraph, TaskNodeStatus


_FORBIDDEN_DIAGNOSTIC_KEYS: Set[str] = {
    "arguments",
    "raw_arguments",
    "normalized_arguments",
    "confirmation_token",
    "confirm_token",
    "raw_token",
    "token_secret",
    "last_assistant_text",
}


def _safe_diag_text(value: Any, *, max_len: int = 512) -> Optional[str]:
    if value is None:
        return None
    scrubbed = _redact_sensitive_text(str(value)) or ""
    if len(scrubbed) > max_len:
        return scrubbed[: max_len - 3] + "..."
    return scrubbed


def _safe_diag_dict(mapping: Any, *, max_bytes: int = 4096) -> Dict[str, Any]:
    if not isinstance(mapping, Mapping):
        return {}
    stripped = {
        str(k): v
        for k, v in mapping.items()
        if str(k).lower() not in _FORBIDDEN_DIAGNOSTIC_KEYS
    }
    cleaned, _, _ = sanitize_ledger_payload(stripped, max_bytes=max_bytes)
    return cleaned if isinstance(cleaned, dict) else {}


def _dedup_preserve(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for item in items:
        s = _safe_diag_text(item, max_len=512) or ""
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


@dataclass(frozen=True)
class ExecutionLifecycleReport:
    """Structured, redacted, read-only end-to-end diagnostic report for a single execution."""

    execution_id: str
    found: bool
    request: Dict[str, Any] = field(default_factory=dict)
    plan_and_graph: Dict[str, Any] = field(default_factory=dict)
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    node_summary: Dict[str, Any] = field(default_factory=dict)
    tools: Dict[str, Any] = field(default_factory=dict)
    policy: Dict[str, Any] = field(default_factory=dict)
    verification: Dict[str, Any] = field(default_factory=dict)
    recovery: Dict[str, Any] = field(default_factory=dict)
    resumability: Dict[str, Any] = field(default_factory=dict)
    persistence_errors: List[Dict[str, Any]] = field(default_factory=list)
    timeline_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExecutionLifecycleDiagnostics:
    """Aggregates ledger, checkpoint, TaskGraph, observation, recovery, and policy into one read-only diagnostic view."""

    @staticmethod
    def diagnose(
        ledger: CentralExecutionLedger,
        execution_id: str,
        *,
        verify_outcomes: bool = False,
        tool_registry: Optional[Any] = None,
    ) -> ExecutionLifecycleReport:
        """Build a strictly read-only end-to-end ExecutionLifecycleReport from persisted state."""
        exec_id = str(execution_id or "")
        record: Optional[ExecutionRecord] = ledger.get_execution(exec_id) if exec_id else None
        persistence_errors = [
            err
            for err in (ledger.get_persistence_errors() if hasattr(ledger, "get_persistence_errors") else [])
            if err.get("execution_id") in (exec_id, None)
        ]

        if record is None:
            blockers = [f"Execution '{exec_id}' does not exist in execution ledger."]
            if persistence_errors:
                blockers.append(
                    f"Ledger persistence error: {persistence_errors[-1].get('error', 'database error')}"
                )
            return ExecutionLifecycleReport(
                execution_id=exec_id,
                found=False,
                request={"execution_id": exec_id, "status": "not_found", "verification_status": "unknown"},
                resumability={
                    "can_resume": False,
                    "resumability_status": "blocked",
                    "requires_explicit_resume": False,
                    "blocking_reasons": _dedup_preserve(blockers),
                    "warnings": [],
                },
                persistence_errors=persistence_errors,
            )

        events: List[ExecutionEvent] = ledger.get_events(record.execution_id)
        observations: List[Dict[str, Any]] = (
            ledger.get_observations(record.execution_id)
            if hasattr(ledger, "get_observations")
            else []
        )
        assessment: ResumeAssessment = CheckpointManager.assess_resume(
            ledger,
            record.execution_id,
            tool_registry=tool_registry,
            verify_outcomes=verify_outcomes,
        )

        rec_dict = record.to_dict()
        planning_state = _safe_diag_dict(assessment.planning_state)
        request_info = {
            "execution_id": record.execution_id,
            "parent_execution_id": record.parent_execution_id,
            "task_prompt": _safe_diag_text(record.task_prompt, max_len=1024) or "",
            "status": record.status,
            "verification_status": record.verification_status,
            "source_component": record.source_component,
            "caller_ring": record.caller_ring,
            "task_local_scoped": bool(record.task_local_scoped),
            "created_at": rec_dict.get("created_at"),
            "updated_at": rec_dict.get("updated_at"),
            "completed_at": rec_dict.get("completed_at"),
            "final_summary": _safe_diag_text(record.final_summary, max_len=1024),
            "error": _safe_diag_text(record.error, max_len=1024),
            "event_count": len(events),
        }

        # ── 1. Plan & Graph ────────────────────────────────────────────────────
        planning_events: List[Dict[str, Any]] = []
        recovery_events: List[Dict[str, Any]] = []
        verification_events: List[Dict[str, Any]] = []
        seen_tools: List[str] = []
        external_tools: List[Dict[str, Any]] = []
        seen_ext: Set[tuple[Optional[str], Optional[str]]] = set()
        arg_keys_by_call: Dict[str, List[str]] = {}
        event_type_counts: Dict[str, int] = {}

        for ev in events:
            event_type_counts[ev.event_type] = event_type_counts.get(ev.event_type, 0) + 1
            meta = ev.metadata if isinstance(ev.metadata, dict) else {}
            if ev.tool_id and ev.tool_id not in seen_tools:
                seen_tools.append(ev.tool_id)
            if ev.is_external and (ev.tool_id, ev.source_id) not in seen_ext:
                seen_ext.add((ev.tool_id, ev.source_id))
                external_tools.append(
                    {
                        "tool_id": ev.tool_id,
                        "source_id": ev.source_id,
                        "source_type": ev.source_type,
                    }
                )
            if ev.correlation_id and isinstance(meta.get("arguments"), Mapping):
                arg_keys_by_call[ev.correlation_id] = sorted(str(k) for k in meta["arguments"].keys())

            if ev.event_type == LedgerEventType.PLANNING_DECISION.value:
                planning_events.append(
                    {
                        "sequence_number": ev.sequence_number,
                        "timestamp": ev.timestamp,
                        "status": ev.status,
                        "mutation_type": meta.get("mutation_type"),
                        "graph_id": meta.get("graph_id"),
                        "node_id": meta.get("node_id"),
                        "graph_status": meta.get("graph_status"),
                        "plan_id": meta.get("plan_id"),
                        "reason": _safe_diag_text(meta.get("reason") or meta.get("detail"), max_len=256),
                    }
                )
            elif ev.event_type == LedgerEventType.RECOVERY_ATTEMPT.value:
                recovery_events.append(
                    {
                        "sequence_number": ev.sequence_number,
                        "timestamp": ev.timestamp,
                        "tool_id": ev.tool_id,
                        "correlation_id": ev.correlation_id,
                        "status": ev.status,
                        "attempt": meta.get("attempt"),
                        "node_id": meta.get("node_id"),
                        "recovery_decision": meta.get("recovery_decision"),
                        "failure_category": meta.get("failure_category"),
                        "next_recovery_action": meta.get("next_recovery_action"),
                        "may_have_side_effect": meta.get("may_have_side_effect"),
                        "reconciliation_required": meta.get("reconciliation_required"),
                        "reason": _safe_diag_text(meta.get("reason") or meta.get("detail"), max_len=256),
                    }
                )
            elif ev.event_type == LedgerEventType.VERIFICATION_RECORDED.value:
                verification_events.append(
                    {
                        "sequence_number": ev.sequence_number,
                        "timestamp": ev.timestamp,
                        "tool_id": ev.tool_id,
                        "correlation_id": ev.correlation_id,
                        "status": ev.status,
                        "target": _safe_diag_text(meta.get("target"), max_len=256),
                        "detail": _safe_diag_text(meta.get("detail"), max_len=256),
                    }
                )

        tg: Optional[TaskGraph] = assessment.task_graph
        graph_overview = build_graph_recovery_overview(tg)
        plan_and_graph = {
            "has_plan": bool(
                planning_state.get("plan_id")
                or planning_state.get("acceptance_criteria")
                or planning_events
                or tg is not None
            ),
            "plan_id": planning_state.get("plan_id"),
            "plan_version": planning_state.get("plan_version"),
            "plan_confirmed": planning_state.get("plan_confirmed"),
            "requires_confirmation": planning_state.get("requires_confirmation"),
            "acceptance_criteria": _safe_diag_text(planning_state.get("acceptance_criteria"), max_len=512),
            "planning_decisions": planning_events,
            "has_graph": tg is not None,
            "graph_id": tg.graph_id if tg is not None else None,
            "graph_version": getattr(tg, "graph_version", None) if tg is not None else None,
            "graph_status": tg.status if tg is not None else None,
            "max_concurrency": getattr(tg, "max_concurrency", None) if tg is not None else None,
            "graph_summary": dict(tg.summary or {}) if tg is not None else {},
            "edge_count": len(getattr(tg, "edges", []) or []) if tg is not None else 0,
        }

        # ── 2. Nodes (Ordered) ─────────────────────────────────────────────────
        nodes_out: List[Dict[str, Any]] = []
        ran_ids: List[str] = []
        pending_ids: List[str] = []
        blocked_ids: List[str] = []
        verified_node_ids: List[str] = []
        unverified_node_ids: List[str] = []
        failed_node_ids: List[str] = []
        contradictory_node_ids: List[str] = []

        if tg is not None:
            for node in sorted(tg.nodes, key=lambda n: (int(n.order_index), str(n.node_id))):
                rs = dict(node.recovery_state) if isinstance(node.recovery_state, dict) else {}
                fail_info = _safe_diag_dict(node.failure_info, max_bytes=1024)
                node_ran = node.execution_status not in (
                    TaskNodeStatus.PENDING.value,
                    TaskNodeStatus.READY.value,
                    TaskNodeStatus.BLOCKED.value,
                )
                if node_ran:
                    ran_ids.append(node.node_id)
                if node.execution_status == TaskNodeStatus.PENDING.value:
                    pending_ids.append(node.node_id)
                elif node.execution_status == TaskNodeStatus.BLOCKED.value:
                    blocked_ids.append(node.node_id)
                elif node.execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value:
                    verified_node_ids.append(node.node_id)
                elif node.execution_status == TaskNodeStatus.COMPLETED_UNVERIFIED.value:
                    unverified_node_ids.append(node.node_id)
                elif node.execution_status == TaskNodeStatus.FAILED.value:
                    failed_node_ids.append(node.node_id)
                if node.contradictory:
                    contradictory_node_ids.append(node.node_id)

                nodes_out.append(
                    {
                        "node_id": node.node_id,
                        "order_index": int(node.order_index),
                        "description": _safe_diag_text(node.description, max_len=256),
                        "proposal_ids": list(node.proposal_ids),
                        "dependency_ids": list(node.dependency_ids),
                        "execution_status": node.execution_status,
                        "verification_status": node.verification_status,
                        "ran": node_ran,
                        "blocked_reason": _safe_diag_text(node.blocked_reason, max_len=256),
                        "contradictory": bool(node.contradictory),
                        "recoverable": node.recoverable,
                        "error": _safe_diag_text(fail_info.get("error"), max_len=256) if fail_info else None,
                        "error_type": fail_info.get("error_type") if fail_info else None,
                        "recovery_attempts": int(rs.get("attempt_count") or 0),
                        "recovery_decision": rs.get("recovery_decision"),
                        "recovery_status": rs.get("recovery_status"),
                        "next_recovery_action": rs.get("next_recovery_action"),
                        "conflict_resources": list(node.conflict_resources or []),
                        "exclusive_resource": node.exclusive_resource,
                    }
                )

        node_summary = {
            "total": len(nodes_out),
            "ran_count": len(ran_ids),
            "ran_node_ids": ran_ids,
            "pending_node_ids": pending_ids,
            "blocked_node_ids": blocked_ids,
            "completed_verified_node_ids": verified_node_ids,
            "completed_unverified_node_ids": unverified_node_ids,
            "failed_node_ids": failed_node_ids,
            "contradictory_node_ids": contradictory_node_ids,
        }

        # ── 3. Tool Calls (Strictly Redacted: Key Names Only, Never Raw Args) ───
        all_ops = (
            list(assessment.completed_verified_ops)
            + list(assessment.completed_unverified_ops)
            + list(assessment.in_flight_ambiguous_ops)
            + list(assessment.failed_ops)
            + list(assessment.contradictory_ops)
            + list(assessment.pending_ops)
        )
        tool_calls_out: List[Dict[str, Any]] = []
        for op in all_ops:
            arg_keys = sorted(str(k) for k in (op.arguments or {}).keys()) or arg_keys_by_call.get(op.call_id, [])
            tool_calls_out.append(
                {
                    "call_id": op.call_id,
                    "tool_id": op.tool_name,
                    "dispatched": bool(op.dispatched),
                    "has_result": bool(op.has_result),
                    "lifecycle_state": op.lifecycle_state,
                    "evidence_level": op.evidence_level,
                    "execution_status": op.execution_status,
                    "verification_status": op.verification_status,
                    "verified": bool(op.verified),
                    "may_have_side_effect": bool(op.may_have_side_effect),
                    "recovery_attempts": int(op.recovery_attempts),
                    "error_type": op.error_type,
                    "error": _safe_diag_text(op.error, max_len=256),
                    "is_external": bool(op.is_external),
                    "source_id": op.source_id,
                    "source_type": op.source_type,
                    "recommended_action": op.recommended_action,
                    "argument_keys": arg_keys,
                }
            )

        tools_info = {
            "unique_tools": seen_tools,
            "external_tools": external_tools,
            "call_count": len(tool_calls_out),
            "calls": tool_calls_out,
        }

        # ── 4. Policy Decisions ────────────────────────────────────────────────
        hist_res = query_policy_history(ledger, execution_id=record.execution_id, limit=500)
        pol_entries = list(hist_res.get("entries") or [])
        if not pol_entries:
            # Also handle fallback if ledger instance lacks helper or used alternate row lookup
            pol_events = [
                ev
                for ev in events
                if ev.event_type == LedgerEventType.AUTHORIZATION_DECISION.value
            ]
            pol_entries = [format_authorization_history_entry(ev) for ev in reversed(pol_events)]

        outcome_counts: Dict[str, int] = {"allow": 0, "require_confirmation": 0, "deny": 0}
        for entry in pol_entries:
            out_k = str(entry.get("outcome") or "").lower()
            outcome_counts[out_k] = outcome_counts.get(out_k, 0) + 1

        policy_info = {
            "decision_count": len(pol_entries),
            "outcome_counts": outcome_counts,
            "decisions": list(reversed(pol_entries)),  # chronological for lifecycle trace
            "resume_reauthorization_checks": _safe_diag_dict(assessment.authorization_checks),
        }

        # ── 5. Verification vs Reported-Only vs Failed/Uncertain ───────────────
        verified_ops = [
            {
                "kind": "operation",
                "call_id": op.call_id,
                "tool_id": op.tool_name,
                "evidence_level": op.evidence_level or "CONFIRMED",
                "verification_status": op.verification_status or "verified",
                "notes": _safe_diag_text(op.notes, max_len=256),
            }
            for op in assessment.completed_verified_ops
        ]
        for ve in verification_events:
            if ve["status"] == VerificationStatus.VERIFIED.value:
                verified_ops.append(
                    {
                        "kind": "verification_event",
                        "sequence_number": ve["sequence_number"],
                        "tool_id": ve["tool_id"],
                        "correlation_id": ve["correlation_id"],
                        "target": ve["target"],
                        "detail": ve["detail"],
                    }
                )

        reported_unverified = [
            {
                "kind": "operation",
                "call_id": op.call_id,
                "tool_id": op.tool_name,
                "evidence_level": op.evidence_level or "REPORTED",
                "verification_status": op.verification_status or "not_verified",
                "reason": "Tool reported success without independent outcome verification.",
            }
            for op in assessment.completed_unverified_ops
        ]
        has_untrusted_claim = bool(
            (assessment.planning_state or {}).get("last_assistant_text")
        )
        if has_untrusted_claim:
            reported_unverified.append(
                {
                    "kind": "assistant_claim",
                    "evidence_level": "REPORTED",
                    "verification_status": "not_verified",
                    "reason": "Assistant text claimed completion or progress; treated as untrusted reported claim, not verification proof.",
                }
            )

        failed_or_uncertain: List[Dict[str, Any]] = []
        for op in assessment.failed_ops:
            failed_or_uncertain.append(
                {
                    "kind": "operation_failed",
                    "call_id": op.call_id,
                    "tool_id": op.tool_name,
                    "lifecycle_state": op.lifecycle_state,
                    "error_type": op.error_type,
                    "error": _safe_diag_text(op.error, max_len=256),
                }
            )
        for op in assessment.in_flight_ambiguous_ops:
            failed_or_uncertain.append(
                {
                    "kind": "operation_ambiguous",
                    "call_id": op.call_id,
                    "tool_id": op.tool_name,
                    "lifecycle_state": op.lifecycle_state,
                    "may_have_side_effect": bool(op.may_have_side_effect),
                    "reason": _safe_diag_text(op.notes, max_len=256),
                }
            )
        for op in assessment.contradictory_ops:
            failed_or_uncertain.append(
                {
                    "kind": "operation_contradictory",
                    "call_id": op.call_id,
                    "tool_id": op.tool_name,
                    "lifecycle_state": op.lifecycle_state,
                    "error": _safe_diag_text(op.error, max_len=256),
                    "reason": _safe_diag_text(op.notes, max_len=256),
                }
            )
        for obs in observations:
            unc = obs.get("remaining_uncertainty")
            ev_lv = obs.get("evidence_level")
            if unc or ev_lv in ("UNKNOWN", "PARTIAL", "CONTRADICTED"):
                failed_or_uncertain.append(
                    {
                        "kind": "observation_uncertainty",
                        "sequence_number": obs.get("sequence_number"),
                        "tool_id": obs.get("tool_id"),
                        "correlation_id": obs.get("correlation_id"),
                        "evidence_level": ev_lv,
                        "execution_status": obs.get("execution_status"),
                        "uncertainty": _safe_diag_text(unc, max_len=256),
                    }
                )

        verification_info = {
            "overall_verification_status": record.verification_status,
            "reported_vs_verified_distinction_preserved": True,
            "verified_count": len(assessment.completed_verified_ops),
            "reported_only_unverified_count": len(assessment.completed_unverified_ops),
            "failed_or_uncertain_count": len(failed_or_uncertain),
            "has_unverified_assistant_claim": has_untrusted_claim,
            "independently_verified": verified_ops,
            "reported_only_unverified": reported_unverified,
            "failed_or_uncertain": failed_or_uncertain,
            "reconciled_verifications": list(assessment.reconciled_verifications),
        }

        # ── 6. Recovery ────────────────────────────────────────────────────────
        total_rec_attempts = max(len(recovery_events), int(graph_overview.get("total_attempts") or 0))
        recovery_info = {
            "attempted": total_rec_attempts > 0 or bool(graph_overview.get("recovering_nodes")),
            "attempt_count": total_rec_attempts,
            "events": recovery_events,
            "graph_overview": _safe_diag_dict(graph_overview, max_bytes=4096),
        }

        # ── 7. Resumability & Blockers ─────────────────────────────────────────
        all_blockers = list(assessment.blocking_reasons)
        for dep_blk in graph_overview.get("blocked_dependents") or []:
            blk_txt = f"Node '{dep_blk.get('node_id')}' blocked: {dep_blk.get('reason')}"
            all_blockers.append(blk_txt)
        for entry in pol_entries:
            if entry.get("outcome") == "deny":
                all_blockers.append(
                    f"Policy denied '{entry.get('tool_id')}' ({entry.get('reason_code')})"
                )
        for err in persistence_errors:
            all_blockers.append(
                f"Ledger persistence error ({err.get('operation', 'io')}): {err.get('error', 'error')}"
            )

        active_pending_cnt = sum(
            1
            for p in assessment.pending_actions
            if p.status in ("pending", "validated", "blocked", "dispatched")
        )
        resumability_info = {
            "can_resume": bool(assessment.can_resume) and not bool(persistence_errors),
            "resumability_status": assessment.resumability_status,
            "requires_explicit_resume": bool(assessment.requires_explicit_resume),
            "checkpoint_id": assessment.checkpoint_id,
            "checkpoint_version": assessment.checkpoint_version,
            "events_after_checkpoint": int(assessment.events_after_checkpoint),
            "pending_proposals_count": active_pending_cnt,
            "dispatchable_proposal_ids": list(assessment.to_dict().get("dispatchable_proposal_ids") or []),
            "blocking_reasons": _dedup_preserve(all_blockers),
            "warnings": _dedup_preserve(assessment.warnings),
            "environment_diff": _safe_diag_dict(assessment.environment_diff, max_bytes=2048),
        }

        timeline_summary = {
            "event_count": len(events),
            "event_type_counts": event_type_counts,
            "first_event_ts": events[0].timestamp if events else None,
            "last_event_ts": events[-1].timestamp if events else None,
        }

        return ExecutionLifecycleReport(
            execution_id=record.execution_id,
            found=True,
            request=request_info,
            plan_and_graph=plan_and_graph,
            nodes=nodes_out,
            node_summary=node_summary,
            tools=tools_info,
            policy=policy_info,
            verification=verification_info,
            recovery=recovery_info,
            resumability=resumability_info,
            persistence_errors=persistence_errors,
            timeline_summary=timeline_summary,
        )


def diagnose_execution_lifecycle(
    ledger: CentralExecutionLedger,
    execution_id: str,
    *,
    verify_outcomes: bool = False,
    tool_registry: Optional[Any] = None,
) -> ExecutionLifecycleReport:
    """Convenience entrypoint returning a read-only ExecutionLifecycleReport."""
    return ExecutionLifecycleDiagnostics.diagnose(
        ledger,
        execution_id,
        verify_outcomes=verify_outcomes,
        tool_registry=tool_registry,
    )


def format_execution_lifecycle_report(report: ExecutionLifecycleReport) -> str:
    """Format an ExecutionLifecycleReport into a concise, human-readable terminal diagnostic view."""
    if not report.found:
        lines = [f"Execution Lifecycle Diagnostic: {report.execution_id} (NOT FOUND)"]
        for reason in report.resumability.get("blocking_reasons") or []:
            lines.append(f"  - {reason}")
        return "\n".join(lines)

    req = report.request
    pg = report.plan_and_graph
    ns = report.node_summary
    tools = report.tools
    pol = report.policy
    ver = report.verification
    rec = report.recovery
    res = report.resumability

    created_ts = req.get("created_at")
    created_str = (
        datetime.datetime.fromtimestamp(float(created_ts)).strftime("%Y-%m-%d %H:%M:%S")
        if created_ts
        else "unknown"
    )
    lines = [
        f"Execution Lifecycle Diagnostic: {report.execution_id}",
        f"  Request:       {req.get('task_prompt') or '(empty)'}",
        f"  Status:        status={req.get('status')}  verification={req.get('verification_status')}  src={req.get('source_component')}  created={created_str}",
    ]
    if req.get("error"):
        lines.append(f"  Error:         {req.get('error')}")

    # Plan & Graph
    if pg.get("has_graph"):
        lines.append(
            f"  Plan & Graph:  graph={pg.get('graph_id')} (v{pg.get('graph_version')}) status={pg.get('graph_status')} "
            f"nodes={ns.get('total', 0)} edges={pg.get('edge_count', 0)} max_conc={pg.get('max_concurrency')}"
        )
    elif pg.get("has_plan"):
        lines.append(
            f"  Plan & Graph:  plan={pg.get('plan_id') or 'present'} (no TaskGraph materialized)"
        )
    else:
        lines.append("  Plan & Graph:  direct/ungraphed execution (no explicit plan/TaskGraph)")

    # Nodes
    if report.nodes:
        lines.append(
            f"  Nodes:         ran={ns.get('ran_count', 0)}/{ns.get('total', 0)}  "
            f"verified={len(ns.get('completed_verified_node_ids', []))}  "
            f"unverified={len(ns.get('completed_unverified_node_ids', []))}  "
            f"failed={len(ns.get('failed_node_ids', []))}  "
            f"blocked={len(ns.get('blocked_node_ids', []))}"
        )
        for n in report.nodes:
            blk = f" blocked={n.get('blocked_reason')}" if n.get("blocked_reason") else ""
            err = f" err={n.get('error_type') or n.get('error')}" if (n.get("error_type") or n.get("error")) else ""
            lines.append(
                f"    [{n['order_index']}] {n['node_id']}: status={n['execution_status']} "
                f"verify={n['verification_status']} deps={n['dependency_ids']}{blk}{err}"
            )

    # Tools
    lines.append(
        f"  Tools Called:  {tools.get('call_count', 0)} call(s) across {tools.get('unique_tools', [])}"
    )
    for tc in tools.get("calls") or []:
        ext_tag = f" [ext:{tc.get('source_id')}]" if tc.get("is_external") else ""
        lines.append(
            f"    - {tc['call_id']} tool={tc['tool_id']}{ext_tag} state={tc['lifecycle_state']} "
            f"evidence={tc.get('evidence_level')} verified={tc['verified']} keys={tc['argument_keys']}"
        )

    # Policy
    oc = pol.get("outcome_counts") or {}
    lines.append(
        f"  Policy:        {pol.get('decision_count', 0)} decision(s) "
        f"(allow={oc.get('allow', 0)}, require_confirmation={oc.get('require_confirmation', 0)}, deny={oc.get('deny', 0)})"
    )
    for pd in pol.get("decisions") or []:
        lines.append(
            f"    - {pd.get('outcome')} tool={pd.get('tool_id')} risk={pd.get('risk_level')} "
            f"stage={pd.get('precedence_stage')} reason={pd.get('reason_code')}"
        )

    # Epistemic verification breakdown
    lines.append(
        f"  Verification:  verified={ver.get('verified_count', 0)}  "
        f"reported_only_unverified={ver.get('reported_only_unverified_count', 0)}  "
        f"failed_or_uncertain={ver.get('failed_or_uncertain_count', 0)}  "
        f"untrusted_claim={ver.get('has_unverified_assistant_claim', False)}"
    )

    # Recovery
    lines.append(
        f"  Recovery:      attempted={rec.get('attempted')} attempts={rec.get('attempt_count', 0)}"
    )
    for rev in rec.get("events") or []:
        lines.append(
            f"    - #{rev.get('sequence_number')} status={rev.get('status')} "
            f"decision={rev.get('recovery_decision')} category={rev.get('failure_category')} "
            f"action={rev.get('next_recovery_action')}"
        )

    # Resumability & Blockers
    lines.append(
        f"  Resumability:  can_resume={res.get('can_resume')} status={res.get('resumability_status')} "
        f"checkpoint={res.get('checkpoint_id') or 'reconstructed'} explicit_confirm={res.get('requires_explicit_resume')}"
    )
    if res.get("blocking_reasons"):
        lines.append("  Blockers:")
        for b in res["blocking_reasons"]:
            lines.append(f"    - {b}")
    if res.get("warnings"):
        lines.append("  Warnings:")
        for w in res["warnings"]:
            lines.append(f"    - {w}")

    return "\n".join(lines)


__all__ = [
    "ExecutionLifecycleDiagnostics",
    "ExecutionLifecycleReport",
    "diagnose_execution_lifecycle",
    "format_execution_lifecycle_report",
]
