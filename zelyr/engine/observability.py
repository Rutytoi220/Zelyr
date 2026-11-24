"""Lightweight, read-only execution and policy observability analytics backed by persisted ledger events."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from zelyr.engine.execution_ledger import (
    CentralExecutionLedger,
    ExecutionEvent,
    ExecutionRecord,
    LedgerEventType,
)


def _as_nonneg_float(val: Any) -> Optional[float]:
    if val is None or isinstance(val, bool):
        return None
    try:
        num = float(val)
        return round(num, 3) if num >= 0.0 else None
    except (TypeError, ValueError):
        return None


def _summarize_durations(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {
            "measurement_kind": "unavailable",
            "measured_count": 0,
            "total_ms": None,
            "avg_ms": None,
            "max_ms": None,
        }
    total = sum(values)
    count = len(values)
    return {
        "measurement_kind": "measured",
        "measured_count": count,
        "total_ms": round(total, 3),
        "avg_ms": round(total / count, 3),
        "max_ms": round(max(values), 3),
    }


@dataclass(frozen=True)
class ExecutionMetricsReport:
    """Measured execution & policy observability report for a single execution."""

    execution_id: str
    found: bool
    status: str = "not_found"
    model_requests: Dict[str, Any] = field(default_factory=dict)
    orchestration_steps: Dict[str, Any] = field(default_factory=dict)
    tool_calls: Dict[str, Any] = field(default_factory=dict)
    tool_duration_ms: Dict[str, Any] = field(default_factory=dict)
    verification_duration_ms: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: Dict[str, Any] = field(default_factory=dict)
    policy_decisions: Dict[str, Any] = field(default_factory=dict)
    confirmation_wait_ms: Dict[str, Any] = field(default_factory=dict)
    failures_and_timeouts: Dict[str, Any] = field(default_factory=dict)
    wall_clock_span_ms: Optional[float] = None
    unmeasured_metrics: List[str] = field(default_factory=list)
    persistence_errors: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _extract_execution_metrics(
    record: ExecutionRecord,
    events: List[ExecutionEvent],
    *,
    persistence_errors: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    tool_call_starts = 0
    tool_results_count = 0
    tool_durations: List[float] = []
    by_tool: Dict[str, Dict[str, Any]] = {}

    ver_durations: List[float] = []
    recovery_count = 0
    recovery_by_status: Dict[str, int] = {}

    policy_outcomes: Dict[str, int] = {"allow": 0, "require_confirmation": 0, "deny": 0}
    policy_total = 0
    conf_wait_durations: List[float] = []

    failed_calls = 0
    timed_out_calls = 0
    cancelled_calls = 0
    rejected_calls = 0

    model_req_count: Optional[int] = None
    model_req_duration_ms: Optional[float] = None
    orch_step_count: Optional[int] = None

    for ev in events:
        meta = ev.metadata if isinstance(ev.metadata, dict) else {}
        et = ev.event_type

        if et == LedgerEventType.PLANNING_DECISION.value and (
            meta.get("metrics_snapshot") is True or ev.status == "metrics_snapshot"
        ):
            if isinstance(meta.get("model_request_count"), int) and not isinstance(meta["model_request_count"], bool):
                prev_cnt = model_req_count or 0
                model_req_count = max(prev_cnt, int(meta["model_request_count"]))
            m_dur = _as_nonneg_float(meta.get("model_request_duration_ms"))
            if m_dur is not None:
                model_req_duration_ms = max(model_req_duration_ms or 0.0, m_dur)
            if isinstance(meta.get("orchestration_step_count"), int) and not isinstance(
                meta["orchestration_step_count"], bool
            ):
                orch_step_count = max(orch_step_count or 0, int(meta["orchestration_step_count"]))

        elif et == LedgerEventType.TOOL_EXECUTION_START.value:
            tool_call_starts += 1

        elif et == LedgerEventType.TOOL_EXECUTION_RESULT.value:
            tool_results_count += 1
            st = str(ev.status or "").lower()
            t_name = str(ev.tool_id or "unknown")
            t_entry = by_tool.setdefault(
                t_name,
                {"calls": 0, "failed": 0, "timed_out": 0, "cancelled": 0, "rejected": 0, "durations_ms": []},
            )
            t_entry["calls"] += 1
            if st == "timed_out":
                timed_out_calls += 1
                t_entry["timed_out"] += 1
            elif st == "cancelled":
                cancelled_calls += 1
                t_entry["cancelled"] += 1
            elif st == "rejected":
                rejected_calls += 1
                t_entry["rejected"] += 1
            elif st in ("failed", "error"):
                failed_calls += 1
                t_entry["failed"] += 1

            dur = _as_nonneg_float(meta.get("duration_ms"))
            if dur is not None:
                tool_durations.append(dur)
                t_entry["durations_ms"].append(dur)

            v_dur_res = _as_nonneg_float(meta.get("verification_duration_ms"))
            if v_dur_res is not None:
                ver_durations.append(v_dur_res)

        elif et == LedgerEventType.VERIFICATION_RECORDED.value:
            v_dur = _as_nonneg_float(meta.get("duration_ms"))
            if v_dur is not None and len(ver_durations) == 0:
                # Only use VERIFICATION_RECORDED if not already recorded on TOOL_EXECUTION_RESULT
                ver_durations.append(v_dur)
            elif v_dur is not None and not any(
                ev2.event_type == LedgerEventType.TOOL_EXECUTION_RESULT.value
                and ev2.correlation_id == ev.correlation_id
                and _as_nonneg_float((ev2.metadata or {}).get("verification_duration_ms")) is not None
                for ev2 in events
            ):
                ver_durations.append(v_dur)

        elif et == LedgerEventType.RECOVERY_ATTEMPT.value:
            recovery_count += 1
            r_st = str(ev.status or "recorded")
            recovery_by_status[r_st] = recovery_by_status.get(r_st, 0) + 1

        elif et == LedgerEventType.AUTHORIZATION_DECISION.value:
            policy_total += 1
            raw_out = str(meta.get("policy_decision") or ev.status or "").strip().lower()
            if raw_out in ("allow", "allowed"):
                policy_outcomes["allow"] += 1
            elif raw_out in ("require_confirmation", "requires_confirmation", "confirmation_required"):
                policy_outcomes["require_confirmation"] += 1
            elif raw_out in ("deny", "denied", "rejected"):
                policy_outcomes["deny"] += 1
            else:
                policy_outcomes[raw_out or "other"] = policy_outcomes.get(raw_out or "other", 0) + 1

            wait_ms = _as_nonneg_float(meta.get("confirmation_wait_ms"))
            if wait_ms is not None:
                conf_wait_durations.append(wait_ms)

    tool_summary_by_tool: Dict[str, Any] = {}
    for t_name, t_data in sorted(by_tool.items()):
        d_summary = _summarize_durations(t_data["durations_ms"])
        tool_summary_by_tool[t_name] = {
            "calls": t_data["calls"],
            "failed": t_data["failed"],
            "timed_out": t_data["timed_out"],
            "cancelled": t_data["cancelled"],
            "rejected": t_data["rejected"],
            "duration_ms": d_summary,
        }

    eff_tool_calls = max(tool_call_starts, tool_results_count)
    wall_span_ms: Optional[float] = None
    if record.completed_at is not None and record.created_at is not None:
        wall_span_ms = round(max(0.0, (float(record.completed_at) - float(record.created_at)) * 1000.0), 3)

    unmeasured: List[str] = ["token_usage"]  # Never fabricate token usage when provider metrics are absent
    if model_req_count is None:
        unmeasured.append("model_request_count")
    if model_req_duration_ms is None:
        unmeasured.append("model_request_duration_ms")
    if orch_step_count is None:
        unmeasured.append("orchestration_step_count")
    if eff_tool_calls > 0 and not tool_durations:
        unmeasured.append("tool_execution_duration_ms")
    if not ver_durations:
        unmeasured.append("verification_duration_ms")

    report = ExecutionMetricsReport(
        execution_id=record.execution_id,
        found=True,
        status=record.status,
        model_requests={
            "measurement_kind": "measured" if model_req_count is not None else "unavailable",
            "count": model_req_count,
            "total_duration_ms": model_req_duration_ms,
            "avg_duration_ms": (
                round(model_req_duration_ms / model_req_count, 3)
                if (model_req_count and model_req_duration_ms is not None)
                else None
            ),
            "token_usage": None,
        },
        orchestration_steps={
            "measurement_kind": "measured" if orch_step_count is not None else "unavailable",
            "count": orch_step_count,
        },
        tool_calls={
            "measurement_kind": "measured",
            "dispatched_starts": tool_call_starts,
            "completed_results": tool_results_count,
            "total_calls": eff_tool_calls,
            "by_tool": tool_summary_by_tool,
        },
        tool_duration_ms=_summarize_durations(tool_durations),
        verification_duration_ms=_summarize_durations(ver_durations),
        recovery_attempts={
            "measurement_kind": "measured",
            "count": recovery_count,
            "by_status": recovery_by_status,
        },
        policy_decisions={
            "measurement_kind": "measured",
            "total": policy_total,
            "by_outcome": policy_outcomes,
        },
        confirmation_wait_ms=_summarize_durations(conf_wait_durations),
        failures_and_timeouts={
            "failed_calls": failed_calls,
            "timed_out_calls": timed_out_calls,
            "cancelled_calls": cancelled_calls,
            "rejected_calls": rejected_calls,
            "total_unsuccessful": failed_calls + timed_out_calls + cancelled_calls + rejected_calls,
        },
        wall_clock_span_ms=wall_span_ms,
        unmeasured_metrics=unmeasured,
        persistence_errors=list(persistence_errors or []),
    )
    return report.to_dict()


def collect_execution_metrics(
    ledger: CentralExecutionLedger,
    execution_id: str,
) -> Dict[str, Any]:
    """Collect measured execution & policy observability metrics for a single execution."""
    exec_id = str(execution_id or "")
    perrs = [
        err
        for err in (ledger.get_persistence_errors() if hasattr(ledger, "get_persistence_errors") else [])
        if err.get("execution_id") in (exec_id, None)
    ]
    rec = ledger.get_execution(exec_id) if exec_id else None
    if rec is None:
        return ExecutionMetricsReport(
            execution_id=exec_id,
            found=False,
            persistence_errors=perrs,
        ).to_dict()
    events = ledger.get_events(rec.execution_id)
    perrs = [
        err
        for err in (ledger.get_persistence_errors() if hasattr(ledger, "get_persistence_errors") else [])
        if err.get("execution_id") in (rec.execution_id, None)
    ]
    return _extract_execution_metrics(rec, events, persistence_errors=perrs)


def collect_fleet_observability(
    ledger: CentralExecutionLedger,
    *,
    limit: int = 20,
) -> Dict[str, Any]:
    """Aggregate measured execution and policy metrics across recent executions."""
    records = ledger.list_executions(limit=limit)
    perrs = list(ledger.get_persistence_errors()) if hasattr(ledger, "get_persistence_errors") else []

    per_exec: List[Dict[str, Any]] = []
    model_req_total = 0
    model_req_dur_total = 0.0
    measured_model_execs = 0
    orch_step_total = 0
    measured_step_execs = 0
    tool_call_total = 0
    tool_durations_all: List[float] = []
    ver_durations_all: List[float] = []
    recovery_total = 0
    policy_outcomes_all: Dict[str, int] = {"allow": 0, "require_confirmation": 0, "deny": 0}
    policy_total = 0
    conf_waits_all: List[float] = []
    failures_total = {"failed_calls": 0, "timed_out_calls": 0, "cancelled_calls": 0, "rejected_calls": 0}

    for rec in records:
        events = ledger.get_events(rec.execution_id)
        rep = _extract_execution_metrics(rec, events)
        per_exec.append(rep)
        mr = rep["model_requests"]
        if mr["measurement_kind"] == "measured" and mr["count"] is not None:
            measured_model_execs += 1
            model_req_total += int(mr["count"])
            if mr["total_duration_ms"] is not None:
                model_req_dur_total += float(mr["total_duration_ms"])
        os_blk = rep["orchestration_steps"]
        if os_blk["measurement_kind"] == "measured" and os_blk["count"] is not None:
            measured_step_execs += 1
            orch_step_total += int(os_blk["count"])
        tool_call_total += int(rep["tool_calls"]["total_calls"])
        for tinfo in (rep["tool_calls"].get("by_tool") or {}).values():
            td = tinfo.get("duration_ms") or {}
            if td.get("total_ms") is not None and td.get("measured_count"):
                # Reconstruct exact sum weight using avg? Preserve exact totals across events directly:
                pass
        for ev in events:
            meta = ev.metadata if isinstance(ev.metadata, dict) else {}
            if ev.event_type == LedgerEventType.TOOL_EXECUTION_RESULT.value:
                d = _as_nonneg_float(meta.get("duration_ms"))
                if d is not None:
                    tool_durations_all.append(d)
                vd = _as_nonneg_float(meta.get("verification_duration_ms"))
                if vd is not None:
                    ver_durations_all.append(vd)
            elif ev.event_type == LedgerEventType.AUTHORIZATION_DECISION.value:
                w = _as_nonneg_float(meta.get("confirmation_wait_ms"))
                if w is not None:
                    conf_waits_all.append(w)

        recovery_total += int(rep["recovery_attempts"]["count"])
        pd_blk = rep["policy_decisions"]
        policy_total += int(pd_blk["total"])
        for k, v in (pd_blk.get("by_outcome") or {}).items():
            policy_outcomes_all[k] = policy_outcomes_all.get(k, 0) + int(v)
        ft = rep["failures_and_timeouts"]
        for k in failures_total:
            failures_total[k] += int(ft.get(k, 0))

    perrs = list(ledger.get_persistence_errors()) if hasattr(ledger, "get_persistence_errors") else perrs
    return {
        "execution_count": len(records),
        "db_path": str(getattr(ledger, "db_path", "")),
        "model_requests": {
            "measurement_kind": "measured" if measured_model_execs > 0 else "unavailable",
            "measured_executions": measured_model_execs,
            "count": model_req_total if measured_model_execs > 0 else None,
            "total_duration_ms": round(model_req_dur_total, 3) if measured_model_execs > 0 else None,
            "token_usage": None,
        },
        "orchestration_steps": {
            "measurement_kind": "measured" if measured_step_execs > 0 else "unavailable",
            "measured_executions": measured_step_execs,
            "count": orch_step_total if measured_step_execs > 0 else None,
        },
        "tool_calls_total": tool_call_total,
        "tool_duration_ms": _summarize_durations(tool_durations_all),
        "verification_duration_ms": _summarize_durations(ver_durations_all),
        "recovery_attempts_total": recovery_total,
        "policy_decisions": {
            "total": policy_total,
            "by_outcome": policy_outcomes_all,
        },
        "confirmation_wait_ms": _summarize_durations(conf_waits_all),
        "failures_and_timeouts": {
            **failures_total,
            "total_unsuccessful": sum(failures_total.values()),
        },
        "persistence_errors": perrs,
        "executions": [
            {
                "execution_id": r["execution_id"],
                "status": r["status"],
                "tool_calls": r["tool_calls"]["total_calls"],
                "model_requests": r["model_requests"]["count"],
                "orchestration_steps": r["orchestration_steps"]["count"],
                "policy_decisions": r["policy_decisions"]["total"],
                "recovery_attempts": r["recovery_attempts"]["count"],
                "unsuccessful_calls": r["failures_and_timeouts"]["total_unsuccessful"],
            }
            for r in per_exec
        ],
    }


def format_observability_summary(summary: Mapping[str, Any]) -> str:
    """Format a single-execution or fleet observability dict into readable terminal output."""
    if "execution_count" in summary:
        pd = summary.get("policy_decisions") or {}
        po = pd.get("by_outcome") or {}
        mr = summary.get("model_requests") or {}
        os_blk = summary.get("orchestration_steps") or {}
        td = summary.get("tool_duration_ms") or {}
        vd = summary.get("verification_duration_ms") or {}
        cw = summary.get("confirmation_wait_ms") or {}
        ft = summary.get("failures_and_timeouts") or {}
        lines = [
            f"Execution & Policy Observability ({summary.get('db_path')}) — {summary.get('execution_count', 0)} execution(s):",
            f"  model_requests:       count={mr.get('count')} total_ms={mr.get('total_duration_ms')} ({mr.get('measurement_kind')})",
            f"  orchestration_steps:  count={os_blk.get('count')} ({os_blk.get('measurement_kind')})",
            f"  tool_calls:           total={summary.get('tool_calls_total', 0)} total_ms={td.get('total_ms')} avg_ms={td.get('avg_ms')} max_ms={td.get('max_ms')} ({td.get('measurement_kind')})",
            f"  verification:         checks={vd.get('measured_count', 0)} total_ms={vd.get('total_ms')} ({vd.get('measurement_kind')})",
            f"  recovery_attempts:    total={summary.get('recovery_attempts_total', 0)}",
            f"  policy_decisions:     total={pd.get('total', 0)} allow={po.get('allow', 0)} confirm={po.get('require_confirmation', 0)} deny={po.get('deny', 0)}",
            f"  confirmation_wait:    waits={cw.get('measured_count', 0)} total_ms={cw.get('total_ms')} avg_ms={cw.get('avg_ms')}",
            f"  failed_or_timed_out:  failed={ft.get('failed_calls', 0)} timed_out={ft.get('timed_out_calls', 0)} cancelled={ft.get('cancelled_calls', 0)} rejected={ft.get('rejected_calls', 0)}",
        ]
        return "\n".join(lines)

    if not summary.get("found", True):
        return f"Execution Metrics: {summary.get('execution_id')} (NOT FOUND)"

    mr = summary.get("model_requests") or {}
    os_blk = summary.get("orchestration_steps") or {}
    tc = summary.get("tool_calls") or {}
    td = summary.get("tool_duration_ms") or {}
    vd = summary.get("verification_duration_ms") or {}
    pd = summary.get("policy_decisions") or {}
    po = pd.get("by_outcome") or {}
    cw = summary.get("confirmation_wait_ms") or {}
    ft = summary.get("failures_and_timeouts") or {}
    lines = [
        f"Execution Metrics: {summary.get('execution_id')} (status={summary.get('status')})",
        f"  model_requests:       count={mr.get('count')} total_ms={mr.get('total_duration_ms')} ({mr.get('measurement_kind')})",
        f"  orchestration_steps:  count={os_blk.get('count')} ({os_blk.get('measurement_kind')})",
        f"  tool_calls:           total={tc.get('total_calls', 0)} total_ms={td.get('total_ms')} avg_ms={td.get('avg_ms')} ({td.get('measurement_kind')})",
        f"  verification:         checks={vd.get('measured_count', 0)} total_ms={vd.get('total_ms')} ({vd.get('measurement_kind')})",
        f"  recovery_attempts:    count={(summary.get('recovery_attempts') or {}).get('count', 0)}",
        f"  policy_decisions:     total={pd.get('total', 0)} allow={po.get('allow', 0)} confirm={po.get('require_confirmation', 0)} deny={po.get('deny', 0)}",
        f"  confirmation_wait:    waits={cw.get('measured_count', 0)} total_ms={cw.get('total_ms')}",
        f"  failed_or_timed_out:  failed={ft.get('failed_calls', 0)} timed_out={ft.get('timed_out_calls', 0)} cancelled={ft.get('cancelled_calls', 0)} rejected={ft.get('rejected_calls', 0)}",
        f"  unmeasured_metrics:   {summary.get('unmeasured_metrics', [])}",
    ]
    return "\n".join(lines)


__all__ = [
    "ExecutionMetricsReport",
    "collect_execution_metrics",
    "collect_fleet_observability",
    "format_observability_summary",
]
