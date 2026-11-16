"""Aggregate reporting and Markdown summary generator for distributed evaluations."""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
from typing import Any, Dict, List, Optional

from zelyr.engine.error_classifier import _redact_secrets
from zelyr.eval.distributed import DistributedTaskResult, DistributedTaskStatus, _sanitize_artifact_segment, _scrub_url_userinfo


def _compute_p95(values: List[float]) -> Optional[float]:
    """Compute P95 only when sample size supports it (n >= 20)."""
    if len(values) < 20:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return round(ordered[idx], 4)


def compute_model_metrics(results: List[DistributedTaskResult]) -> Dict[str, Any]:
    """Compute non-conflating aggregate metrics for a slice of task results."""
    counts: Dict[str, int] = {s.value: 0 for s in DistributedTaskStatus}
    for r in results:
        key = r.status.value if isinstance(r.status, DistributedTaskStatus) else str(r.status)
        counts[key] = counts.get(key, 0) + 1

    total = len(results)
    eligible = [
        r for r in results
        if r.status not in (DistributedTaskStatus.UNSUPPORTED, DistributedTaskStatus.UNAVAILABLE)
    ]
    executed_durations = [
        r.duration_seconds for r in results
        if r.status not in (DistributedTaskStatus.UNSUPPORTED, DistributedTaskStatus.UNAVAILABLE)
        and r.duration_seconds > 0
    ]

    # Tool-call success rate: records in ('success', 'verified') over total dispatched ledger records
    total_tool_records = 0
    accepted_tool_records = 0
    recovery_tasks = 0
    recovery_passed = 0
    prompt_eval_vals: List[int] = []
    eval_count_vals: List[int] = []
    estimated_token_vals: List[int] = []
    ps_sizes: List[int] = []
    ps_vram_sizes: List[int] = []
    queue_wait_vals: List[float] = []

    for r in results:
        recs = list(r.ledger_records or [])
        dispatched = [
            rec for rec in recs
            if rec.get("state") in ("success", "verified", "failed", "partially_completed", "observed", "unverified")
        ]
        total_tool_records += len(dispatched)
        accepted_tool_records += sum(1 for rec in dispatched if rec.get("state") in ("success", "verified"))

        has_prior_failure = any(
            rec.get("state") == "failed" or rec.get("verification_failed") is True
            for rec in recs
        )
        if r.task_id in ("S3", "S8", "S10", "COMPAT_FAILURE_RECOVERY") or (has_prior_failure and r.task_id not in ("S7", "COMPAT_HONEST_REPORT")):
            if r.status not in (DistributedTaskStatus.UNSUPPORTED, DistributedTaskStatus.UNAVAILABLE):
                recovery_tasks += 1
                if r.status == DistributedTaskStatus.PASS:
                    recovery_passed += 1

        rm = r.resource_measurements or {}
        if rm.get("prompt_eval_count_max") is not None:
            prompt_eval_vals.append(int(rm["prompt_eval_count_max"]))
        if rm.get("eval_count_sum") is not None:
            eval_count_vals.append(int(rm["eval_count_sum"]))
        est_obj = getattr(r, "estimated_token_usage", None) or rm.get("estimated_token_usage")
        if isinstance(est_obj, dict) and est_obj.get("estimated_tokens_approx") is not None:
            estimated_token_vals.append(int(est_obj["estimated_tokens_approx"]))
        if (
            rm.get("queued_wait_seconds") is not None
            and r.status not in (DistributedTaskStatus.UNSUPPORTED, DistributedTaskStatus.UNAVAILABLE)
        ):
            queue_wait_vals.append(float(rm["queued_wait_seconds"]))
        for ps_key in ("ps_before", "ps_after"):
            for entry in rm.get(ps_key) or []:
                if entry.get("name") == r.model:
                    if entry.get("size") is not None:
                        ps_sizes.append(int(entry["size"]))
                    if entry.get("size_vram") is not None:
                        ps_vram_sizes.append(int(entry["size_vram"]))

    avg_dur = round(statistics.mean(executed_durations), 3) if executed_durations else None
    med_dur = round(statistics.median(executed_durations), 3) if executed_durations else None
    p95_dur = _compute_p95(executed_durations)
    avg_wait = round(statistics.mean(queue_wait_vals), 3) if queue_wait_vals else None
    max_wait = round(max(queue_wait_vals), 3) if queue_wait_vals else None

    measured_usage = (
        {
            "prompt_eval_count_max": max(prompt_eval_vals) if prompt_eval_vals else None,
            "eval_count_sum": sum(eval_count_vals) if eval_count_vals else None,
        }
        if (prompt_eval_vals or eval_count_vals)
        else None
    )
    estimated_usage = (
        {
            "estimated_tokens_sum": sum(estimated_token_vals),
            "source": "character_heuristic",
        }
        if estimated_token_vals
        else None
    )

    return {
        "total_tasks": total,
        "eligible_tasks": len(eligible),
        "status_counts": counts,
        "task_success_rate_eligible": round(counts["PASS"] / len(eligible), 4) if eligible else None,
        "task_success_rate_total": round(counts["PASS"] / total, 4) if total else None,
        "tool_call_success_rate": round(accepted_tool_records / total_tool_records, 4) if total_tool_records else None,
        "tool_calls_accepted": accepted_tool_records,
        "tool_calls_dispatched": total_tool_records,
        "recovery_rate": round(recovery_passed / recovery_tasks, 4) if recovery_tasks else None,
        "recovery_passed": recovery_passed,
        "recovery_attempted": recovery_tasks,
        "false_success_rate": round(counts["FALSE_SUCCESS"] / len(eligible), 4) if eligible else None,
        "false_success_count": counts["FALSE_SUCCESS"],
        "average_duration_s": avg_dur,
        "median_duration_s": med_dur,
        "p95_duration_s": p95_dur,
        "average_queued_wait_s": avg_wait,
        "max_queued_wait_s": max_wait,
        "context_usage_prompt_tokens_max": max(prompt_eval_vals) if prompt_eval_vals else None,
        "measured_token_usage": measured_usage,
        "estimated_token_usage": estimated_usage,
        "memory_usage_max_size_bytes": max(ps_sizes) if ps_sizes else None,
        "memory_usage_max_vram_bytes": max(ps_vram_sizes) if ps_vram_sizes else None,
        "concurrency_modes_observed": sorted({r.concurrency_mode for r in results}),
    }


def generate_distributed_reports(
    run_output: Dict[str, Any],
    output_dir: Path,
    *,
    title: str = "Zelyr Distributed Multi-Model Evaluation Report",
) -> Dict[str, Path]:
    """Export distributed_report.json, distributed_summary.md, per_model JSONs, and snapshots."""
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    per_model_dir = out_dir / "per_model"
    per_model_dir.mkdir(parents=True, exist_ok=True)

    results: List[DistributedTaskResult] = list(run_output.get("results") or [])
    scheduler_stats = dict(run_output.get("scheduler_stats") or {})
    snapshots = dict(run_output.get("snapshots") or {})
    cleanup = list(run_output.get("cleanup") or [])

    grouped: Dict[str, List[DistributedTaskResult]] = {}
    for r in results:
        key = f"{r.worker_id}::{r.model}"
        grouped.setdefault(key, []).append(r)

    per_model_summary: Dict[str, Any] = {}
    per_model_files: List[Path] = []
    for key, group in sorted(grouped.items()):
        wid, mod = key.split("::", 1)
        metrics = compute_model_metrics(group)
        metrics["worker_id"] = wid
        metrics["model"] = mod
        metrics["model_digest"] = group[0].model_digest
        metrics["endpoint"] = _scrub_url_userinfo(group[0].endpoint)
        metrics["model_settings"] = group[0].model_settings
        per_model_summary[key] = metrics

        safe_wid = _sanitize_artifact_segment(wid, "worker")
        safe_mod = _sanitize_artifact_segment(mod, "model")
        m_path = per_model_dir / f"{safe_wid}__{safe_mod}.json"
        serialized_group = []
        for r in group:
            rd = r.to_dict()
            rd["endpoint"] = _scrub_url_userinfo(rd.get("endpoint", ""))
            if rd.get("error_detail"):
                rd["error_detail"] = _redact_secrets(str(rd["error_detail"]))
            serialized_group.append(rd)
        m_path.write_text(
            json.dumps(
                {
                    "report_version": "1.4",
                    "benchmark_version": "1.4",
                    "summary": metrics,
                    "results": serialized_group,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        per_model_files.append(m_path)

    overall_metrics = compute_model_metrics(results)
    overall_metrics["scheduler_stats"] = scheduler_stats
    cleanup_error_count = sum(len(rec.get("errors") or []) for rec in cleanup)
    cleanup_ok = all(rec.get("cleanup_ok", not rec.get("errors")) for rec in cleanup) if cleanup else True
    if "cleanup_ok" in run_output:
        cleanup_ok = bool(run_output["cleanup_ok"]) and cleanup_ok
    overall_metrics["cleanup_ok"] = cleanup_ok
    overall_metrics["cleanup_error_count"] = cleanup_error_count

    serialized_results = []
    for r in results:
        rd = r.to_dict()
        rd["endpoint"] = _scrub_url_userinfo(rd.get("endpoint", ""))
        if rd.get("error_detail"):
            rd["error_detail"] = _redact_secrets(str(rd["error_detail"]))
        serialized_results.append(rd)

    report_doc = {
        "report_version": "1.4",
        "benchmark_version": "1.4",
        "title": title,
        "overall": overall_metrics,
        "per_model": per_model_summary,
        "results": serialized_results,
        "cleanup": cleanup,
    }
    report_json_path = out_dir / "distributed_report.json"
    report_json_path.write_text(json.dumps(report_doc, indent=2), encoding="utf-8")

    snapshots_path = out_dir / "worker_snapshots.json"
    snapshots_path.write_text(json.dumps(snapshots, indent=2), encoding="utf-8")

    md_lines = [
        f"# {title}",
        "",
        f"- **Concurrency forms observed:** {', '.join(scheduler_stats.get('observed_concurrency_forms') or ['none'])}",
        f"- **Peak simultaneous active workers:** {scheduler_stats.get('peak_simultaneous_workers', 0)}",
        f"- **Peak concurrency by worker:** `{json.dumps(scheduler_stats.get('peak_concurrency_by_worker') or {})}`",
        f"- **Cleanup status:** {'PASS' if cleanup_ok else 'FAIL'} (errors={cleanup_error_count})",
        "",
        "## Per-Model Summary",
        "",
        "| Worker | Model | Num Ctx | Pass | Fail | Timeout | Unver | Unsup | Unavail | Err | Canc | FalseSucc | Success (Elig) | Tool Succ | Recovery | Avg Exec (s) | Med Exec (s) | P95 Exec (s) | Avg Queue (s) | Max Prompt Tok | Max PS Size (MB) | Max PS VRAM (MB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for key, m in per_model_summary.items():
        sc = m["status_counts"]
        succ = f"{m['task_success_rate_eligible'] * 100:.1f}%" if m["task_success_rate_eligible"] is not None else "N/A"
        tsucc = f"{m['tool_call_success_rate'] * 100:.1f}%" if m["tool_call_success_rate"] is not None else "N/A"
        rec = f"{m['recovery_rate'] * 100:.1f}%" if m["recovery_rate"] is not None else "N/A"
        avg_s = f"{m['average_duration_s']:.2f}" if m["average_duration_s"] is not None else "N/A"
        med_s = f"{m['median_duration_s']:.2f}" if m["median_duration_s"] is not None else "N/A"
        p95_s = f"{m['p95_duration_s']:.2f}" if m["p95_duration_s"] is not None else "N/A(n<20)"
        qwait_s = f"{m['average_queued_wait_s']:.2f}" if m["average_queued_wait_s"] is not None else "N/A"
        ptok = str(m["context_usage_prompt_tokens_max"]) if m["context_usage_prompt_tokens_max"] is not None else "N/A"
        ps_mb = f"{m['memory_usage_max_size_bytes'] / (1024**2):.0f}" if m["memory_usage_max_size_bytes"] is not None else "N/A"
        vram_mb = f"{m['memory_usage_max_vram_bytes'] / (1024**2):.0f}" if m["memory_usage_max_vram_bytes"] is not None else "N/A"
        ctx_cfg = m.get("model_settings", {}).get("num_ctx", "-")
        md_lines.append(
            f"| {m['worker_id']} | `{m['model']}` | {ctx_cfg} | {sc['PASS']} | {sc['FAIL']} | {sc['TIMEOUT']} | "
            f"{sc.get('UNVERIFIED', 0)} | {sc['UNSUPPORTED']} | {sc['UNAVAILABLE']} | {sc['ERROR']} | {sc.get('CANCELLED', 0)} | {sc['FALSE_SUCCESS']} | "
            f"{succ} | {tsucc} | {rec} | {avg_s} | {med_s} | {p95_s} | {qwait_s} | {ptok} | {ps_mb} | {vram_mb} |"
        )

    md_lines.extend([
        "",
        "## Individual Task Results",
        "",
        "| Worker | Model | Task | Status | Exit | Duration (s) | Tools | Error / Detail |",
        "|---|---|---|---|---:|---:|---:|---|",
    ])
    for r in results:
        det = _redact_secrets((r.error_detail or "").replace("\n", " ").replace("|", "/"))[:90]
        md_lines.append(
            f"| {r.worker_id} | `{r.model}` | {r.task_id} | **{r.status.value}** | {r.exit_code} | "
            f"{r.duration_seconds:.2f} | {len(r.tool_calls)} | {det} |"
        )

    summary_md_path = out_dir / "distributed_summary.md"
    summary_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "distributed_report_json": report_json_path,
        "distributed_summary_md": summary_md_path,
        "worker_snapshots_json": snapshots_path,
        "per_model_dir": per_model_dir,
    }
