"""Zelyr Evaluation & Benchmark CLI Runner."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from zelyr.config import get_config
from zelyr.eval.harness import EvalHarness
from zelyr.eval.matrix import (
    ModelMatrixEvaluator,
    format_matrix_table,
    render_matrix_scorecard_markdown,
)
from zelyr.eval.metrics import EvalStatus, TaskResult, export_json_report, format_summary_table
from zelyr.eval.scorecard import export_scorecard, generate_benchmark_scorecard
from zelyr.eval.task import EvalTask
from zelyr.eval.tasks.deterministic_p1 import get_deterministic_p1_tasks
from zelyr.eval.tasks.model_tasks_p3 import get_model_tasks_p3, task_live_web_research_synthesis
from zelyr.eval.tasks.model_tasks_p4 import get_model_tasks_p4
from zelyr.eval.tasks.model_tasks_p5 import get_model_tasks_p5
from zelyr.eval.tasks.self_repair_l3 import get_self_repair_l3_tasks
from zelyr.eval.tasks.synthetic_workflows_p2 import get_synthetic_workflows_p2_tasks


def get_all_available_tasks(model: str = "qwen3:8b", base_url: str = "http://127.0.0.1:11434") -> List[EvalTask]:
    """Retrieve all benchmark tasks across Levels 1, 2, 3, 4, and 5."""
    tasks: List[EvalTask] = []
    tasks.extend(get_deterministic_p1_tasks())
    tasks.extend(get_model_tasks_p3(model=model, base_url=base_url))
    tasks.extend(get_synthetic_workflows_p2_tasks())
    tasks.extend(get_self_repair_l3_tasks())
    tasks.extend(get_model_tasks_p4())
    tasks.extend(get_model_tasks_p5())
    return tasks


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zelyr.eval",
        description="Zelyr Deterministic & Live Evaluation Benchmark Suite",
    )
    parser.add_argument(
        "-l", "--level",
        type=str,
        default=None,
        help="Evaluation level to execute (1: deterministic, 2: live model, 3: synthetic workflows, 4: multi-turn self-repair, 5: federation & swarm workflows, 'all': all levels). Defaults to 1.",
    )
    parser.add_argument(
        "-t", "--task",
        type=str,
        default=None,
        help="Filter specific task by ID or substring match.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Enable live model execution for Level 2 tasks (connects to Ollama).",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="Model name for live evaluation (defaults to config.ollama_model or 'qwen3:8b').",
    )
    parser.add_argument(
        "-u", "--base-url",
        type=str,
        default=None,
        help="Ollama base URL for live evaluation (defaults to config.ollama_base_url or 'http://127.0.0.1:11434').",
    )
    parser.add_argument(
        "-j", "--json",
        type=str,
        default=None,
        help="Export benchmark metrics report to the specified JSON file path.",
    )
    parser.add_argument(
        "--scorecard",
        nargs="?",
        const="benchmark_scorecard.md",
        default=None,
        help="Generate comprehensive Markdown scorecard (optional file path, defaults to 'benchmark_scorecard.md').",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        default=False,
        help="List all registered benchmark tasks and exit.",
    )
    parser.add_argument(
        "--live-timeout",
        type=float,
        default=60.0,
        help="Per-task timeout in seconds for live model evaluation (defaults to 60.0s).",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        default=False,
        help="Run the model matrix evaluation battery across archetypes or live models.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Force simulated mock archetype evaluation (default in non-interactive/CI).",
    )
    parser.add_argument(
        "--matrix-models",
        type=str,
        default=None,
        help="Comma-separated list of model names to evaluate under --live (e.g. 'qwen2.5:latest,llama3.2:latest').",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        default=False,
        help="Suppress intermediate execution progress logs.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help="Discover configured distributed workers and output/run multi-worker evaluation.",
    )
    parser.add_argument(
        "--workers-config",
        type=str,
        default=None,
        help="Path to JSON worker configuration file for distributed evaluation.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for distributed evaluation reports and transcripts.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["concurrent", "sequential", "comparison"],
        default="concurrent",
        help="Distributed execution mode (concurrent across workers, sequential baseline, or cross-model comparison).",
    )
    parser.add_argument(
        "--suite",
        type=str,
        choices=["all", "s1_s10", "compat", "smoke"],
        default="all",
        help="Distributed evaluation task suite subset to execute.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Optional cap on number of tasks selected for the distributed run.",
    )
    parser.add_argument(
        "--no-unload",
        action="store_true",
        default=False,
        help="Skip unloading eval-loaded models after completion (preloaded models are never unloaded).",
    )
    return parser


async def run_benchmark_async(
    tasks: List[EvalTask],
    use_live_model: bool = False,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    quiet: bool = False,
    live_timeout: Optional[float] = None,
) -> List[TaskResult]:
    harness = EvalHarness()
    results: List[TaskResult] = []

    for idx, task in enumerate(tasks, start=1):
        if not quiet:
            sys.stdout.write(f"[{idx}/{len(tasks)}] Running {task.task_id} (Level {task.level})... ")
            sys.stdout.flush()

        task_is_live = bool(use_live_model)
        res = await harness.run_task(
            task,
            use_live_model=task_is_live,
            model=model,
            base_url=base_url,
            live_timeout=live_timeout,
        )
        results.append(res)

        if not quiet:
            exp = res.artifacts.get("expected_status") if isinstance(res.artifacts, dict) else None
            status_val = res.status.value if isinstance(res.status, EvalStatus) else str(res.status)
            is_passed = (status_val in ("PASS", "PASSED")) or (exp is not None and status_val == str(exp))
            status_disp = "PASS" if is_passed else status_val
            dur_disp = f"{res.duration_seconds:.2f}s"
            sys.stdout.write(f"{status_disp} ({dur_disp})\n")
            if not is_passed and res.message:
                sys.stdout.write(f"    Notice: {res.message}\n")
            sys.stdout.flush()

    return results


def run_benchmark_cli(args_list: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(args_list)

    if args.live and args.mock:
        print("Error: --live and --mock cannot be used together.", file=sys.stderr)
        return 2
    if args.live_timeout <= 0:
        print("Error: --live-timeout must be greater than 0.", file=sys.stderr)
        return 2
    if args.matrix_models and not args.matrix:
        print("Error: --matrix-models requires --matrix.", file=sys.stderr)
        return 2
    if args.matrix and args.mock and args.matrix_models:
        print("Error: --matrix-models requires --live and cannot be used with --mock.", file=sys.stderr)
        return 2

    cfg = get_config()
    model = args.model or getattr(cfg, "ollama_model", "qwen3:8b")
    base_url = args.base_url or getattr(cfg, "ollama_base_url", "http://127.0.0.1:11434")

    all_tasks = get_all_available_tasks(model=model, base_url=base_url)

    if args.list_tasks:
        print("Available Zelyr Evaluation Benchmark Tasks:")
        for lvl in (1, 2, 3, 4, 5):
            lvl_tasks = [t for t in all_tasks if t.level == lvl]
            print(f"\n  Level {lvl} ({len(lvl_tasks)} tasks):")
            for t in lvl_tasks:
                print(f"    - {t.task_id:<45} {t.description}")
        return 0

    if args.distributed:
        from zelyr.eval.distributed import (
            DistributedEvalRunner,
            ModelMetadata,
            WorkerSnapshot,
            build_compatibility_suite,
            build_matrix_assignments,
            build_standard_s_suite,
            load_worker_configs,
        )
        from zelyr.eval.distributed_report import generate_distributed_reports

        workers_cfg_path: Optional[Path] = None
        if args.workers_config:
            workers_cfg_path = Path(args.workers_config)
            if not workers_cfg_path.is_file():
                print(f"Error: Workers config file not found: {args.workers_config}", file=sys.stderr)
                return 2

        try:
            workers = load_worker_configs(config_path=workers_cfg_path)
        except Exception as exc:
            print(f"Error: Invalid workers config: {exc}", file=sys.stderr)
            return 2

        if not workers:
            print("Error: No workers configured for distributed evaluation.", file=sys.stderr)
            return 2

        if args.base_url and not args.workers_config and workers:
            workers[0].endpoint = args.base_url.rstrip("/")

        if args.suite == "s1_s10":
            tasks = build_standard_s_suite()
        elif args.suite == "compat":
            tasks = build_compatibility_suite()
        elif args.suite == "smoke":
            tasks = [t for t in build_standard_s_suite() if t.task_id == "S1"]
        else:
            tasks = build_standard_s_suite() + build_compatibility_suite()

        if args.task:
            target_filter = args.task.strip()
            tasks = [t for t in tasks if target_filter == t.task_id or target_filter in t.task_id]
            if not tasks:
                print(f"Error: No distributed tasks matched identifier '{args.task}'.", file=sys.stderr)
                return 1
        if args.max_tasks is not None and args.max_tasks > 0:
            tasks = tasks[: args.max_tasks]

        offline_mock = bool(args.mock or not args.live)

        async def _mock_prober(w):
            installed = {
                m: ModelMetadata(
                    name=m,
                    digest="mock-digest",
                    size_bytes=1_000_000_000,
                    capabilities=["completion", "tools"],
                    context_length=32768,
                    compatibility={
                        "workflow_compatible": True,
                        "workflow": "react_tools",
                        "overall_status": "available",
                        "capabilities": {"native_tool_calling": {"status": "AVAILABLE"}},
                    },
                )
                for m in (w.model_allowlist or [model])
            }
            return WorkerSnapshot(
                worker_id=w.worker_id,
                endpoint=w.endpoint,
                healthy=True,
                version="mock",
                installed_models=installed,
                compatibility_by_model={m: dict(meta.compatibility) for m, meta in installed.items()},
                resource_info={
                    "host_ram_total_bytes": None,
                    "host_ram_available_bytes": None,
                    "unknown_metrics": ["host_ram_total", "host_ram_available"],
                },
            )

        async def _mock_executor(w, m, task, ws, case):
            t0 = time.time()
            await asyncio.sleep(0.02)
            t1 = time.time()
            return {
                "exit_code": task.expected_exit_code if isinstance(task.expected_exit_code, int) else 0,
                "cli_json_valid": True,
                "cli_json": {"response": "Mock evaluation completed."},
                "execution_ledger": {
                    "has_unresolved_failures": False,
                    "records": [
                        {
                            "call_id": "mock_1",
                            "tool_name": "verify_outcome",
                            "arguments": {"target": task.task_id},
                            "state": "verified",
                            "verified": True,
                        }
                    ],
                },
                "verifier_checks": [{"check": "mock", "target": task.task_id, "success": True}],
                "telemetry": {
                    "turn_metrics": [],
                    "inference_intervals": [
                        {
                            "turn": 1,
                            "model": m,
                            "endpoint": w.endpoint,
                            "start_ts": t0,
                            "first_chunk_ts": round(t0 + 0.005, 6),
                            "end_ts": t1,
                            "duration_seconds": round(t1 - t0, 4),
                            "first_chunk_latency_seconds": 0.005,
                            "chunk_count": 1,
                        }
                    ],
                },
            }

        runner = DistributedEvalRunner(
            workers=workers,
            output_dir=Path(args.out_dir) if args.out_dir else None,
            task_executor=_mock_executor if offline_mock else None,
            worker_prober=_mock_prober if offline_mock else None,
            ps_fetcher=(lambda _w: asyncio.sleep(0, result=[])) if offline_mock else None,
            model_unloader=(
                (lambda w, _pre, _m: asyncio.sleep(0, result={"worker_id": w.worker_id, "unloaded": [], "errors": []}))
                if offline_mock
                else None
            ),
        )
        snaps = asyncio.run(runner.discover_workers())

        models_by_worker = {}
        for w in workers:
            snap = snaps.get(w.worker_id)
            if w.model_allowlist:
                models_by_worker[w.worker_id] = list(w.model_allowlist)
            elif snap and snap.healthy and snap.installed_models:
                models_by_worker[w.worker_id] = sorted(snap.installed_models.keys())
            else:
                models_by_worker[w.worker_id] = [model]

        assignments = build_matrix_assignments(
            workers,
            snaps,
            tasks,
            models_by_worker=models_by_worker,
            mode=args.mode,
        )

        run_out = asyncio.run(runner.run_mode(args.mode, assignments, unload_after=not args.no_unload))
        paths = generate_distributed_reports(run_out, runner.output_dir)
        if not args.quiet:
            print(paths["distributed_summary_md"].read_text(encoding="utf-8"))
        return 0

    if args.matrix:
        evaluator = ModelMatrixEvaluator()
        use_live = bool(args.live and not args.mock)
        if not args.quiet:
            target_mode = "live Ollama models" if use_live else "simulated mock archetypes"
            print(f"\nExecuting Model Matrix Evaluation ({target_mode})...\n")

        if use_live:
            custom_models = [m.strip() for m in args.matrix_models.split(",")] if args.matrix_models else None
            matrix_results = asyncio.run(evaluator.evaluate_live_matrix(models=custom_models, base_url=base_url))
        else:
            matrix_results = asyncio.run(evaluator.evaluate_mock_matrix())

        table_str = format_matrix_table(matrix_results)
        print("\n" + table_str + "\n")

        if args.json:
            out_path = Path(args.json).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(matrix_results, indent=2), encoding="utf-8")
            if not args.quiet:
                print(f"Matrix report exported to: {out_path}\n")

        if args.scorecard:
            sc_path = Path(args.scorecard).resolve()
            sc_path.parent.mkdir(parents=True, exist_ok=True)
            sc_md = render_matrix_scorecard_markdown(matrix_results)
            sc_path.write_text(sc_md, encoding="utf-8")
            if not args.quiet:
                print(f"Matrix Scorecard exported to: {sc_path}\n")

        if not use_live:
            summary = matrix_results.get("summary", {})
            return 0 if summary.get("all_passed") else 1
        else:
            if not matrix_results.get("live_eval", True):
                if not args.quiet:
                    print(f"Notice: {matrix_results.get('reason')}\n")
            return 0

    # Filter tasks
    selected_tasks = all_tasks

    if args.level is not None:
        lvl_str = str(args.level).strip().lower()
        if lvl_str in ("1", "l1"):
            selected_tasks = [t for t in selected_tasks if t.level == 1]
        elif lvl_str in ("2", "l2"):
            selected_tasks = [t for t in selected_tasks if t.level == 2]
        elif lvl_str in ("3", "l3"):
            selected_tasks = [t for t in selected_tasks if t.level == 3]
        elif lvl_str in ("4", "l4"):
            selected_tasks = [t for t in selected_tasks if t.level == 4]
        elif lvl_str in ("5", "l5"):
            selected_tasks = [t for t in selected_tasks if t.level == 5]
        elif lvl_str in ("all", "*"):
            pass
        else:
            print(f"Error: Unknown level '{args.level}'. Must be 1, 2, 3, 4, 5, or 'all'.", file=sys.stderr)
            return 2
    elif args.task is None:
        # Default to Level 1 if neither level nor specific task is requested
        if args.live:
            selected_tasks = [t for t in selected_tasks if t.level == 2]
        else:
            selected_tasks = [t for t in selected_tasks if t.level == 1]

    if args.task is not None:
        target = args.task.strip()
        selected_tasks = [t for t in selected_tasks if target == t.task_id or target in t.task_id]
        if not selected_tasks:
            print(f"Error: No tasks matched identifier '{args.task}'.", file=sys.stderr)
            return 1

    if not selected_tasks:
        print("No tasks matched selection criteria.", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"\nExecuting {len(selected_tasks)} benchmark task(s)...\n")

    results = asyncio.run(
        run_benchmark_async(
            selected_tasks,
            use_live_model=args.live,
            model=model,
            base_url=base_url,
            quiet=args.quiet,
            live_timeout=args.live_timeout,
        )
    )

    # Print summary table
    table_str = format_summary_table(results)
    print("\n" + table_str + "\n")

    # Export JSON if requested
    if args.json:
        out_path = Path(args.json).resolve()
        export_json_report(results, out_path)
        if not args.quiet:
            print(f"Report exported to: {out_path}\n")

    # Export Scorecard if requested
    if args.scorecard:
        sc_path = Path(args.scorecard).resolve()
        scorecard = generate_benchmark_scorecard(results)
        export_scorecard(scorecard, md_path=sc_path)
        if not args.json:
            json_sc_path = sc_path.with_suffix(".json")
            export_scorecard(scorecard, json_path=json_sc_path)
        if not args.quiet:
            print(f"Benchmark Scorecard exported to: {sc_path}\n")

    # Determine exit code:
    # - If a task defines an expected_status (e.g. containment tests), verify matching status.
    # - If no expected_status, UNVERIFIED does NOT fail CI, while hard failures trigger exit code 1.
    task_map = {t.task_id: t for t in selected_tasks}
    has_failure = False
    for r in results:
        t = task_map.get(r.task_id)
        if t and t.expected_status is not None:
            if r.status != t.expected_status:
                has_failure = True
                break
        else:
            if r.status in (
                EvalStatus.FAIL,
                EvalStatus.FALSE_SUCCESS,
                EvalStatus.STEP_LIMIT_EXCEEDED,
                EvalStatus.TIMEOUT,
                EvalStatus.TOOL_ERROR,
            ):
                has_failure = True
                break

    return 1 if has_failure else 0


def main() -> None:
    sys.exit(run_benchmark_cli(sys.argv[1:]))


if __name__ == "__main__":
    main()
