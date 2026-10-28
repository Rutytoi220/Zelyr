"""Zelyr Evaluation & Benchmark CLI Runner."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import List, Optional

from zelyr.config import get_config
from zelyr.eval.harness import EvalHarness
from zelyr.eval.metrics import EvalStatus, TaskResult, export_json_report, format_summary_table
from zelyr.eval.scorecard import export_scorecard, generate_benchmark_scorecard
from zelyr.eval.task import EvalTask
from zelyr.eval.tasks.deterministic_p1 import get_deterministic_p1_tasks
from zelyr.eval.tasks.model_tasks_p3 import get_model_tasks_p3, task_live_web_research_synthesis
from zelyr.eval.tasks.self_repair_l3 import get_self_repair_l3_tasks
from zelyr.eval.tasks.synthetic_workflows_p2 import get_synthetic_workflows_p2_tasks


def get_all_available_tasks(model: str = "qwen3:8b", base_url: str = "http://127.0.0.1:11434") -> List[EvalTask]:
    """Retrieve all benchmark tasks across Levels 1, 2, and 3."""
    tasks: List[EvalTask] = []
    tasks.extend(get_deterministic_p1_tasks())
    tasks.extend(get_model_tasks_p3(model=model, base_url=base_url))
    tasks.extend(get_synthetic_workflows_p2_tasks())
    tasks.extend(get_self_repair_l3_tasks())
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
        help="Evaluation level to execute (1: deterministic, 2: live model, 3: synthetic workflows, 'all': all levels). Defaults to 1.",
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
        "-q", "--quiet",
        action="store_true",
        default=False,
        help="Suppress intermediate execution progress logs.",
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
            status_disp = res.status.value if isinstance(res.status, EvalStatus) else str(res.status)
            dur_disp = f"{res.duration_seconds:.2f}s"
            sys.stdout.write(f"{status_disp} ({dur_disp})\n")
            if res.status != EvalStatus.PASS and res.message:
                sys.stdout.write(f"    Notice: {res.message}\n")
            sys.stdout.flush()

    return results


def run_benchmark_cli(args_list: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(args_list)

    cfg = get_config()
    model = args.model or getattr(cfg, "ollama_model", "qwen3:8b")
    base_url = args.base_url or getattr(cfg, "ollama_base_url", "http://127.0.0.1:11434")

    all_tasks = get_all_available_tasks(model=model, base_url=base_url)

    if args.list_tasks:
        print("Available Zelyr Evaluation Benchmark Tasks:")
        for lvl in (1, 2, 3):
            lvl_tasks = [t for t in all_tasks if t.level == lvl]
            print(f"\n  Level {lvl} ({len(lvl_tasks)} tasks):")
            for t in lvl_tasks:
                print(f"    - {t.task_id:<36} {t.description}")
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
        elif lvl_str in ("all", "*"):
            pass
        else:
            print(f"Error: Unknown level '{args.level}'. Must be 1, 2, 3, or 'all'.", file=sys.stderr)
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
