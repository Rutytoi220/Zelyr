"""Model Matrix Empirical Evaluation Subsystem.

Evaluates model independence across simulated archetypes and live local Ollama models.
Verifies grounding manifest injection, degraded tool call ingestion, argument coercion,
false-success suppression, and autonomous error recovery routing.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.engine.execution_ledger import ActionState


@dataclass
class MatrixTaskOutcome:
    """Outcome and behavioral metrics for a single matrix task."""

    task_id: str
    passed: bool
    tool_executed: bool
    false_success_suppressed: bool
    refusal_suppressed: bool
    autonomous_recovery: bool
    duration_seconds: float
    details: str = ""
    combination_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArchetypeMetrics:
    """Aggregated evaluation metrics for a single archetype or live model."""

    name: str
    tasks_total: int
    tasks_passed: int
    compliance_rate: float
    false_success_blocked_rate: float
    false_refusal_rate: float
    autonomous_recovery_rate: float
    duration_seconds: float
    task_outcomes: List[MatrixTaskOutcome] = field(default_factory=list)
    model_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["compliance_rate_pct"] = round(self.compliance_rate * 100, 2)
        d["false_success_blocked_pct"] = round(self.false_success_blocked_rate * 100, 2)
        d["false_refusal_pct"] = round(self.false_refusal_rate * 100, 2)
        d["autonomous_recovery_pct"] = round(self.autonomous_recovery_rate * 100, 2)
        d["task_outcomes"] = [o.to_dict() for o in self.task_outcomes]
        return d


class ModelMatrixEvaluator:
    """Empirical multi-model evaluation harness supporting mock archetypes and live Ollama models."""

    STANDARD_TASKS = [
        "task_matrix_grounding_refusal",
        "task_matrix_degraded_markdown_ingestion",
        "task_matrix_scalar_argument_coercion",
        "task_matrix_false_success_interception",
        "task_matrix_anti_consultant_recovery",
    ]

    ARCHETYPES = [
        "Archetype A (Compliant Function Caller)",
        "Archetype B (Degraded 3B Markdown Caller)",
        "Archetype C (Advisory Chat Prior)",
    ]

    @classmethod
    async def probe_ollama_with_metadata(
        cls,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 1.5,
    ) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
        """Probe local Ollama endpoint and collect per-model metadata from /api/tags without inventing capabilities."""
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(f"{base_url.rstrip('/')}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    meta_map: Dict[str, Dict[str, Any]] = {}
                    for m in data.get("models", []):
                        name = m.get("name", "")
                        if name:
                            meta_map[name] = {
                                "name": name,
                                "digest": m.get("digest", ""),
                                "size": m.get("size"),
                                "details": dict(m.get("details") or {}),
                            }
                    return True, meta_map
        except Exception:
            pass
        return False, {}

    @classmethod
    async def probe_ollama(
        cls,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 1.5,
    ) -> Tuple[bool, List[str]]:
        """Probe local Ollama endpoint to verify connectivity and list available models."""
        ok, meta_map = await cls.probe_ollama_with_metadata(base_url=base_url, timeout=timeout)
        return ok, list(meta_map.keys())

    async def run_mock_task(
        self,
        task_id: str,
        archetype: str,
    ) -> MatrixTaskOutcome:
        """Execute a single matrix task against a simulated model archetype."""
        t0 = time.perf_counter()
        orch = NativeOrchestrator()
        turn_counter = 0
        executed_tools: List[str] = []

        async def mock_execute(name: str, **kwargs) -> Any:
            executed_tools.append(name)
            if task_id in ("task_matrix_false_success_interception", "task_matrix_anti_consultant_recovery"):
                if len(executed_tools) == 1:
                    return "Exit code 1\nResource not found"
            return "Exit code 0\nsuccess"

        # Construct deterministic response generator per archetype and task
        async def mock_stream(payload: dict):
            nonlocal turn_counter
            turn_counter += 1

            if task_id == "task_matrix_grounding_refusal":
                if "Archetype A" in archetype:
                    yield {
                        "choices": [{
                            "delta": {
                                "tool_calls": [{
                                    "id": "call_a_1",
                                    "type": "function",
                                    "function": {"name": "execute_command", "arguments": json.dumps({"command": "ls -la"})},
                                }]
                            }
                        }]
                    }
                elif "Archetype B" in archetype:
                    yield {
                        "choices": [{
                            "delta": {
                                "content": "I will inspect files.\n```json\n{\"tool\": \"execute_command\", \"command\": \"ls -la\"}\n```"
                            }
                        }]
                    }
                else:  # Archetype C: initial cloud refusal, then forced action
                    if turn_counter == 1:
                        yield {"choices": [{"delta": {"content": "I cannot access the local file system as an AI assistant."}}]}
                    else:
                        yield {
                            "choices": [{
                                "delta": {
                                    "tool_calls": [{
                                        "id": "call_c_1",
                                        "type": "function",
                                        "function": {"name": "execute_command", "arguments": json.dumps({"command": "ls -la"})},
                                    }]
                                }
                            }]
                        }

            elif task_id == "task_matrix_degraded_markdown_ingestion":
                if "Archetype A" in archetype:
                    yield {
                        "choices": [{
                            "delta": {
                                "tool_calls": [{
                                    "id": "call_a_2",
                                    "type": "function",
                                    "function": {
                                        "name": "manage_workspace_file",
                                        "arguments": json.dumps({"action": "write", "path": "build.status", "content": "matrix_verified"}),
                                    },
                                }]
                            }
                        }]
                    }
                elif "Archetype B" in archetype:
                    inner = json.dumps({"action": "write", "path": "build.status", "content": "matrix_verified"})
                    tag = f"<tool_call>\n{{\"name\": \"manage_workspace_file\", \"arguments\": {json.dumps(inner)}}}\n</tool_call>"
                    yield {"choices": [{"delta": {"content": tag}}]}
                else:
                    inner = json.dumps({"action": "write", "path": "build.status", "content": "matrix_verified"})
                    yield {"choices": [{"delta": {"content": f"```json\n{{\"tool\": \"manage_workspace_file\", \"arguments\": {json.dumps(inner)}}}\n```"}}]}

            elif task_id == "task_matrix_scalar_argument_coercion":
                if "Archetype A" in archetype:
                    yield {
                        "choices": [{
                            "delta": {
                                "tool_calls": [{
                                    "id": "call_a_3",
                                    "type": "function",
                                    "function": {
                                        "name": "manage_desktop_window",
                                        "arguments": json.dumps({"action": "workspace", "target": "1", "verify": True}),
                                    },
                                }]
                            }
                        }]
                    }
                else:
                    # Stringified boolean and integer scalars
                    yield {
                        "choices": [{
                            "delta": {
                                "tool_calls": [{
                                    "id": "call_b_3",
                                    "type": "function",
                                    "function": {
                                        "name": "manage_desktop_window",
                                        "arguments": json.dumps({"action": "workspace", "target": "1", "verify": "true"}),
                                    },
                                }]
                            }
                        }]
                    }

            elif task_id == "task_matrix_false_success_interception":
                if turn_counter == 1:
                    yield {
                        "choices": [{
                            "delta": {
                                "tool_calls": [{
                                    "id": "call_fail_4",
                                    "type": "function",
                                    "function": {"name": "execute_command", "arguments": json.dumps({"command": "cat /var/log/missing.log"})},
                                }]
                            }
                        }]
                    }
                elif turn_counter == 2:
                    if "Archetype C" in archetype:
                        # Emits false-success claim
                        yield {"choices": [{"delta": {"content": "I have successfully read the file and everything is all set!"}}]}
                    else:
                        yield {"choices": [{"delta": {"content": "The file failed to open with exit code 1."}}]}
                else:
                    yield {"choices": [{"delta": {"content": "The file failed to open with exit code 1."}}]}

            elif task_id == "task_matrix_anti_consultant_recovery":
                if turn_counter == 1:
                    yield {
                        "choices": [{
                            "delta": {
                                "tool_calls": [{
                                    "id": "call_fail_5",
                                    "type": "function",
                                    "function": {"name": "execute_command", "arguments": json.dumps({"command": "cat /etc/cluster.conf"})},
                                }]
                            }
                        }]
                    }
                elif turn_counter == 2:
                    if "Archetype C" in archetype:
                        # Emits consultant deferral
                        yield {"choices": [{"delta": {"content": "I apologize, the config is missing. Please check your directory and let me know."}}]}
                    else:
                        yield {
                            "choices": [{
                                "delta": {
                                    "tool_calls": [{
                                        "id": "call_diag_5",
                                        "type": "function",
                                        "function": {"name": "execute_command", "arguments": json.dumps({"command": "find /etc -name cluster.conf"})},
                                    }]
                                }
                            }]
                        }
                else:
                    yield {
                        "choices": [{
                            "delta": {
                                "tool_calls": [{
                                    "id": "call_diag_5_forced",
                                    "type": "function",
                                    "function": {"name": "execute_command", "arguments": json.dumps({"command": "find /etc -name cluster.conf"})},
                                }]
                            }
                        }]
                    }

        prompt_map = {
            "task_matrix_grounding_refusal": "List all files in the current workspace directory.",
            "task_matrix_degraded_markdown_ingestion": "Write 'matrix_verified' to build.status.",
            "task_matrix_scalar_argument_coercion": "Inspect desktop window workspace 1 and verify.",
            "task_matrix_false_success_interception": "Read non-existent system file /var/log/missing.log.",
            "task_matrix_anti_consultant_recovery": "Read the cluster configuration file /etc/cluster.conf.",
        }

        payload = {
            "model": "matrix-evaluator",
            "messages": [{"role": "user", "content": prompt_map[task_id]}],
        }

        streamed: List[str] = []
        try:
            from unittest.mock import patch
            with patch("zelyr.agents.native_orchestrator.execute_tool", side_effect=mock_execute), \
                 patch.object(orch, "get_active_window_context", return_value=None), \
                 patch.object(orch, "get_active_window", return_value=None), \
                 patch.object(NativeOrchestrator, "get_active_window_sync", return_value=None):
                async for chunk in orch.generate_stream(payload, stream_provider=mock_stream):
                    choices = chunk.get("choices", [])
                    if choices and choices[0].get("delta", {}).get("content"):
                        streamed.append(choices[0]["delta"]["content"])
        except Exception as exc:
            dur = time.perf_counter() - t0
            return MatrixTaskOutcome(
                task_id=task_id,
                passed=False,
                tool_executed=bool(executed_tools),
                false_success_suppressed=False,
                refusal_suppressed=False,
                autonomous_recovery=False,
                duration_seconds=dur,
                details=f"Exception: {exc}",
            )

        dur = time.perf_counter() - t0
        full_text = "".join(streamed)

        # Assertions
        tool_exec = len(executed_tools) > 0
        refusal_suppressed = ("I cannot access" not in full_text and "do not have access" not in full_text)
        false_success_suppressed = ("successfully read the file" not in full_text and "all set" not in full_text)
        auto_recovery = (len(executed_tools) >= 2) if task_id in ("task_matrix_anti_consultant_recovery",) else True

        passed = True
        if task_id == "task_matrix_grounding_refusal":
            passed = tool_exec and refusal_suppressed
        elif task_id == "task_matrix_degraded_markdown_ingestion":
            passed = tool_exec
        elif task_id == "task_matrix_scalar_argument_coercion":
            passed = tool_exec
        elif task_id == "task_matrix_false_success_interception":
            passed = false_success_suppressed
        elif task_id == "task_matrix_anti_consultant_recovery":
            passed = auto_recovery and ("I apologize" not in full_text)

        return MatrixTaskOutcome(
            task_id=task_id,
            passed=passed,
            tool_executed=tool_exec,
            false_success_suppressed=false_success_suppressed,
            refusal_suppressed=refusal_suppressed,
            autonomous_recovery=auto_recovery,
            duration_seconds=dur,
            details="Verified" if passed else "Check failed",
            combination_id=f"{archetype}::{task_id}",
        )

    async def evaluate_mock_matrix(self) -> Dict[str, Any]:
        """Run all 5 standard tasks across the 3 simulated model archetypes."""
        t0 = time.perf_counter()
        results: Dict[str, ArchetypeMetrics] = {}

        for archetype in self.ARCHETYPES:
            outcomes: List[MatrixTaskOutcome] = []
            arch_t0 = time.perf_counter()

            for task_id in self.STANDARD_TASKS:
                outcome = await self.run_mock_task(task_id, archetype)
                outcomes.append(outcome)

            arch_dur = time.perf_counter() - arch_t0
            total_tasks = len(outcomes)
            passed_tasks = sum(1 for o in outcomes if o.passed)
            compliance = sum(1 for o in outcomes if o.tool_executed) / total_tasks
            false_succ_blocked = 1.0 if all(o.false_success_suppressed for o in outcomes) else 0.0
            false_refusals = 0.0 if all(o.refusal_suppressed for o in outcomes) else 1.0
            recovery_rate = sum(1 for o in outcomes if o.autonomous_recovery) / total_tasks

            metrics = ArchetypeMetrics(
                name=archetype,
                tasks_total=total_tasks,
                tasks_passed=passed_tasks,
                compliance_rate=round(compliance, 4),
                false_success_blocked_rate=round(false_succ_blocked, 4),
                false_refusal_rate=round(false_refusals, 4),
                autonomous_recovery_rate=round(recovery_rate, 4),
                duration_seconds=round(arch_dur, 4),
                task_outcomes=outcomes,
                model_metadata={"provider": "mock", "model": archetype},
            )
            results[archetype] = metrics

        total_dur = time.perf_counter() - t0
        return {
            "matrix_version": "1.4",
            "mode": "mock",
            "archetypes": {k: v.to_dict() for k, v in results.items()},
            "duration_seconds": round(total_dur, 4),
            "summary": {
                "total_archetypes": len(results),
                "total_evaluations": len(results) * len(self.STANDARD_TASKS),
                "all_passed": all(m.tasks_passed == m.tasks_total for m in results.values()),
            },
        }

    async def evaluate_live_matrix(
        self,
        models: Optional[List[str]] = None,
        base_url: str = "http://127.0.0.1:11434",
    ) -> Dict[str, Any]:
        """Probe local Ollama daemon and evaluate models against the standard task matrix."""
        reachable, available_models = await self.probe_ollama(base_url)
        if not reachable:
            return {
                "matrix_version": "1.4",
                "live_eval": False,
                "reason": f"Ollama daemon not reachable on {base_url}",
                "models": {},
            }

        _, metadata_by_model = await self.probe_ollama_with_metadata(base_url)
        target_models = models if models else available_models[:3]
        if not target_models:
            return {
                "matrix_version": "1.4",
                "live_eval": False,
                "reason": "Ollama reachable, but no models found",
                "models": {},
            }

        from zelyr.config import ZelyrConfig

        model_results: Dict[str, Any] = {}
        t0 = time.perf_counter()

        for mod in target_models:
            if available_models and mod not in available_models:
                model_results[mod] = {
                    "name": mod,
                    "available": False,
                    "status": "UNAVAILABLE",
                    "reason": f"Model '{mod}' is not installed/available on {base_url}",
                    "tasks_total": len(self.STANDARD_TASKS),
                    "tasks_passed": 0,
                    "compliance_rate": 0.0,
                    "false_success_blocked_rate": 0.0,
                    "false_refusal_rate": 0.0,
                    "autonomous_recovery_rate": 0.0,
                    "compliance_rate_pct": 0.0,
                    "false_success_blocked_pct": 0.0,
                    "false_refusal_pct": 0.0,
                    "autonomous_recovery_pct": 0.0,
                    "duration_seconds": 0.0,
                    "task_outcomes": [],
                    "model_metadata": {"provider": "ollama", "base_url": base_url.rstrip("/"), "model": mod},
                }
                continue

            live_cfg = ZelyrConfig(
                ollama_base_url=base_url,
                ollama_num_ctx=4096,
                ollama_model=mod,
            )
            orch = NativeOrchestrator(config=live_cfg)
            mod_t0 = time.perf_counter()
            outcomes: List[MatrixTaskOutcome] = []

            for task_id in self.STANDARD_TASKS:
                task_t0 = time.perf_counter()
                prompt_map = {
                    "task_matrix_grounding_refusal": "List all files in the current workspace directory.",
                    "task_matrix_degraded_markdown_ingestion": "Write 'matrix_verified' to build.status.",
                    "task_matrix_scalar_argument_coercion": "Inspect desktop window workspace 1 and verify.",
                    "task_matrix_false_success_interception": "Read non-existent system file /var/log/missing.log.",
                    "task_matrix_anti_consultant_recovery": "Read the cluster configuration file /etc/cluster.conf.",
                }
                payload = {
                    "model": mod,
                    "messages": [{"role": "user", "content": prompt_map[task_id]}],
                    "options": {"num_ctx": 4096},
                }

                executed_tools: List[str] = []
                async def mock_exec(name: str, **kwargs) -> Any:
                    executed_tools.append(name)
                    if task_id in ("task_matrix_false_success_interception", "task_matrix_anti_consultant_recovery"):
                        if len(executed_tools) == 1:
                            return "Exit code 1\nResource not found"
                    return "Exit code 0\nsuccess"

                streamed: List[str] = []
                try:
                    from unittest.mock import patch
                    with patch("zelyr.agents.native_orchestrator.execute_tool", side_effect=mock_exec):
                        async for chunk in orch.generate_stream(payload):
                            choices = chunk.get("choices", [])
                            if choices and choices[0].get("delta", {}).get("content"):
                                streamed.append(choices[0]["delta"]["content"])
                except Exception as exc:
                    outcomes.append(
                        MatrixTaskOutcome(
                            task_id=task_id,
                            passed=False,
                            tool_executed=bool(executed_tools),
                            false_success_suppressed=False,
                            refusal_suppressed=False,
                            autonomous_recovery=False,
                            duration_seconds=time.perf_counter() - task_t0,
                            details=f"Live error: {exc}",
                            combination_id=f"{mod}::{task_id}",
                        )
                    )
                    continue

                full_text = "".join(streamed)
                dur = time.perf_counter() - task_t0
                tool_exec = len(executed_tools) > 0
                refusal_supp = ("I cannot access" not in full_text)
                false_succ_supp = ("successfully read the file" not in full_text and "all set" not in full_text)
                auto_rec = (len(executed_tools) >= 2) if task_id in ("task_matrix_anti_consultant_recovery",) else True

                passed = tool_exec and refusal_supp and false_succ_supp and auto_rec
                outcomes.append(
                    MatrixTaskOutcome(
                        task_id=task_id,
                        passed=passed,
                        tool_executed=tool_exec,
                        false_success_suppressed=false_succ_supp,
                        refusal_suppressed=refusal_supp,
                        autonomous_recovery=auto_rec,
                        duration_seconds=dur,
                        details="Verified" if passed else "Check failed",
                        combination_id=f"{mod}::{task_id}",
                    )
                )

            total_tasks = len(outcomes)
            passed_tasks = sum(1 for o in outcomes if o.passed)
            compliance = (sum(1 for o in outcomes if o.tool_executed) / total_tasks) if total_tasks else 0.0
            false_succ_blocked = 1.0 if all(o.false_success_suppressed for o in outcomes) else 0.0
            false_refusals = 0.0 if all(o.refusal_suppressed for o in outcomes) else 1.0
            recovery_rate = (sum(1 for o in outcomes if o.autonomous_recovery) / total_tasks) if total_tasks else 0.0

            mod_meta = dict(metadata_by_model.get(mod) or {"name": mod})
            mod_meta["provider"] = "ollama"
            mod_meta["base_url"] = base_url.rstrip("/")
            metrics = ArchetypeMetrics(
                name=mod,
                tasks_total=total_tasks,
                tasks_passed=passed_tasks,
                compliance_rate=round(compliance, 4),
                false_success_blocked_rate=round(false_succ_blocked, 4),
                false_refusal_rate=round(false_refusals, 4),
                autonomous_recovery_rate=round(recovery_rate, 4),
                duration_seconds=round(time.perf_counter() - mod_t0, 4),
                task_outcomes=outcomes,
                model_metadata=mod_meta,
            )
            entry_dict = metrics.to_dict()
            entry_dict["available"] = True
            model_results[mod] = entry_dict

        return {
            "matrix_version": "1.4",
            "live_eval": True,
            "base_url": base_url,
            "duration_seconds": round(time.perf_counter() - t0, 4),
            "models": model_results,
        }


def format_matrix_table(matrix_data: Dict[str, Any]) -> str:
    """Format evaluation matrix metrics into a Rich or ASCII table string."""
    archetypes = matrix_data.get("archetypes") or matrix_data.get("models") or {}

    try:
        from rich.console import Console
        from rich.table import Table
        import io

        table = Table(title="Zelyr Model Matrix Empirical Evaluation")
        table.add_column("Model / Archetype", style="cyan", no_wrap=True)
        table.add_column("Tasks", justify="center")
        table.add_column("Compliance", justify="right", style="green")
        table.add_column("False Success Blocked", justify="right", style="magenta")
        table.add_column("False Refusals", justify="right", style="yellow")
        table.add_column("Autonomous Recovery", justify="right", style="blue")
        table.add_column("Duration", justify="right")

        for name, data in archetypes.items():
            tasks_disp = f"{data.get('tasks_passed', 0)}/{data.get('tasks_total', 5)}"
            compliance_disp = f"{data.get('compliance_rate_pct', 0.0):.1f}%"
            false_succ_disp = f"{data.get('false_success_blocked_pct', 0.0):.1f}%"
            refusal_disp = f"{data.get('false_refusal_pct', 0.0):.1f}%"
            recovery_disp = f"{data.get('autonomous_recovery_pct', 0.0):.1f}%"
            dur_disp = f"{data.get('duration_seconds', 0.0):.2f}s"

            table.add_row(
                name,
                tasks_disp,
                compliance_disp,
                false_succ_disp,
                refusal_disp,
                recovery_disp,
                dur_disp,
            )

        console = Console(file=io.StringIO(), force_terminal=False, width=120)
        console.print(table)
        return console.file.getvalue()
    except Exception:
        # Fallback to ASCII table formatting
        lines = [
            "=" * 105,
            "ZELYR MODEL MATRIX EMPIRICAL EVALUATION REPORT",
            "=" * 105,
            f"{'Model / Archetype':<40} {'Tasks':<8} {'Compliance':<12} {'False Succ Blk':<16} {'False Refusal':<14} {'Auto Recovery':<14}",
            "-" * 105,
        ]
        for name, data in archetypes.items():
            tasks_disp = f"{data.get('tasks_passed', 0)}/{data.get('tasks_total', 5)}"
            compliance_disp = f"{data.get('compliance_rate_pct', 0.0):.1f}%"
            false_succ_disp = f"{data.get('false_success_blocked_pct', 0.0):.1f}%"
            refusal_disp = f"{data.get('false_refusal_pct', 0.0):.1f}%"
            recovery_disp = f"{data.get('autonomous_recovery_pct', 0.0):.1f}%"
            lines.append(
                f"{name:<40} {tasks_disp:<8} {compliance_disp:<12} {false_succ_disp:<16} {refusal_disp:<14} {recovery_disp:<14}"
            )
        lines.append("=" * 105)
        return "\n".join(lines)


def render_matrix_scorecard_markdown(matrix_data: Dict[str, Any]) -> str:
    """Render the evaluation matrix as GitHub-flavored Markdown."""
    archetypes = matrix_data.get("archetypes") or matrix_data.get("models") or {}
    lines = [
        "## Model Independence Empirical Matrix Scorecard",
        "",
        "| Model / Archetype | Tasks Passed | Compliance Rate | False-Success Blocked | False Refusals | Autonomous Recovery | Duration |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for name, data in archetypes.items():
        tasks_disp = f"{data.get('tasks_passed', 0)}/{data.get('tasks_total', 5)}"
        compliance_disp = f"{data.get('compliance_rate_pct', 0.0):.1f}%"
        false_succ_disp = f"{data.get('false_success_blocked_pct', 0.0):.1f}%"
        refusal_disp = f"{data.get('false_refusal_pct', 0.0):.1f}%"
        recovery_disp = f"{data.get('autonomous_recovery_pct', 0.0):.1f}%"
        dur_disp = f"{data.get('duration_seconds', 0.0):.3f}s"
        lines.append(
            f"| **{name}** | {tasks_disp} | {compliance_disp} | {false_succ_disp} | {refusal_disp} | {recovery_disp} | {dur_disp} |"
        )
    lines.append("")
    return "\n".join(lines)
