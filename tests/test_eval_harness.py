"""Comprehensive unit and integration test suite for Zelyr Evaluation Harness."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.eval.context import EvaluationContext
from zelyr.eval.harness import EvalHarness, EvaluationHarness
from zelyr.eval.metrics import EvalStatus, export_json_report, format_summary_table
from zelyr.eval.provider import DeterministicModelProvider, MockTurn
from zelyr.eval.report import BenchmarkReport
from zelyr.eval.suite_l1 import (
    build_eval_l1_exception_cleanup,
    build_eval_l1_false_success,
    build_eval_l1_memory_round_trip,
    build_eval_l1_recovery,
    build_eval_l1_step_limit,
    build_eval_l1_tool_dispatch,
    get_level_1_suite,
)
from zelyr.eval.task import EvalContext, EvaluationTask
from zelyr.eval.tasks.deterministic_p1 import (
    build_task_false_success_rejection,
    build_task_memory_persistence_roundtrip,
    build_task_recovery_envelope,
    build_task_step_limit_enforcement,
    build_task_timeout_containment,
    build_task_tool_dispatch,
    get_deterministic_p1_tasks,
)
from zelyr.eval.tasks.model_tasks_p3 import (
    build_task_live_web_research_synthesis,
    task_live_web_research_synthesis,
)
from zelyr.eval.types import ResultStatus


class TestEvaluationHarness(unittest.IsolatedAsyncioTestCase):
    """Verifies the deterministic evaluation subsystem, isolation boundaries, and L1 suite."""

    async def asyncSetUp(self) -> None:
        self.harness = EvaluationHarness()

    async def test_passing_task_lifecycle(self) -> None:
        """Verify passing task executes, validates observable file artifact, and tears down."""
        task = build_eval_l1_tool_dispatch()
        result = await self.harness.execute_task(task)

        self.assertEqual(result.status, ResultStatus.PASS)
        self.assertTrue(result.is_success)
        self.assertEqual(result.step_count, 2)
        self.assertEqual(result.tool_call_count, 1)
        self.assertEqual(result.tool_sequence, ["write_file"])
        self.assertTrue(result.cleanup_success)
        self.assertIn("verified successfully", result.validator_message)
        self.assertIsNotNone(result.estimated_tokens)
        self.assertIsNone(result.measured_tokens)

    async def test_validator_rejection_of_false_success(self) -> None:
        """Verify independent validator rejects completion prose when side-effect is absent."""
        task = build_eval_l1_false_success()
        result = await self.harness.execute_task(task)

        self.assertEqual(result.status, ResultStatus.FAIL)
        self.assertFalse(result.is_success)
        self.assertEqual(result.failure_category, "VALIDATION_FAILED")
        self.assertIn("Model claimed success", result.validator_message)
        self.assertTrue(result.cleanup_success)

    async def test_tool_dispatch_and_recovery(self) -> None:
        """Verify model handles tool error response and executes recovery tool call."""
        task = build_eval_l1_recovery()
        result = await self.harness.execute_task(task)

        self.assertEqual(result.status, ResultStatus.PASS)
        self.assertTrue(result.is_success)
        self.assertEqual(result.tool_sequence, ["query_status", "initialize_status"])
        self.assertTrue(result.cleanup_success)

    async def test_step_limit_boundaries(self) -> None:
        """Verify non-terminating tool dispatch loops are terminated strictly at step_limit."""
        task = build_eval_l1_step_limit()
        result = await self.harness.execute_task(task)

        self.assertEqual(result.status, ResultStatus.STEP_LIMIT_EXCEEDED)
        self.assertEqual(result.step_count, 3)
        self.assertEqual(result.tool_call_count, 3)
        self.assertTrue(result.cleanup_success)

    async def test_timeout_and_cancellation(self) -> None:
        """Verify timeout boundary terminates execution and cleans up resources."""
        harness = EvaluationHarness()

        async def slow_fake_tool() -> dict:
            await asyncio.sleep(0.5)
            return {"success": True}

        slow_task = EvaluationTask(
            task_id="eval_tool_timeout_test",
            description="Tests timeout during tool execution",
            user_prompt="Run slow tool",
            timeout=0.05,
            allowed_tools=["slow_tool"],
            fake_tools={"slow_tool": slow_fake_tool},
            tool_schemas=[{"type": "function", "function": {"name": "slow_tool"}}],
            mock_turns=[MockTurn.tool_call("slow_tool", {})],
        )

        result = await harness.execute_task(slow_task)
        self.assertEqual(result.status, ResultStatus.TIMEOUT)
        self.assertEqual(result.failure_category, "TIMEOUT")
        self.assertTrue(result.cleanup_success)

    async def test_isolated_database_and_separate_session(self) -> None:
        """Verify persistent memory is written to isolated DB and verified in a separate session."""
        task = build_eval_l1_memory_round_trip()
        result = await self.harness.execute_task(task)

        self.assertEqual(result.status, ResultStatus.PASS)
        self.assertEqual(result.tool_sequence, ["store_memory"])
        self.assertTrue(result.cleanup_success)

    async def test_exception_handling_and_cleanup(self) -> None:
        """Verify injected failures trigger ERROR classification and ensure resource cleanup."""
        task = build_eval_l1_exception_cleanup()
        result = await self.harness.execute_task(task)

        self.assertEqual(result.status, ResultStatus.ERROR)
        self.assertIn("Simulated execution failure", result.error_message or "")
        self.assertTrue(result.cleanup_success)

    async def test_strict_tool_allowlist_enforcement(self) -> None:
        """Verify unauthorized/unregistered tool calls are blocked by security allowlist."""
        task = EvaluationTask(
            task_id="eval_unauthorized_tool",
            description="Attempts unauthorized tool dispatch",
            user_prompt="Execute unauthorized tool",
            allowed_tools=["safe_tool"],
            fake_tools={"safe_tool": lambda: {"success": True}},
            tool_schemas=[
                {"type": "function", "function": {"name": "forbidden_tool"}},
            ],
            mock_turns=[
                MockTurn.tool_call("forbidden_tool", {}),
                MockTurn.text("Tool attempt complete."),
            ],
            validator=None,
        )

        result = await self.harness.execute_task(task)
        self.assertEqual(len(result.tool_sequence), 1)
        self.assertEqual(result.tool_sequence[0], "forbidden_tool")
        self.assertTrue(result.cleanup_success)

    async def test_json_report_validity(self) -> None:
        """Verify BenchmarkReport accurately computes metrics and serializes to valid JSON."""
        t1 = build_eval_l1_tool_dispatch()
        t2 = build_eval_l1_false_success()

        report = await self.harness.run_suite([t1, t2], suite_name="Test Report Suite")

        self.assertEqual(report.total_tasks, 2)
        self.assertEqual(report.passed_count, 1)
        self.assertEqual(report.failed_count, 1)
        self.assertEqual(report.pass_rate, 0.5)

        json_str = report.to_json()
        data = json.loads(json_str)
        self.assertEqual(data["suite_name"], "Test Report Suite")
        self.assertEqual(data["passed"], 1)
        self.assertEqual(data["failed"], 1)
        self.assertEqual(len(data["results"]), 2)

        summary = report.summary_table()
        self.assertIn("=== Benchmark Report: Test Report Suite ===", summary)
        self.assertIn("eval_l1_tool_dispatch", summary)
        self.assertIn("eval_l1_false_success", summary)

    def test_production_default_backward_compatibility(self) -> None:
        """Verify NativeOrchestrator maintains existing production defaults when unconfigured."""
        orch = NativeOrchestrator()
        self.assertEqual(orch.max_steps, 5)
        self.assertIsNone(orch.model_provider)
        self.assertIsNone(orch.tool_schemas)
        from zelyr.core.plugins import execute_tool
        self.assertEqual(orch.execute_tool, execute_tool)

    async def test_no_model_endpoint_invocation_when_using_deterministic_provider(self) -> None:
        """Verify deterministic provider streams completely offline without contacting Ollama."""
        provider = DeterministicModelProvider(
            turns=[MockTurn.text("Offline deterministic answer.")]
        )
        orch = NativeOrchestrator(model_provider=provider)

        payload = {
            "model": "non-existent-model",
            "messages": [{"role": "user", "content": "Offline test"}],
        }

        chunks = []
        async for chunk in orch.generate_stream(payload):
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if "content" in delta:
                chunks.append(delta["content"])

        full_text = "".join(chunks)
        self.assertIn("Offline deterministic answer.", full_text)
        self.assertEqual(provider.request_count, 1)

    async def test_level_1_suite_complete_run(self) -> None:
        """Verify entire Level 1 suite runs deterministically and classifies all 6 tasks."""
        suite = get_level_1_suite()
        self.assertEqual(len(suite), 6)

        report = await self.harness.run_suite(suite, suite_name="Level 1 Benchmark")

        self.assertEqual(report.total_tasks, 6)
        results_by_id = {r.task_id: r for r in report.results}

        self.assertEqual(results_by_id["eval_l1_tool_dispatch"].status, ResultStatus.PASS)
        self.assertEqual(results_by_id["eval_l1_recovery"].status, ResultStatus.PASS)
        self.assertEqual(results_by_id["eval_l1_step_limit"].status, ResultStatus.STEP_LIMIT_EXCEEDED)
        self.assertEqual(results_by_id["eval_l1_false_success"].status, ResultStatus.FAIL)
        self.assertEqual(results_by_id["eval_l1_memory_round_trip"].status, ResultStatus.PASS)
        self.assertEqual(results_by_id["eval_l1_exception_cleanup"].status, ResultStatus.ERROR)

        for r in report.results:
            self.assertTrue(r.cleanup_success, f"Task {r.task_id} failed resource cleanup")


class TestEvalHarnessP1(unittest.IsolatedAsyncioTestCase):
    """Test suite covering deterministic orchestration, independent validators, and metrics."""

    async def asyncSetUp(self) -> None:
        self.harness = EvalHarness()

    async def test_harness_successful_task_passes(self) -> None:
        """Verify valid task lifecycle dispatches tool and achieves PASS status."""
        task = build_task_tool_dispatch()
        result = await self.harness.run_task(task)

        self.assertEqual(result.status, EvalStatus.PASS)
        self.assertTrue(result.is_success)
        self.assertIn("verified file contents", result.message)
        self.assertIn("mock_read_file", result.tool_calls)
        self.assertGreater(result.step_count, 0)
        self.assertGreater(result.estimated_tokens, 0)
        self.assertIsNone(result.measured_tokens)

    async def test_harness_catches_false_success(self) -> None:
        """Verify model prose claiming completion without real side-effects is rejected as FALSE_SUCCESS."""
        task = build_task_false_success_rejection()
        result = await self.harness.run_task(task)

        self.assertEqual(result.status, EvalStatus.FALSE_SUCCESS)
        self.assertFalse(result.is_success)
        self.assertEqual(result.failure_category, "FALSE_SUCCESS")
        self.assertIn("Model claimed success in text without observable state change", result.message)

    async def test_harness_enforces_step_limit(self) -> None:
        """Verify non-terminating loops are strictly halted at max_steps with STEP_LIMIT_EXCEEDED."""
        task = build_task_step_limit_enforcement()
        result = await self.harness.run_task(task)

        self.assertEqual(result.status, EvalStatus.STEP_LIMIT_EXCEEDED)
        self.assertEqual(result.failure_category, "STEP_LIMIT_EXCEEDED")
        self.assertEqual(result.step_count, 3)

    async def test_harness_remedy_recovery_flow(self) -> None:
        """Verify orchestrator recovers from ToolResult structured error and executes remedy tool."""
        task = build_task_recovery_envelope()
        result = await self.harness.run_task(task)

        self.assertEqual(result.status, EvalStatus.PASS)
        self.assertIn("Recovery envelope successfully processed", result.message)
        self.assertEqual(result.tool_calls, ["read_locked_file", "unlock_file", "read_locked_file"])

    async def test_harness_timeout_cancellation(self) -> None:
        """Verify long-running or blocking tasks abort cleanly with TIMEOUT."""
        task = build_task_timeout_containment()
        result = await self.harness.run_task(task)

        self.assertEqual(result.status, EvalStatus.TIMEOUT)
        self.assertEqual(result.failure_category, "TIMEOUT")
        self.assertIn("timed out", result.message)

    async def test_harness_workspace_and_db_isolation(self) -> None:
        """Verify evaluation executes strictly inside temporary workspace without host pollution."""
        task = build_task_memory_persistence_roundtrip()
        result = await self.harness.run_task(task)

        self.assertEqual(result.status, EvalStatus.PASS)
        self.assertIn("query_memory", result.tool_calls)

        self.assertFalse(Path("test_eval.db").exists())
        self.assertFalse(Path("/tmp/test_eval.db").exists())

    async def test_metrics_and_token_distinction(self) -> None:
        """Verify telemetry distinguishes estimated vs measured tokens and serializes valid JSON report."""
        t1 = build_task_tool_dispatch()
        t2 = build_task_false_success_rejection()

        res1 = await self.harness.run_task(t1)
        res2 = await self.harness.run_task(t2)

        results = [res1, res2]
        table_str = format_summary_table(results)
        self.assertIn("Zelyr DETERMINISTIC EVALUATION BENCHMARK REPORT", table_str)
        self.assertIn("task_tool_dispatch", table_str)
        self.assertIn("task_false_success_rejection", table_str)

        with tempfile.TemporaryDirectory() as tmp_dir:
            json_file = Path(tmp_dir) / "eval_report.json"
            export_json_report(results, json_file)
            self.assertTrue(json_file.exists())

            data = json.loads(json_file.read_text(encoding="utf-8"))
            self.assertEqual(data["summary"]["total"], 2)
            self.assertEqual(data["summary"]["passed"], 1)
            self.assertEqual(data["summary"]["false_success"], 1)

            for r in data["results"]:
                self.assertIsNone(r["measured_tokens"])
                self.assertIsInstance(r["estimated_tokens"], int)
                self.assertGreater(r["estimated_tokens"], 0)

    async def test_full_deterministic_p1_suite(self) -> None:
        """Verify the full 6-task deterministic suite executes cleanly with expected classifications."""
        tasks = get_deterministic_p1_tasks()
        self.assertEqual(len(tasks), 6)

        results = {}
        for t in tasks:
            res = await self.harness.run_task(t)
            results[t.task_id] = res

        self.assertEqual(results["task_tool_dispatch"].status, EvalStatus.PASS)
        self.assertEqual(results["task_recovery_envelope"].status, EvalStatus.PASS)
        self.assertEqual(results["task_step_limit_enforcement"].status, EvalStatus.STEP_LIMIT_EXCEEDED)
        self.assertEqual(results["task_false_success_rejection"].status, EvalStatus.FALSE_SUCCESS)
        self.assertEqual(results["task_memory_persistence_roundtrip"].status, EvalStatus.PASS)
        self.assertEqual(results["task_timeout_containment"].status, EvalStatus.TIMEOUT)

    def test_workspace_path_traversal_prevention(self) -> None:
        """Verify that attempting to escape workspace_dir via relative path traversal raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            workspace.mkdir()
            db_path = Path(tmp_dir) / "test.db"
            task = build_task_tool_dispatch()
            ctx = EvalContext(task=task, workspace_dir=workspace, db_path=db_path)

            valid_path = ctx.resolve_path("subdir/file.txt")
            self.assertEqual(valid_path, (workspace / "subdir/file.txt").resolve())

            with self.assertRaises(ValueError) as cm:
                ctx.resolve_path("../../etc/shadow")
            self.assertIn("Path traversal detected", str(cm.exception))

            with self.assertRaises(ValueError) as cm2:
                ctx.resolve_path("../other_folder")
            self.assertIn("Path traversal detected", str(cm2.exception))


class TestEvalLevel2WebResearch(unittest.IsolatedAsyncioTestCase):
    """Verifies Level 2 web search and page extraction workflow in offline deterministic mode."""

    async def asyncSetUp(self) -> None:
        self.harness = EvalHarness()

    async def test_web_research_happy_path(self) -> None:
        """Test 1: Happy path - search -> fetch -> grounded synthesis with source URL citation."""
        task = build_task_live_web_research_synthesis()

        def mock_web_search(query: str, num_results: int = 5) -> dict:
            return {
                "success": True,
                "output": (
                    "### Search Results for 'Project Chimera v2'\n"
                    "1. [Project Chimera v2 Documentation](https://example.com/chimera-v2-docs)\n"
                    "   Official release details and architecture specification for Chimera v2.\n"
                ),
            }

        def mock_fetch_web_page(url: str, max_chars: int = 5000) -> dict:
            return {
                "success": True,
                "output": (
                    '<external_web_content url="https://example.com/chimera-v2-docs">\n'
                    "Project Chimera v2 introduces Quantum-Safe Encryption, releasing 2026-11-15.\n"
                    "Key security architecture revolves around post-quantum cryptographic primitives.\n"
                    "</external_web_content>"
                ),
            }

        def mock_write_file(path: str, content: str, context: EvalContext) -> dict:
            target = context.resolve_path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return {"success": True, "path": path, "bytes_written": len(content)}

        mock_responses = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_search_1",
                        "function": {
                            "name": "web_search",
                            "arguments": {"query": "Project Chimera v2 release date security features"},
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_fetch_1",
                        "function": {
                            "name": "fetch_web_page",
                            "arguments": {"url": "https://example.com/chimera-v2-docs"},
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_write_1",
                        "function": {
                            "name": "write_file",
                            "arguments": {
                                "path": "research_summary.md",
                                "content": (
                                    "# Project Chimera v2 Summary\n\n"
                                    "- Release Date: 2026-11-15\n"
                                    "- Core Security: Quantum-Safe Encryption\n\n"
                                    "Source: https://example.com/chimera-v2-docs\n"
                                ),
                            },
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "Completed research. Summary has been saved to research_summary.md with source URL citations.",
            },
        ]

        custom_tools = {
            "web_search": mock_web_search,
            "fetch_web_page": mock_fetch_web_page,
            "write_file": mock_write_file,
        }

        result = await self.harness.run_task(
            task,
            mock_responses=mock_responses,
            custom_tools=custom_tools,
        )

        self.assertEqual(result.status, EvalStatus.PASS)
        self.assertTrue(result.is_success)
        self.assertEqual(result.score, 1.0)
        self.assertIn("web_search", result.tool_calls)
        self.assertIn("fetch_web_page", result.tool_calls)
        self.assertIn("write_file", result.tool_calls)
        self.assertIn("Web research synthesis verified", result.message)

    async def test_web_research_empty_search_degradation(self) -> None:
        """Test 2: Negative path - empty search results gracefully degrades without hallucination."""
        task = build_task_live_web_research_synthesis()

        def mock_web_search(query: str, num_results: int = 5) -> dict:
            return {
                "success": True,
                "output": "No results found for query: 'Project Chimera v2'",
            }

        def mock_fetch_web_page(url: str, max_chars: int = 5000) -> dict:
            return {"success": False, "error": "Should not be called"}

        def mock_write_file(path: str, content: str, context: EvalContext) -> dict:
            return {"success": True}

        mock_responses = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_search_1",
                        "function": {
                            "name": "web_search",
                            "arguments": {"query": "Project Chimera v2"},
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "I searched for Project Chimera v2 but no results were found. Unable to fetch documentation or produce summary.",
            },
        ]

        custom_tools = {
            "web_search": mock_web_search,
            "fetch_web_page": mock_fetch_web_page,
            "write_file": mock_write_file,
        }

        result = await self.harness.run_task(
            task,
            mock_responses=mock_responses,
            custom_tools=custom_tools,
        )

        self.assertNotEqual(result.status, EvalStatus.PASS)
        self.assertFalse(result.is_success)
        self.assertEqual(result.score, 0.0)
        self.assertIn("web_search", result.tool_calls)
        self.assertNotIn("fetch_web_page", result.tool_calls)

    async def test_web_research_fetch_error_envelope(self) -> None:
        """Test 3: Negative path - HTTP 404 on fetch stops pipeline and rejects false pass."""
        task = build_task_live_web_research_synthesis()

        def mock_web_search(query: str, num_results: int = 5) -> dict:
            return {
                "success": True,
                "output": (
                    "### Search Results\n"
                    "1. [Broken Link](https://example.com/chimera-v2-docs)\n"
                ),
            }

        def mock_fetch_web_page(url: str, max_chars: int = 5000) -> dict:
            return {
                "success": False,
                "error": "Error: HTTP 404 fetching URL: https://example.com/chimera-v2-docs",
            }

        mock_responses = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_search_1",
                        "function": {
                            "name": "web_search",
                            "arguments": {"query": "Project Chimera v2"},
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_fetch_1",
                        "function": {
                            "name": "fetch_web_page",
                            "arguments": {"url": "https://example.com/chimera-v2-docs"},
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "The documentation page returned HTTP 404. I am unable to verify Project Chimera v2 release details.",
            },
        ]

        custom_tools = {
            "web_search": mock_web_search,
            "fetch_web_page": mock_fetch_web_page,
        }

        result = await self.harness.run_task(
            task,
            mock_responses=mock_responses,
            custom_tools=custom_tools,
        )

        self.assertNotEqual(result.status, EvalStatus.PASS)
        self.assertFalse(result.is_success)
        self.assertEqual(result.score, 0.0)
        self.assertIn("fetch_web_page", result.tool_calls)

    async def test_web_research_catches_false_success(self) -> None:
        """Test 4: False-success detection gate rejects model claiming completion without artifact creation."""
        task = build_task_live_web_research_synthesis()

        mock_responses = [
            {
                "role": "assistant",
                "content": "I have researched Project Chimera and completed the task! All results successfully saved.",
            }
        ]

        result = await self.harness.run_task(
            task,
            mock_responses=mock_responses,
        )

        self.assertEqual(result.status, EvalStatus.FALSE_SUCCESS)
        self.assertFalse(result.is_success)
        self.assertEqual(result.failure_category, "FALSE_SUCCESS")
        self.assertEqual(result.score, 0.0)
        self.assertIn("Model claimed success in text without observable state change", result.message)


if __name__ == "__main__":
    unittest.main()
