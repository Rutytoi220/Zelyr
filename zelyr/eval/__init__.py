"""Zelyr Deterministic Evaluation Subsystem."""

from zelyr.eval.context import EvaluationContext
from zelyr.eval.harness import EvalHarness, EvaluationHarness, LiveOllamaStreamAdapter, check_ollama_status_async
from zelyr.eval.metrics import EvalStatus, TaskResult, export_json_report, format_summary_table
from zelyr.eval.provider import DeterministicModelProvider, MockTurn
from zelyr.eval.report import BenchmarkReport
from zelyr.eval.scorecard import (
    BenchmarkScorecard,
    export_scorecard,
    generate_benchmark_scorecard,
    render_markdown_scorecard,
)
from zelyr.eval.suite_l1 import get_level_1_suite
from zelyr.eval.task import EvalContext, EvalTask, EvaluationContext, EvaluationTask
from zelyr.eval.tasks.deterministic_p1 import get_deterministic_p1_tasks
from zelyr.eval.tasks.model_tasks_p3 import (
    check_ollama_status,
    get_model_tasks_p3,
    task_live_web_research_synthesis,
)
from zelyr.eval.tasks.self_repair_l3 import get_self_repair_l3_tasks
from zelyr.eval.tasks.synthetic_workflows_p2 import get_synthetic_workflows_p2_tasks
from zelyr.eval.types import EvaluationResult, ResultStatus, ValidationResult
from zelyr.eval.validators import (
    assert_file_content,
    assert_file_exists,
    assert_json_field,
    assert_memory_record,
    assert_no_tool_errors,
    assert_regex_match,
    assert_tool_called,
    assert_tool_sequence,
    assert_tool_updated,
    combine_validators,
    validate_file_content,
    validate_file_exists,
    validate_json_field,
    validate_memory_record,
    validate_tool_called,
    validate_tool_sequence,
    validate_tool_updated,
)

__all__ = [
    "EvalTask",
    "EvalContext",
    "EvalHarness",
    "EvalStatus",
    "TaskResult",
    "format_summary_table",
    "export_json_report",
    "get_deterministic_p1_tasks",
    "get_synthetic_workflows_p2_tasks",
    "get_self_repair_l3_tasks",
    "get_model_tasks_p3",
    "task_live_web_research_synthesis",
    "check_ollama_status",
    "check_ollama_status_async",
    "LiveOllamaStreamAdapter",
    "assert_file_exists",
    "assert_file_content",
    "assert_regex_match",
    "assert_no_tool_errors",
    "assert_memory_record",
    "assert_tool_called",
    "assert_tool_sequence",
    "EvaluationTask",
    "EvaluationContext",
    "EvaluationHarness",
    "ResultStatus",
    "ValidationResult",
    "EvaluationResult",
    "MockTurn",
    "DeterministicModelProvider",
    "BenchmarkReport",
    "get_level_1_suite",
    "validate_file_exists",
    "validate_file_content",
    "validate_tool_called",
    "validate_tool_sequence",
    "validate_memory_record",
    "validate_json_field",
    "validate_tool_updated",
    "assert_json_field",
    "assert_tool_updated",
    "combine_validators",
    "BenchmarkScorecard",
    "generate_benchmark_scorecard",
    "render_markdown_scorecard",
    "export_scorecard",
]
