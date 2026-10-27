"""Level 1 benchmark evaluation suite for Zelyr orchestration."""

from __future__ import annotations

from typing import Any, List, Optional

from zelyr.db.memory import MemoryStore
from zelyr.eval.context import EvaluationContext
from zelyr.eval.provider import MockTurn
from zelyr.eval.task import EvaluationTask
from zelyr.eval.types import ResultStatus, ValidationResult
from zelyr.eval.validators import (
    combine_validators,
    validate_file_content,
    validate_file_exists,
    validate_memory_record,
    validate_tool_called,
    validate_tool_sequence,
)


def build_eval_l1_tool_dispatch() -> EvaluationTask:
    """Task 1: Tool dispatch — writes file artifact via allowlisted tool, verified by independent content check."""
    def fake_write_file(path: str, content: str, context: EvaluationContext) -> dict:
        target = context.resolve_workspace_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"success": True, "bytes_written": len(content)}

    tool_schemas = [
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Writes text content to a relative workspace file path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        }
    ]

    mock_turns = [
        MockTurn.tool_call("write_file", {"path": "report.txt", "content": "verification_token_7721"}),
        MockTurn.text("The file report.txt has been created with the requested token."),
    ]

    return EvaluationTask(
        task_id="eval_l1_tool_dispatch",
        description="Dispatches allowlisted write_file tool and verifies resulting file artifact.",
        user_prompt="Please write the evaluation result to report.txt with the content 'verification_token_7721'",
        benchmark_level=1,
        allowed_tools=["write_file"],
        fake_tools={"write_file": fake_write_file},
        tool_schemas=tool_schemas,
        mock_turns=mock_turns,
        max_steps=5,
        validator=combine_validators(
            validate_file_exists("report.txt"),
            validate_file_content("report.txt", exact_match="verification_token_7721"),
            validate_tool_called("write_file", exact_count=1),
        ),
        expected_status=ResultStatus.PASS,
    )


def build_eval_l1_recovery() -> EvaluationTask:
    """Task 2: Recovery — tool returns structured failure and recovery hint; model invokes allowed recovery tool."""
    def fake_query_status() -> dict:
        return {
            "success": False,
            "error": "System state uninitialized",
            "remedy_hint": "Run initialize_status first.",
            "allowed_actions": ["initialize_status"],
        }

    def fake_initialize_status() -> dict:
        return {"success": True, "status": "initialized"}

    tool_schemas = [
        {
            "type": "function",
            "function": {
                "name": "query_status",
                "description": "Query current system status.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "initialize_status",
                "description": "Initialize or reset system status.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    mock_turns = [
        MockTurn.tool_call("query_status", {}),
        MockTurn.tool_call("initialize_status", {}),
        MockTurn.text("System successfully initialized."),
    ]

    return EvaluationTask(
        task_id="eval_l1_recovery",
        description="Handles structured tool failure and recovery hint to invoke allowed recovery action.",
        user_prompt="Inspect the system state and initialize if needed.",
        benchmark_level=1,
        allowed_tools=["query_status", "initialize_status"],
        fake_tools={
            "query_status": fake_query_status,
            "initialize_status": fake_initialize_status,
        },
        tool_schemas=tool_schemas,
        mock_turns=mock_turns,
        max_steps=5,
        validator=validate_tool_sequence(["query_status", "initialize_status"]),
        expected_status=ResultStatus.PASS,
    )


def build_eval_l1_step_limit() -> EvaluationTask:
    """Task 3: Step limit — non-terminating mock sequence stopped at max_steps=3."""
    def fake_check_loop() -> dict:
        return {"success": True, "status": "still_running"}

    tool_schemas = [
        {
            "type": "function",
            "function": {
                "name": "check_loop",
                "description": "Checks status in a continuous loop.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    mock_turns = [
        MockTurn.tool_call("check_loop", {}),
        MockTurn.tool_call("check_loop", {}),
        MockTurn.tool_call("check_loop", {}),
        MockTurn.tool_call("check_loop", {}),
        MockTurn.tool_call("check_loop", {}),
    ]

    return EvaluationTask(
        task_id="eval_l1_step_limit",
        description="Halts deliberate non-terminating tool loop at max_steps=3 boundary.",
        user_prompt="Keep checking loop status until complete.",
        benchmark_level=1,
        allowed_tools=["check_loop"],
        fake_tools={"check_loop": fake_check_loop},
        tool_schemas=tool_schemas,
        mock_turns=mock_turns,
        max_steps=3,
        validator=None,
        expected_status=ResultStatus.STEP_LIMIT_EXCEEDED,
    )


def build_eval_l1_false_success() -> EvaluationTask:
    """Task 4: False success — model claims completion in prose without tool invocation; validator rejects claim as FAIL."""
    def fake_write_file(path: str, content: str, context: EvaluationContext) -> dict:
        p = context.resolve_workspace_path(path)
        p.write_text(content, encoding="utf-8")
        return {"success": True}

    tool_schemas = [
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Writes text content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
            },
        }
    ]

    # Model returns purely conversational prose claiming completion WITHOUT calling any tool
    mock_turns = [
        MockTurn.text("Done! I have successfully created artifact.txt with 'completed_job'. Everything is finished.")
    ]

    return EvaluationTask(
        task_id="eval_l1_false_success",
        description="Rejects model claim of success when required side effect was not created.",
        user_prompt="Create artifact.txt containing 'completed_job'",
        benchmark_level=1,
        allowed_tools=["write_file"],
        fake_tools={"write_file": fake_write_file},
        tool_schemas=tool_schemas,
        mock_turns=mock_turns,
        max_steps=5,
        validator=validate_file_content("artifact.txt", exact_match="completed_job"),
        expected_status=ResultStatus.FAIL,
    )


def build_eval_l1_memory_round_trip() -> EvaluationTask:
    """Task 5: Memory round trip — writes to isolated DB, verifies via independent connection and separate session."""
    def fake_store_memory(category: str, content: str, key: Optional[str] = None, context: EvaluationContext = None) -> dict:
        if context and context.memory_store:
            mid = context.memory_store.store(content=content, category=category, key=key)
            return {"success": True, "memory_id": mid}
        return {"success": False, "error": "No memory store configured"}

    tool_schemas = [
        {
            "type": "function",
            "function": {
                "name": "store_memory",
                "description": "Persist a memory item.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "key": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["category", "content"],
                },
            },
        }
    ]

    mock_turns = [
        MockTurn.tool_call("store_memory", {"category": "environment", "key": "db_port", "content": "Database port is 5432"}),
        MockTurn.text("Saved database port configuration to memory."),
    ]

    def validate_memory_session_round_trip(context: EvaluationContext) -> ValidationResult:
        # 1. Independent direct sqlite3 inspection
        sql_val = validate_memory_record(category="environment", key="db_port", content_substring="5432")(context)
        if not sql_val.success:
            return sql_val

        # 2. Separate session against that database file
        separate_store = MemoryStore(db_path=str(context.db_path))
        try:
            rec = separate_store.get_by_key("db_port")
            if not rec or "5432" not in rec.get("content", ""):
                return ValidationResult(
                    success=False,
                    message=f"Separate MemoryStore session failed to retrieve key: {rec}",
                )
            searched = separate_store.search("database port", category="environment", limit=1)
            if not searched or "5432" not in searched[0].get("content", ""):
                return ValidationResult(
                    success=False,
                    message=f"Separate MemoryStore session failed search query: {searched}",
                )
        finally:
            if hasattr(separate_store, "close"):
                separate_store.close()

        return ValidationResult(
            success=True,
            message="Memory record successfully verified across separate sessions and raw database.",
        )

    return EvaluationTask(
        task_id="eval_l1_memory_round_trip",
        description="Persists memory to isolated DB and validates retrieval in a separate session.",
        user_prompt="Remember that the database port is 5432 in category environment.",
        benchmark_level=1,
        allowed_tools=["store_memory"],
        fake_tools={"store_memory": fake_store_memory},
        tool_schemas=tool_schemas,
        mock_turns=mock_turns,
        max_steps=5,
        validator=validate_memory_session_round_trip,
        expected_status=ResultStatus.PASS,
    )


def build_eval_l1_exception_cleanup() -> EvaluationTask:
    """Task 6: Exception and cleanup — injected failure reports ERROR and cleans up temporary resources."""
    mock_turns = [
        MockTurn.error(RuntimeError("Simulated execution failure in model stream"))
    ]

    def setup_sentinel(context: EvaluationContext) -> None:
        sentinel = context.resolve_workspace_path("sentinel.tmp")
        sentinel.write_text("active_run", encoding="utf-8")

    return EvaluationTask(
        task_id="eval_l1_exception_cleanup",
        description="Injects execution failure, confirms ERROR classification and resource teardown.",
        user_prompt="Run critical computation.",
        benchmark_level=1,
        allowed_tools=[],
        fake_tools={},
        tool_schemas=[],
        mock_turns=mock_turns,
        max_steps=5,
        setup_hook=setup_sentinel,
        validator=None,
        expected_status=ResultStatus.ERROR,
    )


def get_level_1_suite() -> List[EvaluationTask]:
    """Retrieve full Level 1 evaluation task suite."""
    return [
        build_eval_l1_tool_dispatch(),
        build_eval_l1_recovery(),
        build_eval_l1_step_limit(),
        build_eval_l1_false_success(),
        build_eval_l1_memory_round_trip(),
        build_eval_l1_exception_cleanup(),
    ]
