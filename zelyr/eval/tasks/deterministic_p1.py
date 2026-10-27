"""Deterministic Phase 1 evaluation tasks for Zelyr orchestration."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, List, Tuple

from zelyr.db.memory import MemoryStore
from zelyr.eval.metrics import EvalStatus
from zelyr.eval.task import EvalContext, EvalTask
from zelyr.eval.validators import (
    assert_file_content,
    assert_file_exists,
    assert_memory_record,
    assert_tool_called,
    assert_tool_sequence,
)
from zelyr.tools.core import ToolResult


def build_task_tool_dispatch() -> EvalTask:
    """Task 1: Dispatches an allowlisted tool and verifies observable side-effects."""
    def setup(workspace: Path, memory: MemoryStore) -> None:
        (workspace / "status.txt").write_text("system_ok_token_4821", encoding="utf-8")

    def mock_read_file(path: str, context: EvalContext) -> dict:
        target = context.resolve_path(path)
        return {"success": True, "content": target.read_text(encoding="utf-8")}

    mock_responses = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_dispatch_1",
                    "function": {"name": "mock_read_file", "arguments": {"path": "status.txt"}},
                }
            ],
        },
        {
            "role": "assistant",
            "content": "Status read successfully: system_ok_token_4821",
        },
    ]

    async def validator(ctx: EvalContext) -> Tuple[bool, str]:
        ok_called, reason_called = assert_tool_called(ctx.tool_calls, "mock_read_file", min_count=1)
        if not ok_called:
            return False, reason_called
        ok_file, reason_file = assert_file_content(ctx.workspace_dir, "status.txt", "system_ok_token_4821", exact=True)
        if not ok_file:
            return False, reason_file
        return True, "Tool mock_read_file dispatched and verified file contents."

    return EvalTask(
        task_id="task_tool_dispatch",
        description="Verify tool dispatch and observable artifact inspection",
        level=1,
        allowed_tools=["mock_read_file"],
        custom_tools={"mock_read_file": mock_read_file},
        tool_schemas=[
            {
                "type": "function",
                "function": {
                    "name": "mock_read_file",
                    "description": "Read file contents",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            }
        ],
        mock_responses=mock_responses,
        setup_fn=setup,
        validator_fn=validator,
    )


def build_task_recovery_envelope() -> EvalTask:
    """Task 2: Ingests structured ToolResult failure and triggers allowed recovery tool."""
    state = {"unlocked": False}

    def read_locked_file(path: str) -> ToolResult:
        if not state["unlocked"]:
            return ToolResult(
                success=False,
                error="File locked",
                remedy_hint="Use unlock_file first",
                allowed_actions=["unlock_file"],
            )
        return ToolResult(success=True, output="Unlocked file data")

    def unlock_file() -> ToolResult:
        state["unlocked"] = True
        return ToolResult(success=True, output="Lock released")

    mock_responses = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_read_1",
                    "function": {"name": "read_locked_file", "arguments": {"path": "secure.dat"}},
                }
            ],
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_unlock_1",
                    "function": {"name": "unlock_file", "arguments": {}},
                }
            ],
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_read_2",
                    "function": {"name": "read_locked_file", "arguments": {"path": "secure.dat"}},
                }
            ],
        },
        {
            "role": "assistant",
            "content": "File unlocked and read successfully.",
        },
    ]

    async def validator(ctx: EvalContext) -> Tuple[bool, str]:
        ok_seq, reason_seq = assert_tool_sequence(
            ctx.tool_calls, ["read_locked_file", "unlock_file", "read_locked_file"]
        )
        if not ok_seq:
            return False, reason_seq
        return True, "Recovery envelope successfully processed and recovery tool invoked."

    return EvalTask(
        task_id="task_recovery_envelope",
        description="Verify orchestrator recovery loop upon structured ToolResult error",
        level=1,
        allowed_tools=["read_locked_file", "unlock_file"],
        custom_tools={"read_locked_file": read_locked_file, "unlock_file": unlock_file},
        tool_schemas=[
            {
                "type": "function",
                "function": {"name": "read_locked_file", "description": "Read file"},
            },
            {
                "type": "function",
                "function": {"name": "unlock_file", "description": "Unlock file"},
            },
        ],
        mock_responses=mock_responses,
        validator_fn=validator,
    )


def build_task_step_limit_enforcement() -> EvalTask:
    """Task 3: Deliberately non-terminating loop halts at configured max_steps boundary."""
    def endless_poll() -> dict:
        return {"status": "pending"}

    # Mock responses loop indefinitely
    mock_responses = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"call_poll_{i}",
                    "function": {"name": "endless_poll", "arguments": {}},
                }
            ],
        }
        for i in range(10)
    ]

    async def validator(ctx: EvalContext) -> Tuple[bool, str]:
        return True, "Step limit verification handled by harness status"

    return EvalTask(
        task_id="task_step_limit_enforcement",
        description="Verify execution termination at max_steps boundary",
        level=1,
        max_steps=3,
        allowed_tools=["endless_poll"],
        custom_tools={"endless_poll": endless_poll},
        tool_schemas=[{"type": "function", "function": {"name": "endless_poll"}}],
        mock_responses=mock_responses,
        validator_fn=validator,
        expected_status=EvalStatus.STEP_LIMIT_EXCEEDED,
    )


def build_task_false_success_rejection() -> EvalTask:
    """Task 4: Model claims completion in text without creating required side-effects."""
    # Model text claims completion without calling any tool
    mock_responses = [
        {
            "role": "assistant",
            "content": "I have successfully created diagnostic.json with all results!",
        }
    ]

    async def validator(ctx: EvalContext) -> Tuple[bool, str]:
        ok_file, reason = assert_file_exists(ctx.workspace_dir, "diagnostic.json")
        return ok_file, reason

    return EvalTask(
        task_id="task_false_success_rejection",
        description="Verify detection of false success claims absent real side effects",
        level=1,
        allowed_tools=["create_diagnostic"],
        custom_tools={"create_diagnostic": lambda: {"success": True}},
        tool_schemas=[{"type": "function", "function": {"name": "create_diagnostic"}}],
        mock_responses=mock_responses,
        validator_fn=validator,
        expected_status=EvalStatus.FALSE_SUCCESS,
    )


def build_task_memory_persistence_roundtrip() -> EvalTask:
    """Task 5: Stores memory fact and validates direct SQLite database persistence."""
    def setup(workspace: Path, memory: MemoryStore) -> None:
        memory.store(
            content="Host database port is 5432",
            category="environment",
            key="db_port",
        )

    def query_memory(key: str, memory_store: MemoryStore) -> dict:
        rec = memory_store.get_by_key(key)
        return {"success": True, "record": rec}

    mock_responses = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_mem_1",
                    "function": {"name": "query_memory", "arguments": {"key": "db_port"}},
                }
            ],
        },
        {
            "role": "assistant",
            "content": "Database port is 5432.",
        },
    ]

    async def validator(ctx: EvalContext) -> Tuple[bool, str]:
        ok_db, reason_db = assert_memory_record(ctx.db_path, "environment", "db_port", "5432")
        if not ok_db:
            return False, reason_db
        return True, "Memory record successfully retrieved and verified on disk."

    return EvalTask(
        task_id="task_memory_persistence_roundtrip",
        description="Verify memory persistence and direct database record verification",
        level=1,
        allowed_tools=["query_memory"],
        custom_tools={"query_memory": query_memory},
        tool_schemas=[{"type": "function", "function": {"name": "query_memory"}}],
        mock_responses=mock_responses,
        setup_fn=setup,
        validator_fn=validator,
    )


def build_task_timeout_containment() -> EvalTask:
    """Task 6: Blocks execution and verifies timeout abort and cleanup."""
    async def blocking_tool() -> dict:
        await asyncio.sleep(5.0)
        return {"success": True}

    mock_responses = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_block_1",
                    "function": {"name": "blocking_tool", "arguments": {}},
                }
            ],
        }
    ]

    async def validator(ctx: EvalContext) -> Tuple[bool, str]:
        return True, "Timeout verified by harness"

    return EvalTask(
        task_id="task_timeout_containment",
        description="Verify execution cancellation upon timeout limit",
        level=1,
        timeout_seconds=0.5,
        allowed_tools=["blocking_tool"],
        custom_tools={"blocking_tool": blocking_tool},
        tool_schemas=[{"type": "function", "function": {"name": "blocking_tool"}}],
        mock_responses=mock_responses,
        validator_fn=validator,
        expected_status=EvalStatus.TIMEOUT,
    )


def get_deterministic_p1_tasks() -> List[EvalTask]:
    """Retrieve full Level 1 deterministic benchmark battery."""
    return [
        build_task_tool_dispatch(),
        build_task_recovery_envelope(),
        build_task_step_limit_enforcement(),
        build_task_false_success_rejection(),
        build_task_memory_persistence_roundtrip(),
        build_task_timeout_containment(),
    ]
