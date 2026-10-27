"""Targeted evaluation test suite for Level 3: Autonomous Self-Repair & Dynamic Tool Crafting.

Verifies:
1. Runtime Tool Failure Recovery: Intercepting a runtime exception in a tool, synthesizing
   and staging a repaired version, re-executing, and validating the final payload.
2. AST Safety Boundary Enforcement: Rejecting repair candidates violating structural AST,
   contract requirements, or attempting Ring 0 privilege escalation.
3. Max Repair Retry Ceiling: Ensuring repeating tool failures terminate gracefully at
   the configured maximum step threshold without infinite recursion or hangs.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import unittest

from zelyr.eval.cli import run_benchmark_cli
from zelyr.eval.harness import EvalHarness
from zelyr.eval.metrics import EvalStatus
from zelyr.eval.task import EvalContext, EvalTask
from zelyr.tools.core import ToolResult
from zelyr.tools.tool_crafter import craft_tool
from zelyr.tools.validator import ToolValidator


class TestLevel3SelfRepairCoverage(unittest.IsolatedAsyncioTestCase):
    """Deterministic evaluation suite validating Level 3 self-repair, AST gates, and retry limits."""

    async def asyncSetUp(self) -> None:
        self.harness = EvalHarness()

    async def test_task_1_runtime_tool_failure_recovery(self) -> None:
        """Task 1: Tool fails with runtime ZeroDivisionError; orchestrator captures error envelope,

        stages repaired tool via craft_tool, re-executes, and successfully completes.
        """
        initial_failing_code = '''"""Dynamic calculator tool v1 (intentional runtime ZeroDivisionError)."""
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "safe_divider",
        "description": "Divides two numbers with fallback",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
    },
}
REQUIRED_RING = 1
TIER = 1
TUI_HINT = "Dividing v1..."

async def execute(a: float = 10, b: float = 2, **kwargs):
    # Unhandled division by zero in initial implementation
    fault = 1 / 0
    return {"result": fault}
'''

        repaired_code = '''"""Dynamic calculator tool v2 (repaired zero division guard)."""
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "safe_divider",
        "description": "Divides two numbers with fallback",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
    },
}
REQUIRED_RING = 1
TIER = 1
TUI_HINT = "Dividing v2..."
# marker: SAFE_DIVIDER_V2_REPAIRED

async def execute(a: float = 10, b: float = 2, **kwargs):
    if b == 0:
        return {"result": 0.0, "status": "zero_fallback"}
    return {"result": a / b, "status": "ok"}
'''

        mock_responses = [
            # Turn 1: Attempt to craft tool with runtime ZeroDivisionError in staging
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_craft_v1",
                        "function": {
                            "name": "craft_tool",
                            "arguments": {"name": "safe_divider", "code": initial_failing_code},
                        },
                    }
                ],
            },
            # Turn 2: Staging caught division by zero. Model receives error and supplies repaired code
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_craft_v2",
                        "function": {
                            "name": "craft_tool",
                            "arguments": {"name": "safe_divider", "code": repaired_code},
                        },
                    }
                ],
            },
            # Turn 3: Final confirmation turn
            {
                "role": "assistant",
                "content": "Diagnosed ZeroDivisionError during staging, repaired with zero fallback, tool installed successfully.",
            },
        ]

        async def validator(ctx: EvalContext) -> Tuple[bool, str]:
            tool_path = ctx.tools_directory / "safe_divider.py"
            if not tool_path.exists():
                return False, f"Expected safe_divider.py to be installed in {ctx.tools_directory}"
            content = tool_path.read_text(encoding="utf-8")
            if "SAFE_DIVIDER_V2_REPAIRED" not in content:
                return False, "Installed tool does not contain repaired marker"
            return True, "Tool successfully repaired and verified in isolated tools directory"

        async def custom_craft(name: str, code: str, context: EvalContext, **kwargs) -> ToolResult:
            return await craft_tool(
                name=name, code=code, tools_dir=context.tools_directory, db_path=str(context.db_path)
            )

        task = EvalTask(
            task_id="test_l3_runtime_failure_recovery",
            description="Autonomous repair of runtime ZeroDivisionError in dynamic tool staging",
            user_prompt="Create a safe division tool that handles zero divisors gracefully.",
            mock_responses=mock_responses,
            allowed_tools=["craft_tool"],
            custom_tools={
                "craft_tool": custom_craft,
            },
            tool_schemas=[
                {
                    "type": "function",
                    "function": {
                        "name": "craft_tool",
                        "description": "Synthesize a new dynamic tool",
                        "parameters": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}, "code": {"type": "string"}},
                            "required": ["name", "code"],
                        },
                    },
                }
            ],
            validator_fn=validator,
            level=3,
            max_steps=5,
        )

        result = await self.harness.run_task(task)
        self.assertEqual(result.status, EvalStatus.PASS)
        self.assertTrue(result.is_success)
        self.assertEqual(result.tool_calls, ["craft_tool", "craft_tool"])
        self.assertIn("Tool successfully repaired", result.message)

    async def test_task_2_ast_safety_boundary_enforcement(self) -> None:
        """Task 2: Validator and craft_tool reject AST contract violations and privilege escalation."""
        validator = ToolValidator()

        # 1. Missing async def (synchronous execute)
        sync_code = """
TOOL_SCHEMA = {"type": "function", "function": {"name": "sync_tool", "description": "test", "parameters": {"type": "object", "properties": {}}}}
REQUIRED_RING = 1
TIER = 1
TUI_HINT = "sync"

def execute(**kwargs):
    return "sync"
"""
        res_sync = validator.validate_syntax_and_ast(sync_code)
        self.assertFalse(res_sync.is_valid)
        self.assertIn("must be an asynchronous function", res_sync.error)

        # 2. Syntax error
        syntax_err_code = "async def execute():\n    return {"
        res_syntax = validator.validate_syntax_and_ast(syntax_err_code)
        self.assertFalse(res_syntax.is_valid)
        self.assertIn("Syntax error", res_syntax.error)

        # 3. Privilege escalation to Ring 0
        ring0_code = """
TOOL_SCHEMA = {"type": "function", "function": {"name": "ring0_tool", "description": "test", "parameters": {"type": "object", "properties": {}}}}
REQUIRED_RING = 0
TIER = 1
TUI_HINT = "ring0"

async def execute(**kwargs):
    return "compromised"
"""
        val_ring0 = await validator.validate_all(ring0_code)
        self.assertTrue(val_ring0.is_valid)  # Valid syntax/schema
        self.assertEqual(val_ring0.ring, 0)

        # But craft_tool MUST reject Ring 0 registration
        craft_res = await craft_tool(name="ring0_tool", code=ring0_code)
        self.assertFalse(craft_res.success)
        self.assertIn("Privilege escalation rejected", craft_res.error)

        # 4. Path traversal in tool name
        traversal_res = await craft_tool(name="../../evil_escape", code=ring0_code)
        self.assertFalse(traversal_res.success)
        self.assertIn("Invalid tool name", traversal_res.error)

    async def test_task_3_max_repair_retry_ceiling(self) -> None:
        """Task 3: Continually failing tool calls halt safely at max_steps boundary without infinite loop."""
        mock_responses = [
            {
                "role": "assistant",
                "tool_calls": [{"id": f"call_{i}", "function": {"name": "flaky_tool", "arguments": {}}}],
            }
            for i in range(10)
        ]

        async def custom_flaky(context: EvalContext = None, **kw) -> ToolResult:
            return ToolResult(success=False, error="Persistent hardware timeout")

        task = EvalTask(
            task_id="test_l3_max_retry_ceiling",
            description="Test bounded retries when tool fails persistently",
            user_prompt="Execute a failing task repeatedly to test retry limits.",
            mock_responses=mock_responses,
            allowed_tools=["flaky_tool"],
            custom_tools={
                "flaky_tool": custom_flaky,
            },
            tool_schemas=[
                {
                    "type": "function",
                    "function": {
                        "name": "flaky_tool",
                        "description": "A persistently failing tool",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            level=3,
            max_steps=3,
        )

        result = await self.harness.run_task(task)
        self.assertEqual(result.status, EvalStatus.STEP_LIMIT_EXCEEDED)
        self.assertFalse(result.is_success)
        self.assertEqual(result.step_count, 3)
        self.assertEqual(len(result.tool_calls), 3)
        self.assertIn("maximum autonomous ReAct steps (3)", result.message)

    def test_level_3_cli_runner_offline_execution(self) -> None:
        """Verify run_benchmark_cli executes Level 3 tasks cleanly in offline mode."""
        exit_code = run_benchmark_cli(["-l", "3", "-q"])
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
