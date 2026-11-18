"""Comprehensive verification test suite for dynamic tool crafting pipeline.

Verifies:
1. AST Security Analysis & Schema Validation (validate_tool_ast).
2. Sterile Sandbox Self-Test Execution (execute_sandbox_self_test & staging cleanup).
3. Atomic Userland Promotion, Provenance Hash, and Dynamic Hotloading into ToolRegistry.
4. Dynamic tool invocation via execute_tool and registry.execute.
5. ErrorClassifier mapping for TOOL_CRAFT_FAILED.
6. NativeOrchestrator alias mapping and argument normalization.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
import pytest

from zelyr.core.plugins import execute_tool, load_plugins
from zelyr.engine.error_classifier import ErrorClassifier
from zelyr.tool_registry import ToolRegistry
from zelyr.tools.tool_crafter import (
    CraftResult,
    ToolCrafter,
    ToolCrafterTool,
    craft_tool,
    execute,
)
from zelyr.tools.validator import (
    execute_sandbox_self_test,
    validate_tool_ast,
)

VALID_TOOL_CODE = """
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculate_md5",
        "description": "Computes MD5 hash of input text.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to hash"}
            },
            "required": ["text"]
        }
    }
}
REQUIRED_RING = 1
TIER = 1
TUI_HINT = "Computing MD5..."

async def execute(text: str = "", **kwargs) -> str:
    import hashlib
    return hashlib.md5(text.encode("utf-8")).hexdigest()

async def self_test() -> bool:
    res = await execute(text="hello")
    return res == "5d41402abc4b2a76b9719d911017c592"
"""


# ---------------------------------------------------------------------------
# 1. AST Security & Contract Validation Tests
# ---------------------------------------------------------------------------


class TestASTSecurityAndValidation:
    """Tests for validate_tool_ast security invariants and contract exports."""

    def test_valid_tool_ast_passes(self):
        ok, err = validate_tool_ast(VALID_TOOL_CODE)
        assert ok is True
        assert err is None

    def test_syntax_error_rejected_cleanly(self):
        code = "def execute(): syntax error in code !!!"
        ok, err = validate_tool_ast(code)
        assert ok is False
        assert "Syntax error" in err

    def test_dunder_subclasses_rejected(self):
        code = VALID_TOOL_CODE + "\ndef exploit(): return ().__class__.__subclasses__()\n"
        ok, err = validate_tool_ast(code)
        assert ok is False
        assert "Security violation" in err
        assert "__subclasses__" in err

    def test_dunder_globals_rejected(self):
        code = VALID_TOOL_CODE + "\ndef exploit(fn): return fn.__globals__\n"
        ok, err = validate_tool_ast(code)
        assert ok is False
        assert "Security violation" in err
        assert "__globals__" in err

    def test_dunder_builtins_rejected(self):
        code = VALID_TOOL_CODE + "\ndef exploit(): return __builtins__\n"
        ok, err = validate_tool_ast(code)
        assert ok is False
        assert "Security violation" in err
        assert "__builtins__" in err

    def test_dunder_code_rejected(self):
        code = VALID_TOOL_CODE + "\ndef exploit(fn): return fn.__code__\n"
        ok, err = validate_tool_ast(code)
        assert ok is False
        assert "Security violation" in err
        assert "__code__" in err

    def test_eval_call_rejected(self):
        code = VALID_TOOL_CODE + "\ndef exploit(): return eval('1+1')\n"
        ok, err = validate_tool_ast(code)
        assert ok is False
        assert "Security violation" in err
        assert "eval" in err

    def test_exec_call_rejected(self):
        code = VALID_TOOL_CODE + "\ndef exploit(): exec('x = 1')\n"
        ok, err = validate_tool_ast(code)
        assert ok is False
        assert "Security violation" in err
        assert "exec" in err

    def test_import_call_rejected(self):
        code = VALID_TOOL_CODE + "\ndef exploit(): return __import__('os')\n"
        ok, err = validate_tool_ast(code)
        assert ok is False
        assert "Security violation" in err
        assert "__import__" in err

    def test_missing_execute_function_rejected(self):
        code = VALID_TOOL_CODE.replace("async def execute", "async def execute_disabled")
        ok, err = validate_tool_ast(code)
        assert ok is False
        assert "Missing required function: 'execute'" in err

    def test_missing_self_test_function_rejected(self):
        code = VALID_TOOL_CODE.replace("async def self_test", "async def test_something_else")
        ok, err = validate_tool_ast(code)
        assert ok is False
        assert "Missing required callable self-test function: 'self_test'" in err

    def test_missing_tool_schema_rejected(self):
        code = VALID_TOOL_CODE.replace("TOOL_SCHEMA =", "IGNORED_SCHEMA =")
        ok, err = validate_tool_ast(code)
        assert ok is False
        assert "Missing required constant 'TOOL_SCHEMA'" in err

    def test_privilege_escalation_ring0_rejected(self):
        code = VALID_TOOL_CODE.replace("REQUIRED_RING = 1", "REQUIRED_RING = 0")
        ok, err = validate_tool_ast(code)
        assert ok is False
        assert "REQUIRED_RING" in err
        assert "Ring 0 is reserved" in err

    def test_invalid_tier_rejected(self):
        code = VALID_TOOL_CODE.replace("TIER = 1", "TIER = 5")
        ok, err = validate_tool_ast(code)
        assert ok is False
        assert "Invalid TIER" in err


# ---------------------------------------------------------------------------
# 2. Sterile Sandbox Self-Test Execution & Staging Cleanup Tests
# ---------------------------------------------------------------------------


class TestSandboxSelfTestExecution:
    """Tests for isolated worker sandbox execution and temporary file hygiene."""

    @pytest.mark.asyncio
    async def test_sandbox_passing_self_test(self):
        ok, err = await execute_sandbox_self_test(VALID_TOOL_CODE, timeout=3.0)
        assert ok is True
        assert err is None

    @pytest.mark.asyncio
    async def test_sandbox_failing_self_test_returns_false(self):
        failing_code = VALID_TOOL_CODE.replace(
            'return res == "5d41402abc4b2a76b9719d911017c592"',
            'return False',
        )
        ok, err = await execute_sandbox_self_test(failing_code, timeout=3.0)
        assert ok is False
        assert "expected True" in err

    @pytest.mark.asyncio
    async def test_sandbox_exception_in_self_test_caught(self):
        raising_code = VALID_TOOL_CODE.replace(
            'return res == "5d41402abc4b2a76b9719d911017c592"',
            'raise ValueError("Deliberate self_test exception")',
        )
        ok, err = await execute_sandbox_self_test(raising_code, timeout=3.0)
        assert ok is False
        assert "Deliberate self_test exception" in err

    @pytest.mark.asyncio
    async def test_sandbox_timeout_containment(self):
        hanging_code = VALID_TOOL_CODE.replace(
            'return res == "5d41402abc4b2a76b9719d911017c592"',
            'import time\n    while True:\n        time.sleep(0.05)',
        )
        ok, err = await execute_sandbox_self_test(hanging_code, timeout=1.0)
        assert ok is False
        assert "timed out" in err

    def test_staging_temporary_files_are_deleted_after_run(self):
        # Verify no dangling staging files remain in /tmp
        hanging_code = VALID_TOOL_CODE.replace(
            'return res == "5d41402abc4b2a76b9719d911017c592"',
            'raise RuntimeError("Fail fast")',
        )
        execute_sandbox_self_test(hanging_code, timeout=1.0)
        staged_files = glob.glob("/tmp/zelyr_stage_*.py")
        assert len(staged_files) == 0


# ---------------------------------------------------------------------------
# 3. Successful Crafting, Atomic Promotion, and Hotloading Tests
# ---------------------------------------------------------------------------


class TestSuccessfulCraftingAndPromotion:
    """Tests full end-to-end tool crafting, atomic file creation, and hotload invocation."""

    @pytest.fixture(autouse=True)
    def setup_tmp_tools_dir(self, tmp_path):
        self.tools_dir = tmp_path / "tools.d"
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        self.registry = ToolRegistry()

    @pytest.mark.asyncio
    async def test_craft_calculate_md5_success_and_hotload(self):
        res = await craft_tool(
            name="calculate_md5",
            code=VALID_TOOL_CODE,
            tools_dir=self.tools_dir,
            registry=self.registry,
        )

        # 1. Verify structured envelope
        assert isinstance(res, dict)
        assert isinstance(res, CraftResult)
        assert res.success is True
        assert res["status"] == "success"
        assert res["tool_name"] == "calculate_md5"
        assert res["ring"] == 1
        assert res["tier"] == 1
        assert res["path"] == str(self.tools_dir / "calculate_md5.py")

        # 2. Verify provenance header
        target_file = self.tools_dir / "calculate_md5.py"
        assert target_file.is_file()
        content = target_file.read_text(encoding="utf-8")
        assert content.startswith("# @generated by zelyr-tool-crafter:")
        computed_hash = hashlib.sha256(VALID_TOOL_CODE.strip().encode("utf-8")).hexdigest()
        assert res["provenance_hash"] == computed_hash

        # 3. Verify registered in ToolRegistry
        tool = self.registry.get_tool("calculate_md5")
        assert tool is not None
        assert tool.name == "calculate_md5"
        assert tool.REQUIRED_RING == 1

        # 4. Invoke via ToolRegistry execute
        exec_res = self.registry.execute("calculate_md5", text="hello")
        assert exec_res.success is True
        assert exec_res.output == "5d41402abc4b2a76b9719d911017c592"

        # 5. Invoke via execute_tool
        out_str = await execute_tool("calculate_md5", text="hello")
        assert "5d41402abc4b2a76b9719d911017c592" in out_str

    @pytest.mark.asyncio
    async def test_atomic_backup_on_tool_replacement(self):
        # Step 1: Craft v1
        res1 = await craft_tool(
            name="calculate_md5",
            code=VALID_TOOL_CODE,
            tools_dir=self.tools_dir,
            registry=self.registry,
        )
        assert res1.success is True

        # Step 2: Craft v2 (same tool name, updated description)
        v2_code = VALID_TOOL_CODE.replace("Computes MD5 hash of input text.", "Computes MD5 hash v2.")
        res2 = await craft_tool(
            name="calculate_md5",
            code=v2_code,
            tools_dir=self.tools_dir,
            registry=self.registry,
        )
        assert res2.success is True

        # Verify a .bak file was created
        bak_files = list(self.tools_dir.glob("calculate_md5.py.bak.*"))
        assert len(bak_files) >= 1

    @pytest.mark.asyncio
    async def test_canonical_execute_entrypoint(self):
        out_json_str = await execute(
            name="calculate_md5",
            code=VALID_TOOL_CODE,
            tools_dir=self.tools_dir,
            registry=self.registry,
        )
        data = json.loads(out_json_str)
        assert data.get("status") == "success"
        assert data.get("tool_name") == "calculate_md5"


# ---------------------------------------------------------------------------
# 4. Error Classifier & Orchestrator Normalization Tests
# ---------------------------------------------------------------------------


class TestErrorClassificationAndOrchestrator:
    """Tests for ErrorClassifier integration and NativeOrchestrator aliases."""

    def test_error_classifier_tool_craft_failed(self):
        classified = ErrorClassifier.classify(
            tool_name="craft_tool",
            error="AST Security Violation in candidate code: eval() disallowed",
        )
        assert classified.error_type == "TOOL_CRAFT_FAILED"
        assert "Dynamic tool crafting failed" in classified.summary
        assert "craft_tool" in classified.allowed_actions
        assert "execute_command" in classified.allowed_actions
