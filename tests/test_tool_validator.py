"""Unit and integration test suite for ToolValidator, isolated staging, and tool crafting resilience."""

import glob
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from zelyr.core.plugins import get_tool_schemas, load_plugins, reload_plugin
from zelyr.db.memory import MemoryStore
from zelyr.memory.capability_indexer import index_tool_capabilities
from zelyr.tools.tool_crafter import craft_tool
from zelyr.tools.tool_manager import ToolManager, delete_dynamic_tool, update_dynamic_tool
from zelyr.tools.validator import ToolValidator

VALID_TOOL_CODE = '''
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "sample_calc_tool",
        "description": "Calculates the sum of two integers.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First number"},
                "b": {"type": "integer", "description": "Second number"}
            },
            "required": ["a", "b"]
        }
    }
}
REQUIRED_RING = 0
TIER = 1
TUI_HINT = "🧮 Calculating sum..."

async def execute(a: int = 0, b: int = 0, **kwargs) -> str:
    return str(a + b)
'''

SAMPLE_TOOL_V1 = '''
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "resilient_adder",
        "description": "Adds numbers v1.",
        "parameters": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Number to add"}
            },
            "required": ["n"]
        }
    }
}
REQUIRED_RING = 1
TIER = 1
TUI_HINT = "Adding..."

async def execute(n: int = 0, **kwargs) -> str:
    return str(n + 1)
'''

SAMPLE_TOOL_V2 = '''
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "resilient_adder",
        "description": "Adds numbers v2 updated.",
        "parameters": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Number to double"}
            },
            "required": ["n"]
        }
    }
}
REQUIRED_RING = 1
TIER = 1
TUI_HINT = "Doubling..."

async def execute(n: int = 0, **kwargs) -> str:
    return str(n * 2)
'''


class TestToolValidator(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.validator = ToolValidator()

    def test_1_valid_tool_passes_ast_and_schema_and_ring(self):
        """Verify standard compliant tool passes syntax, AST, and contract validation."""
        ast_res = self.validator.validate_syntax_and_ast(VALID_TOOL_CODE)
        self.assertTrue(ast_res.is_valid, f"AST validation failed: {ast_res.error}")
        self.assertEqual(ast_res.tier, 1)
        self.assertEqual(ast_res.ring, 0)

        ring_ok, ring_err = self.validator.validate_ring_conformance(0)
        self.assertTrue(ring_ok)
        self.assertIsNone(ring_err)

        ring_3_ok, _ = self.validator.validate_ring_conformance(3)
        self.assertTrue(ring_3_ok)

        schema = {
            "type": "function",
            "function": {
                "name": "test_echo",
                "description": "Echo input text.",
                "parameters": {
                    "type": "object",
                    "properties": {"msg": {"type": "string"}},
                    "required": ["msg"],
                },
            },
        }
        schema_ok, schema_err = self.validator.validate_schema(schema)
        self.assertTrue(schema_ok, f"Schema validation failed: {schema_err}")

    async def test_2_valid_tool_passes_isolated_staging_and_validate_all(self):
        """Verify full validate_all pipeline completes successfully on valid tool."""
        res = await self.validator.validate_all(VALID_TOOL_CODE, test_params={"a": 5, "b": 10})
        self.assertTrue(res.is_valid, f"Validation failed: {res.error}")
        self.assertEqual(res.tool_name, "sample_calc_tool")
        self.assertEqual(res.tier, 1)
        self.assertEqual(res.ring, 0)
        self.assertIsNotNone(res.schema)

    def test_3_syntax_error_caught_with_line_number(self):
        """Verify syntax error is caught with line number and col, aborting before staging."""
        broken_code = "import json\n\ndef broken_function(\n    return 42\n"
        res = self.validator.validate_syntax_and_ast(broken_code)
        self.assertFalse(res.is_valid)
        self.assertIn("Syntax error at line", res.error)

    def test_4_missing_required_exports(self):
        """Verify missing contract exports produce clear errors."""
        code_no_hint = '''
TOOL_SCHEMA = {"type": "function", "function": {"name": "test", "description": "test", "parameters": {"type": "object", "properties": {}}}}
REQUIRED_RING = 1
TIER = 1
async def execute(**kwargs): return "ok"
'''
        res = self.validator.validate_syntax_and_ast(code_no_hint)
        self.assertFalse(res.is_valid)
        self.assertIn("Missing required top-level export(s)", res.error)
        self.assertIn("TUI_HINT", res.error)

        code_no_exec = '''
TOOL_SCHEMA = {"type": "function", "function": {"name": "test", "description": "test", "parameters": {"type": "object", "properties": {}}}}
REQUIRED_RING = 1
TIER = 1
TUI_HINT = "Testing..."
'''
        res_exec = self.validator.validate_syntax_and_ast(code_no_exec)
        self.assertFalse(res_exec.is_valid)
        self.assertIn("Missing required async function: 'execute'", res_exec.error)

        code_sync_exec = '''
TOOL_SCHEMA = {"type": "function", "function": {"name": "test", "description": "test", "parameters": {"type": "object", "properties": {}}}}
REQUIRED_RING = 1
TIER = 1
TUI_HINT = "Testing..."
def execute(**kwargs): return "ok"
'''
        res_sync = self.validator.validate_syntax_and_ast(code_sync_exec)
        self.assertFalse(res_sync.is_valid)
        self.assertIn("must be an asynchronous function (async def)", res_sync.error)

    def test_5_ring_conformance_checks(self):
        """Verify ring validation rejects negative, out-of-bounds, or non-integer values."""
        ok_neg, err_neg = self.validator.validate_ring_conformance(-1)
        self.assertFalse(ok_neg)
        self.assertIn("must conform to Zelyr ring hierarchy", err_neg)

        ok_high, err_high = self.validator.validate_ring_conformance(4)
        self.assertFalse(ok_high)
        self.assertIn("must conform to Zelyr ring hierarchy", err_high)

        ok_bool, err_bool = self.validator.validate_ring_conformance(True)
        self.assertFalse(ok_bool)
        self.assertIn("must be an integer", err_bool)

        ok_str, err_str = self.validator.validate_ring_conformance("0")
        self.assertFalse(ok_str)
        self.assertIn("must be an integer", err_str)

    def test_6_malformed_tool_schema(self):
        """Verify malformed schemas fail validation cleanly."""
        bad_name_schema = {
            "type": "function",
            "function": {
                "name": "",
                "description": "test",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        ok, err = self.validator.validate_schema(bad_name_schema)
        self.assertFalse(ok)
        self.assertIn("function.name' must be a non-empty string", err)

        bad_props_schema = {
            "type": "function",
            "function": {
                "name": "valid_name",
                "description": "test",
                "parameters": {"type": "object"},
            },
        }
        ok, err = self.validator.validate_schema(bad_props_schema)
        self.assertFalse(ok)
        self.assertIn("properties' dictionary", err)

        bad_type_schema = {
            "type": "function",
            "function": {
                "name": "valid_name",
                "description": "test",
                "parameters": {"type": "string", "properties": {}},
            },
        }
        ok, err = self.validator.validate_schema(bad_type_schema)
        self.assertFalse(ok)
        self.assertIn("type' must be 'object'", err)

    async def test_7_runtime_exception_during_staging_fails_and_cleans_up(self):
        """Verify runtime exception during staging fails validation and cleans up scratch files."""
        failing_code = '''
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "crash_tool",
        "description": "Throws error at runtime.",
        "parameters": {"type": "object", "properties": {}}
    }
}
REQUIRED_RING = 0
TIER = 1
TUI_HINT = "Crashing..."

async def execute(**kwargs) -> str:
    raise ValueError("Simulated runtime failure in execute()")
'''
        before_scratch = set(glob.glob("/tmp/zelyr_stage_*.py"))
        res = await self.validator.validate_all(failing_code)
        self.assertFalse(res.is_valid)
        self.assertIn("Simulated runtime failure in execute()", res.error)

        after_scratch = set(glob.glob("/tmp/zelyr_stage_*.py"))
        diff = after_scratch - before_scratch
        self.assertEqual(len(diff), 0, f"Lingering scratch files detected in /tmp: {diff}")

    def test_8_tier_validation(self):
        """Verify TIER must be an integer in {1, 2, 3}."""
        code_bad_tier = '''
TOOL_SCHEMA = {"type": "function", "function": {"name": "test", "description": "test", "parameters": {"type": "object", "properties": {}}}}
REQUIRED_RING = 0
TIER = 5
TUI_HINT = "Testing..."
async def execute(**kwargs): return "ok"
'''
        res = self.validator.validate_syntax_and_ast(code_bad_tier)
        self.assertFalse(res.is_valid)
        self.assertIn("Invalid TIER '5'", res.error)


class TestToolCraftingResilience(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.tools_dir = Path(self.tmp_dir) / "tools.d"
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(self.tmp_dir) / "test_memory.db"

        os.environ["ZELYR_TOOLS_DIR"] = str(self.tools_dir)
        os.environ["ZELYR_MEMORY_DB"] = str(self.db_path)

        self.store = MemoryStore(db_path=self.db_path)
        load_plugins()

    async def asyncTearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        if "ZELYR_MEMORY_DB" in os.environ:
            del os.environ["ZELYR_MEMORY_DB"]
        if "ZELYR_TOOLS_DIR" in os.environ:
            del os.environ["ZELYR_TOOLS_DIR"]

    async def test_new_tool_reload_failure_rolls_back_file_and_runtime(self):
        """Simulate a new tool where reload_plugin() fails, verifying rollback."""
        tool_file = self.tools_dir / "failed_new_tool.py"

        with patch("zelyr.tools.tool_crafter.reload_plugin", side_effect=RuntimeError("Simulated reload failure")):
            res = await craft_tool(
                name="failed_new_tool",
                code=SAMPLE_TOOL_V1.replace("resilient_adder", "failed_new_tool"),
                test_params={"n": 5},
                tools_dir=self.tools_dir,
                db_path=self.db_path,
            )

        self.assertFalse(res.success)
        self.assertIn("Installation failed during reload", res.error)
        self.assertFalse(tool_file.exists())

        schemas = get_tool_schemas(0)
        schema_names = {s.get("function", {}).get("name") or s.get("name") for s in schemas}
        self.assertNotIn("failed_new_tool", schema_names)

        active_cap = self.store.get_by_key("tool.failed_new_tool")
        self.assertIsNone(active_cap)

    async def test_update_reload_failure_restores_previous_working_version(self):
        """Install v1, attempt v2 update where reload fails, verify restoration of v1."""
        craft_res = await craft_tool(
            name="resilient_adder",
            code=SAMPLE_TOOL_V1,
            test_params={"n": 5},
            tools_dir=self.tools_dir,
            db_path=self.db_path,
        )
        self.assertTrue(craft_res.success)

        tool_file = self.tools_dir / "resilient_adder.py"
        self.assertTrue(tool_file.exists())
        self.assertIn("Adds numbers v1.", tool_file.read_text(encoding="utf-8"))

        cap_v1 = self.store.get_by_key("tool.resilient_adder")
        self.assertIsNotNone(cap_v1)
        self.assertIn("Adds numbers v1.", cap_v1["content"])

        real_reload = reload_plugin
        call_count = 0

        def failing_reload(path):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated reload error for v2")
            return real_reload(path)

        with patch("zelyr.tools.tool_manager.reload_plugin", side_effect=failing_reload):
            up_res = await update_dynamic_tool(
                name="resilient_adder",
                code=SAMPLE_TOOL_V2,
                test_params={"n": 5},
                tools_dir=self.tools_dir,
                db_path=self.db_path,
            )

        self.assertFalse(up_res.success)
        self.assertIn("Update failed during reload", up_res.error)

        current_code = tool_file.read_text(encoding="utf-8")
        self.assertIn("Adds numbers v1.", current_code)
        self.assertNotIn("Adds numbers v2 updated.", current_code)

        schemas = get_tool_schemas(0)
        matching = [s for s in schemas if (s.get("function", {}).get("name") or s.get("name")) == "resilient_adder"]
        self.assertEqual(len(matching), 1)
        self.assertIn("Adds numbers v1.", matching[0]["function"]["description"])

        cap_restored = self.store.get_by_key("tool.resilient_adder")
        self.assertIsNotNone(cap_restored)
        self.assertIn("Adds numbers v1.", cap_restored["content"])

    async def test_manual_deletion_reconciliation(self):
        """Unlink tool file directly from disk; verify index_tool_capabilities reconciles MemoryStore."""
        res = await craft_tool(
            name="manual_del_tool",
            code=SAMPLE_TOOL_V1.replace("resilient_adder", "manual_del_tool"),
            test_params={"n": 1},
            tools_dir=self.tools_dir,
            db_path=self.db_path,
        )
        self.assertTrue(res.success)

        cap = self.store.get_by_key("tool.manual_del_tool")
        self.assertIsNotNone(cap)

        # Also store non-capability memories to verify they are never touched
        self.store.store(content="Dark theme preferred", category="preference", key="ui.theme")
        self.store.store(content="192.168.1.1 gateway", category="homelab", key="network.gw")

        tool_file = self.tools_dir / "manual_del_tool.py"
        self.assertTrue(tool_file.exists())
        tool_file.unlink()
        self.assertFalse(tool_file.exists())

        index_tool_capabilities(store=self.store, tools_dir=self.tools_dir)

        self.assertIsNone(self.store.get_by_key("tool.manual_del_tool"))

        with self.store._get_conn() as conn:
            row = conn.execute("SELECT is_active FROM memories WHERE id = ?", (cap["id"],)).fetchone()
            self.assertEqual(row["is_active"], 0)

        search_res = self.store.search("manual_del_tool", category="capability")
        self.assertEqual(len(search_res), 0)

        # Assert user preferences and homelab records were untouched
        self.assertIsNotNone(self.store.get_by_key("ui.theme"))
        self.assertIsNotNone(self.store.get_by_key("network.gw"))

    async def test_normal_deletion_through_tool_manager_remains_correct(self):
        """Verify normal deletion via ToolManager.delete_tool removes file and cleans runtime and memory."""
        res = await craft_tool(
            name="managed_del_tool",
            code=SAMPLE_TOOL_V1.replace("resilient_adder", "managed_del_tool"),
            test_params={"n": 1},
            tools_dir=self.tools_dir,
            db_path=self.db_path,
        )
        self.assertTrue(res.success)

        del_res = ToolManager.delete_tool("managed_del_tool", tools_dir=self.tools_dir, db_path=self.db_path)
        self.assertTrue(del_res.success)

        self.assertFalse((self.tools_dir / "managed_del_tool.py").exists())
        schemas = get_tool_schemas(0)
        self.assertNotIn("managed_del_tool", [s.get("function", {}).get("name") or s.get("name") for s in schemas])
        self.assertIsNone(self.store.get_by_key("tool.managed_del_tool"))

    async def test_restart_reindex_does_not_revive_deleted_tools(self):
        """Verify running index_tool_capabilities multiple times never revives deleted tools."""
        await craft_tool(
            name="revive_check_tool",
            code=SAMPLE_TOOL_V1.replace("resilient_adder", "revive_check_tool"),
            test_params={"n": 1},
            tools_dir=self.tools_dir,
            db_path=self.db_path,
        )
        delete_dynamic_tool("revive_check_tool", tools_dir=self.tools_dir, db_path=self.db_path)

        for _ in range(3):
            index_tool_capabilities(store=self.store, tools_dir=self.tools_dir)

        self.assertIsNone(self.store.get_by_key("tool.revive_check_tool"))
        search_res = self.store.search("revive_check_tool", category="capability")
        self.assertEqual(len(search_res), 0)


if __name__ == "__main__":
    unittest.main()
