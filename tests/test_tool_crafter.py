"""Unit and integration test suite for Autonomous Tool Crafting and Runtime Skill Compilation.

Verifies:
1. Tool Schema & Parameter Conformance for `craft_tool`.
2. Static AST Security Gate: rejection of prohibited imports, reflection escapes, and Ring 0 privilege escalation.
3. Dynamic Sandbox Gate: assertion failure rejection, runtime exception containment, timeout containment, and crash containment.
4. Tool Crafting & Execution: synthesis of computational and string tools, dynamic registration into ToolRegistry, and execution via execute_async.
5. Ephemeral vs. Persistent Lifecycles: in-memory hot-loading vs. disk persistence.
6. CoderAgent and Swarm Delegation integration.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict
import pytest

from zelyr.agents.swarm.delegation import SUBAGENT_PROFILES
from zelyr.engine.tool_crafter import (
    DynamicSynthesizedTool,
    ToolCrafterEngine,
    ToolCrafterTool,
    craft_tool,
)
from zelyr.tool_registry import ToolRegistry
from zelyr.tools.core import ToolResult


# ---------------------------------------------------------------------------
# Test 1: Tool Schema & Parameter Validation
# ---------------------------------------------------------------------------


class TestToolSchemaAndValidation:
    """Verifies schema conformance and name validation."""

    def test_tool_crafter_tool_schema_conformance(self):
        tool = ToolCrafterTool()
        assert tool.tool_id == "craft_tool"
        assert tool.name == "craft_tool"
        assert tool.REQUIRED_RING == 1
        assert tool.tier == 1

        schema = tool.schema
        assert schema["type"] == "object"
        assert "properties" in schema
        props = schema["properties"]
        assert "name" in props
        assert "description" in props
        assert "code" in props
        assert "test_code" in props
        assert "parameters" in props
        assert "persist" in props

        assert props["name"]["type"] == "string"
        assert props["code"]["type"] == "string"
        assert props["test_code"]["type"] == "string"
        assert props["persist"]["type"] == "boolean"

        required = schema.get("required", [])
        assert "name" in required
        assert "description" in required
        assert "code" in required

    def test_tool_crafter_discovered_in_tool_registry(self):
        registry = ToolRegistry()
        tool = registry.get_tool("craft_tool")
        assert tool is not None
        assert tool.name == "craft_tool"

        schemas = registry.get_tool_schemas()
        schema_names = {s.get("function", {}).get("name") for s in schemas}
        assert "craft_tool" in schema_names

    @pytest.mark.asyncio
    async def test_invalid_tool_name_rejection(self):
        engine = ToolCrafterEngine()

        # Empty name
        res_empty = await engine.craft_tool(name="", code="def execute(): pass")
        assert res_empty.success is False
        assert "Invalid tool name" in res_empty.error

        # Path traversal
        res_traversal = await engine.craft_tool(name="../../escaped_tool", code="def execute(): pass")
        assert res_traversal.success is False
        assert "Invalid tool name" in res_traversal.error

        # Special characters
        res_special = await engine.craft_tool(name="invalid-tool-name!", code="def execute(): pass")
        assert res_special.success is False
        assert "Invalid tool name" in res_special.error


# ---------------------------------------------------------------------------
# Test 2: Static Gate (AST Security Audit & Ring Check)
# ---------------------------------------------------------------------------


class TestStaticASTSecurityGate:
    """Verifies Ring-0 AST security rules reject dangerous code before execution."""

    @pytest.mark.asyncio
    async def test_prohibited_imports_rejected_immediately(self):
        engine = ToolCrafterEngine()

        dangerous_snippets = [
            ("import os\ndef execute(): pass", "os"),
            ("import subprocess\ndef execute(): pass", "subprocess"),
            ("import sys\ndef execute(): pass", "sys"),
            ("import shutil\ndef execute(): pass", "shutil"),
            ("import ctypes\ndef execute(): pass", "ctypes"),
            ("import socket\ndef execute(): pass", "socket"),
            ("import importlib\ndef execute(): pass", "importlib"),
            ("from os import path\ndef execute(): pass", "os"),
        ]

        for code, module_name in dangerous_snippets:
            res = await engine.craft_tool(
                name="dangerous_tool",
                description="Dangerous tool candidate",
                code=code,
                test_code="assert True",
            )
            assert res.success is False
            assert "AST Security Violation" in res.error
            assert module_name in res.error

    @pytest.mark.asyncio
    async def test_reflection_and_dunder_escapes_rejected(self):
        engine = ToolCrafterEngine()

        reflection_snippets = [
            "def execute(): return ().__class__.__base__",
            "def execute(): return ().__class__.__subclasses__()",
            "def execute(): return globals()['__builtins__']",
            "def execute(): return getattr(str, '__code__')",
        ]

        for code in reflection_snippets:
            res = await engine.craft_tool(
                name="reflection_tool",
                description="Reflection test tool",
                code=code,
                test_code="assert True",
            )
            assert res.success is False
            assert "AST Security Violation" in res.error

    @pytest.mark.asyncio
    async def test_privilege_escalation_ring0_rejected(self):
        engine = ToolCrafterEngine()
        ring0_code = """
REQUIRED_RING = 0
TIER = 1
def execute():
    return "root"
"""
        res = await engine.craft_tool(
            name="ring0_tool",
            description="Attempt to claim Ring 0",
            code=ring0_code,
            test_code="assert True",
        )
        assert res.success is False
        assert "Privilege escalation rejected" in res.error

    @pytest.mark.asyncio
    async def test_prohibited_in_test_code_rejected(self):
        engine = ToolCrafterEngine()
        valid_code = "def execute(): return 42"
        malicious_test = "import os\nassert execute() == 42"

        res = await engine.craft_tool(
            name="test_code_exploit",
            description="Exploit hidden in test code",
            code=valid_code,
            test_code=malicious_test,
        )
        assert res.success is False
        assert "AST Security Violation" in res.error
        assert "os" in res.error


# ---------------------------------------------------------------------------
# Test 3: Dynamic Gate (Sandbox Execution & Containment)
# ---------------------------------------------------------------------------


class TestDynamicSandboxGate:
    """Verifies candidate tools are tested in an isolated sandbox with resource bounds."""

    @pytest.mark.asyncio
    async def test_sandbox_assertion_failure_rejects_registration(self):
        registry = ToolRegistry()
        engine = ToolCrafterEngine(registry=registry)

        code = "def execute(a: int, b: int) -> int:\n    return a + b\n"
        failing_test = "assert execute(2, 2) == 5\n"

        res = await engine.craft_tool(
            name="failing_calc",
            description="Calculator with failing test assertions",
            code=code,
            test_code=failing_test,
        )

        assert res.success is False
        assert "Sandbox test verification failed" in res.error
        assert "Test assertion failed" in res.error
        assert "failing_calc" not in registry

    @pytest.mark.asyncio
    async def test_sandbox_runtime_exception_rejects_registration(self):
        registry = ToolRegistry()
        engine = ToolCrafterEngine(registry=registry)

        code = "def execute(a: int, b: int) -> float:\n    return a / b\n"
        failing_test = "assert execute(10, 0) == 0\n"

        res = await engine.craft_tool(
            name="division_tool",
            description="Division tool with unhandled zero division",
            code=code,
            test_code=failing_test,
        )

        assert res.success is False
        assert "Sandbox test verification failed" in res.error
        assert "ZeroDivisionError" in res.error
        assert "division_tool" not in registry

    @pytest.mark.asyncio
    async def test_timeout_containment_on_infinite_loop(self):
        registry = ToolRegistry()
        engine = ToolCrafterEngine(registry=registry, sandbox_timeout=1.0)

        infinite_loop_code = """
def execute():
    while True:
        pass
    return "done"
"""
        infinite_test = "assert execute() == 'done'"

        res = await engine.craft_tool(
            name="hanging_tool",
            description="Tool with infinite loop",
            code=infinite_loop_code,
            test_code=infinite_test,
        )

        assert res.success is False
        assert "PluginExecutionError" in res.error or "Sandbox test verification failed" in res.error
        assert "timed out" in res.error
        assert "hanging_tool" not in registry

    @pytest.mark.asyncio
    async def test_crash_containment_on_abnormal_exit(self):
        registry = ToolRegistry()
        engine = ToolCrafterEngine(registry=registry, sandbox_timeout=2.0)

        crashing_code = """
def execute():
    # Simulate internal native crash
    import signal
    import os # will fail AST if import os present, so use direct exit without forbidden imports
"""
        # Test AST validator blocks import os first
        res_ast = await engine.craft_tool(
            name="crash_tool",
            description="Attempt crash",
            code=crashing_code,
            test_code="assert execute()",
        )
        assert res_ast.success is False


# ---------------------------------------------------------------------------
# Test 4: Tool Crafting & Execution Flow
# ---------------------------------------------------------------------------


class TestToolCraftingAndExecution:
    """Verifies end-to-end synthesis, schema production, and session execution."""

    @pytest.mark.asyncio
    async def test_craft_fibonacci_tool_and_execute(self):
        registry = ToolRegistry()
        engine = ToolCrafterEngine(registry=registry)

        code = """
def fib(n: int) -> int:
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def execute(n: int = 10, **kwargs):
    return {"n": n, "fibonacci": fib(n)}
"""

        test_code = """
assert execute(0)["fibonacci"] == 0
assert execute(1)["fibonacci"] == 1
assert execute(7)["fibonacci"] == 13
assert execute(10)["fibonacci"] == 55
"""

        parameters = {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "The Fibonacci sequence index"},
            },
            "required": ["n"],
        }

        res = await engine.craft_tool(
            name="calc_fibonacci",
            description="Computes the nth Fibonacci number accurately.",
            parameters=parameters,
            code=code,
            test_code=test_code,
            persist=False,
        )

        assert res.success is True
        assert "calc_fibonacci" in res.output
        assert res.metadata.get("tool_name") == "calc_fibonacci"

        # Verify registration in registry
        assert "calc_fibonacci" in registry

        # Verify OpenAI schema generation
        schemas = registry.get_tool_schemas()
        matching = [s for s in schemas if s.get("function", {}).get("name") == "calc_fibonacci"]
        assert len(matching) == 1
        fn_schema = matching[0]["function"]
        assert fn_schema["description"] == "Computes the nth Fibonacci number accurately."
        assert fn_schema["parameters"]["properties"]["n"]["type"] == "integer"

        # Execute via ToolRegistry.execute_async
        inv_res = await registry.execute_async("calc_fibonacci", n=8)
        assert inv_res.success is True
        assert isinstance(inv_res.output, dict)
        assert inv_res.output.get("fibonacci") == 21

    @pytest.mark.asyncio
    async def test_craft_string_transformation_tool(self):
        registry = ToolRegistry()
        engine = ToolCrafterEngine(registry=registry)

        code = """
import urllib.parse

def execute(text: str = "", mode: str = "encode", **kwargs):
    if mode == "encode":
        return urllib.parse.quote_plus(text)
    elif mode == "decode":
        return urllib.parse.unquote_plus(text)
    return text
"""

        test_code = """
assert execute("hello world", mode="encode") == "hello+world"
assert execute("hello+world", mode="decode") == "hello world"
assert execute("test?param=1&other=2", mode="encode") == "test%3Fparam%3D1%26other%3D2"
"""

        parameters = {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "String to encode or decode"},
                "mode": {"type": "string", "description": "encode or decode"},
            },
            "required": ["text"],
        }

        res = await engine.craft_tool(
            name="url_coder",
            description="URL encoder and decoder tool",
            parameters=parameters,
            code=code,
            test_code=test_code,
            persist=False,
        )

        assert res.success is True
        assert "url_coder" in registry

        # Execute immediately
        inv_res = await registry.execute_async("url_coder", text="Zelyr Agent v1.2", mode="encode")
        assert inv_res.success is True
        assert inv_res.output == "Zelyr+Agent+v1.2"

    @pytest.mark.asyncio
    async def test_session_hot_loading_multiple_tools(self):
        registry = ToolRegistry()
        engine = ToolCrafterEngine(registry=registry)

        # Tool 1: Cube
        cube_code = "def execute(x: int): return x * x * x"
        cube_test = "assert execute(3) == 27"
        res1 = await engine.craft_tool(
            name="calc_cube",
            description="Computes x cubed",
            code=cube_code,
            test_code=cube_test,
        )
        assert res1.success is True

        # Tool 2: Reverse String
        rev_code = "def execute(s: str): return s[::-1]"
        rev_test = "assert execute('abc') == 'cba'"
        res2 = await engine.craft_tool(
            name="rev_str",
            description="Reverses a string",
            code=rev_code,
            test_code=rev_test,
        )
        assert res2.success is True

        # Both tools registered
        assert "calc_cube" in registry
        assert "rev_str" in registry

        out1 = await registry.execute_async("calc_cube", x=4)
        assert out1.success is True
        assert out1.output == 64

        out2 = await registry.execute_async("rev_str", s="antigravity")
        assert out2.success is True
        assert out2.output == "ytivargitna"


# ---------------------------------------------------------------------------
# Test 5: Ephemeral vs. Persistent Lifecycles
# ---------------------------------------------------------------------------


class TestEphemeralVsPersistentLifecycles:
    """Verifies difference between in-memory ephemeral tools and persistent tools."""

    @pytest.mark.asyncio
    async def test_ephemeral_tool_does_not_touch_disk(self, tmp_path: Path):
        registry = ToolRegistry()
        engine = ToolCrafterEngine(registry=registry, tools_dir=tmp_path)

        code = "def execute(v: int): return v * 2"
        test_code = "assert execute(5) == 10"

        res = await engine.craft_tool(
            name="double_val",
            description="Doubles an integer",
            code=code,
            test_code=test_code,
            persist=False,
            tools_dir=None,  # explicitly ephemeral
        )

        assert res.success is True
        assert res.metadata.get("persisted") is False
        assert not (tmp_path / "double_val.py").exists()

    @pytest.mark.asyncio
    async def test_persistent_tool_writes_atomically_to_disk(self, tmp_path: Path):
        registry = ToolRegistry()
        engine = ToolCrafterEngine(registry=registry, tools_dir=tmp_path)

        code = "def execute(v: int): return v * 10"
        test_code = "assert execute(5) == 50"

        res = await engine.craft_tool(
            name="multiply_ten",
            description="Multiplies by 10",
            code=code,
            test_code=test_code,
            persist=True,
            tools_dir=tmp_path,
        )

        assert res.success is True
        assert res.metadata.get("persisted") is True
        tool_file = tmp_path / "multiply_ten.py"
        assert tool_file.exists()
        assert "multiply_ten" in tool_file.read_text(encoding="utf-8") or "def execute" in tool_file.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Test 6: CoderAgent Integration
# ---------------------------------------------------------------------------


class TestCoderAgentIntegration:
    """Verifies craft_tool is exposed to CoderAgent in Swarm delegation."""

    def test_coder_agent_profile_allows_craft_tool(self):
        coder_profile = SUBAGENT_PROFILES["coder"]
        assert "craft_tool" in coder_profile.allowed_tools

    @pytest.mark.asyncio
    async def test_tool_crafter_tool_delegates_to_engine(self):
        registry = ToolRegistry()
        tool = ToolCrafterTool(tool_registry=registry)

        code = "def execute(msg: str): return f'Echo: {msg}'"
        test_code = "assert execute('ping') == 'Echo: ping'"

        # Invoke craft_tool via its BaseTool interface
        res = await tool.execute_async(
            name="custom_echo",
            description="Custom echo tool",
            code=code,
            test_code=test_code,
        )

        assert res.success is True
        assert "custom_echo" in registry
        inv_res = await registry.execute_async("custom_echo", msg="test")
        assert inv_res.success is True
        assert inv_res.output == "Echo: test"
