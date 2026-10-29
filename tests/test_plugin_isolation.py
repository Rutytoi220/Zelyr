"""Isolated Regression Tests for Plugin Process Isolation and Resource Sandboxing."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from typing import Any, Dict, Optional
import pytest

from zelyr.core.plugin_sandbox import (
    DEFAULT_MAX_MEMORY_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    PluginExecutionError,
    SandboxedPluginRunner,
)
from zelyr.sdk.tool import ZelyrTool
from zelyr.tool_registry import ToolRegistry
from zelyr.tools.core import BaseTool, ToolResult


# --- Mock Plugins ---


class SafeMathPlugin(ZelyrTool):
    """Safe computational mock plugin."""

    @property
    def name(self) -> str:
        return "safe_math"

    @property
    def description(self) -> str:
        return "Performs safe integer multiplication"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        }

    async def execute(self, a: int = 0, b: int = 0, **kwargs: Any) -> Dict[str, Any]:
        await asyncio.sleep(0.01)
        return {"product": a * b, "echo": kwargs}


class StructuredResultPlugin(BaseTool):
    """Mock plugin returning a structured ToolResult directly."""

    def __init__(self) -> None:
        super().__init__(
            tool_id="structured_result_tool",
            name="structured_result_tool",
            description="Returns explicit ToolResult",
        )

    def execute(self, message: str = "ok", **kwargs: Any) -> ToolResult:
        return ToolResult(
            success=True,
            output=f"Processed: {message}",
            metadata={"source": "isolated_worker"},
        )


def crashing_segfault_plugin(*args: Any, **kwargs: Any) -> Any:
    """Mock plugin deliberately triggering a SIGSEGV signal."""
    os.kill(os.getpid(), signal.SIGSEGV)


def crashing_exit139_plugin(*args: Any, **kwargs: Any) -> Any:
    """Mock plugin terminating abruptly with status 139."""
    os._exit(139)


def crashing_sigabrt_plugin(*args: Any, **kwargs: Any) -> Any:
    """Mock plugin triggering a SIGABRT signal."""
    os.kill(os.getpid(), signal.SIGABRT)


def memory_hog_plugin(*args: Any, **kwargs: Any) -> Any:
    """Mock plugin attempting to allocate memory far exceeding sandbox limit."""
    # Attempt to allocate 300 MB of virtual memory
    blob = bytearray(300 * 1024 * 1024)
    return len(blob)


def infinite_sleep_plugin(*args: Any, **kwargs: Any) -> Any:
    """Mock plugin running an infinite blocking loop."""
    while True:
        time.sleep(0.1)


# --- Test Suite ---


class TestPluginProcessIsolation:
    """Tests covering isolated child process execution and crash containment."""

    def test_normal_sync_and_async_execution(self):
        """Phase 3.1: Normal execution of safe computation returns valid ToolResult."""
        # 1. Sync invocation of ZelyrTool
        tool = SafeMathPlugin()
        runner = SandboxedPluginRunner(tool, timeout=5.0)

        res_sync = runner.run(a=7, b=6)
        assert res_sync.success is True
        assert isinstance(res_sync.output, dict)
        assert res_sync.output.get("product") == 42

        # 2. Tool returning ToolResult directly
        struct_tool = StructuredResultPlugin()
        runner_struct = SandboxedPluginRunner(struct_tool, timeout=5.0)
        res_struct = runner_struct.run(message="hello sandbox")
        assert res_struct.success is True
        assert res_struct.output == "Processed: hello sandbox"
        assert res_struct.metadata.get("source") == "isolated_worker"

    @pytest.mark.asyncio
    async def test_normal_execution_async_unblocked(self):
        """Phase 3.1: Verify run_async executes without blocking event loop."""
        tool = SafeMathPlugin()
        runner = SandboxedPluginRunner(tool, timeout=5.0)

        # Execute concurrent tasks alongside the runner
        async def background_counter():
            count = 0
            for _ in range(5):
                await asyncio.sleep(0.01)
                count += 1
            return count

        bg_task = asyncio.create_task(background_counter())
        res = await runner.run_async(a=12, b=11)
        bg_count = await bg_task

        assert res.success is True
        assert res.output.get("product") == 132
        assert bg_count == 5

    def test_crash_containment_sigsegv(self):
        """Phase 3.2: Verify SIGSEGV is intercepted into structured failure envelope."""
        runner = SandboxedPluginRunner(crashing_segfault_plugin, name="segfault_tool", timeout=5.0)

        res = runner.run()

        assert res.success is False
        assert res.output == ""
        assert "PluginExecutionError" in res.error
        assert "terminated unexpectedly" in res.error
        assert "SIGSEGV" in res.error or "11" in res.error

    def test_crash_containment_exit_code_139(self):
        """Phase 3.2: Verify os._exit(139) is intercepted without crashing the test process."""
        runner = SandboxedPluginRunner(crashing_exit139_plugin, name="exit139_tool", timeout=5.0)

        res = runner.run()

        assert res.success is False
        assert res.output == ""
        assert "PluginExecutionError" in res.error
        assert "terminated unexpectedly" in res.error
        assert "139" in res.error

    def test_crash_containment_sigabrt(self):
        """Phase 3.2: Verify SIGABRT crash does not bring down the host process."""
        runner = SandboxedPluginRunner(crashing_sigabrt_plugin, name="sigabrt_tool", timeout=5.0)

        res = runner.run()

        assert res.success is False
        assert res.output == ""
        assert "PluginExecutionError" in res.error
        assert "terminated unexpectedly" in res.error
        assert "SIGABRT" in res.error or "6" in res.error

    def test_memory_ceiling_enforcement(self):
        """Phase 3.3: Verify memory limit (RLIMIT_AS) prevents excessive RAM allocation."""
        # Restrict sandbox to 64 MB virtual memory ceiling
        runner = SandboxedPluginRunner(
            memory_hog_plugin,
            name="memory_hog",
            max_memory_mb=64.0,
            timeout=5.0,
        )

        res = runner.run()

        assert res.success is False
        assert res.output == ""
        assert "PluginExecutionError" in res.error
        # Must indicate either Memory ceiling exceeded or crash containment on out-of-memory
        err_lower = res.error.lower()
        assert "memory" in err_lower or "terminated unexpectedly" in err_lower

    def test_execution_timeout_and_process_cleanup(self):
        """Phase 3.4: Verify execution timeout terminates child process cleanly."""
        timeout_budget = 0.25
        runner = SandboxedPluginRunner(
            infinite_sleep_plugin,
            name="sleep_loop",
            timeout=timeout_budget,
        )

        start_time = time.monotonic()
        res = runner.run()
        elapsed = time.monotonic() - start_time

        assert res.success is False
        assert res.output == ""
        assert "PluginExecutionError" in res.error
        assert "timed out after" in res.error
        # Verified duration is bounded by timeout plus escalation grace window (< 3.0s)
        assert elapsed < 3.0

    def test_hot_reloading_lifecycle(self):
        """Phase 3.5: Verify reloading a plugin refreshes the target without zombie processes."""
        def plugin_v1():
            return "version 1.0"

        def plugin_v2():
            return "version 2.0"

        runner = SandboxedPluginRunner(plugin_v1, name="versioned_tool", timeout=5.0)

        res1 = runner.run()
        assert res1.success is True
        assert res1.output == "version 1.0"

        # Hot reload with new target implementation
        runner.reload(new_target=plugin_v2)

        res2 = runner.run()
        assert res2.success is True
        assert res2.output == "version 2.0"

    @pytest.mark.asyncio
    async def test_tool_registry_integration_with_sandboxed_plugin(self):
        """Verify ToolRegistry execute_async seamlessly invokes SandboxedPluginRunner."""
        registry = ToolRegistry()
        tool = SafeMathPlugin()
        sandboxed = SandboxedPluginRunner(tool, timeout=5.0)

        registry.register_tool("sandboxed_math", sandboxed)

        res: ToolResult = await registry.execute_async("sandboxed_math", a=9, b=9)
        assert res.success is True
        assert res.output.get("product") == 81

        # Test crash containment through ToolRegistry
        crashing = SandboxedPluginRunner(crashing_segfault_plugin, name="segfault_tool")
        registry.register_tool("crashing_tool", crashing)

        res_crash: ToolResult = await registry.execute_async("crashing_tool")
        assert res_crash.success is False
        assert "PluginExecutionError" in res_crash.error

    @pytest.mark.asyncio
    async def test_dynamic_execute_tool_routing_and_isolation(self, tmp_path, monkeypatch):
        """Verify zelyr.core.plugins.execute_tool isolates dynamic plugin crashes."""
        import json
        from zelyr.core.plugins import execute_tool, _executors, _schemas

        # Register a crashing dynamic executor
        _executors["untrusted_dynamic_tool"] = crashing_sigabrt_plugin

        res_str = await execute_tool("untrusted_dynamic_tool")
        parsed = json.loads(res_str)

        assert parsed.get("success") is False
        assert "PluginExecutionError" in parsed.get("error", "")
        assert "SIGABRT" in parsed.get("error", "") or "6" in parsed.get("error", "")

        # Clean up
        _executors.pop("untrusted_dynamic_tool", None)
