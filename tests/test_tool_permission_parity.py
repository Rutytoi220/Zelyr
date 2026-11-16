"""Adversarial regression tests for authoritative native vs fallback tool permission parity."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from zelyr.agents.fallback_parser import FallbackToolParser
from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.core.plugins import _executors, _rings, _schemas, execute_tool, load_plugins
from zelyr.tool_registry import ToolRegistry
from zelyr.tools.core import BaseTool, ToolParameter, ToolResult


class _FakeAllowedUiTool(BaseTool):
    REQUIRED_RING = 1

    def __init__(self) -> None:
        super().__init__(
            tool_id="interact_with_ui",
            name="interact_with_ui",
            description="Fake Tier-3 UI tool",
        )
        self.add_parameter(ToolParameter(name="instruction", type="string", description="UI instruction", required=False))
        self.calls: List[Dict[str, Any]] = []

    def execute(self, params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> ToolResult:
        self.calls.append(dict(params or kwargs))
        return ToolResult(success=True, output="ui ok")


class _FakeBrowserTool(BaseTool):
    REQUIRED_RING = 0

    def __init__(self) -> None:
        super().__init__(
            tool_id="interact_with_browser",
            name="interact_with_browser",
            description="Fake Tier-2 browser tool",
        )
        self.add_parameter(ToolParameter(name="action", type="string", description="Browser action", required=True))
        self.add_parameter(ToolParameter(name="element_id", type="integer", description="Element ID", required=False))
        self.calls: List[Dict[str, Any]] = []

    def execute(self, params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> ToolResult:
        self.calls.append(dict(params or kwargs))
        return ToolResult(success=True, output="browser ok")


class _FakeInvalidRingTool(BaseTool):
    REQUIRED_RING = "invalid_ring_label"

    def __init__(self) -> None:
        super().__init__(
            tool_id="fake_bad_ring",
            name="fake_bad_ring",
            description="Tool with invalid ring metadata",
        )
        self.calls: List[Dict[str, Any]] = []

    def execute(self, params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> ToolResult:
        self.calls.append(dict(params or kwargs))
        return ToolResult(success=True, output="should not run")


class _FakeSafeFileTool(BaseTool):
    REQUIRED_RING = 1

    def __init__(self) -> None:
        super().__init__(
            tool_id="manage_workspace_file",
            name="manage_workspace_file",
            description="Fake workspace file tool",
        )
        self.add_parameter(ToolParameter(name="action", type="string", description="Action", required=True))
        self.add_parameter(ToolParameter(name="path", type="string", description="Path", required=True))
        self.calls: List[Dict[str, Any]] = []

    def execute(self, params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> ToolResult:
        self.calls.append(dict(params or kwargs))
        return ToolResult(success=True, output="file ok")


def _build_test_registry() -> tuple[ToolRegistry, Dict[str, Any]]:
    reg = ToolRegistry()
    ui_tool = _FakeAllowedUiTool()
    browser_tool = _FakeBrowserTool()
    bad_ring_tool = _FakeInvalidRingTool()
    file_tool = _FakeSafeFileTool()
    reg.register_tool("interact_with_ui", ui_tool)
    reg.register_tool("interact_with_browser", browser_tool)
    reg.register_tool("fake_bad_ring", bad_ring_tool)
    reg.register_tool("manage_workspace_file", file_tool)
    return reg, {
        "interact_with_ui": ui_tool,
        "interact_with_browser": browser_tool,
        "fake_bad_ring": bad_ring_tool,
        "manage_workspace_file": file_tool,
    }


def _native_call_stream(tool_name: str, arguments: Any, call_id: str = "call_nat_1"):
    async def _provider(_payload: Dict[str, Any]):
        yield {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
                        },
                    }]
                }
            }]
        }

    return _provider


@pytest.mark.asyncio
class TestNativeToolPermissionParity:
    async def test_native_call_excluded_by_allowed_tools_is_rejected(self):
        executed: List[tuple[str, Dict[str, Any]]] = []

        async def mock_executor(name: str, args: Dict[str, Any]):
            executed.append((name, args))
            return {"status": "success"}

        reg, _ = _build_test_registry()
        orch = NativeOrchestrator(
            tool_registry=reg,
            tool_executor=mock_executor,
            stream_provider=_native_call_stream("interact_with_browser", {"action": "get_page_snapshot"}),
            allowed_tools=["manage_workspace_file"],
            max_steps=1,
        )
        payload = {"messages": [{"role": "user", "content": "Inspect page"}]}
        async for _ in orch.generate_stream(payload):
            pass

        assert executed == []
        tool_msgs = [m for m in payload["messages"] if m.get("role") == "tool"]
        assert tool_msgs
        body = json.loads(tool_msgs[0]["content"])
        assert body["status"] == "rejected"
        assert body["reason"] == "authorization_denied"

    async def test_native_call_above_caller_ring_is_rejected(self):
        executed: List[tuple[str, Dict[str, Any]]] = []

        async def mock_executor(name: str, args: Dict[str, Any]):
            executed.append((name, args))
            return {"status": "success"}

        reg, _ = _build_test_registry()
        orch = NativeOrchestrator(
            tool_registry=reg,
            tool_executor=mock_executor,
            stream_provider=_native_call_stream("execute_command", {"command": "echo should_not_run"}),
            caller_ring=1,
            max_steps=1,
        )
        payload = {"messages": [{"role": "user", "content": "Run echo"}]}
        async for _ in orch.generate_stream(payload):
            pass

        assert executed == []
        tool_msgs = [m for m in payload["messages"] if m.get("role") == "tool"]
        assert tool_msgs
        body = json.loads(tool_msgs[0]["content"])
        assert body["status"] == "rejected"
        assert body["reason"] == "authorization_denied"

    async def test_native_call_with_invalid_arguments_is_rejected(self):
        executed: List[tuple[str, Dict[str, Any]]] = []

        async def mock_executor(name: str, args: Dict[str, Any]):
            executed.append((name, args))
            return {"status": "success"}

        reg, _ = _build_test_registry()
        orch = NativeOrchestrator(
            tool_registry=reg,
            tool_executor=mock_executor,
            # Missing required "path" argument for manage_workspace_file
            stream_provider=_native_call_stream("manage_workspace_file", {"action": "read"}),
            caller_ring=1,
            allowed_tools=["manage_workspace_file"],
            max_steps=1,
        )
        payload = {"messages": [{"role": "user", "content": "Read file"}]}
        async for _ in orch.generate_stream(payload):
            pass

        assert executed == []
        tool_msgs = [m for m in payload["messages"] if m.get("role") == "tool"]
        assert tool_msgs
        body = json.loads(tool_msgs[0]["content"])
        assert body["status"] == "rejected"
        assert body["reason"] == "schema_validation_error"

    async def test_native_call_cannot_regain_disallowed_essential_tool_via_schema_fallback(self):
        captured_tools: List[List[str]] = []
        executed: List[tuple[str, Dict[str, Any]]] = []

        async def capturing_stream(p: Dict[str, Any]):
            captured_tools.append([
                t.get("function", {}).get("name") or t.get("name")
                for t in (p.get("tools") or [])
            ])
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_ess_1",
                            "type": "function",
                            "function": {
                                "name": "execute_command",
                                "arguments": json.dumps({"command": "whoami"}),
                            },
                        }]
                    }
                }]
            }

        async def mock_executor(name: str, args: Dict[str, Any]):
            executed.append((name, args))
            return {"status": "success"}

        reg, _ = _build_test_registry()
        orch = NativeOrchestrator(
            tool_registry=reg,
            tool_executor=mock_executor,
            stream_provider=capturing_stream,
            caller_ring=1,
            allowed_tools=["manage_workspace_file"],
            max_steps=1,
        )
        payload = {"messages": [{"role": "user", "content": "Open browser and navigate"}]}
        async for _ in orch.generate_stream(payload):
            pass

        assert captured_tools
        assert "execute_command" not in captured_tools[0]
        assert executed == []

    async def test_valid_native_call_within_allowlist_and_ring_executes(self):
        executed: List[tuple[str, Dict[str, Any]]] = []

        async def mock_executor(name: str, args: Dict[str, Any]):
            executed.append((name, args))
            return {"status": "success", "content": "ok"}

        reg, _ = _build_test_registry()
        orch = NativeOrchestrator(
            tool_registry=reg,
            tool_executor=mock_executor,
            stream_provider=_native_call_stream("manage_workspace_file", {"action": "read", "path": "notes.txt"}),
            caller_ring=1,
            allowed_tools=["manage_workspace_file"],
            max_steps=1,
        )
        payload = {"messages": [{"role": "user", "content": "Read notes.txt"}]}
        async for _ in orch.generate_stream(payload):
            pass

        assert executed == [("manage_workspace_file", {"action": "read", "path": "notes.txt"})]


class TestFallbackPermissionParityAndFailClosedRings:
    def test_canonical_rewrite_checked_before_allowlist_in_fallback_parser(self):
        # interact_with_ui is allowed, interact_with_browser is NOT allowed
        text = '```json\n{"name": "interact_with_ui", "arguments": {"action": "click_element", "element_id": 7}}\n```'
        calls = FallbackToolParser.parse_tool_calls(
            text,
            known_tools={"interact_with_ui", "interact_with_browser"},
            allowed_tools=["interact_with_ui"],
            caller_ring=0,
        )
        assert calls == []

        # When interact_with_browser IS allowed and caller_ring permits it, it succeeds
        calls_ok = FallbackToolParser.parse_tool_calls(
            text,
            known_tools={"interact_with_ui", "interact_with_browser"},
            allowed_tools=["interact_with_browser"],
            caller_ring=0,
        )
        assert len(calls_ok) == 1
        assert calls_ok[0].name == "interact_with_browser"

    def test_ring_lookup_exception_fails_closed_in_fallback_parser(self):
        text = '```json\n{"name": "manage_workspace_file", "arguments": {"action": "read", "path": "a.txt"}}\n```'
        with patch("zelyr.tool_registry.ToolRegistry._get_tool_ring", side_effect=RuntimeError("ring db failure")):
            calls = FallbackToolParser.parse_tool_calls(
                text,
                known_tools={"manage_workspace_file"},
                caller_ring=1,
            )
        assert calls == []

    def test_unknown_or_invalid_ring_metadata_fails_closed(self):
        reg, tools = _build_test_registry()
        res_bad_ring = reg.execute("fake_bad_ring", caller_ring=1)
        assert res_bad_ring.success is False
        assert res_bad_ring.error_type == "AUTHORIZATION_DENIED"
        assert tools["fake_bad_ring"].calls == []

        # Unknown tool in fallback parser fails closed under caller_ring
        text = '```json\n{"name": "unregistered_secret_tool", "arguments": {}}\n```'
        calls = FallbackToolParser.parse_tool_calls(
            text,
            known_tools={"unregistered_secret_tool"},
            caller_ring=1,
        )
        assert calls == []


@pytest.mark.asyncio
class TestExecutionBoundaryEnforcement:
    async def test_native_canonical_rewrite_cannot_smuggle_disallowed_target(self):
        executed: List[tuple[str, Dict[str, Any]]] = []

        async def mock_executor(name: str, args: Dict[str, Any]):
            executed.append((name, args))
            return {"status": "success"}

        reg, _ = _build_test_registry()
        # interact_with_ui (Ring 1) is in allowed_tools, interact_with_browser (Ring 0) is NOT
        orch = NativeOrchestrator(
            tool_registry=reg,
            tool_executor=mock_executor,
            stream_provider=_native_call_stream("interact_with_ui", {"action": "click_element", "element_id": 4}),
            caller_ring=1,
            allowed_tools=["interact_with_ui"],
            max_steps=1,
        )
        payload = {"messages": [{"role": "user", "content": "Click button"}]}
        async for _ in orch.generate_stream(payload):
            pass

        assert executed == []
        tool_msgs = [m for m in payload["messages"] if m.get("role") == "tool"]
        assert tool_msgs
        body = json.loads(tool_msgs[0]["content"])
        assert body["status"] == "rejected"
        assert body["reason"] == "authorization_denied"

    async def test_direct_plugins_execute_tool_enforces_caller_ring_and_allowed_tools(self, tmp_path):
        load_plugins()
        marker = tmp_path / "ran_marker.txt"

        async def fake_priv_exec(**_kwargs: Any) -> str:
            from pathlib import Path

            Path(str(marker)).write_text("executed", encoding="utf-8")
            return "ran"

        _executors["fake_priv_plugin"] = fake_priv_exec
        _rings["fake_priv_plugin"] = 0
        _schemas.append({"type": "function", "function": {"name": "fake_priv_plugin", "parameters": {"type": "object", "properties": {}}}})
        try:
            res_ring = json.loads(await execute_tool("fake_priv_plugin", caller_ring=1))
            assert res_ring["success"] is False
            assert res_ring["error_type"] == "AUTHORIZATION_DENIED"
            assert not marker.exists()

            res_allow = json.loads(await execute_tool("fake_priv_plugin", allowed_tools=["manage_workspace_file"]))
            assert res_allow["success"] is False
            assert res_allow["error_type"] == "AUTHORIZATION_DENIED"
            assert not marker.exists()

            # Legacy unrestricted call still works
            res_ok = await execute_tool("fake_priv_plugin")
            assert res_ok == "ran"
            assert marker.read_text(encoding="utf-8") == "executed"
        finally:
            _executors.pop("fake_priv_plugin", None)
            _rings.pop("fake_priv_plugin", None)
            _schemas[:] = [s for s in _schemas if s.get("function", {}).get("name") != "fake_priv_plugin"]

    async def test_unknown_tool_rejected_across_native_and_direct_boundaries(self):
        executed: List[tuple[str, Dict[str, Any]]] = []

        async def mock_executor(name: str, args: Dict[str, Any]):
            executed.append((name, args))
            return {"status": "success"}

        reg, _ = _build_test_registry()
        orch = NativeOrchestrator(
            tool_registry=reg,
            tool_executor=mock_executor,
            stream_provider=_native_call_stream("nonexistent_evil_tool", {"x": 1}),
            max_steps=1,
        )
        payload = {"messages": [{"role": "user", "content": "Run unknown tool"}]}
        async for _ in orch.generate_stream(payload):
            pass

        assert executed == []
        tool_msgs = [m for m in payload["messages"] if m.get("role") == "tool"]
        assert tool_msgs
        body = json.loads(tool_msgs[0]["content"])
        assert body["status"] == "rejected"
        assert body["reason"] == "tool_unavailable"

        direct_res = await execute_tool("nonexistent_evil_tool")
        direct_parsed = json.loads(direct_res)
        assert direct_parsed["success"] is False
        assert direct_parsed["error_type"] == "TOOL_NOT_FOUND"

    async def test_registry_canonical_rewrite_cannot_bypass_allowlist_on_direct_execute(self):
        reg, tools = _build_test_registry()
        res = reg.execute(
            "interact_with_ui",
            caller_ring=1,
            allowed_tools=["interact_with_ui"],
            action="click_element",
            element_id=9,
        )
        assert res.success is False
        assert res.error_type == "AUTHORIZATION_DENIED"
        assert tools["interact_with_browser"].calls == []
        assert tools["interact_with_ui"].calls == []

    async def test_repeated_calls_do_not_bypass_authorization(self):
        executed: List[tuple[str, Dict[str, Any]]] = []
        turns = 0

        async def repeat_stream(_payload: Dict[str, Any]):
            nonlocal turns
            turns += 1
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": f"call_rep_{turns}",
                            "type": "function",
                            "function": {
                                "name": "interact_with_browser",
                                "arguments": json.dumps({"action": "get_page_snapshot"}),
                            },
                        }]
                    }
                }]
            }

        async def mock_executor(name: str, args: Dict[str, Any]):
            executed.append((name, args))
            return {"status": "success"}

        reg, _ = _build_test_registry()
        orch = NativeOrchestrator(
            tool_registry=reg,
            tool_executor=mock_executor,
            stream_provider=repeat_stream,
            caller_ring=1,
            allowed_tools=["manage_workspace_file"],
            max_steps=3,
        )
        payload = {"messages": [{"role": "user", "content": "Inspect page repeatedly"}]}
        async for _ in orch.generate_stream(payload):
            pass

        assert executed == []
