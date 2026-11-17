"""Adversarial regression tests for custom tool_executor trust boundaries and test isolation."""

from __future__ import annotations

import json
from typing import Any, Dict, List
import pytest

from zelyr.agents.fallback_parser import FallbackToolParser
from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.agents.swarm.delegation import DelegatedSubagent, SubagentProfile
from zelyr.core.plugins import execute_tool
from zelyr.eval import EvalStatus
from zelyr.eval.harness import EvalHarness
from zelyr.eval.tasks.deterministic_p1 import build_task_tool_dispatch
from zelyr.eval.tasks.model_tasks_p4 import build_task_autonomous_runtime_bug_patching
from zelyr.eval.tasks.model_tasks_p5 import build_task_mcp_server_gateway_federation
from zelyr.tool_registry import (
    TaskLocalToolScope,
    ToolRegistry,
    authorize_and_validate_tool_call,
    bind_task_local_executor,
)
from tests.test_tool_permission_parity import _build_test_registry, _native_call_stream


def _fallback_call_stream(tool_name: str, arguments: Dict[str, Any]):
    async def _provider(_payload: Dict[str, Any]):
        block = json.dumps({"name": tool_name, "arguments": arguments})
        yield {"choices": [{"delta": {"content": f"```json\n{block}\n```"}}]}

    return _provider


@pytest.mark.asyncio
class TestCustomExecutorIsolation:
    async def test_1_production_custom_executor_rejects_unknown_tool_without_metadata(self):
        """1. A production custom executor cannot execute an unknown tool without authorization metadata."""
        invocation_count = 0

        async def prod_custom_executor(name: str, args: Dict[str, Any]):
            nonlocal invocation_count
            invocation_count += 1
            return {"status": "executed", "tool": name, "args": args}

        # Native call with no tool_registry, no tool_schemas, no caller_ring, no allowed_tools
        orch_native = NativeOrchestrator(
            stream_provider=_native_call_stream("unregistered_custom_tool", {"target": "prod"}),
            tool_executor=prod_custom_executor,
            max_steps=1,
        )
        payload_native = {"messages": [{"role": "user", "content": "Run custom tool"}]}
        async for _ in orch_native.generate_stream(payload_native):
            pass

        assert invocation_count == 0
        tool_msgs = [m for m in payload_native["messages"] if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        body = json.loads(tool_msgs[0]["content"])
        assert body["status"] == "rejected"
        assert body["reason"] == "tool_unavailable"

        # Fallback-recovered call with no metadata must also be rejected before reaching prod_custom_executor
        orch_fb = NativeOrchestrator(
            stream_provider=_fallback_call_stream("unregistered_custom_tool", {"target": "prod"}),
            tool_executor=prod_custom_executor,
            max_steps=1,
        )
        payload_fb = {"messages": [{"role": "user", "content": "Run custom tool via markdown"}]}
        async for _ in orch_fb.generate_stream(payload_fb):
            pass

        assert invocation_count == 0

    async def test_2_production_custom_executor_cannot_bypass_allowed_tools(self):
        """2. A production custom executor cannot bypass allowed_tools (including canonical alias rewrites)."""
        invocation_count = 0

        async def prod_custom_executor(name: str, args: Dict[str, Any]):
            nonlocal invocation_count
            invocation_count += 1
            return {"status": "executed"}

        # Direct disallowed registered tool without explicit tool_registry
        orch_direct = NativeOrchestrator(
            stream_provider=_native_call_stream("execute_command", {"command": "echo forbidden"}),
            tool_executor=prod_custom_executor,
            allowed_tools=["manage_workspace_file"],
            max_steps=1,
        )
        payload_direct = {"messages": [{"role": "user", "content": "Run command"}]}
        async for _ in orch_direct.generate_stream(payload_direct):
            pass

        assert invocation_count == 0
        tool_msgs = [m for m in payload_direct["messages"] if m.get("role") == "tool"]
        assert tool_msgs
        assert json.loads(tool_msgs[0]["content"])["reason"] == "authorization_denied"

        # Canonical rewrite (interact_with_ui + element_id -> interact_with_browser) blocked by allowed_tools
        reg, _ = _build_test_registry()
        orch_alias = NativeOrchestrator(
            tool_registry=reg,
            stream_provider=_native_call_stream("interact_with_ui", {"action": "click_element", "element_id": 5}),
            tool_executor=prod_custom_executor,
            allowed_tools=["interact_with_ui"],
            max_steps=1,
        )
        payload_alias = {"messages": [{"role": "user", "content": "Click element 5"}]}
        async for _ in orch_alias.generate_stream(payload_alias):
            pass

        assert invocation_count == 0
        alias_msgs = [m for m in payload_alias["messages"] if m.get("role") == "tool"]
        assert alias_msgs
        assert json.loads(alias_msgs[0]["content"])["reason"] == "authorization_denied"

    async def test_3_production_custom_executor_cannot_bypass_caller_ring(self):
        """3. A production custom executor cannot bypass caller_ring."""
        invocation_count = 0

        async def prod_custom_executor(name: str, args: Dict[str, Any]):
            nonlocal invocation_count
            invocation_count += 1
            return {"status": "executed"}

        # Without explicit tool_registry: execute_command is Ring 0, caller_ring is 1
        orch = NativeOrchestrator(
            stream_provider=_native_call_stream("execute_command", {"command": "id"}),
            tool_executor=prod_custom_executor,
            caller_ring=1,
            max_steps=1,
        )
        payload = {"messages": [{"role": "user", "content": "Check id"}]}
        async for _ in orch.generate_stream(payload):
            pass

        assert invocation_count == 0
        tool_msgs = [m for m in payload["messages"] if m.get("role") == "tool"]
        assert tool_msgs
        body = json.loads(tool_msgs[0]["content"])
        assert body["status"] == "rejected"
        assert body["reason"] == "authorization_denied"

    async def test_4_invalid_or_missing_production_authorization_metadata_fails_closed(self):
        """4. Invalid or missing production authorization metadata fails closed."""
        invocation_count = 0

        async def prod_custom_executor(name: str, args: Dict[str, Any]):
            nonlocal invocation_count
            invocation_count += 1
            return {"status": "executed"}

        class _EmptyRegistry:
            def get_tool(self, _name: str):
                return None

            def get_schemas(self, compact: bool = False):
                return []

        invalid_configs: List[Dict[str, Any]] = [
            {"caller_ring": -1},
            {"caller_ring": 99},
            {"caller_ring": "ring_0"},  # type: ignore[dict-item]
            {"caller_ring": True},  # bool must not be accepted as int ring 1
            {"allowed_tools": "manage_workspace_file"},  # type: ignore[dict-item]  # bare string instead of collection
            {"allowed_tools": 123},  # type: ignore[dict-item]
            {"allowed_tools": [""]},
            {"tool_registry": "invalid_registry_object"},
            {"tool_registry": object()},
            {"tool_schemas": "not_a_schema_list"},  # type: ignore[dict-item]
            {"tool_schemas": [123]},  # type: ignore[dict-item]
            {"task_local_scope": True},
            {"task_local_scope": "test_mode"},
            {"task_local_scope": {"tool_names": ["manage_workspace_file"]}},
            {"tool_registry": _EmptyRegistry()},  # explicit registry missing the requested tool
        ]

        for cfg in invalid_configs:
            orch = NativeOrchestrator(
                stream_provider=_native_call_stream(
                    "manage_workspace_file",
                    {"action": "read", "path": "README.md"},
                ),
                tool_executor=prod_custom_executor,
                max_steps=1,
                **cfg,
            )
            payload = {"messages": [{"role": "user", "content": "Read file"}]}
            async for _ in orch.generate_stream(payload):
                pass

            assert invocation_count == 0, f"Executor was invoked under invalid config: {cfg}"
            tool_msgs = [m for m in payload["messages"] if m.get("role") == "tool"]
            assert tool_msgs, f"Expected rejection tool message for config: {cfg}"
            body = json.loads(tool_msgs[0]["content"])
            assert body["status"] == "rejected"
            assert body["reason"] in ("authorization_denied", "tool_unavailable")

    async def test_5_explicitly_configured_task_local_test_executors_work(self):
        """5. Explicitly configured task-local test executors still work."""
        invocations: List[tuple[str, Dict[str, Any]]] = []

        async def local_mock_executor(name: str, args: Dict[str, Any]):
            invocations.append((name, dict(args)))
            return {"status": "ok", "echo": args.get("payload")}

        scope = TaskLocalToolScope(
            tool_names=("synthetic_probe",),
            tool_schemas=(
                {
                    "type": "function",
                    "function": {
                        "name": "synthetic_probe",
                        "parameters": {
                            "type": "object",
                            "properties": {"payload": {"type": "string"}},
                            "required": ["payload"],
                        },
                    },
                },
            ),
            tool_rings={"synthetic_probe": 1},
        )

        # 5a. Via explicit task_local_scope on NativeOrchestrator
        orch = NativeOrchestrator(
            stream_provider=_native_call_stream("synthetic_probe", {"payload": "alpha"}),
            tool_executor=local_mock_executor,
            task_local_scope=scope,
            caller_ring=1,
            allowed_tools=["synthetic_probe"],
            max_steps=1,
        )
        payload = {"messages": [{"role": "user", "content": "Probe alpha"}]}
        async for _ in orch.generate_stream(payload):
            pass

        assert invocations == [("synthetic_probe", {"payload": "alpha"})]

        # 5b. Via bind_task_local_executor wrapper
        bound_exec = bind_task_local_executor(
            local_mock_executor,
            tool_names=("bound_local_tool",),
        )
        orch_bound = NativeOrchestrator(
            stream_provider=_native_call_stream("bound_local_tool", {"payload": "beta"}),
            tool_executor=bound_exec,
            max_steps=1,
        )
        payload_bound = {"messages": [{"role": "user", "content": "Probe beta"}]}
        async for _ in orch_bound.generate_stream(payload_bound):
            pass

        assert invocations[-1] == ("bound_local_tool", {"payload": "beta"})

    async def test_6_test_executor_cannot_gain_access_to_production_tool_registries(self):
        """6. A test executor cannot gain access to production tool registries."""
        invocation_count = 0
        prod_leak_results: List[Any] = []

        async def local_test_executor(name: str, args: Dict[str, Any]):
            nonlocal invocation_count
            invocation_count += 1
            # Attempt to reach production plugin executor and production ToolRegistry from inside test callback
            plugin_res = await execute_tool("execute_command", command="echo leaked")
            reg_res = ToolRegistry().execute("execute_command", command="echo leaked")
            prod_leak_results.append((json.loads(plugin_res), reg_res))
            return {"status": "local_ok"}

        scope = TaskLocalToolScope(tool_names=("isolated_test_op",))

        # 6a. Model attempting to invoke production tool (execute_command) under TaskLocalToolScope is rejected
        captured_advertised_tools: List[List[str]] = []

        async def _trying_prod_tool_stream(p: Dict[str, Any]):
            captured_advertised_tools.append([
                t.get("function", {}).get("name") or t.get("name")
                for t in (p.get("tools") or [])
            ])
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_prod_leak",
                            "type": "function",
                            "function": {
                                "name": "execute_command",
                                "arguments": json.dumps({"command": "whoami"}),
                            },
                        }]
                    }
                }]
            }

        orch_prod_attempt = NativeOrchestrator(
            stream_provider=_trying_prod_tool_stream,
            tool_executor=local_test_executor,
            task_local_scope=scope,
            max_steps=1,
        )
        payload_prod_attempt = {"messages": [{"role": "user", "content": "Run whoami"}]}
        async for _ in orch_prod_attempt.generate_stream(payload_prod_attempt):
            pass

        assert invocation_count == 0
        assert captured_advertised_tools == [[]]
        tool_msgs = [m for m in payload_prod_attempt["messages"] if m.get("role") == "tool"]
        assert tool_msgs
        assert json.loads(tool_msgs[0]["content"])["reason"] == "tool_unavailable"

        # 6b. Attempting to combine TaskLocalToolScope with a production ToolRegistry fails closed
        reg, _ = _build_test_registry()
        orch_combined = NativeOrchestrator(
            tool_registry=reg,
            stream_provider=_native_call_stream("isolated_test_op", {}),
            tool_executor=local_test_executor,
            task_local_scope=scope,
            max_steps=1,
        )
        payload_combined = {"messages": [{"role": "user", "content": "Run isolated op"}]}
        async for _ in orch_combined.generate_stream(payload_combined):
            pass

        assert invocation_count == 0

        # 6c. Attempting to use TaskLocalToolScope without a custom tool_executor fails closed (never calls production execute_tool)
        orch_no_exec = NativeOrchestrator(
            stream_provider=_native_call_stream("isolated_test_op", {}),
            task_local_scope=scope,
            max_steps=1,
        )
        payload_no_exec = {"messages": [{"role": "user", "content": "Run without custom executor"}]}
        async for _ in orch_no_exec.generate_stream(payload_no_exec):
            pass

        no_exec_msgs = [m for m in payload_no_exec["messages"] if m.get("role") == "tool"]
        assert no_exec_msgs
        assert json.loads(no_exec_msgs[0]["content"])["reason"] == "authorization_denied"

        # 6d. Even when an allowed task-local tool runs, callback attempts to invoke production registries are blocked
        orch_callback_escape = NativeOrchestrator(
            stream_provider=_native_call_stream("isolated_test_op", {}),
            tool_executor=local_test_executor,
            task_local_scope=scope,
            max_steps=1,
        )
        payload_escape = {"messages": [{"role": "user", "content": "Run allowed local op"}]}
        async for _ in orch_callback_escape.generate_stream(payload_escape):
            pass

        assert invocation_count == 1
        assert len(prod_leak_results) == 1
        plugin_attempt, reg_attempt = prod_leak_results[0]
        assert plugin_attempt["success"] is False
        assert plugin_attempt["error_type"] == "AUTHORIZATION_DENIED"
        assert reg_attempt.success is False
        assert reg_attempt.error_type == "AUTHORIZATION_DENIED"

    async def test_7_native_fallback_registry_and_direct_execution_share_boundary(self):
        """7. Native, fallback, registry, subagent, and direct execution continue using the shared authorization boundary."""
        reg, tools = _build_test_registry()

        # Direct boundary decision check
        dec = authorize_and_validate_tool_call(
            "interact_with_ui",
            {"action": "click_element", "element_id": 11},
            caller_ring=1,
            allowed_tools=["interact_with_ui"],
            tool_registry=reg,
        )
        assert not dec.allowed
        assert dec.tool_name == "interact_with_browser"
        assert dec.error_type == "AUTHORIZATION_DENIED"

        # Fallback parser rejects canonical alias smuggling
        fb_calls = FallbackToolParser.parse_tool_calls(
            '```json\n{"name": "interact_with_ui", "arguments": {"action": "click_element", "element_id": 11}}\n```',
            known_tools={"interact_with_ui", "interact_with_browser"},
            allowed_tools=["interact_with_ui"],
            caller_ring=0,
            tool_registry=reg,
        )
        assert fb_calls == []

        # Registry direct execute rejects canonical alias smuggling
        reg_res = reg.execute(
            "interact_with_ui",
            caller_ring=1,
            allowed_tools=["interact_with_ui"],
            action="click_element",
            element_id=11,
        )
        assert reg_res.success is False
        assert reg_res.error_type == "AUTHORIZATION_DENIED"
        assert tools["interact_with_browser"].calls == []
        assert tools["interact_with_ui"].calls == []

        # Subagent delegation execute_tool_call canonicalizes before allowlist check and fails closed on bad ring
        profile = SubagentProfile(
            role="coder",
            agent_name="CoderAgent",
            description="Test coder subagent",
            allowed_tools=["interact_with_ui", "fake_bad_ring"],
            system_prompt="Test prompt",
        )
        subagent = DelegatedSubagent(
            profile=profile,
            session_id="test_sub_isolation",
            tool_registry=reg,
            caller_ring=1,
        )
        sub_alias_out = await subagent.execute_tool_call(
            "interact_with_ui",
            {"action": "click_element", "element_id": 11},
        )
        assert "AuthorizationError" in sub_alias_out
        assert "interact_with_browser" in sub_alias_out

        sub_bad_ring_out = await subagent.execute_tool_call("fake_bad_ring", {})
        assert "AuthorizationError" in sub_bad_ring_out
        assert tools["fake_bad_ring"].calls == []

    async def test_8_existing_evaluation_tasks_retain_intended_behavior(self):
        """8. Existing evaluation tasks retain their intended behavior under explicit TaskLocalToolScope."""
        harness = EvalHarness()

        res_l1 = await harness.run_task(build_task_tool_dispatch())
        assert res_l1.status == EvalStatus.PASS
        assert res_l1.tool_calls == ["mock_read_file"]

        res_l4 = await harness.run_task(build_task_autonomous_runtime_bug_patching())
        assert res_l4.status == EvalStatus.PASS
        # Verify read_file is no longer rewritten to production manage_workspace_file
        assert res_l4.tool_calls == ["read_file", "write_file", "run_service_test"]

        res_l5 = await harness.run_task(build_task_mcp_server_gateway_federation())
        assert res_l5.status == EvalStatus.PASS
        assert res_l5.tool_calls == ["execute_mcp_federation"]
