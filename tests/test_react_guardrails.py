"""Test Suite: ReAct Protocol Conformance, Cycle Detection, and Subprocess Lifecycle Hardening.

Covers:
1. Wire payload conformance: rejected/blocked tool calls produce valid role: 'tool' envelopes
   with matching tool_call_id, name, and parsed status='rejected'.
2. Cycle detection Rule 1: immediately retrying an identical failed tool call is blocked.
3. Cycle detection Rule 2: 3rd identical invocation is blocked (call ceiling).
4. Cycle detection Rule 3: alternating loop (A -> B -> A -> B) with error outcomes is blocked.
5. Step ceiling enforcement: max_steps enforcement on actual tool executions, forcing a synthesis
   turn with tools=[] (tools disabled).
6. Subprocess lifecycle: execute_command with a process tree surviving timeout is fully reaped
   via process group termination without surviving active or zombie processes.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import uuid
import pytest

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.core.plugins import execute_tool, load_plugins
from zelyr.engine.observation_framing import ObservationFramingEngine


def _parse_msg(content: str) -> dict:
    try:
        return json.loads(content)
    except Exception:
        return ObservationFramingEngine.parse_observation(content)


class TestWirePayloadConformance:
    """Verifies that rejected/blocked tool calls conform strictly to OpenAI / wire tool response schema."""

    @pytest.mark.asyncio
    async def test_rejected_tool_call_wire_schema(self):
        orch = NativeOrchestrator()
        orch.max_steps = 1

        call_count = 0

        async def mock_stream(payload: dict):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "id": "call_wire_step_1",
                                "type": "function",
                                "function": {"name": "test_echo", "arguments": json.dumps({"msg": "hello"})},
                            }]
                        }
                    }]
                }
            elif call_count == 2:
                # Step limit reached -> this call should be rejected
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "id": "call_wire_step_2",
                                "type": "function",
                                "function": {"name": "test_echo", "arguments": json.dumps({"msg": "second"})},
                            }]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Final answer without tools."}}]}

        async def mock_exec(name: str, **kwargs):
            return {"status": "ok", "msg": kwargs.get("msg")}

        payload = {"messages": [{"role": "user", "content": "Execute steps"}]}
        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=mock_exec,
            tool_schemas=[{"type": "function", "function": {"name": "test_echo"}}],
        ):
            pass

        # Inspect messages for the rejected tool call envelope
        tool_messages = [m for m in payload["messages"] if m.get("role") == "tool"]
        assert len(tool_messages) >= 2

        rejected_msg = tool_messages[1]
        assert rejected_msg["role"] == "tool"
        assert rejected_msg["tool_call_id"] == "call_wire_step_2"
        assert rejected_msg["name"] == "test_echo"

        parsed_content = json.loads(rejected_msg["content"])
        assert parsed_content["tool"] == "test_echo"
        assert parsed_content["status"] == "rejected"
        assert parsed_content["reason"] == "step_limit_reached"
        assert "detail" in parsed_content


class TestCycleDetectionGuardrails:
    """Verifies turn-scoped cycle detection rules (Rule 1, Rule 2, Rule 3)."""

    @pytest.mark.asyncio
    async def test_rule_1_retry_after_error_blocked(self):
        """Immediately retrying an identical failed tool call is blocked."""
        orch = NativeOrchestrator()
        orch.max_steps = 5

        call_count = 0

        async def mock_stream(payload: dict):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "id": f"call_retry_{call_count}",
                                "type": "function",
                                "function": {"name": "flaky_tool", "arguments": json.dumps({"param": "foo"})},
                            }]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Handled failure."}}]}

        async def mock_exec(name: str, **kwargs):
            return "Error: destination unreachable"

        payload = {"messages": [{"role": "user", "content": "Try failing tool"}]}
        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=mock_exec,
            tool_schemas=[{"type": "function", "function": {"name": "flaky_tool"}}],
        ):
            pass

        tool_messages = [m for m in payload["messages"] if m.get("role") == "tool"]
        assert len(tool_messages) == 2

        # First call was executed and returned error
        call_1_content = _parse_msg(tool_messages[0]["content"])
        assert call_1_content["status"] == "error"

        # Second call was blocked by cycle detection rule 1
        call_2_content = _parse_msg(tool_messages[1]["content"])
        assert call_2_content["status"] == "rejected"
        assert call_2_content["reason"] == "cycle_detected"
        assert "retry after immediate failure" in call_2_content["detail"]

    @pytest.mark.asyncio
    async def test_rule_2_identical_call_ceiling(self):
        """3rd identical invocation is blocked regardless of success status."""
        orch = NativeOrchestrator()
        orch.max_steps = 5

        call_count = 0

        async def mock_stream(payload: dict):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "id": f"call_ceiling_{call_count}",
                                "type": "function",
                                "function": {"name": "repeat_tool", "arguments": json.dumps({"param": "bar"})},
                            }]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Ceiling reached, summarizing."}}]}

        async def mock_exec(name: str, **kwargs):
            return {"status": "ok", "param": kwargs.get("param")}

        payload = {"messages": [{"role": "user", "content": "Repeat call"}]}
        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=mock_exec,
            tool_schemas=[{"type": "function", "function": {"name": "repeat_tool"}}],
        ):
            pass

        tool_messages = [m for m in payload["messages"] if m.get("role") == "tool"]
        assert len(tool_messages) == 3

        call_1_content = _parse_msg(tool_messages[0]["content"])
        call_2_content = _parse_msg(tool_messages[1]["content"])
        call_3_content = _parse_msg(tool_messages[2]["content"])

        assert call_1_content["status"] == "success"
        assert call_2_content["status"] == "success"
        assert call_3_content["status"] == "rejected"
        assert call_3_content["reason"] == "cycle_detected"
        assert "ceiling reached" in call_3_content["detail"]

    @pytest.mark.asyncio
    async def test_rule_3_alternating_failure_loop_blocked(self):
        """Alternating failure cycle (A -> B -> A -> B) is blocked on second B."""
        orch = NativeOrchestrator()
        orch.max_steps = 6

        sequence = ["tool_a", "tool_b", "tool_a", "tool_b"]
        call_idx = 0

        async def mock_stream(payload: dict):
            nonlocal call_idx
            if call_idx < len(sequence):
                t_name = sequence[call_idx]
                call_idx += 1
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "id": f"call_alt_{call_idx}",
                                "type": "function",
                                "function": {"name": t_name, "arguments": json.dumps({"mode": "test"})},
                            }]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Cycle broken, synthesized."}}]}

        async def mock_exec(name: str, **kwargs):
            return "Error: temporary outage"

        payload = {"messages": [{"role": "user", "content": "Alternating test"}]}
        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=mock_exec,
            tool_schemas=[
                {"type": "function", "function": {"name": "tool_a"}},
                {"type": "function", "function": {"name": "tool_b"}},
            ],
        ):
            pass

        tool_messages = [m for m in payload["messages"] if m.get("role") == "tool"]
        assert len(tool_messages) == 4

        # 1st A: error
        assert _parse_msg(tool_messages[0]["content"])["status"] == "error"
        # 1st B: error
        assert _parse_msg(tool_messages[1]["content"])["status"] == "error"
        # 2nd A: error
        assert _parse_msg(tool_messages[2]["content"])["status"] == "error"
        # 2nd B: rejected by Rule 3
        call_4_content = _parse_msg(tool_messages[3]["content"])
        assert call_4_content["status"] == "rejected"
        assert call_4_content["reason"] == "cycle_detected"
        assert "Alternating failure cycle detected" in call_4_content["detail"]

    @pytest.mark.asyncio
    async def test_rule_1_active_at_low_step_limits_max_steps_2(self):
        """Cycle detection Rule 1 triggers even when max_steps <= 3 (no eff_step_limit bypass)."""
        orch = NativeOrchestrator(max_steps=2, enable_cycle_detection=True)

        call_count = 0

        async def mock_stream(payload: dict):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "id": f"call_low_{call_count}",
                                "type": "function",
                                "function": {"name": "failing_op", "arguments": json.dumps({"action": "retry"})},
                            }]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Terminated cleanly."}}]}

        async def mock_exec(name: str, **kwargs):
            return "Error: low-level driver failure"

        payload = {"messages": [{"role": "user", "content": "Execute failing op"}]}
        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=mock_exec,
            tool_schemas=[{"type": "function", "function": {"name": "failing_op"}}],
        ):
            pass

        tool_messages = [m for m in payload["messages"] if m.get("role") == "tool"]
        assert len(tool_messages) == 2

        # 1st call failed
        assert _parse_msg(tool_messages[0]["content"])["status"] == "error"
        # 2nd call was rejected by Rule 1 (cycle_detected), NOT deferred to step limit
        call_2_content = _parse_msg(tool_messages[1]["content"])
        assert call_2_content["status"] == "rejected"
        assert call_2_content["reason"] == "cycle_detected"
        assert "retry after immediate failure" in call_2_content["detail"]

    @pytest.mark.asyncio
    async def test_rule_2_active_at_max_steps_3(self):
        """Cycle detection Rule 2 (call ceiling) triggers on 3rd identical call with max_steps=3."""
        orch = NativeOrchestrator(max_steps=3, enable_cycle_detection=True)

        call_count = 0

        async def mock_stream(payload: dict):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "id": f"call_ceiling3_{call_count}",
                                "type": "function",
                                "function": {"name": "read_state", "arguments": "{}"},
                            }]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Summarizing."}}]}

        async def mock_exec(name: str, **kwargs):
            return {"status": "ok", "state": "idle"}

        payload = {"messages": [{"role": "user", "content": "Check idle state"}]}
        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=mock_exec,
            tool_schemas=[{"type": "function", "function": {"name": "read_state"}}],
        ):
            pass

        tool_messages = [m for m in payload["messages"] if m.get("role") == "tool"]
        assert len(tool_messages) == 3

        # First 2 succeeded
        assert _parse_msg(tool_messages[0]["content"])["status"] == "success"
        assert _parse_msg(tool_messages[1]["content"])["status"] == "success"
        # 3rd was rejected by Rule 2 (ceiling reached)
        call_3_content = _parse_msg(tool_messages[2]["content"])
        assert call_3_content["status"] == "rejected"
        assert call_3_content["reason"] == "cycle_detected"
        assert "ceiling reached" in call_3_content["detail"]

    @pytest.mark.asyncio
    async def test_cycle_detection_fingerprint_matches_normalized_floats(self):
        """Cycle detection catches retry when arguments differ only by float formatting or key ordering."""
        orch = NativeOrchestrator(max_steps=5, enable_cycle_detection=True)

        call_count = 0

        async def mock_stream(payload: dict):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Call 1: {"count": 1.0, "label": "test"}
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "id": "call_norm_1",
                                "type": "function",
                                "function": {"name": "norm_tool", "arguments": '{"count": 1.0, "label": "test"}'},
                            }]
                        }
                    }]
                }
            elif call_count == 2:
                # Call 2: {"label": "test", "count": 1} (reversed key order, integer 1 instead of 1.0)
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "id": "call_norm_2",
                                "type": "function",
                                "function": {"name": "norm_tool", "arguments": {"label": "test", "count": 1}},
                            }]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Normalized handled."}}]}

        async def mock_exec(name: str, **kwargs):
            return "Error: failed operation"

        payload = {"messages": [{"role": "user", "content": "Test normalization"}]}
        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=mock_exec,
            tool_schemas=[{"type": "function", "function": {"name": "norm_tool"}}],
        ):
            pass

        tool_messages = [m for m in payload["messages"] if m.get("role") == "tool"]
        assert len(tool_messages) == 2

        assert _parse_msg(tool_messages[0]["content"])["status"] == "error"
        # Call 2 matched fingerprint despite 1.0 vs 1 and key reordering -> blocked by Rule 1
        call_2_content = _parse_msg(tool_messages[1]["content"])
        assert call_2_content["status"] == "rejected"
        assert call_2_content["reason"] == "cycle_detected"

    @pytest.mark.asyncio
    async def test_cycle_detection_disabled_allows_identical_retries(self):
        """When enable_cycle_detection=False, identical failing calls proceed until step limit."""
        orch = NativeOrchestrator(enable_cycle_detection=False)
        orch.max_steps = 3

        call_count = 0

        async def mock_stream(payload: dict):
            nonlocal call_count
            call_count += 1
            if call_count <= 4:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "id": f"call_bypass_{call_count}",
                                "type": "function",
                                "function": {"name": "raw_tool", "arguments": "{}"},
                            }]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Done."}}]}

        executed_count = 0

        async def mock_exec(name: str, **kwargs):
            nonlocal executed_count
            executed_count += 1
            return "Error: simulated failure"

        payload = {"messages": [{"role": "user", "content": "Execute without cycle detection"}]}
        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=mock_exec,
            tool_schemas=[{"type": "function", "function": {"name": "raw_tool"}}],
        ):
            pass

        # All 3 steps executed the tool despite identical failing retries
        assert executed_count == 3
        tool_messages = [m for m in payload["messages"] if m.get("role") == "tool"]
        assert len(tool_messages) == 4
        # 4th call was rejected because step limit 3 was reached, not cycle_detected
        assert json.loads(tool_messages[3]["content"])["reason"] == "step_limit_reached"


class TestArgumentCanonicalization:
    """Verifies deterministic JSON argument canonicalization and type normalization."""

    def test_canonicalize_key_sorting_and_formatting(self):
        norm1 = NativeOrchestrator._canonicalize_tool_args({"z": 100, "a": 1, "m": 50})
        norm2 = NativeOrchestrator._canonicalize_tool_args('{"a": 1, "m": 50, "z": 100}')
        assert norm1 == '{"a":1,"m":50,"z":100}'
        assert norm2 == '{"a":1,"m":50,"z":100}'
        assert norm1 == norm2

    def test_canonicalize_integer_valued_floats(self):
        norm_float = NativeOrchestrator._canonicalize_tool_args({"val": 42.0, "fraction": 42.5})
        norm_int = NativeOrchestrator._canonicalize_tool_args({"val": 42, "fraction": 42.5})
        assert norm_float == '{"fraction":42.5,"val":42}'
        assert norm_float == norm_int

    def test_canonicalize_boolean_preserved_not_coerced_to_int(self):
        norm_bool = NativeOrchestrator._canonicalize_tool_args({"active": True, "count": 1.0})
        norm_int = NativeOrchestrator._canonicalize_tool_args({"active": 1, "count": 1})
        assert norm_bool == '{"active":true,"count":1}'
        assert norm_int == '{"active":1,"count":1}'
        assert norm_bool != norm_int

    def test_canonicalize_empty_and_null_inputs(self):
        assert NativeOrchestrator._canonicalize_tool_args(None) == "{}"
        assert NativeOrchestrator._canonicalize_tool_args("") == "{}"
        assert NativeOrchestrator._canonicalize_tool_args("   ") == "{}"
        assert NativeOrchestrator._canonicalize_tool_args({}) == "{}"
        assert NativeOrchestrator._canonicalize_tool_args("{}") == "{}"

    def test_canonicalize_nested_structures(self):
        nested = {"level1": {"items": [1.0, 2.0, {"sub": 3.0}]}}
        res = NativeOrchestrator._canonicalize_tool_args(nested)
        assert res == '{"level1":{"items":[1,2,{"sub":3}]}}'


class TestStepCeilingEnforcement:
    """Verifies that reaching max_steps blocks additional tool calls and forces synthesis turn."""

    @pytest.mark.asyncio
    async def test_step_ceiling_and_forced_synthesis_turn(self):
        orch = NativeOrchestrator()
        orch.max_steps = 2

        call_count = 0
        received_tools_counts = []

        async def mock_stream(payload: dict):
            nonlocal call_count
            received_tools_counts.append(len(payload.get("tools", [])))
            call_count += 1
            if call_count <= 3:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "id": f"call_step_{call_count}",
                                "type": "function",
                                "function": {"name": f"step_tool_{call_count}", "arguments": "{}"},
                            }]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Final synthesis without tools."}}]}

        async def mock_exec(name: str, **kwargs):
            return {"status": "ok"}

        payload = {"messages": [{"role": "user", "content": "Execute 3 distinct steps"}]}
        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=mock_exec,
            tool_schemas=[
                {"type": "function", "function": {"name": "step_tool_1"}},
                {"type": "function", "function": {"name": "step_tool_2"}},
                {"type": "function", "function": {"name": "step_tool_3"}},
            ],
        ):
            pass

        tool_messages = [m for m in payload["messages"] if m.get("role") == "tool"]
        assert len(tool_messages) == 3

        # First 2 succeeded
        assert _parse_msg(tool_messages[0]["content"])["status"] == "success"
        assert _parse_msg(tool_messages[1]["content"])["status"] == "success"

        # 3rd tool call rejected by step ceiling
        call_3_content = _parse_msg(tool_messages[2]["content"])
        assert call_3_content["status"] == "rejected"
        assert call_3_content["reason"] == "step_limit_reached"

        # 4th stream call must have received 0 tools (forced synthesis turn)
        assert len(received_tools_counts) >= 4
        assert received_tools_counts[3] == 0

    @pytest.mark.asyncio
    async def test_rejection_increments_step_count_and_enforces_ceiling(self):
        """Verifies that synthetic rejection increments actual_step_count, halting runaway rejection loops."""
        orch = NativeOrchestrator(enable_cycle_detection=True)
        orch.max_steps = 2

        call_count = 0
        received_tools_counts = []

        async def mock_stream(payload: dict):
            nonlocal call_count
            tools_count = len(payload.get("tools", []))
            received_tools_counts.append(tools_count)
            call_count += 1
            if tools_count > 0:
                # Repeatedly attempt identical failing tool calls while tools are enabled
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "id": f"call_reject_loop_{call_count}",
                                "type": "function",
                                "function": {"name": "failing_op", "arguments": json.dumps({"q": 1})},
                            }]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Forced synthesis completion."}}]}

        async def mock_exec(name: str, **kwargs):
            return "Error: failed"

        payload = {"messages": [{"role": "user", "content": "Trigger loop"}]}
        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=mock_exec,
            tool_schemas=[{"type": "function", "function": {"name": "failing_op"}}],
        ):
            pass

        tool_messages = [m for m in payload["messages"] if m.get("role") == "tool"]
        # Step 1: executed -> error (step count = 1)
        # Step 2: rejected by cycle detection (step count = 2 -> ceiling reached)
        # Turn 3: disable_tools=True -> tools=[], model must synthesize
        assert len(tool_messages) == 2
        assert _parse_msg(tool_messages[0]["content"])["status"] == "error"
        assert _parse_msg(tool_messages[1]["content"])["status"] == "rejected"
        assert _parse_msg(tool_messages[1]["content"])["reason"] == "cycle_detected"

        # Tools disabled on turn 3
        assert len(received_tools_counts) >= 3
        assert received_tools_counts[2] == 0


class TestSubprocessLifecycle:
    """Verifies that execute_command reaps detached process trees on timeout."""

    @pytest.mark.asyncio
    async def test_timeout_reaps_process_group_completely(self):
        load_plugins()

        marker = "zelyr_probe_" + uuid.uuid4().hex[:8]
        # Command forks a child process that sleeps in the background
        fork_cmd = f'python3 -c "import time, os; os.fork(); time.sleep(30)" # {marker}'

        try:
            result = await execute_tool("execute_command", command=fork_cmd, timeout=1.0)
            assert "timed out after" in result
        finally:
            # Query ps strictly for our unique marker
            ps = subprocess.run(["ps", "-eo", "pid,ppid,stat,args"], capture_output=True, text=True)
            current_pid = os.getpid()
            surviving = [
                line for line in ps.stdout.splitlines()
                if marker in line and str(current_pid) not in line and "ps -eo" not in line
            ]

            # In case any survived, clean up in finally to ensure no orphan persists
            for line in surviving:
                parts = line.strip().split()
                if parts:
                    try:
                        os.kill(int(parts[0]), 9)
                    except OSError:
                        pass

            assert len(surviving) == 0, f"Expected 0 surviving processes, found: {surviving}"
