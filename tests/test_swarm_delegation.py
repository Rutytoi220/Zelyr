"""Comprehensive test suite for Hierarchical Multi-Agent Subagent Delegation and Blackboard Coordination.

Verifies:
1. Tool schema validation: JSON Schema compliance and parameter validation for `delegate_subtask`.
2. Successful multi-turn delegation: isolated ReAct execution writing to SwarmBlackboard.
3. Context isolation: parent orchestrator context is shielded from subagent internal message bloat.
4. Blackboard rollback on error: unhandled exceptions trigger atomic rollback with zero orphaned artifacts.
5. Timeout containment: hanging subagents timeout cleanly and trigger rollback without hanging parent.
6. Recursion depth ceiling: enforcement of max_depth=2 prevents runaway delegation trees.
7. Capability ring invariance: subagents cannot invoke tools exceeding session's permitted ring level.
8. Role profile tool curation: role-based tool gating and canonical alias resolution.
9. NativeOrchestrator integration: end-to-end delegation flow.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from zelyr.agents.base import AgentResult
from zelyr.agents.swarm.delegation import (
    DelegatedSubagent,
    SubagentDelegationTool,
    SubagentProfile,
    SubagentRole,
    SUBAGENT_PROFILES,
    _current_delegation_depth,
    get_subagent_profile,
)
from zelyr.memory.blackboard import BlackboardStore
from zelyr.tool_registry import BuiltinToolDef, ToolRegistry
from zelyr.tools.core import BaseTool, ToolResult


# ---------------------------------------------------------------------------
# Test 1: Tool Schema & Parameter Validation
# ---------------------------------------------------------------------------

class TestToolSchemaAndParameterValidation:
    def test_schema_conforms_to_json_schema(self):
        tool = SubagentDelegationTool()
        assert tool.tool_id == "delegate_subtask"
        assert tool.name == "delegate_subtask"
        assert tool.REQUIRED_RING == 1
        assert tool.tier == 1

        schema = tool.schema
        assert schema["type"] == "object"
        assert "properties" in schema
        props = schema["properties"]
        assert "role" in props
        assert "task" in props
        assert "context" in props
        assert "timeout_seconds" in props

        assert props["role"]["type"] == "string"
        assert props["task"]["type"] == "string"
        assert props["context"]["type"] == "object"
        assert props["timeout_seconds"]["type"] == "integer"

        required = schema.get("required", [])
        assert "role" in required
        assert "task" in required

    @pytest.mark.asyncio
    async def test_parameter_validation_rejects_empty_or_unknown_roles(self):
        tool = SubagentDelegationTool()

        # Empty role
        res = await tool.execute(role="", task="Do something")
        assert not res.success
        assert "InvalidRoleError" in res.error

        # Unknown role
        res = await tool.execute(role="space_pilot", task="Fly rocket")
        assert not res.success
        assert "InvalidRoleError" in res.error
        assert "Supported roles" in res.error

        # Empty task
        res = await tool.execute(role="coder", task="")
        assert not res.success
        assert "InvalidTaskError" in res.error


# ---------------------------------------------------------------------------
# Test 2: Successful Multi-Turn Delegation with Blackboard
# ---------------------------------------------------------------------------

class TestSuccessfulDelegationWithBlackboard:
    @pytest.mark.asyncio
    async def test_coder_subagent_executes_3_turns_and_commits_blackboard(self):
        bb = BlackboardStore()
        registry = ToolRegistry()
        session_id = "test_session_multi_turn"

        # Mock LLM simulating 3 ReAct turns:
        # Turn 1: subagent calls blackboard_write to post draft code
        # Turn 2: subagent calls echo to verify syntax
        # Turn 3: subagent produces final text summary
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            {
                "content": "Writing fibonacci implementation to blackboard.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "blackboard_write",
                            "arguments": {
                                "key": "draft_patch",
                                "value": "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)",
                            },
                        },
                    }
                ],
            },
            {
                "content": "Verifying code syntax with echo.",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "function": {
                            "name": "echo",
                            "arguments": {"message": "syntax_ok"},
                        },
                    }
                ],
            },
            {
                "content": "Successfully implemented fibonacci function and saved patch to blackboard.",
                "tool_calls": [],
            },
        ]

        tool = SubagentDelegationTool(
            blackboard=bb,
            tool_registry=registry,
            llm_client=mock_llm,
            session_id=session_id,
        )

        res = await tool.execute(
            role="coder",
            task="Implement a recursive fibonacci function and save to blackboard.",
            session_id=session_id,
        )

        assert res.success is True
        assert "Successfully implemented fibonacci" in res.output
        assert "draft_patch" in res.metadata["blackboard_keys"]
        assert res.metadata["role"] == "coder"
        assert res.metadata["agent_name"] == "CoderAgent"

        # Assert blackboard committed state
        patch_val = bb.read(session_id, "CoderAgent", "draft_patch")
        assert "def fib(n):" in patch_val
        final_val = bb.read(session_id, "CoderAgent", "final_output")
        assert "Successfully implemented fibonacci" in final_val


# ---------------------------------------------------------------------------
# Test 3: Context Isolation Between Parent and Subagent
# ---------------------------------------------------------------------------

class TestContextIsolation:
    @pytest.mark.asyncio
    async def test_parent_messages_remain_clean_and_unpolluted(self):
        bb = BlackboardStore()
        registry = ToolRegistry()
        session_id = "test_context_isolation"

        # Mock LLM generating 4 tool turns inside subagent
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            {
                "content": "Step 1: checking system.",
                "tool_calls": [{"id": "c1", "function": {"name": "echo", "arguments": {"message": "A"}}}],
            },
            {
                "content": "Step 2: checking files.",
                "tool_calls": [{"id": "c2", "function": {"name": "echo", "arguments": {"message": "B"}}}],
            },
            {
                "content": "Step 3: storing intermediate data.",
                "tool_calls": [{"id": "c3", "function": {"name": "blackboard_write", "arguments": {"key": "inter", "value": "xyz"}}}],
            },
            {
                "content": "Synthesized finding: All checks passed cleanly.",
                "tool_calls": [],
            },
        ]

        # Parent conversation context
        parent_messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "You are Zelyr main orchestrator."},
            {"role": "user", "content": "Please audit system integrity and report."},
        ]

        tool = SubagentDelegationTool(
            blackboard=bb,
            tool_registry=registry,
            llm_client=mock_llm,
            session_id=session_id,
        )

        # Parent calls delegate_subtask
        delegation_result = await tool.execute(
            role="general",
            task="Perform system integrity audit.",
            session_id=session_id,
        )

        # Parent appends delegation call and result
        parent_messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_delegate",
                "function": {
                    "name": "delegate_subtask",
                    "arguments": '{"role": "general", "task": "Perform system integrity audit."}',
                },
            }],
        })
        parent_messages.append({
            "role": "tool",
            "name": "delegate_subtask",
            "tool_call_id": "call_delegate",
            "content": delegation_result.output,
        })

        # Assert Parent Message History Invariants:
        # Parent contains exactly 4 messages (System, User, Assistant tool_call, Tool result)
        assert len(parent_messages) == 4
        # None of the subagent's intermediate tool calls ("echo A", "echo B") or scratchpad exist in parent
        parent_text = str(parent_messages)
        assert "Step 1: checking system" not in parent_text
        assert "Step 2: checking files" not in parent_text
        assert "All checks passed cleanly" in parent_messages[-1]["content"]


# ---------------------------------------------------------------------------
# Test 4: Blackboard Rollback on Error
# ---------------------------------------------------------------------------

class TestBlackboardRollbackOnError:
    @pytest.mark.asyncio
    async def test_unhandled_exception_reverts_blackboard_with_zero_orphaned_keys(self):
        bb = BlackboardStore()
        session_id = "test_session_rollback"

        # Pre-seed session with an existing safe artifact
        bb.write(session_id, "CoderAgent", "existing_safe_key", "valid_pre_state")

        # Custom worker that writes intermediate dirty artifact then crashes
        class FaultyWorker:
            def __init__(self, blackboard, session_id):
                self.blackboard = blackboard
                self.session_id = session_id

            async def run(self, task, context=None):
                # Write dirty artifact to blackboard
                self.blackboard.write(self.session_id, "CoderAgent", "uncommitted_key", "corrupt_partial_data")
                # Simulate unhandled crash
                raise RuntimeError("Catastrophic AST parser crash in worker agent!")

        tool = SubagentDelegationTool(blackboard=bb, session_id=session_id)
        faulty = FaultyWorker(bb, session_id)

        res = await tool.execute(
            role="coder",
            task="Generate AST",
            session_id=session_id,
            subagent_override=faulty,
        )

        # Assert fault containment
        assert res.success is False
        assert "SubagentExecutionError" in res.error
        assert "Catastrophic AST parser crash" in res.error

        # Assert blackboard transaction rollback
        # 1. Pre-existing safe key remains intact
        assert bb.read(session_id, "CoderAgent", "existing_safe_key") == "valid_pre_state"
        # 2. Uncommitted key from failed execution was rolled back and purged
        assert bb.read(session_id, "CoderAgent", "uncommitted_key") is None
        keys = bb.list_keys(session_id, "CoderAgent")
        assert "uncommitted_key" not in keys
        assert keys == ["existing_safe_key"]


# ---------------------------------------------------------------------------
# Test 5: Timeout Containment and Rollback
# ---------------------------------------------------------------------------

class TestTimeoutContainment:
    @pytest.mark.asyncio
    async def test_hanging_subagent_times_out_and_rolls_back_blackboard(self):
        bb = BlackboardStore()
        session_id = "test_session_timeout"
        bb.write(session_id, "CoderAgent", "safe_checkpoint", "ok")

        class HangingWorker:
            def __init__(self, blackboard, session_id):
                self.blackboard = blackboard
                self.session_id = session_id

            async def run(self, task, context=None):
                self.blackboard.write(self.session_id, "CoderAgent", "hanging_artifact", "in_flight")
                await asyncio.sleep(10.0)  # Hang
                return AgentResult(success=True, output="Done")

        tool = SubagentDelegationTool(blackboard=bb, session_id=session_id)
        hanging = HangingWorker(bb, session_id)

        # Run with short 0.2s timeout
        res = await tool.execute(
            role="coder",
            task="Long task",
            timeout_seconds=1,  # tool clamps to min 1.0s or test override
            session_id=session_id,
            subagent_override=hanging,
        )

        assert res.success is False
        assert "SubagentExecutionError" in res.error
        assert "timed out" in res.error.lower()

        # Check blackboard rolled back
        assert bb.read(session_id, "CoderAgent", "hanging_artifact") is None
        assert bb.read(session_id, "CoderAgent", "safe_checkpoint") == "ok"


# ---------------------------------------------------------------------------
# Test 6: Recursion Depth Ceiling Enforcement
# ---------------------------------------------------------------------------

class TestRecursionDepthCeiling:
    @pytest.mark.asyncio
    async def test_delegation_beyond_max_depth_is_strictly_rejected(self):
        bb = BlackboardStore()
        session_id = "test_depth_session"

        # Level 0 tool
        tool_level_0 = SubagentDelegationTool(
            blackboard=bb,
            session_id=session_id,
            max_depth=2,
            current_depth=0,
        )

        # Level 1 tool
        tool_level_1 = SubagentDelegationTool(
            blackboard=bb,
            session_id=session_id,
            max_depth=2,
            current_depth=1,
        )

        # Level 2 tool (at ceiling)
        tool_level_2 = SubagentDelegationTool(
            blackboard=bb,
            session_id=session_id,
            max_depth=2,
            current_depth=2,
        )

        # Level 1 delegation succeeds
        res_l1 = await tool_level_1.execute(role="general", task="Task L1")
        assert res_l1.success is True

        # Level 2 delegation at max_depth rejects with DelegationDepthExceeded
        res_l2 = await tool_level_2.execute(role="general", task="Task L2")
        assert res_l2.success is False
        assert "DelegationDepthExceeded" in res_l2.error
        assert "Maximum delegation depth of 2 reached" in res_l2.error

    @pytest.mark.asyncio
    async def test_contextvar_depth_tracking_prevents_recursive_runaway(self):
        bb = BlackboardStore()
        session_id = "test_contextvar_depth"

        tool = SubagentDelegationTool(blackboard=bb, session_id=session_id, max_depth=2)

        # Simulate nested worker trying to delegate again
        class RecursiveDelegatingWorker:
            def __init__(self, parent_tool):
                self.parent_tool = parent_tool

            async def run(self, task, context=None):
                # Attempt to delegate child subtask (at depth 2)
                return await self.parent_tool.execute(role="coder", task="Child recursive task")

        worker = RecursiveDelegatingWorker(tool)

        # First delegation sets depth to 1. The worker internally delegates again, which hits depth 2.
        # If worker tries once more, it must be rejected.
        res = await tool.execute(role="coder", task="Root task", subagent_override=worker)
        assert res.success is True  # depth 1 calling depth 2 succeeded

        # Now simulate worker at depth 2 trying to delegate to depth 3
        token = _current_delegation_depth.set(2)
        try:
            res_exceeded = await tool.execute(role="coder", task="Attempt depth 3")
            assert res_exceeded.success is False
            assert "DelegationDepthExceeded" in res_exceeded.error
        finally:
            _current_delegation_depth.reset(token)


# ---------------------------------------------------------------------------
# Test 7: Capability Ring Enforcement
# ---------------------------------------------------------------------------

class TestCapabilityRingEnforcement:
    @pytest.mark.asyncio
    async def test_subagent_cannot_execute_tool_exceeding_permitted_ring(self):
        bb = BlackboardStore()
        registry = ToolRegistry()
        session_id = "test_ring_enforcement"

        # Mock an unprivileged caller (Ring 1)
        # Attempt to run 'execute_command' (REQUIRED_RING = 0)
        profile = SUBAGENT_PROFILES["coder"]
        worker = DelegatedSubagent(
            profile=profile,
            session_id=session_id,
            blackboard=bb,
            tool_registry=registry,
            caller_ring=1,  # Unprivileged Ring 1
        )

        # Directly attempt tool call on a Ring 0 tool ('shell' requires Ring 0)
        # In Zelyr ring architecture, Ring 0 is root privilege. A caller with Ring 1 (1 > 0) is denied.
        result_str = await worker.execute_tool_call("shell", {"command": "whoami"})

        assert "AuthorizationError" in result_str
        assert "requires higher permissions" in result_str


# ---------------------------------------------------------------------------
# Test 8: Role Profiles & Tool Curation
# ---------------------------------------------------------------------------

class TestRoleProfilesAndToolCuration:
    def test_canonical_profiles_and_aliases(self):
        # All 5 standard profiles exist
        for role in ("coder", "tester", "researcher", "reviewer", "general"):
            p = get_subagent_profile(role)
            assert p.role == role
            assert len(p.allowed_tools) > 0

        # Aliases resolve to canonical roles
        assert get_subagent_profile("critic").role == "reviewer"
        assert get_subagent_profile("qa").role == "tester"
        assert get_subagent_profile("developer").role == "coder"
        assert get_subagent_profile("search").role == "researcher"

    @pytest.mark.asyncio
    async def test_reviewer_cannot_invoke_modifying_or_shell_tools(self):
        bb = BlackboardStore()
        registry = ToolRegistry()
        session_id = "test_curation"

        reviewer_profile = SUBAGENT_PROFILES["reviewer"]
        worker = DelegatedSubagent(
            profile=reviewer_profile,
            session_id=session_id,
            blackboard=bb,
            tool_registry=registry,
            caller_ring=1,
        )

        # Reviewer cannot call 'file_write'
        res = await worker.execute_tool_call("file_write", {"path": "a.txt", "content": "x"})
        assert "AuthorizationError" in res
        assert "not in allowed tools list for role 'reviewer'" in res

        # Reviewer cannot call 'shell'
        res2 = await worker.execute_tool_call("shell", {"command": "ls"})
        assert "AuthorizationError" in res2
        assert "not in allowed tools list for role 'reviewer'" in res2


# ---------------------------------------------------------------------------
# Test 9: NativeOrchestrator Integration Flow
# ---------------------------------------------------------------------------

class TestNativeOrchestratorIntegrationFlow:
    @pytest.mark.asyncio
    async def test_orchestrator_delegates_to_subagent_and_synthesizes_response(self):
        from zelyr.agents.native_orchestrator import NativeOrchestrator

        bb = BlackboardStore()
        registry = ToolRegistry()
        session_id = "test_orchestrator_subagent"

        # Initialize SubagentDelegationTool
        delegation_tool = SubagentDelegationTool(
            blackboard=bb,
            tool_registry=registry,
            session_id=session_id,
        )

        # Mock subagent output
        async def mock_execute_tool(name: str, **kwargs):
            if name == "delegate_subtask":
                return await delegation_tool.execute(**kwargs)
            return ToolResult(success=True, output=f"Executed {name}")

        orchestrator = NativeOrchestrator(
            execute_tool=mock_execute_tool,
        )

        # Execute direct delegation via orchestrator's executor
        tool_call_res = await orchestrator.execute_tool(
            "delegate_subtask",
            role="researcher",
            task="Find documentation on async sqlite transactions",
            session_id=session_id,
        )

        assert tool_call_res.success is True
        assert "Subagent [ResearchAgent] executed task" in tool_call_res.output
        assert tool_call_res.metadata["role"] == "researcher"
        assert bb.read(session_id, "ResearchAgent", "task_result") is not None
