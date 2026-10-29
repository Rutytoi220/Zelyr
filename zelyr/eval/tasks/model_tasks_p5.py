"""Level 5 Benchmark Evaluation Tasks: Autonomous Federation, Tool Synthesis & Multi-Agent Swarm Workflows.

Evaluates Zelyr's autonomous capabilities in Level 5 advanced architectures:
1. task_mcp_server_gateway_federation: Outbound MCP server gateway stdio JSON-RPC handshake, tool listing, tool invocation, and memory resource reads.
2. task_autonomous_routine_lifecycle: Autonomous background routine scheduling, interval evaluation, execution triggering, and SQLite history persistence.
3. task_adaptive_context_budget_compaction: Adaptive context window budgeting and historical tool compaction under pressure while preserving inviolate anchors.
4. task_hierarchical_subagent_delegation: Multi-agent delegation via SubagentDelegationTool and SwarmBlackboard with context bubble isolation.
5. task_autonomous_tool_crafting_and_hotload: Autonomous tool crafting via craft_tool: AST security gate, SandboxedPluginRunner test gate, dynamic registry hotload, and invocation.
6. task_multi_turn_session_continuity: Multi-turn chat session persistence via SessionManager: durable SQLite storage, tool-call tracking, and restart continuity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import anyio
from mcp.client.session import ClientSession

from zelyr.agents.swarm.delegation import SubagentDelegationTool
from zelyr.core.routine import RoutineDatabase, RoutineEngine
from zelyr.db.memory import MemoryStore
from zelyr.engine.budget_manager import ContextBudgetEngine, estimate_messages_tokens
from zelyr.engine.tool_crafter import ToolCrafterEngine
from zelyr.eval.task import EvalContext, EvalTask
from zelyr.mcp.server import MCPServer
from zelyr.memory.blackboard import BlackboardStore
from zelyr.memory.sessions import SessionManager
from zelyr.tool_registry import ToolRegistry


# ---------------------------------------------------------------------------
# Task 1: task_mcp_server_gateway_federation
# ---------------------------------------------------------------------------


def build_task_mcp_server_gateway_federation() -> EvalTask:
    """Level 5 Task 1: Outbound MCP server gateway stdio JSON-RPC handshake, tool enumeration, tool invocation, and memory resource reads."""

    def setup(workspace: Path, memory: MemoryStore) -> None:
        if memory is not None:
            memory.store(
                "Federated cluster node alpha active and healthy",
                category="environment",
                key="cluster.node.alpha",
            )
            memory.store(
                "Primary orchestration tier set to high-availability",
                category="preference",
                key="cluster.tier.mode",
            )

    async def execute_mcp_federation(
        action: str,
        text: str,
        context: EvalContext,
    ) -> Dict[str, Any]:
        """Custom tool executing full MCP protocol handshake and client operations."""
        server = MCPServer(
            memory_store=context.memory_store,
            expose_memory=True,
            server_name="zelyr",
            server_version="1.2.0",
        )

        s2c_send, s2c_recv = anyio.create_memory_object_stream[Any](20)
        c2s_send, c2s_recv = anyio.create_memory_object_stream[Any](20)

        tg = anyio.create_task_group()
        await tg.__aenter__()

        tg.start_soon(
            server.server.run,
            c2s_recv,
            s2c_send,
            server.server.create_initialization_options(),
        )

        client = ClientSession(s2c_recv, c2s_send)
        await client.__aenter__()

        try:
            # 1. Handshake
            init_res = await client.initialize()
            server_info = {
                "name": init_res.serverInfo.name,
                "version": init_res.serverInfo.version,
                "protocolVersion": init_res.protocolVersion,
            }

            # 2. Tool enumeration
            tools_res = await client.list_tools()
            tool_names = [t.name for t in tools_res.tools]

            # 3. Tool invocation
            call_res = await client.call_tool("echo", {"text": text})
            call_content = [c.text for c in call_res.content if hasattr(c, "text")]
            is_error = getattr(call_res, "isError", False)

            # 4. Resource reading
            res_recent = await client.read_resource("zelyr://memory/recent")
            raw_recent_text = res_recent.contents[0].text if res_recent.contents else "[]"
            recent_memories = json.loads(raw_recent_text)

            handshake_data = {
                "server_info": server_info,
                "tools": tool_names,
                "tool_call": {
                    "tool": "echo",
                    "input": text,
                    "output": call_content,
                    "isError": is_error,
                },
                "resources": {
                    "uri": "zelyr://memory/recent",
                    "memories": recent_memories,
                },
            }

            out_file = context.resolve_path("mcp_handshake.json")
            out_file.write_text(json.dumps(handshake_data, indent=2), encoding="utf-8")

            return {
                "success": True,
                "server_name": server_info["name"],
                "protocol_version": server_info["protocolVersion"],
                "tools_count": len(tool_names),
                "echo_response": call_content[0] if call_content else "",
                "memory_count": len(recent_memories),
            }
        finally:
            await client.__aexit__(None, None, None)
            tg.cancel_scope.cancel()
            await tg.__aexit__(None, None, None)

    mock_responses = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_mcp_fed",
                    "function": {
                        "name": "execute_mcp_federation",
                        "arguments": {
                            "action": "handshake_and_call",
                            "text": "federated_mcp_handshake_verified",
                        },
                    },
                }
            ],
        },
        {
            "role": "assistant",
            "content": "Successfully established MCP server gateway federation, enumerated registered tools, verified echo invocation, and queried memory resources via zelyr://memory/recent.",
        },
    ]

    def validator(context: EvalContext) -> Tuple[bool, str]:
        report_file = context.resolve_path("mcp_handshake.json")
        if not report_file.is_file():
            return False, "mcp_handshake.json was not created"

        try:
            data = json.loads(report_file.read_text(encoding="utf-8"))
        except Exception as e:
            return False, f"Failed to parse mcp_handshake.json: {e}"

        server_info = data.get("server_info", {})
        if server_info.get("name") != "zelyr":
            return False, f"Expected server name 'zelyr', got '{server_info.get('name')}'"
        if not server_info.get("protocolVersion"):
            return False, "Missing protocolVersion in MCP handshake"

        tools = data.get("tools", [])
        if "echo" not in tools:
            return False, f"Expected 'echo' tool in MCP tools list: {tools}"

        tool_call = data.get("tool_call", {})
        if tool_call.get("isError") is not False:
            return False, f"MCP tool call returned error: {tool_call}"
        if "federated_mcp_handshake_verified" not in str(tool_call.get("output", [])):
            return False, f"Expected echo output in tool_call: {tool_call}"

        resources = data.get("resources", {})
        memories = resources.get("memories", [])
        if not any("Federated cluster node alpha" in str(m) for m in memories):
            return False, f"Expected seeded memory in resource response: {memories}"

        return True, "MCP server gateway federation verified successfully"

    return EvalTask(
        task_id="task_mcp_server_gateway_federation",
        description="Outbound MCP server gateway stdio JSON-RPC handshake, tool listing, tool invocation, and memory resource reads",
        level=5,
        max_steps=5,
        timeout_seconds=15.0,
        setup_fn=setup,
        validator_fn=validator,
        user_prompt="Connect to Zelyr's outbound MCP server gateway, verify handshake negotiation, list available tools, invoke the echo tool, and retrieve recent memory resources.",
        mock_responses=mock_responses,
        custom_tools={
            "execute_mcp_federation": execute_mcp_federation,
        },
        metadata={"benchmark_level": 5},
    )


# ---------------------------------------------------------------------------
# Task 2: task_autonomous_routine_lifecycle
# ---------------------------------------------------------------------------


def build_task_autonomous_routine_lifecycle() -> EvalTask:
    """Level 5 Task 2: Autonomous background routine scheduling, interval evaluation, execution triggering, and SQLite history persistence."""

    def setup(workspace: Path, memory: MemoryStore) -> None:
        pass

    async def manage_routine_lifecycle(
        name: str,
        schedule_expr: str,
        action_type: str,
        action_payload: Dict[str, Any],
        context: EvalContext,
    ) -> Dict[str, Any]:
        """Custom tool registering, persisting, and executing a background routine."""
        db_file = context.resolve_path("routines.db")
        engine = RoutineEngine(db_path=db_file)
        await engine.db.initialize()

        routine = await engine.add_routine_async(
            name=name,
            schedule_expr=schedule_expr,
            schedule_type="interval",
            action_type=action_type,
            action_payload=action_payload,
            timeout_seconds=10.0,
        )

        run = await engine.execute_routine_once(routine)

        # Retrieve updated state and history
        refreshed = await engine.db.get_routine(routine.id)
        history = await engine.db.get_history(routine.id)

        report = {
            "routine_id": routine.id,
            "routine_name": routine.name,
            "schedule_expr": refreshed.schedule_expr if refreshed else schedule_expr,
            "last_run_at": refreshed.last_run_at if refreshed else None,
            "next_run_at": refreshed.next_run_at if refreshed else None,
            "run_status": run.status,
            "run_output": run.output,
            "run_error": run.error,
            "history_count": len(history),
        }

        out_file = context.resolve_path("routine_lifecycle_report.json")
        out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

        return {
            "success": run.status == "success",
            "routine_id": routine.id,
            "run_status": run.status,
            "output": run.output,
        }

    mock_responses = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_routine_mgr",
                    "function": {
                        "name": "manage_routine_lifecycle",
                        "arguments": {
                            "name": "routine_telemetry_probe",
                            "schedule_expr": "300",
                            "action_type": "tool",
                            "action_payload": {"tool": "echo", "args": {"text": "telemetry_probe_ok"}},
                        },
                    },
                }
            ],
        },
        {
            "role": "assistant",
            "content": "Registered background routine 'routine_telemetry_probe', executed manual single-shot trigger, verified status 'success', and confirmed SQLite execution history logging.",
        },
    ]

    async def validator(context: EvalContext) -> Tuple[bool, str]:
        report_file = context.resolve_path("routine_lifecycle_report.json")
        if not report_file.is_file():
            return False, "routine_lifecycle_report.json was not created"

        try:
            report = json.loads(report_file.read_text(encoding="utf-8"))
        except Exception as e:
            return False, f"Failed to parse routine_lifecycle_report.json: {e}"

        if report.get("run_status") != "success":
            return False, f"Expected run_status 'success', got '{report.get('run_status')}'"
        if "telemetry_probe_ok" not in str(report.get("run_output", "")):
            return False, f"Expected 'telemetry_probe_ok' in run output: {report.get('run_output')}"

        # Verify DB directly (restart invariance check)
        db_file = context.resolve_path("routines.db")
        if not db_file.is_file():
            return False, "routines.db SQLite database file missing"

        db = RoutineDatabase(db_file)
        routines = await db.list_routines(include_disabled=True)
        matching = [r for r in routines if r.name == "routine_telemetry_probe"]
        if not matching:
            return False, "Routine 'routine_telemetry_probe' not found in SQLite database"

        r = matching[0]
        if r.last_run_at is None or r.next_run_at is None:
            return False, f"Routine timestamps unpopulated: last_run_at={r.last_run_at}, next_run_at={r.next_run_at}"
        if r.next_run_at <= r.last_run_at:
            return False, f"next_run_at ({r.next_run_at}) must be greater than last_run_at ({r.last_run_at})"

        history = await db.get_history(r.id)
        if len(history) != 1:
            return False, f"Expected exactly 1 execution history record, found {len(history)}"
        if history[0].status != "success":
            return False, f"Expected history status 'success', got '{history[0].status}'"

        return True, "Autonomous routine lifecycle and persistence verified"

    return EvalTask(
        task_id="task_autonomous_routine_lifecycle",
        description="Autonomous background routine scheduling, interval evaluation, execution triggering, and SQLite history persistence",
        level=5,
        max_steps=5,
        timeout_seconds=15.0,
        setup_fn=setup,
        validator_fn=validator,
        user_prompt="Schedule an autonomous routine 'routine_telemetry_probe' with a 300-second interval, trigger its execution, verify run success, and confirm persistence in the routine database.",
        mock_responses=mock_responses,
        custom_tools={
            "manage_routine_lifecycle": manage_routine_lifecycle,
        },
        metadata={"benchmark_level": 5},
    )


# ---------------------------------------------------------------------------
# Task 3: task_adaptive_context_budget_compaction
# ---------------------------------------------------------------------------


def build_task_adaptive_context_budget_compaction() -> EvalTask:
    """Level 5 Task 3: Adaptive context window budgeting and historical tool compaction under pressure while preserving inviolate anchors."""

    def setup(workspace: Path, memory: MemoryStore) -> None:
        bulky_payload = "Log entry analysis chunk: " + ("metric_key=value_token_delta " * 350)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "You are Zelyr, a resilient autonomous agent. Follow safety standards."},
            # Historical Turn 1
            {"role": "user", "content": "Analyze network ingress logs for anomaly patterns."},
            {"role": "assistant", "content": "Executing ingress inspection.", "tool_calls": [{"name": "inspect_logs"}]},
            {"role": "tool", "name": "inspect_logs", "content": bulky_payload},
            # Historical Turn 2
            {"role": "user", "content": "Extract suspicious IP addresses and payload signatures."},
            {"role": "assistant", "content": "Executing payload extraction.", "tool_calls": [{"name": "extract_ips"}]},
            {"role": "tool", "name": "extract_ips", "content": bulky_payload},
            # Historical Turn 3
            {"role": "user", "content": "Cross-reference IP addresses against threat intelligence database."},
            {"role": "assistant", "content": "Querying threat database.", "tool_calls": [{"name": "threat_intel"}]},
            {"role": "tool", "name": "threat_intel", "content": bulky_payload},
            # Recent Turn 4 (2nd most recent user turn)
            {"role": "user", "content": "Check active firewall rules to verify blocklist coverage."},
            {"role": "assistant", "content": "Inspecting firewall blocklist.", "tool_calls": [{"name": "check_firewall"}]},
            {"role": "tool", "name": "check_firewall", "content": bulky_payload},
            # Recent Turn 5 (Latest user turn - inviolate anchor)
            {"role": "user", "content": "Synthesize final incident mitigation summary and confirm actions taken."},
        ]

        (workspace / "uncompacted_session.json").write_text(
            json.dumps(messages, indent=2),
            encoding="utf-8",
        )

    def compact_conversation_context(
        input_file: str,
        output_file: str,
        tool_output_token_limit: int,
        context: EvalContext,
    ) -> Dict[str, Any]:
        """Custom tool applying ContextBudgetEngine compaction to multi-turn conversation."""
        in_path = context.resolve_path(input_file)
        if not in_path.is_file():
            return {"success": False, "error": f"Input file '{input_file}' not found"}

        messages: List[Dict[str, Any]] = json.loads(in_path.read_text(encoding="utf-8"))
        tokens_before = estimate_messages_tokens(messages)

        engine = ContextBudgetEngine(
            tool_output_token_limit=tool_output_token_limit,
            recent_turns_to_preserve=2,
        )

        compacted, compacted_count = engine.stage1_compact_tool_results(
            messages,
            recent_turns_to_preserve=2,
            tool_output_token_limit=tool_output_token_limit,
        )
        tokens_after = estimate_messages_tokens(compacted)

        out_path = context.resolve_path(output_file)
        result_payload = {
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "reduction_pct": round((1.0 - (tokens_after / tokens_before)) * 100, 2),
            "compacted_tool_count": compacted_count,
            "messages": compacted,
        }
        out_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")

        return {
            "success": True,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "compacted_count": compacted_count,
            "reduction_pct": result_payload["reduction_pct"],
        }

    mock_responses = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_compact",
                    "function": {
                        "name": "compact_conversation_context",
                        "arguments": {
                            "input_file": "uncompacted_session.json",
                            "output_file": "compacted_session.json",
                            "tool_output_token_limit": 200,
                        },
                    },
                }
            ],
        },
        {
            "role": "assistant",
            "content": "Conversation context successfully compacted: Stage 1 compacted historical tool outputs older than 2 turns, reduced token volume by over 40%, and preserved root system prompt and latest user turn anchors intact.",
        },
    ]

    def validator(context: EvalContext) -> Tuple[bool, str]:
        out_file = context.resolve_path("compacted_session.json")
        if not out_file.is_file():
            return False, "compacted_session.json was not generated"

        try:
            data = json.loads(out_file.read_text(encoding="utf-8"))
        except Exception as e:
            return False, f"Failed to parse compacted_session.json: {e}"

        compacted_count = data.get("compacted_tool_count", 0)
        if compacted_count != 3:
            return False, f"Expected 3 historical tool outputs compacted, got {compacted_count}"

        reduction = data.get("reduction_pct", 0.0)
        if reduction < 40.0:
            return False, f"Expected token reduction > 40%, got {reduction}%"

        messages = data.get("messages", [])
        if len(messages) != 14:
            return False, f"Expected 14 total messages, got {len(messages)}"

        # Inviolate anchor 1: System prompt preserved verbatim
        root_sys = messages[0]
        if root_sys.get("role") != "system" or "resilient autonomous agent" not in root_sys.get("content", ""):
            return False, f"Root system prompt anchor altered or corrupt: {root_sys}"

        # Inviolate anchor 2: Latest user turn preserved verbatim
        last_turn = messages[-1]
        if last_turn.get("role") != "user" or "Synthesize final incident mitigation summary" not in last_turn.get("content", ""):
            return False, f"Latest user turn anchor altered or corrupt: {last_turn}"

        # Historical tools compacted
        for idx in (3, 6, 9):
            tool_msg = messages[idx]
            if "[... output truncated:" not in tool_msg.get("content", ""):
                return False, f"Historical tool message at index {idx} was not truncated: {tool_msg.get('content')[:100]}"

        # Recent tool preserved untouched
        recent_tool = messages[11] if len(messages) > 11 else messages[-2]
        # In the 12-message sequence:
        # 0: sys
        # 1: u1, 2: a1, 3: t1 (compacted)
        # 4: u2, 5: a2, 6: t2 (compacted)
        # 7: u3, 8: a3, 9: t3 (compacted)
        # 10: u4, 11: a4, 12 is t4... wait, let's verify exact index
        check_fw_tool = next((m for m in messages if m.get("name") == "check_firewall"), None)
        if not check_fw_tool:
            return False, "Recent tool 'check_firewall' missing from compacted messages"
        if "[... output truncated:" in check_fw_tool.get("content", ""):
            return False, "Recent tool 'check_firewall' (<2 turns old) was erroneously truncated"

        return True, "Adaptive context budgeting compaction verified"

    return EvalTask(
        task_id="task_adaptive_context_budget_compaction",
        description="Adaptive context window budgeting and historical tool compaction under pressure while preserving inviolate anchors",
        level=5,
        max_steps=5,
        timeout_seconds=15.0,
        setup_fn=setup,
        validator_fn=validator,
        user_prompt="Enforce context window budgeting on 'uncompacted_session.json': compact historical tool results older than 2 turns while preserving system prompt and recent turns intact.",
        mock_responses=mock_responses,
        custom_tools={
            "compact_conversation_context": compact_conversation_context,
        },
        metadata={"benchmark_level": 5},
    )


# ---------------------------------------------------------------------------
# Task 4: task_hierarchical_subagent_delegation
# ---------------------------------------------------------------------------


def build_task_hierarchical_subagent_delegation() -> EvalTask:
    """Level 5 Task 4: Multi-agent delegation via SubagentDelegationTool and SwarmBlackboard with context bubble isolation."""

    def setup(workspace: Path, memory: MemoryStore) -> None:
        target_code = (
            '"""Data structure sorting utility."""\n\n'
            'def sort_records(records: list) -> list:\n'
            '    return sorted(records, key=lambda r: r.get("priority", 0), reverse=True)\n'
        )
        (workspace / "delegation_target.py").write_text(target_code, encoding="utf-8")

    async def delegate_subagent_task(
        role: str,
        task_description: str,
        session_id: str,
        context: EvalContext,
    ) -> Dict[str, Any]:
        """Custom tool executing SubagentDelegationTool with isolated blackboard write."""
        blackboard = BlackboardStore()
        registry = ToolRegistry()

        # Mock subagent LLM performing isolated ReAct execution
        mock_subagent_llm = MagicMock()
        mock_subagent_llm.generate.side_effect = [
            {
                "content": "Analyzing delegation_target.py and posting audit findings to blackboard.",
                "tool_calls": [
                    {
                        "id": "sub_call_1",
                        "function": {
                            "name": "blackboard_write",
                            "arguments": {
                                "key": "security_audit_report",
                                "value": json.dumps({
                                    "target": "delegation_target.py",
                                    "status": "PASS",
                                    "findings": "Deterministic sort on priority key verified safe.",
                                }),
                            },
                        },
                    }
                ],
            },
            {
                "content": "Coder subagent completed task: security audit report committed to SwarmBlackboard under key 'security_audit_report'.",
                "tool_calls": [],
            },
        ]

        delegation_tool = SubagentDelegationTool(
            blackboard=blackboard,
            tool_registry=registry,
            llm_client=mock_subagent_llm,
            session_id=session_id,
        )

        res = await delegation_tool.execute(
            role=role,
            task=task_description,
            session_id=session_id,
        )

        # Read committed value from blackboard
        report_val = blackboard.read(session_id, "CoderAgent", "security_audit_report")

        out_data = {
            "success": res.success,
            "role": role,
            "agent_name": res.metadata.get("agent_name", ""),
            "output": res.output,
            "blackboard_keys": res.metadata.get("blackboard_keys", []),
            "committed_audit_report": report_val,
        }

        out_file = context.resolve_path("blackboard_state.json")
        out_file.write_text(json.dumps(out_data, indent=2), encoding="utf-8")

        return {
            "success": res.success,
            "output": res.output,
            "blackboard_keys": res.metadata.get("blackboard_keys", []),
            "report_stored": bool(report_val),
        }

    mock_responses = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_delegate",
                    "function": {
                        "name": "delegate_subagent_task",
                        "arguments": {
                            "role": "coder",
                            "task_description": "Audit delegation_target.py and commit findings to blackboard",
                            "session_id": "eval_swarm_session_l5",
                        },
                    },
                }
            ],
        },
        {
            "role": "assistant",
            "content": "Hierarchical multi-agent delegation succeeded: Coder subagent analyzed delegation_target.py, wrote findings to SwarmBlackboard under 'security_audit_report', and returned concise summary without parent context bloat.",
        },
    ]

    def validator(context: EvalContext) -> Tuple[bool, str]:
        bb_file = context.resolve_path("blackboard_state.json")
        if not bb_file.is_file():
            return False, "blackboard_state.json was not created"

        try:
            data = json.loads(bb_file.read_text(encoding="utf-8"))
        except Exception as e:
            return False, f"Failed to parse blackboard_state.json: {e}"

        if not data.get("success"):
            return False, f"Subagent delegation indicated failure: {data}"

        if data.get("role") != "coder" or data.get("agent_name") != "CoderAgent":
            return False, f"Expected CoderAgent subagent role, got {data.get('role')} / {data.get('agent_name')}"

        if "security_audit_report" not in data.get("blackboard_keys", []):
            return False, f"Expected 'security_audit_report' in blackboard keys: {data.get('blackboard_keys')}"

        report_raw = data.get("committed_audit_report")
        if not report_raw or "Deterministic sort" not in str(report_raw):
            return False, f"Blackboard committed report corrupt or missing: {report_raw}"

        # Verify parent context isolation: output must be concise summary, not raw tool-call trace
        output = data.get("output", "")
        if "Coder subagent completed task" not in output:
            return False, f"Expected concise synthesized subagent output, got: {output}"

        return True, "Hierarchical subagent delegation and blackboard coordination verified"

    return EvalTask(
        task_id="task_hierarchical_subagent_delegation",
        description="Multi-agent delegation via SubagentDelegationTool and SwarmBlackboard with context bubble isolation",
        level=5,
        max_steps=5,
        timeout_seconds=15.0,
        setup_fn=setup,
        validator_fn=validator,
        user_prompt="Dispatch a specialized coder subagent to audit delegation_target.py, commit security findings to SwarmBlackboard, and report the synthesized summary.",
        mock_responses=mock_responses,
        custom_tools={
            "delegate_subagent_task": delegate_subagent_task,
        },
        metadata={"benchmark_level": 5},
    )


# ---------------------------------------------------------------------------
# Task 5: task_autonomous_tool_crafting_and_hotload
# ---------------------------------------------------------------------------


def build_task_autonomous_tool_crafting_and_hotload() -> EvalTask:
    """Level 5 Task 5: Autonomous tool crafting via craft_tool: AST security gate, SandboxedPluginRunner test gate, dynamic registry hotload, and invocation."""

    def setup(workspace: Path, memory: MemoryStore) -> None:
        pass

    async def synthesize_and_execute_tool(
        name: str,
        description: str,
        code: str,
        test_code: str,
        parameters: Dict[str, Any],
        test_arguments: Dict[str, Any],
        context: EvalContext,
    ) -> Dict[str, Any]:
        """Custom tool synthesizing a runtime tool, validating via AST and sandbox, and executing it."""
        registry = ToolRegistry()
        engine = ToolCrafterEngine(registry=registry)

        craft_res = await engine.craft_tool(
            name=name,
            description=description,
            code=code,
            test_code=test_code,
            parameters=parameters,
            persist=False,
        )

        if not craft_res.success:
            return {
                "success": False,
                "error": craft_res.error,
                "remedy_hint": craft_res.remedy_hint,
            }

        tool = registry.get_tool(name)
        if tool is None:
            return {"success": False, "error": f"Tool '{name}' not found in registry after crafting"}

        exec_res = await registry.execute_async(name, **test_arguments)

        output_data = {
            "craft_success": craft_res.success,
            "tool_name": name,
            "tool_description": description,
            "parameters": parameters,
            "test_arguments": test_arguments,
            "execution_success": exec_res.success,
            "execution_output": exec_res.output,
            "execution_error": exec_res.error,
        }

        out_file = context.resolve_path("crafted_tool_output.json")
        out_file.write_text(json.dumps(output_data, indent=2), encoding="utf-8")

        return {
            "success": exec_res.success,
            "tool_name": name,
            "output": exec_res.output,
        }

    tool_code = (
        'def execute(numbers: list = None, **kwargs):\n'
        '    nums = numbers or []\n'
        '    if not nums:\n'
        '        return {"count": 0, "sum": 0, "average": 0.0, "max": None, "min": None}\n'
        '    total = sum(nums)\n'
        '    return {\n'
        '        "count": len(nums),\n'
        '        "sum": total,\n'
        '        "average": round(total / len(nums), 3),\n'
        '        "max": max(nums),\n'
        '        "min": min(nums),\n'
        '    }\n'
    )

    test_code = (
        'assert execute([])["count"] == 0\n'
        'res = execute([10, 20, 30])\n'
        'assert res["count"] == 3\n'
        'assert res["sum"] == 60\n'
        'assert res["average"] == 20.0\n'
        'assert res["max"] == 30\n'
        'assert res["min"] == 10\n'
    )

    tool_params = {
        "type": "object",
        "properties": {
            "numbers": {
                "type": "array",
                "description": "List of numeric values to analyze",
            }
        },
        "required": ["numbers"],
    }

    mock_responses = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_craft",
                    "function": {
                        "name": "synthesize_and_execute_tool",
                        "arguments": {
                            "name": "compute_dataset_metrics",
                            "description": "Computes count, sum, average, min, and max for numeric lists.",
                            "code": tool_code,
                            "test_code": test_code,
                            "parameters": tool_params,
                            "test_arguments": {"numbers": [5, 15, 25, 35, 45]},
                        },
                    },
                }
            ],
        },
        {
            "role": "assistant",
            "content": "Tool 'compute_dataset_metrics' synthesized, AST validated, dynamic sandbox unit tested, registered into ToolRegistry, and executed cleanly with expected results.",
        },
    ]

    def validator(context: EvalContext) -> Tuple[bool, str]:
        out_file = context.resolve_path("crafted_tool_output.json")
        if not out_file.is_file():
            return False, "crafted_tool_output.json was not created"

        try:
            data = json.loads(out_file.read_text(encoding="utf-8"))
        except Exception as e:
            return False, f"Failed to parse crafted_tool_output.json: {e}"

        if not data.get("craft_success"):
            return False, f"Tool crafting failed: {data}"
        if not data.get("execution_success"):
            return False, f"Synthesized tool execution failed: {data.get('execution_error')}"

        out = data.get("execution_output", {})
        if out.get("count") != 5:
            return False, f"Expected count 5, got {out.get('count')}"
        if out.get("sum") != 125:
            return False, f"Expected sum 125, got {out.get('sum')}"
        if out.get("average") != 25.0:
            return False, f"Expected average 25.0, got {out.get('average')}"
        if out.get("max") != 45 or out.get("min") != 5:
            return False, f"Expected min=5, max=45, got min={out.get('min')}, max={out.get('max')}"

        return True, "Autonomous tool crafting, sandbox testing, and hotloading verified"

    return EvalTask(
        task_id="task_autonomous_tool_crafting_and_hotload",
        description="Autonomous tool crafting via craft_tool: AST security gate, SandboxedPluginRunner test gate, dynamic registry hotload, and invocation",
        level=5,
        max_steps=5,
        timeout_seconds=15.0,
        setup_fn=setup,
        validator_fn=validator,
        user_prompt="Synthesize a new tool 'compute_dataset_metrics' to compute statistical properties of numeric arrays, validate it with unit test assertions, register it into ToolRegistry, and execute it on a sample dataset.",
        mock_responses=mock_responses,
        custom_tools={
            "synthesize_and_execute_tool": synthesize_and_execute_tool,
        },
        metadata={"benchmark_level": 5},
    )


# ---------------------------------------------------------------------------
# Task 6: task_multi_turn_session_continuity
# ---------------------------------------------------------------------------


def build_task_multi_turn_session_continuity() -> EvalTask:
    """Level 5 Task 6: Multi-turn chat session persistence via SessionManager: durable SQLite storage, tool-call tracking, and restart continuity."""

    def setup(workspace: Path, memory: MemoryStore) -> None:
        pass

    def record_and_verify_session(
        session_id: str,
        title: str,
        model_name: str,
        context: EvalContext,
    ) -> Dict[str, Any]:
        """Custom tool writing multi-turn history to SQLite and verifying continuity across simulated restart."""
        db_file = context.resolve_path("chat_sessions.db")
        manager1 = SessionManager(db_file)

        # 1. Create session
        sid = manager1.create_session(session_id=session_id, title=title, model_name=model_name)

        # 2. Append 4 multi-turn messages
        manager1.append_message(sid, "user", "What is the optimal caching strategy for Redis clusters?")
        manager1.append_message(
            sid,
            "assistant",
            "Let me check existing cluster configuration.",
            tool_calls=[{"id": "call_1", "function": {"name": "cache_inspect", "arguments": {"mode": "cluster"}}}],
        )
        manager1.append_message(sid, "tool", json.dumps({"redis_nodes": 6, "replication_factor": 1, "maxmemory_policy": "allkeys-lru"}))
        manager1.append_message(
            sid,
            "assistant",
            "The cluster operates with 6 nodes and allkeys-lru eviction. Recommend enabling key partitioning and read replicas.",
        )

        # 3. Simulate process restart by opening fresh SessionManager instance
        manager2 = SessionManager(db_file)
        session_meta = manager2.get_session(sid)
        messages = manager2.get_messages(sid)

        audit_data = {
            "session_id": sid,
            "title": session_meta["title"] if session_meta else "",
            "model_name": session_meta["model_name"] if session_meta else "",
            "message_count": len(messages),
            "messages": messages,
        }

        out_file = context.resolve_path("session_continuity_audit.json")
        out_file.write_text(json.dumps(audit_data, indent=2), encoding="utf-8")

        return {
            "success": len(messages) == 4,
            "session_id": sid,
            "message_count": len(messages),
            "title": session_meta["title"] if session_meta else "",
        }

    mock_responses = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_session",
                    "function": {
                        "name": "record_and_verify_session",
                        "arguments": {
                            "session_id": "session_continuity_l5_001",
                            "title": "Redis Optimization Session",
                            "model_name": "qwen3:8b",
                        },
                    },
                }
            ],
        },
        {
            "role": "assistant",
            "content": "Multi-turn session 'session_continuity_l5_001' persisted to SQLite: 4 conversation turns with tool-call metadata preserved, verified across instance restart with zero data degradation.",
        },
    ]

    def validator(context: EvalContext) -> Tuple[bool, str]:
        audit_file = context.resolve_path("session_continuity_audit.json")
        if not audit_file.is_file():
            return False, "session_continuity_audit.json was not created"

        try:
            audit = json.loads(audit_file.read_text(encoding="utf-8"))
        except Exception as e:
            return False, f"Failed to parse session_continuity_audit.json: {e}"

        if audit.get("session_id") != "session_continuity_l5_001":
            return False, f"Expected session_id 'session_continuity_l5_001', got '{audit.get('session_id')}'"
        if audit.get("title") != "Redis Optimization Session":
            return False, f"Expected title 'Redis Optimization Session', got '{audit.get('title')}'"
        if audit.get("message_count") != 4:
            return False, f"Expected 4 messages, got {audit.get('message_count')}"

        # Re-verify independently from database file (restart check)
        db_file = context.resolve_path("chat_sessions.db")
        if not db_file.is_file():
            return False, "chat_sessions.db SQLite database file missing"

        direct_manager = SessionManager(db_file)
        direct_msgs = direct_manager.get_messages("session_continuity_l5_001")
        if len(direct_msgs) != 4:
            return False, f"Expected 4 messages from direct SQLite query, found {len(direct_msgs)}"

        # Check turn roles
        expected_roles = ["user", "assistant", "tool", "assistant"]
        actual_roles = [m["role"] for m in direct_msgs]
        if actual_roles != expected_roles:
            return False, f"Roles mismatch: expected {expected_roles}, got {actual_roles}"

        # Check tool call structured preservation
        turn2 = direct_msgs[1]
        if not turn2.get("tool_calls"):
            return False, f"Turn 2 missing tool_calls metadata: {turn2}"
        tc_fn = turn2["tool_calls"][0]["function"]["name"]
        if tc_fn != "cache_inspect":
            return False, f"Expected tool call 'cache_inspect', got '{tc_fn}'"

        return True, "Multi-turn session persistence and continuity verified"

    return EvalTask(
        task_id="task_multi_turn_session_continuity",
        description="Multi-turn chat session persistence via SessionManager: durable SQLite storage, tool-call tracking, and restart continuity",
        level=5,
        max_steps=5,
        timeout_seconds=15.0,
        setup_fn=setup,
        validator_fn=validator,
        user_prompt="Persist a 4-turn chat session with tool calls into SQLite using SessionManager, simulate restart by opening a new connection, and verify continuity.",
        mock_responses=mock_responses,
        custom_tools={
            "record_and_verify_session": record_and_verify_session,
        },
        metadata={"benchmark_level": 5},
    )


# ---------------------------------------------------------------------------
# Collective Task Discovery Helper
# ---------------------------------------------------------------------------


def get_model_tasks_p5() -> List[EvalTask]:
    """Retrieve the complete suite of 6 Level 5 evaluation benchmark tasks."""
    return [
        build_task_mcp_server_gateway_federation(),
        build_task_autonomous_routine_lifecycle(),
        build_task_adaptive_context_budget_compaction(),
        build_task_hierarchical_subagent_delegation(),
        build_task_autonomous_tool_crafting_and_hotload(),
        build_task_multi_turn_session_continuity(),
    ]
