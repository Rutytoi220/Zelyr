"""Hierarchical Multi-Agent Subagent Delegation and Blackboard Coordination.

Provides the ``SubagentDelegationTool`` (``delegate_subtask``) and ``DelegatedSubagent``
engine connecting Zelyr's ``NativeOrchestrator`` and background routines to specialized
worker subagents in isolated context execution bubbles.

Subagents coordinate intermediate artifacts through the transactional ``SwarmBlackboard``,
enforcing strict recursion limits (max depth 2), bounded execution timeouts, capability
ring inheritance, and fault containment without polluting parent conversation context.
"""
from __future__ import annotations

import asyncio
import contextvars
import enum
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from zelyr.agents.base import AgentResult
from zelyr.agents.swarm.base_swarm import BaseSubagent
from zelyr.memory.blackboard import BlackboardStore
from zelyr.tools.core import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)

# Context variable to track active delegation depth across async task hierarchies
_current_delegation_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
    "_current_delegation_depth", default=0
)


class SubagentRole(str, enum.Enum):
    """Recognized specialized subagent roles."""
    CODER = "coder"
    TESTER = "tester"
    RESEARCHER = "researcher"
    REVIEWER = "reviewer"
    GENERAL = "general"
    VISION = "vision"


ROLE_ALIASES: Dict[str, str] = {
    "critic": "reviewer",
    "review": "reviewer",
    "audit": "reviewer",
    "auditor": "reviewer",
    "code": "coder",
    "coding": "coder",
    "developer": "coder",
    "test": "tester",
    "testing": "tester",
    "qa": "tester",
    "research": "researcher",
    "search": "researcher",
    "assistant": "general",
    "default": "general",
}


@dataclass
class SubagentProfile:
    """Specialized worker role specification with curated tool allocations."""
    role: str
    agent_name: str
    description: str
    allowed_tools: List[str]
    system_prompt: str
    default_ring: int = 1
    topic: str = "swarm.delegation"
    metadata: Dict[str, Any] = field(default_factory=dict)


# Standard curated subagent role profiles
SUBAGENT_PROFILES: Dict[str, SubagentProfile] = {
    "coder": SubagentProfile(
        role="coder",
        agent_name="CoderAgent",
        description="Specialized in writing code, editing files, syntax checks, and repository modifications.",
        allowed_tools=[
            "file_read",
            "file_write",
            "manage_workspace_file",
            "shell",
            "execute_command",
            "echo",
            "blackboard_write",
            "blackboard_read",
            "craft_tool",
        ],
        system_prompt=(
            "You are CoderAgent, a specialist subagent in software development and refactoring. "
            "You write clean, bug-free code, inspect files, and execute shell or build commands. "
            "Record generated patches, AST modifications, and key findings on the shared Swarm blackboard."
        ),
        default_ring=1,
        topic="swarm.task.code",
    ),
    "tester": SubagentProfile(
        role="tester",
        agent_name="TesterAgent",
        description="Specialized in test execution, assertion verification, and execution trace inspection.",
        allowed_tools=[
            "shell",
            "execute_command",
            "file_read",
            "query_system_journal",
            "echo",
            "blackboard_write",
            "blackboard_read",
        ],
        system_prompt=(
            "You are TesterAgent, a specialist subagent in test execution and verification. "
            "You run test suites, assert correctness, and inspect execution traces and journal logs. "
            "Record test reports, failure summaries, and diagnostics on the shared Swarm blackboard."
        ),
        default_ring=1,
        topic="swarm.task.test",
    ),
    "researcher": SubagentProfile(
        role="researcher",
        agent_name="ResearchAgent",
        description="Specialized in vector memory retrieval, file discovery, and web information gathering.",
        allowed_tools=[
            "remember_fact",
            "file_search",
            "web_search",
            "fetch_web_page",
            "file_read",
            "echo",
            "blackboard_write",
            "blackboard_read",
        ],
        system_prompt=(
            "You are ResearchAgent, a specialist subagent in memory querying, documentation research, "
            "and web synthesis. You explore vector memory, search local repositories, and query web resources. "
            "Synthesize facts clearly and do not execute modifying operations or destructive commands."
        ),
        default_ring=1,
        topic="swarm.task.research",
    ),
    "reviewer": SubagentProfile(
        role="reviewer",
        agent_name="ReviewerAgent",
        description="Specialized in read-only code review, AST audits, and security vulnerability analysis.",
        allowed_tools=[
            "file_read",
            "file_search",
            "echo",
            "blackboard_write",
            "blackboard_read",
        ],
        system_prompt=(
            "You are ReviewerAgent, a specialist subagent in read-only code review and security audits. "
            "You evaluate patches, verify architectural invariants, and check for bugs, vulnerabilities, "
            "and style regressions. You operate strictly in read-only mode without modifying files."
        ),
        default_ring=1,
        topic="swarm.task.review",
    ),
    "general": SubagentProfile(
        role="general",
        agent_name="GeneralAgent",
        description="General-purpose worker subagent assisting with multi-step operational tasks using safe tools.",
        allowed_tools=[
            "echo",
            "system_info",
            "remember_fact",
            "task_scheduler",
            "file_read",
            "blackboard_write",
            "blackboard_read",
        ],
        system_prompt=(
            "You are GeneralAgent, an autonomous worker subagent. You assist with general multi-step "
            "operational tasks using standard safe unprivileged tools. Store intermediate outputs on the blackboard."
        ),
        default_ring=1,
        topic="swarm.task.general",
    ),
    "vision": SubagentProfile(
        role="vision",
        agent_name="VisionAgent",
        description="Specialized in visual inspection and desktop screenshot analysis.",
        allowed_tools=[
            "vision",
            "interact_with_ui",
            "echo",
            "blackboard_write",
            "blackboard_read",
        ],
        system_prompt=(
            "You are VisionAgent, a specialist subagent in visual UI inspection and screen analysis."
        ),
        default_ring=1,
        topic="swarm.task.vision",
    ),
}


def get_subagent_profile(role_name: str) -> SubagentProfile:
    """Resolve a role name or alias to its formal SubagentProfile."""
    key = (role_name or "").strip().lower()
    canonical = ROLE_ALIASES.get(key, key)
    if canonical in SUBAGENT_PROFILES:
        return SUBAGENT_PROFILES[canonical]
    raise ValueError(
        f"Unknown subagent role '{role_name}'. Supported roles: "
        f"{', '.join(sorted(SUBAGENT_PROFILES.keys()))}."
    )


class DelegatedSubagent(BaseSubagent):
    """Specialized worker subagent executing in an isolated context bubble.

    Operates with private message histories to prevent prompt bloat in the parent
    orchestrator. Enforces capability rings, allowed tool gating, and writes
    artifacts directly to the shared SwarmBlackboard.
    """

    def __init__(
        self,
        profile: SubagentProfile,
        session_id: str,
        blackboard: Optional[BlackboardStore] = None,
        tool_registry: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        caller_ring: int = 1,
        event_bus: Optional[Any] = None,
        current_depth: int = 1,
        max_depth: int = 2,
    ):
        super().__init__(
            name=profile.agent_name,
            description=profile.description,
            topic=profile.topic,
            event_bus=event_bus,
            tool_registry=tool_registry,
            llm_client=llm_client,
            blackboard=blackboard,
            session_id=session_id,
        )
        from zelyr.tool_registry import _is_valid_ring_value

        self.profile = profile
        if _is_valid_ring_value(caller_ring) and _is_valid_ring_value(profile.default_ring):
            self.caller_ring = max(caller_ring, profile.default_ring)
        else:
            # Preserve invalid ring metadata so execution fails closed instead of clamping
            self.caller_ring = caller_ring if not _is_valid_ring_value(caller_ring) else profile.default_ring
        self.allowed_tools = list(profile.allowed_tools)
        self.current_depth = current_depth
        self.max_depth = max_depth
        self.isolated_messages: List[Dict[str, Any]] = []
        self.set_system_prompt(profile.system_prompt)

    def write_blackboard(self, key: str, value: Any) -> None:
        """Write an artifact to this subagent's blackboard namespace."""
        self._write_to_blackboard(key, value)

    def read_blackboard(self, key: str, default: Any = None) -> Any:
        """Read an artifact from this subagent's blackboard namespace."""
        return self._read_from_blackboard(key, default)

    async def execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool call with strict role allowed-tools and capability ring enforcement."""
        import copy
        import uuid
        from contextlib import nullcontext
        from zelyr.engine.execution_ledger import (
            ExecutionStatus,
            LedgerEventType,
            VerificationStatus,
            execution_scope,
            get_active_execution_context,
            get_default_central_ledger,
        )
        from zelyr.tool_registry import (
            _is_valid_ring_value,
            authorize_and_validate_tool_call,
            canonicalize_tool_invocation,
        )

        ctx = get_active_execution_context()
        ledger = (ctx.ledger if ctx is not None else None) or getattr(self, "execution_ledger", None) or get_default_central_ledger()
        own_exec = ctx is None
        eff_ring = getattr(self, "caller_ring", None)
        if own_exec:
            exec_rec = ledger.create_execution(
                f"swarm_subagent:{self.name}:{tool_name}",
                source_component="swarm_delegation",
                caller_ring=eff_ring if isinstance(eff_ring, int) and not isinstance(eff_ring, bool) else None,
                status=ExecutionStatus.RUNNING,
            )
            exec_id = exec_rec.execution_id
        else:
            exec_id = ctx.execution_id  # type: ignore[union-attr]
        call_id = f"sub_{uuid.uuid4().hex[:12]}"

        def _finish_sub_call(res_status: str, *, err: Optional[str] = None, meta: Optional[Dict[str, Any]] = None, is_ext: bool = False, src_id: Optional[str] = None, src_type: Optional[str] = None) -> None:
            try:
                ledger.record_event(
                    exec_id,
                    LedgerEventType.TOOL_EXECUTION_RESULT,
                    tool_id=tool_name,
                    source_id=src_id,
                    source_type=src_type,
                    is_external=is_ext,
                    status=res_status,
                    correlation_id=call_id,
                    caller_ring=eff_ring if isinstance(eff_ring, int) and not isinstance(eff_ring, bool) else None,
                    metadata={**(meta or {}), **({"error": err} if err else {})},
                    idempotency_key=f"{exec_id}:{call_id}:result",
                )
                if own_exec:
                    ledger.finalize_execution(
                        exec_id,
                        status=ExecutionStatus.COMPLETED_UNVERIFIED if res_status == "succeeded" else ExecutionStatus.FAILED,
                        verification_status=VerificationStatus.NOT_VERIFIED,
                        error=err,
                    )
            except Exception:
                pass

        try:
            ledger.record_event(
                exec_id,
                LedgerEventType.TOOL_SELECTED,
                tool_id=tool_name,
                status="selected",
                correlation_id=call_id,
                caller_ring=eff_ring if isinstance(eff_ring, int) and not isinstance(eff_ring, bool) else None,
                metadata={"role": self.profile.role, "arguments": arguments if isinstance(arguments, dict) else {}},
                idempotency_key=f"{exec_id}:{call_id}:selected",
            )
        except Exception:
            pass

        if not _is_valid_ring_value(eff_ring):
            msg = f"AuthorizationError: Invalid caller_ring {eff_ring!r}."
            _finish_sub_call("rejected", err=msg)
            return msg

        if not isinstance(arguments, dict):
            msg = f"Error: Arguments for '{tool_name}' must be a JSON object/dictionary."
            _finish_sub_call("failed", err=msg)
            return msg

        raw_arguments = copy.deepcopy(arguments)
        tool_name, arguments = canonicalize_tool_invocation(tool_name, raw_arguments)

        # 1. Allowed tools gating (evaluated on canonical target name)
        if not isinstance(self.profile.allowed_tools, (list, tuple, set)) or tool_name not in self.profile.allowed_tools:
            msg = (
                f"AuthorizationError: Tool '{tool_name}' is not in allowed tools list "
                f"for role '{self.profile.role}'."
            )
            try:
                ledger.record_event(
                    exec_id,
                    LedgerEventType.AUTHORIZATION_DECISION,
                    tool_id=tool_name,
                    status="denied",
                    correlation_id=call_id,
                    caller_ring=eff_ring,
                    metadata={"reason": "tool_not_in_allowed_tools", "role": self.profile.role},
                    idempotency_key=f"{exec_id}:{call_id}:auth",
                )
            except Exception:
                pass
            _finish_sub_call("rejected", err=msg)
            return msg

        # 2. Built-in blackboard tool handlers
        if tool_name in ("blackboard_write", "blackboard_post"):
            key = arguments.get("key")
            value = arguments.get("value")
            if value is None:
                value = arguments.get("content") or arguments.get("data")
            if not key:
                msg = "Error: 'key' is required for blackboard_write."
                _finish_sub_call("failed", err=msg)
                return msg
            self.write_blackboard(key, value)
            _finish_sub_call("succeeded", meta={"key": key})
            return json.dumps({"success": True, "key": key, "recorded": True})

        if tool_name in ("blackboard_read", "read_blackboard"):
            key = arguments.get("key")
            if not key:
                msg = "Error: 'key' is required for blackboard_read."
                _finish_sub_call("failed", err=msg)
                return msg
            val = self.read_blackboard(key)
            _finish_sub_call("succeeded", meta={"key": key})
            return json.dumps({"success": True, "key": key, "value": val})

        # 3. ToolRegistry dispatch with authoritative boundary validation
        if self.tool_registry is None:
            msg = f"Error: No tool registry available to execute '{tool_name}'."
            _finish_sub_call("failed", err=msg)
            return msg

        try:
            tool_obj = (
                self.tool_registry.get_tool(tool_name)
                if hasattr(self.tool_registry, "get_tool")
                else None
            )
        except Exception as exc:
            msg = f"AuthorizationError: Registry lookup failed for '{tool_name}': {exc}"
            _finish_sub_call("rejected", err=msg)
            return msg

        authorized_target = getattr(tool_obj, "_target", getattr(tool_obj, "target", None)) if tool_obj is not None else None
        authorized_execute = getattr(tool_obj, "execute", None) if tool_obj is not None else None
        is_ext = bool(getattr(tool_obj, "is_external", False)) if tool_obj is not None else False
        src_id = getattr(tool_obj, "source_id", None) if tool_obj is not None else None
        src_type = getattr(tool_obj, "source_type", None) if tool_obj is not None else None

        decision = authorize_and_validate_tool_call(
            tool_name,
            arguments,
            caller_ring=self.caller_ring,
            allowed_tools=self.profile.allowed_tools,
            tool_registry=self.tool_registry if hasattr(self.tool_registry, "get_tool") else None,
            tool=tool_obj,
            argument_source="model",
        )
        if not decision.allowed:
            msg = decision.detail or f"AuthorizationError: Tool '{tool_name}' execution denied."
            try:
                ledger.record_event(
                    exec_id,
                    LedgerEventType.AUTHORIZATION_DECISION,
                    tool_id=tool_name,
                    source_id=src_id,
                    source_type=src_type,
                    is_external=is_ext,
                    status="denied",
                    correlation_id=call_id,
                    caller_ring=eff_ring,
                    metadata={"reason": decision.reason or msg},
                    idempotency_key=f"{exec_id}:{call_id}:auth",
                )
            except Exception:
                pass
            _finish_sub_call("rejected", err=msg, is_ext=is_ext, src_id=src_id, src_type=src_type)
            return msg

        tool_name = decision.tool_name
        validated_arguments = copy.deepcopy(decision.normalized_arguments)

        if hasattr(self.tool_registry, "get_tool"):
            if (
                self.tool_registry.get_tool(tool_name) is not tool_obj
                or (tool_obj is not None and getattr(tool_obj, "_target", getattr(tool_obj, "target", None)) is not authorized_target)
                or (tool_obj is not None and getattr(tool_obj, "execute", None) != authorized_execute)
            ):
                msg = f"AuthorizationError: Authorized callable for '{tool_name}' changed before execution."
                _finish_sub_call("rejected", err=msg, is_ext=is_ext, src_id=src_id, src_type=src_type)
                return msg

        try:
            ledger.record_event(
                exec_id,
                LedgerEventType.AUTHORIZATION_DECISION,
                tool_id=tool_name,
                source_id=src_id,
                source_type=src_type,
                is_external=is_ext,
                status="allowed",
                correlation_id=call_id,
                caller_ring=eff_ring,
                idempotency_key=f"{exec_id}:{call_id}:auth",
            )
            ledger.record_event(
                exec_id,
                LedgerEventType.TOOL_EXECUTION_START,
                tool_id=tool_name,
                source_id=src_id,
                source_type=src_type,
                is_external=is_ext,
                status="started",
                correlation_id=call_id,
                caller_ring=eff_ring,
                metadata={"arguments": validated_arguments},
                idempotency_key=f"{exec_id}:{call_id}:started",
            )
        except Exception:
            pass

        try:
            with execution_scope(
                ledger,
                exec_id,
                caller_ring=eff_ring,
                source_component="swarm_delegation",
                correlation_id=call_id,
                owned_by_outer=True,
            ):
                if hasattr(self.tool_registry, "execute_async"):
                    res = await self.tool_registry.execute_async(
                        tool_name,
                        caller_ring=self.caller_ring,
                        allowed_tools=self.profile.allowed_tools,
                        **copy.deepcopy(validated_arguments),
                    )
                elif hasattr(self.tool_registry, "execute"):
                    res = self.tool_registry.execute(
                        tool_name,
                        caller_ring=self.caller_ring,
                        allowed_tools=self.profile.allowed_tools,
                        **copy.deepcopy(validated_arguments),
                    )
                elif tool_obj:
                    import inspect
                    raw_res = tool_obj(**copy.deepcopy(validated_arguments))
                    res = await raw_res if inspect.isawaitable(raw_res) else raw_res
                else:
                    msg = f"Tool '{tool_name}' not found."
                    _finish_sub_call("failed", err=msg)
                    return msg

            if isinstance(res, ToolResult):
                output_str = res.output if res.success else (res.error or "Tool execution failed")
                ok = bool(res.success)
                res_meta = getattr(res, "metadata", None)
            elif hasattr(res, "output"):
                output_str = str(res.output)
                ok = True
                res_meta = getattr(res, "metadata", None)
            else:
                output_str = str(res)
                ok = True
                res_meta = None

            _finish_sub_call(
                "succeeded" if ok else "failed",
                err=None if ok else str(output_str),
                meta=res_meta if isinstance(res_meta, dict) else {"output_preview": str(output_str)[:512]},
                is_ext=is_ext,
                src_id=src_id,
                src_type=src_type,
            )
            self.write_blackboard(f"{tool_name}_result", output_str)
            return output_str
        except Exception as exc:
            err = f"Error executing '{tool_name}': {exc}"
            _finish_sub_call("failed", err=str(exc), is_ext=is_ext, src_id=src_id, src_type=src_type)
            self.write_blackboard(f"{tool_name}_error", str(exc))
            return err

    async def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Execute the subagent task in an isolated private context bubble."""
        from zelyr.engine.execution_ledger import (
            ExecutionStatus,
            LedgerEventType,
            VerificationStatus,
            execution_scope,
            get_default_central_ledger,
        )

        sub_ledger = getattr(self, "execution_ledger", None) or get_default_central_ledger()
        eff_ring = getattr(self, "caller_ring", None)
        sub_exec = sub_ledger.create_execution(
            task,
            source_component="swarm_delegation",
            caller_ring=eff_ring if isinstance(eff_ring, int) and not isinstance(eff_ring, bool) else None,
            status=ExecutionStatus.RUNNING,
            metadata={"role": self.profile.role, "agent_name": self.name},
        )
        sub_exec_id = sub_exec.execution_id
        self._emit_telemetry("swarm.agent.started", {"assigned_task": task, "role": self.profile.role, "execution_id": sub_exec_id})
        self.isolated_messages = []

        try:
            with execution_scope(
                sub_ledger,
                sub_exec_id,
                caller_ring=eff_ring if isinstance(eff_ring, int) and not isinstance(eff_ring, bool) else None,
                source_component="swarm_delegation",
                owned_by_outer=False,
            ):
                # System prompt
                if self.system_prompt:
                    self.isolated_messages.append({"role": "system", "content": self.system_prompt})

                # Pre-seed blackboard from context if provided
                if context:
                    self.isolated_messages.append({"role": "system", "content": f"[Shared Context]: {json.dumps(context)}"})
                    if isinstance(context, dict):
                        for k, v in context.items():
                            if not str(k).startswith("_"):
                                self.write_blackboard(str(k), v)

                # Task query
                self.isolated_messages.append({"role": "user", "content": task})

                # Collect allowed tool schemas
                tool_schemas: List[Dict[str, Any]] = []
                if self.tool_registry is not None and hasattr(self.tool_registry, "list_tools"):
                    registered = self.tool_registry.list_tools()
                    for t_name in self.profile.allowed_tools:
                        if t_name in registered:
                            tool_instance = registered[t_name]
                            if hasattr(tool_instance, "schema"):
                                tool_schemas.append({
                                    "type": "function",
                                    "function": {
                                        "name": t_name,
                                        "description": getattr(tool_instance, "description", ""),
                                        "parameters": tool_instance.schema,
                                    },
                                })

                # Add synthetic schemas for blackboard operations if allowed
                if "blackboard_write" in self.profile.allowed_tools:
                    tool_schemas.append({
                        "type": "function",
                        "function": {
                            "name": "blackboard_write",
                            "description": "Write a named artifact to the shared Swarm blackboard.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "key": {"type": "string", "description": "Artifact storage key."},
                                    "value": {"type": "string", "description": "Artifact content or JSON."},
                                },
                                "required": ["key", "value"],
                            },
                        },
                    })
                if "blackboard_read" in self.profile.allowed_tools:
                    tool_schemas.append({
                        "type": "function",
                        "function": {
                            "name": "blackboard_read",
                            "description": "Read a named artifact from the shared Swarm blackboard.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "key": {"type": "string", "description": "Artifact storage key to retrieve."},
                                },
                                "required": ["key"],
                            },
                        },
                    })

                max_turns = 10
                final_output = ""
                content = ""

                # Deterministic ReAct loop
                if self.llm_client is not None:
                    for turn in range(max_turns):
                        try:
                            # Invoke model
                            if hasattr(self.llm_client, "generate"):
                                resp = self.llm_client.generate(self.isolated_messages, tools=tool_schemas)
                            elif callable(self.llm_client):
                                resp = self.llm_client(self.isolated_messages, tools=tool_schemas)
                            else:
                                resp = {"content": f"Subagent {self.name} finished task."}

                            if asyncio.iscoroutine(resp):
                                resp = await resp

                            if isinstance(resp, str):
                                resp = {"content": resp}

                            content = resp.get("content", "")
                            tool_calls = resp.get("tool_calls", [])

                            if tool_calls:
                                self.isolated_messages.append({
                                    "role": "assistant",
                                    "content": content or "",
                                    "tool_calls": tool_calls,
                                })

                                for tc in tool_calls:
                                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                                    name = (fn.get("name") if isinstance(fn, dict) else None) or (tc.get("name") if isinstance(tc, dict) else None) or ""
                                    if isinstance(fn, dict) and "arguments" in fn:
                                        args = fn["arguments"]
                                    elif isinstance(tc, dict) and "arguments" in tc:
                                        args = tc["arguments"]
                                    else:
                                        args = {}
                                    if isinstance(args, str):
                                        try:
                                            args = json.loads(args)
                                        except Exception:
                                            # Preserve malformed string so execute_tool_call rejects it as non-dict
                                            pass

                                    tc_id = (tc.get("id") if isinstance(tc, dict) else None) or f"call_{name}_{turn}"
                                    tool_out = await self.execute_tool_call(name, args)

                                    self.isolated_messages.append({
                                        "role": "tool",
                                        "name": name,
                                        "tool_call_id": tc_id,
                                        "content": str(tool_out),
                                    })
                            else:
                                final_output = content
                                break
                        except Exception as exc:
                            logger.error("[%s] Error during ReAct turn %d: %s", self.name, turn, exc)
                            raise exc

                    if not final_output:
                        final_output = content or f"Subagent [{self.name}] completed {max_turns} execution turns."
                else:
                    # Deterministic fallback when no LLM is configured (e.g. offline tests)
                    final_output = f"Subagent [{self.name}] executed task: {task}"
                    self.write_blackboard("task_result", final_output)

                self.write_blackboard("final_output", final_output)
                self._emit_telemetry("swarm.agent.completed", {"result_summary": final_output})
                sub_ledger.finalize_execution(
                    sub_exec_id,
                    status=ExecutionStatus.COMPLETED_UNVERIFIED,
                    verification_status=VerificationStatus.NOT_VERIFIED,
                    final_summary=final_output,
                    metadata={"role": self.profile.role, "turns": len(self.isolated_messages)},
                )
                return AgentResult(
                    success=True,
                    output=final_output,
                    steps_taken=[f"turn_{i}" for i in range(len(self.isolated_messages))],
                    metadata={"role": self.profile.role, "turns": len(self.isolated_messages), "execution_id": sub_exec_id},
                )
        except Exception as exc:
            sub_ledger.finalize_execution(
                sub_exec_id,
                status=ExecutionStatus.FAILED,
                verification_status=VerificationStatus.INTERRUPTED,
                error=str(exc),
            )
            raise


class SubagentDelegationTool(BaseTool):
    """Tool enabling the parent orchestrator to delegate subtasks to specialized agents.

    Subagents execute inside isolated context bubbles with curated toolsets,
    coordinating state through transactional SwarmBlackboard savepoints.
    """

    REQUIRED_RING: int = 1
    required_ring: int = 1

    def __init__(
        self,
        blackboard: Optional[BlackboardStore] = None,
        tool_registry: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        event_bus: Optional[Any] = None,
        session_id: str = "default_session",
        max_depth: int = 2,
        current_depth: int = 0,
        caller_ring: int = 1,
        **kwargs: Any,
    ):
        super().__init__(
            tool_id="delegate_subtask",
            name="delegate_subtask",
            description=(
                "Delegate a focused subtask to a specialized worker subagent "
                "(coder, tester, researcher, reviewer, general) in an isolated context bubble. "
                "The subagent executes its own ReAct loop without polluting the parent conversation "
                "history, coordinates state through the shared transactional blackboard, and returns "
                "a synthesized summary with artifact references."
            ),
            tier=1,
            is_core=False,
            token_cost=150,
        )
        self.blackboard = blackboard
        self.tool_registry = tool_registry
        self.llm_client = llm_client
        self.event_bus = event_bus
        self.session_id = session_id
        self.max_depth = max_depth
        self.current_depth = current_depth
        self.caller_ring = caller_ring
        self.is_async = True

        self.add_parameter(
            ToolParameter(
                name="role",
                type="string",
                description="Target specialized subagent role ('coder', 'tester', 'researcher', 'reviewer', 'general').",
                required=True,
            )
        )
        self.add_parameter(
            ToolParameter(
                name="task",
                type="string",
                description="Concise, self-contained description of the subtask objective.",
                required=True,
            )
        )
        self.add_parameter(
            ToolParameter(
                name="context",
                type="object",
                description="Optional dictionary of input data, file paths, or blackboard keys relevant to the subtask.",
                required=False,
                default={},
            )
        )
        self.add_parameter(
            ToolParameter(
                name="timeout_seconds",
                type="integer",
                description="Execution timeout in seconds (default 60, min 1, max 300).",
                required=False,
                default=60,
            )
        )

    async def execute_async(self, *args: Any, **kwargs: Any) -> ToolResult:
        """Async invocation surface."""
        return await self.execute(*args, **kwargs)

    async def execute(  # type: ignore[override]
        self,
        role: str = "",
        task: str = "",
        context: Optional[Dict[str, Any]] = None,
        timeout_seconds: int = 60,
        caller_ring: Optional[int] = None,
        session_id: Optional[str] = None,
        subagent_override: Optional[Any] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute subagent delegation with depth, timeout, and transactional rollback protection."""
        # 1. Parameter validation: role
        if not role or not isinstance(role, str) or not role.strip():
            return ToolResult(
                success=False,
                error="InvalidRoleError: 'role' must be a non-empty string.",
            )

        norm_role = role.strip().lower()
        canonical_role = ROLE_ALIASES.get(norm_role, norm_role)
        if canonical_role not in SUBAGENT_PROFILES:
            return ToolResult(
                success=False,
                error=(
                    f"InvalidRoleError: Unknown subagent role '{role}'. "
                    f"Supported roles: {', '.join(sorted(SUBAGENT_PROFILES.keys()))}."
                ),
            )

        # 2. Parameter validation: task
        if not task or not isinstance(task, str) or not task.strip():
            return ToolResult(
                success=False,
                error="InvalidTaskError: 'task' must be a non-empty string.",
            )

        # 3. Parameter validation: context & timeout
        if isinstance(context, str):
            try:
                context = json.loads(context)
            except Exception:
                context = {"raw_context": context}
        elif context is None:
            context = {}

        try:
            timeout = float(timeout_seconds if timeout_seconds is not None else 60.0)
            timeout = min(max(timeout, 1.0), 300.0)
        except (ValueError, TypeError):
            timeout = 60.0

        # 4. Recursion depth ceiling enforcement
        active_depth = _current_delegation_depth.get()
        effective_current_depth = max(self.current_depth, active_depth)
        if effective_current_depth >= self.max_depth:
            return ToolResult(
                success=False,
                error=(
                    f"DelegationDepthExceeded: Maximum delegation depth of {self.max_depth} "
                    f"reached (current depth: {effective_current_depth})."
                ),
            )

        # 5. Dependency resolution & capability ring
        eff_session_id = session_id or self.session_id or "default_session"
        eff_ring = max(self.caller_ring, caller_ring if caller_ring is not None else 1)

        if self.blackboard is None:
            self.blackboard = BlackboardStore()

        profile = SUBAGENT_PROFILES[canonical_role]

        # 6. Transactional savepoint: wrap execution in a Blackboard snapshot
        txn_id = self.blackboard.begin_transaction(eff_session_id)
        pre_keys = set(self.blackboard.list_keys(eff_session_id, profile.agent_name))

        # 7. Instantiate worker agent in isolated context
        worker = subagent_override or DelegatedSubagent(
            profile=profile,
            session_id=eff_session_id,
            blackboard=self.blackboard,
            tool_registry=self.tool_registry,
            llm_client=self.llm_client,
            caller_ring=eff_ring,
            event_bus=self.event_bus,
            current_depth=effective_current_depth + 1,
            max_depth=self.max_depth,
        )

        # Increment delegation depth for child tasks
        token = _current_delegation_depth.set(effective_current_depth + 1)
        try:
            # 8. Dispatch with bounded timeout
            result: AgentResult = await asyncio.wait_for(
                worker.run(task, context=context),
                timeout=timeout,
            )

            if result.success:
                # Commit transactional savepoint on success
                self.blackboard.commit_transaction(eff_session_id, txn_id)
                post_keys = set(self.blackboard.list_keys(eff_session_id, profile.agent_name))
                new_keys = list(post_keys - pre_keys) or list(post_keys)

                return ToolResult(
                    success=True,
                    output=result.output,
                    metadata={
                        "blackboard_keys": new_keys,
                        "role": canonical_role,
                        "session_id": eff_session_id,
                        "agent_name": profile.agent_name,
                        "delegation_depth": effective_current_depth + 1,
                    },
                )
            else:
                # Rollback on worker reported failure
                self.blackboard.rollback_transaction(eff_session_id, txn_id)
                return ToolResult(
                    success=False,
                    error=f"SubagentExecutionError: {result.error or 'Subagent reported failure.'}",
                )

        except asyncio.TimeoutError:
            self.blackboard.rollback_transaction(eff_session_id, txn_id)
            return ToolResult(
                success=False,
                error=f"SubagentExecutionError: Subagent '{canonical_role}' timed out after {timeout} seconds.",
            )
        except Exception as exc:
            self.blackboard.rollback_transaction(eff_session_id, txn_id)
            return ToolResult(
                success=False,
                error=f"SubagentExecutionError: {exc}",
            )
        finally:
            _current_delegation_depth.reset(token)
