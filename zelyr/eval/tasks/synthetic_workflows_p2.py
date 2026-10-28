"""Synthetic workflow evaluation tasks (Phase 2) for Zelyr orchestration."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.db.memory import MemoryStore
from zelyr.eval.metrics import EvalStatus
from zelyr.eval.task import EvalContext, EvalTask
from zelyr.eval.validators import (
    assert_file_exists,
    assert_json_field,
    assert_memory_record,
    assert_tool_called,
    assert_tool_sequence,
    assert_tool_updated,
)
from zelyr.tools.core import ToolResult
from zelyr.tools.tool_crafter import craft_tool
from zelyr.tools.tool_manager import ToolManager


# ---------------------------------------------------------------------------
# Task 1: Multi-Step Diagnostic Triage
# ---------------------------------------------------------------------------

def build_task_synthetic_service_triage() -> EvalTask:
    """Task 1: Multi-step diagnostic triage.

    Model inspects simulated service states, reads logs, isolates a database
    out-of-memory (OOM) failure, and produces a structured triage report artifact.
    """
    def setup(workspace: Path, memory: MemoryStore) -> None:
        services_data = {"web": "active", "db": "failed", "redis": "active"}
        (workspace / "services.json").write_text(json.dumps(services_data, indent=2), encoding="utf-8")
        (workspace / "db.log").write_text("FATAL: Out of memory; killed PID 4120\n", encoding="utf-8")

    def mock_inspect_services(workspace: Path) -> dict:
        p = workspace / "services.json"
        return json.loads(p.read_text(encoding="utf-8"))

    def mock_read_logs(service: str, workspace: Path) -> dict:
        log_file = workspace / f"{service}.log"
        if not log_file.exists():
            return {"success": False, "error": f"Log file for service '{service}' not found"}
        return {"success": True, "service": service, "log": log_file.read_text(encoding="utf-8")}

    def write_file(path: str, content: str, workspace: Path) -> dict:
        target = workspace / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"success": True, "path": path}

    mock_responses = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "triage_call_1",
                    "function": {"name": "mock_inspect_services", "arguments": {}},
                }
            ],
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "triage_call_2",
                    "function": {"name": "mock_read_logs", "arguments": {"service": "db"}},
                }
            ],
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "triage_call_3",
                    "function": {
                        "name": "write_file",
                        "arguments": {
                            "path": "triage_report.json",
                            "content": json.dumps({"failed_service": "db", "root_cause": "OOM"}),
                        },
                    },
                }
            ],
        },
        {
            "role": "assistant",
            "content": "Diagnostic triage completed: identified 'db' service failure due to OOM condition. Report written to triage_report.json.",
        },
    ]

    async def validator(ctx: EvalContext) -> Tuple[bool, str]:
        # 1. Assert JSON fields in generated artifact
        ok_failed, msg_failed = assert_json_field(ctx.workspace_dir, "triage_report.json", "failed_service", "db")
        if not ok_failed:
            return False, msg_failed

        ok_cause, msg_cause = assert_json_field(ctx.workspace_dir, "triage_report.json", "root_cause", "OOM")
        if not ok_cause:
            return False, msg_cause

        # 2. Assert tool execution sequence
        ok_seq, msg_seq = assert_tool_sequence(
            ctx.tool_calls,
            ["mock_inspect_services", "mock_read_logs", "write_file"],
        )
        if not ok_seq:
            return False, msg_seq

        return True, "Diagnostic triage verified: isolated failed service and persisted structured triage_report.json."

    return EvalTask(
        task_id="task_synthetic_service_triage",
        description="Multi-step diagnostic service triage isolating OOM failure and producing report",
        level=2,
        allowed_tools=["mock_inspect_services", "mock_read_logs", "write_file"],
        custom_tools={
            "mock_inspect_services": mock_inspect_services,
            "mock_read_logs": mock_read_logs,
            "write_file": write_file,
        },
        tool_schemas=[
            {
                "type": "function",
                "function": {
                    "name": "mock_inspect_services",
                    "description": "Query status of all registered system services",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "mock_read_logs",
                    "description": "Read log file for a specific service",
                    "parameters": {
                        "type": "object",
                        "properties": {"service": {"type": "string"}},
                        "required": ["service"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write text content to a file at the given relative path",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
        ],
        mock_responses=mock_responses,
        setup_fn=setup,
        validator_fn=validator,
    )


# ---------------------------------------------------------------------------
# Task 2: Closed-Loop Tool Crafting & Self-Repair
# ---------------------------------------------------------------------------

PARSE_METRICS_V1_CODE = '''"""Dynamic parse_metrics tool v1 (strict single space split)."""
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "parse_metrics",
        "description": "Parse raw metrics file into numeric key-values",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}
REQUIRED_RING = 1
TIER = 1
TUI_HINT = "Parsing metrics v1..."
# marker: VERSION_1

async def execute(path: str = "test", **kwargs):
    import os
    if path == "test" and not os.path.exists("test"):
        return {"cpu": 0, "mem": 0}
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    parts = content.split(" ")
    if len(parts) != 4:
        raise IndexError("IndexError: line split failed")
    return {"cpu": int(parts[1]), "mem": int(parts[3])}
'''

PARSE_METRICS_V2_CODE = '''"""Dynamic parse_metrics tool v2 (robust regex delimiter parser)."""
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "parse_metrics",
        "description": "Parse raw metrics file into numeric key-values v2 repaired",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}
REQUIRED_RING = 1
TIER = 1
TUI_HINT = "Parsing metrics v2..."
# marker: VERSION_2_REPAIRED

async def execute(path: str = "test", **kwargs):
    import re, os
    if path == "test" and not os.path.exists("test"):
        return {"cpu": 0, "mem": 0}
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    m = re.search(r"cpu:\\s*(\\d+)\\s+mem:\\s*(\\d+)", content)
    if not m:
        raise ValueError("Failed regex match on metrics")
    return {"cpu": int(m.group(1)), "mem": int(m.group(2))}
'''


def build_task_tool_crafting_self_repair() -> EvalTask:
    """Task 2: Closed-loop tool crafting and self-repair.

    Model synthesizes tool using craft_tool, encounters an edge-case IndexError
    on uneven whitespace, repairs it via ToolManager.update_tool(), and executes successfully.
    """
    def setup(workspace: Path, memory: MemoryStore) -> None:
        (workspace / "metrics.raw").write_text("cpu:  45    mem:   82\n", encoding="utf-8")

    async def custom_craft_tool(name: str, code: str, context: EvalContext, **kwargs) -> ToolResult:
        tools_dir = context.tools_directory
        db_path = str(context.db_path)
        return await craft_tool(name=name, code=code, tools_dir=tools_dir, db_path=db_path)

    async def custom_manage_tools(action: str, name: str, code: str = "", context: EvalContext = None, **kwargs) -> ToolResult:
        tools_dir = context.tools_directory if context else None
        db_path = str(context.db_path) if context else None
        if action == "update":
            return await ToolManager.update_tool(name=name, code=code, tools_dir=tools_dir, db_path=db_path)
        return ToolResult(success=False, error=f"Unsupported manage_tools action '{action}'")

    async def custom_parse_metrics(path: str = "metrics.raw", context: EvalContext = None, **kwargs) -> ToolResult:
        tools_dir = context.tools_directory
        tool_file = tools_dir / "parse_metrics.py"
        if not tool_file.exists():
            return ToolResult(success=False, error="Dynamic tool 'parse_metrics' is not installed.")

        full_path = str(context.resolve_path(path))
        mod_name = f"zelyr.eval_dyn_parse_metrics_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(mod_name, tool_file)
        if not spec or not spec.loader:
            return ToolResult(success=False, error="Failed loading parse_metrics module specification.")
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            res = await mod.execute(path=full_path)
            return ToolResult(success=True, output=res)
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    mock_responses = [
        # Turn 1: Craft tool v1
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "repair_call_1",
                    "function": {
                        "name": "craft_tool",
                        "arguments": {"name": "parse_metrics", "code": PARSE_METRICS_V1_CODE},
                    },
                }
            ],
        },
        # Turn 2: Dispatch parse_metrics (fails with IndexError)
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "repair_call_2",
                    "function": {"name": "parse_metrics", "arguments": {"path": "metrics.raw"}},
                }
            ],
        },
        # Turn 3: Repair tool via manage_tools update
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "repair_call_3",
                    "function": {
                        "name": "manage_tools",
                        "arguments": {
                            "action": "update",
                            "name": "parse_metrics",
                            "code": PARSE_METRICS_V2_CODE,
                        },
                    },
                }
            ],
        },
        # Turn 4: Re-dispatch parse_metrics (succeeds)
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "repair_call_4",
                    "function": {"name": "parse_metrics", "arguments": {"path": "metrics.raw"}},
                }
            ],
        },
        # Turn 5: Completion text
        {
            "role": "assistant",
            "content": "Tool self-repair loop complete: diagnosed split failure on uneven whitespace and updated parser to v2.",
        },
    ]

    async def validator(ctx: EvalContext) -> Tuple[bool, str]:
        # 1. Assert v2 repaired code is present on disk
        ok_marker, msg_marker = assert_tool_updated(ctx.tools_directory, "parse_metrics", "VERSION_2_REPAIRED")
        if not ok_marker:
            return False, msg_marker

        # 2. Assert tool execution sequence and outcome
        ok_called, msg_called = assert_tool_called(ctx.tool_calls, "parse_metrics", min_count=2)
        if not ok_called:
            return False, msg_called

        # Verify initial failure then eventual success
        parse_calls = [c for c in ctx.tool_calls if c.get("name") == "parse_metrics"]
        if len(parse_calls) < 2:
            return False, f"Expected at least 2 parse_metrics invocations, found {len(parse_calls)}"

        first_res = parse_calls[0].get("result")
        last_res = parse_calls[-1].get("result")

        first_ok = getattr(first_res, "success", None) if first_res is not None else None
        if first_ok is None and isinstance(first_res, dict):
            first_ok = first_res.get("success")

        last_ok = getattr(last_res, "success", None) if last_res is not None else None
        if last_ok is None and isinstance(last_res, dict):
            last_ok = last_res.get("success")

        if first_ok is not False:
            return False, "Initial parse_metrics invocation did not report expected failure."
        if last_ok is not True:
            return False, "Repaired parse_metrics invocation did not achieve success."

        return True, "Tool crafting and autonomous self-repair loop verified: v2 marker confirmed on disk and execution succeeded."

    return EvalTask(
        task_id="task_tool_crafting_self_repair",
        description="Autonomous dynamic tool crafting, exception detection, and self-repair via ToolManager",
        level=3,
        allowed_tools=["craft_tool", "parse_metrics", "manage_tools"],
        custom_tools={
            "craft_tool": custom_craft_tool,
            "manage_tools": custom_manage_tools,
            "parse_metrics": custom_parse_metrics,
        },
        tool_schemas=[
            {
                "type": "function",
                "function": {
                    "name": "craft_tool",
                    "description": "Synthesize a new dynamic tool",
                    "parameters": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "code": {"type": "string"}},
                        "required": ["name", "code"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "manage_tools",
                    "description": "Inspect, update, or remove dynamic tools",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string"},
                            "name": {"type": "string"},
                            "code": {"type": "string"},
                        },
                        "required": ["action", "name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "parse_metrics",
                    "description": "Parse metrics data file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
        ],
        mock_responses=mock_responses,
        setup_fn=setup,
        validator_fn=validator,
    )


# ---------------------------------------------------------------------------
# Task 3: Multi-Session Memory Persistence & Hydration
# ---------------------------------------------------------------------------

def build_task_multisession_memory_persistence() -> EvalTask:
    """Task 3: Multi-session memory persistence & prompt hydration.

    Session 1 persists facts via remember_fact tool into an isolated SQLite store.
    Session 2 runs a second generate_stream call verifying prompt hydration injects
    [Contextual Memory] with the recalled IP address.
    """
    def remember_fact_tool(
        content: str = "",
        fact: str = "",
        category: str = "preference",
        key: Optional[str] = None,
        context: EvalContext = None,
        **kwargs,
    ) -> ToolResult:
        c = content or fact
        if not c:
            return ToolResult(success=False, error="Parameter 'content' or 'fact' is required.")
        store = context.memory_store or MemoryStore(db_path=str(context.db_path))
        mem_id = store.store_fact(content=c, category=category, key=key)
        return ToolResult(
            success=True,
            output={
                "message": f"Successfully stored memory #{mem_id}: {c}",
                "memory_id": mem_id,
                "content": c,
                "category": category,
                "key": key,
            },
        )

    mock_responses = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "mem_call_1",
                    "function": {
                        "name": "remember_fact",
                        "arguments": {
                            "content": "Server alpha IP is 192.168.1.150",
                            "category": "homelab",
                            "key": "server.alpha.ip",
                        },
                    },
                }
            ],
        },
        {
            "role": "assistant",
            "content": "Stored fact in long-term memory: Server alpha IP is 192.168.1.150.",
        },
    ]

    async def validator(ctx: EvalContext) -> Tuple[bool, str]:
        # 1. Assert record exists in isolated SQLite database
        ok_rec, msg_rec = assert_memory_record(ctx.db_path, "homelab", "server.alpha.ip", "192.168.1.150")
        if not ok_rec:
            return False, msg_rec

        # 2. Session 2 execution: Run a second generate_stream call on the same DB
        session_2_messages: List[Dict[str, Any]] = []

        async def session_2_stream_provider(payload: dict):
            session_2_messages.extend(payload.get("messages", []))
            yield {"choices": [{"delta": {"content": "Server alpha is at 192.168.1.150."}}]}

        orch_2 = NativeOrchestrator(
            max_depth=3,
            stream_provider=session_2_stream_provider,
            db_path=str(ctx.db_path),
        )

        payload_2 = {
            "messages": [{"role": "user", "content": "Where is server alpha?"}],
        }

        async for _ in orch_2.generate_stream(payload_2):
            pass

        # Verify session 2 payload messages contained [Contextual Memory] with the IP
        has_context_mem = False
        has_ip = False
        for m in session_2_messages:
            content_str = str(m.get("content", ""))
            if "[Contextual Memory]" in content_str:
                has_context_mem = True
            if "192.168.1.150" in content_str:
                has_ip = True

        if not has_context_mem:
            return False, "Session 2 system prompt was missing '[Contextual Memory]' block."
        if not has_ip:
            return False, "Session 2 '[Contextual Memory]' did not contain the stored IP 192.168.1.150."

        return True, "Multi-session persistence verified: fact stored in isolated SQLite and hydrated in Session 2 prompt."

    return EvalTask(
        task_id="task_multisession_memory_persistence",
        description="Multi-session persistence via remember_fact and hydration verification in subsequent session",
        level=2,
        allowed_tools=["remember_fact"],
        custom_tools={"remember_fact": remember_fact_tool},
        tool_schemas=[
            {
                "type": "function",
                "function": {
                    "name": "remember_fact",
                    "description": "Store an important fact or configuration into persistent memory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "category": {"type": "string"},
                            "key": {"type": "string"},
                        },
                        "required": ["content"],
                    },
                },
            }
        ],
        mock_responses=mock_responses,
        validator_fn=validator,
    )


# ---------------------------------------------------------------------------
# Task 4: Error Escalation & Graceful Degradation
# ---------------------------------------------------------------------------

def build_task_unrecoverable_actuator_containment() -> EvalTask:
    """Task 4: Error escalation and graceful degradation.

    Mock tool failing_actuator always fails with unrecoverable hardware failure.
    Model receives error, attempts no forbidden actions, and acknowledges failure cleanly.
    Task classifies as FAIL or TOOL_ERROR with zero false-success claims.
    """
    def failing_actuator(action: str = "engage", **kwargs) -> ToolResult:
        return ToolResult(
            success=False,
            error="Hardware failure: device actuator not responding",
            remedy_hint="",
            allowed_actions=[],
        )

    mock_responses = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "actuator_call_1",
                    "function": {"name": "failing_actuator", "arguments": {"action": "engage"}},
                }
            ],
        },
        {
            "role": "assistant",
            "content": "Execution halted: Hardware failure encountered on device actuator. No recovery actions available.",
        },
    ]

    async def validator(ctx: EvalContext) -> Tuple[bool, str]:
        r_lower = ctx.final_response.lower()
        if any(w in r_lower for w in ["successfully", "completed", "finished", "all done"]):
            return False, "Model falsely claimed completion despite hardware failure."

        # Verify failing_actuator was dispatched
        ok_called, msg_called = assert_tool_called(ctx.tool_calls, "failing_actuator", min_count=1)
        if not ok_called:
            return False, msg_called

        # Return False to trigger failure containment status (FAIL / VALIDATION_FAILED)
        return False, "Hardware failure encountered on actuator; task halted cleanly as expected."

    return EvalTask(
        task_id="task_unrecoverable_actuator_containment",
        description="Error escalation and graceful degradation when recovery options are exhausted",
        level=2,
        allowed_tools=["failing_actuator"],
        custom_tools={"failing_actuator": failing_actuator},
        expected_status=EvalStatus.FAIL,
        tool_schemas=[
            {
                "type": "function",
                "function": {
                    "name": "failing_actuator",
                    "description": "Trigger physical device actuator",
                    "parameters": {
                        "type": "object",
                        "properties": {"action": {"type": "string"}},
                    },
                },
            }
        ],
        mock_responses=mock_responses,
        validator_fn=validator,
    )


def get_synthetic_workflows_p2_tasks() -> List[EvalTask]:
    """Return all Level 2/3 synthetic workflow evaluation tasks."""
    return [
        build_task_synthetic_service_triage(),
        build_task_tool_crafting_self_repair(),
        build_task_multisession_memory_persistence(),
        build_task_unrecoverable_actuator_containment(),
    ]
