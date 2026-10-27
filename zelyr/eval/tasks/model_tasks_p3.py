"""Opt-in Level 2 real-model evaluation tasks (Phase 3) for Zelyr orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from zelyr.db.memory import MemoryStore
from zelyr.eval.harness import check_ollama_status_async
from zelyr.eval.task import EvalContext, EvalTask
from zelyr.eval.validators import (
    assert_file_content,
    assert_file_exists,
    assert_json_field,
    assert_no_tool_errors,
    assert_regex_match,
    assert_tool_called,
)
from zelyr.tools.core import ToolResult


def check_ollama_status(
    url: str = "http://127.0.0.1:11434",
    model_name: Optional[str] = None,
    timeout: float = 1.0,
) -> Tuple[bool, str]:
    """Synchronously verify whether local Ollama daemon is reachable and whether model is present."""
    try:
        resp = httpx.get(f"{url.rstrip('/')}/api/tags", timeout=timeout)
        if resp.status_code != 200:
            return False, f"Ollama HTTP {resp.status_code}"
        if model_name:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            clean_target = model_name.split(":")[0]
            if not any(m == model_name or m.startswith(clean_target) for m in models):
                return False, f"Model '{model_name}' not found in Ollama models: {models}"
        return True, "Ollama ready"
    except Exception as exc:
        return False, f"Could not connect to Ollama at {url}: {exc}"


# Standard schemas for file operations used by real models
FILE_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the content of a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file in workspace"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to target file in workspace"},
                    "content": {"type": "string", "description": "Content string to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
]


def _make_file_tools():
    def read_file(path: str, context: EvalContext) -> dict:
        target = context.resolve_path(path)
        if not target.is_file():
            return {"success": False, "error": f"File '{path}' does not exist"}
        return {"success": True, "path": path, "content": target.read_text(encoding="utf-8")}

    def write_file(path: str, content: str, context: EvalContext) -> dict:
        target = context.resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"success": True, "path": path, "bytes_written": len(content)}

    return {"read_file": read_file, "write_file": write_file}


def build_task_live_workspace_diagnostic(
    model: str = "qwen3:8b",
    base_url: str = "http://127.0.0.1:11434",
) -> EvalTask:
    """Level 2 task: Real model inspects crash log and writes a diagnosis JSON file."""
    def setup(workspace: Path, memory: MemoryStore) -> None:
        log_content = (
            "[2026-10-09 10:14:02.129] CRITICAL core_worker: Worker thread 4 crashed "
            "with OutOfMemoryError in worker_pool.py:128\n"
            "[2026-10-09 10:14:02.130] ERROR supervisor: Child process died unexpectedly\n"
        )
        (workspace / "crash.log").write_text(log_content, encoding="utf-8")

    async def validator(ctx: EvalContext) -> Tuple[bool, str]:
        diag_path = ctx.workspace_dir / "diagnosis.json"
        if not diag_path.is_file():
            return False, "Required artifact 'diagnosis.json' was not created."

        try:
            data = json.loads(diag_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as err:
            return False, f"'diagnosis.json' is not valid JSON: {err}"

        # Tolerant extraction of component and error fields
        component = str(data.get("component", "") or data.get("failing_component", "")).lower()
        error = str(data.get("error_cause", "") or data.get("error", "") or data.get("cause", "")).lower()

        if "worker" not in component and "core_worker" not in component:
            return False, f"Expected failing component 'worker' or 'core_worker', got '{component}'"

        if "memory" not in error and "oom" not in error and "outofmemory" not in error:
            return False, f"Expected error cause to reference OutOfMemory/OOM, got '{error}'"

        return True, "Workspace crash diagnosis successfully identified component and OutOfMemoryError."

    return EvalTask(
        task_id="task_live_workspace_diagnostic",
        description="Diagnose workspace crash log and generate diagnosis.json",
        level=2,
        allowed_tools=["read_file", "write_file"],
        custom_tools=_make_file_tools(),
        tool_schemas=FILE_TOOL_SCHEMAS,
        max_steps=5,
        timeout_seconds=45.0,
        setup_fn=setup,
        validator_fn=validator,
        user_prompt=(
            "Inspect the system crash log in 'crash.log' within your workspace. Identify the failing component "
            "and the root error cause. Then use the write_file tool to save a JSON report into 'diagnosis.json' "
            "with keys 'component' and 'error_cause'."
        ),
        metadata={
            "use_live_model": True,
            "level": 2,
            "model": model,
            "base_url": base_url,
        },
    )


def build_task_live_metric_formatting(
    model: str = "qwen3:8b",
    base_url: str = "http://127.0.0.1:11434",
) -> EvalTask:
    """Level 2 task: Real model computes metric averages and records structured JSON summary."""
    def setup(workspace: Path, memory: MemoryStore) -> None:
        raw_metrics = [
            {"request_id": 1, "latency_ms": 100.0, "status": "ok"},
            {"request_id": 2, "latency_ms": 200.0, "status": "error"},
            {"request_id": 3, "latency_ms": 300.0, "status": "ok"},
        ]
        (workspace / "metrics.json").write_text(json.dumps(raw_metrics, indent=2), encoding="utf-8")

    async def validator(ctx: EvalContext) -> Tuple[bool, str]:
        sum_path = ctx.workspace_dir / "summary.json"
        if not sum_path.is_file():
            return False, "Required artifact 'summary.json' was not created."

        try:
            data = json.loads(sum_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as err:
            return False, f"'summary.json' is not valid JSON: {err}"

        # Validate avg_latency (expected 200.0)
        avg_lat = data.get("avg_latency") or data.get("average_latency")
        if avg_lat is None:
            return False, "Missing 'avg_latency' in summary.json"
        try:
            val_lat = float(avg_lat)
            if not (190.0 <= val_lat <= 210.0):
                return False, f"Expected avg_latency ~200.0, got {val_lat}"
        except (ValueError, TypeError):
            return False, f"'avg_latency' is not numeric: {avg_lat}"

        # Validate total_errors (expected 1)
        err_cnt = data.get("total_errors") or data.get("errors") or data.get("error_count")
        if err_cnt is None:
            return False, "Missing 'total_errors' in summary.json"
        try:
            val_err = int(err_cnt)
            if val_err != 1:
                return False, f"Expected total_errors 1, got {val_err}"
        except (ValueError, TypeError):
            return False, f"'total_errors' is not integer: {err_cnt}"

        return True, "Summary metrics accurately calculated: avg_latency=200, total_errors=1."

    return EvalTask(
        task_id="task_live_metric_formatting",
        description="Compute summary statistics from metrics.json into summary.json",
        level=2,
        allowed_tools=["read_file", "write_file"],
        custom_tools=_make_file_tools(),
        tool_schemas=FILE_TOOL_SCHEMAS,
        max_steps=5,
        timeout_seconds=45.0,
        setup_fn=setup,
        validator_fn=validator,
        user_prompt=(
            "Read 'metrics.json' in your workspace. Calculate the average latency ('avg_latency') across all entries "
            "and count the total number of entries with status 'error' ('total_errors'). "
            "Use the write_file tool to write a JSON document to 'summary.json' with keys 'avg_latency' and 'total_errors'."
        ),
        metadata={
            "use_live_model": True,
            "level": 2,
            "model": model,
            "base_url": base_url,
        },
    )


WEB_RESEARCH_TOOL_SCHEMAS = [
    *FILE_TOOL_SCHEMAS,
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Searches the web for candidate information and links.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query keywords"},
                    "num_results": {"type": "integer", "description": "Number of results to return (1-6)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_web_page",
            "description": "Fetches and extracts clean, readable text from a web page URL with SSRF protection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The HTTP or HTTPS URL of the webpage to fetch."},
                    "max_chars": {"type": "integer", "description": "Maximum characters to extract (default 5000, max 8000)"},
                },
                "required": ["url"],
            },
        },
    },
]


def _make_web_research_tools():
    tools = _make_file_tools()

    async def web_search(query: str, num_results: int = 5, **kwargs) -> dict:
        try:
            from zelyr.tools.web_search import web_search as real_web_search
            output = await real_web_search(query, num_results=num_results)
            return {"success": True, "output": output}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def fetch_web_page(url: str, max_chars: int = 5000, **kwargs) -> dict:
        try:
            from zelyr.tools.web_search import fetch_web_page as real_fetch_web_page
            output = await real_fetch_web_page(url, max_chars=max_chars)
            return {"success": True, "output": output}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    tools["web_search"] = web_search
    tools["fetch_web_page"] = fetch_web_page
    return tools


async def _validate_web_research_synthesis(ctx: EvalContext) -> Tuple[bool, str]:
    """Validate observable state and trajectory for web research and synthesis workflow."""
    # 1. Artifact Check: Verify research_summary.md was created
    ok_file, msg_file = assert_file_exists(ctx, "research_summary.md")
    if not ok_file:
        return False, msg_file

    # 2. Grounded Facts Check: Verify core facts from fetched documentation are present
    ok_feat, msg_feat = assert_file_content(ctx, "research_summary.md", "Quantum-Safe Encryption")
    if not ok_feat:
        return False, f"Missing core feature fact: {msg_feat}"

    ok_date, msg_date = assert_file_content(ctx, "research_summary.md", "2026-11-15")
    if not ok_date:
        return False, f"Missing release date fact: {msg_date}"

    # 3. Source Attribution Check: Verify exact source URL is cited
    ok_url, msg_url = assert_regex_match(
        ctx,
        "research_summary.md",
        r"https://example\.com/chimera-v2-docs",
    )
    if not ok_url:
        return False, f"Missing source attribution URL: {msg_url}"

    # 4. Tool Trajectory Check: Verify both web_search and fetch_web_page were invoked
    ok_search, msg_search = assert_tool_called(ctx, "web_search", min_count=1)
    if not ok_search:
        return False, msg_search

    ok_fetch, msg_fetch = assert_tool_called(ctx, "fetch_web_page", min_count=1)
    if not ok_fetch:
        return False, msg_fetch

    # 5. Tool Health Check: Verify no unhandled tool exceptions
    ok_health, msg_health = assert_no_tool_errors(ctx)
    if not ok_health:
        return False, msg_health

    return True, "Web research synthesis verified: search executed, page fetched, facts grounded, and source URL cited."


def build_task_live_web_research_synthesis(
    model: str = "qwen3:8b",
    base_url: str = "http://127.0.0.1:11434",
) -> EvalTask:
    """Level 2 task: Multi-step web research, document fetch, and cited synthesis."""
    return EvalTask(
        task_id="task_live_web_research_synthesis",
        description="Multi-step web research: search topic, fetch candidate documentation, synthesize summary with citations.",
        level=2,
        allowed_tools=["web_search", "fetch_web_page", "write_file"],
        custom_tools=_make_web_research_tools(),
        tool_schemas=WEB_RESEARCH_TOOL_SCHEMAS,
        max_steps=5,
        timeout_seconds=60.0,
        validator_fn=_validate_web_research_synthesis,
        user_prompt=(
            "Research the release date and core security features of 'Project Chimera v2'. "
            "Search the web, fetch the official documentation page, and synthesize a concise "
            "summary into 'research_summary.md' with explicit source URL citations."
        ),
        metadata={
            "use_live_model": True,
            "level": 2,
            "model": model,
            "base_url": base_url,
        },
    )


task_live_web_research_synthesis = build_task_live_web_research_synthesis()


def get_model_tasks_p3(
    model: str = "qwen3:8b",
    base_url: str = "http://127.0.0.1:11434",
) -> List[EvalTask]:
    """Retrieve all Level 2 live model evaluation tasks."""
    return [
        build_task_live_workspace_diagnostic(model=model, base_url=base_url),
        build_task_live_metric_formatting(model=model, base_url=base_url),
        build_task_live_web_research_synthesis(model=model, base_url=base_url),
    ]


__all__ = [
    "check_ollama_status",
    "build_task_live_workspace_diagnostic",
    "build_task_live_metric_formatting",
    "build_task_live_web_research_synthesis",
    "task_live_web_research_synthesis",
    "get_model_tasks_p3",
]
