"""Offline evaluation harness executing tasks with mocked streams and isolated sandboxes."""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import httpx

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.db.memory import MemoryStore
from zelyr.eval.metrics import EvalStatus, TaskResult
from zelyr.eval.provider import MockTurn
from zelyr.eval.task import EvalContext, EvalTask
from zelyr.eval.types import EvaluationResult, ResultStatus, ValidationResult
from zelyr.tools.core import ToolResult


async def check_ollama_status_async(
    url: str = "http://127.0.0.1:11434",
    model_name: Optional[str] = None,
    timeout: float = 1.0,
) -> Tuple[bool, str]:
    """Check whether local Ollama daemon is reachable and whether model is pulled."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{url.rstrip('/')}/api/tags")
            if resp.status_code != 200:
                return False, f"Ollama HTTP {resp.status_code}"
            if model_name:
                models = [m.get("name", "") for m in resp.json().get("models", [])]
                has_model = (
                    model_name in models
                    or f"{model_name}:latest" in models
                    or (":" not in model_name and any(m.startswith(f"{model_name}:") for m in models))
                )
                if not has_model:
                    return False, f"Model '{model_name}' not found in Ollama models: {models}"
            return True, "Ollama ready"
    except Exception as exc:
        return False, f"Could not connect to Ollama at {url}: {exc}"


class LiveOllamaStreamAdapter:
    """Live streaming adapter connecting to local Ollama with token and latency telemetry."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str = "qwen3:8b"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.measured_tokens: int = 0
        self.model_latencies: List[float] = []

    async def stream_chat(self, payload: dict):
        payload["model"] = self.model
        t0 = time.perf_counter()

        from zelyr.config import get_config
        options = dict(payload.get("options") or {})
        eff_ctx = get_config().get_effective_num_ctx(options.get("num_ctx"))
        options["num_ctx"] = eff_ctx
        options.setdefault("temperature", 0.1)

        timeout_config = httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": payload.get("messages", []),
                    "tools": payload.get("tools", []),
                    "stream": True,
                    "options": options,
                },
            ) as resp:
                if resp.status_code != 200:
                    err_b = await resp.aread()
                    yield {"_api_error": f"HTTP {resp.status_code}: {err_b.decode('utf-8', errors='replace')}"}
                    return

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Terminal metadata chunk from Ollama
                    if chunk.get("done"):
                        pec = chunk.get("prompt_eval_count") or 0
                        ec = chunk.get("eval_count") or 0
                        self.measured_tokens += (pec + ec)

                    # Also support OpenAI-style usage dict if present
                    if "usage" in chunk:
                        u = chunk["usage"]
                        pt = u.get("prompt_tokens") or 0
                        ct = u.get("completion_tokens") or 0
                        self.measured_tokens += (pt + ct)

                    msg = chunk.get("message", {})
                    content = msg.get("content", "")
                    tool_calls = msg.get("tool_calls", [])

                    delta = {}
                    if content:
                        delta["content"] = content
                    if tool_calls:
                        delta["tool_calls"] = tool_calls

                    if delta:
                        yield {"choices": [{"delta": delta}]}

        latency = time.perf_counter() - t0
        self.model_latencies.append(latency)


class EvalHarness:
    """100% offline, deterministic orchestration evaluation harness."""

    async def run_task(
        self,
        task: EvalTask,
        mock_responses: Optional[List[Dict[str, Any]]] = None,
        mock_stream_fn: Optional[Callable[[dict], Any]] = None,
        custom_tools: Optional[Dict[str, Callable[..., Any]]] = None,
        use_live_model: bool = False,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        live_timeout: Optional[float] = None,
    ) -> TaskResult:
        """Execute an EvalTask inside an isolated workspace and validate observable state."""
        start_time = time.perf_counter()

        requires_live = bool(
            (task.metadata and task.metadata.get("use_live_model", False))
            and mock_responses is None
            and not task.mock_responses
            and mock_stream_fn is None
        )
        if requires_live and not use_live_model:
            return TaskResult(
                task_id=task.task_id,
                status=EvalStatus.UNVERIFIED,
                message="Live model evaluation skipped in offline mode (pass --live to execute).",
                step_count=0,
                tool_calls=[],
                duration_seconds=0.0,
                measured_tokens=None,
                estimated_tokens=0,
                failure_category="UNVERIFIED",
                artifacts={"reason": "offline_mode", "use_live_model": False, "expected_status": None},
            )

        eff_use_live_model = bool(
            use_live_model
            and mock_responses is None
            and mock_stream_fn is None
        )
        eff_model = model or (task.metadata.get("model") if task.metadata else None) or "qwen3:8b"
        eff_base_url = base_url or (task.metadata.get("base_url") if task.metadata else None) or "http://127.0.0.1:11434"

        if eff_use_live_model:
            effective_timeout = live_timeout if live_timeout is not None else 60.0
        else:
            effective_timeout = task.timeout_seconds

        if eff_use_live_model:
            is_ready, ready_reason = await check_ollama_status_async(eff_base_url, eff_model)
            if not is_ready:
                return TaskResult(
                    task_id=task.task_id,
                    status=EvalStatus.UNVERIFIED,
                    message=f"Ollama unavailable: {ready_reason}",
                    step_count=0,
                    tool_calls=[],
                    duration_seconds=0.0,
                    measured_tokens=None,
                    estimated_tokens=0,
                    failure_category="UNVERIFIED",
                    artifacts={"reason": ready_reason, "use_live_model": True},
                )

        temp_dir = Path(tempfile.mkdtemp(prefix=f"zelyr_eval_{task.task_id}_")).resolve()
        db_path = temp_dir / "test_eval.db"
        tools_dir = temp_dir / "tools.d"
        tools_dir.mkdir(parents=True, exist_ok=True)

        prev_env_db = os.environ.get("ZELYR_MEMORY_DB")
        os.environ["ZELYR_MEMORY_DB"] = str(db_path)

        memory_store = MemoryStore(db_path=str(db_path))

        context = EvalContext(
            task=task,
            workspace_dir=temp_dir,
            db_path=db_path,
            memory_store=memory_store,
            tools_dir=tools_dir,
        )

        status: EvalStatus = EvalStatus.UNVERIFIED
        status_message: str = ""
        failure_category: Optional[str] = None
        tool_names_called: List[str] = []
        estimated_token_count: int = 0
        step_counter: int = 0
        model_step_latencies: List[float] = []
        tool_step_latencies: List[float] = []
        live_adapter: Optional[LiveOllamaStreamAdapter] = None

        # Merge tool handlers
        tools_map = dict(task.custom_tools or task.fake_tools or {})
        if custom_tools:
            tools_map.update(custom_tools)

        # Scoped tool executor
        async def scoped_tool_executor(name: str, args: Any) -> Any:
            tool_names_called.append(name)
            if task.allowed_tools is not None and name not in task.allowed_tools:
                err_env = ToolResult(
                    success=False,
                    error=f"Security Violation: Tool '{name}' is not in allowed_tools {task.allowed_tools}",
                    remedy_hint=f"Use allowed tools: {task.allowed_tools}",
                    allowed_actions=list(task.allowed_tools),
                ).to_dict(tool=name, arguments=args if isinstance(args, dict) else {})
                context.record_tool_invocation(name, args if isinstance(args, dict) else {}, err_env)
                return err_env

            handler = tools_map.get(name)
            if handler is None:
                # Check for dynamic tool installed in tools_dir
                dynamic_tool_file = tools_dir / f"{name}.py"
                if dynamic_tool_file.is_file():
                    mod_name = f"zelyr.eval_dyn_{task.task_id}_{name}"
                    spec = importlib.util.spec_from_file_location(mod_name, dynamic_tool_file)
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        try:
                            spec.loader.exec_module(mod)
                            if hasattr(mod, "execute"):
                                handler = getattr(mod, "execute")
                        except Exception as load_err:
                            err_env = ToolResult(
                                success=False,
                                error=f"Dynamic tool '{name}' failed to load: {load_err}",
                            ).to_dict(tool=name, arguments=args if isinstance(args, dict) else {})
                            context.record_tool_invocation(name, args if isinstance(args, dict) else {}, err_env)
                            return err_env

            if handler is None:
                err_env = ToolResult(
                    success=False,
                    error=f"Tool '{name}' has no registered implementation.",
                ).to_dict(tool=name, arguments=args if isinstance(args, dict) else {})
                context.record_tool_invocation(name, args if isinstance(args, dict) else {}, err_env)
                return err_env

            call_kwargs = dict(args) if isinstance(args, dict) else {}
            sig = inspect.signature(handler)
            if "context" in sig.parameters and "context" not in call_kwargs:
                call_kwargs["context"] = context
            if "workspace" in sig.parameters and "workspace" not in call_kwargs:
                call_kwargs["workspace"] = temp_dir
            if "db_path" in sig.parameters and "db_path" not in call_kwargs:
                call_kwargs["db_path"] = db_path
            if "tools_dir" in sig.parameters and "tools_dir" not in call_kwargs:
                call_kwargs["tools_dir"] = tools_dir
            if "memory_store" in sig.parameters and "memory_store" not in call_kwargs:
                call_kwargs["memory_store"] = memory_store

            t_tool_0 = time.perf_counter()
            try:
                # Call handler with kwargs or positional args if signature doesn't take kwargs
                if inspect.iscoroutinefunction(handler):
                    try:
                        res = await handler(**call_kwargs)
                    except TypeError:
                        res = await handler(call_kwargs)
                else:
                    try:
                        res = handler(**call_kwargs)
                    except TypeError:
                        res = handler(call_kwargs)

                if inspect.isawaitable(res):
                    res = await res
            except Exception as exc:
                res = ToolResult(
                    success=False,
                    error=f"Tool execution failed: {exc}",
                )
            tool_step_latencies.append(time.perf_counter() - t_tool_0)

            context.record_tool_invocation(name, call_kwargs, res)
            return res

        # Build mock stream provider
        turns_source = mock_responses or task.mock_responses or task.mock_turns or []

        class MockStreamManager:
            def __init__(self, turns: List[Any], custom_fn: Optional[Callable] = None):
                self.turns = list(turns)
                self.custom_fn = custom_fn
                self.req_idx = 0

            async def stream_chat(self, payload: dict):
                nonlocal step_counter, estimated_token_count
                step_counter += 1
                t_model_0 = time.perf_counter()

                # Token accounting via NativeOrchestrator.count_tokens
                for m in payload.get("messages", []):
                    c = m.get("content", "")
                    if isinstance(c, str):
                        estimated_token_count += NativeOrchestrator.count_tokens(c)

                try:
                    if self.custom_fn:
                        res = self.custom_fn(payload)
                        if inspect.isasyncgen(res):
                            async for chunk in res:
                                yield chunk
                            return
                        elif inspect.isgenerator(res):
                            for chunk in res:
                                yield chunk
                            return
                        elif inspect.isawaitable(res):
                            res = await res

                    if self.req_idx < len(self.turns):
                        turn = self.turns[self.req_idx]
                        self.req_idx += 1

                        if isinstance(turn, MockTurn):
                            if turn.exception is not None:
                                raise turn.exception
                            for chunk in turn.chunks:
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                c = delta.get("content", "")
                                if c:
                                    estimated_token_count += NativeOrchestrator.count_tokens(c)
                                yield chunk
                        elif isinstance(turn, dict):
                            # Could be assistant message format or raw chunk
                            if "choices" in turn:
                                yield turn
                            elif turn.get("role") == "assistant":
                                # Convert assistant message format to SSE delta chunks
                                if "tool_calls" in turn:
                                    yield {
                                        "choices": [
                                            {
                                                "delta": {
                                                    "tool_calls": [
                                                        {
                                                            "index": 0,
                                                            "id": tc.get("id", "mock_call_1"),
                                                            "type": "function",
                                                            "function": {
                                                                "name": tc.get("function", {}).get("name") or tc.get("name"),
                                                                "arguments": (
                                                                    tc.get("function", {}).get("arguments")
                                                                    if isinstance(tc.get("function", {}).get("arguments"), str)
                                                                    else json.dumps(tc.get("function", {}).get("arguments") or tc.get("arguments", {}))
                                                                ),
                                                            },
                                                        }
                                                        for tc in turn.get("tool_calls", [])
                                                    ]
                                                }
                                            }
                                        ]
                                    }
                                elif "content" in turn:
                                    content = turn.get("content", "")
                                    estimated_token_count += NativeOrchestrator.count_tokens(content)
                                    yield {"choices": [{"delta": {"content": content}}]}
                            else:
                                yield turn
                        elif isinstance(turn, str):
                            estimated_token_count += NativeOrchestrator.count_tokens(turn)
                            yield {"choices": [{"delta": {"content": turn}}]}
                    else:
                        yield {"choices": [{"delta": {"content": "[Mock turns exhausted]"}}]}
                finally:
                    model_step_latencies.append(time.perf_counter() - t_model_0)

        if eff_use_live_model:
            live_adapter = LiveOllamaStreamAdapter(base_url=eff_base_url, model=eff_model)

            async def _live_stream_wrapper(p: dict):
                nonlocal step_counter
                step_counter += 1
                async for chunk in live_adapter.stream_chat(p):
                    yield chunk

            stream_chat_fn = _live_stream_wrapper
        else:
            stream_mgr = MockStreamManager(turns_source, custom_fn=mock_stream_fn or task.mock_stream_fn)
            stream_chat_fn = stream_mgr.stream_chat

        try:
            # Setup hook
            if task.setup_fn:
                s_res = task.setup_fn(temp_dir, memory_store)
                if inspect.isawaitable(s_res):
                    await s_res
            elif task.setup_hook:
                s_res = task.setup_hook(context)
                if inspect.isawaitable(s_res):
                    await s_res

            orchestrator = NativeOrchestrator(
                max_depth=task.max_steps,
                stream_provider=stream_chat_fn,
                tool_schemas=task.tool_schemas,
                tool_executor=scoped_tool_executor,
                db_path=str(db_path),
            )

            payload = {
                "model": eff_model,
                "messages": [{"role": "user", "content": task.user_prompt or task.description}],
                "tools": list(task.tool_schemas or []),
            }

            async def _execute_stream():
                async for chunk in orchestrator.generate_stream(payload):
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    c = delta.get("content", "")
                    if c:
                        context.final_response += c

            await asyncio.wait_for(_execute_stream(), timeout=effective_timeout)

            # Step limit enforcement
            if f"Reached maximum autonomous ReAct steps ({task.max_steps})" in context.final_response:
                context.step_limit_hit = True
                status = EvalStatus.STEP_LIMIT_EXCEEDED
                failure_category = "STEP_LIMIT_EXCEEDED"
                status_message = f"Reached maximum autonomous ReAct steps ({task.max_steps})."
            else:
                # Independent state validator
                val_fn = task.validator_fn or task.validator
                if val_fn:
                    v_res = val_fn(context)
                    if inspect.isawaitable(v_res):
                        v_res = await v_res

                    if isinstance(v_res, tuple):
                        ok, reason = v_res
                    elif hasattr(v_res, "success"):
                        ok, reason = v_res.success, v_res.message
                    else:
                        ok, reason = bool(v_res), str(v_res)

                    if ok:
                        status = EvalStatus.PASS
                        status_message = reason
                    else:
                        # Check for FALSE_SUCCESS claim
                        r_lower = context.final_response.lower()
                        success_claim_words = [
                            "successfully", "created", "completed", "finished",
                            "done", "have written", "has been saved", "all results",
                        ]
                        claims_completion = any(w in r_lower for w in success_claim_words)
                        if claims_completion or not context.tool_calls:
                            status = EvalStatus.FALSE_SUCCESS
                            failure_category = "FALSE_SUCCESS"
                            status_message = f"Model claimed success in text without observable state change ({reason})"
                        else:
                            status = EvalStatus.FAIL
                            failure_category = "VALIDATION_FAILED"
                            status_message = reason
                else:
                    status = EvalStatus.UNVERIFIED
                    status_message = "No validator configured."

        except asyncio.TimeoutError:
            status = EvalStatus.TIMEOUT
            failure_category = "TIMEOUT"
            status_message = f"Task execution timed out after {effective_timeout}s"
        except Exception as exc:
            context.execution_error = exc
            status = EvalStatus.TOOL_ERROR
            failure_category = "TOOL_ERROR"
            status_message = str(exc)

        finally:
            if prev_env_db is not None:
                os.environ["ZELYR_MEMORY_DB"] = prev_env_db
            else:
                os.environ.pop("ZELYR_MEMORY_DB", None)

            context.duration_seconds = time.perf_counter() - start_time
            context.step_count = step_counter

            if task.cleanup_fn:
                try:
                    c_res = task.cleanup_fn(temp_dir)
                    if inspect.isawaitable(c_res):
                        await c_res
                except Exception:
                    pass
            elif task.cleanup_hook:
                try:
                    c_res = task.cleanup_hook(context)
                    if inspect.isawaitable(c_res):
                        await c_res
                except Exception:
                    pass

            # Always clean up temporary workspace
            shutil.rmtree(temp_dir, ignore_errors=True)

        measured_tokens_val = (
            live_adapter.measured_tokens
            if (eff_use_live_model and live_adapter is not None)
            else None
        )
        latencies_model = (
            live_adapter.model_latencies
            if (eff_use_live_model and live_adapter is not None)
            else list(model_step_latencies)
        )

        return TaskResult(
            task_id=task.task_id,
            status=status,
            message=status_message,
            step_count=context.step_count,
            tool_calls=tool_names_called,
            duration_seconds=context.duration_seconds,
            measured_tokens=measured_tokens_val,
            estimated_tokens=estimated_token_count,
            failure_category=failure_category,
            artifacts={
                "final_response": context.final_response,
                "workspace_dir": str(temp_dir),
                "db_path": str(db_path),
                "expected_status": (
                    task.expected_status.value
                    if hasattr(task.expected_status, "value")
                    else (str(task.expected_status) if task.expected_status is not None else None)
                ),
                "latency_breakdown": {
                    "model_latencies": latencies_model,
                    "tool_latencies": list(tool_step_latencies),
                    "total_model_latency": round(sum(latencies_model), 4),
                    "total_tool_latency": round(sum(tool_step_latencies), 4),
                },
            },
        )

    # Backward compatibility with EvaluationHarness.execute_task
    async def execute_task(self, task: Any) -> EvaluationResult:
        res = await self.run_task(task)
        # Map EvalStatus to ResultStatus
        status_map = {
            EvalStatus.PASS: ResultStatus.PASS,
            EvalStatus.FAIL: ResultStatus.FAIL,
            EvalStatus.FALSE_SUCCESS: ResultStatus.FAIL,
            EvalStatus.STEP_LIMIT_EXCEEDED: ResultStatus.STEP_LIMIT_EXCEEDED,
            EvalStatus.TIMEOUT: ResultStatus.TIMEOUT,
            EvalStatus.TOOL_ERROR: ResultStatus.ERROR,
            EvalStatus.UNVERIFIED: ResultStatus.UNVERIFIED,
        }
        mapped_status = status_map.get(res.status, ResultStatus.FAIL)
        failure_cat = res.failure_category
        if failure_cat == "FALSE_SUCCESS":
            failure_cat = "VALIDATION_FAILED"

        return EvaluationResult(
            task_id=res.task_id,
            status=mapped_status,
            benchmark_level=getattr(task, "benchmark_level", 1),
            step_count=res.step_count,
            model_requests_count=res.step_count,
            tool_call_count=len(res.tool_calls),
            tool_sequence=list(res.tool_calls),
            duration_seconds=res.duration_seconds,
            failure_category=failure_cat,
            validator_message=res.message,
            error_message=res.message if res.status == EvalStatus.TOOL_ERROR else None,
            measured_tokens=res.measured_tokens,
            estimated_tokens={"total_tokens": res.estimated_tokens},
            cleanup_success=True,
        )

    async def run_suite(self, tasks: List[Any], suite_name: str = "Benchmark Suite") -> Any:
        from zelyr.eval.report import BenchmarkReport
        results = [await self.execute_task(t) for t in tasks]
        return BenchmarkReport(suite_name=suite_name, results=results)


# Compatibility alias
EvaluationHarness = EvalHarness
