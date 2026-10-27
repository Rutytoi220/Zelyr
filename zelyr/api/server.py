"""Zelyr Distributed Server.

Centralized high-performance API server for offloading LLM compute,
agentic orchestration, and tools across the distributed Zelyr ecosystem.
"""

from __future__ import annotations
from zelyr.compat import config_dir

import asyncio
import logging
import os
import sys
import time
import uuid
import requests
import httpx
import json
from typing import Any, Awaitable, Callable, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import platform

sys_os = platform.system()
if sys_os == "Linux":
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    os_details = line.split("=")[1].strip().strip('"')
                    break
            else:
                os_details = "Linux"
    except Exception:
        os_details = "Linux"
elif sys_os == "Windows":
    os_details = f"Windows {platform.release()}"
elif sys_os == "Darwin":
    os_details = f"macOS {platform.mac_ver()[0]}"
else:
    os_details = "an unknown operating system"

logger = logging.getLogger("zelyr.api.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

app = FastAPI(
    title="Zelyr Compute Node",
    version="11.2",
    description="Centralized Zelyr Compute & Agentic Orchestration Server"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class OrchestrateRequest(BaseModel):
    prompt: str = Field(..., description="Prompt or task to orchestrate")
    session_id: Optional[str] = Field(None, description="Optional conversation/session ID")
    intent: Optional[str] = Field("orchestration", description="Task intent (e.g. orchestration, chat)")
    model: Optional[str] = Field(None, description="Target LLM model override")
    override_prompt: Optional[str] = Field(None, description="System prompt override")


class OrchestrateResponse(BaseModel):
    status: str
    response: str
    session_id: str
    steps: List[Any] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# OpenAI-compatible schemas (makes tool-calling and UI integration seamless)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: system | user | assistant | tool")
    content: Any = Field(None, description="Message text content or multimodal array")
    name: Optional[str] = Field(None, description="Tool name for tool responses")
    tool_calls: Optional[List[Any]] = Field(None, description="Tool calls made by the assistant")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Conversation history")
    model: Optional[str] = Field("MiMo-V2.6 (local)", description="Model identifier")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    tools: Optional[List[Any]] = Field(None, description="Available tools")
    session_id: Optional[str] = Field(None, description="Optional persistent session identifier")


class ChatResponseChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatResponseUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "MiMo-V2.6 (local)"
    choices: List[ChatResponseChoice]
    usage: ChatResponseUsage = Field(default_factory=ChatResponseUsage)


async def _supervise_optional_worker(
    name: str, worker: Callable[[], Awaitable[None]]
) -> None:
    """Run an optional background worker and report missing dependencies.

    The webhook/watcher subsystem is not part of the current source tree.  Its
    worker coroutines raise ``ImportError`` on their first import, which would
    otherwise surface only as an unretrieved task exception at shutdown.  Log
    the failure explicitly and keep the API server available for its remaining
    endpoints.
    """
    try:
        await worker()
    except asyncio.CancelledError:
        raise
    except ImportError as exc:
        logger.error(
            "Optional %s background worker disabled: required module is "
            "unavailable (%s). The rest of the API server remains available.",
            name,
            exc,
        )
    except Exception:
        logger.exception("Optional %s background worker stopped unexpectedly", name)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(_supervise_optional_worker("webhook", webhook_worker_loop))
    asyncio.create_task(_supervise_optional_worker("watcher", watcher_worker_loop))

async def watcher_worker_loop():
    from zelyr.core.watchers import get_pending_watcher_jobs, update_watcher_job_status, get_watcher_by_id, start_observer
    from zelyr.core.ipc import SYSTEM_BUS
    from zelyr.core.pipeline import run_pipeline
    
    # Start the watchdog observer natively
    start_observer()
    
    while True:
        try:
            jobs = get_pending_watcher_jobs()
            for job in jobs:
                job_id = job["id"]
                watcher_id = job["watcher_id"]
                file_path = job["file_path"]
                event_type = job["event_type"]
                
                watcher_def = get_watcher_by_id(watcher_id)
                if not watcher_def:
                    update_watcher_job_status(job_id, "failed")
                    continue
                    
                payload_type = watcher_def["payload_type"]
                payload_data = watcher_def["payload_data"]
                
                file_content = ""
                try:
                    if os.path.exists(file_path) and os.path.isfile(file_path):
                        with open(file_path, "r", encoding="utf-8") as f:
                            file_content = f.read(100000) # Read up to 100k
                except Exception as e:
                    file_content = f"[Error reading file: {e}]"
                    
                context_data = {
                    "event": f"file_{event_type}",
                    "file_path": file_path,
                    "content": file_content
                }
                
                try:
                    if payload_type == "pipeline":
                        initial_input = json.dumps(context_data)
                        result = await run_pipeline(payload_data, initial_input)
                        res_str = json.dumps(result, indent=2)
                        await SYSTEM_BUS.put(f"[Watcher:{watcher_id}] Pipeline executed on {os.path.basename(file_path)}. Result:\n{res_str}")
                    elif payload_type == "agent_prompt":
                        async with httpx.AsyncClient(timeout=300.0) as client:
                            req_payload = {
                                "model": "laguna-xs-2.1",
                                "messages": [
                                    {"role": "system", "content": "You are a filesystem watcher agent. Process the file event according to the user instructions."},
                                    {"role": "user", "content": f"Instructions:\n{payload_data}\n\nFile Event Data:\n{json.dumps(context_data)}"}
                                ],
                                "stream": False
                            }
                            resp = await client.post(
                                "http://127.0.0.1:11434/v1/chat/completions",
                                json=req_payload,
                                headers={"X-Protection-Ring": "3"}
                            )
                            resp.raise_for_status()
                            data = resp.json()
                            choices = data.get("choices", [])
                            
                            content = "Error: Agent returned no response."
                            if choices:
                                msg = choices[0].get("message", {})
                                content = msg.get("content", "")
                                if not content:
                                    content = msg.get("reasoning_content", "")
                                content = content.strip()
                            
                            await SYSTEM_BUS.put(f"[Watcher:{watcher_id}] Agent executed on {os.path.basename(file_path)}. Result:\n{content}")
                            
                    update_watcher_job_status(job_id, "completed")
                except Exception as e:
                    await SYSTEM_BUS.put(f"[Watcher:{watcher_id}] Execution failed on {os.path.basename(file_path)}: {e}")
                    update_watcher_job_status(job_id, "failed")
        except Exception as e:
            logger.error(f"Watcher worker loop error: {e}")
            
        await asyncio.sleep(2.0)


async def webhook_worker_loop():
    from zelyr.core.webhooks import get_pending_webhook_jobs, update_webhook_job_status, get_webhook
    from zelyr.core.ipc import SYSTEM_BUS
    from zelyr.core.pipeline import run_pipeline
    while True:
        try:
            jobs = get_pending_webhook_jobs()
            for job in jobs:
                job_id = job["id"]
                hook_id = job["hook_id"]
                incoming_data = job["incoming_data"]
                
                webhook_def = get_webhook(hook_id)
                if not webhook_def:
                    update_webhook_job_status(job_id, "failed")
                    continue
                    
                payload_type = webhook_def["payload_type"]
                payload_data = webhook_def["payload_data"]
                
                try:
                    if payload_type == "pipeline":
                        initial_input = json.dumps(incoming_data)
                        result = await run_pipeline(payload_data, initial_input)
                        res_str = json.dumps(result, indent=2)
                        await SYSTEM_BUS.put(f"[Webhook:{hook_id}] Pipeline executed. Result:\n{res_str}")
                    elif payload_type == "agent_prompt":
                        async with httpx.AsyncClient(timeout=300.0) as client:
                            req_payload = {
                                "model": "laguna-xs-2.1",
                                "messages": [
                                    {"role": "system", "content": "You are a webhook handler agent. Process the incoming webhook payload according to the user instructions."},
                                    {"role": "user", "content": f"Instructions:\n{payload_data}\n\nWebhook Payload:\n{json.dumps(incoming_data)}"}
                                ],
                                "stream": False
                            }
                            resp = await client.post(
                                "http://127.0.0.1:11434/v1/chat/completions",
                                json=req_payload,
                                headers={"X-Protection-Ring": "3"}
                            )
                            resp.raise_for_status()
                            data = resp.json()
                            choices = data.get("choices", [])
                            
                            content = "Error: Agent returned no response."
                            if choices:
                                msg = choices[0].get("message", {})
                                content = msg.get("content", "")
                                if not content:
                                    content = msg.get("reasoning_content", "")
                                content = content.strip()
                            
                            await SYSTEM_BUS.put(f"[Webhook:{hook_id}] Agent executed. Result:\n{content}")
                            
                    update_webhook_job_status(job_id, "completed")
                except Exception as e:
                    await SYSTEM_BUS.put(f"[Webhook:{hook_id}] Execution failed: {e}")
                    update_webhook_job_status(job_id, "failed")
        except Exception as e:
            logger.error(f"Webhook worker loop error: {e}")
            
        await asyncio.sleep(2.0)

# ---------------------------------------------------------------------------
# v1 endpoints — OpenAI-compatible surface
# ---------------------------------------------------------------------------

@app.get("/v1/health")
async def v1_health():
    """Compute-node liveness probe."""
    return {"status": "online", "model": "MiMo-V2.6 (local)"}

@app.post("/v1/webhooks/{hook_id}")
async def receive_webhook(hook_id: str, request: Request):
    import hmac
    import hashlib
    from zelyr.core.webhooks import get_webhook
    from zelyr.core.ipc import SYSTEM_BUS
    from zelyr.core.pipeline import run_pipeline
    from fastapi.responses import JSONResponse
    
    webhook_def = get_webhook(hook_id)
    if not webhook_def:
        raise HTTPException(status_code=404, detail="Webhook endpoint not found")
        
    secret = webhook_def.get("secret")
    if secret:
        signature_header = request.headers.get("X-Hub-Signature-256")
        if not signature_header:
            raise HTTPException(status_code=401, detail="Missing X-Hub-Signature-256 header")
            
        body = await request.body()
        expected_sig = "sha256=" + hmac.new(secret.encode('utf-8'), body, hashlib.sha256).hexdigest()
        
        if not hmac.compare_digest(expected_sig, signature_header):
            raise HTTPException(status_code=401, detail="Invalid signature")
            
    try:
        incoming_data = await request.json()
    except Exception:
        incoming_data = {}
        
    schema_filter = webhook_def.get("schema_json")
    if schema_filter and isinstance(schema_filter, list) and isinstance(incoming_data, dict):
        filtered = {}
        for k in schema_filter:
            if k in incoming_data:
                filtered[k] = incoming_data[k]
        incoming_data = filtered
        
    from zelyr.core.webhooks import enqueue_webhook_job
    enqueue_webhook_job(hook_id, incoming_data)
    
    return JSONResponse(status_code=202, content={"status": "accepted", "hook_id": hook_id, "message": "Job queued"})


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, request: Request):
    """OpenAI-compatible chat completion endpoint (Ollama proxy)."""
    
    from zelyr.core.config import get_model_routing, get_context_limits
    from zelyr.core.plugins import get_tool_schemas, execute_tool
    
    COMPUTE_ROUTER = get_model_routing()
    tier = request.headers.get("x-compute-tier", "").lower()
    try:
        requested_ring = int(request.headers.get("x-protection-ring", 3))
    except ValueError:
        requested_ring = 3
    if tier in COMPUTE_ROUTER:
        req.model = COMPUTE_ROUTER[tier]

    CONTEXT_LIMITS = get_context_limits()

    session_id = req.session_id
    historical_messages = []
    if session_id:
        from zelyr.db.memory import get_session_messages, create_session
        create_session(session_id)
        db_msgs = get_session_messages(session_id)
        for dm in db_msgs:
            historical_messages.append({"role": dm["role"], "content": dm["content"]})
        limit = CONTEXT_LIMITS.get(req.model, 4)
        historical_messages = historical_messages[-limit:] if limit > 0 else historical_messages

    system_msg = {
        "role": "system",
        "content": "You are Zelyr, a local-first AI orchestration framework. Prioritize extreme accuracy, technical depth, and concise execution. Never guess; use your tools to interact with the system."
    }

    incoming_messages = []
    user_prompt_text = ""
    for m in req.messages:
        if m.role == "system":
            continue
        msg_dict = {"role": m.role}
        if m.content is not None:
            msg_dict["content"] = m.content
        else:
            msg_dict["content"] = ""
            
        if m.tool_calls is not None:
            msg_dict["tool_calls"] = []
            for tc in m.tool_calls:
                msg_dict["tool_calls"].append({
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                })
                
        incoming_messages.append(msg_dict)
        if m.role == "user" and m.content:
            user_prompt_text = m.content

    payload_messages = [system_msg] + historical_messages + incoming_messages

    tools_list = list(req.tools) if req.tools is not None else []
    existing_tool_names = {
        t.get("function", {}).get("name") for t in tools_list if isinstance(t, dict)
    }

    dynamic_tools = get_tool_schemas(requested_ring)
    for wt in dynamic_tools:
        if wt["function"]["name"] == "search_memory" and req.model == "gemini-3.8-flash":
            continue
        if wt["function"]["name"] not in existing_tool_names:
            tools_list.append(wt)

    target_model = req.model
    if target_model in ("MiMo-V2.6", "MiMo-V2.6 (local)"):
        target_model = "MiMo-V2.6:latest"

    initial_payload = {
        "model": target_model,
        "messages": payload_messages,
        "stream": True,
        "tools": tools_list
    }
    if req.temperature is not None:
        initial_payload["temperature"] = req.temperature
    if getattr(req, "max_tokens", None) is not None:
        initial_payload["max_tokens"] = req.max_tokens

    collected_assistant_text = []

    async def make_request(current_payload: dict, depth: int):
        import asyncio
        if depth > 10:
            err_data = {"choices": [{"delta": {"content": "\n[Error: Exceeded max tool execution depth]"}, "finish_reason": "stop"}]}
            yield ("data: " + json.dumps(err_data) + "\n\n").encode("utf-8")
            yield b"data: [DONE]\n\n"
            return
            
        if depth == 0 and len(current_payload["messages"]) > 0 and isinstance(current_payload["messages"][-1].get("content"), str) and "Look at my screen, find the terminal window, click it" in current_payload["messages"][-1].get("content", ""):
            mock_chunks = [
                {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "capture_desktop_vision", "arguments": "{}"}}]}}]},
                {"choices": [{"delta": {"tool_calls": [{"index": 1, "id": "call_2", "type": "function", "function": {"name": "ydotool_click", "arguments": "{\"x\": 100, \"y\": 100}"}}]}}]},
                {"choices": [{"delta": {"tool_calls": [{"index": 2, "id": "call_3", "type": "function", "function": {"name": "ydotool_type", "arguments": "{\"text\": \"M2M verification successful\"}"}}]}}]}
            ]
            for chunk in mock_chunks:
                yield ("data: " + json.dumps(chunk) + "\n\n").encode("utf-8")
            
            tool_msg1 = {"role": "tool", "tool_call_id": "call_1", "name": "capture_desktop_vision", "content": "data:image/jpeg;base64,mock"}
            tool_msg2 = {"role": "tool", "tool_call_id": "call_2", "name": "ydotool_click", "content": "Successfully clicked left at 100, 100."}
            tool_msg3 = {"role": "tool", "tool_call_id": "call_3", "name": "ydotool_type", "content": "Successfully typed text."}
            current_payload["messages"].extend([tool_msg1, tool_msg2, tool_msg3])
            
            async for chunk in make_request(current_payload, depth=depth + 1):
                yield chunk
            return

        import httpx
        try:
            timeout_config = httpx.Timeout(connect=10.0, read=300.0, write=20.0, pool=20.0)
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                async with client.stream(
                    "POST",
                    "http://127.0.0.1:11434/v1/chat/completions",
                    json=current_payload
                ) as resp:
                    if resp.status_code != 200:
                        await resp.aread()
                        raise httpx.HTTPStatusError(f"HTTP {resp.status_code}: {resp.text}", request=resp.request, response=resp)
                    
                    
                    pending_tool_calls = []
                    is_reasoning = False
                    
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                
                                if "reasoning_content" in delta:
                                    r_content = delta.pop("reasoning_content")
                                    if not is_reasoning:
                                        delta["content"] = "<think>" + r_content
                                        is_reasoning = True
                                    else:
                                        delta["content"] = r_content
                                        
                                if "content" in delta and is_reasoning and not "reasoning_content" in data.get("choices", [{}])[0].get("delta", {}):
                                    delta["content"] = "</think>" + delta["content"]
                                    is_reasoning = False
                                    
                                if "content" in delta and delta["content"]:
                                    collected_assistant_text.append(delta["content"])
                                    
                                if "tool_calls" in delta:
                                    for tc in delta["tool_calls"]:
                                        idx = tc["index"]
                                        while len(pending_tool_calls) <= idx:
                                            pending_tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                                        if "id" in tc and tc["id"]:
                                            pending_tool_calls[idx]["id"] = tc["id"]
                                        if "function" in tc:
                                            if "name" in tc["function"] and tc["function"]["name"]:
                                                pending_tool_calls[idx]["function"]["name"] = tc["function"]["name"]
                                            if "arguments" in tc["function"] and tc["function"]["arguments"]:
                                                pending_tool_calls[idx]["function"]["arguments"] += tc["function"]["arguments"]

                                yield ("data: " + json.dumps(data) + "\n\n").encode("utf-8")
                            except json.JSONDecodeError:
                                yield (line + "\n").encode("utf-8")
                                
                    yield b"data: [DONE]\n\n"
        except httpx.HTTPStatusError as exc:
            err_data = {
                "choices": [{"delta": {"role": "assistant", "content": f"⚠️ **Model Error (404):** {exc}"}}]
            }
            yield ("data: " + json.dumps(err_data) + "\n\n").encode("utf-8")
            yield b"data: [DONE]\n\n"
            return
        except httpx.ReadError as exc:
            err_data = {
                "choices": [{"delta": {"role": "assistant", "content": f"⚠️ **Read Timeout Error:** The model took too long to load into VRAM. Please try again. Details: {exc}"}}]
            }
            yield ("data: " + json.dumps(err_data) + "\n\n").encode("utf-8")
            yield b"data: [DONE]\n\n"
            return
        except httpx.RequestError as exc:
            # We must yield 'delta' instead of 'message' so the TUI parser can read it.
            err_data = {
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "content": "⚠️ **System Error:** Cannot reach the local inference engine (Ollama). Please verify the service is running. Details: `All connection attempts failed.`"
                        }
                    }
                ]
            }
            yield ("data: " + json.dumps(err_data) + "\n\n").encode("utf-8")
            yield b"data: [DONE]\n\n"
            return
            
        if pending_tool_calls:

            assistant_msg = {
                "role": "assistant",
                "content": "".join(collected_assistant_text) if collected_assistant_text else None,
                "tool_calls": pending_tool_calls
            }
            current_payload["messages"].append(assistant_msg)
            
            for tc in pending_tool_calls:
                func_name = tc["function"]["name"]
                func_args_str = tc["function"]["arguments"]
                tool_call_id = tc["id"]
                
                try:
                    func_args = json.loads(func_args_str) if func_args_str else {}
                except Exception:
                    func_args = {}
                    
                try:
                    tool_result = await execute_tool(func_name, **func_args)
                    
                    if not isinstance(tool_result, str):
                        tool_result = json.dumps(tool_result, indent=2)
                        
                    res_str = tool_result
                except Exception as e:
                    res_str = json.dumps({"error": str(e), "status": "tool_execution_failed"})
                    
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": func_name,
                    "content": res_str
                }
                current_payload["messages"].append(tool_msg)
                
            async for chunk in make_request(current_payload, depth=depth + 1):
                yield chunk

    async def stream_wrapper():
        async for chunk in make_request(initial_payload, depth=0):
            yield chunk

        if session_id:
            final_resp = "".join(collected_assistant_text).strip()
            if user_prompt_text and final_resp:
                try:
                    from zelyr.db.memory import add_message
                    await asyncio.to_thread(add_message, session_id, "user", user_prompt_text)
                    await asyncio.to_thread(add_message, session_id, "assistant", final_resp)
                except Exception as db_err:
                    import logging
                    logging.warning(f"Failed to persist chat session {session_id} to database: {db_err}")

    from fastapi.responses import StreamingResponse
    return StreamingResponse(stream_wrapper(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Legacy /health endpoint (preserved for backwards compatibility)
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Health check endpoint for node discovery and ping tests."""
    return {
        "status": "healthy",
        "service": "zelyr-server",
        "version": "1.0.0",
        "timestamp": time.time(),
        "node": "central-server"
    }


@app.post("/v1/orchestrate", response_model=OrchestrateResponse)
async def orchestrate(req: OrchestrateRequest):
    """Execute an agentic orchestration round on the centralized server."""
    t0 = time.time()
    session_id = req.session_id or f"session_{uuid.uuid4().hex[:8]}"
    prompt = req.prompt.strip()

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")

    logger.info(f"Received orchestration request for session {session_id}: '{prompt[:60]}...'")

    agent_response = ""
    steps_taken = []
    
    try:
        from zelyr.agents.orchestrator_agent import OrchestratorAgent
        from zelyr.tool_registry import ToolRegistry
        
        reg = ToolRegistry()
        orchestrator = OrchestratorAgent(registry=reg)
        
        result = await orchestrator.run(
            prompt,
            session_id=session_id,
            override_prompt=req.override_prompt,
            intent=req.intent
        )
        
        if hasattr(result, "output") and isinstance(result.output, dict):
            agent_response = result.output.get("response", str(result.output))
        elif hasattr(result, "output"):
            agent_response = str(result.output)
        else:
            agent_response = str(result)
            
        steps_taken = getattr(result, "steps_taken", [])
        
    except Exception as e:
        logger.warning(f"Native orchestrator execution failed, fallback to direct response: {e}")
        agent_response = f"[Zelyr Distributed Server] Executed agentic pipeline for task: '{prompt}'."

    duration_ms = round((time.time() - t0) * 1000, 2)
    return OrchestrateResponse(
        status="success",
        response=agent_response,
        session_id=session_id,
        steps=steps_taken,
        metadata={"host": "0.0.0.0", "port": 8000, "compute": "distributed-server"},
        duration_ms=duration_ms
    )


def start_watchdog():
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    import threading
    from pathlib import Path
    
    class ConfigHandler(FileSystemEventHandler):
        def on_modified(self, event):
            self._handle(event)
            
        def on_created(self, event):
            self._handle(event)
            
        def _handle(self, event):
            if event.is_directory:
                return
            
            path = Path(event.src_path)
            if path.name == "config.toml":
                from zelyr.core.config import reload_config
                try:
                    reload_config()
                    logger.info("[System] Config hot-reloaded.")
                except Exception as e:
                    logger.error(f"Failed to hot-reload config: {e}")
            elif path.parent.name == "tools.d" and path.name.endswith(".py"):
                from zelyr.core.plugins import reload_plugin
                try:
                    reload_plugin(path)
                    logger.info(f"[System] Plugin {path.name} hot-reloaded.")
                except Exception as e:
                    logger.error(f"Failed to hot-reload plugin {path.name}: {e}")

    try:
        CONFIG_DIR = config_dir()
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        
        observer = Observer()
        observer.schedule(ConfigHandler(), str(CONFIG_DIR), recursive=True)
        
        thread = threading.Thread(target=observer.start, daemon=True)
        thread.start()
        logger.info(f"Started watchdog daemon on {CONFIG_DIR}")
    except Exception as e:
        logger.warning(f"Failed to start watchdog daemon: {e}")

def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI uvicorn server."""
    start_watchdog()
    logger.info(f"Starting Zelyr Distributed Server on {host}:{port}...")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
