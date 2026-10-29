import asyncio
import inspect
import json
import re
import shutil
import httpx
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional
from zelyr.core.plugins import get_tool_schemas, execute_tool, get_tier_timeout

REFUSAL_PHRASES = [
    "cannot see your screen",
    "can't see your screen",
    "as an ai",
    "don't have eyes",
    "cannot interact with your computer",
]

class NativeOrchestrator:
    def __init__(
        self,
        max_context_tokens: Optional[int] = None,
        max_steps: int = 5,
        model_provider: Optional[Any] = None,
        tool_schemas: Optional[list] = None,
        execute_tool: Optional[Any] = None,
        stream_provider: Optional[Callable[[dict], AsyncGenerator[dict, None]]] = None,
        tool_executor: Optional[Callable[[str, dict], Awaitable[Any]]] = None,
        db_path: Optional[str] = None,
        max_depth: Optional[int] = None,
    ):
        self._explicit_max_context_tokens = max_context_tokens
        self.max_steps = max_depth if max_depth is not None else max_steps
        self.max_depth = self.max_steps
        self.stream_provider = stream_provider or model_provider
        self.model_provider = self.stream_provider
        self.tool_schemas = tool_schemas
        self.tool_executor = tool_executor
        self._execute_tool = execute_tool or tool_executor
        self.db_path = db_path

        from zelyr.engine.budget_manager import ContextBudgetEngine
        self.budget_engine = ContextBudgetEngine(
            default_max_tokens=self._explicit_max_context_tokens or 8192,
        )

    @property
    def max_context_tokens(self) -> int:
        from zelyr.config import get_config
        return get_config().get_effective_num_ctx(self._explicit_max_context_tokens)

    @max_context_tokens.setter
    def max_context_tokens(self, val: Optional[int]):
        self._explicit_max_context_tokens = val

    @property
    def execute_tool(self):
        return self._execute_tool or execute_tool

    @execute_tool.setter
    def execute_tool(self, val):
        self._execute_tool = val

    @classmethod
    def is_refusal(cls, text: str) -> bool:
        """Checks if text contains signature conversational refusal phrases."""
        if not text:
            return False
        t_lower = text.lower()
        return any(phrase in t_lower for phrase in REFUSAL_PHRASES)

    @classmethod
    def is_interface_action_query(cls, text: str) -> bool:
        """Determines if a query targets an OS, browser, or interface action."""
        if not text:
            return False
        t_lower = text.lower()
        patterns = [
            r"\b(click|press|tap|double-click|right-click)\b",
            r"\b(open|launch|start|run|close|switch|focus)\b",
            r"\b(window|windows|workspace|workspaces|desktop|screen|display|monitor)\b",
            r"\b(browser|webpage|website|url|tab|button|icon|link|menu)\b",
            r"\b(inspect|navigate|scroll|type|look at|see|find|show me|view)\b",
        ]
        return any(re.search(pat, t_lower) for pat in patterns)

    @staticmethod
    async def get_active_window() -> Optional[Dict[str, Any]]:
        """Query hyprctl activewindow -j non-blockingly and return active window dictionary."""
        if not shutil.which("hyprctl"):
            return None
        try:
            proc = await asyncio.create_subprocess_exec(
                "hyprctl", "activewindow", "-j",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=1.0)
            if proc.returncode == 0 and stdout:
                data = json.loads(stdout.decode())
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return None

    @classmethod
    def get_active_window_sync(cls) -> Optional[Dict[str, Any]]:
        """Query hyprctl activewindow -j synchronously with quick timeout."""
        if not shutil.which("hyprctl"):
            return None
        try:
            import subprocess
            res = subprocess.run(
                ["hyprctl", "activewindow", "-j"],
                capture_output=True,
                timeout=0.5,
            )
            if res.returncode == 0 and res.stdout:
                data = json.loads(res.stdout.decode())
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return None

    async def get_active_window_context(self) -> Optional[str]:
        """Query hyprctl activewindow -j non-blockingly and format telemetry header."""
        data = await self.get_active_window()
        if not data:
            return None
        title = data.get("title", "")
        w_class = data.get("class", "")
        workspace = data.get("workspace", {})
        w_id = workspace.get("id", "") if isinstance(workspace, dict) else workspace
        if title or w_class:
            return f'[Active Window: title="{title}", class="{w_class}", workspace={w_id}]'
        return None

    def _build_system_prompt(self, base_directives: Optional[str] = None) -> str:
        """Compile hierarchical system directives, global user instructions, and local project rules."""
        from zelyr.core.instructions import InstructionManager
        manager = InstructionManager()
        return manager.get_compiled_instructions(base_directives=base_directives)

    @staticmethod
    def count_tokens(text: str) -> int:
        """Estimate token count for a text string."""
        if not text:
            return 0
        words = len(text.split())
        return max(len(text) // 4, int(words * 1.3), 1)

    @classmethod
    def count_message_tokens(cls, msg: dict) -> int:
        """Estimate token footprint of a chat completion message."""
        content = msg.get("content", "")
        tokens = 0
        if isinstance(content, str):
            tokens = cls.count_tokens(content)
        elif isinstance(content, list):
            tokens = sum(cls.count_tokens(c.get("text", "")) for c in content if isinstance(c, dict))
        elif isinstance(content, dict):
            tokens = cls.count_tokens(json.dumps(content))
        else:
            tokens = cls.count_tokens(str(content))

        if "tool_calls" in msg:
            tokens += cls.count_tokens(json.dumps(msg["tool_calls"]))
        return tokens + 4

    def prune_messages(
        self,
        messages: list[dict],
        max_context_tokens: Optional[int] = None,
        reserve_tokens: int = 1000,
        model_name: Optional[str] = None,
    ) -> list[dict]:
        """Adaptive context window budgeting: applies Stage 1 tool compaction and Stage 2
        sliding-window turn pruning with semantic summarization.
        """
        eff_max = max_context_tokens or self.max_context_tokens
        return self.budget_engine.apply_context_budget(
            messages,
            model_name=model_name,
            max_context_tokens=eff_max,
            reserve_tokens=reserve_tokens,
        )

    async def fetch_dynamic_context_length(self, model: str, base_url: str) -> Optional[int]:
        """Attempt to read context length dynamically from Ollama /api/show."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.post(f"{base_url}/api/show", json={"name": model})
                if resp.status_code == 200:
                    data = resp.json()
                    model_info = data.get("model_info", {})
                    for k, v in model_info.items():
                        if "context_length" in k and isinstance(v, int):
                            return v
        except Exception:
            pass
    @classmethod
    def extract_first_tool_call(
        cls,
        text: str,
        known_tools: Optional[set] = None,
    ) -> Optional[dict]:
        """Tolerantly extract the first valid JSON/XML tool call from model text.

        Resolves issues with small models (e.g. Qwen-7B) emitting concatenated
        JSON objects, markdown code fences embedded in prose, or reasoning text before tool calls.
        Safely extracts and returns the first valid tool call normalized to dual format:
        {"type": "function", "function": {"name": ..., "arguments": ...}, "name": ..., "arguments": ...}
        """
        if not text:
            return None

        def _format_res(name: Any, args: Any) -> Optional[dict]:
            if not name or not isinstance(name, str):
                return None
            clean_name = name
            if clean_name.startswith("functions."):
                clean_name = clean_name[len("functions."):]
            elif clean_name.startswith("tools."):
                clean_name = clean_name[len("tools."):]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            if not isinstance(args, dict):
                args = {}
            # Guard against model confusing interact_with_ui with interact_with_browser on web elements
            if clean_name == "interact_with_ui" and ("element_id" in args or args.get("action") in ("click_element", "fill_element")):
                clean_name = "interact_with_browser"
            if known_tools and clean_name not in known_tools:
                return None
            return {
                "type": "function",
                "function": {
                    "name": clean_name,
                    "arguments": args,
                },
                "name": clean_name,
                "arguments": args,
            }

        # 1. XML <tool_call> tags
        if "<tool_call>" in text:
            from zelyr.core.system_prompt import extract_xml_tool_call
            xml_res = extract_xml_tool_call(text)
            if xml_res and "function" in xml_res:
                fn = xml_res["function"]
                res = _format_res(fn.get("name", ""), fn.get("arguments", {}))
                if res:
                    return res

        # 2. Markdown code fences embedded anywhere in the text
        # Handles models placing conversational commentary before ```json ... ``` blocks
        for match in re.finditer(r"```(?:json)?\s*([\{\[].*?[\}\]])\s*```", text, re.DOTALL | re.IGNORECASE):
            block = match.group(1).strip()
            decoder = json.JSONDecoder()
            idx = 0
            while idx < len(block):
                brace_pos = block.find("{", idx)
                bracket_pos = block.find("[", idx)
                if brace_pos != -1 and bracket_pos != -1:
                    start_pos = min(brace_pos, bracket_pos)
                elif brace_pos != -1:
                    start_pos = brace_pos
                elif bracket_pos != -1:
                    start_pos = bracket_pos
                else:
                    break

                try:
                    obj, end_pos = decoder.raw_decode(block, start_pos)
                    if isinstance(obj, list) and obj:
                        first_item = obj[0]
                        if isinstance(first_item, dict):
                            fn_obj = first_item.get("function") if isinstance(first_item.get("function"), dict) else first_item
                            name = fn_obj.get("name") or fn_obj.get("tool")
                            args = fn_obj.get("arguments", fn_obj.get("parameters", fn_obj.get("args", {})))
                            res = _format_res(name, args)
                            if res:
                                return res
                    elif isinstance(obj, dict):
                        fn_obj = obj.get("function") if isinstance(obj.get("function"), dict) else obj
                        name = fn_obj.get("name") or fn_obj.get("tool")
                        args = fn_obj.get("arguments", fn_obj.get("parameters", fn_obj.get("args", {})))
                        res = _format_res(name, args)
                        if res:
                            return res
                    idx = end_pos
                except json.JSONDecodeError:
                    idx = start_pos + 1

        candidate = text.strip()
        # Clean enclosing markdown fences
        if candidate.startswith("```json"):
            candidate = candidate[7:]
        elif candidate.startswith("```"):
            candidate = candidate[3:]
        if candidate.endswith("```"):
            candidate = candidate[:-3]
        candidate = candidate.strip()

        # 3. Check if text starts with a JSON list of tool calls
        if candidate.startswith("["):
            try:
                decoder = json.JSONDecoder()
                parsed_list, _ = decoder.raw_decode(candidate, 0)
                if isinstance(parsed_list, list) and parsed_list:
                    first_item = parsed_list[0]
                    if isinstance(first_item, dict):
                        fn_obj = first_item.get("function") if isinstance(first_item.get("function"), dict) else first_item
                        name = fn_obj.get("name") or fn_obj.get("tool")
                        args = fn_obj.get("arguments", fn_obj.get("parameters", fn_obj.get("args", {})))
                        res = _format_res(name, args)
                        if res:
                            return res
            except Exception:
                pass

        # 4. Iterative JSONDecoder().raw_decode starting at each '{'
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(candidate):
            brace_pos = candidate.find("{", idx)
            if brace_pos == -1:
                break
            try:
                obj, end_pos = decoder.raw_decode(candidate, brace_pos)
                if isinstance(obj, dict):
                    fn_obj = obj.get("function") if isinstance(obj.get("function"), dict) else obj
                    name = fn_obj.get("name") or fn_obj.get("tool")
                    args = fn_obj.get("arguments", fn_obj.get("parameters", fn_obj.get("args", {})))
                    res = _format_res(name, args)
                    if res:
                        return res
                idx = brace_pos + 1
            except json.JSONDecodeError:
                idx = brace_pos + 1

        # 5. Fallback regex search for single-level JSON tool call objects
        for match in re.finditer(r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*\}', candidate):
            try:
                extracted = json.loads(match.group(0))
                if isinstance(extracted, dict) and "name" in extracted:
                    name = extracted["name"]
                    args = extracted.get("arguments", extracted.get("parameters", extracted.get("args", {})))
                    res = _format_res(name, args)
                    if res:
                        return res
            except Exception:
                pass

        return None

    async def _stream_model_events(self, payload: dict, base_url: str, stream_provider: Optional[Any] = None) -> AsyncGenerator[dict, None]:
        active_provider = stream_provider or getattr(self, "stream_provider", None) or getattr(self, "model_provider", None)
        if active_provider is not None:
            # Deterministic test provider injection: skip external Ollama calls
            stream_fn = getattr(active_provider, "stream_chat", None) or getattr(active_provider, "generate_stream", None)
            if callable(stream_fn):
                stream_iter = stream_fn(payload)
            elif callable(active_provider):
                stream_iter = active_provider(payload)
            else:
                raise TypeError(f"Invalid stream_provider {type(active_provider)}. Expected callable or object with stream_chat().")

            async for item in stream_iter:
                if isinstance(item, str):
                    line = item.strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        line = line[6:].strip()
                    if line == "[DONE]":
                        break
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
                elif isinstance(item, dict):
                    yield item
            return

        # Default production path: Ollama over HTTP
        options = payload.get("options", {})
        from zelyr.config import get_config
        prewarm_ctx = get_config().get_effective_num_ctx(
            options.get("num_ctx") or self._explicit_max_context_tokens
        )
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                await client.post(
                    f"{base_url}/api/generate",
                    json={
                        "model": payload["model"],
                        "options": {"num_ctx": prewarm_ctx, "num_gpu": 99},
                        "keep_alive": "5m",
                    },
                )
        except Exception:
            pass

        timeout_config = httpx.Timeout(connect=10.0, read=300.0, write=20.0, pool=20.0)
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            async with client.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                json=payload
            ) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    yield {"_api_error": f"HTTP {resp.status_code}: {resp.text}"}
                    return

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            yield json.loads(data_str)
                        except json.JSONDecodeError:
                            pass

    async def generate_stream(
        self,
        payload: dict,
        depth: int = 0,
        is_retry: bool = False,
        stream_provider: Optional[Callable[[dict], AsyncGenerator[dict, None]]] = None,
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
        tool_executor: Optional[Callable[[str, dict], Awaitable[Any]]] = None,
        db_path: Optional[str] = None,
        max_depth: Optional[int] = None,
    ) -> AsyncGenerator[dict, None]:
        eff_max_depth = max_depth if max_depth is not None else getattr(self, "max_depth", self.max_steps)
        eff_stream_provider = stream_provider or getattr(self, "stream_provider", None) or getattr(self, "model_provider", None)
        eff_tool_schemas = tool_schemas if tool_schemas is not None else getattr(self, "tool_schemas", None)
        eff_tool_executor = tool_executor or getattr(self, "tool_executor", None)
        eff_db_path = db_path or getattr(self, "db_path", None)

        if depth >= eff_max_depth:
            yield {"choices": [{"delta": {"content": f"\n[Notice: Reached maximum autonomous ReAct steps ({eff_max_depth}).]\n"}}]}
            return

        from zelyr.config import get_config
        config = get_config()
        base_url = getattr(config, "ollama_base_url", "http://127.0.0.1:11434").rstrip("/")

        # Inspect if preceding turn was a failed tool execution requiring actuator recovery
        is_recovery_turn = False
        recovery_hint = ""
        allowed_recovery_actions = []

        messages = payload.get("messages", [])
        if messages:
            for m in reversed(messages):
                if m.get("role") == "user" and "[ACTUATOR RECOVERY REQUIRED]" in str(m.get("content", "")):
                    continue
                if m.get("role") == "tool":
                    c = str(m.get("content", ""))
                    if "[Tool Error]:" in c:
                        is_recovery_turn = True
                        recovery_hint = m.get("remedy_hint") or ""
                        if not recovery_hint and "Recovery Hint:" in c:
                            recovery_hint = c.split("Recovery Hint:", 1)[1].strip()
                        allowed_recovery_actions = m.get("allowed_actions") or []
                    break
                elif m.get("role") == "assistant" and m.get("tool_calls"):
                    break

        if depth == 0:
            if eff_stream_provider is None:
                # Eagerly ensure WebExtension bridge server is running on 127.0.0.1:41144
                try:
                    from zelyr.tools.browser_extension import get_bridge
                    bridge = get_bridge()
                    if bridge.server is None:
                        asyncio.create_task(bridge.start_server())
                except Exception:
                    pass

                active_win_header = await self.get_active_window_context()
                if active_win_header:
                    for msg in reversed(payload.get("messages", [])):
                        if msg.get("role") == "user":
                            content = msg.get("content", "")
                            if isinstance(content, str):
                                if "[Active Window:" not in content:
                                    msg["content"] = f"{active_win_header}\n{content}"
                            elif isinstance(content, list):
                                for part in content:
                                    if isinstance(part, dict) and part.get("type") == "text":
                                        t = part.get("text", "")
                                        if "[Active Window:" not in t:
                                            part["text"] = f"{active_win_header}\n{t}"
                                        break
                            break

            system_prompt_content = self._build_system_prompt()

            # Query relevance-based persistent memory (Phase 2 token-budgeted hydration)
            user_prompt = ""
            for msg in reversed(payload.get("messages", [])):
                if msg.get("role") == "user":
                    c = msg.get("content", "")
                    if isinstance(c, str):
                        user_prompt = c
                    elif isinstance(c, list):
                        user_prompt = " ".join(part.get("text", "") for part in c if isinstance(part, dict))
                    break

            if user_prompt and not is_retry and not is_recovery_turn:
                try:
                    from zelyr.memory.hydration import hydrate_context_memory
                    clean_search = re.sub(r"\[Active Window:.*?\]\n?", "", user_prompt).strip()
                    mem_block = hydrate_context_memory(clean_search, db_path=eff_db_path, max_tokens=350)
                    if mem_block:
                        system_prompt_content += f"\n\n{mem_block}"
                    elif eff_stream_provider is None:
                        from zelyr.memory.semantic import search_memories
                        recalled = search_memories(clean_search, top_k=3)
                        if recalled:
                            mem_block = (
                                "\n\n[Contextual Memory]:\n"
                                "<recalled_memory>\n"
                                + "\n".join(f"- {m}" for m in recalled)
                                + "\n</recalled_memory>"
                            )
                            system_prompt_content += mem_block
                except Exception:
                    pass

            system_prompt = {
                "role": "system",
                "content": system_prompt_content,
            }

            if not payload.get("messages") or payload["messages"][0].get("role") != "system":
                payload.setdefault("messages", []).insert(0, system_prompt)
            else:
                payload["messages"][0]["content"] += f"\n\n{system_prompt_content}"

        # Apply adaptive context window budgeting and historical tool compaction
        eff_model = payload.get("model") or config.ollama_model
        payload["messages"] = self.prune_messages(
            payload.get("messages", []),
            max_context_tokens=self.max_context_tokens,
            reserve_tokens=1000,
            model_name=eff_model,
        )
        user_query_for_intent = ""
        for msg in reversed(payload.get("messages", [])):
            if msg.get("role") == "user":
                c = msg.get("content", "")
                if isinstance(c, str):
                    user_query_for_intent = c
                elif isinstance(c, list):
                    user_query_for_intent = " ".join(part.get("text", "") for part in c if isinstance(part, dict))
                break
        if user_query_for_intent.startswith("[Active Window:"):
            parts = user_query_for_intent.split("\n", 1)
            if len(parts) > 1:
                user_query_for_intent = parts[1]
            else:
                user_query_for_intent = re.sub(r"^\[Active Window:.*?\]\s*", "", user_query_for_intent)
        else:
            user_query_for_intent = re.sub(r"^\[Active Window:[^\]]*\]\s*", "", user_query_for_intent)
        is_interface_action = self.is_interface_action_query(user_query_for_intent)

        # Inspect query intent, active window, and bridge connectivity to filter active schemas
        active_window = await self.get_active_window()
        bridge_connected = False
        try:
            from zelyr.tools.browser_extension import get_bridge
            bridge_connected = get_bridge().is_connected()
        except Exception:
            pass

        tier = self.route_tier(
            user_query_for_intent,
            active_window=active_window,
            bridge_connected=bridge_connected,
        )
        if is_recovery_turn and allowed_recovery_actions:
            tier2_actions = {"get_page_snapshot", "scroll_page", "click_element", "fill_element", "list_tabs", "open_tab", "switch_tab", "navigate_url"}
            if any(a in tier2_actions for a in allowed_recovery_actions):
                tier = "tier2_browser"

        from zelyr.core.plugins import filter_tool_schemas_by_tier
        from zelyr.agents.actuator_arbiter import is_browser_window
        browser_active = is_browser_window(active_window)

        if eff_tool_schemas is not None:
            payload["tools"] = list(eff_tool_schemas)
        else:
            active_tools = payload.get("tools") or get_tool_schemas(0)
            payload["tools"] = filter_tool_schemas_by_tier(active_tools, tier, is_browser=browser_active)

            # Strict negative gating: purge Tier 3 interact_with_ui whenever a browser window is active
            if browser_active:
                payload["tools"] = [
                    t for t in payload["tools"]
                    if (t.get("function", {}).get("name") or t.get("name")) != "interact_with_ui"
                ]

        payload["model"] = payload.get("model") or config.ollama_model

        # Scrub images from payload to prevent context-window collapse
        for msg in payload.get("messages", []):
            if "images" in msg:
                del msg["images"]
            if isinstance(msg.get("content"), list):
                # Filter out image_url items
                text_parts = [c.get("text", "") for c in msg["content"] if c.get("type") == "text"]
                msg["content"] = "\n".join(text_parts)

        # Enforce Two-Brain VRAM Eviction: Evict vision engine from CUDA memory before text streaming
        try:
            from zelyr.core.vision_engine.singleton import evict_grounding_engine
            evict_grounding_engine()
        except Exception:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # Enforce 100% GPU layer offload and adequate context size in Ollama
        options = payload.setdefault("options", {})
        from zelyr.config import get_config
        eff_ctx = get_config().get_effective_num_ctx(
            options.get("num_ctx") or self._explicit_max_context_tokens
        )
        options["num_ctx"] = eff_ctx
        options.setdefault("num_gpu", 99)

        pending_tool_calls = []
        is_reasoning = False
        collected_assistant_text = []
        buffered_chunks = []
        buffering = (depth == 0 and not is_retry and is_interface_action) or (depth > 0) or is_recovery_turn
        refusal_detected = False

        try:
            async for data in self._stream_model_events(payload, base_url, stream_provider=eff_stream_provider):
                if "_api_error" in data:
                    yield {"choices": [{"delta": {"content": f"\n⚠️ [API Error] {data['_api_error']}\n"}}]}
                    return

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
                        idx = tc.get("index", 0)
                        if idx > 0:
                            continue  # Enforce strictly ONE tool call per response
                        while len(pending_tool_calls) <= idx:
                            pending_tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                        if "id" in tc and tc["id"]:
                            pending_tool_calls[idx]["id"] = tc["id"]
                        if "function" in tc and isinstance(tc["function"], dict):
                            if "name" in tc["function"] and tc["function"]["name"]:
                                pending_tool_calls[idx]["function"]["name"] = tc["function"]["name"]
                            if "arguments" in tc["function"] and tc["function"]["arguments"] is not None:
                                incoming_args = tc["function"]["arguments"]
                                if isinstance(incoming_args, dict):
                                    pending_tool_calls[idx]["function"]["arguments"] = json.dumps(incoming_args)
                                else:
                                    pending_tool_calls[idx]["function"]["arguments"] += str(incoming_args)
                        elif "name" in tc and tc["name"]:
                            pending_tool_calls[idx]["function"]["name"] = tc["name"]
                            if "arguments" in tc and tc["arguments"] is not None:
                                incoming_args = tc["arguments"]
                                if isinstance(incoming_args, dict):
                                    pending_tool_calls[idx]["function"]["arguments"] = json.dumps(incoming_args)
                                else:
                                    pending_tool_calls[idx]["function"]["arguments"] += str(incoming_args)

                if buffering:
                    if "tool_calls" in delta or pending_tool_calls:
                        buffering = False
                        clean_so_far = "".join(collected_assistant_text)
                        if not is_recovery_turn and not clean_so_far.lstrip().startswith(("{", "[", "```", "<")):
                            for b in buffered_chunks:
                                yield b
                        buffered_chunks.clear()
                        yield data
                    else:
                        buffered_chunks.append(data)
                        text_so_far = "".join(collected_assistant_text)
                        clean_text = re.sub(r"<think>.*?</think>", "", text_so_far, flags=re.DOTALL)
                        stripped = clean_text.lstrip()
                        if self.is_refusal(clean_text):
                            refusal_detected = True
                            break
                        elif stripped and not stripped.startswith(("{", "[", "```", "<")):
                            if not is_recovery_turn and (depth > 0 or len(clean_text) > 250):
                                buffering = False
                                for b in buffered_chunks:
                                    yield b
                                buffered_chunks.clear()
                        elif not is_recovery_turn and len(clean_text) > 4000:
                            buffering = False
                            for b in buffered_chunks:
                                yield b
                            buffered_chunks.clear()
                else:
                    yield data
        except httpx.ConnectError as exc:
            yield {"choices": [{"delta": {"content": f"\n⚠️ [Connection Error] Could not connect to local AI engine at {base_url}: {exc}. Ensure Ollama is running.\n"}}]}
            return
        except httpx.RequestError as exc:
            yield {"choices": [{"delta": {"content": f"\n⚠️ [Request Error] Communication failure with {base_url}: {exc}\n"}}]}
            return

        if not refusal_detected and (depth == 0 and not is_retry and is_interface_action):
            text_so_far = "".join(collected_assistant_text)
            clean_text = re.sub(r"<think>.*?</think>", "", text_so_far, flags=re.DOTALL)
            if not pending_tool_calls and self.is_refusal(clean_text):
                refusal_detected = True

        if refusal_detected:
            buffered_chunks.clear()
            collected_assistant_text.clear()
            pending_tool_calls.clear()

            override_msg = {
                "role": "user",
                "content": "[SYSTEM OVERRIDE]: You possess tools to inspect and interact with the environment. Call the appropriate tool immediately without commentary."
            }
            payload.setdefault("messages", []).append(override_msg)
            payload.setdefault("options", {})["temperature"] = 0.1
            payload["temperature"] = 0.1

            async for retry_chunk in self.generate_stream(payload, depth=depth, is_retry=True):
                yield retry_chunk
            return

        # Enforce single tool execution constraint if multiple tool calls were received
        if len(pending_tool_calls) > 1:
            pending_tool_calls = pending_tool_calls[:1]

        if not pending_tool_calls and collected_assistant_text:
            full_text = "".join(collected_assistant_text).strip()
            known_tools = set()
            if payload.get("tools"):
                for t in payload["tools"]:
                    if isinstance(t, dict):
                        if "function" in t and "name" in t["function"]:
                            known_tools.add(t["function"]["name"])
                        elif "name" in t:
                            known_tools.add(t["name"])

            tool_call = self.extract_first_tool_call(full_text, known_tools)
            if tool_call:
                func_name = tool_call.get("function", {}).get("name") or tool_call.get("name")
                func_args = tool_call.get("function", {}).get("arguments") or tool_call.get("arguments", {})
                if isinstance(func_args, str):
                    try:
                        func_args = json.loads(func_args)
                    except Exception:
                        pass
                if not isinstance(func_args, dict):
                    func_args = {}
                func_args_str = json.dumps(func_args)
                call_id = "call_native_0"
                pending_tool_calls = [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": func_args_str
                    },
                    "name": func_name,
                    "arguments": func_args
                }]
                buffered_chunks.clear()
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": func_name,
                                    "arguments": func_args_str
                                }
                            }]
                        }
                    }]
                }
                collected_assistant_text = []

        # Closed-loop error recovery enforcement (Anti-consultant rule):
        # If the model responds to a recovery hint with conversational text instead of
        # an actuator tool call, and depth < 4, do not terminate the loop.
        # Reject conversational prose, clear buffers, and trigger a fast retry directing
        # the model to invoke the recovery tool immediately.
        if is_recovery_turn and not pending_tool_calls and depth < (eff_max_depth - 1):
            buffered_chunks.clear()
            collected_assistant_text.clear()

            hint_text = recovery_hint or "Execute the recovery tool immediately."
            allowed_clause = f" Allowed recovery actions: {', '.join(allowed_recovery_actions)}." if allowed_recovery_actions else ""
            override_msg = {
                "role": "user",
                "content": (
                    f"[ACTUATOR RECOVERY REQUIRED]: A previous action failed with recovery hint: '{hint_text}'.{allowed_clause} "
                    f"You must NOT output conversational apologies, explanations, or tutorials. "
                    f"Call the appropriate actuator recovery tool immediately."
                ),
            }
            payload.setdefault("messages", []).append(override_msg)
            payload.setdefault("options", {})["temperature"] = 0.1
            payload["temperature"] = 0.1

            async for retry_chunk in self.generate_stream(
                payload,
                depth=depth + 1,
                is_retry=True,
                stream_provider=eff_stream_provider,
                tool_schemas=eff_tool_schemas,
                tool_executor=eff_tool_executor,
                db_path=eff_db_path,
                max_depth=eff_max_depth,
            ):
                yield retry_chunk
            return

        if not pending_tool_calls and buffered_chunks:
            for b in buffered_chunks:
                yield b
            buffered_chunks.clear()

        if pending_tool_calls:
            # Normalize tool calls: Ollama /api/chat requires function.arguments to be a native JSON dict
            normalized_tool_calls = []
            for tc in pending_tool_calls:
                fn_name = tc.get("function", {}).get("name") or tc.get("name") or ""
                fn_args_raw = tc.get("function", {}).get("arguments") or tc.get("arguments", {})
                tc_id = tc.get("id") or "call_native_0"

                if isinstance(fn_args_raw, dict):
                    fn_args = fn_args_raw
                elif isinstance(fn_args_raw, str):
                    raw_str = fn_args_raw.strip()
                    if not raw_str:
                        fn_args = {}
                    else:
                        try:
                            fn_args = json.loads(raw_str)
                        except Exception:
                            try:
                                import ast
                                val = ast.literal_eval(raw_str)
                                fn_args = val if isinstance(val, dict) else {}
                            except Exception:
                                fn_args = {}
                else:
                    fn_args = {}

                if not isinstance(fn_args, dict):
                    fn_args = {}

                # Guard against model confusing interact_with_ui with interact_with_browser on web elements
                if fn_name == "interact_with_ui" and ("element_id" in fn_args or fn_args.get("action") in ("click_element", "fill_element")):
                    fn_name = "interact_with_browser"

                normalized_tool_calls.append({
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "arguments": fn_args,
                    }
                })

            assistant_msg = {
                "role": "assistant",
                "content": "".join(collected_assistant_text) if collected_assistant_text else "",
                "tool_calls": normalized_tool_calls,
            }
            payload["messages"].append(assistant_msg)

            for tc in normalized_tool_calls:
                func_name = tc["function"]["name"]
                func_args = tc["function"]["arguments"]
                tool_call_id = tc["id"]

                timeout = get_tier_timeout(func_name)
                timeout_int = int(timeout)
                try:
                    if eff_tool_executor is not None:
                        try:
                            exec_call = eff_tool_executor(func_name, func_args)
                            if inspect.isawaitable(exec_call):
                                tool_result = await asyncio.wait_for(exec_call, timeout=timeout)
                            else:
                                tool_result = exec_call
                        except TypeError:
                            exec_call = eff_tool_executor(func_name, **func_args)
                            if inspect.isawaitable(exec_call):
                                tool_result = await asyncio.wait_for(exec_call, timeout=timeout)
                            else:
                                tool_result = exec_call
                    else:
                        tool_result = await asyncio.wait_for(
                            self.execute_tool(func_name, **func_args),
                            timeout=timeout,
                        )
                    
                    if hasattr(tool_result, "to_dict"):
                        tool_dict = tool_result.to_dict(tool=func_name, arguments=func_args)
                        tool_result = json.dumps(tool_dict, indent=2)
                    elif not isinstance(tool_result, str):
                        tool_result = json.dumps(tool_result, indent=2)
                    res_str = tool_result
                except (TimeoutError, asyncio.TimeoutError):
                    res_str = json.dumps({
                        "success": False,
                        "error": f"Tool execution timed out after {timeout_int}s. Try a faster Tier 1 command or verify active window.",
                        "remedy_hint": "Try a faster Tier 1 command or verify active window.",
                        "allowed_actions": ["execute_command", "manage_desktop_window"],
                    })
                except Exception as e:
                    res_str = json.dumps({
                        "success": False,
                        "error": str(e),
                        "status": "tool_execution_failed",
                        "remedy_hint": "Inspect error or try alternative action.",
                        "allowed_actions": [],
                    })

                # Format tool output: if tool execution failed, format as structured recovery observation
                is_failed = False
                err_text = ""
                remedy_hint_out = ""
                allowed_actions_out = []

                try:
                    parsed_res = json.loads(res_str) if isinstance(res_str, str) else res_str
                    if isinstance(parsed_res, dict):
                        if parsed_res.get("success") is False:
                            is_failed = True
                        elif "error" in parsed_res and parsed_res["error"]:
                            is_failed = True
                        elif isinstance(parsed_res.get("result"), dict) and parsed_res["result"].get("error"):
                            is_failed = True

                        if is_failed:
                            err_text = (
                                parsed_res.get("error")
                                or (parsed_res.get("result", {}).get("error") if isinstance(parsed_res.get("result"), dict) else None)
                                or "Tool execution failed"
                            )
                            remedy_hint_out = (
                                parsed_res.get("remedy_hint")
                                or (parsed_res.get("result", {}).get("remedy_hint") if isinstance(parsed_res.get("result"), dict) else None)
                                or (parsed_res.get("result", {}).get("metadata", {}).get("remedy_hint") if isinstance(parsed_res.get("result"), dict) else None)
                                or ""
                            )
                            allowed_actions_out = (
                                parsed_res.get("allowed_actions")
                                or (parsed_res.get("result", {}).get("allowed_actions") if isinstance(parsed_res.get("result"), dict) else None)
                                or (parsed_res.get("result", {}).get("metadata", {}).get("allowed_actions") if isinstance(parsed_res.get("result"), dict) else None)
                                or []
                            )
                except Exception:
                    pass

                if is_failed and err_text:
                    err_clean = str(err_text).rstrip(".")
                    if remedy_hint_out:
                        content_for_msg = f"[Tool Error]: {err_clean}. Recovery Hint: {remedy_hint_out}"
                    else:
                        content_for_msg = f"[Tool Error]: {err_clean}"
                else:
                    content_for_msg = res_str

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": func_name,
                    "content": content_for_msg,
                }
                if is_failed and remedy_hint_out:
                    tool_msg["remedy_hint"] = remedy_hint_out
                    tool_msg["allowed_actions"] = allowed_actions_out
                payload["messages"].append(tool_msg)

                session_id = payload.get("session_id")
                if session_id:
                    try:
                        from zelyr.db.memory import add_message
                        add_message(session_id, "tool", f"[{func_name}] {res_str}", db_path=eff_db_path)
                    except Exception:
                        pass

            if depth < eff_max_depth:
                async for chunk in self.generate_stream(
                    payload,
                    depth=depth + 1,
                    stream_provider=eff_stream_provider,
                    tool_schemas=eff_tool_schemas,
                    tool_executor=eff_tool_executor,
                    db_path=eff_db_path,
                    max_depth=eff_max_depth,
                ):
                    yield chunk

    @classmethod
    def route_tier(
        cls,
        task_description: str,
        active_window: Optional[Dict[str, Any]] = None,
        bridge_connected: Optional[bool] = None,
    ) -> str:
        """Determines the automation tier based on task description, active window, and bridge status.

        Returns:
            'tier1_ipc': For window/workspace/OS management
            'tier2_browser': For browser/web/DOM interaction
            'tier3_vision': Fallback for non-accessible canvas/games (when not a browser)
        """
        from zelyr.agents.actuator_arbiter import ActuatorArbiter

        if active_window is None:
            active_window = cls.get_active_window_sync()

        if bridge_connected is None:
            try:
                from zelyr.tools.browser_extension import get_bridge
                bridge = get_bridge()
                bridge_connected = bridge.is_connected()
            except Exception:
                bridge_connected = False

        return ActuatorArbiter.arbitrate(
            task=task_description,
            active_window=active_window,
            bridge_connected=bridge_connected,
        )

    @classmethod
    def get_tier_tool(
        cls,
        task_description: str,
        active_window: Optional[Dict[str, Any]] = None,
        bridge_connected: Optional[bool] = None,
    ) -> str:
        """Returns the primary tool name corresponding to the routed tier."""
        tier = cls.route_tier(
            task_description,
            active_window=active_window,
            bridge_connected=bridge_connected,
        )
        task_lower = task_description.lower()
        if tier == "tier1_ipc":
            if any(k in task_lower for k in ("clipboard", "wl-copy", "wl-paste", "copy to clipboard", "paste")):
                return "manage_system_clipboard"
            if any(k in task_lower for k in ("journal", "journalctl", "systemd log", "service log")):
                return "query_system_journal"
            if any(k in task_lower for k in ("process", "pid", "kill")):
                return "manage_system_process"
            if any(k in task_lower for k in ("notify", "notification", "alert")):
                return "send_desktop_notification"
            if any(k in task_lower for k in ("network", "gateway", "tailscale", "dns", "ping", "vpn")):
                return "inspect_network"
            if any(k in task_lower for k in ("media", "music", "volume", "playerctl", "playback")):
                return "manage_media_playback"
            if any(k in task_lower for k in ("file", "rollback", "undo", "patch")):
                return "manage_workspace_file"
            return "manage_desktop_window"
        tier_map = {
            "tier1_ipc": "manage_desktop_window",
            "tier2_browser": "interact_with_browser",
            "tier3_vision": "interact_with_ui",
        }
        return tier_map.get(tier, "interact_with_ui")

