import asyncio
import inspect
import json
import re
import shutil
import uuid
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
        enable_cycle_detection: bool = True,
    ):
        self._explicit_max_context_tokens = max_context_tokens
        self.max_steps = max_depth if max_depth is not None else max_steps
        self.max_depth = self.max_steps
        self.enable_cycle_detection = enable_cycle_detection
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
    SHELL_TOOL_ALIASES = {
        "shell_exec",
        "bash",
        "sh",
        "terminal",
        "run_command",
        "run_shell",
        "cmd",
        "exec_command",
        "shell_command",
        "shell",
    }

    FILE_TOOL_ALIASES = {
        "read_file",
        "view_file",
    }

    VERIFY_TOOL_ALIASES = {
        "assert_outcome",
        "check_state",
        "verify_state",
        "verify",
    }

    DESKTOP_TOOL_ALIASES = {
        "switch_workspace",
        "focus_window",
        "desktop_window",
        "hyprland_ipc",
        "hyprland",
        "window_manager",
    }

    BROWSER_TOOL_ALIASES = {
        "browser",
        "cdp",
        "web_browser",
        "chrome",
        "chromium",
    }

    MEMORY_TOOL_ALIASES = {
        "remember",
        "remember_fact",
        "search_memory",
        "save_memory",
        "recall_memory",
    }

    CRAFT_TOOL_ALIASES = {
        "create_tool",
        "write_tool",
        "hotload_tool",
    }

    ROUTINE_TOOL_ALIASES = {
        "schedule_task",
        "create_routine",
        "schedule_routine",
        "cron",
        "cron_job",
    }

    TOOL_ALIAS_MAP = {
        **{alias: "execute_command" for alias in SHELL_TOOL_ALIASES},
        **{alias: "manage_workspace_file" for alias in FILE_TOOL_ALIASES},
        **{alias: "verify_outcome" for alias in VERIFY_TOOL_ALIASES},
        **{alias: "manage_desktop_window" for alias in DESKTOP_TOOL_ALIASES},
        **{alias: "interact_with_browser" for alias in BROWSER_TOOL_ALIASES},
        **{alias: "manage_memory" for alias in MEMORY_TOOL_ALIASES},
        **{alias: "craft_tool" for alias in CRAFT_TOOL_ALIASES},
        **{alias: "manage_routine" for alias in ROUTINE_TOOL_ALIASES},
    }

    @staticmethod
    def _strip_tool_prefixes(name: Any) -> str:
        """Strip common prefixes ('tool.', 'tools.', 'functions.', 'function.') without alias mapping."""
        if not name or not isinstance(name, str):
            return ""
        clean = name.strip()
        changed = True
        while changed:
            changed = False
            for prefix in ("tool.", "tools.", "functions.", "function."):
                if clean.startswith(prefix):
                    clean = clean.removeprefix(prefix).strip()
                    changed = True
        return clean

    @classmethod
    def normalize_tool_name(cls, name: Any, active_schemas: Optional[list] = None) -> str:
        """Normalize a tool name by stripping common prefixes and mapping hallucinated aliases to canonical tools."""
        clean = cls._strip_tool_prefixes(name)
        if not clean:
            return ""
        if active_schemas:
            explicit_names = set()
            for t in active_schemas:
                if isinstance(t, dict):
                    fn = t.get("function")
                    if isinstance(fn, dict) and fn.get("name"):
                        explicit_names.add(cls._strip_tool_prefixes(fn["name"]))
                    elif t.get("name"):
                        explicit_names.add(cls._strip_tool_prefixes(t["name"]))
            if clean in explicit_names:
                return clean
        target = cls.TOOL_ALIAS_MAP.get(clean)
        if target and target in cls.get_known_tool_names(active_schemas):
            return target
        return clean

    @classmethod
    def normalize_tool_arguments(
        cls,
        clean_name: str,
        args: Any,
        original_name: Optional[str] = None,
    ) -> dict:
        """Normalize tool arguments for aliased or hallucinated tool calls."""
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass
        if not isinstance(args, dict):
            args = {}
        else:
            args = dict(args)

        raw_orig = cls._strip_tool_prefixes(original_name) if original_name else ""

        if clean_name == "execute_command":
            if "command" not in args:
                for alt_key in ("cmd", "script", "shell_command", "command_line", "input"):
                    if alt_key in args:
                        args["command"] = args.pop(alt_key)
                        break

        elif clean_name == "manage_workspace_file":
            if raw_orig in cls.FILE_TOOL_ALIASES or "action" not in args:
                args.setdefault("action", "read")
            if "path" not in args:
                for alt_key in ("file_path", "filepath", "file", "filename"):
                    if alt_key in args:
                        args["path"] = args.pop(alt_key)
                        break

        elif clean_name == "verify_outcome":
            if "check" not in args:
                for alt_key in ("assertion", "type", "check_type", "action"):
                    if alt_key in args:
                        args["check"] = args.pop(alt_key)
                        break
            if "target" not in args:
                for alt_key in ("path", "file", "file_path", "filepath", "cmd", "command", "process", "port", "name"):
                    if alt_key in args:
                        args["target"] = args.pop(alt_key)
                        break
            if "expected" not in args:
                for alt_key in ("pattern", "content", "value", "hash", "sha256", "regex"):
                    if alt_key in args:
                        args["expected"] = args.pop(alt_key)
                        break

        elif clean_name == "manage_desktop_window":
            if raw_orig == "switch_workspace":
                args.setdefault("action", "workspace")
                if "target" not in args:
                    for alt_key in ("workspace", "workspace_id", "ws", "name", "id"):
                        if alt_key in args:
                            args["target"] = args.pop(alt_key)
                            break
            elif raw_orig == "focus_window":
                args.setdefault("action", "focus")
                if "target" not in args:
                    for alt_key in ("window", "address", "title", "class_name", "app", "name"):
                        if alt_key in args:
                            args["target"] = args.pop(alt_key)
                            break
            elif "action" not in args:
                args.setdefault("action", "get_active")

        elif clean_name == "interact_with_browser":
            act = args.get("action")
            if not act:
                if "url" in args:
                    args["action"] = "navigate_url"
                elif "element_id" in args:
                    args["action"] = "click_element"
                else:
                    args["action"] = "get_page_snapshot"
            else:
                act_str = str(act).strip().lower()
                if act_str in ("snapshot", "page_snapshot"):
                    args["action"] = "get_page_snapshot"
                elif act_str in ("navigate", "goto"):
                    args["action"] = "navigate_url"
                elif act_str == "click" and "element_id" in args:
                    args["action"] = "click_element"
                elif act_str == "type" and "element_id" in args:
                    args["action"] = "fill_element"
                elif act_str in ("tabs",):
                    args["action"] = "list_tabs"
            if "url" not in args:
                for alt_key in ("link", "target_url", "uri", "site"):
                    if alt_key in args:
                        args["url"] = args.pop(alt_key)
                        break

        elif clean_name == "manage_memory":
            if "content" not in args and "fact" in args:
                args["content"] = args.pop("fact")
            if "action" not in args:
                if "content" in args:
                    args["action"] = "store"
                elif "query" in args:
                    args["action"] = "search"
                elif "memory_id" in args:
                    args["action"] = "get"
                else:
                    args["action"] = "store"

        elif clean_name == "craft_tool":
            if "code" not in args:
                for alt_key in ("source", "source_code", "tool_code", "implementation"):
                    if alt_key in args:
                        args["code"] = args.pop(alt_key)
                        break
            if "name" not in args:
                for alt_key in ("tool_name", "id", "tool_id"):
                    if alt_key in args:
                        args["name"] = args.pop(alt_key)
                        break

        elif clean_name == "manage_routine":
            if "schedule" not in args:
                for alt_key in ("cron_expression", "interval", "cron", "schedule_expr", "time"):
                    if alt_key in args:
                        args["schedule"] = args.pop(alt_key)
                        break
            if "action_payload" not in args:
                for alt_key in ("payload", "command", "instruction", "prompt", "task"):
                    if alt_key in args:
                        args["action_payload"] = args.pop(alt_key)
                        break
            if "action" not in args:
                if "schedule" in args:
                    args["action"] = "create"
                elif "name" in args:
                    args["action"] = "run"
                else:
                    args["action"] = "list"

        return args

    @classmethod
    def get_known_tool_names(cls, active_schemas: Optional[list] = None) -> set:
        """Return the set of valid tool names from active schemas, dynamic plugins, and ToolRegistry."""
        known: set = set()
        if active_schemas:
            for t in active_schemas:
                if isinstance(t, dict):
                    fn = t.get("function")
                    if isinstance(fn, dict) and fn.get("name"):
                        known.add(cls._strip_tool_prefixes(fn["name"]))
                    elif t.get("name"):
                        known.add(cls._strip_tool_prefixes(t["name"]))
        try:
            from zelyr.core.plugins import PROVISIONED_TOOLS, _schemas
            known.update(PROVISIONED_TOOLS)
            for s in _schemas:
                if isinstance(s, dict):
                    fn = s.get("function")
                    if isinstance(fn, dict) and fn.get("name"):
                        known.add(cls._strip_tool_prefixes(fn["name"]))
                    elif s.get("name"):
                        known.add(cls._strip_tool_prefixes(s["name"]))
        except Exception:
            pass
        try:
            from zelyr.tool_registry import BUILTIN_TOOLS, SWARM_TOOLS, ToolRegistry
            known.update(BUILTIN_TOOLS.keys())
            known.update(SWARM_TOOLS.keys())
            known.update({
                "safe_file_search",
                "file_opener",
                "read_file",
                "write_file",
                "shell_command",
                "run_command",
            })
            reg = ToolRegistry()
            known.update(reg.list_tools().keys())
        except Exception:
            pass
        return known

    @classmethod
    def extract_first_tool_call(
        cls,
        text: str,
        known_tools: Optional[set] = None,
        active_schemas: Optional[list] = None,
    ) -> Optional[dict]:
        """Tolerantly extract the first valid JSON/XML tool call from model text.

        Resolves issues with small models (e.g. Qwen-7B) emitting concatenated
        JSON objects, markdown code fences embedded in prose, or reasoning text before tool calls.
        Safely extracts and returns the first valid tool call normalized to dual format:
        {"type": "function", "function": {"name": ..., "arguments": ...}, "name": ..., "arguments": ...}
        """
        if not text:
            return None

        effective_known = (
            {cls.normalize_tool_name(k, active_schemas=active_schemas) for k in known_tools if isinstance(k, str)}
            if known_tools is not None
            else cls.get_known_tool_names(active_schemas)
        )

        def _extract_raw_args(fn_obj: dict) -> Any:
            for key in ("arguments", "parameters", "args"):
                if key in fn_obj:
                    return fn_obj[key]
            # Handle flat tool call objects like {"name": "shell_exec", "command": "uptime -p"}
            extra = {
                k: v
                for k, v in fn_obj.items()
                if k not in ("name", "tool", "type", "id", "index", "function")
            }
            return extra if extra else {}

        def _format_res(name: Any, args: Any) -> Optional[dict]:
            clean_name = cls.normalize_tool_name(name, active_schemas=active_schemas)
            if not clean_name:
                return None
            args = cls.normalize_tool_arguments(clean_name, args, original_name=str(name) if name else None)
            # Guard against model confusing interact_with_ui with interact_with_browser on web elements
            if clean_name == "interact_with_ui" and ("element_id" in args or args.get("action") in ("click_element", "fill_element")):
                clean_name = "interact_with_browser"
            if effective_known and clean_name not in effective_known:
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
                            args = _extract_raw_args(fn_obj)
                            res = _format_res(name, args)
                            if res:
                                return res
                    elif isinstance(obj, dict):
                        fn_obj = obj.get("function") if isinstance(obj.get("function"), dict) else obj
                        name = fn_obj.get("name") or fn_obj.get("tool")
                        args = _extract_raw_args(fn_obj)
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
                        args = _extract_raw_args(fn_obj)
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
                    args = _extract_raw_args(fn_obj)
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
                    args = _extract_raw_args(extracted)
                    res = _format_res(name, args)
                    if res:
                        return res
            except Exception:
                pass

        return None

    @classmethod
    def format_tool_observation(cls, tool_name: str, tool_result: Any) -> str:
        """Format tool execution observations with explicit status metadata and remediation hints."""
        # 1. Unpack ToolResult or object with to_dict
        if hasattr(tool_result, "to_dict"):
            try:
                tool_result = tool_result.to_dict(tool=tool_name)
            except TypeError:
                tool_result = tool_result.to_dict()

        parsed: Optional[Dict[str, Any]] = None
        raw_str = ""

        if isinstance(tool_result, dict):
            parsed = tool_result
            raw_str = json.dumps(tool_result)
        elif isinstance(tool_result, str):
            raw_str = tool_result.strip()
            try:
                data = json.loads(raw_str)
                if isinstance(data, dict):
                    parsed = data
            except Exception:
                pass
        else:
            raw_str = str(tool_result)

        exit_code = 0
        status = "success"
        output = ""
        error = ""
        remedy_hint = ""
        allowed_actions: List[str] = []

        if parsed is not None:
            if "allowed_actions" in parsed and isinstance(parsed["allowed_actions"], list):
                allowed_actions = list(parsed["allowed_actions"])
            elif isinstance(parsed.get("result"), dict) and "allowed_actions" in parsed["result"]:
                allowed_actions = list(parsed["result"]["allowed_actions"] or [])
            elif isinstance(parsed.get("result"), dict) and isinstance(parsed["result"].get("metadata"), dict) and "allowed_actions" in parsed["result"]["metadata"]:
                allowed_actions = list(parsed["result"]["metadata"]["allowed_actions"] or [])

            # Check success / error flags
            is_error = False
            if parsed.get("success") is False:
                is_error = True
            elif parsed.get("status") in ("error", "failed", "tool_execution_failed"):
                is_error = True
            elif parsed.get("error"):
                is_error = True
            elif isinstance(parsed.get("result"), dict) and parsed["result"].get("error"):
                is_error = True

            # Extract exit_code
            if "exit_code" in parsed:
                try:
                    exit_code = int(parsed["exit_code"])
                except Exception:
                    exit_code = 1 if is_error else 0
            elif "returncode" in parsed:
                try:
                    exit_code = int(parsed["returncode"])
                except Exception:
                    exit_code = 1 if is_error else 0
            else:
                exit_code = 1 if is_error else 0

            status = "error" if (is_error or exit_code != 0) else "success"

            # Extract output
            if "output" in parsed:
                out_val = parsed["output"]
            elif "stdout" in parsed:
                out_val = parsed["stdout"]
            elif "result" in parsed:
                res_val = parsed["result"]
                if isinstance(res_val, dict) and "output" in res_val:
                    out_val = res_val["output"]
                elif isinstance(res_val, dict) and "error" in res_val and is_error:
                    out_val = ""
                else:
                    out_val = res_val
            else:
                out_val = raw_str if status == "success" else ""

            output = json.dumps(out_val) if isinstance(out_val, dict) else str(out_val if out_val is not None else "")

            # Extract error
            if is_error or exit_code != 0:
                err_val = (
                    parsed.get("error")
                    or parsed.get("stderr")
                    or (parsed.get("result", {}).get("error") if isinstance(parsed.get("result"), dict) else None)
                    or "Tool execution failed"
                )
                error = str(err_val)
                remedy_hint = (
                    parsed.get("remedy_hint")
                    or (parsed.get("result", {}).get("remedy_hint") if isinstance(parsed.get("result"), dict) else None)
                    or (parsed.get("result", {}).get("metadata", {}).get("remedy_hint") if isinstance(parsed.get("result"), dict) else None)
                    or ""
                )
        else:
            # Parse raw text (e.g. "Exit code 0\n...", "Exit code 1\n...", "Error executing command: ...")
            exit_code_match = re.match(r"^Exit code\s+(\d+)(?:\n([\s\S]*))?$", raw_str, re.IGNORECASE)
            if exit_code_match:
                exit_code = int(exit_code_match.group(1))
                body = (exit_code_match.group(2) or "").strip()
                if exit_code == 0:
                    status = "success"
                    output = body
                    error = ""
                else:
                    status = "error"
                    output = ""
                    error = body or f"Command exited with code {exit_code}"
            elif raw_str.startswith(("Error executing command:", "Tool execution failed:", "Error:")):
                status = "error"
                exit_code = 1
                output = ""
                error = raw_str
            else:
                status = "success"
                exit_code = 0
                output = raw_str
                error = ""

        # Remediation hint handling on failure
        if status == "error" or exit_code != 0:
            from zelyr.engine.error_classifier import classify_tool_error

            classification = classify_tool_error(tool_name, exit_code, output, error)
            error_type = classification.error_type
            if not remedy_hint:
                remedy_hint = classification.remedy_hint
            if not allowed_actions:
                allowed_actions = classification.allowed_actions

            err_clean = error.rstrip(".")
            if "[Tool Error]:" not in err_clean:
                err_clean = f"[Tool Error]: {err_clean}" if err_clean else "[Tool Error]: Command failed"
            if "Recovery Hint:" not in err_clean:
                err_clean = f"{err_clean}. Recovery Hint: {remedy_hint}"
            error = err_clean

            structured = {
                "tool": tool_name,
                "status": "error",
                "error_type": error_type,
                "exit_code": exit_code,
                "output": output,
                "error": error,
                "remedy_hint": remedy_hint,
                "allowed_actions": allowed_actions,
            }
        else:
            structured = {
                "tool": tool_name,
                "status": "success",
                "exit_code": exit_code,
                "output": output,
                "error": error,
            }

        return json.dumps(structured, indent=2)

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
            async with httpx.AsyncClient(timeout=15.0) as client:
                await client.post(
                    f"{base_url}/api/generate",
                    json={
                        "model": payload["model"],
                        "options": {"num_ctx": prewarm_ctx},
                        "keep_alive": "5m",
                    },
                )
        except Exception:
            pass

        # OpenAI-compatible /v1/chat/completions requires assistant tool_calls.function.arguments
        # to be a JSON-encoded string, whereas in-memory payload["messages"] keeps native dicts.
        wire_payload = dict(payload)
        wire_payload["stream"] = True
        if "messages" in wire_payload and isinstance(wire_payload["messages"], list):
            wire_msgs = []
            for m in wire_payload["messages"]:
                if isinstance(m, dict) and m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list):
                    m_copy = dict(m)
                    if isinstance(m_copy.get("content"), str) and "<think>" in m_copy["content"]:
                        m_copy["content"] = re.sub(
                            r"<think>.*?(?:</think>|$)", "", m_copy["content"], flags=re.DOTALL
                        ).strip()
                    tc_copies = []
                    for tc in m["tool_calls"]:
                        if isinstance(tc, dict) and isinstance(tc.get("function"), dict):
                            tc_copy = dict(tc)
                            fn_copy = dict(tc["function"])
                            fn_args = fn_copy.get("arguments", {})
                            if not isinstance(fn_args, str):
                                fn_copy["arguments"] = json.dumps(fn_args)
                            tc_copy["function"] = fn_copy
                            tc_copies.append(tc_copy)
                        else:
                            tc_copies.append(tc)
                    m_copy["tool_calls"] = tc_copies
                    wire_msgs.append(m_copy)
                else:
                    wire_msgs.append(m)
            wire_payload["messages"] = wire_msgs

        timeout_config = httpx.Timeout(connect=10.0, read=300.0, write=20.0, pool=20.0)
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            async with client.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                json=wire_payload
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

    @staticmethod
    def _canonicalize_tool_args(args: Any) -> str:
        """Canonicalize tool arguments into a deterministic JSON string.

        Handles JSON strings, dicts, key ordering, and int-equivalent floats (1.0 -> 1).
        """
        if args is None:
            return "{}"

        raw = args
        if isinstance(raw, str):
            trimmed = raw.strip()
            if not trimmed:
                return "{}"
            if trimmed.startswith(("{", "[")):
                try:
                    raw = json.loads(trimmed)
                except Exception:
                    pass

        def _normalize(obj: Any) -> Any:
            if isinstance(obj, bool):
                return obj
            if isinstance(obj, float) and obj.is_integer():
                return int(obj)
            if isinstance(obj, dict):
                return {str(k): _normalize(v) for k, v in sorted(obj.items(), key=lambda item: str(item[0]))}
            if isinstance(obj, list):
                return [_normalize(item) for item in obj]
            return obj

        try:
            norm = _normalize(raw)
            return json.dumps(norm, sort_keys=True, separators=(",", ":"))
        except Exception:
            return "{}" if not raw else str(raw)

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
        turn_tool_history: Optional[List[tuple]] = None,
        actual_step_count: int = 0,
        disable_tools: bool = False,
        enable_cycle_detection: Optional[bool] = None,
    ) -> AsyncGenerator[dict, None]:
        eff_max_depth = max_depth if max_depth is not None else getattr(self, "max_depth", self.max_steps)
        eff_stream_provider = stream_provider or getattr(self, "stream_provider", None) or getattr(self, "model_provider", None)
        eff_tool_schemas = tool_schemas if tool_schemas is not None else getattr(self, "tool_schemas", None)
        eff_tool_executor = tool_executor or getattr(self, "tool_executor", None)
        eff_db_path = db_path or getattr(self, "db_path", None)
        curr_tool_history = turn_tool_history if turn_tool_history is not None else []
        eff_cycle_detection = self.enable_cycle_detection if enable_cycle_detection is None else enable_cycle_detection

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
        recovery_error_type = ""

        messages = payload.get("messages", [])
        if messages:
            for m in reversed(messages):
                if m.get("role") == "user" and "[ACTUATOR RECOVERY REQUIRED]" in str(m.get("content", "")):
                    continue
                if m.get("role") == "tool":
                    c = str(m.get("content", ""))
                    m_status = m.get("status")
                    m_remedy = m.get("remedy_hint")
                    m_actions = m.get("allowed_actions")
                    m_error_type = m.get("error_type")

                    parsed_c = None
                    try:
                        c_stripped = c.strip()
                        if c_stripped.startswith("{") and c_stripped.endswith("}"):
                            parsed_c = json.loads(c_stripped)
                    except Exception:
                        pass

                    if isinstance(parsed_c, dict):
                        if parsed_c.get("status") == "error" or parsed_c.get("exit_code", 0) != 0:
                            is_recovery_turn = True
                        if parsed_c.get("remedy_hint"):
                            is_recovery_turn = True
                            recovery_hint = str(parsed_c.get("remedy_hint") or "")
                        if parsed_c.get("allowed_actions"):
                            allowed_recovery_actions = list(parsed_c.get("allowed_actions") or [])
                        if parsed_c.get("error_type"):
                            recovery_error_type = str(parsed_c.get("error_type") or "")

                    if m_status == "error":
                        is_recovery_turn = True
                    if m_remedy:
                        is_recovery_turn = True
                        recovery_hint = str(m_remedy)
                    if m_actions:
                        allowed_recovery_actions = list(m_actions)
                    if m_error_type:
                        recovery_error_type = str(m_error_type)

                    if "[Tool Error]:" in c:
                        is_recovery_turn = True
                        if not recovery_hint and "Recovery Hint:" in c:
                            recovery_hint = c.split("Recovery Hint:", 1)[1].strip()

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
                    from zelyr.engine.memory_hydration import MemoryHydrator
                    clean_search = re.sub(r"\[Active Window:.*?\]\n?", "", user_prompt).strip()
                    hydrator = MemoryHydrator(db_path=eff_db_path, max_tokens=350)
                    recalled_mems = hydrator.get_relevant_memories(clean_search)
                    mem_block = hydrator.format_recalled_memory_block(recalled_mems)
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
            elif system_prompt_content not in payload["messages"][0].get("content", ""):
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

        if disable_tools:
            payload["tools"] = []
        elif eff_tool_schemas is not None:
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

        # Enforce adequate context size in Ollama (allow Ollama to auto-calculate GPU/CPU layer split)
        options = payload.setdefault("options", {})
        from zelyr.config import get_config
        eff_ctx = get_config().get_effective_num_ctx(
            options.get("num_ctx") or self._explicit_max_context_tokens
        )
        options["num_ctx"] = eff_ctx

        pending_tool_calls = []
        is_reasoning = False
        collected_assistant_text = []
        buffered_chunks = []
        buffering = True
        refusal_detected = False

        try:
            async for data in self._stream_model_events(payload, base_url, stream_provider=eff_stream_provider):
                if "_api_error" in data:
                    yield {"choices": [{"delta": {"content": f"\n⚠️ [API Error] {data['_api_error']}\n"}}]}
                    return

                delta = data.get("choices", [{}])[0].get("delta", {})

                if "reasoning_content" in delta or "reasoning" in delta:
                    r_content = delta.pop("reasoning_content", None) or delta.pop("reasoning", "")
                    if not is_reasoning:
                        delta["content"] = "<think>" + r_content
                        is_reasoning = True
                    else:
                        delta["content"] = r_content
                elif "content" in delta and delta["content"] and is_reasoning:
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
                                raw_fn_name = tc["function"]["name"]
                                pending_tool_calls[idx]["raw_name"] = raw_fn_name
                                pending_tool_calls[idx]["function"]["name"] = self.normalize_tool_name(
                                    raw_fn_name, active_schemas=eff_tool_schemas
                                )
                            if "arguments" in tc["function"] and tc["function"]["arguments"] is not None:
                                incoming_args = tc["function"]["arguments"]
                                if isinstance(incoming_args, dict):
                                    pending_tool_calls[idx]["function"]["arguments"] = json.dumps(incoming_args)
                                else:
                                    pending_tool_calls[idx]["function"]["arguments"] += str(incoming_args)
                        elif "name" in tc and tc["name"]:
                            raw_fn_name = tc["name"]
                            pending_tool_calls[idx]["raw_name"] = raw_fn_name
                            pending_tool_calls[idx]["function"]["name"] = self.normalize_tool_name(
                                raw_fn_name, active_schemas=eff_tool_schemas
                            )
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
                        clean_text = re.sub(r"<think>.*$", "", clean_text, flags=re.DOTALL)
                        stripped = clean_text.lstrip()
                        if is_interface_action and self.is_refusal(clean_text):
                            refusal_detected = True
                            break
                        elif (
                            stripped
                            and not stripped.startswith(("{", "[", "```", "<"))
                            and "```" not in stripped
                            and '{"name"' not in stripped
                            and '{"type"' not in stripped
                            and "<tool_call>" not in stripped
                        ):
                            if not is_recovery_turn and len(clean_text) > 250:
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

        if is_reasoning:
            collected_assistant_text.append("</think>")
            if buffering:
                buffered_chunks.append({"choices": [{"delta": {"content": "</think>"}}]})
            else:
                yield {"choices": [{"delta": {"content": "</think>"}}]}
            is_reasoning = False

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
            non_think_text = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL)
            non_think_text = re.sub(r"<think>.*$", "", non_think_text, flags=re.DOTALL).strip()
            known_tools = self.get_known_tool_names(payload.get("tools"))

            tool_call = (
                self.extract_first_tool_call(
                    non_think_text, known_tools, active_schemas=payload.get("tools")
                )
                if non_think_text
                else None
            )
            if tool_call:
                raw_name = tool_call.get("function", {}).get("name") or tool_call.get("name")
                func_name = self.normalize_tool_name(
                    raw_name, active_schemas=payload.get("tools")
                )
                func_args = tool_call.get("function", {}).get("arguments") or tool_call.get("arguments", {})
                func_args = self.normalize_tool_arguments(func_name, func_args, original_name=raw_name)
                func_args_str = json.dumps(func_args)
                call_id = "call_native_0"
                pending_tool_calls = [{
                    "id": call_id,
                    "type": "function",
                    "raw_name": raw_name,
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
        # an actuator tool call, and depth < min(4, eff_max_depth - 1), do not terminate the loop.
        # Reject conversational prose, clear buffers, and trigger a fast retry directing
        # the model to invoke the recovery tool immediately.
        if is_recovery_turn and not pending_tool_calls and depth < min(4, eff_max_depth - 1):
            buffered_chunks.clear()
            collected_assistant_text.clear()

            err_type_str = f" ({recovery_error_type})" if recovery_error_type else ""
            tools_list = ", ".join(allowed_recovery_actions) if allowed_recovery_actions else "execute_command, manage_workspace_file, verify_outcome"
            allowed_clause = f" Allowed tools: {tools_list}."
            hint_clause = f" Recovery hint: '{recovery_hint}'." if recovery_hint else ""

            override_msg = {
                "role": "user",
                "content": (
                    f"[ACTUATOR RECOVERY REQUIRED] Your previous tool action failed{err_type_str}. "
                    f"You MUST attempt a corrective tool call to diagnose or resolve the issue.{allowed_clause}{hint_clause} "
                    f"Do not output conversational apologies without calling a tool."
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
                raw_fn_name = tc.get("raw_name") or tc.get("function", {}).get("name") or tc.get("name") or ""
                fn_name = self.normalize_tool_name(raw_fn_name, active_schemas=eff_tool_schemas)
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

                fn_args = self.normalize_tool_arguments(fn_name, fn_args, original_name=raw_fn_name)

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

            force_disable_next = False
            for tc in normalized_tool_calls:
                func_name = tc["function"]["name"]
                func_args = tc["function"]["arguments"]
                tool_call_id = tc["id"]

                # Fingerprint tool invocation: canonical JSON args
                canonical_args = self._canonicalize_tool_args(func_args)
                fingerprint = (func_name, canonical_args)

                # ReAct step limit and cycle detection rules
                rejection_reason = None
                rejection_detail = None

                eff_step_limit = getattr(self, "max_steps", 5)
                if actual_step_count >= eff_step_limit:
                    rejection_reason = "step_limit_reached"
                    rejection_detail = f"Maximum execution step limit ({eff_step_limit}) reached for current turn. Explain the limitation and synthesize the answer."
                    force_disable_next = True
                elif eff_cycle_detection:
                    if curr_tool_history and curr_tool_history[-1][0] == fingerprint and curr_tool_history[-1][1] == "error":
                        # Identical Retry After Error: immediate preceding had same fingerprint and ended in error
                        rejection_reason = "cycle_detected"
                        rejection_detail = "Identical tool call retry after immediate failure is blocked. Alter parameters or explain the limitation."
                    elif sum(1 for prior_fp, _ in curr_tool_history if prior_fp == fingerprint) >= 2:
                        # Identical Call Ceiling: attempted for the 3rd time
                        rejection_reason = "cycle_detected"
                        rejection_detail = "Identical tool call ceiling reached (attempted 3 or more times). Alter parameters or explain the limitation."
                    elif len(curr_tool_history) >= 3:
                        # Alternating Failure Loop: A -> B -> A -> B where both A and B returned error
                        A_prev = curr_tool_history[-1][0]
                        B_prev = curr_tool_history[-2][0]
                        A_init = curr_tool_history[-3][0]
                        if (
                            fingerprint == B_prev
                            and A_prev == A_init
                            and A_prev != B_prev
                            and curr_tool_history[-1][1] == "error"
                            and curr_tool_history[-2][1] == "error"
                            and curr_tool_history[-3][1] == "error"
                        ):
                            rejection_reason = "cycle_detected"
                            rejection_detail = "Alternating failure cycle detected (A -> B -> A -> B). Alter approach or explain limitation."

                if rejection_reason is not None:
                    actual_step_count += 1
                    if actual_step_count >= eff_step_limit:
                        force_disable_next = True
                    synth_call_id = tool_call_id or f"call_synth_{uuid.uuid4().hex[:8]}"
                    synth_content = json.dumps({
                        "tool": func_name,
                        "status": "rejected",
                        "reason": rejection_reason,
                        "detail": rejection_detail,
                    })
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": synth_call_id,
                        "name": func_name,
                        "content": synth_content,
                    }
                    payload["messages"].append(tool_msg)
                    curr_tool_history.append((fingerprint, "rejected"))
                    continue

                actual_step_count += 1
                if actual_step_count >= eff_step_limit:
                    force_disable_next = True
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

                # Format tool output with structured status metadata and remediation hints
                content_for_msg = self.format_tool_observation(func_name, res_str)

                is_failed = False
                remedy_hint_out = ""
                allowed_actions_out = []
                error_type_out = ""

                try:
                    obs_dict = json.loads(content_for_msg) if isinstance(content_for_msg, str) else content_for_msg
                    if isinstance(obs_dict, dict):
                        if obs_dict.get("status") == "error" or obs_dict.get("exit_code", 0) != 0:
                            is_failed = True
                        if obs_dict.get("remedy_hint"):
                            remedy_hint_out = str(obs_dict.get("remedy_hint") or "")
                        elif "Recovery Hint:" in str(obs_dict.get("error") or ""):
                            remedy_hint_out = str(obs_dict.get("error") or "").split("Recovery Hint:", 1)[1].strip()
                        elif is_failed:
                            remedy_hint_out = "Command failed with non-zero exit code. Inspect stderr and adjust arguments."

                        if obs_dict.get("allowed_actions"):
                            allowed_actions_out = list(obs_dict.get("allowed_actions") or [])
                        if obs_dict.get("error_type"):
                            error_type_out = str(obs_dict.get("error_type") or "")
                except Exception:
                    pass

                try:
                    if not allowed_actions_out:
                        parsed_raw = json.loads(res_str) if isinstance(res_str, str) else res_str
                        if isinstance(parsed_raw, dict):
                            allowed_actions_out = (
                                parsed_raw.get("allowed_actions")
                                or (parsed_raw.get("result", {}).get("allowed_actions") if isinstance(parsed_raw.get("result"), dict) else None)
                                or []
                            )
                except Exception:
                    pass

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": func_name,
                    "content": content_for_msg,
                    "status": "error" if is_failed else "success",
                }
                if is_failed:
                    if remedy_hint_out:
                        tool_msg["remedy_hint"] = remedy_hint_out
                    if allowed_actions_out:
                        tool_msg["allowed_actions"] = allowed_actions_out
                    if error_type_out:
                        tool_msg["error_type"] = error_type_out
                payload["messages"].append(tool_msg)
                curr_tool_history.append((fingerprint, "error" if is_failed else "success"))

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
                    turn_tool_history=curr_tool_history,
                    actual_step_count=actual_step_count,
                    disable_tools=force_disable_next,
                    enable_cycle_detection=eff_cycle_detection,
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

