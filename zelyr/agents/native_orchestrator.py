import asyncio
import inspect
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
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
        config: Optional[Any] = None,
        tool_registry: Optional[Any] = None,
        caller_ring: Optional[int] = None,
        allowed_tools: Optional[List[str]] = None,
        task_local_scope: Optional[Any] = None,
        central_ledger: Optional[Any] = None,
    ):
        self._explicit_max_context_tokens = max_context_tokens
        self._max_steps = max_depth if max_depth is not None else max_steps
        self._max_depth = self._max_steps
        self.enable_cycle_detection = enable_cycle_detection
        self.stream_provider = stream_provider or model_provider
        self.model_provider = self.stream_provider
        self.tool_schemas = tool_schemas
        self.tool_executor = tool_executor
        self._execute_tool = execute_tool or tool_executor
        self.db_path = db_path
        self.config = config
        self.tool_registry = tool_registry
        self.caller_ring = caller_ring
        self.allowed_tools = allowed_tools
        self.task_local_scope = task_local_scope
        self.central_ledger = central_ledger
        self.current_ledger: Optional[Any] = None
        self.last_finish_reason: Optional[str] = None

        from zelyr.engine.budget_manager import ContextBudgetEngine
        self.budget_engine = ContextBudgetEngine(
            default_max_tokens=self._explicit_max_context_tokens or 8192,
        )

    @property
    def max_steps(self) -> int:
        return getattr(self, "_max_steps", 5)

    @max_steps.setter
    def max_steps(self, value: int) -> None:
        self._max_steps = value
        # Ensure raising max_steps also expands max_depth (e.g. CLI/runners setting orch.max_steps = 8),
        # while lowering max_steps below default keeps depth headroom for rejection/synthesis turns.
        self._max_depth = max(getattr(self, "_max_depth", 5), value)

    @property
    def max_depth(self) -> int:
        return getattr(self, "_max_depth", getattr(self, "_max_steps", 5))

    @max_depth.setter
    def max_depth(self, value: int) -> None:
        self._max_depth = value

    def _active_config(self) -> Any:
        """Return session-scoped config if attached, otherwise global config."""
        cfg = getattr(self, "config", None)
        if cfg is not None:
            return cfg
        from zelyr.config import get_config
        return get_config()

    @property
    def max_context_tokens(self) -> int:
        return self._active_config().get_effective_num_ctx(self._explicit_max_context_tokens)

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

    def get_runtime_manifest(
        self,
        *,
        refresh: bool = False,
        probe_services: bool = False,
        probe_timeout: float = 0.5,
        caller_ring: Optional[int] = None,
        allowed_tools: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Return the structured runtime environment manifest for this orchestrator context."""
        from zelyr.engine.environment import RuntimeEnvironmentManifest

        cfg = getattr(self, "config", None)
        if cfg is None:
            try:
                from zelyr.config import get_config

                cfg = get_config()
            except Exception:
                cfg = None
        registry = getattr(self, "tool_registry", None)
        eff_ring = caller_ring if caller_ring is not None else getattr(self, "caller_ring", None)
        eff_allowed = allowed_tools if allowed_tools is not None else getattr(self, "allowed_tools", None)
        return RuntimeEnvironmentManifest.collect_manifest(
            config=cfg,
            tool_registry=registry,
            caller_ring=eff_ring,
            allowed_tools=eff_allowed,
            probe_services=probe_services,
            probe_timeout=probe_timeout,
            refresh=refresh,
        )

    def refresh_runtime_manifest(
        self,
        *,
        probe_services: bool = False,
        probe_timeout: float = 0.5,
    ) -> Dict[str, Any]:
        """Invalidate cached environment state and collect a fresh runtime manifest."""
        return self.get_runtime_manifest(
            refresh=True,
            probe_services=probe_services,
            probe_timeout=probe_timeout,
        )

    def get_planning_environment_context(
        self,
        *,
        caller_ring: Optional[int] = None,
        allowed_tools: Optional[List[str]] = None,
    ) -> str:
        """Return the compact (<120 token) planning environment manifest block."""
        from zelyr.engine.environment import RuntimeEnvironmentManifest

        cfg = getattr(self, "config", None)
        if cfg is None:
            try:
                from zelyr.config import get_config

                cfg = get_config()
            except Exception:
                cfg = None
        registry = getattr(self, "tool_registry", None)
        eff_ring = caller_ring if caller_ring is not None else getattr(self, "caller_ring", None)
        eff_allowed = allowed_tools if allowed_tools is not None else getattr(self, "allowed_tools", None)
        return RuntimeEnvironmentManifest.build_manifest_block(
            config=cfg,
            tool_registry=registry,
            caller_ring=eff_ring,
            allowed_tools=eff_allowed,
        )

    def _build_system_prompt(
        self,
        base_directives: Optional[str] = None,
        caller_ring: Optional[int] = None,
        allowed_tools: Optional[List[str]] = None,
    ) -> str:
        """Compile hierarchical system directives, global user instructions, and local project rules."""
        from zelyr.core.instructions import InstructionManager
        manager = InstructionManager()
        cfg = getattr(self, "config", None)
        if cfg is None:
            try:
                from zelyr.config import get_config
                cfg = get_config()
            except Exception:
                cfg = None
        registry = getattr(self, "tool_registry", None)
        eff_ring = caller_ring if caller_ring is not None else getattr(self, "caller_ring", None)
        eff_allowed = allowed_tools if allowed_tools is not None else getattr(self, "allowed_tools", None)
        return manager.get_compiled_instructions(
            base_directives=base_directives,
            config=cfg,
            tool_registry=registry,
            caller_ring=eff_ring,
            allowed_tools=eff_allowed,
        )

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
        tools: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Any] = None,
    ) -> list[dict]:
        """Adaptive context window budgeting: applies Stage 1 tool compaction and Stage 2
        sliding-window turn pruning with semantic summarization.
        """
        eff_max = max_context_tokens or self.max_context_tokens
        eff_cfg = config if config is not None else self._active_config()
        return self.budget_engine.apply_context_budget(
            messages,
            model_name=model_name,
            max_context_tokens=eff_max,
            reserve_tokens=reserve_tokens,
            tools=tools,
            config=eff_cfg,
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
    def normalize_tool_name(
        cls,
        name: Any,
        active_schemas: Optional[list] = None,
        task_local_scope: Optional[Any] = None,
    ) -> str:
        """Normalize a tool name by stripping common prefixes and mapping hallucinated aliases to canonical tools."""
        clean = cls._strip_tool_prefixes(name)
        if not clean:
            return ""
        from zelyr.tool_registry import TaskLocalToolScope

        if isinstance(task_local_scope, TaskLocalToolScope):
            scope_names = task_local_scope.get_known_tools()
            if clean in scope_names:
                return clean
            target = cls.TOOL_ALIAS_MAP.get(clean)
            if target and target in scope_names:
                return target
            return clean
        if isinstance(active_schemas, (list, tuple)):
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
        if target and target in cls.get_known_tool_names(
            active_schemas if isinstance(active_schemas, (list, tuple)) else None
        ):
            return target
        return clean

    @classmethod
    def normalize_tool_arguments(
        cls,
        clean_name: str,
        args: Any,
        original_name: Optional[str] = None,
        active_schemas: Optional[list] = None,
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

        # Apply SafeArgumentNormalizer for deterministic type coercions
        from zelyr.engine.normalizer import SafeArgumentNormalizer

        target_schema = cls._find_tool_schema(clean_name, active_schemas=active_schemas)
        if target_schema:
            args, _ = SafeArgumentNormalizer.normalize(target_schema, args, tool_name=clean_name)

        return args

    def cancel(self, reason: str = "task_cancelled") -> None:
        """Cancel active orchestration and discard any pending proposals."""
        self._cancelled = True
        if hasattr(self, "current_ledger") and self.current_ledger is not None:
            self.current_ledger.discard_pending_proposals(reason=reason)

    @classmethod
    def _find_tool_schema(cls, tool_name: str, active_schemas: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
        """Find tool schema definition in active_schemas or ToolRegistry."""
        clean_name = cls.normalize_tool_name(tool_name, active_schemas=active_schemas)
        if active_schemas:
            for s in active_schemas:
                if isinstance(s, dict):
                    fn = s.get("function")
                    s_name = fn.get("name") if isinstance(fn, dict) else s.get("name")
                    if s_name in (clean_name, tool_name):
                        return s
        try:
            from zelyr.tool_registry import ToolRegistry
            reg = ToolRegistry()
            t = reg.get_tool(clean_name) or reg.get_tool(tool_name)
            if t:
                return getattr(t, "schema", None) or getattr(t, "tool_schema", None)
        except Exception:
            pass
        return None

    @classmethod
    def _is_intentional_abandonment(cls, text: str) -> bool:
        """Detect if model text explicitly and intentionally abandons pending actions."""
        if not text or not isinstance(text, str):
            return False
        clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        clean = re.sub(r"<think>.*$", "", clean, flags=re.DOTALL).strip()
        if not clean:
            return False
        if cls.is_refusal(clean):
            return True
        abandon_patterns = [
            r"\b(?:stop|stopping|cancel|cancelling|cancelled|abort|aborting|aborted|abandon|abandoning)\b",
            r"\bwill not proceed\b",
            r"\bdecided not to\b",
            r"\bcannot proceed\b",
            r"\bdeclining to\b",
            r"\bno further actions?\b",
            r"\bstopping here\b",
            r"\btask aborted\b",
        ]
        # Scoped file/step completion (e.g. "no further actions required for this file, I will proceed to the next task")
        # must not be misread as task-wide abandonment.
        has_continuation_intent = bool(
            re.search(
                r"\b(?:proceed\s+to\s+(?:the\s+)?next|moving\s+on\s+to|i\s+will\s+(?:now\s+)?(?:create|write|append|modify|run|execute|proceed))\b",
                clean,
                re.IGNORECASE,
            )
        )
        has_scoped_file_no_actions = bool(
            re.search(
                r"\bno further actions?\s+(?:are\s+)?(?:required|needed)\s+for\s+(?:this|that|the)\s+(?:file|target|step)\b",
                clean,
                re.IGNORECASE,
            )
        )
        for pat in abandon_patterns:
            if pat == r"\bno further actions?\b" and (has_continuation_intent or has_scoped_file_no_actions):
                continue
            if re.search(pat, clean, re.IGNORECASE):
                return True
        return False

    @staticmethod
    def _extract_target_from_args(func_name: str, func_args: Dict[str, Any]) -> Optional[str]:
        """Extract target resource identifier (e.g. file path) from tool arguments."""
        if not isinstance(func_args, dict):
            return None
        for k in ("path", "filepath", "file", "target", "filename", "dst", "destination"):
            val = func_args.get(k)
            if val and isinstance(val, str) and val.strip():
                return val.strip()
        if (func_name or "").lower() in ("execute_command", "bash", "sh"):
            cmd = str(func_args.get("command", "") or func_args.get("cmd", "")).strip()
            if cmd:
                from zelyr.engine.execution_ledger import extract_file_targets_from_text
                extracted = extract_file_targets_from_text(cmd)
                if extracted:
                    return extracted[0]
        return None

    @classmethod
    def _is_explicit_repeat_requested(cls, prompt: str) -> bool:
        """Determine whether user prompt explicitly authorizes action repetition."""
        if not prompt or not isinstance(prompt, str):
            return False

        # 1. Strip negated clauses ("do not repeat", "never retry", "without repeating", etc.)
        clean = re.sub(
            r"\b(?:do\s+not|don'?t|never|without|should\s+not|shouldn'?t|must\s+not|cannot|avoid|stop|prevent)\s+[^.!?;\n]+",
            "",
            prompt,
            flags=re.IGNORECASE,
        )

        # 2. Filter out non-command idiomatic usages of 'again'
        clean = re.sub(
            r"\b(?:time\s+and\s+time\s+again|again\s+and\s+again|never\s+again|once\s+again|over\s+and\s+over\s+again|born\s+again|here\s+again)\b",
            " ",
            clean,
            flags=re.IGNORECASE,
        )

        # 3. Check for genuine imperative repeat directives
        has_repeat_directive = bool(re.search(
            r"\b(?:repeat|repeatedly|rerun|re-execute|retry)\b",
            clean,
            flags=re.IGNORECASE,
        )) or bool(re.search(
            r"\b(?:run|do|execute|write|append|perform|call)\s+(?:this|it|the\s+command|the\s+action)?\s*(?:again|twice|thrice|\d+\s+times|multiple\s+times|another\s+time|once\s+more|second\s+time)\b",
            clean,
            flags=re.IGNORECASE,
        )) or bool(re.search(
            r"\b(?:twice|thrice|\d+\s+times)\b",
            clean,
            flags=re.IGNORECASE,
        ))

        return has_repeat_directive

    @staticmethod
    def _extract_prohibited_targets(prompt: str, ledger: Any = None) -> List[str]:
        """Extract canonical file/directory targets the user explicitly forbade mutating."""
        from zelyr.engine.execution_ledger import extract_prohibited_targets
        return extract_prohibited_targets(prompt, ledger=ledger)

    @staticmethod
    def _replan_budget_available(
        ledger: Any,
        attempts: int,
        max_attempts: int,
        verified_outcomes: Optional[List[str]] = None,
        unfulfilled_outcomes: Optional[List[str]] = None,
    ) -> bool:
        """Bounded replanning; extra attempts (hard cap +3) only if the previous replan verified genuinely new task progress."""
        if max_attempts <= 0:
            return False
        if attempts < max_attempts:
            return True
        if attempts >= max_attempts + 3:
            return False
        try:
            current_targets = set(ledger.get_verified_targets())
            current_count = len(current_targets)
        except Exception:
            return False
        prev_marker = getattr(ledger, "_replan_verified_marker", current_count)
        if current_count <= prev_marker:
            return False

        # If task-specific outcome lists are provided, require genuine task progress:
        # at least one new task-verified outcome AND fewer unfulfilled task outcomes than before.
        if verified_outcomes is not None and unfulfilled_outcomes is not None:
            prev_v_outcomes = getattr(ledger, "_replan_verified_outcomes", None)
            prev_u_count = getattr(ledger, "_replan_unfulfilled_count", None)
            if prev_v_outcomes is not None:
                new_v = set(verified_outcomes) - set(prev_v_outcomes)
                if not new_v:
                    return False
            if prev_u_count is not None:
                if len(unfulfilled_outcomes) >= prev_u_count:
                    return False

        return True

    @classmethod
    def _is_sequential_step_requested(
        cls,
        prompt: str,
        target: str,
        action: str,
        ledger: Any,
    ) -> bool:
        """Determine whether a subsequent distinct modification on a target was explicitly requested."""
        if not prompt or not action:
            return False

        action_lower = action.lower()
        from zelyr.engine.execution_ledger import canonicalize_target, targets_match
        norm_target = canonicalize_target(target)

        # 1. Respect explicit prohibitions against subsequent modification
        prohibition_patterns = [
            r"\b(?:do\s+not|don'?t|never|without|avoid|refrain\s+from)\s+(?:modify|modifying|change|changing|edit|editing|update|updating|append|appending|alter|altering|touch|touching|rewrite|rewriting|overwrite|overwriting|patch|patching|replace|replacing)\b",
            r"\bdo\s+not\s+[^.!?;\n]+\s+afterward\b",
            r"\bwithout\s+modifying\b",
            r"\bleave\s+[^.!?;\n]+\s+(?:untouched|alone|as\s+is|unchanged|intact)\b",
        ]
        for pat in prohibition_patterns:
            if re.search(pat, prompt, flags=re.IGNORECASE):
                return False

        # 2. If the same action on this target has already succeeded, it's not a new sequential step
        if hasattr(ledger, "successful_actions"):
            for rec in ledger.successful_actions():
                if getattr(rec, "verification_failed", False):
                    continue
                rec_action = str(rec.arguments.get("action", "")).lower()
                rec_target = cls._extract_target_from_args(rec.tool_name, rec.arguments)
                if rec_action == action_lower and rec_target and targets_match(rec_target, norm_target):
                    return False

        # 3. Map action to allowed directive verbs
        action_verbs = {
            "append": r"(?:append|appending|add|adding|extend|extending|attach|attaching)",
            "modify": r"(?:modify|modifying|update|updating|edit|editing|patch|patching|change|changing)",
            "update": r"(?:update|updating|modify|modifying|edit|editing|patch|patching|change|changing)",
            "edit": r"(?:edit|editing|update|updating|modify|modifying|patch|patching)",
            "patch": r"(?:patch|patching|modify|modifying|update|updating)",
            "write": r"(?:write|writing|overwrite|overwriting|replace|replacing)",
            "delete": r"(?:delete|deleting|remove|removing|erase|erasing|unlink)",
            "execute_command": r"(?:run|execute|launch|start|invoke|append|appending|add|adding|write|writing|modify|modifying|update|updating|patch|patching)",
            "bash": r"(?:run|execute|launch|start|invoke|append|appending|add|adding|write|writing|modify|modifying|update|updating|patch|patching)",
            "sh": r"(?:run|execute|launch|start|invoke|append|appending|add|adding|write|writing|modify|modifying|update|updating|patch|patching)",
        }
        verbs_pat = action_verbs.get(action_lower, re.escape(action_lower))

        # 4. Check for explicit sequential instruction with transition markers
        pat_transition = (
            rf"\b(?:then|and\s+then|afterwards?|subsequently|followed\s+by|next|secondly|step\s*2|as\s+a\s+second\s+step|now)\s+"
            rf"(?:[a-zA-Z0-9_\-'\"]+\s+){{0,5}}{verbs_pat}\b"
        )
        if re.search(pat_transition, prompt, flags=re.IGNORECASE):
            neg_check = rf"\b(?:then|and\s+then|afterwards?|subsequently|next)\s+(?:do\s+not|don'?t|never|avoid|without)\s+{verbs_pat}\b"
            if not re.search(neg_check, prompt, flags=re.IGNORECASE):
                return True

        # 5. Check coordination conjunctions (e.g. 'create X and (then) update/append it')
        pat_coord = rf"\b(?:create|write|make)\s+[^.!?;\n]+\s+and\s+(?:then\s+)?{verbs_pat}\b"
        if re.search(pat_coord, prompt, flags=re.IGNORECASE):
            return True

        return False

    @classmethod
    def _get_mutation_targets(
        cls,
        func_name: str,
        func_args: Dict[str, Any],
        verified_targets: Optional[List[str]] = None,
        allow_dir_containment: bool = False,
    ) -> List[str]:
        """Identify which canonical targets (or verified/prohibited targets) are targeted for mutation by the tool call."""
        if not isinstance(func_args, dict):
            return []

        from zelyr.engine.execution_ledger import canonicalize_target, targets_match
        mutated: List[str] = []
        fname_lower = func_name.lower()

        def _match_or_canon(val_str: str) -> None:
            matched_any = False
            if verified_targets:
                for vt in verified_targets:
                    if targets_match(
                        val_str,
                        vt,
                        verified_targets,
                        allow_dir_containment=allow_dir_containment,
                    ):
                        if vt not in mutated:
                            mutated.append(vt)
                        matched_any = True
            if not matched_any:
                canon = canonicalize_target(val_str)
                if canon and canon not in mutated:
                    mutated.append(canon)

        if fname_lower == "manage_workspace_file":
            action = str(func_args.get("action", "read")).lower()
            if action in ("read", "view", "list", "list_tree", "stat", "exists", "inspect", "check"):
                return []

            if action in (
                "write", "overwrite", "create", "append", "patch", "edit", "modify",
                "update", "replace", "rollback", "delete", "remove", "unlink", "rm",
                "truncate", "move", "rename", "copy", "cp", "touch", "alter", "change",
            ):
                for k in ("path", "filepath", "file", "target", "filename", "src", "source"):
                    val = func_args.get(k)
                    if val and isinstance(val, str) and val.strip():
                        # For copy/cp, only destination is mutated, not source
                        if action in ("copy", "cp") and k in ("src", "source"):
                            continue
                        _match_or_canon(val.strip())

            if action in ("move", "rename", "copy", "cp"):
                for k in ("destination", "dst", "to", "new_path"):
                    val = func_args.get(k)
                    if val and isinstance(val, str) and val.strip():
                        _match_or_canon(val.strip())

        elif fname_lower in ("execute_command", "bash", "sh"):
            cmd = str(func_args.get("command", "") or func_args.get("cmd", "")).strip()
            if cmd:
                # Extract potential path arguments following mutating shell constructs
                mut_shell_pat = re.compile(
                    r"(?:>{1,2}\s*|(?:\b(?:tee|rm|unlink|truncate|mv|cp|touch|mkdir|rmdir)\b(?:\s+-[a-zA-Z0-9\-]+)*|\bsed\s+-[a-zA-Z]*i[a-zA-Z]*)\s+)([^;&|\n]+)",
                    flags=re.IGNORECASE,
                )
                candidate_tokens: List[str] = []
                for m in mut_shell_pat.finditer(cmd):
                    seg = m.group(1)
                    for tok in re.findall(r"[\"'`]?([~/a-zA-Z0-9_.\-]+(?:/[a-zA-Z0-9_.\-]+)*)[\"'`]?", seg):
                        tok_clean = tok.strip("\"'`.,:;)")
                        if tok_clean and not tok_clean.startswith("-") and ("/" in tok_clean or "." in tok_clean):
                            candidate_tokens.append(tok_clean)

                for tok in candidate_tokens:
                    _match_or_canon(tok)

        return mutated

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

    @staticmethod
    def _unique_ledger_call_id(
        ledger: Any,
        candidate_id: str,
        reserved_ids: Optional[set] = None,
    ) -> str:
        """Ensure a call_id is unique against recorded ledger entries and in-flight turn IDs."""
        if not candidate_id:
            candidate_id = f"call_{uuid.uuid4().hex[:8]}"

        def _is_taken(cid: str) -> bool:
            if reserved_ids is not None and cid in reserved_ids:
                return True
            if ledger is not None and hasattr(ledger, "get_record") and ledger.get_record(cid) is not None:
                return True
            return False

        if not _is_taken(candidate_id):
            return candidate_id
        suffix = 2
        while _is_taken(f"{candidate_id}_{suffix}"):
            suffix += 1
        return f"{candidate_id}_{suffix}"

    @classmethod
    def extract_first_tool_call(
        cls,
        text: str,
        known_tools: Optional[set] = None,
        active_schemas: Optional[list] = None,
        caller_ring: Optional[int] = None,
        allowed_tools: Optional[List[str]] = None,
    ) -> Optional[dict]:
        """Tolerantly extract the first valid JSON/XML tool call from model text.

        Resolves issues with models emitting concatenated JSON objects, markdown code fences,
        XML pseudo-tags embedded in prose, or reasoning text before tool calls.
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

        from zelyr.agents.fallback_parser import FallbackToolParser

        calls = FallbackToolParser.parse_tool_calls(
            text,
            known_tools=effective_known,
            active_schemas=active_schemas,
            max_calls=1,
            caller_ring=caller_ring,
            allowed_tools=allowed_tools,
        )
        if calls:
            c = calls[0]
            return {
                "type": "function",
                "function": {
                    "name": c.name,
                    "arguments": c.arguments,
                },
                "name": c.name,
                "arguments": c.arguments,
                "tool_call_id": c.tool_call_id,
                "cleaned_text": c.cleaned_text,
            }
        return None

    @classmethod
    def format_tool_observation(cls, tool_name: str, tool_result: Any, as_json: bool = False) -> str:
        """Format tool execution observations with deterministic semantic anchors and status metadata."""
        from zelyr.engine.observation_framing import ObservationFramingEngine

        return ObservationFramingEngine.frame_observation(tool_name, tool_result, as_json=as_json)

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
        active_cfg = self._active_config()
        prewarm_ctx = active_cfg.get_effective_num_ctx(
            options.get("num_ctx") or self._explicit_max_context_tokens
        )
        eff_keep_alive = (
            active_cfg.get_effective_keep_alive(payload.get("keep_alive"))
            if hasattr(active_cfg, "get_effective_keep_alive")
            else (payload.get("keep_alive") if payload.get("keep_alive") is not None else "5m")
        )
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                await client.post(
                    f"{base_url}/api/generate",
                    json={
                        "model": payload["model"],
                        "options": {"num_ctx": prewarm_ctx},
                        "keep_alive": eff_keep_alive,
                    },
                )
        except Exception:
            pass

        # Native Ollama /api/chat honors options.num_ctx natively (whereas /v1/chat/completions
        # silently ignores options.num_ctx and falls back to 4096) and requires assistant
        # tool_calls.function.arguments to be native JSON dicts rather than JSON strings.
        wire_payload = dict(payload)
        wire_payload["stream"] = True
        wire_payload.setdefault("keep_alive", eff_keep_alive)
        if "temperature" in wire_payload and wire_payload["temperature"] is not None:
            opts_copy = dict(wire_payload.get("options") or {})
            opts_copy.setdefault("temperature", wire_payload["temperature"])
            wire_payload["options"] = opts_copy
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
                            if isinstance(fn_args, str):
                                try:
                                    parsed_args = json.loads(fn_args)
                                    fn_copy["arguments"] = parsed_args if isinstance(parsed_args, dict) else {}
                                except json.JSONDecodeError:
                                    fn_copy["arguments"] = {}
                            elif not isinstance(fn_args, dict):
                                fn_copy["arguments"] = {}
                            tc_copy["function"] = fn_copy
                            tc_copies.append(tc_copy)
                        else:
                            tc_copies.append(tc)
                    m_copy["tool_calls"] = tc_copies
                    wire_msgs.append(m_copy)
                else:
                    wire_msgs.append(m)
            wire_payload["messages"] = wire_msgs

        read_timeout = (
            active_cfg.get_effective_read_timeout(payload.get("read_timeout"))
            if hasattr(active_cfg, "get_effective_read_timeout")
            else 300.0
        )
        timeout_config = httpx.Timeout(connect=10.0, read=read_timeout, write=20.0, pool=20.0)
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            async with client.stream(
                "POST",
                f"{base_url}/api/chat",
                json=wire_payload
            ) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    yield {"_api_error": f"HTTP {resp.status_code}: {resp.text}"}
                    return

                async for line in resp.aiter_lines():
                    is_done, events = self._parse_stream_line_to_events(line)
                    for evt in events:
                        yield evt
                    if is_done:
                        break

    @staticmethod
    def _normalize_done_reason(chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Propagate Ollama/OpenAI done_reason onto choices[0].finish_reason."""
        done_r = chunk.get("done_reason") or chunk.get("finish_reason")
        if done_r and isinstance(chunk.get("choices"), list) and chunk["choices"]:
            c0 = chunk["choices"][0]
            if isinstance(c0, dict) and not c0.get("finish_reason"):
                c0["finish_reason"] = done_r
        return chunk

    @classmethod
    def _parse_stream_line_to_events(cls, line: str) -> tuple[bool, List[Dict[str, Any]]]:
        """Decode either OpenAI SSE lines or native Ollama /api/chat NDJSON lines."""
        if not line:
            return False, []
        if line.startswith("data: "):
            data_str = line[6:]
            if data_str == "[DONE]":
                return True, []
            try:
                parsed = json.loads(data_str)
            except json.JSONDecodeError:
                return False, []
            if isinstance(parsed, dict):
                return False, [cls._normalize_done_reason(parsed)]
            return False, [parsed]

        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            return False, []
        if not isinstance(parsed, dict):
            return False, []
        if "error" in parsed and "choices" not in parsed:
            return True, [{"_api_error": str(parsed["error"])}]
        if "choices" in parsed:
            return bool(parsed.get("done")), [cls._normalize_done_reason(parsed)]

        msg = parsed.get("message")
        msg = msg if isinstance(msg, dict) else {}
        thinking_text = msg.get("thinking") or ""
        content_text = msg.get("content") or ""
        norm_tcs: List[Dict[str, Any]] = []
        raw_tcs = msg.get("tool_calls")
        if isinstance(raw_tcs, list):
            for tc_i, tc in enumerate(raw_tcs):
                if not isinstance(tc, dict):
                    continue
                tc_copy = dict(tc)
                fn_block = tc_copy.get("function")
                fn_idx = fn_block.get("index") if isinstance(fn_block, dict) else None
                if tc_copy.get("index") is None:
                    tc_copy["index"] = fn_idx if fn_idx is not None else tc_i
                tc_copy.setdefault("type", "function")
                norm_tcs.append(tc_copy)

        done_flag = bool(parsed.get("done"))
        done_r = parsed.get("done_reason") or parsed.get("finish_reason") or ("stop" if done_flag else None)

        events: List[Dict[str, Any]] = []
        if thinking_text and (content_text or norm_tcs):
            events.append({"choices": [{"delta": {"reasoning_content": thinking_text}}]})
            thinking_text = ""

        delta: Dict[str, Any] = {}
        if thinking_text:
            delta["reasoning_content"] = thinking_text
        if content_text:
            delta["content"] = content_text
        if norm_tcs:
            delta["tool_calls"] = norm_tcs

        if delta or done_r:
            choice0: Dict[str, Any] = {"delta": delta}
            if done_r:
                choice0["finish_reason"] = done_r
            events.append({"choices": [choice0]})
        return done_flag, events

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
        ledger: Optional[Any] = None,
        pending_proposals: Optional[List[Dict[str, Any]]] = None,
        replanning_attempts: int = 0,
        max_replanning_attempts: Optional[int] = None,
        task_local_scope: Optional[Any] = None,
    ) -> AsyncGenerator[dict, None]:
        """Stream ReAct execution while recording a root persistent execution ledger entry."""
        if depth > 0 or is_retry:
            async for chunk in self._generate_stream_core(
                payload,
                depth=depth,
                is_retry=is_retry,
                stream_provider=stream_provider,
                tool_schemas=tool_schemas,
                tool_executor=tool_executor,
                db_path=db_path,
                max_depth=max_depth,
                turn_tool_history=turn_tool_history,
                actual_step_count=actual_step_count,
                disable_tools=disable_tools,
                enable_cycle_detection=enable_cycle_detection,
                ledger=ledger,
                pending_proposals=pending_proposals,
                replanning_attempts=replanning_attempts,
                max_replanning_attempts=max_replanning_attempts,
                task_local_scope=task_local_scope,
            ):
                yield chunk
            return

        from zelyr.engine.execution_ledger import (
            ActionState,
            ExecutionStatus,
            LedgerEventType,
            TurnExecutionLedger,
            VerificationStatus,
            detect_false_success,
            get_default_central_ledger,
        )

        eff_ledger = ledger if ledger is not None else TurnExecutionLedger()
        self.current_ledger = eff_ledger

        eff_caller_ring = payload.get("caller_ring") if "caller_ring" in payload else getattr(self, "caller_ring", None)
        eff_task_local_scope = task_local_scope if task_local_scope is not None else getattr(self, "task_local_scope", None)
        if eff_task_local_scope is None:
            eff_exec = tool_executor or getattr(self, "tool_executor", None)
            if eff_exec is not None:
                eff_task_local_scope = getattr(eff_exec, "_task_local_tool_scope", None)

        central_ledger = (
            getattr(self, "central_ledger", None)
            or payload.get("central_ledger")
            or getattr(eff_ledger, "_central_ledger", None)
            or get_default_central_ledger()
        )
        exec_id = getattr(eff_ledger, "_execution_id", None)
        is_resumed_root = bool(payload.get("_resumed_from_checkpoint") and depth == 0 and not is_retry)
        owns_root_exec = False
        user_prompt = ""
        for msg in reversed(payload.get("messages", [])):
            if msg.get("role") == "user":
                c = msg.get("content", "")
                if isinstance(c, list):
                    c = " ".join(part.get("text", "") for part in c if isinstance(part, dict))
                if isinstance(c, str) and not c.startswith(("[SYSTEM", "[ACTUATOR RECOVERY")):
                    user_prompt = c
                    break

        acceptance_spec = payload.get("acceptance_criteria") or getattr(self, "acceptance_criteria", None)
        if not exec_id:
            owns_root_exec = True
            rec = central_ledger.create_execution(
                user_prompt,
                source_component="native_orchestrator",
                caller_ring=eff_caller_ring,
                task_local_scoped=bool(eff_task_local_scope),
                status=ExecutionStatus.RUNNING,
                metadata={
                    "model": payload.get("model"),
                    "has_acceptance_criteria": bool(acceptance_spec),
                },
            )
            exec_id = rec.execution_id

        eff_ledger.attach_central_ledger(
            central_ledger,
            exec_id,
            caller_ring=eff_caller_ring,
            task_local_scoped=bool(eff_task_local_scope),
            source_component="native_orchestrator",
        )
        eff_ledger._root_meta = {
            "cancelled": False,
            "timed_out": False,
            "error": None,
            "connection_error": False,
            "provider_error": None,
            "checkpoint_failed": False,
            "acceptance_criteria": acceptance_spec,
            "model": payload.get("model"),
            "task_prompt": user_prompt,
        }

        raw_task_graph = payload.get("task_graph") or (
            eff_ledger.get_task_graph() if hasattr(eff_ledger, "get_task_graph") else None
        )
        if raw_task_graph is not None:
            from zelyr.engine.task_graph import TaskGraph, TaskGraphValidationError

            if isinstance(raw_task_graph, TaskGraph):
                if raw_task_graph.execution_id and raw_task_graph.execution_id != exec_id:
                    raise TaskGraphValidationError(
                        f"Cross-execution isolation violation: TaskGraph '{raw_task_graph.graph_id}' belongs to '{raw_task_graph.execution_id}', cannot run under '{exec_id}'."
                    )
                if not raw_task_graph.execution_id:
                    raw_task_graph.execution_id = exec_id
                active_graph = raw_task_graph
            elif isinstance(raw_task_graph, dict):
                active_graph = TaskGraph.from_dict(raw_task_graph, default_execution_id=exec_id, verify_integrity=False)
                if active_graph.execution_id != exec_id:
                    raise TaskGraphValidationError(
                        f"Cross-execution isolation violation: TaskGraph '{active_graph.graph_id}' belongs to '{active_graph.execution_id}', cannot run under '{exec_id}'."
                    )
            else:
                raise TaskGraphValidationError("payload['task_graph'] must be a TaskGraph or JSON dictionary.")
            active_graph.validate(
                known_proposals=eff_ledger.get_tracked_proposals() if hasattr(eff_ledger, "get_tracked_proposals") else None,
                allow_blocked=True,
            )
            active_graph.recompute_schedule(
                known_proposals=eff_ledger.get_tracked_proposals() if hasattr(eff_ledger, "get_tracked_proposals") else None,
            )
            if hasattr(eff_ledger, "attach_task_graph"):
                eff_ledger.attach_task_graph(active_graph)

        active_tg_meta = {}
        active_tg_now = eff_ledger.get_task_graph() if hasattr(eff_ledger, "get_task_graph") else None
        if active_tg_now is not None:
            active_tg_meta = {"graph_id": active_tg_now.graph_id, "graph_status": active_tg_now.status}

        central_ledger.record_event(
            exec_id,
            LedgerEventType.PLANNING_DECISION,
            status="planned" if not is_resumed_root else "resumed_plan",
            caller_ring=eff_caller_ring,
            task_local_scoped=bool(eff_task_local_scope),
            metadata={
                "max_steps": getattr(self, "max_steps", 5),
                "has_acceptance_criteria": bool(acceptance_spec),
                **active_tg_meta,
            },
            idempotency_key=f"{exec_id}:plan:{'resume' if is_resumed_root else 'root'}:{actual_step_count}",
        )
        if owns_root_exec:
            from zelyr.engine.checkpoint import CheckpointManager

            cp_init = CheckpointManager.capture_checkpoint(
                central_ledger,
                exec_id,
                turn_ledger=eff_ledger,
                status=ExecutionStatus.RUNNING,
                verification_status=VerificationStatus.NOT_VERIFIED,
                task_prompt=user_prompt,
                planning_state={
                    "depth": depth,
                    "actual_step_count": actual_step_count,
                    "replanning_attempts": replanning_attempts,
                    "max_steps": getattr(self, "max_steps", 5),
                    "acceptance_criteria": acceptance_spec,
                    "model": payload.get("model"),
                },
                emit_event=False,
                strict=None,
            )
            if cp_init is None and central_ledger.has_persistence_errors():
                eff_ledger._root_meta["checkpoint_failed"] = True

        collected_out: List[str] = []
        caught_exc: Optional[Exception] = None
        was_cancelled = False
        try:
            async for chunk in self._generate_stream_core(
                payload,
                depth=depth,
                is_retry=is_retry,
                stream_provider=stream_provider,
                tool_schemas=tool_schemas,
                tool_executor=tool_executor,
                db_path=db_path,
                max_depth=max_depth,
                turn_tool_history=turn_tool_history,
                actual_step_count=actual_step_count,
                disable_tools=disable_tools,
                enable_cycle_detection=enable_cycle_detection,
                ledger=eff_ledger,
                pending_proposals=pending_proposals,
                replanning_attempts=replanning_attempts,
                max_replanning_attempts=max_replanning_attempts,
                task_local_scope=task_local_scope,
            ):
                for ch in chunk.get("choices", []) if isinstance(chunk, dict) else []:
                    delta = ch.get("delta", {}) if isinstance(ch, dict) else {}
                    txt = delta.get("content")
                    if isinstance(txt, str) and txt:
                        collected_out.append(txt)
                yield chunk
        except asyncio.CancelledError:
            was_cancelled = True
            raise
        except Exception as exc:
            caught_exc = exc
            raise
        finally:
            if (owns_root_exec or is_resumed_root) and getattr(eff_ledger, "_execution_id", None) == exec_id:
                existing_rec = central_ledger.get_execution(exec_id)
                already_finalized = (
                    existing_rec is not None
                    and not is_resumed_root
                    and existing_rec.status
                    in (
                        ExecutionStatus.COMPLETED.value,
                        ExecutionStatus.COMPLETED_UNVERIFIED.value,
                        ExecutionStatus.FAILED.value,
                        ExecutionStatus.CANCELLED.value,
                        ExecutionStatus.TIMED_OUT.value,
                        ExecutionStatus.INTERRUPTED.value,
                        ExecutionStatus.ABANDONED.value,
                    )
                )
                if not already_finalized:
                    root_meta = getattr(eff_ledger, "_root_meta", {}) or {}
                    final_text = "".join(collected_out)
                    all_records = [eff_ledger._records[cid] for cid in eff_ledger._order]
                    has_ver_fail = any(getattr(r, "verification_failed", False) for r in all_records)
                    has_unresolved = eff_ledger.has_unresolved_failures()
                    has_verified = bool(eff_ledger.verified_actions() or eff_ledger.get_verified_targets())
                    total_acts = eff_ledger.total_actions()
                    interruption_reason: Optional[str] = None

                    if (
                        was_cancelled
                        or getattr(self, "_cancelled", False)
                        or payload.get("cancelled", False)
                        or root_meta.get("cancelled")
                    ):
                        final_status = ExecutionStatus.CANCELLED
                        ver_status = VerificationStatus.INTERRUPTED
                        err_msg = "Task cancelled"
                        interruption_reason = "user_cancelled"
                    elif (root_meta.get("timed_out") and has_unresolved) or isinstance(caught_exc, (TimeoutError, asyncio.TimeoutError)):
                        final_status = ExecutionStatus.TIMED_OUT
                        ver_status = VerificationStatus.INTERRUPTED
                        err_msg = str(caught_exc) if isinstance(caught_exc, (TimeoutError, asyncio.TimeoutError)) else (root_meta.get("error") or "Tool execution timed out")
                        interruption_reason = "task_timeout" if isinstance(caught_exc, (TimeoutError, asyncio.TimeoutError)) else "tool_timeout"
                    elif root_meta.get("connection_error") or root_meta.get("provider_error"):
                        final_status = ExecutionStatus.INTERRUPTED
                        ver_status = VerificationStatus.INTERRUPTED
                        err_msg = str(root_meta.get("provider_error") or root_meta.get("error") or "Model provider failure")
                        interruption_reason = "provider_failure"
                    elif caught_exc is not None:
                        final_status = ExecutionStatus.INTERRUPTED
                        ver_status = VerificationStatus.INTERRUPTED
                        err_msg = str(caught_exc) or type(caught_exc).__name__
                        interruption_reason = f"exception:{type(caught_exc).__name__}"
                    elif has_ver_fail:
                        final_status = ExecutionStatus.FAILED
                        ver_status = VerificationStatus.VERIFICATION_FAILED
                        err_msg = "Outcome verification failed"
                        interruption_reason = "verification_failed"
                    elif has_unresolved:
                        ver_status = (
                            VerificationStatus.CLAIMED_UNVERIFIED
                            if detect_false_success(final_text, has_unresolved=True)
                            else VerificationStatus.NOT_VERIFIED
                        )
                        final_status = ExecutionStatus.FAILED
                        unresolved = eff_ledger.unresolved_failures()
                        err_msg = unresolved[-1].error if unresolved else "Unresolved action failure"
                        interruption_reason = "unresolved_failure"
                    elif has_verified:
                        final_status = ExecutionStatus.COMPLETED
                        ver_status = VerificationStatus.VERIFIED
                        err_msg = None
                    elif total_acts > 0:
                        if acceptance_spec and detect_false_success(final_text, has_unresolved=False):
                            ver_status = VerificationStatus.CLAIMED_UNVERIFIED
                        else:
                            ver_status = VerificationStatus.NOT_VERIFIED
                        final_status = ExecutionStatus.COMPLETED_UNVERIFIED
                        err_msg = None
                    else:
                        if acceptance_spec:
                            final_status = ExecutionStatus.FAILED
                            ver_status = VerificationStatus.NOT_VERIFIED
                            err_msg = "Task ended without executing or verifying required acceptance criteria"
                            interruption_reason = "unfulfilled_criteria"
                        else:
                            final_status = ExecutionStatus.COMPLETED
                            ver_status = VerificationStatus.NOT_VERIFIED
                            err_msg = None

                    eff_ledger.set_verification_status(ver_status)
                    from zelyr.engine.checkpoint import CheckpointManager

                    cp_err: Optional[str] = None
                    try:
                        cp_saved = CheckpointManager.capture_checkpoint(
                            central_ledger,
                            exec_id,
                            turn_ledger=eff_ledger,
                            status=final_status,
                            verification_status=ver_status,
                            task_prompt=user_prompt,
                            planning_state={
                                "depth": depth,
                                "actual_step_count": actual_step_count,
                                "replanning_attempts": replanning_attempts,
                                "max_steps": getattr(self, "max_steps", 5),
                                "acceptance_criteria": acceptance_spec,
                                "model": payload.get("model"),
                                "interruption_reason": interruption_reason,
                                "last_assistant_text": final_text[:512] if final_text else None,
                            },
                            pending_proposals=getattr(eff_ledger, "_active_pending_queue", None),
                            emit_event=False,
                            strict=None,
                        )
                        if cp_saved is None and central_ledger.has_persistence_errors():
                            cp_err = "Checkpoint persistence failed; state could not be durably saved."
                    except Exception as exc:
                        cp_err = f"Checkpoint persistence failed: {exc}"
                        if central_ledger.raise_on_persistence_error:
                            central_ledger.finalize_execution(
                                exec_id,
                                status=final_status,
                                verification_status=ver_status,
                                final_summary=final_text[:1024] if final_text else None,
                                error=err_msg,
                                metadata={"total_actions": total_acts, "verified_targets": eff_ledger.get_verified_targets(), "checkpoint_saved": False},
                            )
                            raise

                    if cp_err:
                        root_meta["checkpoint_failed"] = True
                        root_meta["checkpoint_error"] = cp_err

                    central_ledger.finalize_execution(
                        exec_id,
                        status=final_status,
                        verification_status=ver_status,
                        final_summary=final_text[:1024] if final_text else None,
                        error=err_msg,
                        metadata={
                            "total_actions": total_acts,
                            "verified_targets": eff_ledger.get_verified_targets(),
                            "checkpoint_saved": not bool(root_meta.get("checkpoint_failed")),
                            **({"interruption_reason": interruption_reason} if interruption_reason else {}),
                            **({"checkpoint_error": cp_err} if cp_err else {}),
                        },
                    )

    async def _generate_stream_core(
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
        ledger: Optional[Any] = None,
        pending_proposals: Optional[List[Dict[str, Any]]] = None,
        replanning_attempts: int = 0,
        max_replanning_attempts: Optional[int] = None,
        task_local_scope: Optional[Any] = None,
    ) -> AsyncGenerator[dict, None]:
        eff_max_depth = max_depth if max_depth is not None else getattr(self, "max_depth", self.max_steps)
        eff_max_replanning = max_replanning_attempts if max_replanning_attempts is not None else getattr(self, "max_replanning_attempts", 1)
        eff_stream_provider = stream_provider or getattr(self, "stream_provider", None) or getattr(self, "model_provider", None)
        eff_tool_schemas = tool_schemas if tool_schemas is not None else getattr(self, "tool_schemas", None)
        eff_tool_executor = tool_executor or getattr(self, "tool_executor", None)
        eff_task_local_scope = (
            task_local_scope
            if task_local_scope is not None
            else getattr(self, "task_local_scope", None)
        )
        if eff_task_local_scope is None and eff_tool_executor is not None:
            eff_task_local_scope = getattr(eff_tool_executor, "_task_local_tool_scope", None)
        eff_db_path = db_path or getattr(self, "db_path", None)
        curr_tool_history = turn_tool_history if turn_tool_history is not None else []
        eff_cycle_detection = self.enable_cycle_detection if enable_cycle_detection is None else enable_cycle_detection

        from zelyr.engine.execution_ledger import (
            ActionState,
            LedgerEventType,
            TurnExecutionLedger,
            VerificationStatus,
            detect_consultant_deferral,
            detect_false_success,
            execution_scope,
            has_imperative_action_intent,
            has_valid_inaction_justification,
            is_truthful_failure_report,
            is_unresolvable_terminal_blocker,
        )

        messages = payload.get("messages", [])

        if ledger is not None:
            eff_ledger = ledger
            self.current_ledger = ledger
        elif depth == 0:
            eff_ledger = TurnExecutionLedger()
            self.current_ledger = eff_ledger
        else:
            eff_ledger = getattr(self, "current_ledger", None) or TurnExecutionLedger()
            self.current_ledger = eff_ledger

        if eff_ledger.total_actions() == 0 and messages:
            eff_ledger.hydrate_from_messages(messages)

        if pending_proposals is not None:
            curr_pending_proposals = list(pending_proposals)
        elif depth == 0 and not payload.get("_resumed_from_checkpoint"):
            curr_pending_proposals = []
        else:
            curr_pending_proposals = list(payload.get("_pending_proposals") or [])
        proposals_before_turn = list(curr_pending_proposals)
        payload["_pending_proposals"] = curr_pending_proposals
        eff_ledger._active_pending_queue = curr_pending_proposals

        if getattr(self, "_cancelled", False) or payload.get("cancelled", False):
            if hasattr(eff_ledger, "_root_meta") and isinstance(eff_ledger._root_meta, dict):
                eff_ledger._root_meta["cancelled"] = True
            if eff_ledger:
                eff_ledger.discard_pending_proposals(reason="task_cancelled")
            curr_pending_proposals.clear()
            return

        eff_step_limit = getattr(self, "max_steps", 5)
        if depth >= eff_max_depth:
            if eff_ledger:
                eff_ledger.discard_pending_proposals(reason="max_depth_reached")
                if getattr(eff_ledger, "_central_ledger", None) and getattr(eff_ledger, "_execution_id", None):
                    from zelyr.engine.execution_ledger import LedgerEventType

                    eff_ledger._central_ledger.record_event(
                        eff_ledger._execution_id,
                        LedgerEventType.PLANNING_DECISION,
                        status="max_depth_reached",
                        metadata={"depth": depth, "max_depth": eff_max_depth},
                        idempotency_key=f"{eff_ledger._execution_id}:plan:max_depth:{depth}",
                    )
            curr_pending_proposals.clear()
            yield {"choices": [{"delta": {"content": f"\n[Notice: Reached maximum autonomous ReAct steps ({eff_max_depth}).]\n"}}]}
            return

        if actual_step_count >= eff_step_limit:
            if eff_ledger:
                eff_ledger.discard_pending_proposals(reason="step_limit_reached")
                if getattr(eff_ledger, "_central_ledger", None) and getattr(eff_ledger, "_execution_id", None):
                    from zelyr.engine.execution_ledger import LedgerEventType

                    eff_ledger._central_ledger.record_event(
                        eff_ledger._execution_id,
                        LedgerEventType.PLANNING_DECISION,
                        status="step_limit_reached",
                        metadata={"actual_step_count": actual_step_count, "step_limit": eff_step_limit},
                        idempotency_key=f"{eff_ledger._execution_id}:plan:step_limit:{actual_step_count}",
                    )
            curr_pending_proposals.clear()
            disable_tools = True

        acceptance_spec = payload.get("acceptance_criteria") or getattr(self, "acceptance_criteria", None)
        if acceptance_spec and "acceptance_criteria" not in payload:
            payload["acceptance_criteria"] = acceptance_spec

        eff_caller_ring = payload.get("caller_ring") if "caller_ring" in payload else getattr(self, "caller_ring", None)
        eff_allowed_tools = payload.get("allowed_tools") if "allowed_tools" in payload else getattr(self, "allowed_tools", None)

        config = self._active_config()
        base_url = (
            config.get_effective_ollama_base_url()
            if hasattr(config, "get_effective_ollama_base_url")
            else getattr(config, "ollama_base_url", "http://127.0.0.1:11434").rstrip("/")
        )

        # Inspect if preceding turn was a failed tool execution requiring actuator recovery
        is_recovery_turn = False
        recovery_hint = ""
        allowed_recovery_actions = []
        recovery_error_type = ""

        if messages:
            for m in reversed(messages):
                if m.get("role") == "user" and (
                    "[ACTUATOR RECOVERY REQUIRED]" in str(m.get("content", ""))
                    or "[SYSTEM DIRECTIVE:" in str(m.get("content", ""))
                ):
                    continue
                if m.get("role") == "tool":
                    c = str(m.get("content", ""))
                    m_status = m.get("status")
                    m_remedy = m.get("remedy_hint")
                    m_actions = m.get("allowed_actions")
                    m_error_type = m.get("error_type")

                    from zelyr.engine.observation_framing import ObservationFramingEngine

                    parsed_c = ObservationFramingEngine.parse_observation(c)

                    if parsed_c.get("status") == "error" or (parsed_c.get("exit_code") is not None and parsed_c.get("exit_code") != 0):
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

            system_prompt_content = self._build_system_prompt(
                caller_ring=eff_caller_ring,
                allowed_tools=eff_allowed_tools,
            )

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
            tools=payload.get("tools") or eff_tool_schemas,
            config=config,
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

        # Enforce adequate context size in Ollama and determine schema compaction mode
        options = payload.setdefault("options", {})
        eff_ctx = config.get_effective_num_ctx(
            options.get("num_ctx") or self._explicit_max_context_tokens
        )
        options["num_ctx"] = eff_ctx

        budget_pressure_ratio = 0.0
        if hasattr(self, "budget_engine") and self.budget_engine is not None:
            try:
                headroom = self.budget_engine.get_context_headroom(
                    payload.get("messages", []),
                    max_context_tokens=eff_ctx,
                    tools=payload.get("tools") or eff_tool_schemas,
                    config=config,
                )
                budget_pressure_ratio = headroom.get("utilization", 0.0)
            except Exception:
                pass

        use_compact_schemas = bool(
            getattr(self.config, "compact_schemas", False)
            or getattr(config, "compact_schemas", False)
            or eff_ctx <= 8192
            or budget_pressure_ratio >= 0.5
        )

        from zelyr.core.plugins import filter_tool_schemas_by_tier
        from zelyr.agents.actuator_arbiter import is_browser_window
        browser_active = is_browser_window(active_window)

        from zelyr.tool_registry import authorize_and_validate_tool_call

        from zelyr.tool_registry import TaskLocalToolScope

        is_valid_task_scope = isinstance(eff_task_local_scope, TaskLocalToolScope)
        has_valid_registry = self.tool_registry is not None and callable(getattr(self.tool_registry, "get_schemas", None))
        has_valid_schemas_list = (
            isinstance(eff_tool_schemas, (list, tuple))
            and all(isinstance(s, dict) for s in eff_tool_schemas)
        )

        def _is_schema_tool_authorized(t_name: str, schemas_ctx: Optional[List[Dict[str, Any]]] = None) -> bool:
            is_ext_candidate = False
            if self.tool_registry is not None:
                if hasattr(self.tool_registry, "get_external_tool_entry") and self.tool_registry.get_external_tool_entry(t_name) is not None:
                    is_ext_candidate = True
                elif hasattr(self.tool_registry, "get_tool") and callable(getattr(self.tool_registry, "get_tool", None)):
                    try:
                        reg_t = self.tool_registry.get_tool(t_name)
                        if reg_t is not None and getattr(reg_t, "is_external", False):
                            is_ext_candidate = True
                    except Exception:
                        pass
            if not is_ext_candidate and eff_allowed_tools is None and eff_caller_ring is None and eff_task_local_scope is None:
                return True
            dec = authorize_and_validate_tool_call(
                t_name,
                {},
                caller_ring=eff_caller_ring,
                allowed_tools=eff_allowed_tools,
                active_schemas=schemas_ctx,
                tool_registry=self.tool_registry,
                task_local_scope=eff_task_local_scope,
                normalize_args=False,
                validate_schema=False,
            )
            return dec.allowed

        if disable_tools:
            payload["tools"] = []
        elif eff_task_local_scope is not None:
            if is_valid_task_scope and self.tool_registry is None:
                scope_schemas = (
                    list(eff_tool_schemas)
                    if has_valid_schemas_list
                    else list(eff_task_local_scope.tool_schemas)
                )
                payload["tools"] = [
                    t for t in scope_schemas
                    if _is_schema_tool_authorized(
                        (t.get("function", {}).get("name") or t.get("name") or ""),
                        scope_schemas,
                    )
                ]
            else:
                payload["tools"] = []
        elif eff_tool_schemas is not None:
            if has_valid_schemas_list:
                payload["tools"] = [
                    t for t in eff_tool_schemas
                    if _is_schema_tool_authorized(
                        (t.get("function", {}).get("name") or t.get("name") or ""),
                        list(eff_tool_schemas),
                    )
                ]
            else:
                payload["tools"] = []
        else:
            if self.tool_registry is not None:
                if has_valid_registry:
                    active_tools = payload.get("tools") or self.tool_registry.get_schemas(compact=use_compact_schemas)
                else:
                    active_tools = []
            else:
                active_tools = payload.get("tools") or get_tool_schemas(0)

            # Keep authorized, live external tools eligible across automation tiers while preserving native tier gating
            external_names: set[str] = set()
            if self.tool_registry is not None and callable(getattr(self.tool_registry, "get_tool", None)):
                for s in active_tools:
                    s_name = (s.get("function", {}).get("name") or s.get("name") or "") if isinstance(s, dict) else ""
                    if s_name:
                        try:
                            reg_t = self.tool_registry.get_tool(s_name)
                            if reg_t is not None and getattr(reg_t, "is_external", False):
                                external_names.add(s_name)
                        except Exception:
                            pass
            native_tools_only = [
                s for s in active_tools
                if (s.get("function", {}).get("name") or s.get("name") or "") not in external_names
            ]
            external_tools_only = [
                s for s in active_tools
                if (s.get("function", {}).get("name") or s.get("name") or "") in external_names
            ]
            payload["tools"] = filter_tool_schemas_by_tier(native_tools_only, tier, is_browser=browser_active)
            payload["tools"].extend(external_tools_only)

            # Strict negative gating: purge Tier 3 interact_with_ui whenever a browser window is active
            if browser_active:
                payload["tools"] = [
                    t for t in payload["tools"]
                    if (t.get("function", {}).get("name") or t.get("name")) != "interact_with_ui"
                ]

            payload["tools"] = [
                t for t in payload["tools"]
                if _is_schema_tool_authorized(
                    (t.get("function", {}).get("name") or t.get("name") or ""),
                    payload["tools"],
                )
            ]

        # Core Sandbox Protection: Ensure essential actuators (execute_command, manage_workspace_file) are never dropped,
        # UNLESS running in an isolated task-local test scope or excluded by caller restrictions.
        if (
            not disable_tools
            and eff_task_local_scope is None
            and (self.tool_registry is None or has_valid_registry)
            and (eff_tool_schemas is None or has_valid_schemas_list)
            and payload.get("tools") is not None
        ):
            existing_tool_names = {
                t.get("function", {}).get("name") or t.get("name")
                for t in payload["tools"]
                if isinstance(t, dict)
            }
            registry_for_essential = self.tool_registry
            if registry_for_essential is None:
                from zelyr.tool_registry import ToolRegistry
                registry_for_essential = ToolRegistry()
            all_registry_tools = registry_for_essential.get_schemas(compact=use_compact_schemas)
            for essential in ("execute_command", "manage_workspace_file"):
                if essential not in existing_tool_names:
                    found = False
                    for cand in all_registry_tools:
                        cand_name = cand.get("function", {}).get("name") or cand.get("name")
                        if cand_name == essential:
                            if _is_schema_tool_authorized(essential, [cand]):
                                payload["tools"].append(cand)
                                existing_tool_names.add(essential)
                            found = True
                            break
                    if not found and self.tool_registry is None:
                        for cand in get_tool_schemas(0):
                            cand_name = cand.get("function", {}).get("name") or cand.get("name")
                            if cand_name == essential:
                                if _is_schema_tool_authorized(essential, [cand]):
                                    payload["tools"].append(cand)
                                    existing_tool_names.add(essential)
                                break

        # Active tool scoping during recovery: prioritize allowed_actions + foundational diagnostic tools
        unresolved_exec_failures = [
            r for r in eff_ledger.unresolved_failures()
            if not getattr(r, "verification_failed", False)
        ] if hasattr(eff_ledger, "unresolved_failures") else []
        if (is_recovery_turn or unresolved_exec_failures) and not disable_tools and payload.get("tools"):
            effective_allowed = set(allowed_recovery_actions)
            if not effective_allowed and unresolved_exec_failures:
                last_failed_for_tools = unresolved_exec_failures[-1]
                from zelyr.engine.error_classifier import ErrorClassifier
                classification_for_tools = ErrorClassifier.classify(
                    last_failed_for_tools.tool_name,
                    exit_code=last_failed_for_tools.exit_code,
                    error=last_failed_for_tools.error or "",
                )
                effective_allowed.update(classification_for_tools.allowed_actions)
                if last_failed_for_tools.tool_name:
                    effective_allowed.add(last_failed_for_tools.tool_name)

            foundational_tools = {"execute_command", "manage_workspace_file"}
            permitted_recovery_tools = effective_allowed | foundational_tools
            scoped_tools = [
                t for t in payload["tools"]
                if (t.get("function", {}).get("name") or t.get("name")) in permitted_recovery_tools
            ]
            if scoped_tools:
                payload["tools"] = scoped_tools

        if not disable_tools and use_compact_schemas and payload.get("tools"):
            from zelyr.engine.schema_compact import CompactSchemaGenerator
            payload["tools"] = CompactSchemaGenerator.compact_schemas(payload["tools"])

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

        # Context size in Ollama already finalized in options["num_ctx"]

        pending_tool_calls = []
        is_reasoning = False
        open_think_buffered = False
        open_think_yielded = False
        collected_assistant_text = []
        buffered_chunks = []
        buffering = True
        refusal_detected = False
        stream_finish_reason = None

        try:
            async for data in self._stream_model_events(payload, base_url, stream_provider=eff_stream_provider):
                if "_api_error" in data:
                    if hasattr(eff_ledger, "_root_meta") and isinstance(eff_ledger._root_meta, dict):
                        eff_ledger._root_meta["provider_error"] = str(data["_api_error"])
                        eff_ledger._root_meta["error"] = str(data["_api_error"])
                    eff_ledger._active_pending_queue = list(curr_pending_proposals)
                    yield {"choices": [{"delta": {"content": f"\n⚠️ [API Error] {data['_api_error']}\n"}}]}
                    return

                choices_list = data.get("choices")
                if choices_list and isinstance(choices_list, list) and choices_list[0]:
                    fr = choices_list[0].get("finish_reason") or data.get("done_reason") or data.get("finish_reason")
                    if fr:
                        stream_finish_reason = fr
                        self.last_finish_reason = fr
                elif data.get("done_reason") or data.get("finish_reason"):
                    stream_finish_reason = data.get("done_reason") or data.get("finish_reason")
                    self.last_finish_reason = stream_finish_reason

                delta = data.get("choices", [{}])[0].get("delta", {})

                if "reasoning_content" in delta or "reasoning" in delta:
                    r_content = delta.pop("reasoning_content", None) or delta.pop("reasoning", "")
                    if not is_reasoning:
                        delta["content"] = "<think>" + r_content
                        is_reasoning = True
                        if buffering:
                            open_think_buffered = True
                        else:
                            open_think_yielded = True
                    else:
                        delta["content"] = r_content
                elif "content" in delta and delta["content"] and is_reasoning:
                    if (buffering and open_think_buffered) or (not buffering and open_think_yielded):
                        delta["content"] = "</think>" + delta["content"]
                    else:
                        collected_assistant_text.append("</think>")
                    is_reasoning = False
                    open_think_buffered = False
                    open_think_yielded = False

                if "content" in delta and delta["content"]:
                    collected_assistant_text.append(delta["content"])

                if "tool_calls" in delta:
                    for tc in delta["tool_calls"]:
                        idx = tc.get("index", 0)
                        while len(pending_tool_calls) <= idx:
                            pending_tool_calls.append(
                                {"id": "", "type": "function", "function": {"name": "", "arguments": ""}, "_has_arguments": False}
                            )
                        if "id" in tc and tc["id"]:
                            pending_tool_calls[idx]["id"] = tc["id"]
                        for opt_k in ("proposal_id", "depends_on", "expires_at", "external_source", "node_id"):
                            if opt_k in tc and tc[opt_k] is not None:
                                pending_tool_calls[idx][opt_k] = tc[opt_k]
                        if "function" in tc and isinstance(tc["function"], dict):
                            if "name" in tc["function"] and tc["function"]["name"]:
                                raw_fn_name = tc["function"]["name"]
                                pending_tool_calls[idx]["raw_name"] = raw_fn_name
                                pending_tool_calls[idx]["function"]["name"] = self.normalize_tool_name(
                                    raw_fn_name, active_schemas=eff_tool_schemas
                                )
                            if "arguments" in tc["function"] and tc["function"]["arguments"] is not None:
                                pending_tool_calls[idx]["_has_arguments"] = True
                                incoming_args = tc["function"]["arguments"]
                                if isinstance(incoming_args, dict):
                                    pending_tool_calls[idx]["function"]["arguments"] = json.dumps(incoming_args)
                                elif isinstance(incoming_args, str):
                                    pending_tool_calls[idx]["function"]["arguments"] += incoming_args
                                else:
                                    pending_tool_calls[idx]["function"]["arguments"] = json.dumps(incoming_args)
                        elif "name" in tc and tc["name"]:
                            raw_fn_name = tc["name"]
                            pending_tool_calls[idx]["raw_name"] = raw_fn_name
                            pending_tool_calls[idx]["function"]["name"] = self.normalize_tool_name(
                                raw_fn_name, active_schemas=eff_tool_schemas
                            )
                            if "arguments" in tc and tc["arguments"] is not None:
                                pending_tool_calls[idx]["_has_arguments"] = True
                                incoming_args = tc["arguments"]
                                if isinstance(incoming_args, dict):
                                    pending_tool_calls[idx]["function"]["arguments"] = json.dumps(incoming_args)
                                elif isinstance(incoming_args, str):
                                    pending_tool_calls[idx]["function"]["arguments"] += incoming_args
                                else:
                                    pending_tool_calls[idx]["function"]["arguments"] = json.dumps(incoming_args)

                if buffering:
                    if "tool_calls" in delta or pending_tool_calls:
                        buffering = False
                        clean_so_far = "".join(collected_assistant_text)
                        if not is_recovery_turn and not clean_so_far.lstrip().startswith(("{", "[", "```", "<")):
                            for b in buffered_chunks:
                                yield b
                            if open_think_buffered:
                                open_think_yielded = True
                        buffered_chunks.clear()
                        open_think_buffered = False
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
                            if (
                                not is_recovery_turn
                                and not eff_ledger.has_unresolved_failures()
                                and not (depth == 0 and has_imperative_action_intent(user_query_for_intent))
                                and len(clean_text) > 250
                            ):
                                buffering = False
                                for b in buffered_chunks:
                                    yield b
                                if open_think_buffered:
                                    open_think_yielded = True
                                buffered_chunks.clear()
                                open_think_buffered = False
                        elif (
                            not is_recovery_turn
                            and not eff_ledger.has_unresolved_failures()
                            and not (depth == 0 and has_imperative_action_intent(user_query_for_intent))
                            and len(clean_text) > 4000
                        ):
                            buffering = False
                            for b in buffered_chunks:
                                yield b
                            if open_think_buffered:
                                open_think_yielded = True
                            buffered_chunks.clear()
                            open_think_buffered = False
                else:
                    yield data
        except httpx.ConnectError as exc:
            if hasattr(eff_ledger, "_root_meta") and isinstance(eff_ledger._root_meta, dict):
                eff_ledger._root_meta["connection_error"] = True
                eff_ledger._root_meta["error"] = str(exc)
            eff_ledger._active_pending_queue = list(curr_pending_proposals)
            yield {"choices": [{"delta": {"content": f"\n⚠️ [Connection Error] Could not connect to local AI engine at {base_url}: {exc}. Ensure Ollama is running.\n"}}]}
            return
        except httpx.RequestError as exc:
            if hasattr(eff_ledger, "_root_meta") and isinstance(eff_ledger._root_meta, dict):
                eff_ledger._root_meta["connection_error"] = True
                eff_ledger._root_meta["error"] = str(exc)
            eff_ledger._active_pending_queue = list(curr_pending_proposals)
            err_detail = str(exc).strip()
            err_label = f"{type(exc).__name__}: {err_detail}" if err_detail else type(exc).__name__
            yield {"choices": [{"delta": {"content": f"\n⚠️ [Request Error] Communication failure with {base_url}: {err_label}\n"}}]}
            return

        if is_reasoning:
            collected_assistant_text.append("</think>")
            if buffering and open_think_buffered:
                buffered_chunks.append({"choices": [{"delta": {"content": "</think>"}}]})
            elif not buffering and open_think_yielded:
                yield {"choices": [{"delta": {"content": "</think>"}}]}
            is_reasoning = False
            open_think_buffered = False
            open_think_yielded = False

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

            async for retry_chunk in self.generate_stream(
                payload,
                depth=depth,
                is_retry=True,
                stream_provider=eff_stream_provider,
                tool_schemas=eff_tool_schemas,
                tool_executor=eff_tool_executor,
                db_path=eff_db_path,
                max_depth=eff_max_depth,
                turn_tool_history=curr_tool_history,
                actual_step_count=actual_step_count,
                disable_tools=disable_tools,
                ledger=eff_ledger,
                enable_cycle_detection=eff_cycle_detection,
                task_local_scope=eff_task_local_scope,
            ):
                yield retry_chunk
            return

        new_pending_proposals: List[Dict[str, Any]] = []

        # Enforce single tool execution constraint: execute first call in this turn,
        # retain remaining proposed calls in pending queue for subsequent turns.
        if len(pending_tool_calls) > 1:
            additional_native = pending_tool_calls[1:]
            pending_tool_calls = pending_tool_calls[:1]
            for idx_rem, anc in enumerate(additional_native, 1):
                anc_raw = anc.get("raw_name") or anc.get("function", {}).get("name") or anc.get("name") or ""
                anc_name = self.normalize_tool_name(
                    anc_raw,
                    active_schemas=eff_tool_schemas,
                    task_local_scope=eff_task_local_scope,
                )
                anc_args = anc.get("function", {}).get("arguments") or anc.get("arguments", {})
                if isinstance(anc_args, str):
                    try:
                        anc_args = json.loads(anc_args)
                    except Exception:
                        anc_args = {}
                raw_anc_id = anc.get("id") or f"call_native_rem_{depth}_{idx_rem}_{uuid.uuid4().hex[:6]}"
                anc_id = self._unique_ledger_call_id(eff_ledger, raw_anc_id)
                anc_deps = anc.get("depends_on") if isinstance(anc.get("depends_on"), list) else None
                anc_exp = anc.get("expires_at")
                anc_pid = anc.get("proposal_id")
                anc_ext = anc.get("external_source") if isinstance(anc.get("external_source"), dict) else None
                if anc_ext is None and self.tool_registry is not None:
                    try:
                        ext_e = (
                            self.tool_registry.get_external_tool_entry(anc_name)
                            if callable(getattr(self.tool_registry, "get_external_tool_entry", None))
                            else None
                        )
                        if ext_e is not None:
                            anc_ext = {
                                "is_external": True,
                                "source_id": getattr(ext_e, "source_id", None) or "external",
                                "source_type": getattr(ext_e, "source_type", None) or "external",
                                "external_tool_name": getattr(ext_e, "external_tool_name", None) or anc_name,
                            }
                    except Exception:
                        anc_ext = None
                eff_ledger.record_proposed(
                    anc_id,
                    anc_name,
                    anc_args if isinstance(anc_args, dict) else {},
                    turn=depth,
                    raw_name=anc_raw,
                    proposal_id=anc_pid,
                    depends_on=anc_deps,
                    expires_at=anc_exp,
                    external_source=anc_ext,
                )
                p_meta = eff_ledger.get_proposal_metadata(anc_id) if hasattr(eff_ledger, "get_proposal_metadata") else None
                new_pending_proposals.append({
                    "id": anc_id,
                    "proposal_id": (p_meta or {}).get("proposal_id") or anc_pid or f"prop_{anc_id}",
                    "tool_name": anc_name,
                    "raw_name": anc_raw,
                    "arguments": anc_args if isinstance(anc_args, dict) else {},
                    "turn": depth,
                    "source": "native",
                    "depends_on": list(anc_deps or []),
                    "expires_at": anc_exp,
                    "external_source": anc_ext,
                    "creation_event_id": (p_meta or {}).get("creation_event_id"),
                    "creation_seq": (p_meta or {}).get("creation_seq", 0),
                    "proposal_order": (p_meta or {}).get("proposal_order", idx_rem),
                })

        if not disable_tools and not pending_tool_calls and collected_assistant_text:
            full_text = "".join(collected_assistant_text).strip()
            non_think_text = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL)
            non_think_text = re.sub(r"<think>.*$", "", non_think_text, flags=re.DOTALL).strip()
            if is_valid_task_scope:
                known_tools = eff_task_local_scope.get_known_tools()
            else:
                known_tools = self.get_known_tool_names(payload.get("tools"))

            from zelyr.agents.fallback_parser import FallbackToolParser

            fallback_calls = (
                FallbackToolParser.parse_tool_calls(
                    non_think_text,
                    known_tools=known_tools,
                    active_schemas=payload.get("tools"),
                    max_calls=None,
                    caller_ring=eff_caller_ring,
                    allowed_tools=eff_allowed_tools,
                    tool_registry=self.tool_registry,
                    task_local_scope=eff_task_local_scope,
                )
                if non_think_text
                else []
            )
            if fallback_calls:
                fc = fallback_calls[0]
                func_name = fc.name
                func_args = fc.arguments
                func_args_str = json.dumps(func_args)
                call_id = self._unique_ledger_call_id(eff_ledger, fc.tool_call_id)
                pending_tool_calls = [{
                    "id": call_id,
                    "type": "function",
                    "raw_name": func_name,
                    "function": {
                        "name": func_name,
                        "arguments": func_args_str,
                    },
                    "name": func_name,
                    "arguments": func_args,
                }]
                if len(fallback_calls) > 1:
                    for fc_rem in fallback_calls[1:]:
                        rem_call_id = self._unique_ledger_call_id(eff_ledger, fc_rem.tool_call_id)
                        eff_ledger.record_proposed(
                            rem_call_id,
                            fc_rem.name,
                            fc_rem.arguments,
                            turn=depth,
                            raw_name=fc_rem.name,
                        )
                        new_pending_proposals.append({
                            "id": rem_call_id,
                            "tool_name": fc_rem.name,
                            "raw_name": fc_rem.name,
                            "arguments": fc_rem.arguments,
                            "turn": depth,
                            "source": "fallback",
                        })
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
                                    "arguments": func_args_str,
                                },
                            }]
                        }
                    }]
                }
                cleaned_text = fc.cleaned_text
                collected_assistant_text = [cleaned_text] if cleaned_text else []

        # Prune any TTL-expired proposals before reconciliation
        if proposals_before_turn:
            now_check = time.time()
            unexpired_before: List[Dict[str, Any]] = []
            for p in proposals_before_turn:
                p_exp = p.get("expires_at")
                if p_exp is not None and not isinstance(p_exp, bool) and now_check >= float(p_exp):
                    eff_ledger.cancel_pending_proposal(p["id"], reason="proposal_expired")
                    if hasattr(eff_ledger, "_update_proposal_lifecycle"):
                        eff_ledger._update_proposal_lifecycle(p["id"], "expired", blocked_reason="Proposal TTL expired.")
                else:
                    unexpired_before.append(p)
            proposals_before_turn = unexpired_before

        # Reconcile new turn output with pre-existing pending proposals (proposals_before_turn)
        if pending_tool_calls and proposals_before_turn:
            p0 = proposals_before_turn[0]
            new_call_name = (
                pending_tool_calls[0].get("raw_name")
                or pending_tool_calls[0].get("function", {}).get("name")
                or pending_tool_calls[0].get("name")
            )
            norm_new_name = self.normalize_tool_name(
                new_call_name,
                active_schemas=eff_tool_schemas,
                task_local_scope=eff_task_local_scope,
            )
            p0_args_canon = self._canonicalize_tool_args(p0.get("arguments", {}))
            new_args_raw = (
                pending_tool_calls[0].get("function", {}).get("arguments")
                or pending_tool_calls[0].get("arguments", {})
            )
            if isinstance(new_args_raw, str):
                try:
                    new_args_raw = json.loads(new_args_raw)
                except Exception:
                    new_args_raw = {}
            new_args_canon = self._canonicalize_tool_args(new_args_raw if isinstance(new_args_raw, dict) else {})

            if norm_new_name == p0["tool_name"] and new_args_canon == p0_args_canon:
                # Exact match: model explicitly re-emitted/confirmed the proposal
                eff_ledger.cancel_pending_proposal(p0["id"], reason="reconfirmed_and_dispatched")
                if p0.get("depends_on") and not pending_tool_calls[0].get("depends_on"):
                    pending_tool_calls[0]["depends_on"] = list(p0["depends_on"])
                if p0.get("expires_at") is not None and pending_tool_calls[0].get("expires_at") is None:
                    pending_tool_calls[0]["expires_at"] = p0["expires_at"]
                if p0.get("proposal_id") and not pending_tool_calls[0].get("proposal_id"):
                    pending_tool_calls[0]["proposal_id"] = p0["proposal_id"]
                if isinstance(p0.get("external_source"), dict) and not pending_tool_calls[0].get("external_source"):
                    pending_tool_calls[0]["external_source"] = dict(p0["external_source"])
                proposals_before_turn.pop(0)
                curr_pending_proposals = proposals_before_turn + new_pending_proposals
            else:
                # Model emitted a different tool or different arguments: superseded by model!
                for p in proposals_before_turn:
                    eff_ledger.cancel_pending_proposal(p["id"], reason="superseded_by_model")
                proposals_before_turn.clear()
                curr_pending_proposals = new_pending_proposals
        had_unconfirmed_proposals = False
        if not pending_tool_calls:
            full_text = "".join(collected_assistant_text)
            clean_text = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL)
            clean_text = re.sub(r"<think>.*$", "", clean_text, flags=re.DOTALL).strip()

            is_abandoned = self._is_intentional_abandonment(clean_text)
            reason = "abandoned_by_model" if is_abandoned else "unconfirmed_by_model"
            if proposals_before_turn:
                eff_ledger.discard_pending_proposals(reason=reason)
                proposals_before_turn.clear()
                if not is_abandoned:
                    had_unconfirmed_proposals = True
            curr_pending_proposals = new_pending_proposals
        else:
            curr_pending_proposals = proposals_before_turn + new_pending_proposals
        eff_ledger._active_pending_queue = list(curr_pending_proposals)

        # Check model terminal text if no tools were called
        if not pending_tool_calls:
            full_text = "".join(collected_assistant_text)
            clean_text = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL)
            clean_text = re.sub(r"<think>.*$", "", clean_text, flags=re.DOTALL).strip()

            # Generation exhaustion check: if model reached length limit without emitting any tools or usable text
            if stream_finish_reason in ("length", "max_tokens") and not clean_text:
                buffered_chunks.clear()
                collected_assistant_text.clear()
                yield {
                    "choices": [{
                        "delta": {
                            "content": (
                                "\n⚠️ [Model Error] Model generation truncated: "
                                "token limit reached during reasoning without producing terminal response or tool call.\n"
                            )
                        },
                        "finish_reason": "length",
                    }]
                }
                return

            # 1. Authoritative Runtime Invariants: Unresolved Failure Gate
            unresolved_exec_for_gate = [
                r for r in eff_ledger.unresolved_failures()
                if not getattr(r, "verification_failed", False)
            ]
            structured_for_gate = (
                [
                    ob
                    for ob in eff_ledger.get_structured_observations()
                    if getattr(ob, "verification_status", None) != "verification_failed"
                ]
                if hasattr(eff_ledger, "get_structured_observations")
                else []
            )
            claim_contradicted = False
            if structured_for_gate:
                from zelyr.engine.observation_framing import ObservationFramingEngine

                claim_check = ObservationFramingEngine.reconcile_model_claim(clean_text, structured_for_gate)
                if not claim_check.get("consistent") and claim_check.get("claim_status") == "contradicted":
                    claim_contradicted = True

            if (unresolved_exec_for_gate or claim_contradicted) and depth < min(4, eff_max_depth - 1):
                if claim_contradicted or detect_false_success(clean_text, has_unresolved=True):
                    if getattr(eff_ledger, "_central_ledger", None) and getattr(eff_ledger, "_execution_id", None):
                        from zelyr.engine.execution_ledger import VerificationStatus as _VS

                        eff_ledger._central_ledger.record_event(
                            eff_ledger._execution_id,
                            LedgerEventType.VERIFICATION_RECORDED,
                            status=_VS.CLAIMED_UNVERIFIED.value,
                            metadata={"detail": "Model claimed completion despite unresolved failure", "depth": depth},
                            idempotency_key=f"{eff_ledger._execution_id}:claimed_unverified:{depth}",
                        )
                        eff_ledger._central_ledger.record_event(
                            eff_ledger._execution_id,
                            LedgerEventType.RECOVERY_ATTEMPT,
                            status="remediation_requested",
                            metadata={"reason": "false_success_gate", "depth": depth},
                            idempotency_key=f"{eff_ledger._execution_id}:false_success_recovery:{depth}",
                        )
                    buffered_chunks.clear()
                    collected_assistant_text.clear()

                    last_failed = unresolved_exec_for_gate[-1] if unresolved_exec_for_gate else None
                    failed_tool = (last_failed.tool_name if last_failed else None) or "unknown_action"
                    failed_error = (last_failed.error if last_failed else None) or (
                        f"exit code {last_failed.exit_code}"
                        if last_failed and last_failed.exit_code is not None
                        else "action failed or verification contradicted"
                    )

                    override_msg = {
                        "role": "user",
                        "content": (
                            f"[SYSTEM NOTICE: Unresolved Failure] Action '{failed_tool}' failed with error: {failed_error}. "
                            "Your statement claiming success contradicts the machine state. "
                            "You must either attempt a corrective tool action or truthfully report the failure to the user."
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
                        turn_tool_history=curr_tool_history,
                        actual_step_count=actual_step_count,
                        disable_tools=disable_tools,
                        ledger=eff_ledger,
                        enable_cycle_detection=eff_cycle_detection,
                        replanning_attempts=replanning_attempts,
                        max_replanning_attempts=eff_max_replanning,
                        task_local_scope=eff_task_local_scope,
                    ):
                        yield retry_chunk
                    return

            # 2. Closed-loop error recovery enforcement (Anti-consultant gate):
            # If preceding turn was a failure or unresolved failure exists, and model gave conversational
            # advice without calling tools, reject prose and inject directed remediation until terminal ceiling.
            has_unresolved = eff_ledger.has_unresolved_failures()
            unresolved = eff_ledger.unresolved_failures() if has_unresolved else []
            unresolved_exec = [r for r in unresolved if not getattr(r, "verification_failed", False)]
            last_failed = unresolved_exec[-1] if unresolved_exec else (unresolved[-1] if unresolved else None)

            eff_step_limit = getattr(self, "max_steps", 5)
            if (
                not disable_tools
                and actual_step_count < eff_step_limit
                and is_recovery_turn
                and depth < eff_max_depth - 1
            ):
                failed_tool = (last_failed.tool_name if last_failed else None) or "unknown_action"
                failed_error = (last_failed.error if last_failed else None) or (
                    f"exit code {last_failed.exit_code}"
                    if last_failed and last_failed.exit_code is not None
                    else "action failed"
                )

                from zelyr.engine.error_classifier import ErrorClassifier
                classification = ErrorClassifier.classify(
                    failed_tool,
                    exit_code=last_failed.exit_code if last_failed else None,
                    error=failed_error,
                )
                err_type = classification.error_type or recovery_error_type or "ACTION_FAILED"
                remedy = classification.remedy_hint or recovery_hint or "Inspect error details and adjust arguments."
                allowed_act = classification.allowed_actions or allowed_recovery_actions or ["execute_command", "manage_workspace_file"]
                allowed_act_str = ", ".join(allowed_act) if isinstance(allowed_act, (list, tuple, set)) else str(allowed_act)

                failed_so = (
                    eff_ledger.get_structured_observation(last_failed.call_id)
                    if (last_failed and hasattr(eff_ledger, "get_structured_observation"))
                    else None
                )
                so_guidance = getattr(failed_so, "recovery_guidance", {}) if failed_so is not None else {}
                so_recoverable = bool(so_guidance.get("is_recoverable", True)) if isinstance(so_guidance, dict) else True
                so_side_effect = bool(so_guidance.get("may_have_side_effect", False)) if isinstance(so_guidance, dict) else False

                is_unresolvable = is_unresolvable_terminal_blocker(
                    error_type=err_type,
                    error_message=failed_error,
                    tool_name=failed_tool,
                )
                current_attempts = eff_ledger.get_recovery_attempts(last_failed.call_id if last_failed else None)
                is_terminal = last_failed.terminal if last_failed else False

                if (
                    is_unresolvable
                    or not getattr(classification, "is_recoverable", True)
                    or not so_recoverable
                    or current_attempts >= 2
                    or is_terminal
                ):
                    # Terminal blocker ceiling reached (max 2 attempts) or unresolvable boundary:
                    # Do NOT trigger another recovery override. Permit model to output truthful terminal failure explanation.
                    if last_failed:
                        eff_ledger.mark_terminal(last_failed.call_id)
                else:
                    is_deferral = detect_consultant_deferral(clean_text)
                    is_truthful = is_truthful_failure_report(clean_text)

                    # Intercept consultant deferral or failure evasions
                    if is_deferral or not is_truthful:
                        if last_failed:
                            eff_ledger.increment_recovery_attempts(last_failed.call_id)
                        buffered_chunks.clear()
                        collected_assistant_text.clear()

                        side_effect_caution = (
                            f"\nSide-Effect Caution: {so_guidance.get('next_step') or 'Prior execution was dispatched and may already have modified host state; inspect host state before repeating.'}"
                            if so_side_effect
                            else ""
                        )
                        directive_msg = {
                            "role": "user",
                            "content": (
                                f"[SYSTEM DIRECTIVE: Autonomous Diagnostic & Remediation Required]\n"
                                f"[ACTUATOR RECOVERY REQUIRED]\n"
                                f"[ACTUATOR RECOVERY REQUIRED: {err_type}]\n"
                                f"The action '{failed_tool}' failed with error: {failed_error}.\n"
                                f"Do NOT ask the user to investigate, verify files, or run commands.\n"
                                f"You possess full host capabilities to inspect and resolve this state.\n"
                                f"Remedy Hint: {remedy}\n"
                                f"Allowed Actions: {allowed_act_str}\n"
                                f"You must invoke an authorized tool to diagnose or fix this failure now.{side_effect_caution}"
                            ),
                        }
                        payload.setdefault("messages", []).append(directive_msg)
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
                            turn_tool_history=curr_tool_history,
                            actual_step_count=actual_step_count,
                            ledger=eff_ledger,
                            enable_cycle_detection=eff_cycle_detection,
                            replanning_attempts=replanning_attempts,
                            max_replanning_attempts=eff_max_replanning,
                            task_local_scope=eff_task_local_scope,
                        ):
                            yield retry_chunk
                        return

            # 3. Authoritative Runtime Invariants: Action Intent & Non-Actuation Interception
            if (
                depth == 0
                and not is_retry
                and eff_ledger.total_actions() == 0
                and has_imperative_action_intent(user_query_for_intent)
                and not has_valid_inaction_justification(clean_text)
            ):
                buffered_chunks.clear()
                collected_assistant_text.clear()

                advisory_msg = {
                    "role": "user",
                    "content": (
                        "[SYSTEM NOTICE: Action Not Executed] You proposed steps in text but did not invoke any tools. "
                        "To execute this request on the host, you must emit the appropriate tool call."
                    ),
                }
                payload.setdefault("messages", []).append(advisory_msg)
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
                    turn_tool_history=curr_tool_history,
                    actual_step_count=actual_step_count,
                    ledger=eff_ledger,
                    enable_cycle_detection=eff_cycle_detection,
                    replanning_attempts=replanning_attempts,
                    max_replanning_attempts=eff_max_replanning,
                    task_local_scope=eff_task_local_scope,
                ):
                    yield retry_chunk
                return

            # 4. Goal-Aware Replanning Interception:
            # If the model emitted text without tools while requested outcomes remain unfulfilled,
            # provide a structured, bounded replanning opportunity.
            # Never auto-execute or pop old proposals; require fresh, validated model tool calls.
            if (
                not disable_tools
                and not getattr(self, "_cancelled", False)
                and not payload.get("cancelled", False)
                and stream_finish_reason not in ("length", "max_tokens")
                and depth < eff_max_depth - 1
                and actual_step_count < eff_step_limit
                and not is_abandoned
                and not is_truthful_failure_report(clean_text)
            ):
                from zelyr.engine.execution_ledger import find_unfulfilled_task_requirements

                clean_user_prompt = ""
                for m in payload.get("messages", []):
                    if m.get("role") == "user":
                        c_txt = m.get("content", "")
                        if isinstance(c_txt, list):
                            c_txt = " ".join(part.get("text", "") for part in c_txt if isinstance(part, dict))
                        if any(marker in c_txt for marker in (
                            "[SYSTEM DIRECTIVE:",
                            "[SYSTEM NOTICE:",
                            "[ACTUATOR RECOVERY REQUIRED",
                            "[SYSTEM EVALUATION",
                        )):
                            continue
                        c_clean = re.sub(r"^\[Active Window:[^\]]*\]\s*", "", c_txt).strip()
                        if c_clean:
                            clean_user_prompt = c_clean
                            break

                verified_outcomes, unfulfilled_outcomes = find_unfulfilled_task_requirements(
                    clean_user_prompt,
                    eff_ledger,
                    acceptance_spec=acceptance_spec,
                )

                if had_unconfirmed_proposals and not unfulfilled_outcomes:
                    unfulfilled_outcomes.append("Previously proposed action was unconfirmed and discarded; task remains incomplete")

                if unfulfilled_outcomes and self._replan_budget_available(
                    eff_ledger,
                    replanning_attempts,
                    eff_max_replanning,
                    verified_outcomes=verified_outcomes,
                    unfulfilled_outcomes=unfulfilled_outcomes,
                ):
                    buffered_chunks.clear()
                    collected_assistant_text.clear()

                    if clean_text:
                        payload.setdefault("messages", []).append({
                            "role": "assistant",
                            "content": clean_text,
                        })

                    verified_str = "\n".join(f"- {o}" for o in verified_outcomes) if verified_outcomes else "- None yet verified"
                    unfulfilled_str = "\n".join(f"- {o}" for o in unfulfilled_outcomes)

                    active_tools = eff_tool_schemas or payload.get("tools") or []
                    tool_names = [
                        (t.get("function", {}).get("name") or t.get("name"))
                        for t in active_tools
                        if (t.get("function", {}).get("name") or t.get("name"))
                    ]
                    tools_str = ", ".join(tool_names) if tool_names else "manage_workspace_file, execute_command"

                    active_tg = eff_ledger.get_task_graph() if hasattr(eff_ledger, "get_task_graph") else None
                    tg_section = ""
                    tg_plan_meta: Dict[str, Any] = {}
                    if active_tg is not None:
                        try:
                            active_tg.recompute_schedule(
                                known_proposals=eff_ledger.get_tracked_proposals() if hasattr(eff_ledger, "get_tracked_proposals") else None
                            )
                            next_gn = active_tg.get_next_eligible_node()
                            g_sum = active_tg.summary or {}
                            blk_desc = ", ".join(f"{b.node_id}({b.blocked_reason})" for b in active_tg.get_blocked_nodes()[:5]) or "none"
                            fail_desc_g = ", ".join(f"{f.node_id}" for f in active_tg.get_failed_nodes()[:5]) or "none"
                            tg_section = (
                                f"\n\n5. Task Dependency Graph Status (graph_id={active_tg.graph_id}, status={active_tg.status}):\n"
                                f"- Verified Nodes: {g_sum.get('completed_verified_count', 0)}/{g_sum.get('total_nodes', len(active_tg.nodes))}\n"
                                f"- Next Eligible Node: {next_gn.node_id if next_gn else 'none'}\n"
                                f"- Failed Nodes: {fail_desc_g}\n"
                                f"- Blocked Nodes: {blk_desc}"
                            )
                            tg_plan_meta = {
                                "graph_id": active_tg.graph_id,
                                "graph_status": active_tg.status,
                                "next_eligible_node_id": next_gn.node_id if next_gn else None,
                            }
                        except Exception:
                            tg_section = ""

                    replanning_directive = (
                        "[SYSTEM DIRECTIVE: Goal-Aware Replanning]\n"
                        "The model produced a text response without tool actions, but the original task remains incomplete.\n\n"
                        f"1. Original Task:\n{clean_user_prompt}\n\n"
                        f"2. Independently Verified Outcomes (DO NOT REPEAT OR REGRESS):\n{verified_str}\n\n"
                        f"3. Unfulfilled Task Requirements:\n{unfulfilled_str}\n\n"
                        f"4. Available Actuator Tools:\n{tools_str}"
                        f"{tg_section}\n\n"
                        "CRITICAL INSTRUCTIONS FOR REPLANNING:\n"
                        "- Earlier proposals were NOT executed and do NOT constitute authorization.\n"
                        "- Reassess the current host machine state strictly using the actual observations.\n"
                        "- Do NOT repeat, modify, or regress already verified outcomes.\n"
                        "- If previous operations failed, consider the actual error output.\n"
                        "- If unfulfilled requirements require host machine actions, you must explicitly emit a new, valid tool call now.\n"
                        "- If you determine from observations that no further actions are appropriate or possible, explain why in your response."
                    )
                    payload.setdefault("messages", []).append({
                        "role": "user",
                        "content": replanning_directive,
                    })
                    payload.setdefault("options", {})["temperature"] = 0.1
                    payload["temperature"] = 0.1

                    try:
                        eff_ledger._replan_verified_marker = len(eff_ledger.get_verified_targets())
                        eff_ledger._replan_verified_outcomes = list(verified_outcomes)
                        eff_ledger._replan_unfulfilled_count = len(unfulfilled_outcomes)
                        if getattr(eff_ledger, "_central_ledger", None) and getattr(eff_ledger, "_execution_id", None):
                            eff_ledger._central_ledger.record_event(
                                eff_ledger._execution_id,
                                LedgerEventType.PLANNING_DECISION,
                                status="replanning",
                                metadata={
                                    "replanning_attempt": replanning_attempts + 1,
                                    "unfulfilled_count": len(unfulfilled_outcomes),
                                    "verified_count": len(verified_outcomes),
                                    **tg_plan_meta,
                                },
                                idempotency_key=f"{eff_ledger._execution_id}:replan:{replanning_attempts + 1}:{depth}",
                            )
                    except Exception:
                        pass

                    async for retry_chunk in self.generate_stream(
                        payload,
                        depth=depth + 1,
                        is_retry=True,
                        stream_provider=eff_stream_provider,
                        tool_schemas=eff_tool_schemas,
                        tool_executor=eff_tool_executor,
                        db_path=eff_db_path,
                        max_depth=eff_max_depth,
                        turn_tool_history=curr_tool_history,
                        actual_step_count=actual_step_count,
                        ledger=eff_ledger,
                        enable_cycle_detection=eff_cycle_detection,
                        replanning_attempts=replanning_attempts + 1,
                        max_replanning_attempts=eff_max_replanning,
                        task_local_scope=eff_task_local_scope,
                    ):
                        yield retry_chunk
                    return

        if not pending_tool_calls and buffered_chunks:
            for b in buffered_chunks:
                yield b
            buffered_chunks.clear()

        if pending_tool_calls:
            from zelyr.engine.normalizer import strict_json_loads
            from zelyr.tool_registry import (
                activate_task_local_scope,
                authorize_and_validate_tool_call,
                canonicalize_tool_invocation,
            )

            # Normalize tool calls: Ollama /api/chat requires function.arguments to be a native JSON dict
            normalized_tool_calls = []
            reserved_turn_ids: set = set()
            for idx, tc in enumerate(pending_tool_calls):
                raw_fn_name = tc.get("raw_name") or tc.get("function", {}).get("name") or tc.get("name") or ""
                fn_name = self.normalize_tool_name(
                    raw_fn_name,
                    active_schemas=eff_tool_schemas,
                    task_local_scope=eff_task_local_scope,
                )
                tc_fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                has_explicit_args = tc.get("_has_arguments", None)
                if "arguments" in tc_fn and tc_fn["arguments"] is not None:
                    if has_explicit_args is False and tc_fn["arguments"] == "":
                        fn_args_raw = {}
                    else:
                        fn_args_raw = tc_fn["arguments"]
                elif "arguments" in tc and tc["arguments"] is not None:
                    fn_args_raw = tc["arguments"]
                else:
                    fn_args_raw = {}

                raw_tc_id = tc.get("id") or f"call_native_{idx}_{uuid.uuid4().hex[:6]}"
                tc_id = self._unique_ledger_call_id(eff_ledger, raw_tc_id, reserved_ids=reserved_turn_ids)
                reserved_turn_ids.add(tc_id)

                raw_for_auth: Any = fn_args_raw
                parsed_dict_ok = False
                if isinstance(fn_args_raw, dict):
                    fn_args = fn_args_raw
                    parsed_dict_ok = True
                elif isinstance(fn_args_raw, str):
                    raw_str = fn_args_raw.strip()
                    if not raw_str:
                        fn_args = {}
                    else:
                        try:
                            parsed_raw = strict_json_loads(raw_str)
                            if isinstance(parsed_raw, dict):
                                fn_args = parsed_raw
                                parsed_dict_ok = True
                                raw_for_auth = parsed_raw
                            else:
                                fn_args = {}
                        except Exception:
                            fn_args = {}
                else:
                    fn_args = {}

                # Canonicalize alias rewrites (e.g., interact_with_ui + element_id -> interact_with_browser)
                # BEFORE argument normalization against the canonical schema and BEFORE permission checks.
                if parsed_dict_ok:
                    fn_name, fn_args = canonicalize_tool_invocation(fn_name, fn_args)
                    fn_args = self.normalize_tool_arguments(
                        fn_name, fn_args, original_name=raw_fn_name, active_schemas=eff_tool_schemas
                    )
                    raw_for_auth = fn_args

                tc_entry: Dict[str, Any] = {
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "arguments": fn_args,
                    },
                    "_raw_arguments": raw_for_auth,
                    "from_pending_proposal": bool(
                        tc.get("from_pending_proposal", False) or tc.get("source") == "pending_queue"
                    ),
                }
                normalized_tool_calls.append(tc_entry)

            wire_assistant_tool_calls = [
                {
                    "id": tc["id"],
                    "type": tc["type"],
                    "function": tc["function"],
                }
                for tc in normalized_tool_calls
            ]
            assistant_msg = {
                "role": "assistant",
                "content": "".join(collected_assistant_text) if collected_assistant_text else "",
                "tool_calls": wire_assistant_tool_calls,
            }
            payload["messages"].append(assistant_msg)

            for tc in normalized_tool_calls:
                if tc["id"] not in eff_ledger._records:
                    if hasattr(eff_ledger, "set_call_source_metadata") and self.tool_registry is not None:
                        try:
                            ext_entry = (
                                self.tool_registry.get_external_tool_entry(tc["function"]["name"])
                                if callable(getattr(self.tool_registry, "get_external_tool_entry", None))
                                else None
                            )
                            reg_t = (
                                self.tool_registry.get_tool(tc["function"]["name"])
                                if callable(getattr(self.tool_registry, "get_tool", None))
                                else None
                            )
                            is_ext = bool(ext_entry is not None or (reg_t is not None and getattr(reg_t, "is_external", False)))
                            src_id = (
                                getattr(ext_entry, "source_id", None)
                                if ext_entry is not None
                                else (getattr(reg_t, "source_id", None) if reg_t is not None else "native")
                            )
                            src_type = (
                                getattr(ext_entry, "source_type", None)
                                if ext_entry is not None
                                else (getattr(reg_t, "source_type", None) if reg_t is not None else "native")
                            )
                            ext_tname = (
                                getattr(ext_entry, "external_tool_name", None)
                                if ext_entry is not None
                                else (getattr(reg_t, "external_tool_name", None) if reg_t is not None else None)
                            )
                            eff_ledger.set_call_source_metadata(
                                tc["id"],
                                is_external=is_ext,
                                source_id=src_id,
                                source_type=src_type,
                                extra={"external_tool_name": ext_tname} if ext_tname else {},
                            )
                        except Exception:
                            pass
                    extra_tc_meta: Dict[str, Any] = {}
                    for cand_ptc in pending_tool_calls:
                        if cand_ptc.get("id") and cand_ptc.get("id") == tc["id"]:
                            extra_tc_meta = cand_ptc
                            break
                    if not extra_tc_meta and len(pending_tool_calls) == 1 and len(normalized_tool_calls) == 1:
                        extra_tc_meta = pending_tool_calls[0]
                    eff_prop_id = tc.get("proposal_id") or extra_tc_meta.get("proposal_id")
                    eff_deps = (
                        tc.get("depends_on")
                        if isinstance(tc.get("depends_on"), list)
                        else (extra_tc_meta.get("depends_on") if isinstance(extra_tc_meta.get("depends_on"), list) else None)
                    )
                    eff_exp = tc.get("expires_at") if tc.get("expires_at") is not None else extra_tc_meta.get("expires_at")
                    eff_ext = (
                        tc.get("external_source")
                        if isinstance(tc.get("external_source"), dict)
                        else (extra_tc_meta.get("external_source") if isinstance(extra_tc_meta.get("external_source"), dict) else None)
                    )
                    if eff_prop_id:
                        tc["proposal_id"] = eff_prop_id
                    if eff_deps is not None:
                        tc["depends_on"] = list(eff_deps)
                    if eff_exp is not None:
                        tc["expires_at"] = eff_exp
                    if eff_ext is not None:
                        tc["external_source"] = dict(eff_ext)
                    if extra_tc_meta.get("node_id") and not tc.get("node_id"):
                        tc["node_id"] = extra_tc_meta["node_id"]
                    eff_ledger.record_proposed(
                        tc["id"],
                        tc["function"]["name"],
                        tc["function"]["arguments"],
                        turn=depth,
                        proposal_id=eff_prop_id,
                        depends_on=eff_deps,
                        expires_at=eff_exp,
                        external_source=eff_ext,
                    )

            force_disable_next = False
            for tc in normalized_tool_calls:
                func_name = tc["function"]["name"]
                func_args = tc["function"]["arguments"]
                tool_call_id = tc["id"]

                # Fingerprint tool invocation: canonical JSON args
                canonical_args = self._canonicalize_tool_args(func_args)
                fingerprint = (func_name, canonical_args)

                # Authoritative runtime permission, known-tool, and argument-schema validation
                # applied identically to native, pending, and fallback tool calls before execution.
                rejection_reason = None
                rejection_detail = None
                decision = None

                if eff_task_local_scope is not None and eff_tool_executor is None:
                    rejection_reason = "authorization_denied"
                    rejection_detail = (
                        "AuthorizationError: TaskLocalToolScope requires an isolated custom "
                        "tool_executor and cannot dispatch to production execute_tool."
                    )
                else:
                    auth_schemas = eff_tool_schemas if eff_tool_schemas is not None else payload.get("tools")
                    known_for_auth = (
                        self.get_known_tool_names(list(eff_tool_schemas))
                        if (eff_tool_schemas is not None and has_valid_schemas_list and eff_task_local_scope is None)
                        else None
                    )
                    raw_auth_args = tc["_raw_arguments"] if "_raw_arguments" in tc else func_args
                    decision = authorize_and_validate_tool_call(
                        func_name,
                        raw_auth_args,
                        caller_ring=eff_caller_ring,
                        allowed_tools=eff_allowed_tools,
                        active_schemas=auth_schemas,
                        tool_registry=self.tool_registry,
                        known_tools=known_for_auth,
                        task_local_scope=eff_task_local_scope,
                        require_registered=True,
                        normalize_args=True,
                        validate_schema=True,
                        argument_source="model",
                    )
                    if not decision.allowed:
                        rejection_reason = decision.reason or "authorization_denied"
                        rejection_detail = decision.detail or "Tool invocation denied by runtime authorization boundary."
                    else:
                        func_name = decision.tool_name
                        func_args = decision.normalized_arguments
                        tc["function"]["name"] = func_name
                        tc["function"]["arguments"] = func_args
                        fingerprint = (func_name, self._canonicalize_tool_args(func_args))

                eff_step_limit = getattr(self, "max_steps", 5)
                if rejection_reason is None and actual_step_count >= eff_step_limit:
                    rejection_reason = "step_limit_reached"
                    rejection_detail = f"Maximum execution step limit ({eff_step_limit}) reached for current turn. Explain the limitation and synthesize the answer."
                    force_disable_next = True
                elif rejection_reason is None and eff_cycle_detection:
                    all_user_prompts = " ".join(
                        (m.get("content") if isinstance(m.get("content"), str) else "")
                        for m in payload.get("messages", [])
                        if m.get("role") == "user" and not str(m.get("content", "")).startswith("[SYSTEM")
                    )
                    is_explicit_repeat = self._is_explicit_repeat_requested(all_user_prompts)

                    # Check if action is an idempotent read/inspection check (allowed to poll up to ceiling)
                    func_lower = func_name.lower()
                    is_read_tool = (
                        func_lower.startswith(("read_", "get_", "list_", "check_", "inspect_", "verify_", "stat_"))
                        or func_lower in ("get_page_snapshot", "list_tabs", "read_file", "view_file", "list_dir", "list_tree", "read_state", "verify_outcome")
                        or (func_lower == "manage_workspace_file" and str(func_args.get("action", "")).lower() in ("read", "view", "list", "list_tree", "stat", "exists", "inspect", "check"))
                    )

                    has_prior_success = (
                        (curr_tool_history and curr_tool_history[-1][0] == fingerprint and curr_tool_history[-1][1] == "success")
                        or (hasattr(eff_ledger, "has_identical_successful_action") and eff_ledger.has_identical_successful_action(func_name, canonical_args))
                    )

                    # Check if action mutates any already-verified resource
                    from zelyr.engine.execution_ledger import (
                        _is_inspection_action,
                        canonicalize_target,
                        criterion_applies_to_target,
                        extract_file_targets_from_text,
                        has_remaining_explicit_steps,
                        parse_acceptance_criteria,
                        targets_match,
                    )
                    from zelyr.engine.verifier import OutcomeVerifier

                    mutated_targets = self._get_mutation_targets(
                        func_name, func_args, eff_ledger.get_verified_targets()
                    )

                    target_already_verified = False
                    verified_target_hit = None
                    parsed_c = parse_acceptance_criteria(acceptance_spec) if acceptance_spec else []
                    all_c_targets = [c.get("target", "") for c in parsed_c if c.get("target")]
                    for mt in mutated_targets:
                        matching_c = [
                            c for c in parsed_c
                            if c.get("target") and criterion_applies_to_target(c, mt, all_c_targets)
                        ]
                        if matching_c:
                            if all(OutcomeVerifier.verify(check=c["check"], target=c["target"], expected=c.get("expected")).success for c in matching_c):
                                target_already_verified = True
                                verified_target_hit = mt
                                eff_ledger.record_target_verified(mt, check_type=matching_c[0].get("check"))
                                break
                            else:
                                if hasattr(eff_ledger, "invalidate_target_verification"):
                                    eff_ledger.invalidate_target_verification(mt, check_type=matching_c[0].get("check"))
                        elif eff_ledger.is_target_verified(mt):
                            target_already_verified = True
                            verified_target_hit = mt
                            break

                    # Check authorizations: explicit repeat, sequential step, recovery
                    if func_name == "manage_workspace_file":
                        action_param = str(func_args.get("action", "")).lower()
                    elif func_name in ("execute_command", "bash", "sh"):
                        cmd_str = str(func_args.get("command", "") or func_args.get("cmd", ""))
                        if ">>" in cmd_str:
                            action_param = "append"
                        elif ">" in cmd_str or "| tee" in cmd_str:
                            action_param = "write"
                        elif any(op in cmd_str for op in ("rm ", "unlink ", "truncate ")):
                            action_param = "delete"
                        else:
                            action_param = func_name
                    else:
                        action_param = func_name
                    is_authorized_sequential = (
                        target_already_verified
                        and verified_target_hit is not None
                        and self._is_sequential_step_requested(
                            all_user_prompts,
                            verified_target_hit,
                            action_param,
                            eff_ledger,
                        )
                    )

                    if target_already_verified and not is_explicit_repeat and not is_authorized_sequential and not is_read_tool:
                        has_unresolved_on_target = any(
                            (rec.state == ActionState.FAILED or getattr(rec, "verification_failed", False))
                            and self._extract_target_from_args(rec.tool_name, rec.arguments)
                            and targets_match(self._extract_target_from_args(rec.tool_name, rec.arguments), verified_target_hit)
                            for rec in eff_ledger.unresolved_failures()
                        )
                        if not has_unresolved_on_target:
                            rejection_reason = "outcome_already_verified"
                            rejection_detail = (
                                f"Target resource '{verified_target_hit}' already satisfies the verified acceptance criteria. "
                                f"Mutations via '{func_name}' are blocked to prevent regressing verified outcomes. "
                                "Synthesize your final response or proceed to uncompleted tasks."
                            )
                            if not has_remaining_explicit_steps(all_user_prompts, eff_ledger, current_target=verified_target_hit):
                                force_disable_next = True
                    elif (
                        not is_explicit_repeat
                        and not is_read_tool
                        and not getattr(eff_ledger, "_allow_unverified_retry", False)
                        and fingerprint in (getattr(eff_ledger, "_ambiguous_fingerprints", None) or ())
                    ):
                        rejection_reason = "ambiguous_in_flight_operation"
                        rejection_detail = (
                            f"Action '{func_name}' was previously dispatched before interruption without a persisted result. "
                            "Do not blindly replay ambiguous mutating actions; inspect or verify host state first."
                        )
                    elif (
                        not is_explicit_repeat
                        and not is_read_tool
                        and fingerprint in (getattr(eff_ledger, "_resumed_verified_fingerprints", None) or ())
                    ):
                        rejection_reason = "outcome_already_verified"
                        rejection_detail = (
                            f"Action '{func_name}' was already completed and verified prior to interruption; "
                            "skip repeating this verified operation."
                        )
                    elif has_prior_success and not is_explicit_repeat and not is_read_tool:
                        rejection_reason = "redundant_action"
                        rejection_detail = (
                            f"Action '{func_name}' with identical parameters has already succeeded. "
                            "Do not repeat completed actions; provide the final response or proceed to the next distinct step."
                        )
                    elif curr_tool_history and curr_tool_history[-1][0] == fingerprint and curr_tool_history[-1][1] == "error":
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

                # Enforce durable pending-proposal runtime guards (expiration, blocked/completed state, dependencies)
                if rejection_reason is None:
                    tc_exp = tc.get("expires_at")
                    if tc_exp is not None and not isinstance(tc_exp, bool) and time.time() >= float(tc_exp):
                        rejection_reason = "proposal_expired"
                        rejection_detail = f"Pending proposal '{tool_call_id}' ({func_name}) expired prior to dispatch."
                        if hasattr(eff_ledger, "_update_proposal_lifecycle"):
                            eff_ledger._update_proposal_lifecycle(tool_call_id, "expired", blocked_reason=rejection_detail)
                    else:
                        guarded_map = payload.get("_guarded_pending_proposals") if isinstance(payload.get("_guarded_pending_proposals"), dict) else {}
                        g_entry = guarded_map.get(tool_call_id) or guarded_map.get(tc.get("proposal_id") or "")
                        if isinstance(g_entry, dict):
                            g_st = str(g_entry.get("status") or "")
                            if g_st == "dispatched" and not getattr(eff_ledger, "_allow_unverified_retry", False):
                                rejection_reason = "ambiguous_in_flight_operation"
                                rejection_detail = (
                                    f"Proposal '{g_entry.get('proposal_id') or tool_call_id}' ({func_name}) is already dispatched without result; inspect host state before retrying."
                                )
                            elif g_st == "completed" and not is_read_tool:
                                rejection_reason = "outcome_already_verified"
                                rejection_detail = f"Proposal '{g_entry.get('proposal_id') or tool_call_id}' ({func_name}) already completed prior to resume."
                            elif g_st in ("expired", "blocked", "cancelled", "failed"):
                                rejection_reason = f"proposal_{g_st}"
                                rejection_detail = str(g_entry.get("blocked_reason") or f"Proposal is {g_st} and cannot dispatch.")

                if rejection_reason is None:
                    tc_deps = tc.get("depends_on") if isinstance(tc.get("depends_on"), list) else []
                    if not tc_deps and hasattr(eff_ledger, "get_proposal_metadata"):
                        pm = eff_ledger.get_proposal_metadata(tool_call_id)
                        if isinstance(pm, dict) and isinstance(pm.get("depends_on"), list):
                            tc_deps = list(pm["depends_on"])
                    if tc_deps:
                        tracked_props = eff_ledger.get_tracked_proposals() if hasattr(eff_ledger, "get_tracked_proposals") else []
                        for dep_ref in tc_deps:
                            dep_matches = [
                                tp
                                for tp in tracked_props
                                if str(dep_ref) in (str(tp.get("proposal_id") or ""), str(tp.get("original_call_id") or ""))
                            ]
                            dep_rec = eff_ledger._records.get(dep_ref)
                            if dep_rec is None:
                                for tp in dep_matches:
                                    cand_rec = eff_ledger._records.get(str(tp.get("original_call_id") or ""))
                                    if cand_rec is not None:
                                        dep_rec = cand_rec
                                        if cand_rec.state in (
                                            ActionState.VERIFIED,
                                            ActionState.SUCCESS,
                                            ActionState.OBSERVED,
                                            ActionState.UNVERIFIED,
                                        ):
                                            break
                            dep_ok = (
                                dep_rec is not None
                                and dep_rec.state in (
                                    ActionState.VERIFIED,
                                    ActionState.SUCCESS,
                                    ActionState.OBSERVED,
                                    ActionState.UNVERIFIED,
                                )
                            ) or any(tp.get("status") == "completed" for tp in dep_matches)
                            if dep_ok:
                                continue
                            has_failed_match = (
                                dep_rec is not None
                                and dep_rec.state in (ActionState.FAILED, ActionState.REJECTED)
                            ) or (
                                dep_rec is not None
                                and dep_rec.state == ActionState.CANCELLED
                                and "reconfirmed_and_dispatched" not in str(dep_rec.error or "")
                            ) or any(
                                tp.get("status") in ("failed", "blocked", "cancelled", "expired")
                                and "reconfirmed_and_dispatched" not in str(tp.get("blocked_reason") or "")
                                for tp in dep_matches
                            )
                            if has_failed_match:
                                rejection_reason = "dependency_blocked"
                                rejection_detail = f"Prerequisite '{dep_ref}' failed/blocked; dependent call '{func_name}' cannot execute."
                            else:
                                rejection_reason = "dependency_not_satisfied"
                                rejection_detail = f"Prerequisite '{dep_ref}' has not completed yet; execute/verify prerequisite before '{func_name}'."
                            break

                # Enforce TaskGraph prerequisite verification & node readiness before dispatch
                active_tg_step = eff_ledger.get_task_graph() if hasattr(eff_ledger, "get_task_graph") else None
                bound_tg_nodes: List[Any] = []
                if active_tg_step is not None:
                    from zelyr.engine.task_graph import TaskGraphValidationError, TaskNodeStatus

                    try:
                        tracked_for_tg = eff_ledger.get_tracked_proposals() if hasattr(eff_ledger, "get_tracked_proposals") else []
                        active_tg_step.recompute_schedule(known_proposals=tracked_for_tg)
                        active_tg_step.validate(known_proposals=tracked_for_tg, allow_blocked=True)
                    except TaskGraphValidationError as tg_exc:
                        rejection_reason = "invalid_task_graph"
                        rejection_detail = str(tg_exc)
                    else:
                        bound_tg_nodes = active_tg_step.find_nodes_for_call(tool_call_id, proposal_id=tc.get("proposal_id"))
                        explicit_nid = tc.get("node_id")
                        if explicit_nid:
                            exp_node = active_tg_step.get_node(str(explicit_nid))
                            if exp_node is not None and exp_node not in bound_tg_nodes:
                                bound_tg_nodes.append(exp_node)
                        if rejection_reason is None:
                            for gn in bound_tg_nodes:
                                if gn.execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value:
                                    rejection_reason = "outcome_already_verified"
                                    rejection_detail = f"TaskGraph node '{gn.node_id}' is already completed and verified."
                                    break
                                if gn.execution_status in (TaskNodeStatus.PENDING.value, TaskNodeStatus.BLOCKED.value):
                                    rejection_reason = (
                                        "dependency_blocked"
                                        if gn.execution_status == TaskNodeStatus.BLOCKED.value
                                        else "dependency_not_satisfied"
                                    )
                                    rejection_detail = gn.blocked_reason or (
                                        f"TaskGraph node '{gn.node_id}' cannot execute until prerequisite dependencies are independently verified."
                                    )
                                    break
                                if gn.execution_status in (TaskNodeStatus.FAILED.value, TaskNodeStatus.CANCELLED.value):
                                    rejection_reason = f"node_{gn.execution_status}"
                                    rejection_detail = gn.blocked_reason or f"TaskGraph node '{gn.node_id}' is {gn.execution_status} and cannot execute."
                                    break

                # Runtime-enforced user prohibitions: an explicit "do not create/write/delete <path>"
                # in the user's own request cannot be overridden by model-proposed tool calls.
                if rejection_reason is None:
                    from zelyr.engine.execution_ledger import targets_match
                    _user_text = " ".join(
                        (m.get("content") if isinstance(m.get("content"), str) else "")
                        for m in payload.get("messages", [])
                        if m.get("role") == "user" and not str(m.get("content", "")).startswith("[SYSTEM")
                    )
                    prohibited_targets = self._extract_prohibited_targets(_user_text, ledger=eff_ledger)
                    if prohibited_targets:
                        prohibited_hit = self._get_mutation_targets(
                            func_name,
                            func_args,
                            prohibited_targets,
                            allow_dir_containment=True,
                        )
                        prohibited_hit = [
                            t for t in prohibited_hit
                            if any(
                                targets_match(t, p, prohibited_targets, allow_dir_containment=True)
                                for p in prohibited_targets
                            )
                        ]
                        if prohibited_hit:
                            rejection_reason = "prohibited_by_user"
                            rejection_detail = (
                                f"The user explicitly prohibited mutating '{prohibited_hit[0]}'. "
                                f"The call to '{func_name}' is blocked; do not retry it."
                            )

                if rejection_reason is not None:
                    actual_step_count += 1
                    if actual_step_count >= eff_step_limit:
                        force_disable_next = True
                    synth_call_id = tool_call_id or f"call_synth_{uuid.uuid4().hex[:8]}"
                    err_payload = {
                        "tool": func_name,
                        "status": "rejected",
                        "reason": rejection_reason,
                        "detail": rejection_detail,
                        "error_type": (
                            decision.error_type
                            if (decision is not None and not decision.allowed)
                            else ("AUTHORIZATION_DENIED" if decision is None else rejection_reason.upper())
                        ),
                    }
                    if decision is not None and not decision.allowed and isinstance(getattr(decision, "validation_error", None), dict):
                        for k, v in decision.validation_error.items():
                            if k not in err_payload:
                                err_payload[k] = v
                    synth_content = json.dumps(err_payload)
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": synth_call_id,
                        "name": func_name,
                        "content": synth_content,
                    }
                    payload["messages"].append(tool_msg)
                    curr_tool_history.append((fingerprint, "rejected"))
                    if getattr(eff_ledger, "_central_ledger", None) and getattr(eff_ledger, "_execution_id", None):
                        from zelyr.engine.execution_ledger import LedgerEventType

                        c_id = eff_ledger._execution_id
                        is_schema_err = err_payload.get("error_type") == "SCHEMA_VALIDATION_ERROR"
                        eff_ledger._central_ledger.record_event(
                            c_id,
                            LedgerEventType.ARGUMENT_VALIDATED if is_schema_err else LedgerEventType.AUTHORIZATION_DECISION,
                            tool_id=func_name,
                            status="invalid" if is_schema_err else "denied",
                            correlation_id=tool_call_id,
                            caller_ring=eff_caller_ring,
                            task_local_scoped=bool(eff_task_local_scope),
                            metadata={"reason": rejection_reason, "detail": rejection_detail},
                            idempotency_key=f"{c_id}:{tool_call_id}:{'validated' if is_schema_err else 'auth'}",
                        )
                    from zelyr.engine.observation_framing import ObservationFramingEngine

                    rej_obs = ObservationFramingEngine.frame_structured(
                        func_name,
                        err_payload,
                        requested_tool=func_name,
                        requested_args=func_args,
                        executed_args=func_args,
                        dispatched=False,
                        execution_id=getattr(eff_ledger, "_execution_id", None),
                        correlation_id=tool_call_id,
                    )
                    tool_msg["evidence_level"] = rej_obs.evidence_level
                    tool_msg["execution_status"] = rej_obs.execution_status
                    eff_ledger.record_dispatched(tool_call_id, emit_event=False)
                    eff_ledger.record_observation(
                        tool_call_id,
                        status="rejected",
                        output=synth_content,
                        error=f"[{rejection_reason}] {rejection_detail}",
                        rejection_reason=rejection_reason,
                        structured_observation=rej_obs,
                    )
                    if active_tg_step is not None and bound_tg_nodes and rejection_reason not in (
                        "dependency_not_satisfied",
                        "dependency_blocked",
                        "outcome_already_verified",
                        "node_failed",
                        "node_cancelled",
                    ):
                        for gn in bound_tg_nodes:
                            try:
                                if gn.execution_status not in ("completed_verified", "blocked", "cancelled"):
                                    active_tg_step.apply_observation(
                                        gn.node_id,
                                        rej_obs,
                                        action_record=eff_ledger._records.get(tool_call_id),
                                        known_proposals=eff_ledger.get_tracked_proposals() if hasattr(eff_ledger, "get_tracked_proposals") else None,
                                    )
                            except Exception:
                                pass
                    continue

                if active_tg_step is not None and bound_tg_nodes:
                    from zelyr.engine.task_graph import TaskNodeStatus

                    for gn in bound_tg_nodes:
                        if gn.execution_status == TaskNodeStatus.READY.value:
                            gn.execution_status = TaskNodeStatus.RUNNING.value
                            gn.updated_at = time.time()

                actual_step_count += 1
                if actual_step_count >= eff_step_limit:
                    force_disable_next = True
                if getattr(eff_ledger, "_central_ledger", None) and getattr(eff_ledger, "_execution_id", None):
                    from zelyr.engine.execution_ledger import LedgerEventType

                    eff_ledger._central_ledger.record_event(
                        eff_ledger._execution_id,
                        LedgerEventType.AUTHORIZATION_DECISION,
                        tool_id=func_name,
                        status="allowed",
                        correlation_id=tool_call_id,
                        caller_ring=eff_caller_ring,
                        task_local_scoped=bool(eff_task_local_scope),
                        idempotency_key=f"{eff_ledger._execution_id}:{tool_call_id}:auth",
                    )
                eff_ledger.record_validated(tool_call_id)
                eff_ledger.record_dispatched(tool_call_id)
                timeout = get_tier_timeout(func_name)
                timeout_int = int(timeout)
                from contextlib import nullcontext
                from zelyr.engine.execution_ledger import execution_scope

                ledger_scope_cm = (
                    execution_scope(
                        eff_ledger._central_ledger,
                        eff_ledger._execution_id,
                        caller_ring=eff_caller_ring,
                        task_local_scoped=bool(eff_task_local_scope),
                        source_component="native_orchestrator",
                        correlation_id=tool_call_id,
                        owned_by_outer=True,
                    )
                    if (getattr(eff_ledger, "_central_ledger", None) and getattr(eff_ledger, "_execution_id", None))
                    else nullcontext(None)
                )
                was_timed_out = False
                was_cancelled = False
                raw_tool_payload: Any = None
                ext_src_for_frame: Optional[Dict[str, Any]] = None
                try:
                    with ledger_scope_cm as call_ctx:
                        if eff_tool_executor is not None:
                            with activate_task_local_scope(
                                eff_task_local_scope if is_valid_task_scope else None
                            ):
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
                            exec_fn = self.execute_tool
                            dispatched_via_instance_registry = False
                            if exec_fn is execute_tool and self.tool_registry is not None and callable(getattr(self.tool_registry, "execute_async", None)):
                                try:
                                    from zelyr.core.plugins import _executors as _plugin_executors
                                    reg_tool = self.tool_registry.get_tool(func_name) if callable(getattr(self.tool_registry, "get_tool", None)) else None
                                    if reg_tool is not None and (
                                        getattr(reg_tool, "is_external", False) or func_name not in _plugin_executors
                                    ):
                                        dispatched_via_instance_registry = True
                                except Exception:
                                    reg_tool = None
                            if dispatched_via_instance_registry:
                                exec_coro = self.tool_registry.execute_async(
                                    func_name,
                                    caller_ring=eff_caller_ring,
                                    allowed_tools=eff_allowed_tools,
                                    **func_args,
                                )
                            elif exec_fn is execute_tool and (eff_caller_ring is not None or eff_allowed_tools is not None):
                                exec_coro = exec_fn(
                                    func_name,
                                    caller_ring=eff_caller_ring,
                                    allowed_tools=eff_allowed_tools,
                                    **func_args,
                                )
                            else:
                                exec_coro = exec_fn(func_name, **func_args)
                            tool_result = await asyncio.wait_for(
                                exec_coro,
                                timeout=timeout,
                            )

                    extra_telemetry: Dict[str, Any] = {}
                    raw_res_meta = getattr(tool_result, "metadata", None)
                    if isinstance(raw_res_meta, dict):
                        extra_telemetry.update(raw_res_meta)
                    bag = getattr(call_ctx, "telemetry_bag", None)
                    if isinstance(bag, dict) and bag:
                        extra_telemetry.update(bag)
                    if extra_telemetry.get("is_external") or extra_telemetry.get("external") or extra_telemetry.get("source_id"):
                        ext_src_for_frame = {
                            "is_external": True,
                            "source_id": extra_telemetry.get("source_id") or "external",
                            "source_type": extra_telemetry.get("source_type") or "external",
                            "external_tool_name": extra_telemetry.get("external_tool_name") or func_name,
                        }
                    if extra_telemetry and hasattr(eff_ledger, "set_call_source_metadata"):
                        eff_ledger.set_call_source_metadata(
                            tool_call_id,
                            is_external=extra_telemetry.get("is_external", extra_telemetry.get("external")),
                            source_id=extra_telemetry.get("source_id"),
                            source_type=extra_telemetry.get("source_type"),
                            extra=extra_telemetry,
                        )

                    if hasattr(tool_result, "to_dict"):
                        tool_dict = tool_result.to_dict(tool=func_name, arguments=func_args)
                        raw_tool_payload = tool_dict
                        tool_result = json.dumps(tool_dict, indent=2)
                    elif not isinstance(tool_result, str):
                        raw_tool_payload = tool_result
                        tool_result = json.dumps(tool_result, indent=2)
                    else:
                        raw_tool_payload = tool_result
                    res_str = tool_result
                except asyncio.CancelledError:
                    was_cancelled = True
                    from zelyr.engine.observation_framing import ObservationFramingEngine

                    cancel_payload = {
                        "success": False,
                        "status": "cancelled",
                        "error": f"Tool '{func_name}' execution was cancelled.",
                        "error_type": "CANCELLED",
                    }
                    cancel_obs = ObservationFramingEngine.frame_structured(
                        func_name,
                        cancel_payload,
                        requested_tool=func_name,
                        requested_args=func_args,
                        executed_args=func_args,
                        dispatched=True,
                        cancelled=True,
                        execution_id=getattr(eff_ledger, "_execution_id", None),
                        correlation_id=tool_call_id,
                        external_source=ext_src_for_frame,
                    )
                    eff_ledger.record_observation(
                        tool_call_id,
                        status="error",
                        output="",
                        error=cancel_payload["error"],
                        exit_code=1,
                        structured_observation=cancel_obs,
                    )
                    if active_tg_step is not None and bound_tg_nodes:
                        for gn in bound_tg_nodes:
                            try:
                                active_tg_step.apply_observation(
                                    gn.node_id,
                                    cancel_obs,
                                    action_record=eff_ledger._records.get(tool_call_id),
                                    known_proposals=eff_ledger.get_tracked_proposals() if hasattr(eff_ledger, "get_tracked_proposals") else None,
                                )
                            except Exception:
                                pass
                    raise
                except (TimeoutError, asyncio.TimeoutError):
                    was_timed_out = True
                    if hasattr(eff_ledger, "_root_meta") and isinstance(eff_ledger._root_meta, dict):
                        eff_ledger._root_meta["timed_out"] = True
                        eff_ledger._root_meta["error"] = f"Tool execution timed out after {timeout_int}s"
                    if hasattr(eff_ledger, "set_call_source_metadata"):
                        eff_ledger.set_call_source_metadata(tool_call_id, extra={"timed_out": True, "timeout_s": timeout_int})
                    raw_tool_payload = {
                        "success": False,
                        "status": "timed_out",
                        "error": f"Tool execution timed out after {timeout_int}s. Try a faster Tier 1 command or verify active window.",
                        "error_type": "COMMAND_TIMEOUT",
                        "remedy_hint": "Try a faster Tier 1 command or verify active window.",
                        "allowed_actions": ["execute_command", "manage_desktop_window"],
                    }
                    res_str = json.dumps(raw_tool_payload)
                except Exception as e:
                    raw_tool_payload = {
                        "success": False,
                        "error": str(e),
                        "status": "tool_execution_failed",
                        "remedy_hint": "Inspect error or try alternative action.",
                        "allowed_actions": [],
                    }
                    res_str = json.dumps(raw_tool_payload)

                from zelyr.engine.observation_framing import ObservationFramingEngine

                structured_obs = ObservationFramingEngine.frame_structured(
                    func_name,
                    raw_tool_payload if raw_tool_payload is not None else res_str,
                    requested_tool=func_name,
                    requested_args=func_args,
                    executed_args=func_args,
                    dispatched=True,
                    timed_out=was_timed_out,
                    cancelled=was_cancelled,
                    execution_id=getattr(eff_ledger, "_execution_id", None),
                    correlation_id=tool_call_id,
                    external_source=ext_src_for_frame,
                )

                # Format tool output with structured status metadata and remediation hints
                fmt_fn = self.format_tool_observation
                if getattr(fmt_fn, "__func__", None) is getattr(NativeOrchestrator.format_tool_observation, "__func__", None):
                    content_for_msg = ObservationFramingEngine.frame_observation(
                        func_name,
                        res_str,
                        structured=structured_obs,
                        execution_id=getattr(eff_ledger, "_execution_id", None),
                        correlation_id=tool_call_id,
                        external_source=ext_src_for_frame,
                    )
                else:
                    content_for_msg = fmt_fn(func_name, res_str)

                obs_dict = ObservationFramingEngine.parse_observation(content_for_msg)

                is_failed = (obs_dict.get("status") == "error" or (obs_dict.get("exit_code") is not None and obs_dict.get("exit_code") != 0))
                remedy_hint_out = str(obs_dict.get("remedy_hint") or "")
                allowed_actions_out = list(obs_dict.get("allowed_actions") or [])
                error_type_out = str(obs_dict.get("error_type") or "")
                is_verified = (obs_dict.get("verified") is True)
                verification_detail = obs_dict.get("verification_detail") or obs_dict.get("summary")

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

                tool_target = self._extract_target_from_args(func_name, func_args)
                criteria_failed = False
                failed_criteria_records: List[Dict[str, Any]] = []
                verified_targets_for_obs: List[str] = []
                failed_targets_for_obs: List[str] = []
                if acceptance_spec and not is_failed:
                    from zelyr.engine.execution_ledger import (
                        _is_inspection_action,
                        canonicalize_target,
                        criterion_applies_to_target,
                        extract_file_targets_from_text,
                        has_remaining_explicit_steps,
                        parse_acceptance_criteria,
                        targets_match,
                    )
                    from zelyr.engine.verifier import OutcomeVerifier
                    parsed_c = parse_acceptance_criteria(acceptance_spec)
                    all_c_targets = [c.get("target", "") for c in parsed_c if c.get("target")]
                    c_results = []
                    by_target: Dict[str, List[Any]] = {}
                    for c in parsed_c:
                        vr = OutcomeVerifier.verify(check=c["check"], target=c["target"], expected=c.get("expected"))
                        c_results.append((c, vr))
                        norm_ct = canonicalize_target(c.get("target", ""), check_type=c.get("check")) if c.get("target") else ""
                        if norm_ct:
                            by_target.setdefault(norm_ct, []).append((c, vr))
                        if str(c.get("check", "")).lower() in ("command_output", "command", "cmd") and c.get("target"):
                            for ref_ft in extract_file_targets_from_text(str(c["target"])):
                                norm_ft = canonicalize_target(ref_ft)
                                if norm_ft:
                                    by_target.setdefault(norm_ft, []).append(
                                        ({"check": "file_exists", "target": ref_ft, "expected": c.get("expected")}, vr)
                                    )

                    for norm_ct, items in by_target.items():
                        first_c = items[0][0]
                        if all(r.success for _, r in items):
                            eff_ledger.record_target_verified(
                                first_c["target"],
                                detail=f"{first_c['check']} verified",
                                check_type=first_c.get("check"),
                            )
                        else:
                            if hasattr(eff_ledger, "invalidate_target_verification"):
                                eff_ledger.invalidate_target_verification(
                                    first_c["target"],
                                    check_type=first_c.get("check"),
                                )

                    candidate_targets: List[str] = [tool_target] if tool_target else []
                    if func_name in ("execute_command", "bash", "sh"):
                        for mt_cand in self._get_mutation_targets(func_name, func_args, all_c_targets):
                            if mt_cand not in candidate_targets:
                                candidate_targets.append(mt_cand)
                        if candidate_targets and not tool_target:
                            tool_target = candidate_targets[0]

                    matching_pairs = []
                    if candidate_targets:
                        matching_pairs = [
                            (c, r) for c, r in c_results
                            if c.get("target") and any(
                                criterion_applies_to_target(c, ct, all_c_targets)
                                for ct in candidate_targets
                            )
                        ]

                    if (
                        not matching_pairs
                        and func_name in ("execute_command", "bash", "sh")
                        and not _is_inspection_action(func_name, func_args)
                    ):
                        cmd_str = str(func_args.get("command", "") or func_args.get("cmd", ""))
                        matching_pairs = [
                            (c, r) for c, r in c_results
                            if (
                                str(c.get("check", "")).lower() in ("command_output", "command", "cmd")
                                or (c.get("target") and Path(str(c["target"])).name and Path(str(c["target"])).name in cmd_str)
                            )
                        ]
                        if matching_pairs and not tool_target and matching_pairs[0][0].get("target"):
                            tool_target = str(matching_pairs[0][0]["target"])

                    is_pure_inspection = (
                        _is_inspection_action(func_name, func_args)
                        and func_name != "verify_outcome"
                    )

                    if matching_pairs and all(r.success for _, r in matching_pairs):
                        if not is_pure_inspection:
                            is_verified = True
                            criteria_failed = False
                            verified_targets_for_obs = [
                                canonicalize_target(str(c.get("target")), check_type=c.get("check"))
                                for c, _ in matching_pairs
                                if c.get("target")
                            ]
                            verification_detail = f"Verified: {matching_pairs[0][0]['check']} passed on {tool_target}"
                    elif matching_pairs and not is_pure_inspection:
                        is_verified = False
                        criteria_failed = True
                        for c, r in matching_pairs:
                            if not r.success:
                                chk = c.get("check", "check")
                                tgt = c.get("target") or tool_target
                                exp = c.get("expected")
                                reason = r.error or "Observed state did not match expected criteria"
                                crit_str = f"{chk} on {tgt}" + (f" (expected '{exp}')" if exp is not None else "")
                                req_str = f"Satisfy {chk} on {tgt}" + (f" with expected '{exp}'" if exp is not None else "")
                                if tgt:
                                    canon_ft = canonicalize_target(str(tgt), check_type=chk)
                                    if canon_ft and canon_ft not in failed_targets_for_obs:
                                        failed_targets_for_obs.append(canon_ft)
                                failed_criteria_records.append({
                                    "target": tgt,
                                    "check": chk,
                                    "expected": exp,
                                    "criterion": crit_str,
                                    "reason": reason,
                                    "requirement": req_str,
                                })
                        if failed_criteria_records:
                            first_fc = failed_criteria_records[0]
                            verification_detail = (
                                f"Verification failed for target '{first_fc['target']}': "
                                f"criterion '{first_fc['criterion']}' not satisfied ({first_fc['reason']}). "
                                f"Required: {first_fc['requirement']}."
                            )

                if func_name == "verify_outcome" and str(res_str).lstrip().startswith("[VERIFIED]") and not is_failed and not criteria_failed:
                    is_verified = True
                    if tool_target:
                        eff_ledger.record_target_verified(tool_target)
                        if tool_target not in verified_targets_for_obs:
                            verified_targets_for_obs.append(tool_target)
                elif not is_failed and not acceptance_spec and tool_target:
                    # If an authorized mutating tool altered a previously verified target without re-verifying it,
                    # invalidate stale verification until re-verified.
                    muts = self._get_mutation_targets(func_name, func_args, eff_ledger.get_verified_targets())
                    for mt in muts:
                        if not is_verified and hasattr(eff_ledger, "invalidate_target_verification"):
                            eff_ledger.invalidate_target_verification(mt)

                if is_verified and not is_failed:
                    ver_meta_ok = {
                        "passed": True,
                        "check": (
                            str(c_results[-1][0].get("check", "outcome"))
                            if (acceptance_spec and c_results)
                            else ("verify_outcome" if func_name == "verify_outcome" else "outcome")
                        ),
                        "target": (
                            str(c_results[-1][0].get("target", ""))
                            if (acceptance_spec and c_results)
                            else str(tool_target or "")
                        ),
                        "observed": verification_detail or "Acceptance criteria satisfied",
                    }
                    structured_obs = ObservationFramingEngine.frame_structured(
                        func_name,
                        raw_tool_payload if raw_tool_payload is not None else res_str,
                        requested_tool=func_name,
                        requested_args=func_args,
                        executed_args=func_args,
                        dispatched=True,
                        timed_out=was_timed_out,
                        cancelled=was_cancelled,
                        execution_id=getattr(eff_ledger, "_execution_id", None),
                        correlation_id=tool_call_id,
                        external_source=ext_src_for_frame,
                        independent_verification=ver_meta_ok,
                    )
                    content_for_msg = ObservationFramingEngine.frame_observation(
                        func_name,
                        {
                            "output": res_str,
                            "status": "success",
                            "verified": True,
                            "verification_detail": verification_detail or "Acceptance criteria satisfied",
                        },
                        structured=structured_obs,
                        execution_id=getattr(eff_ledger, "_execution_id", None),
                        correlation_id=tool_call_id,
                        external_source=ext_src_for_frame,
                    )
                elif criteria_failed and not is_failed and failed_criteria_records:
                    first_fc = failed_criteria_records[0]
                    all_crit_str = "; ".join(fc["criterion"] for fc in failed_criteria_records)
                    all_reason_str = "; ".join(fc["reason"] for fc in failed_criteria_records)
                    all_req_str = "; ".join(fc["requirement"] for fc in failed_criteria_records)
                    ver_meta_fail = {
                        "passed": False,
                        "check": str(first_fc.get("criterion") or "outcome"),
                        "target": str(first_fc.get("target") or ""),
                        "error": all_reason_str,
                    }
                    structured_obs = ObservationFramingEngine.frame_structured(
                        func_name,
                        raw_tool_payload if raw_tool_payload is not None else res_str,
                        requested_tool=func_name,
                        requested_args=func_args,
                        executed_args=func_args,
                        dispatched=True,
                        timed_out=was_timed_out,
                        cancelled=was_cancelled,
                        execution_id=getattr(eff_ledger, "_execution_id", None),
                        correlation_id=tool_call_id,
                        external_source=ext_src_for_frame,
                        independent_verification=ver_meta_fail,
                    )
                    content_for_msg = ObservationFramingEngine.frame_observation(
                        func_name,
                        {
                            "output": obs_dict.get("output") or res_str,
                            "status": "success",
                            "verified": False,
                            "target": first_fc["target"],
                            "failed_criterion": all_crit_str,
                            "verifier_reason": all_reason_str,
                            "required_correction": (
                                f"{all_req_str}. Do not repeat identical parameters; emit a fresh tool call with corrected arguments."
                            ),
                            "verification_detail": verification_detail,
                        },
                        structured=structured_obs,
                        execution_id=getattr(eff_ledger, "_execution_id", None),
                        correlation_id=tool_call_id,
                        external_source=ext_src_for_frame,
                    )

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": func_name,
                    "content": content_for_msg,
                    "status": "error" if is_failed else "success",
                    "evidence_level": structured_obs.evidence_level,
                    "execution_status": structured_obs.execution_status,
                }
                if structured_obs.remaining_uncertainty:
                    tool_msg["remaining_uncertainty"] = structured_obs.remaining_uncertainty
                if is_failed:
                    if remedy_hint_out:
                        tool_msg["remedy_hint"] = remedy_hint_out
                    if allowed_actions_out:
                        tool_msg["allowed_actions"] = allowed_actions_out
                    if error_type_out:
                        tool_msg["error_type"] = error_type_out
                payload["messages"].append(tool_msg)
                curr_tool_history.append((fingerprint, "error" if is_failed else "success"))

                eff_ledger.record_observation(
                    tool_call_id,
                    status="error" if is_failed else "success",
                    output=obs_dict.get("output") or content_for_msg,
                    error=obs_dict.get("error") if is_failed else (verification_detail if criteria_failed else None),
                    exit_code=obs_dict.get("exit_code"),
                    verified=is_verified,
                    verification_detail=verification_detail,
                    verification_failed=bool(criteria_failed and not is_failed),
                    failed_targets=failed_targets_for_obs,
                    verified_targets=verified_targets_for_obs,
                    structured_observation=structured_obs,
                )
                if active_tg_step is not None and bound_tg_nodes:
                    for gn in bound_tg_nodes:
                        try:
                            active_tg_step.apply_observation(
                                gn.node_id,
                                structured_obs,
                                action_record=eff_ledger._records.get(tool_call_id),
                                outcome_verifier_passed=(True if is_verified else (False if criteria_failed else None)),
                                known_proposals=eff_ledger.get_tracked_proposals() if hasattr(eff_ledger, "get_tracked_proposals") else None,
                            )
                        except Exception:
                            pass

                if acceptance_spec and not is_failed and c_results and all(r.success for _, r in c_results):
                    all_criteria_satisfied = all(
                        OutcomeVerifier.verify(check=c["check"], target=c["target"], expected=c.get("expected")).success
                        for c in (parsed_c or [])
                    )
                    if all_criteria_satisfied:
                        all_user_prompts = " ".join(
                            (m.get("content") if isinstance(m.get("content"), str) else "")
                            for m in payload.get("messages", [])
                            for _ in [1]
                            if m.get("role") == "user" and not str(m.get("content", "")).startswith("[SYSTEM")
                        )
                        if not has_remaining_explicit_steps(all_user_prompts, eff_ledger, current_target=tool_target):
                            force_disable_next = True

                session_id = payload.get("session_id")
                if session_id:
                    try:
                        from zelyr.db.memory import add_message
                        add_message(session_id, "tool", f"[{func_name}] {res_str}", db_path=eff_db_path)
                    except Exception:
                        pass

            if force_disable_next and curr_pending_proposals:
                eff_ledger.discard_pending_proposals(reason="step_limit_reached" if actual_step_count >= eff_step_limit else "task_completed")
                curr_pending_proposals.clear()
            eff_ledger._active_pending_queue = list(curr_pending_proposals)

            if getattr(eff_ledger, "_central_ledger", None) and getattr(eff_ledger, "_execution_id", None):
                from zelyr.engine.checkpoint import CheckpointManager
                from zelyr.engine.execution_ledger import ExecutionStatus

                CheckpointManager.capture_checkpoint(
                    eff_ledger._central_ledger,
                    eff_ledger._execution_id,
                    turn_ledger=eff_ledger,
                    status=ExecutionStatus.RUNNING,
                    verification_status=eff_ledger._verification_status_override or "not_verified",
                    planning_state={
                        "depth": depth + 1,
                        "actual_step_count": actual_step_count,
                        "replanning_attempts": replanning_attempts,
                        "max_steps": getattr(self, "max_steps", 5),
                        "acceptance_criteria": acceptance_spec,
                        "model": payload.get("model"),
                    },
                    pending_proposals=curr_pending_proposals,
                    emit_event=False,
                    strict=None,
                )

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
                    disable_tools=(disable_tools or force_disable_next),
                    enable_cycle_detection=eff_cycle_detection,
                    ledger=eff_ledger,
                    pending_proposals=curr_pending_proposals,
                    replanning_attempts=replanning_attempts,
                    max_replanning_attempts=eff_max_replanning,
                    task_local_scope=eff_task_local_scope,
                ):
                    yield chunk
            elif curr_pending_proposals:
                eff_ledger.discard_pending_proposals(reason="max_depth_reached")
                curr_pending_proposals.clear()

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

