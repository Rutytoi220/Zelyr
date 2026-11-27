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
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
from zelyr.agents.actuator_arbiter import ActuatorArbiter
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
        max_concurrency: int = 1,
    ):
        if not isinstance(max_concurrency, int) or isinstance(max_concurrency, bool) or max_concurrency < 1:
            raise ValueError(f"max_concurrency must be a positive integer >= 1, got {max_concurrency!r}.")
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
        self.max_concurrency = int(max_concurrency)
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
        reserve_tokens: Optional[int] = 1000,
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
        "list_tree",
        "list_dir",
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

    SCREENSHOT_TOOL_ALIASES = {
        "screenshot",
        "screen_capture",
        "take_screenshot",
        "capture_screenshot",
        "observe_screen",
        "observe_desktop",
    }

    DESKTOP_CONTROL_ALIASES = {
        "mouse_move",
        "move_mouse",
        "mouse_click",
        "click_mouse",
        "double_click",
        "mouse_drag",
        "mouse_scroll",
        "keyboard_type",
        "type_text",
        "keyboard_press",
        "press_key",
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
        **{alias: "capture_screen" for alias in SCREENSHOT_TOOL_ALIASES},
        **{alias: "desktop_control" for alias in DESKTOP_CONTROL_ALIASES},
        **{alias: "interact_with_browser" for alias in BROWSER_TOOL_ALIASES},
        **{alias: "manage_memory" for alias in MEMORY_TOOL_ALIASES},
        **{alias: "craft_tool" for alias in CRAFT_TOOL_ALIASES},
        **{alias: "manage_routine" for alias in ROUTINE_TOOL_ALIASES},
    }

    _DESKTOP_READ_ONLY_ACTIONS = ActuatorArbiter.DESKTOP_READ_ONLY_ACTIONS
    _DESKTOP_INPUT_TOOLS = ActuatorArbiter.DESKTOP_INPUT_TOOLS | frozenset(DESKTOP_CONTROL_ALIASES)
    _WINDOW_READ_ONLY_ACTIONS = ActuatorArbiter.WINDOW_READ_ONLY_ACTIONS

    @staticmethod
    def _extract_current_user_prompt(messages: Optional[List[Dict[str, Any]]]) -> str:
        """Return the latest real user prompt, ignoring synthetic system/vision directives and prior turns."""
        return ActuatorArbiter.extract_current_user_prompt(messages)

    @classmethod
    def check_desktop_intent_guard(
        cls,
        tool_name: str,
        tool_args: Optional[Dict[str, Any]],
        *,
        user_prompt: str,
        acceptance_spec: Any = None,
    ) -> Optional[Tuple[str, str]]:
        """Block unrequested desktop pointer/keyboard input or window/workspace mutations across all dispatch paths."""
        return ActuatorArbiter.check_desktop_intent_guard(
            tool_name,
            tool_args,
            user_prompt=user_prompt,
            acceptance_spec=acceptance_spec,
        )

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
            raw_act = str(args.get("action") or "").strip()
            if not raw_act:
                if "search_text" in args or "replace_text" in args:
                    args["action"] = "patch"
                elif "content" in args:
                    args["action"] = "write"
                elif raw_orig in ("list_tree", "list_dir"):
                    args["action"] = "list_tree"
                elif raw_orig in cls.FILE_TOOL_ALIASES or "action" not in args:
                    args["action"] = "read"
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

        elif clean_name == "desktop_control":
            alias_action_map = {
                "mouse_move": "mouse_move",
                "move_mouse": "mouse_move",
                "mouse_click": "mouse_click",
                "click_mouse": "mouse_click",
                "double_click": "double_click",
                "mouse_drag": "mouse_drag",
                "mouse_scroll": "mouse_scroll",
                "keyboard_type": "keyboard_type",
                "type_text": "keyboard_type",
                "keyboard_press": "keyboard_press",
                "press_key": "keyboard_press",
            }
            if not str(args.get("action") or "").strip():
                if raw_orig in alias_action_map:
                    args["action"] = alias_action_map[raw_orig]
                elif "text" in args:
                    args["action"] = "keyboard_type"
                elif "key" in args or "keys" in args or "hotkey" in args:
                    args["action"] = "keyboard_press"
                elif "end_x" in args or "end_y" in args:
                    args["action"] = "mouse_drag"
                elif "x" in args and "y" in args and "button" in args:
                    args["action"] = "mouse_click"
                elif "x" in args and "y" in args:
                    args["action"] = "mouse_move"
            if "key" not in args:
                for alt_key in ("keys", "hotkey", "key_name", "shortcut"):
                    if alt_key in args:
                        args["key"] = args.pop(alt_key)
                        break
            if "coordinate" in args and "x" not in args and "y" not in args:
                coord_val = args.pop("coordinate")
                if isinstance(coord_val, str) and coord_val.strip().startswith("["):
                    try:
                        coord_val = json.loads(coord_val)
                    except Exception:
                        coord_val = None
                if isinstance(coord_val, (list, tuple)) and len(coord_val) == 2:
                    try:
                        args["x"] = int(round(float(coord_val[0])))
                        args["y"] = int(round(float(coord_val[1])))
                    except (TypeError, ValueError):
                        pass
            if "expected_window" not in args and "window" in args and args.get("action") != "inspect_windows":
                args["expected_window"] = args.get("window")
            if args.get("action") in ("screenshot", "observe") and "target" in args:
                tgt_val = str(args.pop("target") or "").strip()
                if tgt_val and "window" not in args and tgt_val.lower() not in (
                    "window",
                    "active_window",
                    "active_monitor",
                    "monitor",
                    "region",
                    "full",
                    "screen",
                    "desktop",
                ):
                    args["window"] = tgt_val

        elif clean_name == "capture_screen":
            valid_capture_targets = {"active_monitor", "active_window", "window", "monitor", "region", "full"}
            t_raw = str(args.get("target") or "").strip()
            if not t_raw:
                if args.get("window"):
                    args["target"] = "window"
                elif args.get("monitor"):
                    args["target"] = "monitor"
                elif args.get("region"):
                    args["target"] = "region"
                else:
                    args["target"] = "active_monitor"
            elif t_raw.lower() in valid_capture_targets:
                args["target"] = t_raw.lower()
            elif t_raw.lower() in ("desktop", "screen", "all", "entire_screen", "full_screen"):
                args["target"] = "full"
            else:
                if not str(args.get("window") or "").strip():
                    args["window"] = t_raw
                args["target"] = "window"

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
        fn_low = (func_name or "").lower()
        if fn_low == "verify_outcome":
            val = func_args.get("target") or func_args.get("window")
            if val and isinstance(val, str) and val.strip():
                base_val = val.strip()
                sel_val = func_args.get("selector")
                chk_low = str(func_args.get("check") or "").strip().lower()
                if (
                    chk_low.startswith("desktop_")
                    and isinstance(sel_val, str)
                    and sel_val.strip()
                    and "::" not in base_val
                ):
                    return f"{base_val}::{sel_val.strip()}"
                return base_val
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
                "mkdir", "create_directory", "create_dir",
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
        pending_vision_b64 = wire_payload.pop("_pending_vision_b64", None)
        force_vision_image = bool(wire_payload.pop("_force_vision_image", False))
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
        vision_attached = False
        if isinstance(pending_vision_b64, str) and pending_vision_b64 and isinstance(wire_payload.get("messages"), list):
            supports_vision = force_vision_image or await self._model_supports_vision(
                str(wire_payload.get("model") or ""), base_url
            )
            if supports_vision:
                wire_payload["messages"] = list(wire_payload["messages"]) + [
                    {
                        "role": "user",
                        "content": (
                            "[Transient desktop screenshot attached for the immediately preceding "
                            "capture_screen/observe tool observation. Carefully read the ACTUAL rendered UI "
                            "text, values, and coordinates from this image. Never assume an earlier click or "
                            "type succeeded unless the resulting state is clearly visible here; if anything is "
                            "missing or reset, emit corrective tool calls now rather than concluding.]"
                        ),
                        "images": [pending_vision_b64],
                    }
                ]
                vision_attached = True

        async with httpx.AsyncClient(timeout=timeout_config) as client:
            async with client.stream(
                "POST",
                f"{base_url}/api/chat",
                json=wire_payload
            ) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    err_text_resp = resp.text
                    if vision_attached and resp.status_code == 400 and any(
                        tok in err_text_resp.lower() for tok in ("vision", "image", "multimodal", "unsupported")
                    ):
                        cache_key = (str(base_url).rstrip("/"), str(wire_payload.get("model") or ""))
                        self._vision_support_cache[cache_key] = False
                        retry_payload = dict(wire_payload)
                        retry_payload["messages"] = list(wire_payload["messages"][:-1])
                        async with client.stream(
                            "POST",
                            f"{base_url}/api/chat",
                            json=retry_payload,
                        ) as retry_resp:
                            if retry_resp.status_code != 200:
                                await retry_resp.aread()
                                yield {"_api_error": f"HTTP {retry_resp.status_code}: {retry_resp.text}"}
                                return
                            async for line in retry_resp.aiter_lines():
                                is_done, events = self._parse_stream_line_to_events(line)
                                for evt in events:
                                    yield evt
                                if is_done:
                                    break
                        return
                    yield {"_api_error": f"HTTP {resp.status_code}: {err_text_resp}"}
                    return

                async for line in resp.aiter_lines():
                    is_done, events = self._parse_stream_line_to_events(line)
                    for evt in events:
                        yield evt
                    if is_done:
                        break

    _vision_support_cache: Dict[tuple, bool] = {}

    async def _model_supports_vision(self, model: str, base_url: str) -> bool:
        """Query and cache whether the target Ollama model advertises vision capability."""
        clean_model = (model or "").strip()
        clean_url = (base_url or "").rstrip("/")
        if not clean_model or not clean_url:
            return False
        cache_key = (clean_url, clean_model)
        if cache_key in self._vision_support_cache:
            return self._vision_support_cache[cache_key]
        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                resp = await client.post(f"{clean_url}/api/show", json={"name": clean_model})
            if resp.status_code == 200:
                data = resp.json()
                caps = data.get("capabilities")
                if isinstance(caps, list):
                    supported = any(str(c).lower() == "vision" for c in caps)
                    self._vision_support_cache[cache_key] = supported
                    return supported
                details = data.get("details") if isinstance(data.get("details"), dict) else {}
                families = [str(f).lower() for f in (details.get("families") or [])]
                supported = any("clip" in f or "vision" in f or "mllama" in f or "vl" in f for f in families)
                self._vision_support_cache[cache_key] = supported
                return supported
        except Exception:
            pass
        return False

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

        reuse_current = bool(payload.pop("_reuse_current_ledger", False))
        defer_root_finalization = bool(payload.pop("_defer_root_finalization", False))
        eff_ledger = (
            ledger
            if ledger is not None
            else (
                getattr(self, "current_ledger", None)
                if reuse_current and getattr(self, "current_ledger", None) is not None
                else TurnExecutionLedger()
            )
        )
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
            active_tg_root = eff_ledger.get_task_graph() if hasattr(eff_ledger, "get_task_graph") else None
            if active_tg_root is not None and payload.get("execute_task_graph_parallel"):
                tg_res = await self.execute_task_graph(
                    active_tg_root,
                    proposals=(
                        payload.get("graph_proposals")
                        or (eff_ledger.get_tracked_proposals() if hasattr(eff_ledger, "get_tracked_proposals") else None)
                        or payload.get("_pending_proposals")
                    ),
                    max_concurrency=payload.get("max_concurrency"),
                    execution_id=exec_id,
                    central_ledger=central_ledger,
                    turn_ledger=eff_ledger,
                    tool_schemas=tool_schemas,
                    tool_executor=tool_executor,
                    caller_ring=eff_caller_ring,
                    allowed_tools=getattr(self, "allowed_tools", None),
                    task_local_scope=eff_task_local_scope,
                    acceptance_criteria=acceptance_spec,
                    allow_unverified_retry=bool(payload.get("allow_unverified_retry", False)),
                    finalize_execution=is_resumed_root,
                    task_prompt=user_prompt,
                )
                summary_txt = f"TaskGraph '{tg_res['graph_id']}' execution finished ({tg_res['graph_status']})."
                collected_out.append(summary_txt)
                yield {"choices": [{"delta": {"content": summary_txt}, "finish_reason": "stop"}]}
                return
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
                try:
                    if hasattr(eff_ledger, "flush_observability_metrics"):
                        eff_ledger.flush_observability_metrics()
                except Exception:
                    pass
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
                if not already_finalized and not defer_root_finalization:
                    from zelyr.engine.execution_ledger import (
                        parse_acceptance_criteria,
                        resolve_root_execution_outcome,
                    )

                    root_meta = getattr(eff_ledger, "_root_meta", {}) or {}
                    final_text = "".join(collected_out)
                    total_acts = eff_ledger.total_actions()
                    prov_err = (
                        str(root_meta.get("provider_error") or root_meta.get("error") or "Model provider failure")
                        if (root_meta.get("connection_error") or root_meta.get("provider_error"))
                        else None
                    )
                    pre_v_results = None
                    if (
                        acceptance_spec is not None
                        and not (was_cancelled or getattr(self, "_cancelled", False) or payload.get("cancelled", False) or root_meta.get("cancelled"))
                        and caught_exc is None
                        and not prov_err
                    ):
                        parsed_root_specs = parse_acceptance_criteria(acceptance_spec)
                        if parsed_root_specs:
                            from zelyr.engine.verifier import OutcomeVerifier

                            pre_v_results = []
                            for spec_item in parsed_root_specs:
                                vr = await OutcomeVerifier.verify_criterion_async(spec_item)
                                pre_v_results.append((spec_item.get("check", ""), spec_item.get("target", ""), vr))

                    res_outcome = resolve_root_execution_outcome(
                        prompt=user_prompt,
                        ledger=eff_ledger,
                        final_text=final_text,
                        acceptance_spec=acceptance_spec,
                        v_results=pre_v_results,
                        was_cancelled=bool(
                            was_cancelled
                            or getattr(self, "_cancelled", False)
                            or payload.get("cancelled", False)
                            or root_meta.get("cancelled")
                        ),
                        was_timed_out=bool((root_meta.get("timed_out") and eff_ledger.has_unresolved_failures()) or isinstance(caught_exc, (TimeoutError, asyncio.TimeoutError))),
                        caught_exc=caught_exc,
                        provider_error=prov_err,
                        is_truncated=(getattr(self, "last_finish_reason", None) in ("length", "max_tokens") and not final_text.strip()),
                    )
                    final_status = res_outcome.execution_status
                    ver_status = res_outcome.verification_status
                    err_msg = res_outcome.error
                    interruption_reason = res_outcome.interruption_reason

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
                                "policy_compliance": res_outcome.policy_compliance,
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
                                metadata={
                                    "total_actions": total_acts,
                                    "verified_targets": res_outcome.verified_targets,
                                    "checkpoint_saved": False,
                                    **({"policy_compliance": res_outcome.policy_compliance} if res_outcome.policy_compliance.get("has_prohibitions") else {}),
                                },
                            )
                            raise

                    if cp_err:
                        root_meta["checkpoint_failed"] = True
                        root_meta["checkpoint_error"] = cp_err

                    if res_outcome.v_results and not (
                        central_ledger.get_events(exec_id, event_type=LedgerEventType.VERIFICATION_RECORDED.value)
                    ):
                        for v_check, v_target, v_res in res_outcome.v_results:
                            central_ledger.record_event(
                                exec_id,
                                LedgerEventType.VERIFICATION_RECORDED,
                                status=VerificationStatus.VERIFIED.value if v_res.success else VerificationStatus.VERIFICATION_FAILED.value,
                                metadata={
                                    "check": v_check,
                                    "target": v_target,
                                    "result": getattr(v_res, "result", "PASS" if v_res.success else "FAIL"),
                                    "evidence_source": getattr(v_res, "evidence_source", None),
                                    "evidence_collected": getattr(v_res, "evidence_collected", None),
                                    "expected": getattr(v_res, "expected", None),
                                    "error": getattr(v_res, "error", None),
                                },
                            )

                    fin_meta: Dict[str, Any] = {
                        "total_actions": total_acts,
                        "verified_targets": res_outcome.verified_targets,
                        "checkpoint_saved": not bool(root_meta.get("checkpoint_failed")),
                    }
                    if interruption_reason:
                        fin_meta["interruption_reason"] = interruption_reason
                    if res_outcome.diagnostic_failures_only:
                        fin_meta["diagnostic_test_failed"] = True
                    if res_outcome.policy_compliance.get("has_prohibitions"):
                        fin_meta["policy_compliance"] = res_outcome.policy_compliance
                    if cp_err:
                        fin_meta["checkpoint_error"] = cp_err

                    central_ledger.finalize_execution(
                        exec_id,
                        status=final_status,
                        verification_status=ver_status,
                        final_summary=final_text[:1024] if final_text else None,
                        error=err_msg,
                        metadata=fin_meta,
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
                # unless running under an isolated Native Messaging socket.
                if not (os.environ.get("ZELYR_BROWSER_NATIVE_SOCKET") or "").strip():
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

        from zelyr.core.plugins import filter_tool_schemas_by_tier
        from zelyr.agents.actuator_arbiter import ActuatorArbiter, is_browser_window
        browser_active = is_browser_window(active_window)

        tier = self.route_tier(
            user_query_for_intent,
            active_window=active_window,
            bridge_connected=bridge_connected,
        )
        if is_recovery_turn and allowed_recovery_actions:
            tier2_actions = {"get_page_snapshot", "scroll_page", "click_element", "fill_element", "list_tabs", "open_tab", "switch_tab", "navigate_url"}
            tier3_actions = {"desktop_control", "capture_screen"}
            if any(a in tier2_actions for a in allowed_recovery_actions):
                tier = "tier2_browser"
            elif any(a in tier3_actions for a in allowed_recovery_actions) and not browser_active:
                tier = "tier3_vision"

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

        has_cross_surface = ActuatorArbiter.has_cross_surface_browser_and_desktop_intent(
            user_query_for_intent
        )
        if depth == 0 and not hasattr(eff_ledger, "_base_payload_tools") and payload.get("tools"):
            eff_ledger._base_payload_tools = list(payload["tools"])

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
                active_tools = list(
                    getattr(eff_ledger, "_base_payload_tools", None)
                    or payload.get("tools")
                    or get_tool_schemas(0)
                )

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
            payload["tools"] = filter_tool_schemas_by_tier(
                native_tools_only,
                tier,
                is_browser=browser_active,
                include_tiers=({2, 3} if has_cross_surface else None),
            )
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

            desktop_recovery_tools = {"desktop_control", "capture_screen", "manage_desktop_window"}
            desktop_failure_seen = any(
                (getattr(f, "tool_name", "") or "").lower()
                in {"interact_with_ui", "desktop_control", "capture_screen", "screen_capture", "manage_desktop_window"}
                for f in unresolved_exec_failures
            )
            foundational_tools = {"execute_command", "manage_workspace_file"}
            permitted_recovery_tools = effective_allowed | foundational_tools
            if desktop_failure_seen:
                permitted_recovery_tools |= desktop_recovery_tools
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

        # Apply adaptive context window budgeting using final scoped/compacted schemas and safe reserve
        if 4096 <= eff_ctx <= 8192:
            eff_reserve: Optional[int] = max(1638, int(eff_ctx * 0.25))
        elif eff_ctx < 4096:
            eff_reserve = max(100, int(eff_ctx * 0.25))
        else:
            eff_reserve = None
        payload["messages"] = self.prune_messages(
            payload.get("messages", []),
            max_context_tokens=eff_ctx,
            reserve_tokens=eff_reserve,
            model_name=payload["model"],
            tools=payload.get("tools") if payload.get("tools") is not None else eff_tool_schemas,
            config=config,
        )

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

        if hasattr(eff_ledger, "record_orchestration_step"):
            eff_ledger.record_orchestration_step()
        _llm_t0 = time.monotonic()
        pending_vis_for_turn = payload.pop("_pending_vision_b64", None)
        if pending_vis_for_turn and eff_ledger is not None:
            eff_ledger._cached_desktop_vision_b64 = pending_vis_for_turn
        stream_payload = payload
        if pending_vis_for_turn:
            stream_payload = dict(payload)
            stream_payload["_pending_vision_b64"] = pending_vis_for_turn
        try:
            async for data in self._stream_model_events(stream_payload, base_url, stream_provider=eff_stream_provider):
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
        finally:
            if hasattr(eff_ledger, "record_model_request"):
                eff_ledger.record_model_request((time.monotonic() - _llm_t0) * 1000.0)

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

            # Mid-task token-limit exhaustion during <think>: retry once concisely rather than discarding prior steps
            if (
                stream_finish_reason in ("length", "max_tokens")
                and len(clean_text) < 120
                and actual_step_count > 0
                and not is_retry
                and depth < eff_max_depth - 1
                and not getattr(self, "_cancelled", False)
                and not payload.get("cancelled", False)
            ):
                buffered_chunks.clear()
                collected_assistant_text.clear()
                for msg in payload.get("messages", []):
                    if isinstance(msg, dict) and msg.get("role") == "assistant" and isinstance(msg.get("content"), str) and "<think>" in msg["content"]:
                        msg["content"] = re.sub(r"<think>.*?(?:</think>|$)", "", msg["content"], flags=re.DOTALL).strip()
                payload.setdefault("messages", []).append({
                    "role": "user",
                    "content": (
                        "[SYSTEM NOTICE: Token Limit Reached During Reasoning] Your previous reasoning "
                        "exhausted the generation limit before emitting a tool call or visible answer. "
                        "Keep internal thinking brief (1-2 sentences) and immediately invoke the single next "
                        "tool call or provide your grounded final response."
                    ),
                })
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

            # Thinking-only emission guard: some reasoning models emit only <think>...</think>
            # with empty visible output and stop. Nudge once to emit a visible final response.
            if (
                not clean_text
                and ("<think>" in full_text or "</think>" in full_text)
                and not is_retry
                and (depth < eff_max_depth - 1 or (depth < eff_max_depth and actual_step_count > 0))
                and not getattr(self, "_cancelled", False)
                and not payload.get("cancelled", False)
            ):
                buffered_chunks.clear()
                collected_assistant_text.clear()
                payload.setdefault("messages", []).append({
                    "role": "user",
                    "content": (
                        "[SYSTEM NOTICE: Empty Visible Response] You placed your output only inside "
                        "<think> reasoning with no visible final answer. Provide a concise, grounded "
                        "visible response now based strictly on your observed tool results "
                        "(do not place the final answer inside <think>)."
                    ),
                })
                payload.setdefault("options", {})["temperature"] = 0.1
                payload["temperature"] = 0.1
                synthesis_only = depth + 1 >= eff_max_depth
                async for retry_chunk in self.generate_stream(
                    payload,
                    depth=depth + 1,
                    is_retry=True,
                    stream_provider=eff_stream_provider,
                    tool_schemas=eff_tool_schemas,
                    tool_executor=eff_tool_executor,
                    db_path=eff_db_path,
                    max_depth=(depth + 2) if synthesis_only else eff_max_depth,
                    turn_tool_history=curr_tool_history,
                    actual_step_count=actual_step_count,
                    disable_tools=bool(disable_tools or synthesis_only),
                    ledger=eff_ledger,
                    enable_cycle_detection=eff_cycle_detection,
                    replanning_attempts=replanning_attempts,
                    max_replanning_attempts=eff_max_replanning,
                    task_local_scope=eff_task_local_scope,
                ):
                    yield retry_chunk
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
                                f"Do NOT write proposed code blocks in prose or say 'Let's fix/write...'; invoke a tool call directly.\n"
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
                from zelyr.engine.execution_ledger import find_unfulfilled_task_requirements_async

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

                verified_outcomes, unfulfilled_outcomes = await find_unfulfilled_task_requirements_async(
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
            if eff_ledger is not None:
                eff_ledger._cached_desktop_vision_b64 = None
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
                current_user_prompt = self._extract_current_user_prompt(payload.get("messages"))
                all_user_prompts = " ".join(
                    (m.get("content") if isinstance(m.get("content"), str) else "")
                    for m in payload.get("messages", [])
                    if m.get("role") == "user" and not str(m.get("content", "")).startswith("[SYSTEM")
                )
                if rejection_reason is None:
                    guard_hit = self.check_desktop_intent_guard(
                        func_name,
                        func_args,
                        user_prompt=current_user_prompt,
                        acceptance_spec=acceptance_spec,
                    )
                    if guard_hit is not None:
                        rejection_reason, rejection_detail = guard_hit

                if rejection_reason is None and actual_step_count >= eff_step_limit:
                    rejection_reason = "step_limit_reached"
                    rejection_detail = f"Maximum execution step limit ({eff_step_limit}) reached for current turn. Explain the limitation and synthesize the answer."
                    force_disable_next = True
                elif rejection_reason is None and eff_cycle_detection:
                    is_explicit_repeat = self._is_explicit_repeat_requested(all_user_prompts)

                    from zelyr.engine.execution_ledger import (
                        _is_inspection_action,
                        canonicalize_target,
                        criterion_applies_to_target,
                        extract_file_targets_from_text,
                        has_remaining_explicit_steps,
                        is_diagnostic_test_command,
                        parse_acceptance_criteria,
                        targets_match,
                    )
                    from zelyr.engine.verifier import OutcomeVerifier

                    # Check if action is an idempotent read/inspection check (allowed to re-observe after mutations)
                    func_lower = func_name.lower()
                    is_read_tool = (
                        func_lower.startswith(("read_", "get_", "list_", "check_", "inspect_", "verify_", "stat_"))
                        or func_lower in (
                            "get_page_snapshot",
                            "list_tabs",
                            "read_file",
                            "view_file",
                            "list_dir",
                            "list_tree",
                            "read_state",
                            "verify_outcome",
                            "capture_screen",
                            "screen_capture",
                        )
                        or (
                            func_lower in ("desktop_control", "desktop_automation", "manage_desktop_window", "interact_with_browser", "browser_cdp")
                            and _is_inspection_action(func_name, func_args)
                        )
                    )
                    # Stateful compositor focus/workspace switching may legitimately need to run again
                    # after focus drifts or an intervening desktop failure (e.g. FOCUS_LOST), though
                    # consecutive back-to-back duplicates with no intervening call remain redundant.
                    is_browser_after_nav = False
                    if (
                        func_lower in ("interact_with_browser", "browser_cdp")
                        and isinstance(func_args, dict)
                        and str(func_args.get("action") or "").strip().lower() in ("click_element", "fill_element", "press_key", "click", "type")
                        and curr_tool_history
                    ):
                        last_same_idx = -1
                        for idx_h, (h_fp, h_st) in enumerate(curr_tool_history):
                            if h_fp == fingerprint and h_st == "success":
                                last_same_idx = idx_h
                        if last_same_idx >= 0:
                            for h_fp, h_st in curr_tool_history[last_same_idx + 1 :]:
                                if h_st != "success" or not isinstance(h_fp, tuple):
                                    continue
                                h_name, h_canon = h_fp
                                if str(h_name or "").lower() in ("interact_with_browser", "browser_cdp"):
                                    try:
                                        h_args = json.loads(h_canon) if isinstance(h_canon, str) and h_canon.startswith("{") else {}
                                    except Exception:
                                        h_args = {}
                                    if str(h_args.get("action") or "").strip().lower() in (
                                        "navigate_url", "open_tab", "switch_tab", "reload_tab", "get_page_snapshot", "scroll_page"
                                    ):
                                        is_browser_after_nav = True
                                        break
                    is_state_recovery_action = (
                        is_browser_after_nav
                        or (
                            func_lower == "manage_desktop_window"
                            and isinstance(func_args, dict)
                            and str(func_args.get("action") or "").strip().lower() in ("focus", "workspace", "focus_window")
                            and (
                                not curr_tool_history
                                or not (curr_tool_history[-1][0] == fingerprint and curr_tool_history[-1][1] == "success")
                            )
                        )
                    )

                    # Check if action mutates any already-verified resource
                    is_test_or_verify_cmd = is_diagnostic_test_command(func_name, func_args) or func_lower in (
                        "verify_outcome",
                        "verify_state",
                        "check_state",
                    )
                    cycle_history = curr_tool_history
                    last_mut_idx = -1
                    if (is_test_or_verify_cmd or is_read_tool) and curr_tool_history:
                        for idx_h, (h_fp, h_st) in enumerate(curr_tool_history):
                            if h_st != "success":
                                continue
                            h_name, h_canon = h_fp
                            try:
                                h_args = json.loads(h_canon) if isinstance(h_canon, str) and h_canon.startswith("{") else {}
                            except Exception:
                                h_args = {}
                            if not _is_inspection_action(h_name, h_args) and not is_diagnostic_test_command(h_name, h_args):
                                last_mut_idx = idx_h
                        if last_mut_idx >= 0:
                            cycle_history = curr_tool_history[last_mut_idx + 1 :]
                    elif (
                        func_lower in ("interact_with_browser", "browser_cdp")
                        and isinstance(func_args, dict)
                        and str(func_args.get("action") or "").strip().lower() in ("click_element", "fill_element", "press_key", "click", "type")
                        and curr_tool_history
                    ):
                        # Reset browser element retry window after a successful DOM refresh or navigation
                        last_refresh_idx = -1
                        for idx_h, (h_fp, h_st) in enumerate(curr_tool_history):
                            if h_st != "success" or not isinstance(h_fp, tuple):
                                continue
                            h_name, h_canon = h_fp
                            if str(h_name or "").lower() in ("interact_with_browser", "browser_cdp"):
                                try:
                                    h_args = json.loads(h_canon) if isinstance(h_canon, str) and h_canon.startswith("{") else {}
                                except Exception:
                                    h_args = {}
                                if str(h_args.get("action") or "").strip().lower() in (
                                    "get_page_snapshot", "navigate_url", "open_tab", "reload_tab", "switch_tab"
                                ):
                                    last_refresh_idx = idx_h
                        if last_refresh_idx >= 0:
                            cycle_history = curr_tool_history[last_refresh_idx + 1 :]
                    elif (
                        func_lower in ("desktop_control", "desktop_automation")
                        and not is_read_tool
                        and curr_tool_history
                    ):
                        # Reset desktop retry ceiling after explicit window focus/workspace recovery
                        last_focus_idx = -1
                        for idx_h, (h_fp, h_st) in enumerate(curr_tool_history):
                            if h_st != "success" or not isinstance(h_fp, tuple):
                                continue
                            h_name, h_canon = h_fp
                            h_low = str(h_name or "").lower()
                            try:
                                h_args = json.loads(h_canon) if isinstance(h_canon, str) and h_canon.startswith("{") else {}
                            except Exception:
                                h_args = {}
                            h_act = str(h_args.get("action") or "").strip().lower()
                            if (
                                (h_low == "manage_desktop_window" and h_act in ("focus", "focus_window", "focuswindow", "workspace"))
                                or (h_low in ("desktop_control", "desktop_automation") and h_act == "focus_window")
                            ):
                                last_focus_idx = idx_h
                        if last_focus_idx >= 0:
                            cycle_history = curr_tool_history[last_focus_idx + 1 :]
                    elif (
                        func_lower == "manage_desktop_window"
                        and isinstance(func_args, dict)
                        and str(func_args.get("action") or "").strip().lower() in ("focus", "focus_window", "focuswindow", "workspace")
                        and curr_tool_history
                    ):
                        # Reset focus-call retry window whenever a newer desktop_control failure (e.g. FOCUS_LOST) occurred
                        last_dc_fail_idx = -1
                        for idx_h, (h_fp, h_st) in enumerate(curr_tool_history):
                            if h_st == "error" and isinstance(h_fp, tuple):
                                if str(h_fp[0] or "").lower() in ("desktop_control", "desktop_automation"):
                                    last_dc_fail_idx = idx_h
                        if last_dc_fail_idx >= 0:
                            cycle_history = curr_tool_history[last_dc_fail_idx + 1 :]

                    if is_test_or_verify_cmd and last_mut_idx >= 0:
                        has_prior_success = any(
                            p_fp == fingerprint and p_st == "success" for p_fp, p_st in cycle_history
                        )
                    else:
                        has_prior_success = (
                            (curr_tool_history and curr_tool_history[-1][0] == fingerprint and curr_tool_history[-1][1] == "success")
                            or (hasattr(eff_ledger, "has_identical_successful_action") and eff_ledger.has_identical_successful_action(func_name, canonical_args))
                        )
                    is_file_mutation_action = (
                        func_name == "manage_workspace_file"
                        and str(func_args.get("action", "")).lower() in ("write", "patch", "rollback")
                    )
                    has_prior_failed_verification = (
                        is_file_mutation_action
                        and not is_test_or_verify_cmd
                        and not is_read_tool
                        and hasattr(eff_ledger, "has_identical_failed_verification_action")
                        and eff_ledger.has_identical_failed_verification_action(func_name, canonical_args)
                    )

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
                            c_ok = True
                            for mc in matching_c:
                                mc_vr = await OutcomeVerifier.verify_criterion_async(mc)
                                if not mc_vr.success:
                                    c_ok = False
                                    break
                            if c_ok:
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
                    last_exec_rec = next(
                        (
                            eff_ledger._records[cid]
                            for cid in reversed(eff_ledger._order)
                            if cid != tool_call_id
                            and cid in eff_ledger._records
                            and eff_ledger._records[cid].state not in (ActionState.PROPOSED, ActionState.REJECTED)
                        ),
                        None,
                    )
                    last_exec_desktop_focus_lost = bool(
                        func_lower in ("desktop_control", "desktop_automation", "interact_with_ui")
                        and not is_read_tool
                        and last_exec_rec is not None
                        and (last_exec_rec.tool_name or "").lower() in ("desktop_control", "desktop_automation", "interact_with_ui")
                        and last_exec_rec.state == ActionState.FAILED
                        and (
                            "focus_lost" in str(last_exec_rec.error or "").lower()
                            or "is not focused" in str(last_exec_rec.error or "").lower()
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
                    elif (
                        func_lower == "interact_with_ui"
                        and isinstance(payload.get("tools") or auth_schemas, list)
                        and bool(payload.get("tools") or auth_schemas)
                        and "interact_with_ui" not in {
                            (s.get("function", {}).get("name") or s.get("name"))
                            for s in (payload.get("tools") or auth_schemas)
                            if isinstance(s, dict)
                        }
                        and "desktop_control" in {
                            (s.get("function", {}).get("name") or s.get("name"))
                            for s in (payload.get("tools") or auth_schemas)
                            if isinstance(s, dict)
                        }
                    ):
                        rejection_reason = "tool_unavailable"
                        rejection_detail = (
                            "Tool 'interact_with_ui' is not enabled in this runtime (Spotless Arms unavailable). "
                            "Use 'capture_screen' to inspect coordinates and 'desktop_control' "
                            "(mouse_click, double_click, mouse_move, keyboard_type, keyboard_press) instead."
                        )
                    elif has_prior_success and not is_explicit_repeat and not is_read_tool and not is_state_recovery_action:
                        rejection_reason = "redundant_action"
                        rejection_detail = (
                            f"Action '{func_name}' with identical parameters has already succeeded. "
                            "Do not repeat completed actions; provide the final response or proceed to the next distinct step."
                        )
                    elif has_prior_failed_verification and not is_explicit_repeat and not is_read_tool:
                        rejection_reason = "redundant_action"
                        rejection_detail = (
                            f"Action '{func_name}' with identical parameters already ran and failed verification. "
                            "Inspect the failing test/verification details using 'manage_workspace_file' and make a distinct fix instead of reverting to failed parameters."
                        )
                    elif last_exec_desktop_focus_lost:
                        rejection_reason = "cycle_detected"
                        rejection_detail = (
                            "Focus was lost (FOCUS_LOST) on the preceding desktop_control action and focus recovery has not run yet. "
                            "Do NOT retry clicks or typing with adjusted coordinates; call manage_desktop_window(action='focus', target=...) first."
                        )
                    elif cycle_history and cycle_history[-1][0] == fingerprint and cycle_history[-1][1] == "error":
                        # Identical Retry After Error: immediate preceding had same fingerprint and ended in error
                        rejection_reason = "cycle_detected"
                        if is_test_or_verify_cmd:
                            rejection_detail = (
                                "Identical tool call retry after immediate failure is blocked. "
                                "Do not repeat the failing test command unchanged; inspect source/test files with 'manage_workspace_file' and fix the code before re-testing."
                            )
                        else:
                            rejection_detail = "Identical tool call retry after immediate failure is blocked. Alter parameters or explain the limitation."
                    elif (
                        sum(1 for prior_fp, _ in cycle_history if prior_fp == fingerprint) >= 2
                        or sum(1 for prior_fp, _ in curr_tool_history if prior_fp == fingerprint) >= 8
                    ):
                        # Identical Call Ceiling: attempted for the 3rd time without intervening state change
                        rejection_reason = "cycle_detected"
                        if is_test_or_verify_cmd:
                            rejection_detail = (
                                "Identical tool call ceiling reached (attempted 3 or more times). "
                                "Do not repeat unchanged test runs; inspect or edit the code with 'manage_workspace_file' before testing again."
                            )
                        else:
                            rejection_detail = "Identical tool call ceiling reached (attempted 3 or more times). Alter parameters or explain the limitation."
                    elif len(cycle_history) >= 3:
                        # Alternating Failure Loop: A -> B -> A -> B where both A and B returned error
                        A_prev = cycle_history[-1][0]
                        B_prev = cycle_history[-2][0]
                        A_init = cycle_history[-3][0]
                        if (
                            fingerprint == B_prev
                            and A_prev == A_init
                            and A_prev != B_prev
                            and cycle_history[-1][1] == "error"
                            and cycle_history[-2][1] == "error"
                            and cycle_history[-3][1] == "error"
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

                # Runtime-enforced user prohibitions: explicit target bans AND workspace-wide read-only directives
                # cannot be overridden by model-proposed tool calls.
                if rejection_reason is None:
                    from zelyr.engine.execution_ledger import (
                        extract_global_mutation_prohibition,
                        is_mutating_tool_call,
                        targets_match,
                    )
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

                    if rejection_reason is None:
                        from zelyr.engine.execution_ledger import (
                            _DESKTOP_MUTATION_SHELL_RE,
                            _is_inspection_action,
                            extract_desktop_interaction_prohibition,
                        )
                        global_prohib = extract_global_mutation_prohibition(_user_text)
                        if global_prohib["no_file_modifications"] or global_prohib["no_mutating_commands"]:
                            allow_tests = bool(
                                global_prohib["is_test_diagnostic"]
                                and not global_prohib["no_mutating_commands"]
                            )
                            is_mut, mut_reason = is_mutating_tool_call(
                                func_name,
                                func_args,
                                allow_read_only_tests=allow_tests,
                            )
                            if is_mut:
                                rejection_reason = "prohibited_by_user"
                                rejection_detail = (
                                    f"The user explicitly prohibited modifying files or running mutating commands ({mut_reason}). "
                                    f"The call to '{func_name}' is blocked; use read-only inspection only and do not retry mutations."
                                )
                        elif extract_desktop_interaction_prohibition(_user_text):
                            t_low_prohib = (func_name or "").lower().strip()
                            is_desktop_mut = (
                                not _is_inspection_action(t_low_prohib, func_args)
                                and (
                                    t_low_prohib in (
                                        "desktop_control",
                                        "desktop_automation",
                                        "interact_with_ui",
                                        "gui_actuate",
                                        "click_tag",
                                        "type_text",
                                        "press_key",
                                        "move_mouse",
                                        "click_mouse",
                                        "manage_desktop_window",
                                    )
                                    or (
                                        t_low_prohib in ("execute_command", "bash", "sh")
                                        and bool(_DESKTOP_MUTATION_SHELL_RE.search(str(func_args.get("command") or func_args.get("cmd") or "")))
                                    )
                                )
                            )
                            if is_desktop_mut:
                                rejection_reason = "prohibited_by_user"
                                rejection_detail = (
                                    f"The user explicitly prohibited interacting with or modifying the desktop UI/window. "
                                    f"The call to '{func_name}' is blocked; use read-only observation only and do not retry interaction."
                                )

                # Centralized Unified Execution Policy gate right before dispatch
                pol_dec: Any = None
                if rejection_reason is None:
                    from zelyr.engine.execution_policy import (
                        PolicyDecisionOutcome,
                        UNTRUSTED_POLICY_OVERRIDE_KEYS,
                        build_execution_policy_engine,
                    )
                    from zelyr.engine.normalizer import strict_json_loads
                    from zelyr.tool_registry import resolve_tool_ring

                    raw_tc_args: Any = func_args
                    cand_raw = tc.get("arguments")
                    if isinstance(cand_raw, dict):
                        raw_tc_args = cand_raw
                    elif isinstance(cand_raw, str) and cand_raw.strip():
                        try:
                            parsed_raw = strict_json_loads(cand_raw.strip())
                            if isinstance(parsed_raw, dict):
                                raw_tc_args = parsed_raw
                        except Exception:
                            raw_tc_args = func_args

                    smuggled_pol = (
                        [str(k) for k in raw_tc_args.keys() if str(k).strip() in UNTRUSTED_POLICY_OVERRIDE_KEYS]
                        if isinstance(raw_tc_args, dict)
                        else []
                    )
                    if smuggled_pol:
                        rejection_reason = "policy_denied"
                        rejection_detail = (
                            f"Policy denied '{func_name}': untrusted authorization/policy keys "
                            f"{sorted(set(smuggled_pol))} rejected."
                        )
                    else:
                        pol_engine = build_execution_policy_engine(
                            policy_engine=getattr(self, "execution_policy_engine", None),
                            policy_config=getattr(self, "execution_policy_config", None),
                            zelyr_config=self._active_config(),
                            confirmation_store=getattr(self, "confirmation_store", None),
                        )
                        reg_tool_obj = (
                            self.tool_registry.get_tool(func_name)
                            if (self.tool_registry is not None and callable(getattr(self.tool_registry, "get_tool", None)))
                            else None
                        )
                        ext_entry_pol = (
                            self.tool_registry.get_external_tool_entry(func_name)
                            if (self.tool_registry is not None and callable(getattr(self.tool_registry, "get_external_tool_entry", None)))
                            else None
                        )
                        eff_schemas_pol = payload.get("tools") or []
                        matched_pol_schema = next(
                            (
                                s
                                for s in eff_schemas_pol
                                if isinstance(s, dict) and (s.get("function") or {}).get("name") == func_name
                            ),
                            None,
                        )
                        req_ring_pol = resolve_tool_ring(
                            func_name,
                            tool=reg_tool_obj,
                            tool_registry=self.tool_registry,
                            active_schemas=eff_schemas_pol,
                            task_local_scope=eff_task_local_scope if is_valid_task_scope else None,
                        )
                        is_ext_pol = bool(
                            ext_entry_pol is not None
                            or getattr(reg_tool_obj, "is_external", False)
                            or str(func_name).startswith("ext__")
                        )
                        src_id_pol = (
                            getattr(ext_entry_pol, "source_id", None)
                            if ext_entry_pol is not None
                            else getattr(reg_tool_obj, "source_id", None)
                        )
                        src_type_pol = (
                            getattr(ext_entry_pol, "source_type", None)
                            if ext_entry_pol is not None
                            else getattr(reg_tool_obj, "source_type", None)
                        )
                        ext_meta_pol = (
                            dict(getattr(ext_entry_pol, "metadata", None) or {})
                            if ext_entry_pol is not None
                            else None
                        )
                        pol_conf_map = getattr(self, "confirmation_tokens", None)
                        pol_tok_pair = pol_conf_map.get(tool_call_id) if isinstance(pol_conf_map, dict) else None
                        pol_tok = pol_tok_pair[0] if isinstance(pol_tok_pair, tuple) else (pol_tok_pair if isinstance(pol_tok_pair, str) else None)
                        pol_cid = pol_tok_pair[1] if isinstance(pol_tok_pair, tuple) and len(pol_tok_pair) > 1 else None
                        pol_dec = await pol_engine.evaluate_async(
                            func_name,
                            raw_tc_args,
                            normalized_arguments=func_args,
                            schema=matched_pol_schema,
                            tool_obj=reg_tool_obj,
                            required_ring=req_ring_pol,
                            auth_decision=decision,
                            is_registered=bool(decision.allowed),
                            is_external=is_ext_pol,
                            source_id=src_id_pol,
                            source_type=src_type_pol,
                            external_metadata=ext_meta_pol,
                            workspace_root=getattr(self, "workspace_root", None),
                            execution_id=getattr(eff_ledger, "_execution_id", None),
                            correlation_id=tool_call_id,
                            confirmation_token=pol_tok,
                            confirmation_id=pol_cid,
                            consume_confirmation=True,
                            confirm_callback=getattr(self, "confirm_callback", None),
                            source_surface="native_orchestrator.react",
                            central_ledger=getattr(eff_ledger, "_central_ledger", None),
                        )
                        if not pol_dec.allowed:
                            rejection_reason = (
                                "confirmation_required"
                                if pol_dec.outcome == PolicyDecisionOutcome.REQUIRE_CONFIRMATION
                                else "policy_denied"
                            )
                            rejection_detail = pol_dec.reason

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
                from zelyr.engine.execution_policy import authorized_policy_context

                policy_scope_cm = (
                    authorized_policy_context(pol_dec)
                    if (pol_dec is not None and pol_dec.allowed)
                    else nullcontext(None)
                )

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
                pending_vision_b64_extracted: Optional[str] = None
                ext_src_for_frame: Optional[Dict[str, Any]] = None
                _tool_t0 = time.monotonic()
                _tool_duration_ms: Optional[float] = None
                try:
                    with policy_scope_cm, ledger_scope_cm as call_ctx:
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
                    pending_vision_b64_extracted: Optional[str] = None
                    raw_res_meta = getattr(tool_result, "metadata", None)
                    if isinstance(raw_res_meta, dict):
                        for mk, mv in raw_res_meta.items():
                            if mk == "_vision_b64" and isinstance(mv, str) and mv:
                                pending_vision_b64_extracted = mv
                            else:
                                extra_telemetry[mk] = mv
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

                    def _pop_vision_b64_recursive(node: Any) -> Optional[str]:
                        if isinstance(node, dict):
                            direct = node.pop("_vision_b64", None)
                            if isinstance(direct, str) and direct:
                                return direct
                            for key in ("output", "result", "data", "metadata"):
                                if key in node:
                                    nested = _pop_vision_b64_recursive(node[key])
                                    if nested:
                                        return nested
                        return None

                    if hasattr(tool_result, "to_dict"):
                        tool_dict = tool_result.to_dict(tool=func_name, arguments=func_args)
                        v_b64 = _pop_vision_b64_recursive(tool_dict)
                        if v_b64:
                            pending_vision_b64_extracted = v_b64
                        raw_tool_payload = tool_dict
                        tool_result = json.dumps(tool_dict, indent=2)
                    elif not isinstance(tool_result, str):
                        if isinstance(tool_result, dict) and "_vision_b64" in str(tool_result)[:4096]:
                            import copy as _copy_vis

                            tool_result = _copy_vis.deepcopy(tool_result)
                            v_b64 = _pop_vision_b64_recursive(tool_result)
                            if v_b64:
                                pending_vision_b64_extracted = v_b64
                        elif isinstance(tool_result, dict) and "_vision_b64" in tool_result:
                            tool_result = dict(tool_result)
                            v_b64 = tool_result.pop("_vision_b64", None)
                            if isinstance(v_b64, str) and v_b64:
                                pending_vision_b64_extracted = v_b64
                        raw_tool_payload = tool_result
                        tool_result = json.dumps(tool_result, indent=2)
                    else:
                        stripped_str = tool_result.strip()
                        if "_vision_b64" in stripped_str and stripped_str.startswith("{") and stripped_str.endswith("}"):
                            try:
                                parsed_tr = json.loads(stripped_str)
                                v_b64 = _pop_vision_b64_recursive(parsed_tr)
                                if v_b64:
                                    pending_vision_b64_extracted = v_b64
                                    tool_result = json.dumps(parsed_tr, indent=2)
                                    raw_tool_payload = parsed_tr
                                else:
                                    raw_tool_payload = tool_result
                            except json.JSONDecodeError:
                                raw_tool_payload = tool_result
                        else:
                            raw_tool_payload = tool_result
                    if pending_vision_b64_extracted:
                        payload["_pending_vision_b64"] = pending_vision_b64_extracted
                        if eff_ledger is not None:
                            eff_ledger._cached_desktop_vision_b64 = pending_vision_b64_extracted
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
                        duration_ms=round((time.monotonic() - _tool_t0) * 1000.0, 3),
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
                finally:
                    _tool_duration_ms = round((time.monotonic() - _tool_t0) * 1000.0, 3)

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
                # manage_desktop_window's verified=True confirms WM IPC focus/dispatch only, never root outcome verification.
                is_verified = (obs_dict.get("verified") is True and func_name != "manage_desktop_window")
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
                _ver_duration_ms: Optional[float] = None
                if acceptance_spec and not is_failed:
                    from zelyr.engine.execution_ledger import (
                        _is_inspection_action,
                        canonicalize_target,
                        criterion_applies_to_target,
                        criterion_canonical_target_raw,
                        extract_file_targets_from_text,
                        has_remaining_explicit_steps,
                        parse_acceptance_criteria,
                        targets_match,
                    )
                    from zelyr.engine.verifier import OutcomeVerifier
                    _ver_t0 = time.monotonic()
                    try:
                        parsed_c = parse_acceptance_criteria(acceptance_spec)
                        all_c_targets = [criterion_canonical_target_raw(c) or c.get("target", "") for c in parsed_c if c.get("target")]
                        c_results = []
                        by_target: Dict[str, List[Any]] = {}
                        for c in parsed_c:
                            vr = await OutcomeVerifier.verify_criterion_async(c)
                            c_results.append((c, vr))
                            eff_ct = criterion_canonical_target_raw(c) or c.get("target", "")
                            norm_ct = canonicalize_target(eff_ct, check_type=c.get("check")) if eff_ct else ""
                            if norm_ct:
                                by_target.setdefault(norm_ct, []).append((c, vr))
                            if str(c.get("check", "")).lower() in ("command_output", "command", "cmd") and c.get("target"):
                                for ref_ft in extract_file_targets_from_text(str(c["target"])):
                                    norm_ft = canonicalize_target(ref_ft)
                                    if norm_ft:
                                        by_target.setdefault(norm_ft, []).append(
                                            ({"check": "file_exists", "target": ref_ft, "expected": c.get("expected")}, vr)
                                        )
                    finally:
                        _ver_duration_ms = round((time.monotonic() - _ver_t0) * 1000.0, 3)

                    for norm_ct, items in by_target.items():
                        first_c = items[0][0]
                        eff_rec_t = criterion_canonical_target_raw(first_c) or first_c["target"]
                        if all(r.success for _, r in items):
                            eff_ledger.record_target_verified(
                                eff_rec_t,
                                detail=f"{first_c['check']} verified",
                                check_type=first_c.get("check"),
                            )
                        else:
                            if hasattr(eff_ledger, "invalidate_target_verification"):
                                eff_ledger.invalidate_target_verification(
                                    eff_rec_t,
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
                        and not _is_inspection_action(func_name, func_args)
                    ):
                        if func_name in ("execute_command", "bash", "sh"):
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
                        elif tool_target:
                            # Attach unmatched command_output checks (e.g. pytest -q) immediately on mutating file edits.
                            matching_pairs = [
                                (c, r) for c, r in c_results
                                if str(c.get("check", "")).lower() in ("command_output", "command", "cmd")
                            ]

                    is_pure_inspection = (
                        _is_inspection_action(func_name, func_args)
                        and func_name != "verify_outcome"
                    )

                    if matching_pairs and all(r.success for _, r in matching_pairs):
                        if not is_pure_inspection:
                            is_verified = True
                            criteria_failed = False
                            verified_targets_for_obs = [
                                canonicalize_target(str(criterion_canonical_target_raw(c) or c.get("target")), check_type=c.get("check"))
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
                                tgt = criterion_canonical_target_raw(c) or c.get("target") or tool_target
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
                    v_check_arg = str(func_args.get("check") or "").strip().lower()
                    v_target_arg = tool_target or str(func_args.get("target") or func_args.get("window") or "").strip()
                    v_sel_arg = str(func_args.get("selector") or "").strip()
                    if v_target_arg and v_sel_arg and v_check_arg.startswith("desktop_") and "::" not in v_target_arg:
                        v_target_arg = f"{v_target_arg}::{v_sel_arg}"
                    if v_target_arg:
                        eff_ledger.record_target_verified(v_target_arg, check_type=v_check_arg or None)
                        canon_vt = canonicalize_target(v_target_arg, check_type=v_check_arg or None)
                        if canon_vt and canon_vt not in verified_targets_for_obs:
                            verified_targets_for_obs.append(canon_vt)
                elif not is_failed and not _is_inspection_action(func_name, func_args):
                    # Invalidate any previously verified targets affected by this later mutation.
                    fn_low = (func_name or "").strip().lower()
                    if not acceptance_spec and tool_target:
                        muts = self._get_mutation_targets(func_name, func_args, eff_ledger.get_verified_targets())
                        for mt in muts:
                            if not is_verified and hasattr(eff_ledger, "invalidate_target_verification"):
                                eff_ledger.invalidate_target_verification(mt)
                    if fn_low in ("interact_with_browser", "browser_cdp", "browser_navigate"):
                        if hasattr(eff_ledger, "invalidate_surface_verifications"):
                            eff_ledger.invalidate_surface_verifications(
                                "browser",
                                exclude_targets=set(verified_targets_for_obs),
                            )
                    elif fn_low in ("desktop_control", "desktop_automation", "manage_desktop_window", "interact_with_ui", "gui_actuate"):
                        if hasattr(eff_ledger, "invalidate_surface_verifications"):
                            eff_win_tgt = (
                                tool_target
                                or str(func_args.get("expected_window") or func_args.get("window") or "").strip()
                                or None
                            )
                            eff_ledger.invalidate_surface_verifications(
                                "desktop",
                                exclude_targets=set(verified_targets_for_obs),
                                window_target=eff_win_tgt,
                            )

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
                    duration_ms=_tool_duration_ms,
                    verification_duration_ms=_ver_duration_ms,
                )
                cached_vis = getattr(eff_ledger, "_cached_desktop_vision_b64", None) if eff_ledger is not None else None
                if not pending_vision_b64_extracted and cached_vis:
                    fn_low = (func_name or "").strip().lower()
                    act_low = str((func_args or {}).get("action") or "").strip().lower()
                    preserve_for_recovery = (
                        (fn_low in ("desktop_control", "desktop_automation", "interact_with_ui") and is_failed)
                        or (
                            fn_low == "manage_desktop_window"
                            and not is_failed
                            and act_low in ("focus", "list", "get_active", "monitors", "geometry", "inspect")
                        )
                    )
                    if preserve_for_recovery:
                        payload["_pending_vision_b64"] = cached_vis
                    else:
                        eff_ledger._cached_desktop_vision_b64 = None
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

                _desktop_done_with_capture = False
                if (
                    acceptance_spec
                    and not is_failed
                    and not (c_results and all(r.success for _, r in c_results))
                    and func_name in ("capture_screen", "desktop_control")
                    and parsed_c
                ):
                    from zelyr.engine.execution_ledger import _assess_desktop_verification_state

                    _d_st = _assess_desktop_verification_state(eff_ledger)
                    _desktop_done_with_capture = (
                        _d_st["last_desktop_mut_idx"] >= 0
                        and not _d_st["trailing_unobserved_mutation"]
                    )

                if acceptance_spec and not is_failed and (
                    (c_results and all(r.success for _, r in c_results))
                    or _desktop_done_with_capture
                ):
                    all_criteria_satisfied = True
                    for c in (parsed_c or []):
                        c_vr = await OutcomeVerifier.verify_criterion_async(c)
                        if not c_vr.success:
                            all_criteria_satisfied = False
                            break
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
        if tier == "tier3_vision":
            try:
                from zelyr.core.vision_engine.singleton import is_spotless_arms_available
                if not is_spotless_arms_available():
                    return "desktop_control"
            except Exception:
                return "desktop_control"
        tier_map = {
            "tier1_ipc": "manage_desktop_window",
            "tier2_browser": "interact_with_browser",
            "tier3_vision": "interact_with_ui",
        }
        return tier_map.get(tier, "desktop_control")

    @staticmethod
    def _normalize_graph_proposal(item: Any, execution_id: str, fallback_order: int = 0) -> Dict[str, Any]:
        """Normalize a PendingActionProposal or dict into a canonical proposal descriptor."""
        if hasattr(item, "to_dict") and not isinstance(item, dict):
            d = item.to_dict()
        elif isinstance(item, dict):
            d = dict(item)
        else:
            raise ValueError(f"Unsupported graph proposal entry: {type(item).__name__}")

        prop_exec = d.get("execution_id")
        if prop_exec and str(prop_exec) != execution_id:
            from zelyr.engine.task_graph import TaskGraphValidationError

            raise TaskGraphValidationError(
                f"Cross-execution proposal '{d.get('proposal_id') or d.get('original_call_id')}' "
                f"belongs to '{prop_exec}', not '{execution_id}'."
            )

        pid = str(d.get("proposal_id") or "").strip()
        cid = str(d.get("original_call_id") or d.get("id") or "").strip()
        if not cid and pid:
            cid = f"call_{pid}"
        if not pid and cid:
            pid = f"prop_{cid}"
        if not pid and not cid:
            raise ValueError("Graph proposal entry must define 'proposal_id' or 'original_call_id'.")

        tool_id = str(d.get("tool_id") or d.get("tool_name") or "").strip()
        raw_args = d.get("normalized_arguments") if "normalized_arguments" in d else d.get("arguments", {})
        if isinstance(raw_args, str):
            try:
                parsed_a = json.loads(raw_args)
                args_val: Any = parsed_a if isinstance(parsed_a, dict) else raw_args
            except Exception:
                args_val = raw_args
        elif isinstance(raw_args, dict):
            args_val = dict(raw_args)
        else:
            args_val = raw_args

        return {
            "proposal_id": pid,
            "original_call_id": cid,
            "execution_id": execution_id,
            "tool_id": tool_id,
            "raw_name": str(d.get("raw_name") or tool_id),
            "arguments": args_val,
            "normalized_arguments": dict(args_val) if isinstance(args_val, dict) else {},
            "status": str(d.get("status") or "pending"),
            "depends_on": list(d.get("depends_on") or []),
            "external_source": dict(d["external_source"]) if isinstance(d.get("external_source"), dict) else None,
            "expires_at": d.get("expires_at"),
            "proposal_order": int(d.get("proposal_order", fallback_order) or 0),
            "node_id": d.get("node_id"),
            "acceptance_criteria": d.get("acceptance_criteria"),
        }

    def _check_live_external_provenance(self, tool_name: str, ext_source: Optional[Dict[str, Any]]) -> Optional[str]:
        """Revalidate external/fallback tool provenance and availability against live registry."""
        reg = getattr(self, "tool_registry", None)
        ext_entry = None
        if reg is not None and callable(getattr(reg, "get_external_tool_entry", None)):
            try:
                ext_entry = reg.get_external_tool_entry(tool_name)
            except Exception:
                ext_entry = None
        is_ext = bool((ext_source and ext_source.get("is_external", True)) or ext_entry is not None)
        if not is_ext or reg is None:
            return None
        if ext_entry is None:
            return f"External tool '{tool_name}' is no longer registered or available in runtime registry."
        if getattr(ext_entry, "authorized", True) is False:
            return f"External tool '{tool_name}' authorization has been revoked."
        if callable(getattr(reg, "get_tool", None)) and reg.get_tool(tool_name) is None:
            return f"External tool '{tool_name}' adapter is unavailable."
        return None

    async def _run_reserved_graph_node(
        self,
        *,
        graph: Any,
        node_id: str,
        call_id: str,
        proposal: Dict[str, Any],
        owner_id: str,
        reservation_id: str,
        central_ledger: Any,
        turn_ledger: Any,
        prop_by_ref: Dict[str, Dict[str, Any]],
        eff_exec_id: str,
        eff_caller_ring: Optional[int],
        eff_allowed_tools: Optional[List[str]],
        eff_task_local_scope: Optional[Any],
        auth_schemas: Optional[List[Dict[str, Any]]],
        known_for_auth: Optional[set],
        eff_tool_executor: Optional[Callable[[str, dict], Awaitable[Any]]],
        parsed_criteria: List[Dict[str, Any]],
        node_timeout: float,
        verifier_fn: Optional[Callable[..., Any]] = None,
        policy_engine: Optional[Any] = None,
        policy_config: Optional[Any] = None,
        confirm_callback: Optional[Callable[[Any], Any]] = None,
        node_confirmation: Optional[Tuple[Optional[str], Optional[str]]] = None,
        workspace_root: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Execute a single reserved TaskNode inside an isolated task-local & ledger execution scope."""
        from zelyr.engine.execution_ledger import (
            ActionState,
            LedgerEventType,
            _extract_action_target,
            criterion_applies_to_target,
            execution_scope,
            parse_acceptance_criteria,
        )
        from zelyr.engine.observation_framing import ObservationFramingEngine
        from zelyr.engine.task_graph import TaskNodeStatus
        from zelyr.engine.verifier import OutcomeVerifier
        from zelyr.tool_registry import activate_task_local_scope, authorize_and_validate_tool_call

        func_name = proposal["tool_id"]
        func_args = dict(proposal.get("normalized_arguments") or {})
        ext_src = proposal.get("external_source")
        did_dispatch = False
        was_timed_out = False
        was_cancelled = False
        structured_obs: Any = None
        outcome_passed: Optional[bool] = None
        is_failed = False
        is_verified = False

        try:
            # Verify live ownership before executing
            node_cur = graph.get_node(node_id)
            if (
                node_cur is None
                or node_cur.reserved_by != owner_id
                or node_cur.reservation_id != reservation_id
            ):
                raise PermissionError(
                    f"Node '{node_id}' reservation ownership mismatch for '{owner_id}'."
                )

            # Live pre-dispatch re-authorization inside isolated worker right before execution
            ext_err = self._check_live_external_provenance(func_name, ext_src)
            if ext_err is not None:
                err_payload = {
                    "tool": func_name,
                    "status": "rejected",
                    "reason": "authorization_denied",
                    "detail": ext_err,
                    "error_type": "AUTHORIZATION_DENIED",
                }
                structured_obs = ObservationFramingEngine.frame_structured(
                    func_name,
                    err_payload,
                    requested_tool=func_name,
                    requested_args=func_args,
                    executed_args=func_args,
                    dispatched=False,
                    execution_id=eff_exec_id,
                    correlation_id=call_id,
                    external_source=ext_src,
                )
                turn_ledger.record_rejected(call_id, reason=ext_err)
                turn_ledger.record_observation(
                    call_id,
                    status="rejected",
                    output=json.dumps(err_payload),
                    error=ext_err,
                    rejection_reason="authorization_denied",
                    structured_observation=structured_obs,
                )
                is_failed = True
                return {"node_id": node_id, "call_id": call_id, "status": "rejected", "error": ext_err}

            if eff_task_local_scope is not None and eff_tool_executor is None:
                rej_detail = (
                    "AuthorizationError: TaskLocalToolScope requires an isolated custom "
                    "tool_executor and cannot dispatch to production execute_tool."
                )
                err_payload = {
                    "tool": func_name,
                    "status": "rejected",
                    "reason": "authorization_denied",
                    "detail": rej_detail,
                    "error_type": "AUTHORIZATION_DENIED",
                }
                structured_obs = ObservationFramingEngine.frame_structured(
                    func_name,
                    err_payload,
                    requested_tool=func_name,
                    requested_args=func_args,
                    executed_args=func_args,
                    dispatched=False,
                    execution_id=eff_exec_id,
                    correlation_id=call_id,
                    external_source=ext_src,
                )
                turn_ledger.record_rejected(call_id, reason=rej_detail)
                turn_ledger.record_observation(
                    call_id,
                    status="rejected",
                    output=json.dumps(err_payload),
                    error=rej_detail,
                    rejection_reason="authorization_denied",
                    structured_observation=structured_obs,
                )
                is_failed = True
                return {"node_id": node_id, "call_id": call_id, "status": "rejected", "error": rej_detail}

            decision = authorize_and_validate_tool_call(
                func_name,
                proposal.get("arguments", func_args),
                caller_ring=eff_caller_ring,
                allowed_tools=eff_allowed_tools,
                active_schemas=auth_schemas,
                tool_registry=self.tool_registry,
                known_tools=known_for_auth,
                task_local_scope=eff_task_local_scope,
                require_registered=True,
                normalize_args=True,
                validate_schema=True,
                argument_source="internal",
            )
            if not decision.allowed:
                rej_reason = decision.reason or "authorization_denied"
                rej_detail = decision.detail or "Tool invocation denied at worker dispatch."
                err_type = decision.error_type or "AUTHORIZATION_DENIED"
                err_payload = {
                    "tool": func_name,
                    "status": "rejected",
                    "reason": rej_reason,
                    "detail": rej_detail,
                    "error_type": err_type,
                }
                if isinstance(getattr(decision, "validation_error", None), dict):
                    for k, v in decision.validation_error.items():
                        err_payload.setdefault(k, v)
                structured_obs = ObservationFramingEngine.frame_structured(
                    func_name,
                    err_payload,
                    requested_tool=func_name,
                    requested_args=func_args,
                    executed_args=func_args,
                    dispatched=False,
                    execution_id=eff_exec_id,
                    correlation_id=call_id,
                    external_source=ext_src,
                )
                turn_ledger.record_rejected(call_id, reason=rej_detail)
                turn_ledger.record_observation(
                    call_id,
                    status="rejected",
                    output=json.dumps(err_payload),
                    error=rej_detail,
                    rejection_reason=rej_reason,
                    structured_observation=structured_obs,
                )
                is_failed = True
                return {"node_id": node_id, "call_id": call_id, "status": "rejected", "error": rej_detail}

            func_name = decision.tool_name
            func_args = dict(decision.normalized_arguments)

            # Unified Execution Policy evaluation right after runtime auth & normalization
            from zelyr.engine.execution_policy import (
                PolicyDecisionOutcome,
                UNTRUSTED_POLICY_OVERRIDE_KEYS,
                authorized_policy_context,
                build_execution_policy_engine,
            )
            from zelyr.tool_registry import resolve_tool_ring

            raw_prop_args = proposal.get("arguments", func_args)
            smuggled_node_pol = (
                [str(k) for k in raw_prop_args.keys() if str(k).strip() in UNTRUSTED_POLICY_OVERRIDE_KEYS]
                if isinstance(raw_prop_args, dict)
                else []
            )
            eff_pol_engine = build_execution_policy_engine(
                policy_engine=policy_engine or getattr(self, "execution_policy_engine", None),
                policy_config=policy_config or getattr(self, "execution_policy_config", None),
                zelyr_config=self._active_config(),
                confirmation_store=getattr(self, "confirmation_store", None),
            )
            reg_tool_pol = (
                self.tool_registry.get_tool(func_name)
                if (self.tool_registry is not None and callable(getattr(self.tool_registry, "get_tool", None)))
                else None
            )
            ext_entry_pol = (
                self.tool_registry.get_external_tool_entry(func_name)
                if (self.tool_registry is not None and callable(getattr(self.tool_registry, "get_external_tool_entry", None)))
                else None
            )
            matched_node_schema = next(
                (
                    s
                    for s in (auth_schemas or [])
                    if isinstance(s, dict) and (s.get("function") or {}).get("name") == func_name
                ),
                None,
            )
            req_ring_node = resolve_tool_ring(
                func_name,
                tool=reg_tool_pol,
                tool_registry=self.tool_registry,
                active_schemas=auth_schemas,
                task_local_scope=eff_task_local_scope,
            )
            is_ext_node = bool(
                (ext_src and ext_src.get("is_external"))
                or ext_entry_pol is not None
                or getattr(reg_tool_pol, "is_external", False)
                or str(func_name).startswith("ext__")
            )
            src_id_node = (
                ext_src.get("source_id") if isinstance(ext_src, dict) and ext_src.get("source_id") else (
                    getattr(ext_entry_pol, "source_id", None)
                    if ext_entry_pol is not None
                    else getattr(reg_tool_pol, "source_id", None)
                )
            )
            src_type_node = (
                ext_src.get("source_type") if isinstance(ext_src, dict) and ext_src.get("source_type") else (
                    getattr(ext_entry_pol, "source_type", None)
                    if ext_entry_pol is not None
                    else getattr(reg_tool_pol, "source_type", None)
                )
            )
            ext_meta_node = (
                dict(getattr(ext_entry_pol, "metadata", None) or {})
                if ext_entry_pol is not None
                else None
            )
            conf_tok_node = node_confirmation[0] if isinstance(node_confirmation, tuple) and len(node_confirmation) > 0 else None
            conf_id_node = node_confirmation[1] if isinstance(node_confirmation, tuple) and len(node_confirmation) > 1 else None

            if smuggled_node_pol:
                pol_dec_node = eff_pol_engine.evaluate(
                    func_name,
                    raw_prop_args if isinstance(raw_prop_args, dict) else func_args,
                    normalized_arguments=func_args,
                    schema=matched_node_schema,
                    tool_obj=reg_tool_pol,
                    required_ring=req_ring_node,
                    auth_decision=decision,
                    is_registered=True,
                    is_external=is_ext_node,
                    source_id=src_id_node,
                    source_type=src_type_node,
                    external_metadata=ext_meta_node,
                    workspace_root=workspace_root or getattr(self, "workspace_root", None),
                    node_target_resources=getattr(node_cur, "target_resources", ()),
                    execution_id=eff_exec_id,
                    node_id=node_id,
                    correlation_id=call_id,
                    source_surface="native_orchestrator.task_graph",
                    central_ledger=central_ledger,
                )
            else:
                pol_dec_node = await eff_pol_engine.evaluate_async(
                    func_name,
                    raw_prop_args if isinstance(raw_prop_args, dict) else func_args,
                    normalized_arguments=func_args,
                    schema=matched_node_schema,
                    tool_obj=reg_tool_pol,
                    required_ring=req_ring_node,
                    auth_decision=decision,
                    is_registered=True,
                    is_external=is_ext_node,
                    source_id=src_id_node,
                    source_type=src_type_node,
                    external_metadata=ext_meta_node,
                    workspace_root=workspace_root or getattr(self, "workspace_root", None),
                    node_target_resources=getattr(node_cur, "target_resources", ()),
                    execution_id=eff_exec_id,
                    node_id=node_id,
                    correlation_id=call_id,
                    confirmation_token=conf_tok_node,
                    confirmation_id=conf_id_node,
                    consume_confirmation=True,
                    confirm_callback=confirm_callback or getattr(self, "confirm_callback", None),
                    source_surface="native_orchestrator.task_graph",
                    central_ledger=central_ledger,
                )

            if not pol_dec_node.allowed:
                err_payload = pol_dec_node.to_tool_error_payload()
                err_payload["status"] = "rejected"
                err_payload["detail"] = pol_dec_node.reason
                structured_obs = ObservationFramingEngine.frame_structured(
                    func_name,
                    err_payload,
                    requested_tool=func_name,
                    requested_args=func_args,
                    executed_args=func_args,
                    dispatched=False,
                    execution_id=eff_exec_id,
                    correlation_id=call_id,
                    external_source=ext_src,
                )
                if turn_ledger.get_record(call_id) is None:
                    turn_ledger.record_proposed(
                        call_id,
                        func_name,
                        func_args,
                        proposal_id=proposal["proposal_id"],
                        depends_on=proposal.get("depends_on"),
                        external_source=ext_src,
                    )
                turn_ledger.record_rejected(call_id, reason=pol_dec_node.reason)
                turn_ledger.record_observation(
                    call_id,
                    status="rejected",
                    output=json.dumps(err_payload),
                    error=pol_dec_node.reason,
                    rejection_reason=err_payload["reason"],
                    structured_observation=structured_obs,
                )
                is_failed = True
                return {
                    "node_id": node_id,
                    "call_id": call_id,
                    "tool_name": func_name,
                    "status": "rejected",
                    "error": pol_dec_node.reason,
                    "error_type": err_payload["error_type"],
                    "policy_decision": pol_dec_node.to_safe_dict(),
                    "dispatched": False,
                }

            cur_rec = turn_ledger.get_record(call_id)
            if cur_rec is None:
                turn_ledger.record_proposed(
                    call_id,
                    func_name,
                    func_args,
                    proposal_id=proposal["proposal_id"],
                    depends_on=proposal.get("depends_on"),
                    external_source=ext_src,
                )
            cur_rec = turn_ledger.get_record(call_id)
            if cur_rec is not None and cur_rec.state == ActionState.PROPOSED:
                turn_ledger.record_validated(call_id)
                turn_ledger.record_authorized(call_id)
            elif cur_rec is not None and cur_rec.state == ActionState.VALIDATED:
                turn_ledger.record_authorized(call_id)

            # Pre-dispatch transport/tool readiness check before marking dispatched (failure here is retry-safe)
            reg_tool_pre = (
                self.tool_registry.get_tool(func_name)
                if (self.tool_registry is not None and callable(getattr(self.tool_registry, "get_tool", None)))
                else None
            )
            pre_hook = (
                getattr(eff_tool_executor, "pre_dispatch", None)
                if eff_tool_executor is not None
                else None
            ) or (getattr(reg_tool_pre, "pre_dispatch", None) if reg_tool_pre is not None else None)
            if callable(pre_hook):
                try:
                    pre_out = pre_hook(func_name, func_args)
                    if inspect.isawaitable(pre_out):
                        pre_out = await asyncio.wait_for(pre_out, timeout=node_timeout)
                    if isinstance(pre_out, dict) and pre_out.get("status") in ("error", "failed", "rejected"):
                        pre_err = str(pre_out.get("error") or pre_out.get("detail") or "Pre-dispatch readiness failed")
                        pre_type = str(pre_out.get("error_type") or "FAILED_BEFORE_EXECUTION")
                        err_env = {"status": "error", "error": pre_err, "error_type": pre_type}
                        structured_obs = ObservationFramingEngine.frame_structured(
                            func_name,
                            err_env,
                            requested_tool=func_name,
                            requested_args=func_args,
                            executed_args=func_args,
                            dispatched=False,
                            execution_id=eff_exec_id,
                            correlation_id=call_id,
                            external_source=ext_src,
                        )
                        turn_ledger.record_observation(
                            call_id,
                            status="error",
                            output="",
                            error=pre_err,
                            verified=False,
                            structured_observation=structured_obs,
                        )
                        is_failed = True
                        return {
                            "node_id": node_id,
                            "call_id": call_id,
                            "tool_name": func_name,
                            "status": "error",
                            "error_type": pre_type,
                            "dispatched": False,
                            "error": pre_err,
                        }
                except asyncio.CancelledError:
                    raise
                except Exception as pre_exc:
                    pre_err = f"{type(pre_exc).__name__}: {pre_exc}"
                    err_env = {"status": "error", "error": pre_err, "error_type": "FAILED_BEFORE_EXECUTION"}
                    structured_obs = ObservationFramingEngine.frame_structured(
                        func_name,
                        err_env,
                        requested_tool=func_name,
                        requested_args=func_args,
                        executed_args=func_args,
                        dispatched=False,
                        execution_id=eff_exec_id,
                        correlation_id=call_id,
                        external_source=ext_src,
                    )
                    turn_ledger.record_observation(
                        call_id,
                        status="error",
                        output="",
                        error=pre_err,
                        verified=False,
                        structured_observation=structured_obs,
                    )
                    is_failed = True
                    return {
                        "node_id": node_id,
                        "call_id": call_id,
                        "tool_name": func_name,
                        "status": "error",
                        "error_type": "FAILED_BEFORE_EXECUTION",
                        "dispatched": False,
                        "error": pre_err,
                    }

            turn_ledger.record_dispatched(call_id, emit_event=True)
            did_dispatch = True

            raw_res: Any = None
            _node_tool_t0 = time.monotonic()
            with authorized_policy_context(pol_dec_node), execution_scope(
                central_ledger,
                eff_exec_id,
                caller_ring=eff_caller_ring,
                task_local_scoped=bool(eff_task_local_scope is not None),
                source_component="graph_parallel",
                correlation_id=call_id,
                owned_by_outer=True,
            ):
                with activate_task_local_scope(eff_task_local_scope):
                    if eff_tool_executor is not None:
                        call_obj = eff_tool_executor(func_name, func_args)
                        if inspect.isawaitable(call_obj):
                            raw_res = await asyncio.wait_for(call_obj, timeout=node_timeout)
                        else:
                            raw_res = call_obj
                    else:
                        exec_fn = self.execute_tool
                        dispatched_via_reg = False
                        if (
                            exec_fn is execute_tool
                            and self.tool_registry is not None
                            and callable(getattr(self.tool_registry, "execute_async", None))
                        ):
                            try:
                                from zelyr.core.plugins import _executors as _plugin_executors

                                reg_tool = (
                                    self.tool_registry.get_tool(func_name)
                                    if callable(getattr(self.tool_registry, "get_tool", None))
                                    else None
                                )
                                if reg_tool is not None and (
                                    getattr(reg_tool, "is_external", False) or func_name not in _plugin_executors
                                ):
                                    dispatched_via_reg = True
                            except Exception:
                                reg_tool = None
                        if dispatched_via_reg:
                            coro = self.tool_registry.execute_async(
                                func_name,
                                caller_ring=eff_caller_ring,
                                allowed_tools=eff_allowed_tools,
                                **func_args,
                            )
                        else:
                            coro = exec_fn(func_name, func_args)
                        if inspect.isawaitable(coro):
                            raw_res = await asyncio.wait_for(coro, timeout=node_timeout)
                        else:
                            raw_res = coro
            _node_tool_duration_ms = round((time.monotonic() - _node_tool_t0) * 1000.0, 3)

            # Normalize tool output payload
            if raw_res is not None and hasattr(raw_res, "success") and not isinstance(raw_res, dict):
                res_ok = bool(getattr(raw_res, "success", False))
                raw_payload: Dict[str, Any] = {
                    "status": "success" if res_ok else "error",
                    "success": res_ok,
                    "output": getattr(raw_res, "output", ""),
                }
                if getattr(raw_res, "error", None):
                    raw_payload["error"] = getattr(raw_res, "error")
                if getattr(raw_res, "error_type", None):
                    raw_payload["error_type"] = getattr(raw_res, "error_type")
            elif isinstance(raw_res, dict):
                raw_payload = dict(raw_res)
            elif isinstance(raw_res, str):
                trimmed = raw_res.strip()
                parsed_d = None
                if trimmed.startswith("{") and trimmed.endswith("}"):
                    try:
                        cand = json.loads(trimmed)
                        if isinstance(cand, dict):
                            parsed_d = cand
                    except Exception:
                        parsed_d = None
                raw_payload = parsed_d if parsed_d is not None else {"status": "success", "output": raw_res}
            else:
                raw_payload = {"status": "success", "output": str(raw_res) if raw_res is not None else ""}

            exit_code = raw_payload.get("exit_code")
            err_msg = raw_payload.get("error")
            is_failed = (
                raw_payload.get("status") in ("error", "failed", "rejected", "tool_execution_failed")
                or raw_payload.get("success") is False
                or (exit_code is not None and not isinstance(exit_code, bool) and exit_code != 0)
            )
            if is_failed and not err_msg:
                err_msg = str(raw_payload.get("detail") or raw_payload.get("output") or "Tool execution failed")

            # Outcome verification matching node acceptance criteria, verification_requirement, or target
            _SUPPORTED_OUTCOME_CHECKS = frozenset(
                {
                    "file_exists",
                    "exists",
                    "file_contains",
                    "contains",
                    "file_checksum",
                    "checksum",
                    "sha256",
                    "file_json",
                    "json",
                    "process_running",
                    "port_listening",
                    "command_output",
                }
            )
            node_spec = proposal.get("acceptance_criteria")
            node_criteria = list(parse_acceptance_criteria(str(node_spec))) if node_spec else []
            if not node_criteria and node_cur is not None and getattr(node_cur, "verification_requirement", None):
                for crit in parse_acceptance_criteria(str(node_cur.verification_requirement)):
                    if (
                        isinstance(crit, dict)
                        and str(crit.get("check") or "").strip().lower() in _SUPPORTED_OUTCOME_CHECKS
                        and str(crit.get("target") or "").strip()
                    ):
                        node_criteria.append(crit)
            tool_target = _extract_action_target(func_name, func_args) or (
                str(raw_payload.get("target")) if isinstance(raw_payload.get("target"), str) else None
            )
            applicable_criteria = list(node_criteria)
            if parsed_criteria and tool_target:
                for crit in parsed_criteria:
                    if criterion_applies_to_target(crit, tool_target) and crit not in applicable_criteria:
                        applicable_criteria.append(crit)

            criteria_failed = False
            ver_meta: Optional[Dict[str, Any]] = None
            verified_targets_for_obs: List[str] = []
            failed_targets_for_obs: List[str] = []
            verification_detail: Optional[str] = None
            _node_ver_duration_ms: Optional[float] = None

            if not is_failed:
                if applicable_criteria:
                    v_fn = verifier_fn or OutcomeVerifier.verify
                    c_results = []
                    _node_ver_t0 = time.monotonic()
                    try:
                        for crit in applicable_criteria:
                            vr = v_fn(
                                check=crit["check"],
                                target=crit["target"],
                                expected=crit.get("expected"),
                            )
                            c_results.append((crit, vr))
                    finally:
                        _node_ver_duration_ms = round((time.monotonic() - _node_ver_t0) * 1000.0, 3)
                    failed_c = [
                        (c, r) for c, r in c_results if not getattr(r, "success", False)
                    ]
                    if failed_c:
                        criteria_failed = True
                        is_verified = False
                        outcome_passed = False
                        first_c, first_r = failed_c[0]
                        verification_detail = getattr(first_r, "error", None) or "Outcome verification failed."
                        ver_meta = {
                            "verified": False,
                            "passed": False,
                            "failed_criterion": f"{first_c['check']}:{first_c['target']}",
                            "target": str(first_c.get("target") or ""),
                            "error": verification_detail,
                        }
                        for fc, _ in failed_c:
                            tgt_s = str(fc.get("target") or "")
                            if tgt_s and tgt_s not in failed_targets_for_obs:
                                failed_targets_for_obs.append(tgt_s)
                            if tgt_s:
                                turn_ledger.invalidate_target_verification(tgt_s, check_type=fc.get("check"))
                    else:
                        is_verified = True
                        outcome_passed = True
                        verification_detail = "All matching acceptance criteria independently verified."
                        ver_meta = {"verified": True, "passed": True, "detail": verification_detail}
                        for vc, _ in c_results:
                            tgt_s = str(vc.get("target") or "")
                            if tgt_s:
                                if tgt_s not in verified_targets_for_obs:
                                    verified_targets_for_obs.append(tgt_s)
                                turn_ledger.record_target_verified(
                                    tgt_s,
                                    detail=verification_detail,
                                    check_type=vc.get("check"),
                                )
                elif raw_payload.get("verified") is True and func_name != "manage_desktop_window":
                    is_verified = True
                    outcome_passed = True
                    verification_detail = str(raw_payload.get("verification_detail") or "Verified")
                    ver_meta = {"verified": True, "passed": True, "detail": verification_detail}
                    if tool_target:
                        verified_targets_for_obs.append(tool_target)
                        turn_ledger.record_target_verified(tool_target, detail=verification_detail)
                elif raw_payload.get("verified") is False and raw_payload.get("verification_failed") is True:
                    criteria_failed = True
                    is_verified = False
                    outcome_passed = False
                    verification_detail = str(raw_payload.get("verifier_reason") or raw_payload.get("verification_detail") or "Verification failed")
                    ver_meta = {
                        "verified": False,
                        "passed": False,
                        "error": verification_detail,
                        "target": tool_target or "",
                    }
                    if tool_target:
                        failed_targets_for_obs.append(tool_target)
                        turn_ledger.invalidate_target_verification(tool_target)
            else:
                if tool_target:
                    turn_ledger.invalidate_target_verification(tool_target)

            structured_obs = ObservationFramingEngine.frame_structured(
                func_name,
                raw_payload,
                requested_tool=func_name,
                requested_args=func_args,
                executed_args=func_args,
                dispatched=True,
                timed_out=False,
                cancelled=False,
                execution_id=eff_exec_id,
                correlation_id=call_id,
                external_source=ext_src,
                independent_verification=ver_meta,
            )
            turn_ledger.record_observation(
                call_id,
                status="error" if is_failed else "success",
                output=raw_payload.get("output") if raw_payload.get("output") is not None else raw_payload,
                error=str(err_msg) if is_failed else (verification_detail if criteria_failed else None),
                exit_code=exit_code if not isinstance(exit_code, bool) else None,
                verified=is_verified,
                verification_detail=verification_detail,
                verification_failed=bool(criteria_failed and not is_failed),
                failed_targets=failed_targets_for_obs,
                verified_targets=verified_targets_for_obs,
                structured_observation=structured_obs,
                duration_ms=_node_tool_duration_ms,
                verification_duration_ms=_node_ver_duration_ms,
            )
            return {
                "node_id": node_id,
                "call_id": call_id,
                "tool_name": func_name,
                "status": "error" if is_failed else ("verified" if is_verified else "unverified"),
                "verified": is_verified,
                "error": str(err_msg) if is_failed else None,
                "output": raw_payload.get("output"),
            }
        except asyncio.TimeoutError:
            was_timed_out = True
            is_failed = True
            timeout_err = f"Node '{node_id}' tool '{func_name}' timed out after {node_timeout}s."
            structured_obs = ObservationFramingEngine.frame_structured(
                func_name,
                {"status": "error", "error": timeout_err, "error_type": "TIMEOUT_ERROR"},
                requested_tool=func_name,
                requested_args=func_args,
                executed_args=func_args,
                dispatched=did_dispatch,
                timed_out=True,
                cancelled=False,
                execution_id=eff_exec_id,
                correlation_id=call_id,
                external_source=ext_src,
            )
            turn_ledger.record_observation(
                call_id,
                status="timed_out",
                output="",
                error=timeout_err,
                verified=False,
                structured_observation=structured_obs,
                duration_ms=round((time.monotonic() - _node_tool_t0) * 1000.0, 3),
            )
            return {"node_id": node_id, "call_id": call_id, "status": "timed_out", "error": timeout_err}
        except asyncio.CancelledError:
            was_cancelled = True
            cancel_err = f"Node '{node_id}' execution cancelled while {'running' if did_dispatch else 'pending'}."
            if turn_ledger.get_record(call_id) is None:
                turn_ledger.record_proposed(call_id, func_name, func_args, proposal_id=proposal["proposal_id"], external_source=ext_src)
            turn_ledger.record_cancelled(call_id, reason=cancel_err)
            structured_obs = ObservationFramingEngine.frame_structured(
                func_name,
                {"status": "cancelled", "error": cancel_err},
                requested_tool=func_name,
                requested_args=func_args,
                executed_args=func_args,
                dispatched=did_dispatch,
                timed_out=False,
                cancelled=True,
                execution_id=eff_exec_id,
                correlation_id=call_id,
                external_source=ext_src,
            )
            turn_ledger._structured_observations[call_id] = structured_obs
            raise
        except Exception as exc:
            is_failed = True
            exc_err = f"{type(exc).__name__}: {exc}"
            structured_obs = ObservationFramingEngine.frame_structured(
                func_name,
                {"status": "error", "error": exc_err},
                requested_tool=func_name,
                requested_args=func_args,
                executed_args=func_args,
                dispatched=did_dispatch,
                timed_out=False,
                cancelled=False,
                execution_id=eff_exec_id,
                correlation_id=call_id,
                external_source=ext_src,
            )
            if turn_ledger.get_record(call_id) is None:
                turn_ledger.record_proposed(call_id, func_name, func_args, proposal_id=proposal["proposal_id"], external_source=ext_src)
            turn_ledger.record_observation(
                call_id,
                status="error",
                output="",
                error=exc_err,
                verified=False,
                structured_observation=structured_obs,
            )
            return {"node_id": node_id, "call_id": call_id, "status": "error", "error": exc_err}
        finally:
            # Always release only OUR reservation & transition graph atomically under graph._lock
            with graph._lock:
                node_chk = graph.get_node(node_id)
                if (
                    node_chk is not None
                    and node_chk.reserved_by == owner_id
                    and node_chk.reservation_id == reservation_id
                ):
                    temp_status = (
                        TaskNodeStatus.CANCELLED.value
                        if was_cancelled
                        else (TaskNodeStatus.FAILED.value if (is_failed or was_timed_out) else TaskNodeStatus.PENDING.value)
                    )
                    graph.release_reservation(
                        node_id,
                        owner_id=owner_id,
                        reservation_id=reservation_id,
                        final_status=temp_status,
                        known_proposals=prop_by_ref,
                    )
                    turn_ledger.record_node_reservation_event(
                        "reservation_released",
                        node_id=node_id,
                        owner_id=owner_id,
                        reservation_id=reservation_id,
                        call_id=call_id,
                        tool_id=func_name,
                        status="released",
                        idempotency_suffix=f"release:{reservation_id}",
                    )
                if structured_obs is not None:
                    graph.apply_observation(
                        node_id,
                        structured_obs,
                        action_record=turn_ledger.get_record(call_id),
                        outcome_verifier_passed=outcome_passed,
                        known_proposals=prop_by_ref,
                    )
                if was_cancelled and node_chk is not None:
                    prev_rs = dict(node_chk.recovery_state) if isinstance(node_chk.recovery_state, dict) else {}
                    hist = list(prev_rs.get("recovery_history") or [])
                    hist.append(
                        {
                            "attempt": int(prev_rs.get("attempt_count") or 0),
                            "timestamp": time.time(),
                            "failure_category": "CANCELLED",
                            "decision": "blocked",
                            "previous_status": TaskNodeStatus.RUNNING.value,
                            "new_status": TaskNodeStatus.CANCELLED.value,
                            "reason": "Node cancelled while active; preserve recovery history and potential side effects.",
                            "verified": False,
                        }
                    )
                    node_chk.recovery_state = {
                        "recovery_version": 1,
                        "attempt_count": int(prev_rs.get("attempt_count") or 0),
                        **({"max_attempts": int(prev_rs["max_attempts"])} if prev_rs.get("max_attempts") is not None else {}),
                        **({"retry_deadline": float(prev_rs["retry_deadline"])} if prev_rs.get("retry_deadline") is not None else {}),
                        "recovery_decision": "blocked",
                        "recovery_status": "cancelled",
                        "previous_failure_category": "CANCELLED",
                        "may_have_side_effect": bool(did_dispatch),
                        "execution_started": bool(did_dispatch),
                        "reconciliation_required": bool(did_dispatch),
                        "verified_outcome": False,
                        "next_recovery_action": "manual_intervention_or_replan",
                        "recovery_history": hist[-24:],
                    }

    async def execute_task_graph(
        self,
        graph: Any,
        proposals: Optional[Union[Dict[str, Any], Sequence[Any]]] = None,
        *,
        max_concurrency: Optional[int] = None,
        execution_id: Optional[str] = None,
        central_ledger: Optional[Any] = None,
        turn_ledger: Optional[Any] = None,
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
        tool_executor: Optional[Callable[[str, dict], Awaitable[Any]]] = None,
        caller_ring: Optional[int] = None,
        allowed_tools: Optional[Iterable[str]] = None,
        task_local_scope: Optional[Any] = None,
        acceptance_criteria: Optional[str] = None,
        node_timeouts: Optional[Dict[str, float]] = None,
        default_node_timeout: Optional[float] = None,
        max_steps: Optional[int] = None,
        allow_unverified_retry: bool = False,
        finalize_execution: bool = True,
        task_prompt: Optional[str] = None,
        enable_recovery: bool = True,
        recovery_policy: Optional[Any] = None,
        verifier_fn: Optional[Callable[..., Any]] = None,
        reconcile_on_recovery: bool = False,
        confirm_recovery: bool = False,
        policy_engine: Optional[Any] = None,
        policy_config: Optional[Any] = None,
        confirm_callback: Optional[Callable[[Any], Any]] = None,
        node_confirmations: Optional[Dict[str, Tuple[Optional[str], Optional[str]]]] = None,
        workspace_root: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Execute READY independent nodes in a validated TaskGraph with bounded parallelism."""
        from zelyr.engine.checkpoint import CheckpointManager
        from zelyr.engine.execution_ledger import (
            ActionState,
            ExecutionStatus,
            LedgerEventType,
            TurnExecutionLedger,
            VerificationStatus,
            get_default_central_ledger,
            parse_acceptance_criteria,
        )
        from zelyr.engine.observation_framing import ObservationFramingEngine
        from zelyr.engine.recovery import (
            RecoveryEngine,
            RecoveryPolicy,
            build_graph_recovery_overview,
        )
        from zelyr.engine.task_graph import (
            TaskGraph,
            TaskGraphConflictError,
            TaskGraphStatus,
            TaskGraphValidationError,
            TaskNodeStatus,
            extract_node_conflict_profile,
            profiles_conflict,
        )
        from zelyr.tool_registry import authorize_and_validate_tool_call

        if not isinstance(graph, TaskGraph):
            raise TaskGraphValidationError("execute_task_graph requires a TaskGraph instance.")

        eff_exec_id = str(execution_id or graph.execution_id or "").strip()
        if not eff_exec_id:
            raise TaskGraphValidationError("execute_task_graph requires a non-empty execution_id.")
        if graph.execution_id and graph.execution_id != eff_exec_id:
            raise TaskGraphValidationError(
                f"Cross-execution isolation violation: TaskGraph '{graph.graph_id}' belongs to '{graph.execution_id}', not '{eff_exec_id}'."
            )
        graph.execution_id = eff_exec_id

        eff_max_conc = (
            max_concurrency
            if max_concurrency is not None
            else (graph.max_concurrency if getattr(graph, "max_concurrency", 1) > 1 else getattr(self, "max_concurrency", 1))
        )
        if not isinstance(eff_max_conc, int) or isinstance(eff_max_conc, bool) or eff_max_conc < 1:
            raise TaskGraphValidationError(
                f"max_concurrency must be a positive integer >= 1, got {eff_max_conc!r}."
            )
        graph.configure_max_concurrency(eff_max_conc)

        eff_caller_ring = caller_ring if caller_ring is not None else getattr(self, "caller_ring", None)
        eff_allowed_raw = allowed_tools if allowed_tools is not None else getattr(self, "allowed_tools", None)
        eff_allowed_tools = list(eff_allowed_raw) if eff_allowed_raw is not None else None
        eff_tool_executor = tool_executor or getattr(self, "tool_executor", None)
        eff_task_local_scope = task_local_scope if task_local_scope is not None else getattr(self, "task_local_scope", None)
        if eff_task_local_scope is None and eff_tool_executor is not None:
            eff_task_local_scope = getattr(eff_tool_executor, "_task_local_tool_scope", None)
        eff_tool_schemas = tool_schemas if tool_schemas is not None else getattr(self, "tool_schemas", None)

        eff_central = (
            central_ledger
            or getattr(self, "central_ledger", None)
            or (getattr(turn_ledger, "_central_ledger", None) if turn_ledger is not None else None)
            or get_default_central_ledger()
        )
        eff_central.create_execution(
            task_prompt=task_prompt or graph.objective or "",
            execution_id=eff_exec_id,
            source_component="graph_parallel",
            caller_ring=eff_caller_ring,
            task_local_scoped=bool(eff_task_local_scope is not None),
            status=ExecutionStatus.RUNNING,
        )
        eff_ledger = turn_ledger if turn_ledger is not None else TurnExecutionLedger(
            execution_id=eff_exec_id,
            central_ledger=eff_central,
            caller_ring=eff_caller_ring,
            task_local_scoped=bool(eff_task_local_scope is not None),
            source_component="graph_parallel",
        )
        if getattr(eff_ledger, "_central_ledger", None) is None or getattr(eff_ledger, "_execution_id", None) != eff_exec_id:
            eff_ledger.attach_central_ledger(
                eff_central,
                eff_exec_id,
                caller_ring=eff_caller_ring,
                task_local_scoped=bool(eff_task_local_scope is not None),
                source_component="graph_parallel",
            )
        eff_ledger.attach_task_graph(graph)
        self.current_ledger = eff_ledger

        # Build normalized proposal lookup preserving execution isolation
        prop_by_ref: Dict[str, Dict[str, Any]] = {}
        raw_prop_list: List[Any] = []
        if isinstance(proposals, dict):
            raw_prop_list = list(proposals.values())
        elif proposals is not None:
            raw_prop_list = list(proposals)
        elif hasattr(eff_ledger, "get_tracked_proposals"):
            raw_prop_list = eff_ledger.get_tracked_proposals()

        for idx, p_item in enumerate(raw_prop_list):
            p_norm = self._normalize_graph_proposal(p_item, eff_exec_id, fallback_order=idx)
            prop_by_ref[p_norm["proposal_id"]] = p_norm
            prop_by_ref[p_norm["original_call_id"]] = p_norm
            if hasattr(eff_ledger, "_proposal_meta"):
                eff_ledger._proposal_meta[p_norm["original_call_id"]] = dict(p_norm)

        graph.validate(known_proposals=prop_by_ref, allow_blocked=True)
        graph.recompute_schedule(known_proposals=prop_by_ref)

        parsed_criteria = parse_acceptance_criteria(acceptance_criteria) if acceptance_criteria else []
        eff_default_timeout = float(default_node_timeout if default_node_timeout is not None else 60.0)
        eff_node_timeouts = dict(node_timeouts or {})
        eff_policy = (
            recovery_policy
            if isinstance(recovery_policy, RecoveryPolicy)
            else RecoveryPolicy.from_config(self._active_config())
        )
        eff_max_steps = (
            int(max_steps)
            if max_steps is not None
            else max(int(getattr(self, "max_steps", 5)), (len(graph.nodes) * 2) + (eff_policy.max_attempts * len(graph.nodes)))
        )

        auth_schemas = eff_tool_schemas
        known_for_auth = (
            self.get_known_tool_names(list(eff_tool_schemas))
            if (eff_tool_schemas is not None and eff_task_local_scope is None)
            else None
        )
        schema_by_name: Dict[str, Dict[str, Any]] = {}
        if isinstance(eff_tool_schemas, list):
            for s in eff_tool_schemas:
                if not isinstance(s, dict):
                    continue
                fn = s.get("function") if isinstance(s.get("function"), dict) else {}
                sname = str(fn.get("name") or s.get("name") or "").strip()
                if sname:
                    schema_by_name[sname] = s

        def _collect_checkpoint_proposals() -> List[Dict[str, Any]]:
            uniq: List[Dict[str, Any]] = []
            seen_ids: set[str] = set()
            for p_v in prop_by_ref.values():
                if isinstance(p_v, dict):
                    pid_s = str(p_v.get("proposal_id") or "")
                    if pid_s and pid_s not in seen_ids:
                        seen_ids.add(pid_s)
                        uniq.append(dict(p_v))
            return uniq

        def _capture_cp(status_val: Any) -> None:
            ver_st = (
                VerificationStatus.VERIFIED
                if graph.status == TaskGraphStatus.COMPLETED_VERIFIED.value
                else VerificationStatus.NOT_VERIFIED
            )
            CheckpointManager.capture_checkpoint(
                eff_central,
                eff_exec_id,
                turn_ledger=eff_ledger,
                status=status_val,
                verification_status=ver_st,
                task_prompt=task_prompt or graph.objective or "",
                planning_state={
                    "max_concurrency": eff_max_conc,
                    "acceptance_criteria": acceptance_criteria,
                    "executed_steps": executed_steps,
                    "graph_status": graph.status,
                },
                pending_actions=_collect_checkpoint_proposals() or None,
                task_graph=graph,
                emit_event=False,
                strict=None,
            )

        active_tasks: Dict[asyncio.Task[Dict[str, Any]], Tuple[str, str, str, str]] = {}
        node_results: List[Dict[str, Any]] = []
        recovery_events_out: List[Dict[str, Any]] = []
        dispatched_node_ids: set[str] = set()
        backoff_until_by_node: Dict[str, float] = {}
        for gn_init in graph.nodes:
            if isinstance(getattr(gn_init, "recovery_state", None), dict):
                nra = gn_init.recovery_state.get("next_retry_after")
                if nra is not None:
                    backoff_until_by_node[gn_init.node_id] = float(nra)
        executed_steps = 0
        was_cancelled = False

        def _run_recovery_for_nodes(target_node_ids: Iterable[str]) -> bool:
            if not enable_recovery:
                return False
            changed_any = False
            for r_nid in target_node_ids:
                r_node = graph.get_node(r_nid)
                if r_node is None:
                    continue
                if r_node.reserved_by is not None:
                    continue
                if r_node.execution_status not in (
                    TaskNodeStatus.FAILED.value,
                    TaskNodeStatus.BLOCKED.value,
                    TaskNodeStatus.COMPLETED_UNVERIFIED.value,
                ):
                    continue
                # Do not run node recovery on a downstream node blocked solely by an upstream prerequisite
                if r_node.execution_status == TaskNodeStatus.BLOCKED.value and not r_node.contradictory:
                    has_unsat_parent = any(
                        (graph.get_node(dep_id) is None)
                        or not graph._is_prerequisite_satisfied(
                            graph.get_node(dep_id), r_node.requires_verified_dependencies
                        )
                        for dep_id in r_node.dependency_ids
                    )
                    if has_unsat_parent:
                        continue
                assess = RecoveryEngine.assess_node_recovery(
                    graph,
                    r_nid,
                    central_ledger=eff_central,
                    turn_ledger=eff_ledger,
                    proposals=prop_by_ref,
                    policy=eff_policy,
                    tool_registry=self.tool_registry,
                    active_schemas=auth_schemas,
                    known_tools=known_for_auth,
                    caller_ring=eff_caller_ring,
                    allowed_tools=eff_allowed_tools,
                    task_local_scope=eff_task_local_scope,
                    zelyr_config=self._active_config(),
                    policy_engine=policy_engine or getattr(self, "execution_policy_engine", None),
                    policy_config=policy_config or getattr(self, "execution_policy_config", None),
                    workspace_root=workspace_root or getattr(self, "workspace_root", None),
                    acceptance_criteria=acceptance_criteria,
                    verifier_fn=verifier_fn,
                    reconcile_state=reconcile_on_recovery,
                    confirm_recovery=confirm_recovery,
                )
                try:
                    apply_res = RecoveryEngine.apply_recovery_decision(
                        graph,
                        assess,
                        prop_by_ref=prop_by_ref,
                        central_ledger=eff_central,
                        turn_ledger=eff_ledger,
                        execute_retry=eff_policy.allow_auto_retry,
                    )
                except TaskGraphConflictError:
                    continue
                recovery_events_out.append(apply_res)
                changed_any = True
                if apply_res.get("retried"):
                    dispatched_node_ids.discard(r_nid)
                    backoff_s = float(apply_res.get("backoff_seconds") or 0.0)
                    if backoff_s > 0.0:
                        backoff_until_by_node[r_nid] = time.time() + backoff_s
                    else:
                        backoff_until_by_node.pop(r_nid, None)
            return changed_any

        def _finalize_graph_state() -> None:
            graph.recompute_schedule(known_proposals=prop_by_ref)
            if was_cancelled or getattr(self, "_cancelled", False):
                fin_exec_status = ExecutionStatus.CANCELLED
                fin_ver_status = VerificationStatus.INTERRUPTED
            elif graph.status == TaskGraphStatus.COMPLETED_VERIFIED.value:
                fin_exec_status = ExecutionStatus.COMPLETED
                fin_ver_status = VerificationStatus.VERIFIED
            elif graph.status == TaskGraphStatus.COMPLETED_UNVERIFIED.value:
                fin_exec_status = ExecutionStatus.COMPLETED_UNVERIFIED
                fin_ver_status = VerificationStatus.NOT_VERIFIED
            elif graph.status in (TaskGraphStatus.FAILED.value, TaskGraphStatus.BLOCKED.value):
                fin_exec_status = ExecutionStatus.FAILED
                fin_ver_status = VerificationStatus.NOT_VERIFIED
            elif graph.status == TaskGraphStatus.CANCELLED.value:
                fin_exec_status = ExecutionStatus.CANCELLED
                fin_ver_status = VerificationStatus.INTERRUPTED
            else:
                fin_exec_status = ExecutionStatus.INTERRUPTED if was_cancelled else ExecutionStatus.FAILED
                fin_ver_status = VerificationStatus.NOT_VERIFIED

            try:
                if hasattr(eff_ledger, "flush_observability_metrics"):
                    eff_ledger.flush_observability_metrics()
            except Exception:
                pass
            cp_exc_to_raise: Optional[BaseException] = None
            try:
                _capture_cp(fin_exec_status)
            except Exception as cp_exc:
                if getattr(eff_central, "raise_on_persistence_error", False):
                    cp_exc_to_raise = cp_exc
            if finalize_execution:
                try:
                    eff_central.finalize_execution(
                        eff_exec_id,
                        status=fin_exec_status,
                        verification_status=fin_ver_status,
                        final_summary=f"TaskGraph '{graph.graph_id}' finished with status={graph.status}",
                        error=(
                            "TaskGraph cancelled"
                            if fin_exec_status == ExecutionStatus.CANCELLED
                            else (f"TaskGraph ended in status '{graph.status}'" if fin_exec_status == ExecutionStatus.FAILED else None)
                        ),
                        metadata=(
                            {"checkpoint_saved": False, "checkpoint_error": str(cp_exc_to_raise)}
                            if cp_exc_to_raise is not None
                            else None
                        ),
                    )
                except Exception:
                    if cp_exc_to_raise is None and getattr(eff_central, "raise_on_persistence_error", False):
                        raise
            if cp_exc_to_raise is not None:
                raise cp_exc_to_raise

        # Assess pre-existing failed / unverified nodes on entry (e.g., resumed graphs)
        try:
            initial_candidates = [
                n.node_id
                for n in graph.nodes
                if n.execution_status in (TaskNodeStatus.FAILED.value, TaskNodeStatus.COMPLETED_UNVERIFIED.value)
                or (n.execution_status == TaskNodeStatus.BLOCKED.value and n.contradictory)
            ]
            if initial_candidates:
                _run_recovery_for_nodes(initial_candidates)

            _capture_cp(ExecutionStatus.RUNNING)
        except BaseException:
            _finalize_graph_state()
            raise

        try:
            while executed_steps < eff_max_steps or active_tasks:
                if getattr(self, "_cancelled", False):
                    raise asyncio.CancelledError("Orchestrator execution cancelled.")

                graph.recompute_schedule(known_proposals=prop_by_ref)
                ready_nodes = graph.get_ready_nodes()
                active_profiles = [
                    ap for _, ap in graph._active_profiles_excluding(None)
                ]

                if executed_steps < eff_max_steps and len(active_tasks) < eff_max_conc and ready_nodes:
                    if hasattr(eff_ledger, "record_orchestration_step"):
                        eff_ledger.record_orchestration_step()
                    dispatch_failed_ids: List[str] = []
                    now_loop = time.time()
                    for node in ready_nodes:
                        if len(active_tasks) >= eff_max_conc or executed_steps + len(active_tasks) >= eff_max_steps:
                            break
                        if node.node_id in dispatched_node_ids or node.reserved_by is not None:
                            continue
                        b_until = backoff_until_by_node.get(node.node_id)
                        if b_until is not None and now_loop < b_until:
                            continue
                        backoff_until_by_node.pop(node.node_id, None)

                        # Resolve runnable proposal for this node
                        prop_info: Optional[Dict[str, Any]] = None
                        for pid in node.proposal_ids:
                            if pid in prop_by_ref:
                                prop_info = prop_by_ref[pid]
                                break
                        if prop_info is None and node.node_id in prop_by_ref:
                            prop_info = prop_by_ref[node.node_id]

                        if prop_info is None:
                            fail_obs = ObservationFramingEngine.frame_structured(
                                "task_graph",
                                {"status": "error", "error": f"No proposal bound for node '{node.node_id}'."},
                                dispatched=False,
                                execution_id=eff_exec_id,
                                correlation_id=f"missing_{node.node_id}",
                            )
                            graph.apply_observation(node.node_id, fail_obs, outcome_verifier_passed=False, known_proposals=prop_by_ref)
                            node_results.append({"node_id": node.node_id, "status": "failed", "error": "missing_proposal"})
                            dispatch_failed_ids.append(node.node_id)
                            continue

                        if prop_info.get("expires_at") is not None and float(prop_info["expires_at"]) <= time.time():
                            exp_call_id = prop_info["original_call_id"]
                            exp_err = f"Proposal '{prop_info['proposal_id']}' for node '{node.node_id}' has expired."
                            prop_info["status"] = "expired"
                            if eff_ledger.get_record(exp_call_id) is None:
                                eff_ledger.record_proposed(
                                    exp_call_id,
                                    prop_info["tool_id"],
                                    dict(prop_info.get("normalized_arguments") or {}),
                                    proposal_id=prop_info["proposal_id"],
                                    external_source=prop_info.get("external_source") or node.external_source,
                                )
                            eff_ledger.record_rejected(exp_call_id, reason=exp_err)
                            exp_obs = ObservationFramingEngine.frame_structured(
                                prop_info["tool_id"] or "task_graph",
                                {"status": "rejected", "error": exp_err, "error_type": "PROPOSAL_EXPIRED"},
                                dispatched=False,
                                execution_id=eff_exec_id,
                                correlation_id=exp_call_id,
                            )
                            eff_ledger.record_observation(
                                exp_call_id,
                                status="rejected",
                                output="",
                                error=exp_err,
                                rejection_reason="proposal_expired",
                                structured_observation=exp_obs,
                            )
                            graph.apply_observation(node.node_id, exp_obs, action_record=eff_ledger.get_record(exp_call_id), known_proposals=prop_by_ref)
                            node_results.append({"node_id": node.node_id, "call_id": exp_call_id, "status": "expired", "error": exp_err})
                            dispatch_failed_ids.append(node.node_id)
                            continue

                        call_id = prop_info["original_call_id"]
                        func_name = prop_info["tool_id"]
                        ext_src = prop_info.get("external_source") or node.external_source
                        existing_rec = eff_ledger.get_record(call_id)

                        # Guard against re-executing already completed or ambiguous in-flight operations
                        if existing_rec is not None:
                            if existing_rec.state == ActionState.VERIFIED:
                                node.execution_status = TaskNodeStatus.COMPLETED_VERIFIED.value
                                node.verification_status = VerificationStatus.VERIFIED.value
                                graph.recompute_schedule(known_proposals=prop_by_ref)
                                continue
                            if existing_rec.state in (ActionState.DISPATCHED, ActionState.EXECUTED) and not allow_unverified_retry:
                                node.execution_status = TaskNodeStatus.RUNNING.value
                                node.verification_status = VerificationStatus.INTERRUPTED.value
                                node.blocked_reason = "Ambiguous in-flight operation requires verification before replay."
                                graph.recompute_schedule(known_proposals=prop_by_ref)
                                continue

                        # Build conservative conflict profile and check against running reservations
                        reg_tool_obj = (
                            self.tool_registry.get_tool(func_name)
                            if (self.tool_registry is not None and callable(getattr(self.tool_registry, "get_tool", None)))
                            else None
                        )
                        cand_args_dict = dict(prop_info.get("normalized_arguments") or {})
                        cand_profile = extract_node_conflict_profile(
                            func_name,
                            cand_args_dict,
                            node=node,
                            external_source=ext_src,
                            tool_schema=schema_by_name.get(func_name),
                            tool_obj=reg_tool_obj,
                        )
                        if any(profiles_conflict(cand_profile, ap) is not None for ap in active_profiles):
                            continue

                        # Pre-dispatch validation, external provenance & live authorization check
                        ext_err = self._check_live_external_provenance(func_name, ext_src)
                        if ext_err is not None:
                            if eff_ledger.get_record(call_id) is None:
                                eff_ledger.record_proposed(
                                    call_id, func_name, cand_args_dict,
                                    proposal_id=prop_info["proposal_id"],
                                    depends_on=prop_info.get("depends_on"),
                                    external_source=ext_src,
                                )
                            eff_ledger.record_rejected(call_id, reason=ext_err)
                            eff_central.record_event(
                                eff_exec_id,
                                LedgerEventType.AUTHORIZATION_DECISION,
                                tool_id=func_name,
                                status="denied",
                                correlation_id=call_id,
                                caller_ring=eff_caller_ring,
                                task_local_scoped=bool(eff_task_local_scope is not None),
                                metadata={"reason": "authorization_denied", "detail": ext_err, "node_id": node.node_id},
                                idempotency_key=f"{eff_exec_id}:{call_id}:auth",
                            )
                            rej_obs = ObservationFramingEngine.frame_structured(
                                func_name,
                                {"status": "rejected", "error": ext_err, "error_type": "AUTHORIZATION_DENIED"},
                                requested_tool=func_name,
                                requested_args=cand_args_dict,
                                executed_args=cand_args_dict,
                                dispatched=False,
                                execution_id=eff_exec_id,
                                correlation_id=call_id,
                                external_source=ext_src,
                            )
                            eff_ledger.record_observation(
                                call_id, status="rejected", output="", error=ext_err,
                                rejection_reason="authorization_denied", structured_observation=rej_obs,
                            )
                            graph.apply_observation(node.node_id, rej_obs, action_record=eff_ledger.get_record(call_id), known_proposals=prop_by_ref)
                            node_results.append({"node_id": node.node_id, "call_id": call_id, "status": "rejected", "error": ext_err})
                            dispatch_failed_ids.append(node.node_id)
                            continue

                        pre_dec = authorize_and_validate_tool_call(
                            func_name,
                            prop_info.get("arguments", cand_args_dict),
                            caller_ring=eff_caller_ring,
                            allowed_tools=eff_allowed_tools,
                            active_schemas=auth_schemas,
                            tool_registry=self.tool_registry,
                            known_tools=known_for_auth,
                            task_local_scope=eff_task_local_scope,
                            require_registered=True,
                            normalize_args=True,
                            validate_schema=True,
                            argument_source="internal",
                        )
                        if not pre_dec.allowed:
                            rej_reason = pre_dec.reason or "authorization_denied"
                            rej_detail = pre_dec.detail or "Authorization denied."
                            err_t = pre_dec.error_type or "AUTHORIZATION_DENIED"
                            if eff_ledger.get_record(call_id) is None:
                                eff_ledger.record_proposed(
                                    call_id, func_name, cand_args_dict,
                                    proposal_id=prop_info["proposal_id"],
                                    depends_on=prop_info.get("depends_on"),
                                    external_source=ext_src,
                                )
                            eff_ledger.record_rejected(call_id, reason=rej_detail)
                            is_schema_err = err_t == "SCHEMA_VALIDATION_ERROR"
                            eff_central.record_event(
                                eff_exec_id,
                                LedgerEventType.ARGUMENT_VALIDATED if is_schema_err else LedgerEventType.AUTHORIZATION_DECISION,
                                tool_id=func_name,
                                status="invalid" if is_schema_err else "denied",
                                correlation_id=call_id,
                                caller_ring=eff_caller_ring,
                                task_local_scoped=bool(eff_task_local_scope is not None),
                                metadata={"reason": rej_reason, "detail": rej_detail, "node_id": node.node_id},
                                idempotency_key=f"{eff_exec_id}:{call_id}:{'validated' if is_schema_err else 'auth'}",
                            )
                            rej_obs = ObservationFramingEngine.frame_structured(
                                func_name,
                                {"status": "rejected", "reason": rej_reason, "detail": rej_detail, "error_type": err_t},
                                requested_tool=func_name,
                                requested_args=cand_args_dict,
                                executed_args=cand_args_dict,
                                dispatched=False,
                                execution_id=eff_exec_id,
                                correlation_id=call_id,
                                external_source=ext_src,
                            )
                            eff_ledger.record_observation(
                                call_id, status="rejected", output="", error=rej_detail,
                                rejection_reason=rej_reason, structured_observation=rej_obs,
                            )
                            graph.apply_observation(node.node_id, rej_obs, action_record=eff_ledger.get_record(call_id), known_proposals=prop_by_ref)
                            node_results.append({"node_id": node.node_id, "call_id": call_id, "status": "rejected", "error": rej_detail})
                            dispatch_failed_ids.append(node.node_id)
                            continue

                        # Update normalized tool & args cleanly before reservation
                        func_name = pre_dec.tool_name
                        norm_args = dict(pre_dec.normalized_arguments)
                        prop_info["tool_id"] = func_name
                        prop_info["normalized_arguments"] = norm_args
                        norm_profile = extract_node_conflict_profile(
                            func_name,
                            norm_args,
                            node=node,
                            external_source=ext_src,
                            tool_schema=schema_by_name.get(func_name),
                            tool_obj=reg_tool_obj,
                        )
                        if any(profiles_conflict(norm_profile, ap) is not None for ap in active_profiles):
                            continue

                        # Atomically reserve node & sorted resource locks
                        owner_id = f"worker:{node.node_id}:{uuid.uuid4().hex[:8]}"
                        try:
                            rid = graph.reserve_node(
                                node.node_id,
                                owner_id=owner_id,
                                conflict_profile=norm_profile,
                                max_concurrency=eff_max_conc,
                            )
                        except TaskGraphConflictError:
                            continue

                        dispatched_node_ids.add(node.node_id)
                        active_profiles.append(norm_profile)

                        if isinstance(ext_src, dict) and ext_src:
                            eff_ledger.set_call_source_metadata(
                                call_id,
                                is_external=bool(ext_src.get("is_external", True)),
                                source_id=ext_src.get("source_id"),
                                source_type=ext_src.get("source_type"),
                                extra={"external_tool_name": ext_src.get("external_tool_name")} if ext_src.get("external_tool_name") else None,
                            )
                        if eff_ledger.get_record(call_id) is None:
                            eff_ledger.record_proposed(
                                call_id,
                                func_name,
                                norm_args,
                                proposal_id=prop_info["proposal_id"],
                                depends_on=prop_info.get("depends_on"),
                                external_source=ext_src,
                            )
                        eff_ledger.record_node_reservation_event(
                            "node_reserved",
                            node_id=node.node_id,
                            owner_id=owner_id,
                            reservation_id=rid,
                            call_id=call_id,
                            tool_id=func_name,
                            resources=norm_profile.lock_keys,
                            status="reserved",
                            idempotency_suffix=f"reserve:{rid}",
                        )

                        max_allowed_t = float(default_node_timeout) if default_node_timeout is not None else 600.0
                        node_hint_t = getattr(node, "timeout_seconds", None)
                        if node.node_id in eff_node_timeouts:
                            eff_timeout = float(eff_node_timeouts[node.node_id])
                        elif node_hint_t is not None:
                            eff_timeout = max(0.05, min(float(node_hint_t), max_allowed_t))
                        else:
                            eff_timeout = eff_default_timeout
                        worker_coro = self._run_reserved_graph_node(
                            graph=graph,
                            node_id=node.node_id,
                            call_id=call_id,
                            proposal=prop_info,
                            owner_id=owner_id,
                            reservation_id=rid,
                            central_ledger=eff_central,
                            turn_ledger=eff_ledger,
                            prop_by_ref=prop_by_ref,
                            eff_exec_id=eff_exec_id,
                            eff_caller_ring=eff_caller_ring,
                            eff_allowed_tools=eff_allowed_tools,
                            eff_task_local_scope=eff_task_local_scope,
                            auth_schemas=auth_schemas,
                            known_for_auth=known_for_auth,
                            eff_tool_executor=eff_tool_executor,
                            parsed_criteria=parsed_criteria,
                            node_timeout=eff_timeout,
                            verifier_fn=verifier_fn,
                            policy_engine=policy_engine,
                            policy_config=policy_config,
                            confirm_callback=confirm_callback,
                            node_confirmation=(node_confirmations.get(node.node_id) if isinstance(node_confirmations, dict) else None),
                            workspace_root=workspace_root,
                        )
                        task = asyncio.create_task(worker_coro, name=f"graph_node_{node.node_id}_{owner_id}")
                        active_tasks[task] = (node.node_id, call_id, owner_id, rid)

                    if dispatch_failed_ids and _run_recovery_for_nodes(dispatch_failed_ids):
                        graph.recompute_schedule(known_proposals=prop_by_ref)

                if not active_tasks:
                    # Wait only bounded backoff for any READY node awaiting retry backoff timer
                    pending_wait_nodes = [
                        n.node_id
                        for n in graph.get_ready_nodes()
                        if n.node_id not in dispatched_node_ids
                        and n.node_id in backoff_until_by_node
                        and backoff_until_by_node[n.node_id] > time.time()
                    ]
                    if pending_wait_nodes and executed_steps < eff_max_steps:
                        soonest = min(backoff_until_by_node[nid] for nid in pending_wait_nodes)
                        sleep_for = max(0.0, soonest - time.time())
                        if sleep_for > 0.0:
                            await asyncio.sleep(sleep_for)
                        _capture_cp(ExecutionStatus.RUNNING)
                        continue
                    break

                # Persist active concurrent reservations + RUNNING state before awaiting wave
                _capture_cp(ExecutionStatus.RUNNING)

                done, _ = await asyncio.wait(active_tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
                completed_in_wave: List[str] = []
                for d_task in done:
                    nid_d, cid_d, owner_d, rid_d = active_tasks.pop(d_task)
                    completed_in_wave.append(nid_d)
                    executed_steps += 1
                    try:
                        res_item = d_task.result()
                        if isinstance(res_item, dict):
                            node_results.append(res_item)
                    except asyncio.CancelledError:
                        node_results.append({"node_id": nid_d, "call_id": cid_d, "status": "cancelled"})
                    except Exception as task_exc:
                        node_results.append({"node_id": nid_d, "call_id": cid_d, "status": "error", "error": str(task_exc)})

                if completed_in_wave:
                    _run_recovery_for_nodes(completed_in_wave)

                _capture_cp(ExecutionStatus.RUNNING)
        except asyncio.CancelledError:
            was_cancelled = True
            for pending_t in list(active_tasks.keys()):
                if not pending_t.done():
                    pending_t.cancel()
            if active_tasks:
                await asyncio.gather(*list(active_tasks.keys()), return_exceptions=True)
                active_tasks.clear()
            raise
        finally:
            if active_tasks:
                for leftover_t in list(active_tasks.keys()):
                    if not leftover_t.done():
                        leftover_t.cancel()
                await asyncio.gather(*list(active_tasks.keys()), return_exceptions=True)
                active_tasks.clear()

            _finalize_graph_state()

        return {
            "execution_id": eff_exec_id,
            "graph_id": graph.graph_id,
            "graph_status": graph.status,
            "max_concurrency": graph.max_concurrency,
            "summary": dict(graph.summary),
            "executed_steps": executed_steps,
            "results": node_results,
            "recovery_events": recovery_events_out,
            "recovery_overview": build_graph_recovery_overview(graph),
            "task_graph": graph,
            "turn_ledger": eff_ledger,
        }

    async def plan_to_graph(
        self,
        objective: str,
        *,
        execution_id: Optional[str] = None,
        central_ledger: Optional[Any] = None,
        turn_ledger: Optional[Any] = None,
        stream_provider: Optional[Any] = None,
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
        tool_registry: Optional[Any] = None,
        max_concurrency: Optional[int] = None,
        max_plan_retries: int = 2,
        workspace_root: Optional[Union[str, Path]] = None,
        caller_ring: Optional[int] = None,
        allowed_tools: Optional[Iterable[str]] = None,
        task_local_scope: Optional[Any] = None,
        require_confirmation: Optional[bool] = None,
        confirmed: bool = False,
        model: Optional[str] = None,
        persist_checkpoint: bool = True,
    ) -> Any:
        """Compile a natural-language objective into a validated TaskGraph without executing it."""
        from zelyr.engine.execution_ledger import TurnExecutionLedger, get_default_central_ledger
        from zelyr.planning.compiler import PlanCompiler

        eff_max_conc = (
            max_concurrency
            if max_concurrency is not None
            else getattr(self, "max_concurrency", 1)
        )
        eff_central = (
            central_ledger
            or getattr(self, "central_ledger", None)
            or (getattr(turn_ledger, "_central_ledger", None) if turn_ledger is not None else None)
            or get_default_central_ledger()
        )
        eff_caller_ring = caller_ring if caller_ring is not None else getattr(self, "caller_ring", None)
        eff_allowed_raw = allowed_tools if allowed_tools is not None else getattr(self, "allowed_tools", None)
        eff_allowed = list(eff_allowed_raw) if eff_allowed_raw is not None else None
        eff_schemas = tool_schemas if tool_schemas is not None else getattr(self, "tool_schemas", None)
        eff_registry = tool_registry if tool_registry is not None else getattr(self, "tool_registry", None)
        eff_scope = task_local_scope if task_local_scope is not None else getattr(self, "task_local_scope", None)
        eff_sp = stream_provider or getattr(self, "stream_provider", None) or getattr(self, "model_provider", None)

        cfg = self._active_config()
        eff_model = model or getattr(cfg, "ollama_model", None)
        eff_base_url = getattr(cfg, "ollama_base_url", "http://127.0.0.1:11434")
        env_ctx = ""
        try:
            env_ctx = self.get_planning_environment_context(
                caller_ring=eff_caller_ring,
                allowed_tools=eff_allowed,
            )
        except Exception:
            env_ctx = ""

        async def _planner_stream(req_payload: dict):
            async for ev in self._stream_model_events(req_payload, base_url=eff_base_url, stream_provider=eff_sp):
                yield ev

        compiler = PlanCompiler(max_plan_retries=max_plan_retries)
        eff_turn = turn_ledger or TurnExecutionLedger(
            execution_id=execution_id,
            central_ledger=eff_central,
            caller_ring=eff_caller_ring,
            task_local_scoped=bool(eff_scope is not None),
            source_component="plan_compiler",
        )
        bundle = await compiler.compile_from_objective(
            objective,
            stream_provider=_planner_stream,
            execution_id=execution_id,
            central_ledger=eff_central,
            turn_ledger=eff_turn,
            max_concurrency=eff_max_conc,
            max_plan_retries=max_plan_retries,
            model=eff_model,
            tool_schemas=eff_schemas,
            tool_registry=eff_registry,
            workspace_root=workspace_root,
            environment_context=env_ctx,
            require_confirmation=require_confirmation,
            confirmed=confirmed,
            caller_ring=eff_caller_ring,
            task_local_scoped=bool(eff_scope is not None),
            persist_checkpoint=persist_checkpoint,
        )
        self.current_ledger = eff_turn
        return bundle

    async def run_planned_task(
        self,
        objective: str,
        *,
        execution_id: Optional[str] = None,
        central_ledger: Optional[Any] = None,
        turn_ledger: Optional[Any] = None,
        stream_provider: Optional[Any] = None,
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
        tool_executor: Optional[Callable[[str, dict], Awaitable[Any]]] = None,
        tool_registry: Optional[Any] = None,
        max_concurrency: Optional[int] = None,
        max_plan_retries: int = 2,
        workspace_root: Optional[Union[str, Path]] = None,
        caller_ring: Optional[int] = None,
        allowed_tools: Optional[Iterable[str]] = None,
        task_local_scope: Optional[Any] = None,
        acceptance_criteria: Optional[str] = None,
        require_confirmation: Optional[bool] = None,
        confirmed: Optional[bool] = None,
        confirm_callback: Optional[Callable[[Any], Any]] = None,
        plan_only: bool = False,
        node_timeouts: Optional[Dict[str, float]] = None,
        default_node_timeout: Optional[float] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compile objective into a validated TaskGraph, gate confirmation/policy, and execute."""
        from zelyr.engine.checkpoint import CheckpointManager
        from zelyr.engine.execution_ledger import (
            ExecutionStatus,
            LedgerEventType,
            TurnExecutionLedger,
            VerificationStatus,
            get_default_central_ledger,
        )
        from zelyr.planning.compiler import PlanConfirmationRequiredError

        eff_central = central_ledger or getattr(self, "central_ledger", None) or get_default_central_ledger()
        eff_caller_ring = caller_ring if caller_ring is not None else getattr(self, "caller_ring", None)
        eff_allowed_raw = allowed_tools if allowed_tools is not None else getattr(self, "allowed_tools", None)
        eff_allowed = list(eff_allowed_raw) if eff_allowed_raw is not None else None
        eff_scope = task_local_scope if task_local_scope is not None else getattr(self, "task_local_scope", None)
        eff_turn = turn_ledger or TurnExecutionLedger(
            execution_id=execution_id,
            central_ledger=eff_central,
            caller_ring=eff_caller_ring,
            task_local_scoped=bool(eff_scope is not None),
            source_component="plan_compiler",
        )

        bundle = await self.plan_to_graph(
            objective,
            execution_id=execution_id,
            central_ledger=eff_central,
            turn_ledger=eff_turn,
            stream_provider=stream_provider,
            tool_schemas=tool_schemas,
            tool_registry=tool_registry,
            max_concurrency=max_concurrency,
            max_plan_retries=max_plan_retries,
            workspace_root=workspace_root,
            caller_ring=eff_caller_ring,
            allowed_tools=eff_allowed,
            task_local_scope=eff_scope,
            require_confirmation=require_confirmation,
            confirmed=bool(confirmed is True),
            model=model,
            persist_checkpoint=True,
        )
        eff_exec_id = bundle.task_graph.execution_id

        if plan_only:
            return {
                "status": "planned",
                "executed": False,
                "execution_id": eff_exec_id,
                "graph_id": bundle.task_graph.graph_id,
                "bundle": bundle,
                "task_graph": bundle.task_graph,
                "summary_text": bundle.summary_text,
                "results": [],
            }

        eff_require_conf = require_confirmation if require_confirmation is not None else bundle.requires_confirmation
        is_permitted = True
        if eff_require_conf:
            if confirmed is True:
                is_permitted = True
            elif confirmed is False:
                is_permitted = False
            elif confirm_callback is not None:
                cb_res = confirm_callback(bundle)
                is_permitted = bool(await cb_res if inspect.isawaitable(cb_res) else cb_res)
            elif require_confirmation is True:
                # Explicit require_confirmation=True always fails closed unless confirmed=True or callback approves
                is_permitted = False
            else:
                from zelyr.compat import get_env

                if get_env("AUTOPILOT") == "1" or get_env("TESTING") == "1":
                    is_permitted = True
                else:
                    is_permitted = False

        if not is_permitted:
            eff_central.record_event(
                eff_exec_id,
                LedgerEventType.AUTHORIZATION_DECISION,
                status="denied",
                caller_ring=eff_caller_ring,
                task_local_scoped=bool(eff_scope is not None),
                metadata={
                    "reason": "plan_confirmation_required",
                    "graph_id": bundle.task_graph.graph_id,
                    "plan_id": bundle.structured_plan.plan_id,
                },
                idempotency_key=f"{eff_exec_id}:plan_confirm:denied",
            )
            CheckpointManager.capture_checkpoint(
                eff_central,
                eff_exec_id,
                turn_ledger=eff_turn,
                status=ExecutionStatus.INITIALIZED,
                verification_status=VerificationStatus.NOT_VERIFIED,
                task_prompt=bundle.structured_plan.objective,
                planning_state={
                    "plan_id": bundle.structured_plan.plan_id,
                    "plan_version": bundle.structured_plan.plan_version,
                    "plan_digest": bundle.structured_plan.plan_digest,
                    "plan_confirmed": False,
                    "requires_confirmation": True,
                    "max_concurrency": bundle.task_graph.max_concurrency,
                    "acceptance_criteria": acceptance_criteria,
                },
                pending_actions=bundle.proposals,
                task_graph=bundle.task_graph,
                emit_event=False,
            )
            raise PlanConfirmationRequiredError(
                f"Execution of plan '{bundle.structured_plan.plan_id}' ({eff_exec_id}) requires explicit confirmation.",
                bundle=bundle,
            )

        bundle.confirmed = True
        eff_central.record_event(
            eff_exec_id,
            LedgerEventType.AUTHORIZATION_DECISION,
            status="allowed",
            caller_ring=eff_caller_ring,
            task_local_scoped=bool(eff_scope is not None),
            metadata={
                "reason": "plan_confirmed",
                "graph_id": bundle.task_graph.graph_id,
                "plan_id": bundle.structured_plan.plan_id,
            },
            idempotency_key=f"{eff_exec_id}:plan_confirm:allowed",
        )

        # When plan execution was explicitly confirmed, issue per-node single-use bound
        # ConfirmationRecords strictly in memory (never persisted on TaskGraph/checkpoints).
        plan_node_confirmations: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        if is_permitted and (eff_require_conf or confirmed is True or confirm_callback is not None):
            from zelyr.engine.execution_policy import (
                PolicyDecisionOutcome,
                build_execution_policy_engine,
            )

            plan_pol_engine = build_execution_policy_engine(
                policy_engine=getattr(self, "execution_policy_engine", None),
                policy_config=getattr(self, "execution_policy_config", None),
                zelyr_config=self._active_config(),
                confirmation_store=getattr(self, "confirmation_store", None),
            )
            prop_by_id_plan = {p.proposal_id: p for p in (bundle.proposals or [])}
            eff_plan_schemas = tool_schemas if tool_schemas is not None else getattr(self, "tool_schemas", None)
            for gn in bundle.task_graph.nodes:
                if not gn.proposal_ids:
                    continue
                gp = prop_by_id_plan.get(gn.proposal_ids[0])
                if gp is None:
                    continue
                gp_schema = next(
                    (
                        s
                        for s in (eff_plan_schemas or [])
                        if isinstance(s, dict) and (s.get("function") or {}).get("name") == gp.tool_id
                    ),
                    None,
                )
                gp_tool_obj = (
                    self.tool_registry.get_tool(gp.tool_id)
                    if (self.tool_registry is not None and callable(getattr(self.tool_registry, "get_tool", None)))
                    else None
                )
                pre_pol = plan_pol_engine.evaluate(
                    gp.tool_id,
                    gp.normalized_arguments,
                    normalized_arguments=gp.normalized_arguments,
                    schema=gp_schema,
                    tool_obj=gp_tool_obj,
                    workspace_root=workspace_root or getattr(self, "workspace_root", None),
                    node_target_resources=getattr(gn, "target_resources", ()),
                    is_external=bool(gp.external_source),
                    execution_id=eff_exec_id,
                    node_id=gn.node_id,
                    consume_confirmation=False,
                    record_to_ledger=False,
                    source_surface="plan_compiler",
                )
                if pre_pol.outcome == PolicyDecisionOutcome.REQUIRE_CONFIRMATION:
                    raw_t, rec_t = plan_pol_engine.confirmation_store.issue_confirmation(
                        tool_id=pre_pol.tool_id,
                        arg_digest=pre_pol.arg_digest,
                        resource_scope=pre_pol.resource_scope,
                        execution_id=eff_exec_id,
                        node_id=gn.node_id,
                        ttl_seconds=plan_pol_engine.config.confirmation_ttl_seconds,
                        max_lifetime_seconds=plan_pol_engine.config.max_confirmation_lifetime_seconds,
                        policy_version=plan_pol_engine.config.policy_version,
                        approved_by="plan_confirmation",
                    )
                    plan_node_confirmations[gn.node_id] = (raw_t, rec_t.confirmation_id)

        exec_res = await self.execute_task_graph(
            bundle.task_graph,
            bundle.proposals,
            max_concurrency=bundle.task_graph.max_concurrency,
            execution_id=eff_exec_id,
            central_ledger=eff_central,
            turn_ledger=eff_turn,
            tool_schemas=tool_schemas,
            tool_executor=tool_executor,
            caller_ring=eff_caller_ring,
            allowed_tools=eff_allowed,
            task_local_scope=eff_scope,
            acceptance_criteria=acceptance_criteria,
            node_timeouts=node_timeouts,
            default_node_timeout=default_node_timeout,
            task_prompt=bundle.structured_plan.objective,
            confirm_callback=confirm_callback,
            node_confirmations=plan_node_confirmations or None,
            workspace_root=workspace_root,
        )
        exec_res["bundle"] = bundle
        exec_res["executed"] = True
        return exec_res

    def replan_graph(
        self,
        graph: Any,
        replan_payload: Union[str, Dict[str, Any]],
        *,
        central_ledger: Optional[Any] = None,
        turn_ledger: Optional[Any] = None,
        known_proposals: Optional[Any] = None,
        workspace_root: Optional[Union[str, Path]] = None,
        checkpoint: bool = True,
    ) -> Any:
        """Apply a validated incremental replan while preserving verified nodes & ledger history."""
        from zelyr.planning.compiler import replan_task_graph

        eff_central = central_ledger or getattr(self, "central_ledger", None)
        eff_turn = turn_ledger or getattr(self, "current_ledger", None)
        return replan_task_graph(
            graph,
            replan_payload,
            central_ledger=eff_central,
            turn_ledger=eff_turn,
            known_proposals=known_proposals,
            workspace_root=workspace_root,
            tool_schemas=getattr(self, "tool_schemas", None),
            tool_registry=getattr(self, "tool_registry", None),
            checkpoint=checkpoint,
        )

    def assess_node_recovery(
        self,
        graph: Any,
        node_id: str,
        *,
        central_ledger: Optional[Any] = None,
        turn_ledger: Optional[Any] = None,
        proposals: Optional[Any] = None,
        policy: Optional[Any] = None,
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
        caller_ring: Optional[int] = None,
        allowed_tools: Optional[Iterable[str]] = None,
        task_local_scope: Optional[Any] = None,
        acceptance_criteria: Optional[str] = None,
        verifier_fn: Optional[Callable[..., Any]] = None,
        reconcile_state: bool = False,
        confirm_recovery: bool = False,
    ) -> Any:
        """Evaluate structured evidence, OutcomeVerifier checks, and live auth for a failed node."""
        from zelyr.engine.recovery import RecoveryEngine, RecoveryPolicy

        eff_central = central_ledger or getattr(self, "central_ledger", None)
        eff_turn = turn_ledger or getattr(self, "current_ledger", None)
        eff_policy = (
            policy
            if isinstance(policy, RecoveryPolicy)
            else RecoveryPolicy.from_config(self._active_config())
        )
        eff_schemas = tool_schemas if tool_schemas is not None else getattr(self, "tool_schemas", None)
        eff_ring = caller_ring if caller_ring is not None else getattr(self, "caller_ring", None)
        eff_allowed = (
            list(allowed_tools)
            if allowed_tools is not None
            else (list(getattr(self, "allowed_tools", None)) if getattr(self, "allowed_tools", None) is not None else None)
        )
        eff_scope = task_local_scope if task_local_scope is not None else getattr(self, "task_local_scope", None)
        return RecoveryEngine.assess_node_recovery(
            graph,
            node_id,
            central_ledger=eff_central,
            turn_ledger=eff_turn,
            proposals=proposals,
            policy=eff_policy,
            tool_registry=getattr(self, "tool_registry", None),
            active_schemas=eff_schemas,
            caller_ring=eff_ring,
            allowed_tools=eff_allowed,
            task_local_scope=eff_scope,
            zelyr_config=self._active_config(),
            policy_engine=getattr(self, "execution_policy_engine", None),
            policy_config=getattr(self, "execution_policy_config", None),
            workspace_root=getattr(self, "workspace_root", None),
            acceptance_criteria=acceptance_criteria,
            verifier_fn=verifier_fn,
            reconcile_state=reconcile_state,
            confirm_recovery=confirm_recovery,
        )

    def recover_graph_node(
        self,
        graph: Any,
        node_id: str,
        *,
        proposals: Optional[Any] = None,
        central_ledger: Optional[Any] = None,
        turn_ledger: Optional[Any] = None,
        policy: Optional[Any] = None,
        reconcile_state: bool = False,
        confirm_recovery: bool = False,
        execute_retry: bool = True,
        acceptance_criteria: Optional[str] = None,
        verifier_fn: Optional[Callable[..., Any]] = None,
    ) -> Dict[str, Any]:
        """Run verification-driven recovery assessment + atomic graph/ledger transition for one node."""
        from zelyr.engine.recovery import RecoveryEngine

        eff_central = central_ledger or getattr(self, "central_ledger", None)
        eff_turn = turn_ledger or getattr(self, "current_ledger", None)
        assessment = self.assess_node_recovery(
            graph,
            node_id,
            central_ledger=eff_central,
            turn_ledger=eff_turn,
            proposals=proposals,
            policy=policy,
            acceptance_criteria=acceptance_criteria,
            verifier_fn=verifier_fn,
            reconcile_state=reconcile_state,
            confirm_recovery=confirm_recovery,
        )
        res = RecoveryEngine.apply_recovery_decision(
            graph,
            assessment,
            prop_by_ref=proposals if isinstance(proposals, dict) else None,
            central_ledger=eff_central,
            turn_ledger=eff_turn,
            execute_retry=execute_retry,
        )
        return {"assessment": assessment, **res}

    def replace_failed_graph_node(
        self,
        graph: Any,
        failed_node_id: str,
        *,
        replacement_node_id: str,
        description: str,
        tool_id: str,
        arguments: Dict[str, Any],
        extra_dependencies: Sequence[str] = (),
        verification_requirement: Optional[str] = None,
        expected_outcome: str = "",
        timeout_seconds: Optional[float] = None,
        proposals: Optional[Any] = None,
        central_ledger: Optional[Any] = None,
        turn_ledger: Optional[Any] = None,
        checkpoint: bool = True,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Replace a failed node with an alternative proposal while preserving dependencies and verified work."""
        from zelyr.engine.recovery import RecoveryEngine

        eff_central = central_ledger or getattr(self, "central_ledger", None)
        eff_turn = turn_ledger or getattr(self, "current_ledger", None)
        return RecoveryEngine.recover_node_with_alternative(
            graph,
            failed_node_id,
            replacement_node_id=replacement_node_id,
            description=description,
            tool_id=tool_id,
            arguments=arguments,
            extra_dependencies=extra_dependencies,
            verification_requirement=verification_requirement,
            expected_outcome=expected_outcome,
            timeout_seconds=timeout_seconds,
            prop_by_ref=proposals if isinstance(proposals, dict) else None,
            central_ledger=eff_central,
            turn_ledger=eff_turn,
            checkpoint=checkpoint,
        )

    def add_corrective_graph_node(
        self,
        graph: Any,
        failed_node_id: str,
        *,
        corrective_node_id: str,
        description: str,
        tool_id: str,
        arguments: Dict[str, Any],
        verification_requirement: Optional[str] = None,
        expected_outcome: str = "",
        proposals: Optional[Any] = None,
        central_ledger: Optional[Any] = None,
        turn_ledger: Optional[Any] = None,
        checkpoint: bool = True,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Insert a corrective prerequisite step before ``failed_node_id`` without dropping dependencies."""
        from zelyr.engine.recovery import RecoveryEngine

        eff_central = central_ledger or getattr(self, "central_ledger", None)
        eff_turn = turn_ledger or getattr(self, "current_ledger", None)
        return RecoveryEngine.add_corrective_step(
            graph,
            failed_node_id,
            corrective_node_id=corrective_node_id,
            description=description,
            tool_id=tool_id,
            arguments=arguments,
            verification_requirement=verification_requirement,
            expected_outcome=expected_outcome,
            prop_by_ref=proposals if isinstance(proposals, dict) else None,
            central_ledger=eff_central,
            turn_ledger=eff_turn,
            checkpoint=checkpoint,
        )

