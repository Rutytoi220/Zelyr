"""Autonomous Tool Crafting and Runtime Skill Compilation Engine.

Provides the two-phase gatekeeper (Static AST audit + isolated sandbox test execution),
dynamic ZelyrTool / BaseTool synthesis, and hot-loading into ToolRegistry.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import inspect
import io
import json
import logging
import re
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from zelyr.config import get_config_dir
from zelyr.core.plugin_sandbox import SandboxedPluginRunner
from zelyr.core.plugins import reload_plugin, unregister_plugin
from zelyr.sdk.tool import ZelyrTool
from zelyr.security.ast_validator import ASTValidator, SecurityViolationError
from zelyr.tools.core import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


def _extract_schema_metadata_from_code(
    code: str,
) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """Statically inspect tool candidate code to extract tool name, description, and schema."""
    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "TOOL_SCHEMA":
                        if isinstance(node.value, ast.Dict):
                            for k, v in zip(node.value.keys, node.value.values):
                                if (
                                    isinstance(k, ast.Constant)
                                    and k.value == "function"
                                    and isinstance(v, ast.Dict)
                                ):
                                    name_val = None
                                    desc_val = None
                                    params_val = None
                                    for fk, fv in zip(v.keys, v.values):
                                        if isinstance(fk, ast.Constant):
                                            if fk.value == "name" and isinstance(fv, ast.Constant):
                                                name_val = str(fv.value)
                                            elif fk.value == "description" and isinstance(fv, ast.Constant):
                                                desc_val = str(fv.value)
                                            elif fk.value == "parameters" and isinstance(fv, ast.Dict):
                                                try:
                                                    params_val = ast.literal_eval(fv)
                                                except Exception:
                                                    pass
                                    return name_val, desc_val, params_val
    except Exception:
        pass
    return None, None, None


class _TestHarnessTarget:
    """Worker target executed in an isolated child process to verify tool candidates."""

    def __init__(self, code: str, test_code: str, tool_name: str) -> None:
        self.code = code
        self.test_code = test_code
        self.tool_name = tool_name

    def __call__(self, *args: Any, **kwargs: Any) -> ToolResult:
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        ns: Dict[str, Any] = {"__name__": "__main__", "__builtins__": __builtins__}

        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            try:
                # 1. Execute candidate code
                exec(self.code, ns)

                # 2. Setup sync/async bridge for execute function if defined
                raw_execute = ns.get("execute")
                if raw_execute and inspect.iscoroutinefunction(raw_execute):
                    def _bridge(*b_args: Any, **b_kwargs: Any) -> Any:
                        coro = raw_execute(*b_args, **b_kwargs)
                        try:
                            asyncio.get_running_loop()
                            return coro
                        except RuntimeError:
                            return asyncio.run(coro)

                    ns["execute"] = _bridge

                # 3. Execute companion test code if provided
                if self.test_code and self.test_code.strip():
                    if "await " in self.test_code:
                        lines = self.test_code.splitlines()
                        indented = "\n".join("    " + line for line in lines)
                        wrapper_code = f"async def __test_async_runner__():\n{indented}\n"
                        exec(wrapper_code, ns)
                        asyncio.run(ns["__test_async_runner__"]())
                    else:
                        exec(self.test_code, ns)
                else:
                    # Dry-run execution of candidate execute function if present
                    fn = ns.get("execute")
                    if fn is not None and callable(fn):
                        sig = inspect.signature(fn)
                        dummy_args: Dict[str, Any] = {}
                        for p in sig.parameters.values():
                            if p.default is inspect.Parameter.empty and p.kind not in (
                                inspect.Parameter.VAR_POSITIONAL,
                                inspect.Parameter.VAR_KEYWORD,
                            ):
                                if p.annotation in (int, float):
                                    dummy_args[p.name] = 1
                                elif p.annotation is str:
                                    dummy_args[p.name] = ""
                                elif p.annotation is dict:
                                    dummy_args[p.name] = {}
                                else:
                                    dummy_args[p.name] = 1
                        res = fn(**dummy_args)
                        if inspect.iscoroutine(res):
                            asyncio.run(res)

                out_str = stdout_buf.getvalue()
                err_str = stderr_buf.getvalue()
                return ToolResult(
                    success=True,
                    output=f"All tests passed for '{self.tool_name}'.",
                    metadata={"stdout": out_str, "stderr": err_str},
                )
            except AssertionError as ae:
                tb = traceback.format_exc()
                out_str = stdout_buf.getvalue()
                err_str = stderr_buf.getvalue()
                return ToolResult(
                    success=False,
                    error=f"Test assertion failed: {ae}\n{tb}",
                    output=out_str,
                    metadata={"stdout": out_str, "stderr": err_str, "traceback": tb},
                )
            except Exception as exc:
                tb = traceback.format_exc()
                out_str = stdout_buf.getvalue()
                err_str = stderr_buf.getvalue()
                return ToolResult(
                    success=False,
                    error=f"Test runtime exception ({type(exc).__name__}): {exc}\n{tb}",
                    output=out_str,
                    metadata={"stdout": out_str, "stderr": err_str, "traceback": tb},
                )


class _DynamicSandboxTarget:
    """Worker target executed in an isolated child process to run dynamic tool invocations."""

    def __init__(self, code: str, tool_name: str) -> None:
        self.code = code
        self.tool_name = tool_name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        merged: Dict[str, Any] = {}
        if args and isinstance(args[0], dict):
            merged.update(args[0])
        merged.update(kwargs)

        ns: Dict[str, Any] = {"__name__": "__main__", "__builtins__": __builtins__}
        exec(self.code, ns)

        # 1. Check for execute function
        candidate = ns.get("execute")

        # 2. Check for tool class inheriting from BaseTool or ZelyrTool
        if candidate is None:
            for val in ns.values():
                if inspect.isclass(val) and issubclass(val, (BaseTool, ZelyrTool)) and val not in (BaseTool, ZelyrTool):
                    inst = val()
                    candidate = getattr(inst, "execute", inst)
                    break

        # 3. Check for function named after tool
        if candidate is None and self.tool_name in ns and callable(ns[self.tool_name]):
            candidate = ns[self.tool_name]

        # 4. Check for any defined callable function
        if candidate is None:
            for k, val in ns.items():
                if callable(val) and not k.startswith("_") and getattr(val, "__module__", None) in (None, "__main__"):
                    candidate = val
                    break

        if candidate is None or not callable(candidate):
            raise RuntimeError(f"No executable function or tool class found for '{self.tool_name}'.")

        sig = inspect.signature(candidate)
        params_list = list(sig.parameters.values())
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params_list)
        single_dict_param = len(params_list) == 1 and (
            params_list[0].annotation == Dict[str, Any]
            or params_list[0].name in ("params", "arguments", "kwargs")
        )

        if single_dict_param and not has_var_keyword:
            res = candidate(merged)
        elif has_var_keyword:
            res = candidate(**merged)
        else:
            filtered = {k: v for k, v in merged.items() if k in sig.parameters}
            res = candidate(**filtered)

        if inspect.iscoroutine(res):
            res = asyncio.run(res)

        if isinstance(res, ToolResult):
            return res
        return ToolResult(success=True, output=res)


class DynamicSynthesizedTool(BaseTool):
    """Dynamically compiled tool executed inside a supervised sandboxed runner."""

    REQUIRED_RING: int = 1
    TIER: int = 1

    def __init__(
        self,
        name: str,
        description: str,
        code: str,
        parameters: Optional[Dict[str, Any]] = None,
        test_code: Optional[str] = None,
        sandboxed: bool = True,
        timeout: float = 30.0,
        max_memory_mb: float = 512.0,
    ) -> None:
        super().__init__(
            tool_id=name,
            name=name,
            description=description,
            tier=1,
        )
        self.code = code
        self.test_code = test_code
        self.sandboxed = sandboxed
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self._raw_parameters = parameters or {}
        self._custom_schema: Optional[Dict[str, Any]] = None

        self._init_parameters(parameters)

        if self.sandboxed:
            self._runner = SandboxedPluginRunner(
                target=_DynamicSandboxTarget(self.code, self.name),
                name=self.name,
                description=self.description,
                timeout=self.timeout,
                max_memory_mb=self.max_memory_mb,
            )
        else:
            self._runner = None

    def _init_parameters(self, params: Optional[Dict[str, Any]]) -> None:
        if not params:
            self._custom_schema = {"type": "object", "properties": {}}
            return

        if isinstance(params, dict) and params.get("type") == "object":
            self._custom_schema = params
            props = params.get("properties", {})
            required_set = set(params.get("required", []))
            for p_name, p_info in props.items():
                p_type = p_info.get("type", "string") if isinstance(p_info, dict) else "string"
                p_desc = p_info.get("description", "") if isinstance(p_info, dict) else ""
                self.add_parameter(
                    ToolParameter(
                        name=p_name,
                        type=p_type,
                        description=p_desc,
                        required=p_name in required_set,
                    )
                )
        elif isinstance(params, dict) and "properties" in params:
            self._custom_schema = {"type": "object", **params}
            props = params.get("properties", {})
            required_set = set(params.get("required", []))
            for p_name, p_info in props.items():
                p_type = p_info.get("type", "string") if isinstance(p_info, dict) else "string"
                p_desc = p_info.get("description", "") if isinstance(p_info, dict) else ""
                self.add_parameter(
                    ToolParameter(
                        name=p_name,
                        type=p_type,
                        description=p_desc,
                        required=p_name in required_set,
                    )
                )
        elif isinstance(params, dict):
            props = {}
            for k, v in params.items():
                if isinstance(v, dict):
                    props[k] = v
                else:
                    props[k] = {"type": str(v)}
            self._custom_schema = {"type": "object", "properties": props}
            for k, p_info in props.items():
                self.add_parameter(
                    ToolParameter(
                        name=k,
                        type=p_info.get("type", "string"),
                        description=p_info.get("description", ""),
                        required=True,
                    )
                )

    @property
    def schema(self) -> Dict[str, Any]:
        if self._custom_schema:
            return self._custom_schema
        return super().schema

    def execute(self, params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> ToolResult:
        merged = dict(params or {})
        merged.update(kwargs)
        if self._runner is not None:
            return self._runner.run(**merged)
        target = _DynamicSandboxTarget(self.code, self.name)
        return target(**merged)

    async def execute_async(self, params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> ToolResult:
        merged = dict(params or {})
        merged.update(kwargs)
        if self._runner is not None:
            return await self._runner.run_async(**merged)
        target = _DynamicSandboxTarget(self.code, self.name)
        return await asyncio.to_thread(target, **merged)


ZelyrTool.register(DynamicSynthesizedTool)


class ToolCrafterEngine:
    """Orchestrates static AST security validation, sandboxed testing, and tool registration."""

    def __init__(
        self,
        registry: Optional[Any] = None,
        tools_dir: Optional[Path] = None,
        sandbox_timeout: float = 10.0,
        max_memory_mb: float = 512.0,
    ) -> None:
        self.registry = registry
        self.tools_dir = tools_dir
        self.sandbox_timeout = float(sandbox_timeout)
        self.max_memory_mb = float(max_memory_mb)

    async def craft_tool(
        self,
        name: str,
        code: str,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        test_code: Optional[str] = None,
        persist: bool = False,
        registry: Optional[Any] = None,
        tools_dir: Optional[Path] = None,
        test_params: Optional[Dict[str, Any]] = None,
        db_path: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolResult:
        clean_name = (name or "").strip()
        if not clean_name or not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", clean_name):
            return ToolResult(
                success=False,
                error=f"Invalid tool name '{name}'. Must be a valid Python identifier matching ^[a-zA-Z_][a-zA-Z0-9_]*$.",
            )

        # Duplicate capability prevention if db_path is provided
        if db_path:
            try:
                from zelyr.db.memory import MemoryStore
                store = MemoryStore(db_path=db_path)
                key = f"tool.{clean_name}"
                existing_by_key = store.get_by_key(key)
                if existing_by_key and existing_by_key.get("is_active"):
                    return ToolResult(
                        success=False,
                        error=(
                            f"Duplicate capability rejected: Tool '{clean_name}' already exists in active capabilities. "
                            f"Existing description: {existing_by_key.get('content')}"
                        ),
                    )
            except Exception as e:
                logger.debug("Duplicate capability check skipped: %s", e)

        # Statically extract candidate description and parameters if missing
        cand_name, cand_desc, cand_params = _extract_schema_metadata_from_code(code)
        if cand_name and cand_name != clean_name:
            return ToolResult(
                success=False,
                error=f"Tool name mismatch: Provided name '{clean_name}' does not match TOOL_SCHEMA function name '{cand_name}'.",
            )
        if not description and cand_desc:
            description = cand_desc
        elif not description:
            description = f"Dynamically synthesized tool '{clean_name}'"

        if parameters is None and cand_params:
            parameters = cand_params

        # 1. Static Gate: AST security analysis
        try:
            ASTValidator.validate_code(code)
        except SyntaxError as e:
            return ToolResult(
                success=False,
                error=f"Syntax error in tool candidate code at line {e.lineno or 1}: {e.msg}",
            )
        except SecurityViolationError as e:
            return ToolResult(
                success=False,
                error=f"AST Security Violation in candidate code: {e}",
            )

        if test_code:
            try:
                ASTValidator.validate_code(test_code)
            except SyntaxError as e:
                return ToolResult(
                    success=False,
                    error=f"Syntax error in test code at line {e.lineno or 1}: {e.msg}",
                )
            except SecurityViolationError as e:
                return ToolResult(
                    success=False,
                    error=f"AST Security Violation in test code: {e}",
                )

        # Privilege escalation check: cannot claim REQUIRED_RING = 0
        try:
            tree = ast.parse(code)
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "REQUIRED_RING":
                            if isinstance(node.value, ast.Constant) and node.value.value == 0:
                                return ToolResult(
                                    success=False,
                                    error="Privilege escalation rejected: generated dynamic tools cannot claim REQUIRED_RING = 0.",
                                )
        except Exception:
            pass

        # 2. Dynamic Gate: Sandbox test execution
        effective_test_code = test_code
        if not effective_test_code and test_params is not None:
            effective_test_code = f"__test_args__ = {repr(test_params)}\nexecute(**__test_args__)"

        runner = SandboxedPluginRunner(
            target=_TestHarnessTarget(code, effective_test_code or "", clean_name),
            name=f"test_{clean_name}",
            timeout=self.sandbox_timeout,
            max_memory_mb=self.max_memory_mb,
        )

        test_res = await runner.run_async()
        if not test_res.success:
            return ToolResult(
                success=False,
                error=f"Sandbox test verification failed: {test_res.error}",
                output=test_res.output,
                metadata=test_res.metadata,
            )

        # 3. Dynamic Wrapping & Registration
        tool_instance = DynamicSynthesizedTool(
            name=clean_name,
            description=description,
            code=code,
            parameters=parameters,
            test_code=test_code,
            sandboxed=True,
            timeout=self.sandbox_timeout,
            max_memory_mb=self.max_memory_mb,
        )

        active_reg = registry or self.registry
        if active_reg is None:
            try:
                from zelyr.tool_registry import ToolRegistry
                active_reg = ToolRegistry()
            except Exception:
                active_reg = None

        if active_reg is not None:
            if clean_name in active_reg:
                active_reg.unregister_tool(clean_name)
            active_reg.register_tool(clean_name, tool_instance)

        # 4. Atomic installation if persist is requested or tools_dir is provided
        target_tools_dir = tools_dir or self.tools_dir
        should_persist = persist or (tools_dir is not None)
        target_file = None

        if should_persist:
            final_dir = target_tools_dir or (get_config_dir() / "tools.d")
            final_dir.mkdir(parents=True, exist_ok=True)
            target_file = final_dir / f"{clean_name}.py"
            tmp_file = final_dir / f".tmp_{clean_name}_{uuid.uuid4().hex}.py"

            disk_code = code
            if "TOOL_SCHEMA" not in code:
                plugin_schema = {
                    "type": "function",
                    "function": {
                        "name": clean_name,
                        "description": description or f"Tool {clean_name}",
                        "parameters": tool_instance.schema,
                    },
                }
                schema_json = json.dumps(plugin_schema, indent=4)
                disk_code = (
                    f'"""Auto-generated by Zelyr Autonomous Tool Crafter."""\n'
                    f"TOOL_SCHEMA = {schema_json}\n"
                    f"REQUIRED_RING = 1\n"
                    f"TIER = 1\n"
                    f'TUI_HINT = "Executing {clean_name}..."\n\n'
                    f"{code}\n"
                )

            try:
                tmp_file.write_text(disk_code, encoding="utf-8")
                tmp_file.replace(target_file)
            except Exception as e:
                if tmp_file.exists():
                    try:
                        tmp_file.unlink()
                    except OSError:
                        pass
                return ToolResult(
                    success=False,
                    error=f"Failed to write tool to {target_file}: {e}",
                )

            # Hot reload with rollback containment
            try:
                reload_plugin(target_file)
            except Exception as err:
                logger.warning(
                    "reload_plugin failed for %s: %s. Rolling back installation.",
                    target_file,
                    err,
                )
                if target_file.exists():
                    try:
                        target_file.unlink()
                    except OSError:
                        pass
                try:
                    unregister_plugin(clean_name)
                except Exception:
                    pass
                if active_reg is not None and clean_name in active_reg:
                    active_reg.unregister_tool(clean_name)
                return ToolResult(
                    success=False,
                    error=f"Installation failed during reload: {err}. File and runtime state rolled back.",
                )

            if db_path or (get_config_dir() / "tools.d").exists():
                try:
                    from zelyr.db.memory import MemoryStore
                    store = MemoryStore(db_path=db_path)
                    from zelyr.memory.capability_indexer import index_tool_capabilities
                    index_tool_capabilities(store=store, tools_dir=final_dir)
                except Exception as e:
                    logger.debug("Capability indexing skipped: %s", e)

        return ToolResult(
            success=True,
            output=f"Tool '{clean_name}' crafted and registered successfully.",
            metadata={
                "tool_name": clean_name,
                "persisted": bool(should_persist),
                "file_path": str(target_file) if target_file else None,
                "description": description,
                "schema": tool_instance.schema,
            },
        )


class ToolCrafterTool(BaseTool):
    """Outbound tool enabling agents to synthesize, validate, test, and register new tools."""

    REQUIRED_RING: int = 1
    TIER: int = 1

    def __init__(
        self,
        engine: Optional[ToolCrafterEngine] = None,
        tool_registry: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            tool_id="craft_tool",
            name="craft_tool",
            description=(
                "Synthesize, validate against Ring-0 AST rules, test in an isolated sandbox, "
                "and dynamically register a new Python tool for immediate execution."
            ),
            tier=1,
            is_core=False,
            token_cost=150,
        )
        self.tool_registry = tool_registry
        self.engine = engine or ToolCrafterEngine(registry=tool_registry)
        self._custom_schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Unique snake_case identifier for the tool (e.g. parse_custom_log).",
                },
                "description": {
                    "type": "string",
                    "description": "Clear description of what the tool does and when to use it.",
                },
                "parameters": {
                    "type": "object",
                    "description": "JSON Schema dictionary defining the tool parameters.",
                },
                "code": {
                    "type": "string",
                    "description": "Self-contained Python code implementing execute() or a BaseTool subclass.",
                },
                "test_code": {
                    "type": "string",
                    "description": "Self-contained test script with assertions verifying the tool.",
                },
                "persist": {
                    "type": "boolean",
                    "description": "Whether to persist the tool to disk across sessions.",
                    "default": False,
                },
            },
            "required": ["name", "description", "code"],
        }
        self.add_parameter(ToolParameter("name", "string", "Unique snake_case identifier", required=True))
        self.add_parameter(ToolParameter("description", "string", "Tool purpose description", required=True))
        self.add_parameter(ToolParameter("code", "string", "Python implementation code", required=True))
        self.add_parameter(ToolParameter("parameters", "object", "JSON Schema for parameters", required=False))
        self.add_parameter(ToolParameter("test_code", "string", "Verification test script", required=False))
        self.add_parameter(
            ToolParameter("persist", "boolean", "Persist tool to disk across sessions", required=False, default=False)
        )

    def set_registry(self, registry: Any) -> None:
        """Attach or update the tool registry reference."""
        self.tool_registry = registry
        if self.engine is not None and self.engine.registry is None:
            self.engine.registry = registry

    @property
    def schema(self) -> Dict[str, Any]:
        return self._custom_schema

    def execute(self, params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> ToolResult:
        from zelyr.core.async_bridge import run_sync
        return run_sync(self.execute_async(params=params, **kwargs))

    async def execute_async(self, params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> ToolResult:
        merged = dict(params or {})
        merged.update(kwargs)
        if "registry" not in merged and self.tool_registry is not None:
            merged["registry"] = self.tool_registry
        return await self.engine.craft_tool(**merged)


async def craft_tool(
    name: str,
    code: str,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    test_code: Optional[str] = None,
    persist: bool = False,
    registry: Optional[Any] = None,
    tools_dir: Optional[Path] = None,
    test_params: Optional[Dict[str, Any]] = None,
    db_path: Optional[str] = None,
    **kwargs: Any,
) -> ToolResult:
    """Convenience async entrypoint to craft, validate, test, and register a dynamic tool."""
    engine = ToolCrafterEngine(registry=registry, tools_dir=tools_dir)
    return await engine.craft_tool(
        name=name,
        code=code,
        description=description,
        parameters=parameters,
        test_code=test_code,
        persist=persist,
        registry=registry,
        tools_dir=tools_dir,
        test_params=test_params,
        db_path=db_path,
        **kwargs,
    )
