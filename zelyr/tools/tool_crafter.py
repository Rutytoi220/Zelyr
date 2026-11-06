"""Zelyr Autonomous Tool Crafter & Hardened Dynamic Hotloading Pipeline.

Empowers Zelyr to synthesize, AST validate, test in a sterile sandbox, atomically promote,
and hotload new dynamic tools into ~/.config/zelyr/tools.d/ and ToolRegistry.
Enforces privilege escalation prevention, AST security invariants, and fail-closed staging.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import logging
import os
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from zelyr.config import get_config_dir
from zelyr.core.plugins import get_tools_dir, reload_plugin, unregister_plugin
from zelyr.tools.core import BaseTool, ToolParameter, ToolResult
from zelyr.tools.validator import (
    ToolValidator,
    execute_sandbox_self_test,
    validate_tool_ast,
)

logger = logging.getLogger(__name__)

REQUIRED_RING = 1
TIER = 1
TUI_HINT = "Crafting and validating dynamic tool..."

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "craft_tool",
        "description": "Synthesizes, AST validates, sandboxes, and hotloads a new dynamic Python tool.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Unique identifier for the dynamic tool (valid Python identifier).",
                },
                "code": {
                    "type": "string",
                    "description": "Full Python source code for the tool module.",
                },
            },
            "required": ["name", "code"],
        },
    },
}


class CraftResult(dict, ToolResult):
    """Hybrid result conforming to both dictionary format and ToolResult semantics."""

    def __init__(
        self,
        data: Dict[str, Any],
        success: bool,
        error: Optional[str] = None,
        remedy_hint: Optional[str] = None,
        allowed_actions: Optional[List[str]] = None,
    ) -> None:
        dict.__init__(self, data)
        ToolResult.__init__(
            self,
            success=success,
            output=data if success else None,
            error=error,
            remedy_hint=remedy_hint or data.get("remedy_hint"),
            allowed_actions=allowed_actions or data.get("allowed_actions", []),
        )


def _extract_schema_metadata_from_code(
    code: str,
) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]], int, int]:
    """Statically inspect tool candidate code to extract tool name, description, schema, ring, and tier."""
    name_val: Optional[str] = None
    desc_val: Optional[str] = None
    params_val: Optional[Dict[str, Any]] = None
    ring_val = 1
    tier_val = 1

    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                target_name = None
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    target_name = node.targets[0].id
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    target_name = node.target.id

                val_node = node.value
                if target_name == "REQUIRED_RING" and val_node is not None:
                    if isinstance(val_node, ast.Constant) and isinstance(val_node.value, int):
                        ring_val = val_node.value
                elif target_name == "TIER" and val_node is not None:
                    if isinstance(val_node, ast.Constant) and isinstance(val_node.value, int):
                        tier_val = val_node.value
                elif target_name == "TOOL_SCHEMA" and val_node is not None and isinstance(val_node, ast.Dict):
                    for k, v in zip(val_node.keys, val_node.values):
                        if isinstance(k, ast.Constant):
                            if k.value == "name" and isinstance(v, ast.Constant):
                                name_val = str(v.value)
                            elif k.value == "description" and isinstance(v, ast.Constant):
                                desc_val = str(v.value)
                            elif k.value == "parameters" and isinstance(v, ast.Dict):
                                try:
                                    params_val = ast.literal_eval(v)
                                except Exception:
                                    pass
                            elif k.value == "function" and isinstance(v, ast.Dict):
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
    except Exception:
        pass

    return name_val, desc_val, params_val, ring_val, tier_val


class DynamicPluginTool(BaseTool):
    """Dynamic tool instance registered into ToolRegistry."""

    def __init__(self, name: str, mod: Any) -> None:
        schema = getattr(mod, "TOOL_SCHEMA", {})
        fn_info = schema.get("function", schema) if isinstance(schema, dict) else {}
        desc = fn_info.get("description", f"Dynamic tool {name}")
        ring = getattr(mod, "REQUIRED_RING", 1)
        tier = getattr(mod, "TIER", 1)

        super().__init__(
            tool_id=name,
            name=name,
            description=desc,
            tier=tier,
        )
        self.REQUIRED_RING = ring
        self.required_ring = ring
        self.mod = mod
        self.execute_func = getattr(mod, "execute")
        self._custom_schema = schema

        params_info = fn_info.get("parameters", {}) if isinstance(fn_info, dict) else {}
        if isinstance(params_info, dict) and "properties" in params_info:
            req_keys = set(params_info.get("required", []))
            for p_name, p_data in params_info.get("properties", {}).items():
                p_type = p_data.get("type", "string") if isinstance(p_data, dict) else "string"
                p_desc = p_data.get("description", "") if isinstance(p_data, dict) else ""
                self.add_parameter(
                    ToolParameter(
                        name=p_name,
                        type=p_type,
                        description=p_desc,
                        required=p_name in req_keys,
                    )
                )

    @property
    def schema(self) -> Dict[str, Any]:
        if self._custom_schema:
            return self._custom_schema
        return super().schema

    def _call_target(self, *args: Any, **kwargs: Any) -> Any:
        sig = inspect.signature(self.execute_func)
        params_list = list(sig.parameters.values())
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params_list)
        single_dict_param = len(params_list) == 1 and (
            params_list[0].annotation == Dict[str, Any]
            or params_list[0].name in ("params", "arguments", "kwargs")
        )
        merged: Dict[str, Any] = {}
        if args and isinstance(args[0], dict):
            merged.update(args[0])
        merged.update(kwargs)

        if single_dict_param and not has_var_kw:
            return self.execute_func(merged)
        elif has_var_kw:
            return self.execute_func(*args, **kwargs)
        else:
            filtered = {k: v for k, v in merged.items() if k in sig.parameters}
            return self.execute_func(**filtered)

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        import asyncio

        res = self._call_target(*args, **kwargs)
        if inspect.iscoroutine(res):
            try:
                asyncio.get_running_loop()

                async def _inner():
                    val = await res
                    if isinstance(val, ToolResult):
                        return val
                    return ToolResult(success=True, output=val)

                return _inner()
            except RuntimeError:
                from zelyr.core.async_bridge import run_sync

                res = run_sync(res)
        if isinstance(res, ToolResult):
            return res
        return ToolResult(success=True, output=res)

    async def execute_async(self, *args: Any, **kwargs: Any) -> Any:
        res = self._call_target(*args, **kwargs)
        if inspect.iscoroutine(res):
            res = await res
        if isinstance(res, ToolResult):
            return res
        return ToolResult(success=True, output=res)


async def _legacy_craft_tool(
    name: str,
    code: str,
    test_params: Optional[Dict[str, Any]] = None,
    db_path: Optional[str] = None,
    tools_dir: Optional[Path] = None,
) -> CraftResult:
    clean_name = (name or "").strip()
    if not clean_name or not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", clean_name):
        return CraftResult(
            {
                "status": "error",
                "error_type": "TOOL_CRAFT_FAILED",
                "error": f"Invalid tool name '{name}'. Must be a valid Python identifier matching ^[a-zA-Z_][a-zA-Z0-9_]*$.",
                "remedy_hint": "Inspect syntax, define required TOOL_SCHEMA and self_test(), and remove restricted dunder accesses.",
                "allowed_actions": ["craft_tool", "execute_command"],
            },
            success=False,
            error=f"Invalid tool name '{name}'. Must be a valid Python identifier matching ^[a-zA-Z_][a-zA-Z0-9_]*$.",
        )

    store = None
    if db_path or (get_config_dir() / "tools.d").exists():
        try:
            from zelyr.db.memory import MemoryStore

            store = MemoryStore(db_path=db_path)
            key = f"tool.{clean_name}"
            existing_by_key = store.get_by_key(key)
            if existing_by_key and existing_by_key.get("is_active"):
                return CraftResult(
                    {
                        "status": "error",
                        "error_type": "TOOL_CRAFT_FAILED",
                        "error": (
                            f"Duplicate capability rejected: Tool '{clean_name}' already exists in active capabilities. "
                            f"Existing description: {existing_by_key.get('content')}"
                        ),
                        "remedy_hint": "Inspect syntax, define required TOOL_SCHEMA and self_test(), and remove restricted dunder accesses.",
                        "allowed_actions": ["craft_tool", "execute_command"],
                    },
                    success=False,
                    error=(
                        f"Duplicate capability rejected: Tool '{clean_name}' already exists in active capabilities. "
                        f"Existing description: {existing_by_key.get('content')}"
                    ),
                )
        except Exception:
            pass

    validator = ToolValidator()
    val_res = await validator.validate_all(code, test_params=test_params)
    if not val_res.is_valid:
        return CraftResult(
            {
                "status": "error",
                "error_type": "TOOL_CRAFT_FAILED",
                "error": f"Validation failed: {val_res.error}",
                "remedy_hint": "Inspect syntax, define required TOOL_SCHEMA and self_test(), and remove restricted dunder accesses.",
                "allowed_actions": ["craft_tool", "execute_command"],
            },
            success=False,
            error=f"Validation failed: {val_res.error}",
        )

    if val_res.tool_name and val_res.tool_name != clean_name:
        return CraftResult(
            {
                "status": "error",
                "error_type": "TOOL_CRAFT_FAILED",
                "error": f"Tool name mismatch: Provided name '{clean_name}' does not match TOOL_SCHEMA function name '{val_res.tool_name}'.",
                "remedy_hint": "Inspect syntax, define required TOOL_SCHEMA and self_test(), and remove restricted dunder accesses.",
                "allowed_actions": ["craft_tool", "execute_command"],
            },
            success=False,
            error=f"Tool name mismatch: Provided name '{clean_name}' does not match TOOL_SCHEMA function name '{val_res.tool_name}'.",
        )

    if val_res.ring == 0:
        return CraftResult(
            {
                "status": "error",
                "error_type": "TOOL_CRAFT_FAILED",
                "error": "Privilege escalation rejected: generated dynamic tools cannot claim REQUIRED_RING = 0.",
                "remedy_hint": "Inspect syntax, define required TOOL_SCHEMA and self_test(), and remove restricted dunder accesses.",
                "allowed_actions": ["craft_tool", "execute_command"],
            },
            success=False,
            error="Privilege escalation rejected: generated dynamic tools cannot claim REQUIRED_RING = 0.",
        )

    target_dir = tools_dir or (get_config_dir() / "tools.d")
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / f"{clean_name}.py"

    backup_path = None
    orig_code = None
    if target_file.exists():
        try:
            orig_code = target_file.read_text(encoding="utf-8")
            timestamp = int(time.time() * 1000)
            backup_path = target_file.with_name(f"{target_file.name}.bak.{timestamp}")
            shutil.copy2(target_file, backup_path)
        except Exception as exc:
            logger.warning("Could not backup %s: %s", target_file, exc)

    tmp_file = target_dir / f".tmp_{clean_name}_{uuid.uuid4().hex}.py"
    try:
        tmp_file.write_text(code, encoding="utf-8")
        tmp_file.replace(target_file)
    except Exception as e:
        if tmp_file.exists():
            try:
                tmp_file.unlink()
            except OSError:
                pass
        return CraftResult(
            {
                "status": "error",
                "error_type": "TOOL_CRAFT_FAILED",
                "error": f"Failed to write tool to {target_file}: {e}",
                "remedy_hint": "Inspect syntax, define required TOOL_SCHEMA and self_test(), and remove restricted dunder accesses.",
                "allowed_actions": ["craft_tool", "execute_command"],
            },
            success=False,
            error=f"Failed to write tool to {target_file}: {e}",
        )

    this_mod = sys.modules.get(__name__)
    active_reload = getattr(this_mod, "reload_plugin", reload_plugin)
    try:
        active_reload(target_file)
    except Exception as err:
        logger.warning("reload_plugin failed for %s: %s. Rolling back installation.", target_file, err)
        if backup_path and backup_path.exists() and orig_code is not None:
            try:
                target_file.write_text(orig_code, encoding="utf-8")
            except Exception:
                pass
        elif target_file.exists():
            try:
                target_file.unlink()
            except OSError:
                pass

        try:
            unregister_plugin(clean_name)
        except Exception:
            pass

        if store:
            try:
                cap_record = store.get_by_key(f"tool.{clean_name}")
                if cap_record:
                    store.delete(cap_record["id"], hard_delete=False)
            except Exception:
                pass

        try:
            from zelyr.memory.capability_indexer import reconcile_tool_capabilities

            reconcile_tool_capabilities(store=store, tools_dir=target_dir)
        except Exception:
            pass

        return CraftResult(
            {
                "status": "error",
                "error_type": "TOOL_CRAFT_FAILED",
                "error": f"Installation failed during reload: {err}. File and runtime state rolled back.",
                "remedy_hint": "Inspect syntax, define required TOOL_SCHEMA and self_test(), and remove restricted dunder accesses.",
                "allowed_actions": ["craft_tool", "execute_command"],
            },
            success=False,
            error=f"Installation failed during reload: {err}. File and runtime state rolled back.",
        )

    if store:
        try:
            from zelyr.memory.capability_indexer import index_tool_capabilities

            index_tool_capabilities(store=store, tools_dir=target_dir)
        except Exception as e:
            logger.warning("Capability indexing failed for %s: %s", clean_name, e)

    return CraftResult(
        {
            "status": "success",
            "tool_name": clean_name,
            "path": str(target_file),
            "file_path": str(target_file),
            "ring": val_res.ring,
            "tier": val_res.tier,
            "provenance_hash": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        },
        success=True,
    )


async def craft_tool(
    name: str,
    code: str,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    test_code: Optional[str] = None,
    persist: bool = True,
    registry: Optional[Any] = None,
    test_params: Optional[Dict[str, Any]] = None,
    db_path: Optional[str] = None,
    tools_dir: Optional[Path] = None,
    **kwargs: Any,
) -> CraftResult:
    """Synthesize, AST validate, test in a sandbox, and hotload a dynamic tool."""
    clean_name = (name or "").strip()
    if not clean_name or not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", clean_name):
        err_msg = (
            f"Invalid tool name '{name}'. Must be a valid Python identifier matching ^[a-zA-Z_][a-zA-Z0-9_]*$."
        )
        return CraftResult(
            {
                "status": "error",
                "error_type": "TOOL_CRAFT_FAILED",
                "error": err_msg,
                "remedy_hint": "Inspect syntax, define required TOOL_SCHEMA and self_test(), and remove restricted dunder accesses.",
                "allowed_actions": ["craft_tool", "execute_command"],
            },
            success=False,
            error=err_msg,
        )

    # Legacy staging path when self_test is absent but TOOL_SCHEMA is present
    if test_code is None and "TOOL_SCHEMA" in code and "self_test" not in code:
        return await _legacy_craft_tool(
            name=clean_name,
            code=code,
            test_params=test_params,
            db_path=db_path,
            tools_dir=tools_dir,
        )

    # If companion test script was provided, delegate to ToolCrafterEngine for legacy/companion test pattern
    if test_code is not None:
        from zelyr.engine.tool_crafter import ToolCrafterEngine

        engine = ToolCrafterEngine(registry=registry, tools_dir=tools_dir)
        raw_res = await engine.craft_tool(
            name=clean_name,
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
        if raw_res.success:
            return CraftResult(
                {
                    "status": "success",
                    "tool_name": clean_name,
                    "ring": 1,
                    "tier": 1,
                    "path": raw_res.metadata.get("file_path", ""),
                    "provenance_hash": hashlib.sha256(code.encode("utf-8")).hexdigest(),
                },
                success=True,
                output=raw_res.output,
            )
        else:
            return CraftResult(
                {
                    "status": "error",
                    "error_type": "TOOL_CRAFT_FAILED",
                    "error": raw_res.error or "Tool crafting failed",
                    "remedy_hint": raw_res.remedy_hint or "Inspect syntax, define required TOOL_SCHEMA and self_test(), and remove restricted dunder accesses.",
                    "allowed_actions": ["craft_tool", "execute_command"],
                },
                success=False,
                error=raw_res.error,
            )

    # 1. AST Security and Contract Validation
    ast_ok, ast_err = validate_tool_ast(code)
    if not ast_ok:
        return CraftResult(
            {
                "status": "error",
                "error_type": "TOOL_CRAFT_FAILED",
                "error": ast_err or "AST Security validation failed",
                "remedy_hint": "Inspect syntax, define required TOOL_SCHEMA and self_test(), and remove restricted dunder accesses.",
                "allowed_actions": ["craft_tool", "execute_command"],
            },
            success=False,
            error=ast_err,
        )

    # 2. Sterile Sandbox Self-Test Execution
    sb_ok, sb_err = await execute_sandbox_self_test(code, timeout=3.0)
    if not sb_ok:
        return CraftResult(
            {
                "status": "error",
                "error_type": "TOOL_CRAFT_FAILED",
                "error": sb_err or "Sandbox self-test failed",
                "remedy_hint": "Inspect syntax, define required TOOL_SCHEMA and self_test(), and remove restricted dunder accesses.",
                "allowed_actions": ["craft_tool", "execute_command"],
            },
            success=False,
            error=sb_err,
        )

    # Extract validated metadata
    cand_name, _, _, ring_val, tier_val = _extract_schema_metadata_from_code(code)
    if cand_name and cand_name != clean_name:
        err_msg = f"Tool name mismatch: Provided name '{clean_name}' does not match TOOL_SCHEMA function name '{cand_name}'."
        return CraftResult(
            {
                "status": "error",
                "error_type": "TOOL_CRAFT_FAILED",
                "error": err_msg,
                "remedy_hint": "Inspect syntax, define required TOOL_SCHEMA and self_test(), and remove restricted dunder accesses.",
                "allowed_actions": ["craft_tool", "execute_command"],
            },
            success=False,
            error=err_msg,
        )

    # 3. Provenance Hash & Atomic Userland Promotion
    clean_code = code.strip()
    code_hash = hashlib.sha256(clean_code.encode("utf-8")).hexdigest()
    provenance_header = f"# @generated by zelyr-tool-crafter: {code_hash}\n"
    if clean_code.startswith("# @generated by zelyr-tool-crafter:"):
        formatted_code = clean_code + "\n"
    else:
        formatted_code = provenance_header + clean_code + "\n"

    target_dir = tools_dir or get_tools_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / f"{clean_name}.py"

    orig_mode = 0o644
    if target_file.exists():
        try:
            orig_mode = os.stat(target_file).st_mode
        except Exception:
            pass
        timestamp = int(time.time() * 1000)
        backup_file = target_file.with_name(f"{target_file.name}.bak.{timestamp}")
        try:
            shutil.copy2(target_file, backup_file)
        except Exception as exc:
            logger.warning(f"Could not create backup for {target_file}: {exc}")

    tmp_file = target_dir / f".tmp_{clean_name}_{uuid.uuid4().hex[:8]}.py"
    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(formatted_code)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.chmod(tmp_file, orig_mode)
        except Exception:
            pass
        os.replace(tmp_file, target_file)
    except Exception as exc:
        err_msg = f"Failed to atomically write tool to {target_file}: {exc}"
        return CraftResult(
            {
                "status": "error",
                "error_type": "TOOL_CRAFT_FAILED",
                "error": err_msg,
                "remedy_hint": "Inspect syntax, define required TOOL_SCHEMA and self_test(), and remove restricted dunder accesses.",
                "allowed_actions": ["craft_tool", "execute_command"],
            },
            success=False,
            error=err_msg,
        )
    finally:
        if tmp_file.exists():
            try:
                tmp_file.unlink()
            except OSError:
                pass

    # 4. Dynamic Hotloading into sys.modules, plugins._executors, and ToolRegistry
    module_name = f"zelyr.dynamic_tools.{clean_name}"
    spec = importlib.util.spec_from_file_location(module_name, str(target_file))
    if not spec or not spec.loader:
        err_msg = f"Failed to create module specification for {target_file}"
        return CraftResult(
            {
                "status": "error",
                "error_type": "TOOL_CRAFT_FAILED",
                "error": err_msg,
                "remedy_hint": "Inspect syntax, define required TOOL_SCHEMA and self_test(), and remove restricted dunder accesses.",
                "allowed_actions": ["craft_tool", "execute_command"],
            },
            success=False,
            error=err_msg,
        )

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        err_msg = f"Failed to execute loaded module {target_file}: {exc}"
        return CraftResult(
            {
                "status": "error",
                "error_type": "TOOL_CRAFT_FAILED",
                "error": err_msg,
                "remedy_hint": "Inspect syntax, define required TOOL_SCHEMA and self_test(), and remove restricted dunder accesses.",
                "allowed_actions": ["craft_tool", "execute_command"],
            },
            success=False,
            error=err_msg,
        )

    # Register in zelyr.core.plugins
    try:
        from zelyr.core import plugins

        exec_fn = getattr(mod, "execute", None)
        if exec_fn is not None:
            plugins._executors[clean_name] = exec_fn
        schema_val = getattr(mod, "TOOL_SCHEMA", None)
        if schema_val is not None:
            plugins._schemas.append(schema_val)
        plugins._rings[clean_name] = ring_val
        plugins._tui_hints[clean_name] = getattr(mod, "TUI_HINT", f"Executing {clean_name}...")
        plugins._plugin_file_paths[clean_name] = target_file
    except Exception as exc:
        logger.debug("Plugin state update skipped: %s", exc)

    # Register in ToolRegistry
    active_reg = registry
    if active_reg is None:
        try:
            from zelyr.tool_registry import ToolRegistry

            active_reg = ToolRegistry()
        except Exception:
            active_reg = None

    if active_reg is not None:
        try:
            if clean_name in active_reg:
                active_reg.unregister_tool(clean_name)
            tool_inst = DynamicPluginTool(name=clean_name, mod=mod)
            active_reg.register_tool(clean_name, tool_inst)
        except Exception as exc:
            logger.warning(f"Registration in ToolRegistry failed for {clean_name}: {exc}")

    # Capability Indexing
    if db_path or (get_config_dir() / "tools.d").exists():
        try:
            from zelyr.db.memory import MemoryStore
            from zelyr.memory.capability_indexer import index_tool_capabilities

            store = MemoryStore(db_path=db_path)
            index_tool_capabilities(store=store, tools_dir=target_dir)
        except Exception as exc:
            logger.debug("Capability indexing skipped: %s", exc)

    return CraftResult(
        {
            "status": "success",
            "tool_name": clean_name,
            "ring": ring_val,
            "tier": tier_val,
            "path": str(target_file),
            "provenance_hash": code_hash,
        },
        success=True,
    )


async def execute(name: str, code: str, **kwargs: Any) -> str:
    """Execute dynamic tool crafting and return structured JSON string envelope."""
    res = await craft_tool(name=name, code=code, **kwargs)
    if isinstance(res, dict):
        return json.dumps(res)
    elif isinstance(res, ToolResult):
        if res.success:
            if isinstance(res.output, dict):
                return json.dumps(res.output)
            return json.dumps({"status": "success", "output": res.output})
        return json.dumps({
            "status": "error",
            "error_type": "TOOL_CRAFT_FAILED",
            "error": res.error or "Tool crafting failed",
            "remedy_hint": res.remedy_hint or "Inspect syntax, define required TOOL_SCHEMA and self_test(), and remove restricted dunder accesses.",
            "allowed_actions": res.allowed_actions or ["craft_tool", "execute_command"],
        })
    return str(res)


class ToolCrafter:
    """Dynamic tool crafter coordinating AST validation, sterile sandbox test, and atomic promotion."""

    def __init__(self, tools_dir: Optional[Path] = None, registry: Optional[Any] = None) -> None:
        self.tools_dir = tools_dir or get_tools_dir()
        self.registry = registry

    async def craft(self, name: str, code: str, **kwargs: Any) -> CraftResult:
        return await craft_tool(
            name=name,
            code=code,
            tools_dir=self.tools_dir,
            registry=self.registry,
            **kwargs,
        )


class ToolCrafterTool(BaseTool):
    """Canonical ToolCrafterTool registered in ToolRegistry."""

    REQUIRED_RING: int = 1
    TIER: int = 1
    TUI_HINT: str = "Crafting and validating dynamic tool..."

    def __init__(
        self,
        tool_registry: Optional[Any] = None,
        tools_dir: Optional[Path] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            tool_id="craft_tool",
            name="craft_tool",
            description="Synthesizes, AST validates, sandboxes, and hotloads a new dynamic Python tool.",
            tier=1,
        )
        self.tool_registry = tool_registry
        self.tools_dir = tools_dir
        self.add_parameter(ToolParameter("name", "string", "Unique identifier for the dynamic tool", required=True))
        self.add_parameter(ToolParameter("description", "string", "Description of tool functionality", required=False))
        self.add_parameter(ToolParameter("code", "string", "Full Python source code for the tool module", required=True))
        self.add_parameter(ToolParameter("parameters", "object", "JSON Schema for parameters", required=False))
        self.add_parameter(ToolParameter("test_code", "string", "Optional verification test script", required=False))
        self.add_parameter(ToolParameter("persist", "boolean", "Persist tool to disk across sessions", required=False, default=True))

    def set_registry(self, registry: Any) -> None:
        self.tool_registry = registry

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Unique identifier for the dynamic tool"},
                "description": {"type": "string", "description": "Description of tool functionality"},
                "code": {"type": "string", "description": "Full Python source code for the tool module"},
                "parameters": {"type": "object", "description": "JSON Schema for parameters"},
                "test_code": {"type": "string", "description": "Optional verification test script"},
                "persist": {"type": "boolean", "description": "Persist tool to disk across sessions"},
            },
            "required": ["name", "description", "code"],
        }

    def execute(self, params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        import asyncio

        merged = dict(params or {})
        merged.update(kwargs)
        try:
            loop = asyncio.get_running_loop()
            return loop.create_task(self.execute_async(**merged))
        except RuntimeError:
            return asyncio.run(self.execute_async(**merged))

    async def execute_async(self, params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        merged = dict(params or {})
        merged.update(kwargs)
        name = merged.pop("name", "")
        code = merged.pop("code", "")
        reg = merged.pop("registry", self.tool_registry)
        tdir = merged.pop("tools_dir", self.tools_dir)
        return await execute(name=name, code=code, registry=reg, tools_dir=tdir, **merged)


def __getattr__(attr_name: str) -> Any:
    if attr_name in ("DynamicSynthesizedTool", "ToolCrafterEngine"):
        import zelyr.engine.tool_crafter as etc_mod

        val = getattr(etc_mod, attr_name)
        globals()[attr_name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {attr_name!r}")


__all__ = [
    "craft_tool",
    "execute",
    "validate_tool_ast",
    "execute_sandbox_self_test",
    "CraftResult",
    "ToolCrafter",
    "ToolCrafterTool",
    "DynamicPluginTool",
    "REQUIRED_RING",
    "TIER",
    "TUI_HINT",
    "TOOL_SCHEMA",
    "reload_plugin",
    "unregister_plugin",
]
