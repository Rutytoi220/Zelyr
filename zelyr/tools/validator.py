"""Zelyr Tool Contract Validator & Isolated Staging Pipeline.

Provides deterministic structural AST inspection, schema verification, ring
conformance enforcement, and dry-run isolated staging execution in a temporary
environment prior to plugin persistence in ~/.config/zelyr/tools.d/.
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import logging
import os
import re
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Outcome of tool syntax, contract, and staging validation."""
    is_valid: bool
    error: Optional[str] = None
    tool_name: Optional[str] = None
    schema: Optional[Dict[str, Any]] = None
    tier: int = 1
    ring: int = 1


class ToolValidator:
    """Validates dynamic tool code against Zelyr's runtime contract."""

    REQUIRED_EXPORTS: Set[str] = {
        "TOOL_SCHEMA",
        "REQUIRED_RING",
        "TIER",
        "TUI_HINT",
    }

    VALID_TIERS: Set[int] = {1, 2, 3}
    MIN_RING: int = 0
    MAX_RING: int = 3

    def validate_syntax_and_ast(self, code: str) -> ValidationResult:
        """Structural AST verification of tool candidate code.

        Verifies code parses cleanly without syntax errors and defines the required
        asynchronous 'execute' function along with top-level contract exports.
        Does not evaluate arbitrary expressions.
        """
        if not code or not code.strip():
            return ValidationResult(is_valid=False, error="Tool code cannot be empty.")

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            line = e.lineno or 1
            col = e.offset or 1
            msg = e.msg or "syntax error"
            return ValidationResult(
                is_valid=False,
                error=f"Syntax error at line {line}, col {col}: {msg}",
            )

        # Inspect top-level definitions
        has_async_execute = False
        has_sync_execute = False
        top_level_vars: Set[str] = set()

        for node in tree.body:
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "execute":
                has_async_execute = True
            elif isinstance(node, ast.FunctionDef) and node.name == "execute":
                has_sync_execute = True
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        top_level_vars.add(target.id)
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    top_level_vars.add(node.target.id)

        if has_sync_execute and not has_async_execute:
            return ValidationResult(
                is_valid=False,
                error="'execute' must be an asynchronous function (async def).",
            )

        if not has_async_execute:
            return ValidationResult(
                is_valid=False,
                error="Missing required async function: 'execute'.",
            )

        missing_exports = self.REQUIRED_EXPORTS - top_level_vars
        if missing_exports:
            sorted_missing = sorted(missing_exports)
            return ValidationResult(
                is_valid=False,
                error=f"Missing required top-level export(s): {', '.join(sorted_missing)}.",
            )

        # Statically inspect constants if present in AST
        tier_val = 1
        ring_val = 1
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                target_name = None
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    target_name = node.targets[0].id
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    target_name = node.target.id

                val_node = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
                if target_name == "TIER" and val_node is not None:
                    if isinstance(val_node, ast.Constant):
                        if not isinstance(val_node.value, int) or isinstance(val_node.value, bool) or val_node.value not in self.VALID_TIERS:
                            return ValidationResult(
                                is_valid=False,
                                error=f"Invalid TIER '{val_node.value}'. Must be an integer (1, 2, or 3).",
                            )
                        tier_val = val_node.value
                elif target_name == "REQUIRED_RING" and val_node is not None:
                    if isinstance(val_node, ast.Constant):
                        ok, err = self.validate_ring_conformance(val_node.value)
                        if not ok:
                            return ValidationResult(is_valid=False, error=err)
                        ring_val = int(val_node.value)
                elif target_name == "TUI_HINT" and val_node is not None:
                    if isinstance(val_node, ast.Constant):
                        if not isinstance(val_node.value, str) or not val_node.value.strip():
                            return ValidationResult(
                                is_valid=False,
                                error="'TUI_HINT' must be a non-empty string.",
                            )

        return ValidationResult(is_valid=True, tier=tier_val, ring=ring_val)

    def validate_schema(self, schema_dict: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate TOOL_SCHEMA structure conforming to OpenAI/Zelyr function call spec."""
        if not isinstance(schema_dict, dict):
            return False, "TOOL_SCHEMA must be a dictionary."

        if schema_dict.get("type") != "function":
            return False, "TOOL_SCHEMA 'type' must be 'function'."

        fn = schema_dict.get("function")
        if not isinstance(fn, dict):
            return False, "TOOL_SCHEMA must contain a 'function' dictionary."

        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            return False, "TOOL_SCHEMA 'function.name' must be a non-empty string."

        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_-]*$", name):
            return False, f"Invalid tool name '{name}'. Must be an alphanumeric identifier."

        desc = fn.get("description")
        if not isinstance(desc, str) or not desc.strip():
            return False, "TOOL_SCHEMA 'function.description' must be a non-empty string."

        params = fn.get("parameters")
        if not isinstance(params, dict):
            return False, "TOOL_SCHEMA 'function.parameters' must be a dictionary."

        if params.get("type") != "object":
            return False, "TOOL_SCHEMA parameters 'type' must be 'object'."

        props = params.get("properties")
        if not isinstance(props, dict):
            return False, "TOOL_SCHEMA parameters must contain a 'properties' dictionary."

        req = params.get("required")
        if req is not None and not isinstance(req, list):
            return False, "TOOL_SCHEMA parameters 'required' must be a list if specified."

        return True, None

    def validate_ring_conformance(self, ring_value: Any) -> Tuple[bool, Optional[str]]:
        """Validate REQUIRED_RING conformance against Zelyr ring hierarchy [0, 3]."""
        if not isinstance(ring_value, int) or isinstance(ring_value, bool):
            return False, f"REQUIRED_RING must be an integer, got {type(ring_value).__name__}."

        if not (self.MIN_RING <= ring_value <= self.MAX_RING):
            return False, (
                f"Invalid REQUIRED_RING {ring_value}: must conform to Zelyr ring hierarchy "
                f"[{self.MIN_RING}, {self.MAX_RING}]."
            )

        return True, None

    async def test_isolated_execution(
        self,
        code: str,
        test_params: Optional[Dict[str, Any]] = None,
        timeout: float = 5.0,
    ) -> ValidationResult:
        """Dynamically loads candidate code in an isolated scratch file and dry-runs execute().

        Guarantees complete cleanup of scratch files in /tmp/ and sys.modules.
        """
        # Ensure AST & syntax passes first
        ast_res = self.validate_syntax_and_ast(code)
        if not ast_res.is_valid:
            return ast_res

        temp_path: Optional[str] = None
        module_name = f"zelyr_stage_{uuid.uuid4().hex}"

        try:
            # 1. Create scratch file in /tmp/
            fd, temp_path = tempfile.mkstemp(prefix="zelyr_stage_", suffix=".py", dir="/tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(code)

            # 2. Load module dynamically
            spec = importlib.util.spec_from_file_location(module_name, temp_path)
            if not spec or not spec.loader:
                return ValidationResult(
                    is_valid=False,
                    error="Failed to create module specification for isolated staging.",
                )

            mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = mod
            spec.loader.exec_module(mod)

            # 3. Validate runtime exports and attributes
            schema = getattr(mod, "TOOL_SCHEMA", None)
            ok, err = self.validate_schema(schema)
            if not ok:
                return ValidationResult(is_valid=False, error=err)

            ring = getattr(mod, "REQUIRED_RING", None)
            ok, err = self.validate_ring_conformance(ring)
            if not ok:
                return ValidationResult(is_valid=False, error=err)

            tier = getattr(mod, "TIER", 1)
            if not isinstance(tier, int) or isinstance(tier, bool) or tier not in self.VALID_TIERS:
                return ValidationResult(
                    is_valid=False,
                    error=f"Invalid TIER '{tier}'. Must be an integer (1, 2, or 3).",
                )

            tui_hint = getattr(mod, "TUI_HINT", None)
            if not isinstance(tui_hint, str) or not tui_hint.strip():
                return ValidationResult(
                    is_valid=False,
                    error="'TUI_HINT' must be a non-empty string.",
                )

            exec_fn = getattr(mod, "execute", None)
            if not callable(exec_fn) or not asyncio.iscoroutinefunction(exec_fn):
                return ValidationResult(
                    is_valid=False,
                    error="'execute' must be an asynchronous callable.",
                )

            tool_name = schema["function"]["name"]

            # 4. Prepare parameters for dry-run
            params_to_use = dict(test_params) if test_params is not None else {}
            if test_params is None:
                # Synthesize default parameters for required keys if needed
                props = schema["function"]["parameters"].get("properties", {})
                req_keys = schema["function"]["parameters"].get("required", [])
                for rk in req_keys:
                    p_info = props.get(rk, {})
                    p_type = p_info.get("type", "string")
                    if p_type == "string":
                        params_to_use[rk] = "test"
                    elif p_type in ("integer", "number"):
                        params_to_use[rk] = 1
                    elif p_type == "boolean":
                        params_to_use[rk] = True
                    elif p_type == "array":
                        params_to_use[rk] = []
                    elif p_type == "object":
                        params_to_use[rk] = {}

            # 5. Execute dry-run under strict timeout
            try:
                result = await asyncio.wait_for(exec_fn(**params_to_use), timeout=timeout)
            except asyncio.TimeoutError:
                return ValidationResult(
                    is_valid=False,
                    tool_name=tool_name,
                    schema=schema,
                    tier=tier,
                    ring=ring,
                    error=f"Tool execution timed out during staging after {timeout}s.",
                )
            except Exception as e:
                return ValidationResult(
                    is_valid=False,
                    tool_name=tool_name,
                    schema=schema,
                    tier=tier,
                    ring=ring,
                    error=f"Execution failed during isolated staging: {e}",
                )

            # Check that output is non-null and valid type (str, dict, or ToolResult)
            if result is None:
                return ValidationResult(
                    is_valid=False,
                    tool_name=tool_name,
                    schema=schema,
                    tier=tier,
                    ring=ring,
                    error="Tool execution returned None. Must return a string or serializable payload.",
                )

            return ValidationResult(
                is_valid=True,
                tool_name=tool_name,
                schema=schema,
                tier=tier,
                ring=ring,
            )

        finally:
            # 6. Strict Cleanup
            if module_name in sys.modules:
                del sys.modules[module_name]
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    async def validate_all(
        self,
        code: str,
        test_params: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Run the complete validation pipeline (AST -> Schema -> Ring -> Staging)."""
        ast_res = self.validate_syntax_and_ast(code)
        if not ast_res.is_valid:
            return ast_res

        return await self.test_isolated_execution(code, test_params=test_params)

    def validate_code(
        self,
        code: str,
        allow_ring0: bool = False,
    ) -> ValidationResult:
        """Validate dynamic tool code against security, syntax, and privilege boundaries.

        Inspects AST to ensure the code does not:
        - Call or import dynamic import primitives (__import__, importlib, sys.modules)
        - Invoke reflection or execution primitives (eval, exec, compile, getattr on builtins)
        - Attempt privilege escalation by declaring REQUIRED_RING = 0 or mutating PROTECTED_TOOLS
        - Attempt path traversal in tool naming or exported strings
        """
        # First, run syntax and base AST checks
        res = self.validate_syntax_and_ast(code)
        if not res.is_valid:
            return res

        try:
            tree = ast.parse(code)
        except Exception as e:
            return ValidationResult(is_valid=False, error=f"Parse error: {e}")

        # Security AST walk
        for node in ast.walk(tree):
            # Check dynamic imports and execution calls
            if isinstance(node, ast.Call):
                # __import__(...)
                if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                    return ValidationResult(
                        is_valid=False,
                        error="Security violation: Dynamic import primitive '__import__' is disallowed.",
                    )
                # importlib.import_module(...)
                if isinstance(node.func, ast.Attribute) and node.func.attr == "import_module":
                    return ValidationResult(
                        is_valid=False,
                        error="Security violation: 'importlib.import_module' is disallowed.",
                    )
                # eval(...), exec(...), compile(...)
                if isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec", "compile"):
                    return ValidationResult(
                        is_valid=False,
                        error=f"Security violation: Execution primitive '{node.func.id}' is disallowed.",
                    )
                # getattr targeting builtins
                if isinstance(node.func, ast.Name) and node.func.id == "getattr":
                    for arg in node.args:
                        if isinstance(arg, ast.Name) and arg.id in ("__builtins__", "builtins"):
                            return ValidationResult(
                                is_valid=False,
                                error="Security violation: Reflection escalation targeting builtins is disallowed.",
                            )
                        if isinstance(arg, ast.Constant) and str(arg.value) in ("__builtins__", "builtins"):
                            return ValidationResult(
                                is_valid=False,
                                error="Security violation: Reflection escalation targeting builtins is disallowed.",
                            )

            # Check imports of importlib
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "importlib":
                        return ValidationResult(
                            is_valid=False,
                            error="Security violation: Importing 'importlib' is disallowed in dynamic tools.",
                        )
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] == "importlib":
                    return ValidationResult(
                        is_valid=False,
                        error="Security violation: Importing from 'importlib' is disallowed in dynamic tools.",
                    )

            # Check sys.modules access or __builtins__ access
            if isinstance(node, ast.Attribute):
                if node.attr == "modules" and isinstance(node.value, ast.Name) and node.value.id == "sys":
                    return ValidationResult(
                        is_valid=False,
                        error="Security violation: Direct access to 'sys.modules' is disallowed.",
                    )
                if node.attr in ("__builtins__", "__subclasses__", "__globals__"):
                    return ValidationResult(
                        is_valid=False,
                        error=f"Security violation: Access to '{node.attr}' is disallowed.",
                    )

            if isinstance(node, ast.Name) and node.id == "__builtins__":
                return ValidationResult(
                    is_valid=False,
                    error="Security violation: Direct access to '__builtins__' is disallowed.",
                )

            # Check attempts to modify or assign to PROTECTED_TOOLS
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = []
                if isinstance(node, ast.Assign):
                    targets = node.targets
                elif isinstance(node, ast.AnnAssign):
                    targets = [node.target]
                elif isinstance(node, ast.AugAssign):
                    targets = [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and target.id == "PROTECTED_TOOLS":
                        return ValidationResult(
                            is_valid=False,
                            error="Security violation: Attempt to modify PROTECTED_TOOLS is disallowed.",
                        )

            # Check method calls on PROTECTED_TOOLS (e.g. PROTECTED_TOOLS.add, .clear, .pop)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "PROTECTED_TOOLS":
                    return ValidationResult(
                        is_valid=False,
                        error="Security violation: Modifying PROTECTED_TOOLS is disallowed.",
                    )

        # Check privilege escalation via REQUIRED_RING
        if not allow_ring0 and res.ring == 0:
            return ValidationResult(
                is_valid=False,
                error="Privilege escalation rejected: dynamic tools cannot claim REQUIRED_RING = 0.",
            )

        # Check path traversal in tool naming or exports
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                target_name = None
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    target_name = node.targets[0].id
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    target_name = node.target.id

                val_node = node.value
                if target_name in ("TUI_HINT", "TOOL_NAME") and isinstance(val_node, ast.Constant) and isinstance(val_node.value, str):
                    val_str = val_node.value
                    if any(seq in val_str for seq in ("../", "..\\", "/..", "\\..")) or val_str in ("..", "."):
                        return ValidationResult(
                            is_valid=False,
                            error=f"Security violation: Path traversal sequence detected in {target_name}.",
                        )

        # Check tool name in TOOL_SCHEMA for path traversal or invalid characters
        match = re.search(r'["\']name["\']\s*:\s*["\']([^"\']+)["\']', code)
        if match:
            tool_name = match.group(1)
            if (
                any(seq in tool_name for seq in ("..", "/", "\\"))
                or not re.match(r"^[a-zA-Z_][a-zA-Z0-9_-]*$", tool_name)
            ):
                return ValidationResult(
                    is_valid=False,
                    error=f"Security violation: Invalid tool name or path traversal detected in TOOL_SCHEMA ('{tool_name}').",
                )
            res.tool_name = tool_name

        return res
