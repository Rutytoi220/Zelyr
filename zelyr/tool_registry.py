"""Type-safe registry and invocation surface for Zelyr tools.

``zelyr.tools`` defines two calling conventions for tool implementations:

- The dict-parameter family (``EchoTool``, ``ShellTool``, ``FileReadTool``,
  ``FileWriteTool``, ``SystemInfoTool``, ``FileTool``) whose ``execute``
  methods are ``async def execute(self, params: Dict[str, Any])``.  # type: ignore[override]
- The legacy keyword-argument family (``ShellCommandTool``, ``ReadFileTool``,
  ``WriteFileTool``, ``PythonExecTool``) whose ``execute`` methods are
  synchronous and accept explicit keyword parameters.

Callers that invoke ``tool.execute(...)`` directly must know which calling
convention a given tool uses. ``ToolRegistry`` removes that burden: it
validates registrations against the real :class:`zelyr.tools.BaseTool`,
tracks tools by their ``tool_id``, generates OpenAI-compatible
function-calling schemas for LLM tool-calling loops, and exposes a single
``execute()`` entry point that works for both tool families by delegating to
``BaseTool.__call__`` (which already performs the signature-based dispatch
and sync/async bridging).

``ToolRegistry`` acts as a specialized View/Adapter over
``zelyr.core.registry.Registry``.  When a ``core_registry`` is injected,
all storage operations delegate to it so that tools registered through
the Engine's registry are immediately visible through this adapter, and
vice-versa.  When no ``core_registry`` is provided, an isolated instance
is created for backwards compatibility.
"""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from zelyr.compat import config_dir
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from zelyr.tools.core import BaseTool, ToolResult


@dataclass(frozen=True)
class ToolAuthorizationDecision:
    """Authoritative pre-execution decision returned by shared tool authorization."""

    allowed: bool
    tool_name: str
    normalized_arguments: Dict[str, Any]
    reason: Optional[str] = None
    error_type: Optional[str] = None
    detail: Optional[str] = None
    required_ring: Optional[int] = None
    validation_error: Optional[Dict[str, Any]] = None
    warnings: tuple[str, ...] = ()


def _is_valid_ring_value(val: Any) -> bool:
    """Return True only for integer security rings in [0, 3] (excluding booleans)."""
    return isinstance(val, int) and not isinstance(val, bool) and 0 <= val <= 3


@dataclass(frozen=True)
class TaskLocalToolScope:
    """Explicit, isolated test/evaluation tool scope for task-local mock executors.

    Task-local test executors operate in a sealed sandbox disconnected from
    production ToolRegistry and global zelyr.core.plugins registrations.
    Only tools explicitly declared in ``tool_names``, ``tool_schemas``, or
    discovered in ``tools_dir`` are recognized as registered within the scope.
    """

    tool_names: Tuple[str, ...] = ()
    tool_schemas: Tuple[Dict[str, Any], ...] = ()
    tools_dir: Optional[Any] = None
    tool_rings: Optional[Dict[str, int]] = None

    def __init__(
        self,
        tool_names: Iterable[str] = (),
        tool_schemas: Iterable[Dict[str, Any]] = (),
        tools_dir: Optional[Any] = None,
        tool_rings: Optional[Dict[str, int]] = None,
    ) -> None:
        if isinstance(tool_names, (str, bytes, dict)) or not isinstance(tool_names, Iterable):
            raise ValueError("TaskLocalToolScope.tool_names must be an iterable of tool name strings.")
        cleaned_names: List[str] = []
        for n in tool_names:
            if not isinstance(n, str) or not n.strip():
                raise ValueError("TaskLocalToolScope.tool_names entries must be non-empty strings.")
            cleaned_names.append(n.strip())

        if isinstance(tool_schemas, (str, bytes, dict)) or not isinstance(tool_schemas, Iterable):
            raise ValueError("TaskLocalToolScope.tool_schemas must be an iterable of schema dicts.")
        cleaned_schemas: List[Dict[str, Any]] = []
        for s in tool_schemas:
            if not isinstance(s, dict):
                raise ValueError("TaskLocalToolScope.tool_schemas entries must be dicts.")
            cleaned_schemas.append(s)

        cleaned_rings: Optional[Dict[str, int]] = None
        if tool_rings is not None:
            if not isinstance(tool_rings, dict):
                raise ValueError("TaskLocalToolScope.tool_rings must be a dict mapping tool names to integer rings.")
            cleaned_rings = dict(tool_rings)

        object.__setattr__(self, "tool_names", tuple(cleaned_names))
        object.__setattr__(self, "tool_schemas", tuple(cleaned_schemas))
        object.__setattr__(self, "tools_dir", tools_dir)
        object.__setattr__(self, "tool_rings", cleaned_rings)

    def get_known_tools(self) -> set[str]:
        """Return the set of tool names declared in this isolated task-local scope."""
        names: set[str] = set(self.tool_names)
        for s in self.tool_schemas:
            if isinstance(s, dict):
                fn = s.get("function")
                s_name = fn.get("name") if isinstance(fn, dict) else s.get("name")
                if isinstance(s_name, str) and s_name.strip():
                    names.add(s_name.strip())
        if self.tools_dir is not None:
            from pathlib import Path

            try:
                td = Path(self.tools_dir)
                if td.is_dir():
                    for p in td.glob("*.py"):
                        if p.is_file() and not p.name.startswith("_"):
                            names.add(p.stem)
            except Exception:
                pass
        return names

    def find_schema(
        self,
        tool_id: str,
        active_schemas: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Locate schema strictly within task-local schemas (never consulting production registries)."""
        for pool in (active_schemas or (), self.tool_schemas):
            for s in pool:
                if isinstance(s, dict):
                    fn = s.get("function")
                    s_name = fn.get("name") if isinstance(fn, dict) else s.get("name")
                    if s_name == tool_id:
                        return s
        return None

    def resolve_ring(
        self,
        tool_id: str,
        active_schemas: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[int]:
        """Resolve security ring strictly within task-local metadata (never consulting production registries)."""
        if self.tool_rings is not None and tool_id in self.tool_rings:
            raw_ring = self.tool_rings[tool_id]
            return raw_ring if _is_valid_ring_value(raw_ring) else None
        for pool in (active_schemas or (), self.tool_schemas):
            for s in pool:
                if not isinstance(s, dict):
                    continue
                fn = s.get("function") if isinstance(s.get("function"), dict) else {}
                s_name = fn.get("name") or s.get("name")
                if s_name == tool_id:
                    for src in (fn, s):
                        for key in ("required_ring", "ring"):
                            if key in src and src[key] is not None:
                                return src[key] if _is_valid_ring_value(src[key]) else None
        return None


_ACTIVE_TASK_LOCAL_SCOPE: ContextVar[Optional[TaskLocalToolScope]] = ContextVar(
    "_ACTIVE_TASK_LOCAL_SCOPE",
    default=None,
)


@contextmanager
def activate_task_local_scope(scope: Optional[TaskLocalToolScope]):
    """Bind an active TaskLocalToolScope for the duration of a task-local test executor call."""
    if scope is None:
        yield
        return
    token = _ACTIVE_TASK_LOCAL_SCOPE.set(scope)
    try:
        yield
    finally:
        _ACTIVE_TASK_LOCAL_SCOPE.reset(token)


def get_active_task_local_scope() -> Optional[TaskLocalToolScope]:
    """Return the active TaskLocalToolScope if running inside an isolated test executor."""
    return _ACTIVE_TASK_LOCAL_SCOPE.get()


def bind_task_local_executor(
    executor: Callable[..., Any],
    *,
    tool_names: Iterable[str] = (),
    tool_schemas: Iterable[Dict[str, Any]] = (),
    tools_dir: Optional[Any] = None,
    tool_rings: Optional[Dict[str, int]] = None,
) -> Callable[..., Any]:
    """Attach an explicit TaskLocalToolScope to a task-local test executor callable."""
    if not callable(executor):
        raise ValueError("Task-local executor must be callable.")
    scope = TaskLocalToolScope(
        tool_names=tool_names,
        tool_schemas=tool_schemas,
        tools_dir=tools_dir,
        tool_rings=tool_rings,
    )
    import functools
    import inspect

    if inspect.iscoroutinefunction(executor):
        @functools.wraps(executor)
        async def _async_wrapped(*args: Any, **kwargs: Any) -> Any:
            with activate_task_local_scope(scope):
                return await executor(*args, **kwargs)

        setattr(_async_wrapped, "_task_local_tool_scope", scope)
        return _async_wrapped

    @functools.wraps(executor)
    def _sync_wrapped(*args: Any, **kwargs: Any) -> Any:
        with activate_task_local_scope(scope):
            return executor(*args, **kwargs)

    setattr(_sync_wrapped, "_task_local_tool_scope", scope)
    return _sync_wrapped


_TOOL_REGISTRY_ALIASES: Dict[str, str] = {
    "safe_file_search": "file_search",
    "file_opener": "file_read",
    "read_file": "file_read",
    "write_file": "file_write",
    "shell_command": "shell",
    "run_command": "shell",
    "execute_command": "shell",
    "bash": "shell",
}


def canonicalize_tool_invocation(
    tool_name: str,
    arguments: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Resolve canonical tool target before allowlist, ring, and schema checks.

    Prevents alias smuggling such as invoking ``interact_with_ui`` with browser DOM
    parameters (``element_id`` or ``click_element``/``fill_element``) to bypass
    allowlist or ring restrictions on ``interact_with_browser``.
    """
    clean_name = str(tool_name or "").strip()
    args = dict(arguments) if isinstance(arguments, dict) else {}
    if clean_name == "interact_with_ui" and (
        "element_id" in args or args.get("action") in ("click_element", "fill_element")
    ):
        clean_name = "interact_with_browser"
    return clean_name, args


def _is_mock_object(obj: Any) -> bool:
    """Return True if obj is a unittest.mock proxy object."""
    return obj is not None and type(obj).__module__.startswith("unittest.mock")


def _iter_unwrap_tool_chain(tool: Optional[Any]) -> List[Any]:
    """Return [tool, ...] following .target, ._target, and .__wrapped__ wrappers safely."""
    chain: List[Any] = []
    seen: set[int] = set()
    cur = tool
    while cur is not None and id(cur) not in seen and len(chain) < 8:
        seen.add(id(cur))
        chain.append(cur)
        if _is_mock_object(cur):
            break
        nxt = None
        for attr in ("_target", "target", "__wrapped__"):
            cand = getattr(cur, attr, None)
            if cand is not None and cand is not cur and not isinstance(cand, (str, bytes, int, float, bool, dict, list, tuple, set)):
                nxt = cand
                break
        cur = nxt
    return chain


def _verify_tool_object_identity(
    canonical_name: str,
    tool: Optional[Any],
    allowed_set: Optional[set[str]] = None,
) -> Optional[str]:
    """Verify that a resolved tool object (and any wrapped target) matches the authorized tool identity."""
    if tool is None or _is_mock_object(tool):
        return None
    expected_canonical = _TOOL_REGISTRY_ALIASES.get(canonical_name, canonical_name)
    valid_names = {canonical_name, expected_canonical}

    chain = _iter_unwrap_tool_chain(tool)
    for obj in chain:
        if _is_mock_object(obj):
            continue
        for attr in ("tool_id", "name"):
            val = getattr(obj, attr, None)
            if isinstance(val, str):
                cleaned = val.strip()
                if not cleaned or cleaned == "sandboxed_plugin" or cleaned in valid_names:
                    continue
                if allowed_set is not None and cleaned not in allowed_set:
                    return (
                        f"AuthorizationError: Tool object identity '{cleaned}' ({attr}) "
                        f"does not match requested tool '{canonical_name}' and is not in allowed tools list."
                    )
                if cleaned in BUILTIN_TOOLS or cleaned in SWARM_TOOLS:
                    return (
                        f"AuthorizationError: Built-in tool identity '{cleaned}' ({attr}) "
                        f"cannot be disguised under '{canonical_name}'."
                    )
    return None


def _validate_callable_signature(
    canonical_name: str,
    tool: Optional[Any],
    arguments: Dict[str, Any],
) -> Optional[str]:
    """Verify that normalized arguments can bind to the concrete tool callable when signature is explicit."""
    if tool is None or _is_mock_object(tool):
        return None
    import inspect

    chain = _iter_unwrap_tool_chain(tool)
    # Prefer the innermost unwrapped target so SandboxedPluginRunner(*args, **kwargs) is unwrapped to its target
    for obj in reversed(chain):
        if _is_mock_object(obj):
            continue
        target_fn = None
        if inspect.isclass(obj):
            target_fn = getattr(obj, "execute", None)
        elif hasattr(obj, "execute") and callable(obj.execute):
            target_fn = obj.execute
        elif callable(obj):
            target_fn = obj
        if target_fn is None or _is_mock_object(target_fn):
            continue
        try:
            sig = inspect.signature(target_fn)
        except (ValueError, TypeError):
            continue
        params_list = [
            p for p in sig.parameters.values()
            if p.name != "self"
        ]
        if not params_list:
            if arguments:
                return (
                    f"Callable for '{canonical_name}' accepts no arguments, "
                    f"but received {sorted(arguments.keys())!r}."
                )
            return None
        # Single-dict parameter convention: def execute(self, params: Dict[str, Any])
        if len(params_list) == 1 and params_list[0].kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ) and (
            params_list[0].annotation in (Dict[str, Any], dict, "Dict[str, Any]", "dict")
            or params_list[0].name in ("params", "arguments", "kwargs")
        ):
            return None
        # Dual convention used by test fixtures: def execute(self, params=None, **kwargs)
        if (
            params_list[0].name in ("params", "arguments")
            and params_list[0].default is not inspect.Parameter.empty
            and any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params_list)
        ):
            return None
        # Skip generic (*args, **kwargs) wrapper layer if an outer wrapper was reached
        if (
            len(params_list) == 2
            and params_list[0].kind == inspect.Parameter.VAR_POSITIONAL
            and params_list[1].kind == inspect.Parameter.VAR_KEYWORD
            and obj is not chain[-1]
        ):
            continue
        try:
            sig.bind(**arguments)
        except TypeError as exc:
            return f"Callable signature mismatch for '{canonical_name}': {exc}"
        return None
    return None


def _find_schema_for_tool(
    tool_id: str,
    tool: Optional[Any] = None,
    tool_registry: Optional[Any] = None,
    active_schemas: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Locate function schema from active_schemas, tool object, registry, or plugins."""
    if active_schemas:
        for s in active_schemas:
            if isinstance(s, dict):
                fn = s.get("function")
                s_name = fn.get("name") if isinstance(fn, dict) else s.get("name")
                if s_name == tool_id:
                    return s
    if tool is not None:
        for candidate in _iter_unwrap_tool_chain(tool):
            if _is_mock_object(candidate):
                continue
            s_obj = getattr(candidate, "schema", None) or getattr(candidate, "tool_schema", None)
            if isinstance(s_obj, dict) and (
                s_obj.get("properties") or s_obj.get("required") or "additionalProperties" in s_obj
            ):
                return s_obj
        for candidate in _iter_unwrap_tool_chain(tool):
            if _is_mock_object(candidate):
                continue
            s_obj = getattr(candidate, "schema", None) or getattr(candidate, "tool_schema", None)
            if isinstance(s_obj, dict):
                return s_obj
    if tool_registry is not None and hasattr(tool_registry, "get_tool"):
        t_obj = tool_registry.get_tool(tool_id)
        if t_obj is not None and not _is_mock_object(t_obj):
            s_obj = getattr(t_obj, "schema", None) or getattr(t_obj, "tool_schema", None)
            if isinstance(s_obj, dict):
                return s_obj
    try:
        from zelyr.core.plugins import get_tool_schemas

        for s in get_tool_schemas(0):
            if isinstance(s, dict):
                fn = s.get("function")
                s_name = fn.get("name") if isinstance(fn, dict) else s.get("name")
                if s_name == tool_id:
                    return s
    except Exception:
        pass
    return None


def resolve_tool_ring(
    tool_id: str,
    *,
    tool: Optional[Any] = None,
    tool_registry: Optional[Any] = None,
    active_schemas: Optional[List[Dict[str, Any]]] = None,
) -> Optional[int]:
    """Resolve explicit integer security ring [0, 3], or return None on unknown/invalid ring.

    Does NOT swallow exceptions raised by registry ring lookup so restricted callers
    fail closed on lookup failures.
    """
    import sys

    reg = tool_registry
    if reg is None:
        reg = ToolRegistry()

    # External fallback tools always derive required_ring from host policy, never remote metadata.
    if reg is not None and not _is_mock_object(reg) and hasattr(reg, "get_external_tool_entry"):
        ext_entry = reg.get_external_tool_entry(tool_id)
        if ext_entry is not None:
            raw_ext_ring = getattr(ext_entry, "required_ring", None)
            return raw_ext_ring if _is_valid_ring_value(raw_ext_ring) else None

    eff_tool = tool
    if eff_tool is None and hasattr(reg, "get_tool"):
        eff_tool = reg.get_tool(tool_id)

    # Check explicit ring attributes across the tool and any wrapped targets so invalid values
    # always fail closed and wrapped privileged tools preserve their highest privilege requirement.
    if eff_tool is not None:
        if _is_mock_object(eff_tool):
            # A bare unittest.mock object without explicit integer ring metadata fails closed under ring checks
            raw_mock_ring = None
            if "REQUIRED_RING" in getattr(eff_tool, "__dict__", {}):
                raw_mock_ring = eff_tool.__dict__["REQUIRED_RING"]
            elif "required_ring" in getattr(eff_tool, "__dict__", {}):
                raw_mock_ring = eff_tool.__dict__["required_ring"]
            if raw_mock_ring is not None:
                return raw_mock_ring if _is_valid_ring_value(raw_mock_ring) else None
        else:
            explicit_rings: List[int] = []
            for candidate in _iter_unwrap_tool_chain(eff_tool):
                if _is_mock_object(candidate):
                    continue
                cand_found_explicit = False
                for attr_owner in (candidate, type(candidate)):
                    for attr_name in ("required_ring", "REQUIRED_RING"):
                        if hasattr(attr_owner, attr_name):
                            raw_ring = getattr(attr_owner, attr_name)
                            if raw_ring is not None:
                                if not _is_valid_ring_value(raw_ring):
                                    return None
                                explicit_rings.append(raw_ring)
                                cand_found_explicit = True
                mod_name = type(candidate).__module__
                if mod_name in sys.modules:
                    mod = sys.modules[mod_name]
                    if hasattr(mod, "REQUIRED_RING"):
                        raw_ring = getattr(mod, "REQUIRED_RING")
                        if raw_ring is not None:
                            if not _is_valid_ring_value(raw_ring):
                                return None
                            explicit_rings.append(raw_ring)
                            cand_found_explicit = True
                if not cand_found_explicit:
                    cand_id = getattr(candidate, "tool_id", None) or getattr(candidate, "name", None)
                    if isinstance(cand_id, str) and cand_id in BUILTIN_TOOLS:
                        explicit_rings.append(0)
            if explicit_rings:
                return min(explicit_rings)

    # Honor registry _get_tool_ring (including mocked/overridden implementations)
    if reg is not None and hasattr(reg, "_get_tool_ring") and not _is_mock_object(reg):
        if eff_tool is not None or type(reg)._get_tool_ring is not ToolRegistry._get_tool_ring:
            ring_val = reg._get_tool_ring(tool_id, eff_tool)
            return ring_val if _is_valid_ring_value(ring_val) else None

    from zelyr.core.plugins import _executors, _rings, _schemas, load_plugins

    if not _rings and not _executors and not _schemas:
        load_plugins()

    if tool_id in _rings:
        raw_ring = _rings[tool_id]
        return raw_ring if _is_valid_ring_value(raw_ring) else None

    canonical_alias = _TOOL_REGISTRY_ALIASES.get(tool_id, tool_id)
    eff_tool_id = getattr(eff_tool, "tool_id", None) if eff_tool is not None else None
    if tool_id in BUILTIN_TOOLS or canonical_alias in BUILTIN_TOOLS or eff_tool_id in BUILTIN_TOOLS:
        return 0
    if tool_id in SWARM_TOOLS or eff_tool_id in SWARM_TOOLS:
        return 1

    if active_schemas:
        for s in active_schemas:
            if not isinstance(s, dict):
                continue
            fn = s.get("function") if isinstance(s.get("function"), dict) else {}
            s_name = fn.get("name") or s.get("name")
            if s_name == tool_id:
                for src in (fn, s):
                    for key in ("required_ring", "ring"):
                        if key in src and src[key] is not None:
                            return src[key] if _is_valid_ring_value(src[key]) else None

    if eff_tool is not None:
        return 1

    return None


def authorize_and_validate_tool_call(
    tool_name: str,
    arguments: Any,
    *,
    caller_ring: Optional[int] = None,
    allowed_tools: Optional[Iterable[str]] = None,
    active_schemas: Optional[List[Dict[str, Any]]] = None,
    tool_registry: Optional[Any] = None,
    tool: Optional[Any] = None,
    known_tools: Optional[Iterable[str]] = None,
    task_local_scope: Optional[Any] = None,
    require_registered: bool = True,
    normalize_args: bool = True,
    validate_schema: bool = True,
    argument_source: str = "internal",
) -> ToolAuthorizationDecision:
    """Single authoritative permission, canonicalization, and schema validation gate."""
    from zelyr.engine.normalizer import (
        SafeArgumentNormalizer,
        _schema_error,
        _security_override_error,
        _strict_json_loads_object,
        find_reserved_authorization_key,
        validate_arguments,
    )

    if argument_source not in ("model", "internal"):
        return ToolAuthorizationDecision(
            allowed=False,
            tool_name=str(tool_name or ""),
            normalized_arguments={},
            reason="authorization_denied",
            error_type="AUTHORIZATION_DENIED",
            detail=f"AuthorizationError: Invalid argument_source {argument_source!r}.",
        )

    parsed_dict_args: Optional[Dict[str, Any]] = None
    if isinstance(arguments, dict):
        parsed_dict_args = dict(arguments)
    elif isinstance(arguments, str) and normalize_args:
        parsed_obj, _ = _strict_json_loads_object(arguments)
        if parsed_obj is not None:
            parsed_dict_args = parsed_obj

    args_is_dict = parsed_dict_args is not None
    raw_args_dict = dict(parsed_dict_args) if args_is_dict else {}

    canonical_name, canonical_args = canonicalize_tool_invocation(tool_name, raw_args_dict)
    if not canonical_name:
        return ToolAuthorizationDecision(
            allowed=False,
            tool_name="",
            normalized_arguments=canonical_args,
            reason="tool_unavailable",
            error_type="TOOL_NOT_FOUND",
            detail="Empty tool name rejected.",
        )

    # 0. Fail closed when untrusted model arguments attempt to supply reserved auth metadata.
    if argument_source == "model" and parsed_dict_args is not None:
        reserved_key = find_reserved_authorization_key(parsed_dict_args)
        if reserved_key is not None:
            sec_err = _security_override_error(canonical_name, reserved_key)
            return ToolAuthorizationDecision(
                allowed=False,
                tool_name=canonical_name,
                normalized_arguments={},
                reason="authorization_denied",
                error_type="AUTHORIZATION_DENIED",
                detail=sec_err["error"],
                validation_error=sec_err,
            )

    # 0b. Validate authorization metadata upfront (fail closed on malformed/conflicting metadata).
    if task_local_scope is not None:
        if not isinstance(task_local_scope, TaskLocalToolScope):
            return ToolAuthorizationDecision(
                allowed=False,
                tool_name=canonical_name,
                normalized_arguments=canonical_args,
                reason="authorization_denied",
                error_type="AUTHORIZATION_DENIED",
                detail="AuthorizationError: Invalid task_local_scope metadata.",
            )
        if tool_registry is not None or tool is not None:
            return ToolAuthorizationDecision(
                allowed=False,
                tool_name=canonical_name,
                normalized_arguments=canonical_args,
                reason="authorization_denied",
                error_type="AUTHORIZATION_DENIED",
                detail="AuthorizationError: Isolated task-local test scope cannot access or combine with a production tool_registry.",
            )

    if caller_ring is not None and not _is_valid_ring_value(caller_ring):
        return ToolAuthorizationDecision(
            allowed=False,
            tool_name=canonical_name,
            normalized_arguments=canonical_args,
            reason="authorization_denied",
            error_type="AUTHORIZATION_DENIED",
            detail=f"AuthorizationError: Invalid caller_ring {caller_ring!r}.",
        )

    # 1. Explicit tool allowlist enforcement (evaluated on canonical target name).
    if allowed_tools is not None:
        if isinstance(allowed_tools, (str, bytes, dict)) or not isinstance(allowed_tools, Iterable):
            return ToolAuthorizationDecision(
                allowed=False,
                tool_name=canonical_name,
                normalized_arguments=canonical_args,
                reason="authorization_denied",
                error_type="AUTHORIZATION_DENIED",
                detail="AuthorizationError: Invalid allowed_tools metadata.",
            )
        allowed_set: set[str] = set()
        for item in allowed_tools:
            if not isinstance(item, str) or not item.strip():
                return ToolAuthorizationDecision(
                    allowed=False,
                    tool_name=canonical_name,
                    normalized_arguments=canonical_args,
                    reason="authorization_denied",
                    error_type="AUTHORIZATION_DENIED",
                    detail="AuthorizationError: Invalid allowed_tools entry.",
                )
            allowed_set.add(item.strip())
        if canonical_name not in allowed_set:
            return ToolAuthorizationDecision(
                allowed=False,
                tool_name=canonical_name,
                normalized_arguments=canonical_args,
                reason="authorization_denied",
                error_type="AUTHORIZATION_DENIED",
                detail=f"AuthorizationError: Tool '{canonical_name}' is not in allowed tools list.",
            )

    active_schemas_list: Optional[List[Dict[str, Any]]] = None
    schema_names: set[str] = set()
    if active_schemas is not None:
        if isinstance(active_schemas, (str, bytes, dict)) or not isinstance(active_schemas, Iterable):
            return ToolAuthorizationDecision(
                allowed=False,
                tool_name=canonical_name,
                normalized_arguments=canonical_args,
                reason="authorization_denied",
                error_type="AUTHORIZATION_DENIED",
                detail="AuthorizationError: Invalid active_schemas metadata.",
            )
        active_schemas_list = []
        for s in active_schemas:
            if not isinstance(s, dict):
                return ToolAuthorizationDecision(
                    allowed=False,
                    tool_name=canonical_name,
                    normalized_arguments=canonical_args,
                    reason="authorization_denied",
                    error_type="AUTHORIZATION_DENIED",
                    detail="AuthorizationError: Malformed tool schema in active_schemas.",
                )
            fn = s.get("function")
            s_name = fn.get("name") if isinstance(fn, dict) else s.get("name")
            if not isinstance(s_name, str) or not s_name.strip():
                return ToolAuthorizationDecision(
                    allowed=False,
                    tool_name=canonical_name,
                    normalized_arguments=canonical_args,
                    reason="authorization_denied",
                    error_type="AUTHORIZATION_DENIED",
                    detail="AuthorizationError: Malformed tool schema missing valid tool name.",
                )
            active_schemas_list.append(s)
            schema_names.add(s_name.strip())

    if known_tools is not None:
        if isinstance(known_tools, (str, bytes, dict)) or not isinstance(known_tools, Iterable):
            return ToolAuthorizationDecision(
                allowed=False,
                tool_name=canonical_name,
                normalized_arguments=canonical_args,
                reason="authorization_denied",
                error_type="AUTHORIZATION_DENIED",
                detail="AuthorizationError: Invalid known_tools metadata.",
            )
        known_set: set[str] = set()
        for k in known_tools:
            if not isinstance(k, str) or not k.strip():
                return ToolAuthorizationDecision(
                    allowed=False,
                    tool_name=canonical_name,
                    normalized_arguments=canonical_args,
                    reason="authorization_denied",
                    error_type="AUTHORIZATION_DENIED",
                    detail="AuthorizationError: Invalid known_tools entry.",
                )
            known_set.add(k.strip())
        if canonical_name not in known_set:
            return ToolAuthorizationDecision(
                allowed=False,
                tool_name=canonical_name,
                normalized_arguments=canonical_args,
                reason="tool_unavailable",
                error_type="TOOL_NOT_FOUND",
                detail=f"Tool '{canonical_name}' is not available in the active runtime manifest.",
            )

    resolved_ring: Optional[int] = None
    tool_schema: Optional[Dict[str, Any]] = None

    # 2A. Isolated Task-Local Test Scope (strictly disconnected from production registries).
    if task_local_scope is not None:
        scope_known = task_local_scope.get_known_tools()
        if canonical_name not in scope_known:
            return ToolAuthorizationDecision(
                allowed=False,
                tool_name=canonical_name,
                normalized_arguments=canonical_args,
                reason="tool_unavailable",
                error_type="TOOL_NOT_FOUND",
                detail=f"Tool '{canonical_name}' is not registered in the isolated task-local test scope.",
            )
        if caller_ring is not None:
            resolved_ring = task_local_scope.resolve_ring(canonical_name, active_schemas=active_schemas_list)
            if resolved_ring is None:
                return ToolAuthorizationDecision(
                    allowed=False,
                    tool_name=canonical_name,
                    normalized_arguments=canonical_args,
                    reason="authorization_denied",
                    error_type="AUTHORIZATION_DENIED",
                    detail=f"AuthorizationError: Unknown or invalid security ring for task-local tool '{canonical_name}'.",
                )
            if caller_ring > resolved_ring:
                return ToolAuthorizationDecision(
                    allowed=False,
                    tool_name=canonical_name,
                    normalized_arguments=canonical_args,
                    reason="authorization_denied",
                    error_type="AUTHORIZATION_DENIED",
                    detail=(
                        f"AuthorizationError: Tool '{canonical_name}' requires higher permissions "
                        f"(requires ring {resolved_ring}, caller ring is {caller_ring})."
                    ),
                    required_ring=resolved_ring,
                )
        tool_schema = task_local_scope.find_schema(canonical_name, active_schemas=active_schemas_list)
    else:
        # 2B. Production Execution Boundary (must resolve against authoritative registry/plugins/schemas).
        active_test_scope = get_active_task_local_scope()
        if active_test_scope is not None:
            is_local_reg_tool = (
                tool_registry is not None
                and hasattr(tool_registry, "_core_registry")
                and (
                    tool_registry._core_registry.has_tool(canonical_name)
                    or canonical_name == "echo"
                )
            )
            local_cand = tool_registry.get_tool(canonical_name) if is_local_reg_tool and hasattr(tool_registry, "get_tool") else None
            is_external_escape = (
                (
                    getattr(local_cand, "is_external", False)
                    or (hasattr(tool_registry, "get_external_tool_entry") and tool_registry.get_external_tool_entry(canonical_name) is not None)
                )
                and canonical_name not in active_test_scope.get_known_tools()
            )
            if not is_local_reg_tool or is_external_escape:
                return ToolAuthorizationDecision(
                    allowed=False,
                    tool_name=canonical_name,
                    normalized_arguments=canonical_args,
                    reason="authorization_denied",
                    error_type="AUTHORIZATION_DENIED",
                    detail="AuthorizationError: Isolated task-local test executor cannot access production tool registries.",
                )

        if tool_registry is not None and (
            not hasattr(tool_registry, "get_tool") or not callable(getattr(tool_registry, "get_tool", None))
        ):
            return ToolAuthorizationDecision(
                allowed=False,
                tool_name=canonical_name,
                normalized_arguments=canonical_args,
                reason="authorization_denied",
                error_type="AUTHORIZATION_DENIED",
                detail="AuthorizationError: Invalid tool_registry metadata.",
            )

        reg = tool_registry if tool_registry is not None else ToolRegistry()
        eff_tool = tool
        if eff_tool is None and (tool_registry is not None or require_registered):
            try:
                eff_tool = reg.get_tool(canonical_name)
            except Exception as exc:
                return ToolAuthorizationDecision(
                    allowed=False,
                    tool_name=canonical_name,
                    normalized_arguments=canonical_args,
                    reason="authorization_denied",
                    error_type="AUTHORIZATION_DENIED",
                    detail=f"AuthorizationError: Registry lookup failed for '{canonical_name}': {exc}",
                )

        # External fallback lifecycle & authorization check (never expose or execute unauthorized/disconnected tools)
        ext_entry = (
            reg.get_external_tool_entry(canonical_name)
            if (reg is not None and not _is_mock_object(reg) and hasattr(reg, "get_external_tool_entry"))
            else None
        )
        if ext_entry is not None:
            if not getattr(ext_entry, "authorized", False):
                return ToolAuthorizationDecision(
                    allowed=False,
                    tool_name=canonical_name,
                    normalized_arguments={},
                    reason="authorization_denied",
                    error_type="AUTHORIZATION_DENIED",
                    detail=(
                        f"AuthorizationError: External tool '{canonical_name}' from source "
                        f"'{ext_entry.source_id}' is not authorized for execution."
                    ),
                )
            ext_lc = getattr(getattr(ext_entry, "lifecycle", None), "value", str(getattr(ext_entry, "lifecycle", "")))
            if ext_lc != "AVAILABLE" or eff_tool is None or (
                hasattr(eff_tool, "is_available") and not eff_tool.is_available()
            ):
                return ToolAuthorizationDecision(
                    allowed=False,
                    tool_name=canonical_name,
                    normalized_arguments={},
                    reason="tool_unavailable",
                    error_type="TOOL_UNAVAILABLE",
                    detail=(
                        f"External tool '{canonical_name}' from source '{ext_entry.source_id}' "
                        f"is unavailable ({ext_lc or 'UNAVAILABLE'})."
                    ),
                )

        if eff_tool is not None and getattr(eff_tool, "is_external", False):
            if hasattr(eff_tool, "is_available") and not eff_tool.is_available():
                return ToolAuthorizationDecision(
                    allowed=False,
                    tool_name=canonical_name,
                    normalized_arguments={},
                    reason="tool_unavailable",
                    error_type="TOOL_UNAVAILABLE",
                    detail=f"External tool '{canonical_name}' registration is stale or source disconnected.",
                )

        if tool_registry is not None:
            is_registered_or_schema_known = bool(eff_tool is not None)
        else:
            from zelyr.core.plugins import _executors, _rings, _schemas, load_plugins

            if not _executors and not _schemas and not _rings:
                load_plugins()

            plugin_names = set(_executors.keys()) | {
                (s.get("function", {}).get("name") or s.get("name"))
                for s in _schemas
                if isinstance(s, dict)
            }
            is_registered_or_schema_known = bool(
                eff_tool is not None
                or canonical_name in plugin_names
                or canonical_name in BUILTIN_TOOLS
                or canonical_name in SWARM_TOOLS
                or canonical_name in schema_names
            )

        if require_registered and not is_registered_or_schema_known:
            return ToolAuthorizationDecision(
                allowed=False,
                tool_name=canonical_name,
                normalized_arguments=canonical_args,
                reason="tool_unavailable",
                error_type="TOOL_NOT_FOUND",
                detail=f"Tool '{canonical_name}' is not registered in the active runtime.",
            )

        # 2C. Verify that the resolved tool object (and any wrapped target) matches the authorized tool identity.
        identity_err = _verify_tool_object_identity(
            canonical_name,
            eff_tool,
            allowed_set=allowed_set if allowed_tools is not None else None,
        )
        if identity_err is not None:
            return ToolAuthorizationDecision(
                allowed=False,
                tool_name=canonical_name,
                normalized_arguments=canonical_args,
                reason="authorization_denied",
                error_type="AUTHORIZATION_DENIED",
                detail=identity_err,
            )

        # 3. Caller security-ring enforcement (fail-closed on invalid/unknown ring or lookup error).
        if caller_ring is not None:
            try:
                resolved_ring = resolve_tool_ring(
                    canonical_name,
                    tool=eff_tool,
                    tool_registry=reg,
                    active_schemas=active_schemas_list,
                )
            except Exception as exc:
                return ToolAuthorizationDecision(
                    allowed=False,
                    tool_name=canonical_name,
                    normalized_arguments=canonical_args,
                    reason="authorization_denied",
                    error_type="AUTHORIZATION_DENIED",
                    detail=f"AuthorizationError: Ring lookup failed for '{canonical_name}': {exc}",
                )
            if resolved_ring is None:
                return ToolAuthorizationDecision(
                    allowed=False,
                    tool_name=canonical_name,
                    normalized_arguments=canonical_args,
                    reason="authorization_denied",
                    error_type="AUTHORIZATION_DENIED",
                    detail=f"AuthorizationError: Unknown or invalid security ring for tool '{canonical_name}'.",
                )
            if caller_ring > resolved_ring:
                return ToolAuthorizationDecision(
                    allowed=False,
                    tool_name=canonical_name,
                    normalized_arguments=canonical_args,
                    reason="authorization_denied",
                    error_type="AUTHORIZATION_DENIED",
                    detail=(
                        f"AuthorizationError: Tool '{canonical_name}' requires higher permissions "
                        f"(requires ring {resolved_ring}, caller ring is {caller_ring})."
                    ),
                    required_ring=resolved_ring,
                )

        # 4. Schema-backed normalization and argument validation.
        tool_schema = _find_schema_for_tool(
            canonical_name,
            tool=eff_tool,
            tool_registry=reg if (tool_registry is not None or require_registered) else None,
            active_schemas=active_schemas_list,
        )
    if not args_is_dict:
        val_err: Optional[Dict[str, Any]]
        if normalize_args:
            _, _, val_err = SafeArgumentNormalizer.normalize_and_validate(
                canonical_name,
                tool_schema if validate_schema else None,
                arguments,
                argument_source=argument_source,
            )
        else:
            val_err = validate_arguments(canonical_name, tool_schema, arguments, argument_source=argument_source)
        return ToolAuthorizationDecision(
            allowed=False,
            tool_name=canonical_name,
            normalized_arguments={},
            reason="schema_validation_error",
            error_type=(val_err.get("error_type") if isinstance(val_err, dict) else None) or "SCHEMA_VALIDATION_ERROR",
            detail=val_err.get("error") if isinstance(val_err, dict) else "Invalid non-object tool arguments.",
            required_ring=resolved_ring,
            validation_error=val_err if isinstance(val_err, dict) else None,
        )

    import copy

    eff_args = copy.deepcopy(canonical_args)
    norm_warnings_list: List[str] = []
    if normalize_args:
        eff_args, norm_warnings, err_env = SafeArgumentNormalizer.normalize_and_validate(
            canonical_name,
            tool_schema if validate_schema else None,
            eff_args,
            argument_source=argument_source,
        )
        norm_warnings_list = norm_warnings
        if err_env is not None:
            is_sec = err_env.get("error_type") == "AUTHORIZATION_DENIED"
            return ToolAuthorizationDecision(
                allowed=False,
                tool_name=canonical_name,
                normalized_arguments={} if is_sec else copy.deepcopy(eff_args),
                reason="authorization_denied" if is_sec else "schema_validation_error",
                error_type=err_env.get("error_type", "SCHEMA_VALIDATION_ERROR"),
                detail=err_env.get("error", "Schema validation failed."),
                required_ring=resolved_ring,
                validation_error=err_env,
                warnings=tuple(norm_warnings_list),
            )
    elif validate_schema and tool_schema is not None:
        val_err = validate_arguments(
            canonical_name, tool_schema, eff_args, argument_source=argument_source
        )
        if val_err is not None:
            return ToolAuthorizationDecision(
                allowed=False,
                tool_name=canonical_name,
                normalized_arguments=copy.deepcopy(eff_args),
                reason="schema_validation_error",
                error_type=val_err.get("error_type", "SCHEMA_VALIDATION_ERROR"),
                detail=val_err.get("error", "Schema validation failed."),
                required_ring=resolved_ring,
                validation_error=val_err,
            )

    if validate_schema and task_local_scope is None and eff_tool is not None:
        sig_err = _validate_callable_signature(canonical_name, eff_tool, eff_args)
        if sig_err is not None:
            sig_env = _schema_error(
                canonical_name,
                "arguments",
                "object",
                "",
                sig_err,
                error_category="constraint_violation",
                actual_type="dict",
                value_category="object",
            )
            return ToolAuthorizationDecision(
                allowed=False,
                tool_name=canonical_name,
                normalized_arguments=copy.deepcopy(eff_args),
                reason="schema_validation_error",
                error_type="SCHEMA_VALIDATION_ERROR",
                detail=sig_env["error"],
                required_ring=resolved_ring,
                validation_error=sig_env,
                warnings=tuple(norm_warnings_list),
            )

    return ToolAuthorizationDecision(
        allowed=True,
        tool_name=canonical_name,
        normalized_arguments=copy.deepcopy(eff_args),
        required_ring=resolved_ring,
        warnings=tuple(norm_warnings_list),
    )

class BuiltinToolDef:
    """Lightweight metadata descriptor for a built-in tool."""

    def __init__(
        self,
        tool_id: str,
        description: str,
        module_path: str,
        class_name: str,
        constructor_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.tool_id = tool_id
        self.name = tool_id
        self.description = description
        self.module_path = module_path
        self.class_name = class_name
        self.constructor_kwargs = constructor_kwargs or {}
        self._instance: Optional[BaseTool] = None

    def instantiate(self) -> BaseTool:
        if self._instance is None:
            import importlib
            mod = importlib.import_module(self.module_path)
            cls = getattr(mod, self.class_name)
            self._instance = cls(**self.constructor_kwargs)
        return self._instance


BUILTIN_TOOLS: Dict[str, BuiltinToolDef] = {
    "app_launcher": BuiltinToolDef(
        tool_id="app_launcher",
        description="Launch common default apps (e.g. browser, calculator, terminal, file_explorer)",
        module_path="zelyr.tools.os_assist",
        class_name="AppLauncherTool",
    ),
    "clipboard_read": BuiltinToolDef(
        tool_id="clipboard_read",
        description="Read the current text content of the system clipboard.",
        module_path="zelyr.tools.clipboard_tools",
        class_name="ClipboardReadTool",
    ),
    "clipboard_write": BuiltinToolDef(
        tool_id="clipboard_write",
        description="Write text content to the system clipboard.",
        module_path="zelyr.tools.clipboard_tools",
        class_name="ClipboardWriteTool",
    ),
    "echo": BuiltinToolDef(
        tool_id="echo",
        description="Echoes the input back to verify tool calls are working.",
        module_path="zelyr.tools",
        class_name="EchoTool",
    ),
    "file_read": BuiltinToolDef(
        tool_id="file_read",
        description="Reads files and extracts text. Supports plain text (.txt, .py, .json, .md) and document extraction (.pdf).",
        module_path="zelyr.tools",
        class_name="FileReadTool",
        constructor_kwargs={"base_dir": "."},
    ),
    "file_search": BuiltinToolDef(
        tool_id="file_search",
        description="Find files safely in standard user directories (Desktop, Documents, etc.)",
        module_path="zelyr.tools.os_assist",
        class_name="SafeFileSearchTool",
    ),
    "file_write": BuiltinToolDef(
        tool_id="file_write",
        description="Write, append, or delete files safely.",
        module_path="zelyr.tools",
        class_name="FileWriteTool",
        constructor_kwargs={"base_dir": "."},
    ),
    "shell": BuiltinToolDef(
        tool_id="shell",
        description="Execute Linux shell commands. Do NOT run interactive commands.",
        module_path="zelyr.tools",
        class_name="ShellTool",
    ),
    "system_info": BuiltinToolDef(
        tool_id="system_info",
        description="Retrieve CPU, Memory, Disk, and OS information.",
        module_path="zelyr.tools",
        class_name="SystemInfoTool",
    ),
    "task_scheduler": BuiltinToolDef(
        tool_id="task_scheduler",
        description="Schedule a task to run at a specific future time. Supports natural language like 'in 1 hour', 'in 30 minutes', or a cron expression.",
        module_path="zelyr.tools.task_scheduler",
        class_name="TaskSchedulerTool",
    ),
    "verify_outcome": BuiltinToolDef(
        tool_id="verify_outcome",
        description="Verify system state outcomes deterministically (files, contents, checksums, json, processes, ports, commands).",
        module_path="zelyr.tools.verify_outcome",
        class_name="VerifyOutcomeTool",
    ),
    "manage_desktop_window": BuiltinToolDef(
        tool_id="manage_desktop_window",
        description="Tier 1 Hyprland System IPC: Deterministically manage desktop windows, workspaces, and focus via direct UNIX socket or hyprctl.",
        module_path="zelyr.tools.hyprland_ipc",
        class_name="ManageDesktopWindowTool",
    ),
    "interact_with_browser": BuiltinToolDef(
        tool_id="interact_with_browser",
        description="Tier 2 Semantic UI Automation: Direct Chrome DevTools Protocol (CDP) browser automation and accessibility tree extraction.",
        module_path="zelyr.tools.browser_cdp",
        class_name="InteractWithBrowserTool",
    ),
    "manage_memory": BuiltinToolDef(
        tool_id="manage_memory",
        description="Tier 1 Persistent Memory: Store, search, retrieve, and delete persistent facts, user preferences, and environment configurations.",
        module_path="zelyr.tools.manage_memory",
        class_name="ManageMemoryTool",
    ),
    "manage_routine": BuiltinToolDef(
        tool_id="manage_routine",
        description="Autonomous background routine scheduling, temporal task management, and execution lifecycle actuator.",
        module_path="zelyr.tools.manage_routine",
        class_name="ManageRoutineTool",
    ),
}

SWARM_TOOLS: Dict[str, BuiltinToolDef] = {
    "delegate_subtask": BuiltinToolDef(
        tool_id="delegate_subtask",
        description="Delegate a focused subtask to a specialized worker subagent (coder, tester, researcher, reviewer, general) in an isolated context bubble.",
        module_path="zelyr.tools.subagent_tool",
        class_name="SubagentDelegationTool",
    ),
    "craft_tool": BuiltinToolDef(
        tool_id="craft_tool",
        description="Synthesizes, AST validates, sandboxes, and hotloads a new dynamic Python tool.",
        module_path="zelyr.tools.tool_crafter",
        class_name="ToolCrafterTool",
    ),
}

class ToolRegistryError(Exception):
    """Raised when a tool registry operation is invalid."""

class ToolRegistry:
    """Thread-safe registry and invocation surface for Zelyr tools.

    When *core_registry* is provided, ``ToolRegistry`` delegates all
    storage to it and adds schema generation and safe execution on top.
    When omitted, an isolated ``core.registry.Registry`` is created so
    that existing ``ToolRegistry()`` call-sites continue to work.
    """

    def __init__(self, core_registry=None) -> None:
        """Auto-generated docstring.

Args:
    core_registry: Argument.

Returns:
    Return value.
"""
        from zelyr.core.registry import Registry as CoreRegistry
        self._core_registry = core_registry or CoreRegistry()
        self._lock = threading.RLock()
        self._external_manager: Optional[Any] = None

    @property
    def external_manager(self):
        """Return the lazily initialized ExternalToolManager integrated with this registry."""
        with self._lock:
            if self._external_manager is None:
                from zelyr.tools.external import ExternalToolManager

                self._external_manager = ExternalToolManager(self)
            return self._external_manager

    @property
    def external_policy(self):
        """Return the active ExternalToolPolicy governing external fallback tools."""
        return self.external_manager.policy

    def register_external_source(self, source):
        """Register an ExternalToolSource (without discovering or authorizing tools automatically)."""
        return self.external_manager.register_source(source)

    def remove_external_source(self, source_id: str) -> bool:
        """Remove an external source and invalidate all stale registrations."""
        return self.external_manager.remove_source(source_id)

    async def discover_external_tools(self, source_id: Optional[str] = None):
        """Discover and validate tools from external sources (defaulting newly discovered tools to unauthorized)."""
        return await self.external_manager.discover_all(source_id=source_id)

    async def refresh_external_tools(self, source_id: Optional[str] = None):
        """Refresh external tool discovery and lifecycle state safely."""
        return await self.external_manager.discover_all(source_id=source_id)

    def authorize_external_source(self, source_id: str) -> None:
        """Explicitly allow an external source in host policy (per-tool approval still required)."""
        self.external_manager.authorize_source(source_id)

    def revoke_external_source(self, source_id: str, *, block: bool = False) -> None:
        """Revoke an external source and unregister all its tools cleanly."""
        self.external_manager.revoke_source(source_id, block=block)

    def authorize_external_tool(self, tool_id: str, *, authorize_source: bool = True) -> bool:
        """Explicitly authorize an external tool for registration and execution."""
        return self.external_manager.authorize_tool(tool_id, authorize_source=authorize_source)

    def revoke_external_tool(self, tool_id: str, *, block: bool = False) -> None:
        """Revoke an external tool and remove it from active execution."""
        self.external_manager.revoke_tool(tool_id, block=block)

    def mark_external_source_status(self, source_id: str, health, *, error: Optional[str] = None) -> None:
        """Update external source health and invalidate stale tools if unhealthy."""
        self.external_manager.mark_source_status(source_id, health, error=error)

    def get_external_tool_entry(self, tool_id: str):
        """Return tracked external tool metadata entry, or None."""
        if self._external_manager is None:
            return None
        return self._external_manager.get_tool_entry(tool_id)

    def get_external_manifest_summary(
        self,
        *,
        caller_ring: Optional[int] = None,
        allowed_tools: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Return structured external tool manifest telemetry scoped to caller policy."""
        return self.external_manager.get_manifest_summary(
            caller_ring=caller_ring,
            allowed_tools=allowed_tools,
        )

    def register(self, tool: BaseTool) -> None:
        """Register ``tool`` using its own ``tool_id`` as the registry key."""
        if not isinstance(tool, BaseTool):
            raise ToolRegistryError(f'Component must be an instance of BaseTool, got {type(tool).__name__}')
        tool_id = tool.tool_id
        if not tool_id:
            raise ToolRegistryError('Tool must define a non-empty tool_id')
        self.register_tool(tool_id, tool)

    def register_tool(self, tool_id: str, tool: BaseTool) -> None:
        """Register ``tool`` under an explicit ``tool_id``.

        Mirrors :meth:`zelyr.core.registry.Registry.register_tool` so this
        registry can act as a drop-in replacement for existing tool-storage
        callers such as the CLI.
        """
        if not tool_id:
            raise ToolRegistryError('tool_id cannot be empty')
        if not isinstance(tool, BaseTool):
            raise ToolRegistryError(f'Component must be an instance of BaseTool, got {type(tool).__name__}')
        with self._lock:
            if self._core_registry.has_tool(tool_id):
                raise ToolRegistryError(f"Tool '{tool_id}' is already registered")
            self._core_registry.register_tool(tool_id, tool)

    def unregister_tool(self, tool_id: str) -> bool:
        """Remove a registered tool. Returns False if it was not registered."""
        with self._lock:
            if not self._core_registry.has_tool(tool_id):
                return False
            self._core_registry.unregister_tool(tool_id)
            return True

    def get_tool(self, tool_id: str) -> Optional[BaseTool]:
        """Return the tool registered under ``tool_id``, or ``None``."""
        tool = self._core_registry.get_tool(tool_id)
        if tool is None:
            resolved_id = _TOOL_REGISTRY_ALIASES.get(tool_id, tool_id)
            tool = self._core_registry.get_tool(resolved_id)
            if tool is None:
                if resolved_id in BUILTIN_TOOLS:
                    tool = self._load_builtin_tool(resolved_id)
                elif tool_id in BUILTIN_TOOLS:
                    tool = self._load_builtin_tool(tool_id)
                elif resolved_id in SWARM_TOOLS:
                    tool = self._load_swarm_tool(resolved_id)
                elif tool_id in SWARM_TOOLS:
                    tool = self._load_swarm_tool(tool_id)
        return tool

    def get_builtin_tools(self) -> Dict[str, BuiltinToolDef]:
        """Return metadata descriptors for all standard built-in tools."""
        return dict(BUILTIN_TOOLS)

    def _load_builtin_tool(self, tool_id: str) -> Optional[BaseTool]:
        defn = BUILTIN_TOOLS.get(tool_id)
        if defn is None:
            return None
        with self._lock:
            if not self._core_registry.has_tool(tool_id):
                instance = defn.instantiate()
                self._core_registry.register_tool(tool_id, instance)
                return instance
            return self._core_registry.get_tool(tool_id)

    def _load_swarm_tool(self, tool_id: str) -> Optional[BaseTool]:
        defn = SWARM_TOOLS.get(tool_id)
        if defn is None:
            return None
        with self._lock:
            if not self._core_registry.has_tool(tool_id):
                instance = defn.instantiate()
                if hasattr(instance, "set_registry"):
                    instance.set_registry(self)
                elif hasattr(instance, "tool_registry") and instance.tool_registry is None:
                    instance.tool_registry = self
                self._core_registry.register_tool(tool_id, instance)
                return instance
            return self._core_registry.get_tool(tool_id)

    def list_tools(self) -> Dict[str, BaseTool]:
        """Return a shallow copy of all registered tools keyed by ``tool_id``."""
        tools = self._core_registry.list_tools()
        import json
        from pathlib import Path
        config_path = config_dir() / "settings.json"
        
        disabled = []
        try:
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    disabled = data.get('disabled_plugins', [])
        except Exception:
            pass
            
        if disabled:
            return {k: v for k, v in tools.items() if k not in disabled}
        return tools

    def __contains__(self, tool_id: str) -> bool:
        """Auto-generated docstring.

Args:
    tool_id: Argument.

Returns:
    Return value.
"""
        return self._core_registry.has_tool(tool_id)

    def __len__(self) -> int:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return len(self._core_registry.list_tools())

    def get_schemas(self, compact: bool = False) -> List[Dict[str, Any]]:
        """Return OpenAI-compatible function-calling schemas for every tool."""
        if not self._core_registry.list_tools():
            for tid in BUILTIN_TOOLS:
                self.get_tool(tid)
        schemas: List[Dict[str, Any]] = []
        for tool_id, tool in self.list_tools().items():
            if getattr(tool, 'is_external', False) and hasattr(tool, 'is_available') and not tool.is_available():
                continue
            if not hasattr(tool, 'description') or not hasattr(tool, 'schema'):
                continue
            schemas.append({'type': 'function', 'function': {'name': tool_id, 'description': tool.description or tool_id, 'parameters': tool.schema or {'type': 'object', 'properties': {}}}})
        if compact:
            from zelyr.engine.schema_compact import CompactSchemaGenerator
            return CompactSchemaGenerator.compact_schemas(schemas)
        return schemas

    def get_tool_schemas(self, compact: bool = False) -> List[Dict[str, Any]]:
        """Return OpenAI-compatible function-calling schemas for all active tools."""
        return self.get_schemas(compact=compact)

    def _pre_validate_path(self, tool_id: str, arguments: Dict[str, Any]) -> Optional[ToolResult]:
        """Intercept slang or hallucinatory file paths before execution."""
        from zelyr.tools.core import ToolResult
        import os
        
        path_arg = None
        if 'file_path' in arguments:
            path_arg = arguments['file_path']
        elif 'path' in arguments:
            path_arg = arguments['path']
        elif 'params' in arguments and isinstance(arguments['params'], dict):
            path_arg = arguments['params'].get('file_path') or arguments['params'].get('path')
            
        if path_arg and isinstance(path_arg, str) and tool_id in ('file_opener', 'file_read', 'read_file'):
            try:
                from zelyr.tools import resolve_safe_path
                from pathlib import Path
                
                # Basic slang/placeholder trap and file existence check using smart resolver
                resolved_p = resolve_safe_path(path_arg, Path.cwd())
                if not resolved_p.exists():
                    import json
                    error_payload = {
                        "status": "FATAL_ERROR",
                        "error_type": "PRE_EXECUTION_GUARD",
                        "message": f"[System Guard]: Execution blocked. The path '{path_arg}' (resolved to '{resolved_p}') DOES NOT EXIST.",
                        "action_required": "Run 'file_search' or 'ls' first to get valid absolute paths. Stop guessing filenames."
                    }
                    return ToolResult(success=False, error=json.dumps(error_payload))
            except Exception:
                pass
        return None

    def _get_tool_ring(self, tool_id: str, tool: Optional[BaseTool] = None) -> Optional[int]:
        """Resolve the required security ring for a tool [0, 3]. Ring 0 is highest privilege.

        Returns None when explicit ring metadata is invalid or when no ring is defined
        for an unregistered tool identifier.
        """
        import sys

        ext_entry = self.get_external_tool_entry(tool_id)
        if ext_entry is not None:
            raw_ext_ring = getattr(ext_entry, "required_ring", None)
            return raw_ext_ring if _is_valid_ring_value(raw_ext_ring) else None

        if tool is not None:
            for attr_owner in (tool, type(tool)):
                for attr_name in ("required_ring", "REQUIRED_RING"):
                    if hasattr(attr_owner, attr_name):
                        raw_ring = getattr(attr_owner, attr_name)
                        if raw_ring is not None:
                            return raw_ring if _is_valid_ring_value(raw_ring) else None
            cls = type(tool)
            mod_name = cls.__module__
            if mod_name in sys.modules:
                mod = sys.modules[mod_name]
                if hasattr(mod, "REQUIRED_RING"):
                    raw_ring = getattr(mod, "REQUIRED_RING")
                    if raw_ring is not None:
                        return raw_ring if _is_valid_ring_value(raw_ring) else None

        from zelyr.core.plugins import _rings

        if tool_id in _rings:
            raw_ring = _rings[tool_id]
            return raw_ring if _is_valid_ring_value(raw_ring) else None

        canonical_alias = _TOOL_REGISTRY_ALIASES.get(tool_id, tool_id)
        tool_attr_id = getattr(tool, "tool_id", None) if tool is not None else None
        if tool_id in BUILTIN_TOOLS or canonical_alias in BUILTIN_TOOLS or tool_attr_id in BUILTIN_TOOLS:
            return 0
        if tool_id in SWARM_TOOLS or tool_attr_id in SWARM_TOOLS:
            return 1
        if tool is not None:
            return 1
        return None

    def _authorize_execution(
        self,
        tool_id: str,
        tool: BaseTool,
        caller_ring: Optional[int] = None,
        allowed_tools: Optional[Any] = None,
    ) -> Optional[ToolResult]:
        """Verify that caller has permission to execute the specified tool."""
        decision = authorize_and_validate_tool_call(
            tool_id,
            {},
            caller_ring=caller_ring,
            allowed_tools=allowed_tools,
            tool_registry=self,
            tool=tool,
            normalize_args=False,
            validate_schema=False,
        )
        if not decision.allowed:
            return ToolResult(
                success=False,
                output="",
                error=decision.detail or "AuthorizationError: Tool execution denied.",
                error_type=decision.error_type or "AUTHORIZATION_DENIED",
            )
        return None

    def _normalize_and_validate(
        self,
        tool_id: str,
        tool: BaseTool,
        arguments: Dict[str, Any],
        *,
        argument_source: str = "internal",
    ) -> tuple[Optional[Dict[str, Any]], Optional[ToolResult]]:
        """Normalize unambiguous types and validate arguments via one shared authoritative path."""
        from zelyr.engine.normalizer import SafeArgumentNormalizer, _schema_error

        tool_schema = getattr(tool, "schema", None) or getattr(tool, "tool_schema", None)
        eff_arg_source = "model" if getattr(tool, "is_external", False) else argument_source
        normalized_args, _, val_envelope = SafeArgumentNormalizer.normalize_and_validate(
            tool_id,
            tool_schema,
            arguments,
            argument_source=eff_arg_source,
        )
        if val_envelope is not None:
            return None, ToolResult(
                success=False,
                error=val_envelope["error"],
                output=val_envelope,
                error_type=val_envelope.get("error_type", "SCHEMA_VALIDATION_ERROR"),
                remedy_hint=val_envelope.get("remedy_hint"),
                allowed_actions=[tool_id],
            )
        sig_err = _validate_callable_signature(tool_id, tool, normalized_args)
        if sig_err is not None:
            sig_env = _schema_error(
                tool_id,
                "arguments",
                "object",
                "",
                sig_err,
                error_category="constraint_violation",
                actual_type="dict",
                value_category="object",
            )
            return None, ToolResult(
                success=False,
                error=sig_env["error"],
                output=sig_env,
                error_type="SCHEMA_VALIDATION_ERROR",
                remedy_hint=sig_env.get("remedy_hint"),
                allowed_actions=[tool_id],
            )
        return normalized_args, None

    def execute(
        self,
        tool_id: str,
        caller_ring: Optional[int] = None,
        allowed_tools: Optional[Any] = None,
        **arguments: Any,
    ) -> ToolResult:  # type: ignore[override]
        """Execute a registered tool safely, regardless of its calling convention."""
        if caller_ring is None and "caller_ring" in arguments:
            caller_ring = arguments.pop("caller_ring")
        if allowed_tools is None and "allowed_tools" in arguments:
            allowed_tools = arguments.pop("allowed_tools")

        tool_id, arguments = canonicalize_tool_invocation(tool_id, arguments)
        tool = self.get_tool(tool_id)
        if tool is None:
            ext_entry = self.get_external_tool_entry(tool_id)
            if ext_entry is not None:
                if not getattr(ext_entry, "authorized", False):
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"AuthorizationError: External tool '{tool_id}' from source '{ext_entry.source_id}' is not authorized.",
                        error_type="AUTHORIZATION_DENIED",
                    )
                return ToolResult(
                    success=False,
                    output="",
                    error=f"External tool '{tool_id}' from source '{ext_entry.source_id}' is unavailable ({ext_entry.lifecycle.value}).",
                    error_type="TOOL_UNAVAILABLE",
                )
            return ToolResult(success=False, error=f"Tool not found: {tool_id}", error_type="TOOL_NOT_FOUND")

        auth_error = self._authorize_execution(tool_id, tool, caller_ring, allowed_tools)
        if auth_error:
            return auth_error

        if getattr(tool, 'requires_approval', False):
            try:
                from zelyr.services.governor import GovernorService
                governor = GovernorService.instance()
                if governor and governor.is_strict_mode():
                    return ToolResult(success=False, error="This tool requires approval, but was invoked synchronously. Governor gates are only supported in async mode.")
            except Exception:
                pass
        val_error = self._pre_validate_path(tool.tool_id, arguments)
        if val_error:
            return val_error

        normalized_args, schema_error = self._normalize_and_validate(tool_id, tool, arguments)
        if schema_error is not None or normalized_args is None:
            return schema_error  # type: ignore[return-value]
        arguments = normalized_args

        try:
            res = tool(**arguments)
            if isinstance(res, ToolResult):
                return res
            return ToolResult(success=True, output=res)
        except Exception as exc:
            import traceback
            import logging
            tb = traceback.format_exc()
            if "zelyr/dynamic_plugins" in tb or ".config/zelyr/plugins" in tb or ".config/axiom/plugins" in tb:
                logging.getLogger("zelyr.plugin_crash").error(tb)
            from zelyr.utils.optional_deps import OptionalDependencyError
            if isinstance(exc, (ImportError, OptionalDependencyError)):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Tool requires optional dependency: {exc}",
                    error_type="TOOL_UNAVAILABLE",
                )
            return ToolResult(success=False, error=str(exc))

    async def execute_async(
        self,
        tool_id: str,
        caller_ring: Optional[int] = None,
        allowed_tools: Optional[Any] = None,
        **arguments: Any,
    ) -> ToolResult:
        """Async version of execute.

        Directly invokes the tool's execute method and awaits it if it is
        asynchronous, bypassing the sync-bridging logic in __call__.
        """
        import asyncio
        import inspect
        from zelyr.tools.core import ToolResult

        if caller_ring is None and "caller_ring" in arguments:
            caller_ring = arguments.pop("caller_ring")
        if allowed_tools is None and "allowed_tools" in arguments:
            allowed_tools = arguments.pop("allowed_tools")

        tool_id, arguments = canonicalize_tool_invocation(tool_id, arguments)
        tool = self.get_tool(tool_id)
        if tool is None:
            ext_entry = self.get_external_tool_entry(tool_id)
            if ext_entry is not None:
                if not getattr(ext_entry, "authorized", False):
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"AuthorizationError: External tool '{tool_id}' from source '{ext_entry.source_id}' is not authorized.",
                        error_type="AUTHORIZATION_DENIED",
                    )
                return ToolResult(
                    success=False,
                    output="",
                    error=f"External tool '{tool_id}' from source '{ext_entry.source_id}' is unavailable ({ext_entry.lifecycle.value}).",
                    error_type="TOOL_UNAVAILABLE",
                )
            return ToolResult(success=False, error=f"Tool not found: {tool_id}", error_type="TOOL_NOT_FOUND")

        auth_error = self._authorize_execution(tool_id, tool, caller_ring, allowed_tools)
        if auth_error:
            return auth_error

        if getattr(tool, 'requires_approval', False):
            try:
                from zelyr.services.governor import GovernorService
                governor = GovernorService.instance()
                if governor and governor.is_strict_mode():
                    approved = await governor.request_approval(tool.name, arguments)
                    if not approved:
                        return ToolResult(
                            success=False,
                            error="User denied execution via The Governor.",
                            error_type="AUTHORIZATION_DENIED",
                        )
            except Exception as e:
                return ToolResult(success=False, error=f"Governor failed: {e}")
        val_error = self._pre_validate_path(tool.tool_id, arguments)
        if val_error:
            return val_error

        normalized_args, schema_error = self._normalize_and_validate(tool_id, tool, arguments)
        if schema_error is not None or normalized_args is None:
            return schema_error  # type: ignore[return-value]
        arguments = normalized_args

        try:
            sig = inspect.signature(tool.execute)
            params_list = list(sig.parameters.values())
            single_dict_param = len(params_list) == 1 and (params_list[0].annotation == Dict[str, Any] or params_list[0].name in ('params', 'arguments', 'kwargs'))
            if single_dict_param:
                result = tool.execute(arguments)
            else:
                result = tool.execute(**arguments)
            if asyncio.iscoroutine(result):
                return await result
            return result
        except Exception as exc:
            import traceback
            import logging
            tb = traceback.format_exc()
            if "zelyr/dynamic_plugins" in tb or ".config/zelyr/plugins" in tb or ".config/axiom/plugins" in tb:
                logging.getLogger("zelyr.plugin_crash").error(tb)
            from zelyr.utils.optional_deps import OptionalDependencyError
            if isinstance(exc, (ImportError, OptionalDependencyError)):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Tool requires optional dependency: {exc}",
                    error_type="TOOL_UNAVAILABLE",
                )
            return ToolResult(success=False, error=str(exc))
