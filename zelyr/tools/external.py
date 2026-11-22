"""External / Fallback Tool Discovery, Adaptation, Policy & Lifecycle Engine for Zelyr 1.4.

Provides a unified, fail-closed integration pipeline for external tools (starting
with MCP sources) inside Zelyr's existing ToolRegistry without creating a separate
parallel registry, without auto-installing packages or executing untrusted code,
and without bypassing runtime authorization or argument normalization.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
from enum import Enum
import logging
import re
import threading
import time
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Set, Tuple
import urllib.parse

from zelyr.engine.normalizer import (
    RESERVED_AUTHORIZATION_KEYS,
    SafeArgumentNormalizer,
    _redact_sensitive_text,
)
from zelyr.tools.core import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger("zelyr.tools.external")

DEFAULT_EXTERNAL_REQUIRED_RING: int = 1
MAX_SCHEMA_DEPTH: int = 10
MAX_EXTERNAL_DESCRIPTION_LEN: int = 240
MAX_ENDPOINT_LEN: int = 160

_SAFE_EXTERNAL_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SAFE_SLUG_STRIP_RE = re.compile(r"[^a-zA-Z0-9_]+")
_CTRL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]+")

_SUPPORTED_JSON_SCHEMA_TYPES: frozenset[str] = frozenset(
    {"string", "integer", "number", "boolean", "object", "array", "null"}
)
_UNSUPPORTED_SCHEMA_KEYWORDS: frozenset[str] = frozenset(
    {
        "$ref",
        "$dynamicRef",
        "$recursiveRef",
        "not",
        "if",
        "then",
        "else",
        "dependentSchemas",
        "patternProperties",
        "unevaluatedProperties",
        "unevaluatedItems",
    }
)
_PROMPT_CONTROL_MARKERS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?is)<\s*/?\s*(?:system|instruction|assistant|developer|tool_call|function_call)\s*>"),
    re.compile(r"<\|[^|]{1,64}\|>"),
    re.compile(r"(?i)\[\s*(?:SYSTEM|TOOL\s+OBSERVATION|OVERRIDE|ADMIN)[^\]]{0,80}\]"),
    re.compile(
        r"(?i)\b(?:ignore|disregard|forget)\s+(?:all\s+|any\s+)?(?:previous|prior|system)\s+(?:instructions?|rules?|prompts?)\b"
    ),
    re.compile(r"(?i)\b(?:override|bypass)\s+(?:the\s+)?(?:system\s+prompt|security\s+ring|authorization)\b"),
)


class ExternalToolLifecycle(str, Enum):
    """Canonical lifecycle states for discovered external tools."""

    DISCOVERED = "DISCOVERED"
    VALIDATED = "VALIDATED"
    AUTHORIZED = "AUTHORIZED"
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    DEGRADED = "DEGRADED"


class ExternalSourceHealth(str, Enum):
    """Health/connectivity states for an external tool source."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"
    DISCONNECTED = "DISCONNECTED"


def sanitize_source_endpoint(endpoint: Any) -> str:
    """Redact userinfo, query tokens, fragments, and secrets from a source endpoint."""
    raw = str(endpoint or "").strip()
    if not raw:
        return "unknown"
    raw = _CTRL_CHARS_RE.sub(" ", raw).strip()

    if "://" in raw and not raw.startswith(("unix://", "stdio://")):
        try:
            parsed = urllib.parse.urlparse(raw)
            host = parsed.hostname or ""
            if host:
                if ":" in host and not host.startswith("["):
                    host = f"[{host}]"
                port_part = f":{parsed.port}" if parsed.port else ""
                path = parsed.path or ""
                rebuilt = f"{parsed.scheme}://{host}{port_part}{path}"
                scrubbed = _redact_sensitive_text(rebuilt)
                return scrubbed[:MAX_ENDPOINT_LEN]
        except Exception:
            pass

    scrubbed = _redact_sensitive_text(raw.split("?", 1)[0].split("#", 1)[0])
    if len(scrubbed) > MAX_ENDPOINT_LEN:
        scrubbed = scrubbed[:MAX_ENDPOINT_LEN] + "...[truncated]"
    return scrubbed or "unknown"


def sanitize_external_text(raw: Any, *, max_len: int = MAX_EXTERNAL_DESCRIPTION_LEN) -> str:
    """Strip control markers, prompt-override directives, and secrets from untrusted text."""
    text = _CTRL_CHARS_RE.sub(" ", str(raw or "")).strip()
    text = re.sub(r"\s+", " ", text)
    for pattern in _PROMPT_CONTROL_MARKERS:
        text = pattern.sub("[filtered]", text)
    text = _redact_sensitive_text(text).strip()
    if len(text) > max_len:
        text = text[: max_len - 3].rstrip() + "..."
    return text


def sanitize_external_description(source_id: str, description: Any, *, max_len: int = MAX_EXTERNAL_DESCRIPTION_LEN) -> str:
    """Format an untrusted external tool description with unmistakable source attribution."""
    safe_src = sanitize_external_text(source_id, max_len=48) or "external"
    body = sanitize_external_text(description, max_len=max_len)
    prefix = f"[External:{safe_src}]"
    if not body:
        return f"{prefix} External tool provided by source '{safe_src}'."
    if body.startswith(prefix):
        return body[:max_len]
    combined = f"{prefix} {body}"
    if len(combined) > max_len:
        combined = combined[: max_len - 3].rstrip() + "..."
    return combined


def _slugify_component(raw: str, *, fallback: str = "item") -> str:
    cleaned = _SAFE_SLUG_STRIP_RE.sub("_", str(raw or "").strip().lower()).strip("_")
    return cleaned or fallback


def build_external_tool_id(source_id: str, external_tool_name: str, *, prefix: str = "ext") -> str:
    """Derive a deterministic namespaced Zelyr tool_id distinct from native tools."""
    src_slug = _slugify_component(source_id, fallback="source")
    tool_slug = _slugify_component(external_tool_name, fallback="tool")
    clean_prefix = _slugify_component(prefix, fallback="ext")
    return f"{clean_prefix}__{src_slug}__{tool_slug}"


@dataclass(frozen=True)
class SchemaCompatibilityResult:
    """Outcome returned by external JSON schema validation."""

    valid: bool
    normalized_schema: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


def _schema_compat_error(
    tool_id: str,
    source_id: str,
    reason: str,
    detail: str,
    *,
    error_type: str = "UNSUPPORTED_TOOL_SCHEMA",
) -> Dict[str, Any]:
    return {
        "tool": tool_id,
        "tool_id": tool_id,
        "source_id": source_id,
        "status": "error",
        "error_type": error_type,
        "reason": reason,
        "error": _redact_sensitive_text(detail),
    }


def _sanitize_and_validate_schema_node(
    node: Any,
    tool_id: str,
    source_id: str,
    *,
    depth: int,
    seen: Set[int],
    path: str = "inputSchema",
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if depth > MAX_SCHEMA_DEPTH:
        return None, _schema_compat_error(
            tool_id,
            source_id,
            "schema_depth_exceeded",
            f"External schema for '{tool_id}' exceeds max nesting depth ({MAX_SCHEMA_DEPTH}) at '{path}'.",
        )
    if not isinstance(node, dict):
        return None, _schema_compat_error(
            tool_id,
            source_id,
            "invalid_schema_node",
            f"Schema node '{path}' for '{tool_id}' must be a JSON object.",
        )

    obj_id = id(node)
    if obj_id in seen:
        return None, _schema_compat_error(
            tool_id,
            source_id,
            "cyclic_schema",
            f"Cyclic schema reference detected in '{tool_id}' at '{path}'.",
        )
    seen.add(obj_id)
    try:
        for bad_kw in _UNSUPPORTED_SCHEMA_KEYWORDS:
            if bad_kw in node:
                return None, _schema_compat_error(
                    tool_id,
                    source_id,
                    "unsupported_schema_keyword",
                    f"Unsupported JSON Schema keyword '{bad_kw}' at '{path}' for tool '{tool_id}'.",
                )

        raw_type = node.get("type")
        if raw_type is not None:
            if isinstance(raw_type, str):
                if raw_type not in _SUPPORTED_JSON_SCHEMA_TYPES:
                    return None, _schema_compat_error(
                        tool_id,
                        source_id,
                        "unsupported_schema_type",
                        f"Unsupported JSON Schema type {raw_type!r} at '{path}'.",
                    )
            elif isinstance(raw_type, list) and raw_type:
                if not all(isinstance(t, str) and t in _SUPPORTED_JSON_SCHEMA_TYPES for t in raw_type):
                    return None, _schema_compat_error(
                        tool_id,
                        source_id,
                        "unsupported_schema_type",
                        f"Unsupported union schema type {raw_type!r} at '{path}'.",
                    )
            else:
                return None, _schema_compat_error(
                    tool_id,
                    source_id,
                    "invalid_schema_type",
                    f"Invalid JSON Schema type declaration at '{path}'.",
                )

        out: Dict[str, Any] = {}
        for k, v in node.items():
            if not isinstance(k, str):
                return None, _schema_compat_error(
                    tool_id, source_id, "invalid_schema_key", f"Non-string schema key at '{path}'."
                )
            if k == "description" and isinstance(v, str):
                out[k] = sanitize_external_text(v, max_len=200)
            elif k == "properties":
                if not isinstance(v, dict):
                    return None, _schema_compat_error(
                        tool_id, source_id, "invalid_properties", f"'properties' at '{path}' must be an object."
                    )
                clean_props: Dict[str, Any] = {}
                for prop_name, prop_def in v.items():
                    if not isinstance(prop_name, str) or not prop_name.strip():
                        return None, _schema_compat_error(
                            tool_id, source_id, "invalid_property_name", f"Empty or non-string property name under '{path}'."
                        )
                    if prop_name in RESERVED_AUTHORIZATION_KEYS:
                        return None, _schema_compat_error(
                            tool_id,
                            source_id,
                            "reserved_authorization_key_in_schema",
                            f"External tool schema '{tool_id}' attempts to declare reserved authorization parameter '{prop_name}'.",
                            error_type="AUTHORIZATION_DENIED",
                        )
                    clean_prop, err = _sanitize_and_validate_schema_node(
                        prop_def,
                        tool_id,
                        source_id,
                        depth=depth + 1,
                        seen=seen,
                        path=f"{path}.properties.{prop_name}",
                    )
                    if err is not None:
                        return None, err
                    clean_props[prop_name] = clean_prop
                out["properties"] = clean_props
            elif k == "required":
                if not isinstance(v, list) or not all(isinstance(r, str) and r.strip() for r in v):
                    return None, _schema_compat_error(
                        tool_id, source_id, "invalid_required_list", f"'required' at '{path}' must be a list of non-empty strings."
                    )
                for req_key in v:
                    if req_key in RESERVED_AUTHORIZATION_KEYS:
                        return None, _schema_compat_error(
                            tool_id,
                            source_id,
                            "reserved_authorization_key_in_schema",
                            f"External tool schema '{tool_id}' attempts to require reserved authorization parameter '{req_key}'.",
                            error_type="AUTHORIZATION_DENIED",
                        )
                out["required"] = list(v)
            elif k == "additionalProperties":
                if isinstance(v, bool):
                    out["additionalProperties"] = v
                elif isinstance(v, dict):
                    clean_add, err = _sanitize_and_validate_schema_node(
                        v,
                        tool_id,
                        source_id,
                        depth=depth + 1,
                        seen=seen,
                        path=f"{path}.additionalProperties",
                    )
                    if err is not None:
                        return None, err
                    out["additionalProperties"] = clean_add
                else:
                    return None, _schema_compat_error(
                        tool_id,
                        source_id,
                        "invalid_additional_properties",
                        f"'additionalProperties' at '{path}' must be a boolean or object schema.",
                    )
            elif k == "items":
                if not isinstance(v, dict):
                    return None, _schema_compat_error(
                        tool_id,
                        source_id,
                        "unsupported_items_schema",
                        f"'items' at '{path}' must be a single object schema (tuple-array items lists are unsupported).",
                    )
                clean_items, err = _sanitize_and_validate_schema_node(
                    v,
                    tool_id,
                    source_id,
                    depth=depth + 1,
                    seen=seen,
                    path=f"{path}.items",
                )
                if err is not None:
                    return None, err
                out["items"] = clean_items
            elif k in ("anyOf", "oneOf", "allOf"):
                if not isinstance(v, list) or not v:
                    return None, _schema_compat_error(
                        tool_id, source_id, "invalid_combinator", f"'{k}' at '{path}' must be a non-empty list."
                    )
                clean_branches: List[Dict[str, Any]] = []
                for idx, branch in enumerate(v):
                    clean_branch, err = _sanitize_and_validate_schema_node(
                        branch,
                        tool_id,
                        source_id,
                        depth=depth + 1,
                        seen=seen,
                        path=f"{path}.{k}[{idx}]",
                    )
                    if err is not None:
                        return None, err
                    assert clean_branch is not None
                    clean_branches.append(clean_branch)
                out[k] = clean_branches
            elif k == "pattern" and isinstance(v, str):
                try:
                    re.compile(v)
                except re.error as exc:
                    return None, _schema_compat_error(
                        tool_id,
                        source_id,
                        "invalid_regex_pattern",
                        f"Invalid regex pattern at '{path}': {exc}",
                    )
                out[k] = v
            else:
                out[k] = copy.deepcopy(v)

        return out, None
    finally:
        seen.discard(obj_id)


def validate_external_tool_schema(
    tool_id: str,
    external_tool_name: str,
    source_id: str,
    raw_schema: Any,
) -> SchemaCompatibilityResult:
    """Validate and adapt an external tool's JSON Schema without mutating or losing constraints."""
    if not isinstance(external_tool_name, str) or not _SAFE_EXTERNAL_TOOL_NAME_RE.fullmatch(external_tool_name.strip()):
        return SchemaCompatibilityResult(
            valid=False,
            error=_schema_compat_error(
                tool_id,
                source_id,
                "invalid_external_tool_name",
                f"External tool name {external_tool_name!r} from source '{source_id}' is invalid.",
            ),
        )

    if not isinstance(raw_schema, dict):
        return SchemaCompatibilityResult(
            valid=False,
            error=_schema_compat_error(
                tool_id,
                source_id,
                "missing_or_invalid_input_schema",
                f"External tool '{tool_id}' inputSchema must be a JSON object dictionary.",
            ),
        )

    top_type = raw_schema.get("type")
    if top_type is not None and top_type != "object":
        return SchemaCompatibilityResult(
            valid=False,
            error=_schema_compat_error(
                tool_id,
                source_id,
                "unsupported_root_schema_type",
                f"External tool '{tool_id}' root inputSchema type must be 'object', received {top_type!r}.",
            ),
        )
    if top_type is None and not isinstance(raw_schema.get("properties"), dict):
        return SchemaCompatibilityResult(
            valid=False,
            error=_schema_compat_error(
                tool_id,
                source_id,
                "missing_root_object_type",
                f"External tool '{tool_id}' inputSchema must declare type='object'.",
            ),
        )

    seen: Set[int] = set()
    clean_schema, err = _sanitize_and_validate_schema_node(
        raw_schema, tool_id, source_id, depth=0, seen=seen
    )
    if err is not None or clean_schema is None:
        return SchemaCompatibilityResult(valid=False, error=err)

    clean_schema.setdefault("type", "object")
    clean_schema.setdefault("properties", {})
    return SchemaCompatibilityResult(valid=True, normalized_schema=clean_schema)


@dataclass(frozen=True)
class DiscoveredExternalTool:
    """Untrusted discovery metadata for a single external tool."""

    source_id: str
    source_type: str
    source_endpoint: str
    external_tool_name: str
    tool_id: str
    name: str
    description: str
    input_schema: Any
    availability_status: ExternalToolLifecycle
    required_permissions: tuple[str, ...]
    required_ring: int
    discovered_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExternalSourceRecord:
    """Tracked runtime source metadata and health/epoch state."""

    source: ExternalToolSource
    source_id: str
    source_type: str
    endpoint_identity: str
    health: ExternalSourceHealth = ExternalSourceHealth.HEALTHY
    epoch: int = 1
    last_discovered_at: Optional[float] = None
    last_error: Optional[str] = None
    validation_failures: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ExternalToolEntry:
    """Unified lifecycle, authorization, and adaptation state for one external tool."""

    tool_id: str
    source_id: str
    source_type: str
    source_endpoint: str
    external_tool_name: str
    description: str
    discovered_tool: DiscoveredExternalTool
    lifecycle: ExternalToolLifecycle = ExternalToolLifecycle.DISCOVERED
    authorized: bool = False
    schema_valid: bool = False
    required_ring: int = DEFAULT_EXTERNAL_REQUIRED_RING
    required_permissions: tuple[str, ...] = ()
    discovered_at: float = 0.0
    source_epoch: int = 1
    adapted_tool: Optional[ExternalAdaptedTool] = None
    validation_error: Optional[Dict[str, Any]] = None
    last_error: Optional[str] = None


class ExternalToolSource:
    """Abstract discovery & invocation interface for supported external tool sources."""

    source_id: str
    source_type: str

    def __init__(self, source_id: str, *, source_type: str = "external", endpoint_identity: str = "") -> None:
        clean_id = str(source_id or "").strip()
        if not clean_id:
            raise ValueError("ExternalToolSource requires a non-empty source_id.")
        self.source_id = clean_id
        self.source_type = str(source_type or "external").strip()
        self._endpoint_identity = sanitize_source_endpoint(endpoint_identity or clean_id)

    @property
    def endpoint_identity(self) -> str:
        return self._endpoint_identity

    def is_available(self) -> bool:
        return False

    async def check_health(self) -> ExternalSourceHealth:
        return ExternalSourceHealth.HEALTHY if self.is_available() else ExternalSourceHealth.UNAVAILABLE

    async def discover_tools(self) -> List[DiscoveredExternalTool]:
        raise NotImplementedError

    async def invoke_tool(
        self,
        external_tool_name: str,
        arguments: Dict[str, Any],
        *,
        timeout: float = 30.0,
    ) -> ToolResult:
        raise NotImplementedError


class MCPExternalToolSource(ExternalToolSource):
    """Discovery & invocation adapter wrapping existing Zelyr/official MCP clients.

    Works with:
    - ``zelyr.mcp.sse_client.MCPSSEClient``
    - ``zelyr.mcp.stdio_client.MCPStdioClientAsync``
    - ``zelyr.mcp.client.MCPStdioClient``
    - Official ``mcp.client.session.ClientSession`` or duck-typed MCP clients
    """

    def __init__(
        self,
        source_id: str,
        client: Any,
        *,
        server_name: Optional[str] = None,
        endpoint_identity: Optional[str] = None,
        tool_prefix: str = "ext",
        required_permissions: Tuple[str, ...] = ("mcp.remote_tool",),
    ) -> None:
        self.client = client
        self.server_name = server_name or getattr(client, "name", None) or source_id
        self.tool_prefix = tool_prefix
        self._required_permissions = tuple(required_permissions)
        raw_ep = endpoint_identity or getattr(client, "url", None) or getattr(client, "command", None) or self.server_name
        src_type = (
            "mcp_sse"
            if getattr(client, "url", None)
            else ("mcp_stdio" if getattr(client, "command", None) else "mcp")
        )
        super().__init__(source_id=source_id, source_type=src_type, endpoint_identity=raw_ep)

    def is_available(self) -> bool:
        c = self.client
        if c is None:
            return False
        if hasattr(c, "is_available") and callable(c.is_available):
            try:
                return bool(c.is_available())
            except Exception:
                return False
        if hasattr(c, "_closed") and getattr(c, "_closed", False):
            return False
        if hasattr(c, "_connected"):
            if not getattr(c, "_connected", False):
                return False
            if hasattr(c, "_running") and not getattr(c, "_running", False):
                return False
            return True
        if hasattr(c, "_running"):
            return bool(getattr(c, "_running", False))
        # Official ClientSession / lightweight connected client with list_tools/call_tool
        return callable(getattr(c, "list_tools", None)) and callable(getattr(c, "call_tool", None))

    @staticmethod
    def _coerce_tool_item(item: Any) -> Optional[Dict[str, Any]]:
        if isinstance(item, dict):
            return dict(item)
        # Support mcp.types.Tool or objects with .name / .inputSchema
        name = getattr(item, "name", None)
        if not name:
            return None
        return {
            "name": name,
            "description": getattr(item, "description", "") or "",
            "inputSchema": copy.deepcopy(getattr(item, "inputSchema", None)),
        }

    async def discover_tools(self) -> List[DiscoveredExternalTool]:
        if not self.is_available():
            raise ConnectionError(f"MCP source '{self.source_id}' is disconnected or unavailable.")

        list_fn = getattr(self.client, "list_tools", None)
        if not callable(list_fn):
            raise RuntimeError(f"MCP source '{self.source_id}' client does not implement list_tools().")

        raw_result = list_fn()
        if asyncio.iscoroutine(raw_result) or isinstance(raw_result, asyncio.Future):
            raw_result = await raw_result

        if hasattr(raw_result, "tools"):
            raw_items = getattr(raw_result, "tools", [])
        elif isinstance(raw_result, dict) and "tools" in raw_result:
            raw_items = raw_result.get("tools", [])
        elif isinstance(raw_result, list):
            raw_items = raw_result
        else:
            raw_items = []

        now = time.time()
        discovered: List[DiscoveredExternalTool] = []
        for raw_item in raw_items:
            item = self._coerce_tool_item(raw_item)
            if item is None:
                continue
            ext_name = str(item.get("name") or "").strip()
            tool_id = build_external_tool_id(self.source_id, ext_name, prefix=self.tool_prefix)
            safe_desc = sanitize_external_description(self.source_id, item.get("description", ""))
            raw_schema = item.get("inputSchema")
            # Intentionally ignore any authorization/ring metadata supplied by untrusted MCP server payloads.
            discovered.append(
                DiscoveredExternalTool(
                    source_id=self.source_id,
                    source_type=self.source_type,
                    source_endpoint=self.endpoint_identity,
                    external_tool_name=ext_name,
                    tool_id=tool_id,
                    name=tool_id,
                    description=safe_desc,
                    input_schema=copy.deepcopy(raw_schema),
                    availability_status=ExternalToolLifecycle.DISCOVERED,
                    required_permissions=self._required_permissions,
                    required_ring=DEFAULT_EXTERNAL_REQUIRED_RING,
                    discovered_at=now,
                    metadata={
                        "external": True,
                        "source_id": self.source_id,
                        "source_type": self.source_type,
                        "source_endpoint": self.endpoint_identity,
                        "external_tool_name": ext_name,
                    },
                )
            )
        return discovered

    async def invoke_tool(
        self,
        external_tool_name: str,
        arguments: Dict[str, Any],
        *,
        timeout: float = 30.0,
    ) -> ToolResult:
        base_meta: Dict[str, Any] = {
            "external": True,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "source_endpoint": self.endpoint_identity,
            "external_tool_name": external_tool_name,
        }
        if not self.is_available():
            return ToolResult(
                success=False,
                output="",
                error=f"External source '{self.source_id}' is disconnected or unavailable.",
                error_type="TOOL_UNAVAILABLE",
                metadata=base_meta,
            )

        call_fn = getattr(self.client, "call_tool", None)
        if not callable(call_fn):
            return ToolResult(
                success=False,
                output="",
                error=f"MCP source '{self.source_id}' client does not support call_tool().",
                error_type="TOOL_UNAVAILABLE",
                metadata=base_meta,
            )

        try:
            if asyncio.iscoroutinefunction(call_fn):
                try:
                    raw = await asyncio.wait_for(
                        call_fn(external_tool_name, arguments, timeout=timeout),
                        timeout=timeout,
                    )
                except TypeError:
                    raw = await asyncio.wait_for(call_fn(external_tool_name, arguments), timeout=timeout)
            else:
                raw = await asyncio.wait_for(
                    asyncio.to_thread(call_fn, external_tool_name, arguments),
                    timeout=timeout,
                )
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                output="",
                error=f"External MCP tool '{external_tool_name}' on source '{self.source_id}' timed out after {timeout:.1f}s.",
                error_type="TIMEOUT",
                remedy_hint="Check external source responsiveness or choose an alternative tool.",
                metadata=base_meta,
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                output="",
                error=_redact_sensitive_text(f"External source invocation failed: {exc}"),
                error_type="EXTERNAL_TOOL_ERROR",
                metadata=base_meta,
            )

        if isinstance(raw, ToolResult):
            merged_meta = dict(raw.metadata or {})
            merged_meta.update(base_meta)
            return ToolResult(
                success=raw.success,
                output=raw.output,
                error=_redact_sensitive_text(raw.error) if raw.error else None,
                metadata=merged_meta,
                remedy_hint=raw.remedy_hint,
                allowed_actions=raw.allowed_actions,
                error_type=raw.error_type,
            )

        if isinstance(raw, dict):
            if not raw.get("success", False):
                err_txt = _redact_sensitive_text(str(raw.get("error") or "Unknown external MCP error"))
                return ToolResult(
                    success=False,
                    output="",
                    error=err_txt,
                    error_type=raw.get("error_type") or "EXTERNAL_TOOL_ERROR",
                    metadata=base_meta,
                )
            return ToolResult(
                success=True,
                output=raw.get("output", ""),
                metadata=base_meta,
            )

        # Support mcp.types.CallToolResult objects
        if hasattr(raw, "isError") and hasattr(raw, "content"):
            blocks = getattr(raw, "content", []) or []
            texts = [str(getattr(b, "text", "")) for b in blocks if getattr(b, "text", None) is not None]
            joined = " ".join(texts)
            if getattr(raw, "isError", False):
                return ToolResult(
                    success=False,
                    output="",
                    error=_redact_sensitive_text(joined or "External MCP tool returned error."),
                    error_type="EXTERNAL_TOOL_ERROR",
                    metadata=base_meta,
                )
            return ToolResult(success=True, output=joined, metadata=base_meta)

        return ToolResult(success=True, output=str(raw), metadata=base_meta)


class ExternalToolPolicy:
    """Explicit fail-closed authorization policy for external fallback tools.

    Newly discovered tools are unauthorized by default until both their source
    and their tool identity are permitted by host policy (not by remote metadata).
    """

    def __init__(
        self,
        *,
        allowed_sources: Optional[Iterable[str]] = None,
        allowed_tools: Optional[Iterable[str]] = None,
        blocked_sources: Optional[Iterable[str]] = None,
        blocked_tools: Optional[Iterable[str]] = None,
        default_required_ring: int = DEFAULT_EXTERNAL_REQUIRED_RING,
        source_rings: Optional[Dict[str, int]] = None,
        tool_rings: Optional[Dict[str, int]] = None,
    ) -> None:
        self._lock = threading.RLock()
        self.allowed_sources: Set[str] = {str(s).strip() for s in (allowed_sources or ()) if str(s).strip()}
        self.allowed_tools: Set[str] = {str(t).strip() for t in (allowed_tools or ()) if str(t).strip()}
        self.blocked_sources: Set[str] = {str(s).strip() for s in (blocked_sources or ()) if str(s).strip()}
        self.blocked_tools: Set[str] = {str(t).strip() for t in (blocked_tools or ()) if str(t).strip()}
        self.default_required_ring: int = self._validate_ring(default_required_ring)
        self.source_rings: Dict[str, int] = {
            str(k).strip(): self._validate_ring(v) for k, v in (source_rings or {}).items() if str(k).strip()
        }
        self.tool_rings: Dict[str, int] = {
            str(k).strip(): self._validate_ring(v) for k, v in (tool_rings or {}).items() if str(k).strip()
        }

    @staticmethod
    def _validate_ring(ring: int) -> int:
        if not isinstance(ring, int) or isinstance(ring, bool) or not (0 <= ring <= 3):
            raise ValueError(f"Invalid required security ring {ring!r}; expected integer in [0, 3].")
        return ring

    def authorize_source(self, source_id: str) -> None:
        clean = str(source_id or "").strip()
        if not clean:
            raise ValueError("source_id cannot be empty.")
        with self._lock:
            self.blocked_sources.discard(clean)
            self.allowed_sources.add(clean)

    def revoke_source(self, source_id: str, *, block: bool = False) -> None:
        clean = str(source_id or "").strip()
        with self._lock:
            self.allowed_sources.discard(clean)
            if block and clean:
                self.blocked_sources.add(clean)

    def authorize_tool(self, tool_id: str) -> None:
        clean = str(tool_id or "").strip()
        if not clean:
            raise ValueError("tool_id cannot be empty.")
        with self._lock:
            self.blocked_tools.discard(clean)
            self.allowed_tools.add(clean)

    def revoke_tool(self, tool_id: str, *, block: bool = False) -> None:
        clean = str(tool_id or "").strip()
        with self._lock:
            self.allowed_tools.discard(clean)
            if block and clean:
                self.blocked_tools.add(clean)

    def set_tool_ring(self, tool_id: str, ring: int) -> None:
        valid_ring = self._validate_ring(ring)
        with self._lock:
            self.tool_rings[str(tool_id).strip()] = valid_ring

    def set_source_ring(self, source_id: str, ring: int) -> None:
        valid_ring = self._validate_ring(ring)
        with self._lock:
            self.source_rings[str(source_id).strip()] = valid_ring

    def is_source_allowed(self, source_id: str) -> bool:
        clean = str(source_id or "").strip()
        with self._lock:
            return bool(clean and clean not in self.blocked_sources and clean in self.allowed_sources)

    def is_tool_authorized(self, source_id: str, tool_id: str, external_tool_name: Optional[str] = None) -> bool:
        clean_src = str(source_id or "").strip()
        clean_tid = str(tool_id or "").strip()
        scoped_ext = f"{clean_src}:{external_tool_name}" if external_tool_name else ""
        with self._lock:
            if not clean_src or not clean_tid:
                return False
            if clean_src in self.blocked_sources or clean_tid in self.blocked_tools:
                return False
            if clean_src not in self.allowed_sources:
                return False
            return clean_tid in self.allowed_tools or (bool(scoped_ext) and scoped_ext in self.allowed_tools)

    def resolve_required_ring(self, source_id: str, tool_id: str) -> int:
        with self._lock:
            if tool_id in self.tool_rings:
                return self.tool_rings[tool_id]
            if source_id in self.source_rings:
                return self.source_rings[source_id]
            return self.default_required_ring


class ExternalAdaptedTool(BaseTool):
    """Zelyr BaseTool adapter wrapping a validated, policy-governed external tool."""

    is_external: bool = True

    def __init__(
        self,
        *,
        tool_id: str,
        description: str,
        input_schema: Dict[str, Any],
        source: ExternalToolSource,
        external_tool_name: str,
        required_ring: int = DEFAULT_EXTERNAL_REQUIRED_RING,
        required_permissions: Tuple[str, ...] = (),
        discovered_at: float = 0.0,
        source_epoch: int = 1,
        manager: Optional[ExternalToolManager] = None,
    ) -> None:
        super().__init__(
            tool_id=tool_id,
            name=tool_id,
            description=description,
            tier=1,
            is_core=False,
            requires_bridge=True,
        )
        self.is_external = True
        self.source_id = source.source_id
        self.source_type = source.source_type
        self.source_endpoint = source.endpoint_identity
        self.external_tool_name = external_tool_name
        self.required_ring = required_ring
        self.required_permissions = tuple(required_permissions)
        self.discovered_at = discovered_at
        self._source = source
        self._source_epoch = source_epoch
        self._manager = manager
        self._preserved_input_schema = copy.deepcopy(input_schema)

        props = self._preserved_input_schema.get("properties", {})
        req = self._preserved_input_schema.get("required", [])
        if isinstance(props, dict):
            req_set = set(req) if isinstance(req, list) else set()
            for p_name, p_meta in props.items():
                if not isinstance(p_meta, dict):
                    continue
                p_type = p_meta.get("type", "string")
                if isinstance(p_type, list):
                    non_null = [t for t in p_type if isinstance(t, str) and t != "null"]
                    p_type = non_null[0] if non_null else "string"
                elif not isinstance(p_type, str):
                    p_type = "string"
                self.add_parameter(
                    ToolParameter(
                        name=p_name,
                        type=p_type,
                        description=str(p_meta.get("description", "")),
                        required=p_name in req_set,
                        default=p_meta.get("default"),
                    )
                )

    @property
    def schema(self) -> Dict[str, Any]:
        """Preserve the full validated external JSON inputSchema without lossy flattening."""
        return copy.deepcopy(self._preserved_input_schema)

    def is_available(self) -> bool:
        """Fail closed unless this exact adapted tool instance matches live authorized healthy state."""
        if not self._source.is_available():
            return False
        if self._manager is not None:
            return self._manager.is_adapted_tool_live(self)
        return True

    async def execute(self, params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> ToolResult:
        """Normalize arguments, re-verify live authorization/staleness, and invoke the external source."""
        meta = {
            "external": True,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "source_endpoint": self.source_endpoint,
            "external_tool_name": self.external_tool_name,
            "tool_id": self.tool_id,
        }

        if self._manager is not None:
            entry = self._manager.get_tool_entry(self.tool_id)
            if entry is None or entry.source_epoch != self._source_epoch or entry.adapted_tool is not self:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"External tool '{self.tool_id}' registration is stale and cannot execute.",
                    error_type="TOOL_UNAVAILABLE",
                    metadata=meta,
                )
            if not entry.authorized:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"AuthorizationError: External tool '{self.tool_id}' is not authorized for execution.",
                    error_type="AUTHORIZATION_DENIED",
                    metadata=meta,
                )
            if entry.lifecycle != ExternalToolLifecycle.AVAILABLE:
                return ToolResult(
                    success=False,
                    output="",
                    error=(
                        f"External tool '{self.tool_id}' from source '{self.source_id}' is currently "
                        f"{entry.lifecycle.value}."
                    ),
                    error_type="TOOL_UNAVAILABLE",
                    metadata=meta,
                )

        if not self._source.is_available():
            if self._manager is not None:
                self._manager.mark_source_status(
                    self.source_id,
                    ExternalSourceHealth.DISCONNECTED,
                    error="External source is no longer connected.",
                )
            return ToolResult(
                success=False,
                output="",
                error=f"External source '{self.source_id}' is disconnected or unavailable.",
                error_type="TOOL_UNAVAILABLE",
                metadata=meta,
            )

        raw_args: Dict[str, Any] = dict(params or {}) if isinstance(params, dict) else {}
        if kwargs:
            raw_args.update(kwargs)

        norm_args, _, err_env = SafeArgumentNormalizer.normalize_and_validate(
            self.tool_id,
            self.schema,
            raw_args,
            argument_source="model",
        )
        if err_env is not None:
            return ToolResult(
                success=False,
                output=err_env,
                error=err_env["error"],
                error_type=err_env.get("error_type", "SCHEMA_VALIDATION_ERROR"),
                remedy_hint=err_env.get("remedy_hint"),
                allowed_actions=[self.tool_id],
                metadata=meta,
            )

        result = await self._source.invoke_tool(self.external_tool_name, norm_args)
        merged_meta = dict(getattr(result, "metadata", None) or {})
        merged_meta.update(meta)
        result.metadata = merged_meta
        return result


class ExternalToolManager:
    """Manages external source discovery, adaptation, policy enforcement, and lifecycle within ToolRegistry."""

    def __init__(self, registry: Any, *, policy: Optional[ExternalToolPolicy] = None) -> None:
        self._registry = registry
        self.policy = policy or ExternalToolPolicy()
        self._lock = threading.RLock()
        self._sources: Dict[str, ExternalSourceRecord] = {}
        self._entries: Dict[str, ExternalToolEntry] = {}
        self._state_version: int = 0

    def state_signature(self) -> Tuple[Any, ...]:
        """Return a deterministic signature for manifest cache invalidation."""
        with self._lock:
            src_sig = tuple(
                sorted(
                    (sid, rec.health.value, bool(rec.source.is_available()), rec.epoch, len(rec.validation_failures))
                    for sid, rec in self._sources.items()
                )
            )
            tool_sig = tuple(
                sorted(
                    (tid, e.lifecycle.value, e.authorized, e.required_ring, e.source_epoch)
                    for tid, e in self._entries.items()
                )
            )
            return (self._state_version, src_sig, tool_sig)

    def is_adapted_tool_live(self, tool: ExternalAdaptedTool) -> bool:
        with self._lock:
            entry = self._entries.get(tool.tool_id)
            if entry is None:
                return False
            src_rec = self._sources.get(entry.source_id)
            if src_rec is None or src_rec.health != ExternalSourceHealth.HEALTHY:
                return False
            return (
                entry.adapted_tool is tool
                and entry.source_epoch == tool._source_epoch
                and entry.source_epoch == src_rec.epoch
                and entry.schema_valid
                and entry.authorized
                and entry.lifecycle == ExternalToolLifecycle.AVAILABLE
                and src_rec.source.is_available()
            )

    def get_tool_entry(self, tool_id: str) -> Optional[ExternalToolEntry]:
        with self._lock:
            return self._entries.get(tool_id)

    def get_source_record(self, source_id: str) -> Optional[ExternalSourceRecord]:
        with self._lock:
            return self._sources.get(source_id)

    def register_source(self, source: ExternalToolSource) -> ExternalSourceRecord:
        with self._lock:
            existing = self._sources.get(source.source_id)
            if existing is not None:
                # Re-binding the same source_id replaces previous registration safely and invalidates stale epoch.
                self._invalidate_source_tools_locked(
                    source.source_id,
                    lifecycle=ExternalToolLifecycle.UNAVAILABLE,
                    reason="Source re-registered; awaiting discovery refresh.",
                    bump_epoch=True,
                )
            rec = ExternalSourceRecord(
                source=source,
                source_id=source.source_id,
                source_type=source.source_type,
                endpoint_identity=source.endpoint_identity,
                health=(
                    ExternalSourceHealth.HEALTHY
                    if source.is_available()
                    else ExternalSourceHealth.UNAVAILABLE
                ),
                epoch=(existing.epoch + 1) if existing else 1,
            )
            self._sources[source.source_id] = rec
            self._state_version += 1
            return rec

    def remove_source(self, source_id: str) -> bool:
        """Unregister all tools from *source_id*, invalidate stale instances, and remove the source."""
        with self._lock:
            if source_id not in self._sources:
                return False
            self._invalidate_source_tools_locked(
                source_id,
                lifecycle=ExternalToolLifecycle.UNAVAILABLE,
                reason="Source removed.",
                bump_epoch=True,
            )
            for tid in [k for k, e in self._entries.items() if e.source_id == source_id]:
                del self._entries[tid]
            del self._sources[source_id]
            self._state_version += 1
            return True

    def _is_native_or_conflicting_id_locked(self, tool_id: str, source_id: str) -> Optional[str]:
        from zelyr.tool_registry import BUILTIN_TOOLS, SWARM_TOOLS, _TOOL_REGISTRY_ALIASES

        if tool_id in BUILTIN_TOOLS or tool_id in SWARM_TOOLS or tool_id in _TOOL_REGISTRY_ALIASES:
            return f"External tool_id '{tool_id}' conflicts with reserved native tool identifier."
        core_reg = getattr(self._registry, "_core_registry", None)
        if core_reg is not None and core_reg.has_tool(tool_id):
            existing = core_reg.get_tool(tool_id)
            if existing is not None and not getattr(existing, "is_external", False):
                return f"External tool_id '{tool_id}' conflicts with already registered native tool."
        other_entry = self._entries.get(tool_id)
        if other_entry is not None and other_entry.source_id != source_id:
            return (
                f"Duplicate tool_id '{tool_id}' already registered by external source "
                f"'{other_entry.source_id}'."
            )
        return None

    def _sync_entry_registration_locked(self, entry: ExternalToolEntry, src_rec: ExternalSourceRecord) -> None:
        core_reg = getattr(self._registry, "_core_registry", None)
        is_live_source = src_rec.health == ExternalSourceHealth.HEALTHY and src_rec.source.is_available()
        is_authorized = self.policy.is_tool_authorized(
            entry.source_id, entry.tool_id, entry.external_tool_name
        )
        entry.authorized = is_authorized
        entry.required_ring = self.policy.resolve_required_ring(entry.source_id, entry.tool_id)

        if entry.validation_error is not None or not entry.schema_valid:
            entry.lifecycle = ExternalToolLifecycle.UNAVAILABLE
            if core_reg is not None and core_reg.has_tool(entry.tool_id):
                existing = core_reg.get_tool(entry.tool_id)
                if getattr(existing, "is_external", False):
                    core_reg.unregister_tool(entry.tool_id)
            return

        if not is_authorized:
            entry.lifecycle = ExternalToolLifecycle.VALIDATED
            if core_reg is not None and core_reg.has_tool(entry.tool_id):
                existing = core_reg.get_tool(entry.tool_id)
                if getattr(existing, "is_external", False):
                    core_reg.unregister_tool(entry.tool_id)
            return

        if not is_live_source:
            entry.lifecycle = (
                ExternalToolLifecycle.DEGRADED
                if src_rec.health == ExternalSourceHealth.DEGRADED
                else ExternalToolLifecycle.UNAVAILABLE
            )
            if core_reg is not None and core_reg.has_tool(entry.tool_id):
                existing = core_reg.get_tool(entry.tool_id)
                if getattr(existing, "is_external", False):
                    core_reg.unregister_tool(entry.tool_id)
            return

        # Transition: AUTHORIZED -> AVAILABLE
        entry.lifecycle = ExternalToolLifecycle.AUTHORIZED
        adapted = ExternalAdaptedTool(
            tool_id=entry.tool_id,
            description=entry.description,
            input_schema=entry.discovered_tool.input_schema,
            source=src_rec.source,
            external_tool_name=entry.external_tool_name,
            required_ring=entry.required_ring,
            required_permissions=entry.required_permissions,
            discovered_at=entry.discovered_at,
            source_epoch=src_rec.epoch,
            manager=self,
        )
        entry.adapted_tool = adapted
        entry.source_epoch = src_rec.epoch
        entry.lifecycle = ExternalToolLifecycle.AVAILABLE
        if core_reg is not None:
            core_reg.register_tool(entry.tool_id, adapted)

    def _invalidate_source_tools_locked(
        self,
        source_id: str,
        *,
        lifecycle: ExternalToolLifecycle,
        reason: str,
        bump_epoch: bool = True,
    ) -> None:
        src_rec = self._sources.get(source_id)
        if src_rec is not None and bump_epoch:
            src_rec.epoch += 1
        core_reg = getattr(self._registry, "_core_registry", None)
        for entry in self._entries.values():
            if entry.source_id != source_id:
                continue
            if bump_epoch and src_rec is not None:
                entry.source_epoch = src_rec.epoch
            entry.lifecycle = lifecycle
            entry.last_error = _redact_sensitive_text(reason)
            if core_reg is not None and core_reg.has_tool(entry.tool_id):
                existing = core_reg.get_tool(entry.tool_id)
                if getattr(existing, "is_external", False):
                    core_reg.unregister_tool(entry.tool_id)

    def mark_source_status(
        self,
        source_id: str,
        health: ExternalSourceHealth,
        *,
        error: Optional[str] = None,
    ) -> None:
        """Update source health state cleanly; mark tools unavailable and invalidate stale instances when unhealthy."""
        with self._lock:
            src_rec = self._sources.get(source_id)
            if src_rec is None:
                return
            src_rec.health = health
            src_rec.last_error = _redact_sensitive_text(error) if error else None
            if health == ExternalSourceHealth.HEALTHY:
                for entry in self._entries.values():
                    if entry.source_id == source_id:
                        self._sync_entry_registration_locked(entry, src_rec)
            else:
                target_lc = (
                    ExternalToolLifecycle.DEGRADED
                    if health == ExternalSourceHealth.DEGRADED
                    else ExternalToolLifecycle.UNAVAILABLE
                )
                self._invalidate_source_tools_locked(
                    source_id,
                    lifecycle=target_lc,
                    reason=error or f"Source '{source_id}' became {health.value}.",
                    bump_epoch=True,
                )
            self._state_version += 1

    def authorize_source(self, source_id: str) -> None:
        with self._lock:
            self.policy.authorize_source(source_id)
            src_rec = self._sources.get(source_id)
            if src_rec is not None:
                for entry in self._entries.values():
                    if entry.source_id == source_id:
                        self._sync_entry_registration_locked(entry, src_rec)
            self._state_version += 1

    def revoke_source(self, source_id: str, *, block: bool = False) -> None:
        with self._lock:
            self.policy.revoke_source(source_id, block=block)
            src_rec = self._sources.get(source_id)
            if src_rec is not None:
                src_rec.epoch += 1
            for entry in self._entries.values():
                if entry.source_id == source_id and src_rec is not None:
                    self._sync_entry_registration_locked(entry, src_rec)
            self._state_version += 1

    def authorize_tool(self, tool_id: str, *, authorize_source: bool = True) -> bool:
        with self._lock:
            entry = self._entries.get(tool_id)
            if entry is not None and authorize_source and entry.source_id not in self.policy.blocked_sources:
                self.policy.authorize_source(entry.source_id)
            self.policy.authorize_tool(tool_id)
            if entry is None:
                self._state_version += 1
                return True
            src_rec = self._sources.get(entry.source_id)
            if src_rec is not None:
                self._sync_entry_registration_locked(entry, src_rec)
            self._state_version += 1
            return entry.authorized

    def revoke_tool(self, tool_id: str, *, block: bool = False) -> None:
        with self._lock:
            self.policy.revoke_tool(tool_id, block=block)
            entry = self._entries.get(tool_id)
            if entry is not None:
                src_rec = self._sources.get(entry.source_id)
                if src_rec is not None:
                    src_rec.epoch += 1
                    self._sync_entry_registration_locked(entry, src_rec)
            self._state_version += 1

    async def discover_source(self, source_id: str) -> Dict[str, Any]:
        """Discover, schema-validate, and reconcile tools from a single registered source."""
        with self._lock:
            src_rec = self._sources.get(source_id)
            if src_rec is None:
                raise KeyError(f"Unknown external tool source '{source_id}'.")
            source = src_rec.source

        try:
            discovered_list = await source.discover_tools()
        except Exception as exc:
            err_msg = _redact_sensitive_text(str(exc))
            self.mark_source_status(source_id, ExternalSourceHealth.DISCONNECTED, error=err_msg)
            return {
                "source_id": source_id,
                "health": ExternalSourceHealth.DISCONNECTED.value,
                "discovered": 0,
                "validated": 0,
                "available": 0,
                "errors": [{"source_id": source_id, "error_type": "SOURCE_UNAVAILABLE", "error": err_msg}],
            }

        with self._lock:
            src_rec = self._sources.get(source_id)
            if src_rec is None:
                return {"source_id": source_id, "health": ExternalSourceHealth.DISCONNECTED.value, "discovered": 0, "validated": 0, "available": 0, "errors": []}

            src_rec.health = ExternalSourceHealth.HEALTHY if source.is_available() else ExternalSourceHealth.UNAVAILABLE
            src_rec.last_discovered_at = time.time()
            src_rec.last_error = None
            src_rec.validation_failures = []
            src_rec.epoch += 1

            seen_in_batch: Set[str] = set()
            seen_tool_ids: Set[str] = set()
            validated_count = 0
            available_count = 0

            for raw_tool in discovered_list:
                tid = raw_tool.tool_id
                # Track duplicates within the same source payload or against native/other sources.
                conflict_msg: Optional[str] = None
                if raw_tool.external_tool_name in seen_in_batch or tid in seen_tool_ids:
                    conflict_msg = f"Duplicate external tool identifier '{tid}' returned by source '{source_id}'."
                else:
                    conflict_msg = self._is_native_or_conflicting_id_locked(tid, source_id)

                if conflict_msg is not None:
                    dup_err = _schema_compat_error(
                        tid,
                        source_id,
                        "duplicate_tool_id",
                        conflict_msg,
                        error_type="DUPLICATE_TOOL_ID",
                    )
                    src_rec.validation_failures.append(dup_err)
                    # Preserve any already valid distinct entry when another duplicate collides in batch
                    if tid not in self._entries:
                        self._entries[tid] = ExternalToolEntry(
                            tool_id=tid,
                            source_id=source_id,
                            source_type=src_rec.source_type,
                            source_endpoint=raw_tool.source_endpoint,
                            external_tool_name=raw_tool.external_tool_name,
                            description=raw_tool.description,
                            discovered_tool=raw_tool,
                            lifecycle=ExternalToolLifecycle.UNAVAILABLE,
                            authorized=False,
                            schema_valid=False,
                            required_ring=self.policy.resolve_required_ring(source_id, tid),
                            required_permissions=raw_tool.required_permissions,
                            discovered_at=raw_tool.discovered_at,
                            source_epoch=src_rec.epoch,
                            validation_error=dup_err,
                            last_error=conflict_msg,
                        )
                    continue

                seen_in_batch.add(raw_tool.external_tool_name)
                seen_tool_ids.add(tid)

                schema_res = validate_external_tool_schema(
                    tid,
                    raw_tool.external_tool_name,
                    source_id,
                    raw_tool.input_schema,
                )
                req_ring = self.policy.resolve_required_ring(source_id, tid)
                if not schema_res.valid or schema_res.normalized_schema is None:
                    if schema_res.error:
                        src_rec.validation_failures.append(schema_res.error)
                    bad_entry = ExternalToolEntry(
                        tool_id=tid,
                        source_id=source_id,
                        source_type=src_rec.source_type,
                        source_endpoint=raw_tool.source_endpoint,
                        external_tool_name=raw_tool.external_tool_name,
                        description=raw_tool.description,
                        discovered_tool=raw_tool,
                        lifecycle=ExternalToolLifecycle.UNAVAILABLE,
                        authorized=False,
                        schema_valid=False,
                        required_ring=req_ring,
                        required_permissions=raw_tool.required_permissions,
                        discovered_at=raw_tool.discovered_at,
                        source_epoch=src_rec.epoch,
                        validation_error=schema_res.error,
                        last_error=(schema_res.error or {}).get("error"),
                    )
                    self._entries[tid] = bad_entry
                    self._sync_entry_registration_locked(bad_entry, src_rec)
                    continue

                validated_count += 1
                clean_discovered = DiscoveredExternalTool(
                    source_id=raw_tool.source_id,
                    source_type=raw_tool.source_type,
                    source_endpoint=raw_tool.source_endpoint,
                    external_tool_name=raw_tool.external_tool_name,
                    tool_id=tid,
                    name=tid,
                    description=raw_tool.description,
                    input_schema=schema_res.normalized_schema,
                    availability_status=ExternalToolLifecycle.VALIDATED,
                    required_permissions=raw_tool.required_permissions,
                    required_ring=req_ring,
                    discovered_at=raw_tool.discovered_at,
                    metadata=dict(raw_tool.metadata),
                )
                entry = ExternalToolEntry(
                    tool_id=tid,
                    source_id=source_id,
                    source_type=src_rec.source_type,
                    source_endpoint=raw_tool.source_endpoint,
                    external_tool_name=raw_tool.external_tool_name,
                    description=raw_tool.description,
                    discovered_tool=clean_discovered,
                    lifecycle=ExternalToolLifecycle.VALIDATED,
                    authorized=False,
                    schema_valid=True,
                    required_ring=req_ring,
                    required_permissions=raw_tool.required_permissions,
                    discovered_at=raw_tool.discovered_at,
                    source_epoch=src_rec.epoch,
                )
                self._entries[tid] = entry
                self._sync_entry_registration_locked(entry, src_rec)
                if entry.lifecycle == ExternalToolLifecycle.AVAILABLE:
                    available_count += 1

            # Mark previously known tools from this source that vanished on refresh as unavailable.
            core_reg = getattr(self._registry, "_core_registry", None)
            for tid, existing_entry in list(self._entries.items()):
                if existing_entry.source_id == source_id and tid not in seen_tool_ids:
                    existing_entry.lifecycle = ExternalToolLifecycle.UNAVAILABLE
                    existing_entry.source_epoch = src_rec.epoch
                    existing_entry.last_error = f"Tool '{tid}' no longer advertised by source '{source_id}'."
                    if core_reg is not None and core_reg.has_tool(tid):
                        reg_tool = core_reg.get_tool(tid)
                        if getattr(reg_tool, "is_external", False):
                            core_reg.unregister_tool(tid)

            self._state_version += 1
            return {
                "source_id": source_id,
                "health": src_rec.health.value,
                "discovered": len(discovered_list),
                "validated": validated_count,
                "available": available_count,
                "errors": list(src_rec.validation_failures),
            }

    async def discover_all(self, source_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if source_id is not None:
            return [await self.discover_source(source_id)]
        with self._lock:
            source_ids = list(self._sources.keys())
        reports = []
        for sid in source_ids:
            reports.append(await self.discover_source(sid))
        return reports

    def get_manifest_summary(
        self,
        *,
        caller_ring: Optional[int] = None,
        allowed_tools: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """Build structured, secret-free external tool telemetry for RuntimeEnvironmentManifest."""
        allowed_set: Optional[Set[str]] = None
        if allowed_tools is not None:
            if isinstance(allowed_tools, (str, bytes, dict)) or not isinstance(allowed_tools, Iterable):
                allowed_set = set()
            else:
                allowed_set = {str(x).strip() for x in allowed_tools if isinstance(x, str) and x.strip()}

        with self._lock:
            sources_summary: Dict[str, Dict[str, Any]] = {}
            all_failures: List[Dict[str, Any]] = []
            available_ids: List[str] = []
            unavailable_ids: List[str] = []
            degraded_ids: List[str] = []
            authorized_ids: List[str] = []
            discovered_ids: List[str] = []
            details: Dict[str, Dict[str, Any]] = {}

            for sid, src_rec in sorted(self._sources.items()):
                # Check live availability cleanly without blocking
                effective_health = src_rec.health
                if effective_health == ExternalSourceHealth.HEALTHY and not src_rec.source.is_available():
                    effective_health = ExternalSourceHealth.DISCONNECTED

                src_entries = [e for e in self._entries.values() if e.source_id == sid]
                src_failures = list(src_rec.validation_failures)
                all_failures.extend(src_failures)

                src_auth = sum(1 for e in src_entries if e.authorized)
                src_avail = sum(
                    1
                    for e in src_entries
                    if e.lifecycle == ExternalToolLifecycle.AVAILABLE and effective_health == ExternalSourceHealth.HEALTHY
                )
                sources_summary[sid] = {
                    "source_id": sid,
                    "source_type": src_rec.source_type,
                    "endpoint": sanitize_source_endpoint(src_rec.endpoint_identity),
                    "health": effective_health.value,
                    "discovered_count": len(src_entries),
                    "authorized_count": src_auth,
                    "available_count": src_avail,
                    "unavailable_count": len(src_entries) - src_avail,
                    "validation_failure_count": len(src_failures),
                    "last_discovered_at": src_rec.last_discovered_at,
                    "last_error": _redact_sensitive_text(src_rec.last_error) if src_rec.last_error else None,
                }

            for tid, entry in sorted(self._entries.items()):
                discovered_ids.append(tid)
                src_rec = self._sources.get(entry.source_id)
                src_live = (
                    src_rec is not None
                    and src_rec.health == ExternalSourceHealth.HEALTHY
                    and src_rec.source.is_available()
                )
                eff_lc = entry.lifecycle
                if eff_lc == ExternalToolLifecycle.AVAILABLE and not src_live:
                    eff_lc = ExternalToolLifecycle.UNAVAILABLE

                caller_authorized = bool(entry.authorized)
                if caller_authorized and allowed_set is not None and tid not in allowed_set:
                    caller_authorized = False
                if caller_authorized and caller_ring is not None:
                    if not isinstance(caller_ring, int) or isinstance(caller_ring, bool) or caller_ring > entry.required_ring:
                        caller_authorized = False

                if entry.authorized and caller_authorized:
                    authorized_ids.append(tid)

                if eff_lc == ExternalToolLifecycle.AVAILABLE:
                    available_ids.append(tid)
                elif eff_lc == ExternalToolLifecycle.DEGRADED:
                    degraded_ids.append(tid)
                    unavailable_ids.append(tid)
                else:
                    unavailable_ids.append(tid)

                details[tid] = {
                    "tool_id": tid,
                    "source_id": entry.source_id,
                    "source_type": entry.source_type,
                    "external_tool_name": entry.external_tool_name,
                    "lifecycle": eff_lc.value,
                    "authorized": entry.authorized,
                    "caller_authorized": caller_authorized,
                    "schema_valid": entry.schema_valid,
                    "required_ring": entry.required_ring,
                    "required_permissions": list(entry.required_permissions),
                    "discovered_at": entry.discovered_at,
                    "is_external": True,
                    "last_error": _redact_sensitive_text(entry.last_error) if entry.last_error else None,
                }

            healthy_sources = sum(
                1 for s in sources_summary.values() if s["health"] == ExternalSourceHealth.HEALTHY.value
            )
            return {
                "sources": sources_summary,
                "counts": {
                    "sources": len(sources_summary),
                    "healthy_sources": healthy_sources,
                    "discovered": len(discovered_ids),
                    "validated": sum(1 for e in self._entries.values() if e.schema_valid),
                    "authorized": len(authorized_ids),
                    "available": len(available_ids),
                    "unavailable": len(unavailable_ids),
                    "degraded": len(degraded_ids),
                    "schema_failures": len(all_failures),
                },
                "available": available_ids,
                "unavailable": unavailable_ids,
                "degraded": degraded_ids,
                "authorized": authorized_ids,
                "validation_failures": all_failures,
                "details": details,
            }
