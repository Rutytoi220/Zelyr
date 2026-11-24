"""Validated structured plan schema, compiler, review, and replanning bridge for Zelyr.

Converts structured model planning output into validated ``TaskGraph`` DAGs and
``PendingActionProposal`` descriptors while strictly enforcing schema versions,
dependency integrity, conservative resource/conflict rules, and security boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import uuid

from zelyr.engine.checkpoint import (
    CheckpointManager,
    PendingActionProposal,
    ProposalStatus,
)
from zelyr.engine.execution_ledger import (
    CentralExecutionLedger,
    ExecutionStatus,
    LedgerEventType,
    TurnExecutionLedger,
    VerificationStatus,
    canonicalize_target,
    get_default_central_ledger,
    parse_acceptance_criteria,
    sanitize_ledger_payload,
    sanitize_ledger_text,
)
from zelyr.engine.normalizer import (
    RESERVED_AUTHORIZATION_KEYS,
    find_reserved_authorization_key,
    strict_json_loads,
)
from zelyr.engine.task_graph import (
    TaskGraph,
    TaskGraphEdge,
    TaskGraphStatus,
    TaskGraphValidationError,
    TaskNode,
    TaskNodeStatus,
    extract_node_conflict_profile,
)
from zelyr.planning.planner import PlanError

STRUCTURED_PLAN_VERSION: int = 1
SUPPORTED_PLAN_VERSIONS: frozenset[int] = frozenset({1})
MAX_NODE_ID_LENGTH: int = 64
DEFAULT_MAX_NODE_TIMEOUT_SECONDS: float = 600.0
MIN_NODE_TIMEOUT_SECONDS: float = 0.05

_NODE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
_VAGUE_DESCRIPTION_PATTERNS: Tuple[re.Pattern[str], ...] = tuple(
    re.compile(pat, re.IGNORECASE)
    for pat in (
        r"^(?:continue|keep)\s+working\.?$",
        r"^do\s+(?:the\s+)?(?:task|work|it|something)\.?$",
        r"^(?:work\s+on\s+(?:task|it)|todo|tbd|placeholder|step\s*\d*)\.?$",
    )
)

# Security-sensitive fields that untrusted planner JSON may NEVER set or override.
FORBIDDEN_PLAN_SECURITY_KEYS: frozenset[str] = frozenset(
    {
        *RESERVED_AUTHORIZATION_KEYS,
        "caller_ring_grant",
        "allowed_tools_grant",
        "_caller_ring",
        "_allowed_tools",
        "permissions",
        "privileges",
        "execution_privileges",
        "admin",
        "root",
        "bypass_auth",
        "bypass_confirmation",
        "skip_authorization",
        "skip_verification",
        "unrestricted",
        "unrestricted_tools",
        "auto_approve",
        "workspace_root",
        "filesystem_root",
        "sandbox_root",
        "root_dir",
        "token",
        "secret",
        "api_key",
        "password",
        "authorization",
        "credentials",
    }
)

FORBIDDEN_PLAN_CONCURRENCY_KEYS: frozenset[str] = frozenset(
    {
        "max_concurrency",
        "concurrency_limit",
        "parallel_limit",
        "max_parallel",
    }
)

_IGNORED_MODEL_STATUS_KEYS: frozenset[str] = frozenset(
    {
        "execution_status",
        "verification_status",
        "verified",
        "completed",
        "done",
        "status",
    }
)


class PlanCompilationError(PlanError):
    """Base error raised during structured plan parsing, validation, or compilation."""


class PlanValidationError(PlanCompilationError):
    """Raised when structured planner output violates schema, DAG, deliverable, or security rules."""


class UnsupportedPlanVersionError(PlanCompilationError):
    """Raised when a plan declares an unsupported structured schema version."""


class PlanConfirmationRequiredError(PlanCompilationError):
    """Raised when graph execution requires explicit confirmation before dispatch."""

    def __init__(self, message: str, *, bundle: Optional[Any] = None) -> None:
        super().__init__(message)
        self.bundle = bundle


def _sanitize_plan_text(raw: Any, *, max_bytes: int = 512) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    cleaned, _, _ = sanitize_ledger_text(raw, max_bytes=max_bytes, is_external=True, strip_injection=True)
    return (cleaned or "").strip()


def _normalize_node_id(raw_id: Any, *, field_label: str = "node_id") -> str:
    if not isinstance(raw_id, str) or not raw_id.strip():
        raise PlanValidationError(f"Plan entry missing non-empty '{field_label}'.")
    cleaned = raw_id.strip()
    if len(cleaned) > MAX_NODE_ID_LENGTH or not _NODE_ID_PATTERN.match(cleaned):
        raise PlanValidationError(
            f"Invalid identifier '{cleaned}' for '{field_label}': must match [A-Za-z0-9][A-Za-z0-9_.:-]{{0,63}}."
        )
    return cleaned


def _validate_timeout_value(raw_timeout: Any, *, max_timeout: float = DEFAULT_MAX_NODE_TIMEOUT_SECONDS) -> Optional[float]:
    if raw_timeout is None:
        return None
    if isinstance(raw_timeout, bool) or not isinstance(raw_timeout, (int, float)):
        raise PlanValidationError(f"timeout_seconds must be a positive finite number, got {raw_timeout!r}.")
    val = float(raw_timeout)
    if not math.isfinite(val) or val <= 0.0:
        raise PlanValidationError(f"timeout_seconds must be > 0 and finite, got {raw_timeout!r}.")
    return max(MIN_NODE_TIMEOUT_SECONDS, min(val, float(max_timeout)))


def _is_vague_description(desc: str) -> bool:
    clean = desc.strip()
    if len(clean) < 3:
        return True
    return any(pat.match(clean) for pat in _VAGUE_DESCRIPTION_PATTERNS)


def _format_verification_requirement(raw_vr: Any) -> Optional[str]:
    if raw_vr is None:
        return None
    if isinstance(raw_vr, str):
        cleaned = _sanitize_plan_text(raw_vr, max_bytes=256)
        return cleaned or None
    if isinstance(raw_vr, dict):
        chk = _sanitize_plan_text(str(raw_vr.get("check") or raw_vr.get("type") or ""), max_bytes=64)
        tgt = _sanitize_plan_text(str(raw_vr.get("target") or ""), max_bytes=160)
        exp = _sanitize_plan_text(str(raw_vr.get("expected") or ""), max_bytes=128) if raw_vr.get("expected") is not None else ""
        if chk and tgt:
            return f"{chk}:{tgt}:{exp}" if exp else f"{chk}:{tgt}"
        if chk or tgt:
            return _sanitize_plan_text(f"{chk} {tgt}".strip(), max_bytes=256)
    raise PlanValidationError(f"verification_requirement must be a string, object, or null, got {type(raw_vr).__name__}.")


def _resolve_resource_within_workspace(raw_res: str, workspace_root: Optional[Path]) -> str:
    """Canonicalize explicit resource identifiers and reject traversal outside trusted workspace_root."""
    s = raw_res.strip()
    if not s:
        raise PlanValidationError("target_resources entries must be non-empty strings.")

    lower = s.lower()
    if lower in ("desktop_gui", "gui", "exclusive:desktop_gui"):
        return "exclusive:desktop_gui"
    if lower in ("browser", "browser_session", "exclusive:browser_session"):
        return "exclusive:browser_session"
    if lower in ("shell", "opaque_shell", "exclusive:opaque_shell"):
        return "exclusive:opaque_shell"
    if lower.startswith("exclusive:") or lower.startswith("external:"):
        return s

    path_candidate = s
    for prefix in ("file:", "path:", "target:"):
        if lower.startswith(prefix):
            path_candidate = s[len(prefix) :].strip()
            break

    is_path_like = (
        lower.startswith(("file:", "path:", "target:"))
        or "/" in path_candidate
        or "\\" in path_candidate
        or "." in path_candidate
    )
    if not is_path_like:
        return s

    if not path_candidate:
        raise PlanValidationError(f"Empty path in target resource '{s}'.")

    # Reject explicit parent-directory traversal attempts regardless of workspace configuration
    parts = Path(path_candidate).parts
    if ".." in parts:
        raise PlanValidationError(f"Path traversal ('..') is forbidden in target resource '{s}'.")

    if workspace_root is not None:
        ws_resolved = workspace_root.resolve()
        p_obj = Path(path_candidate)
        cand_resolved = p_obj.resolve() if p_obj.is_absolute() else (ws_resolved / p_obj).resolve()
        try:
            cand_resolved.relative_to(ws_resolved)
        except ValueError as exc:
            raise PlanValidationError(
                f"Target resource '{s}' escapes configured workspace root '{ws_resolved}'."
            ) from exc
        return f"target:{cand_resolved}"

    canon = canonicalize_target(path_candidate, check_type="file_exists")
    return f"target:{canon}" if canon else f"target:{path_candidate}"


@dataclass(frozen=True)
class PlannedTaskNode:
    """Validated unit of work inside a StructuredPlan prior to execution."""

    node_id: str
    description: str
    expected_outcome: str
    depends_on: Tuple[str, ...] = ()
    verification_requirement: Optional[str] = None
    requires_verified_dependencies: bool = True
    target_resources: Tuple[str, ...] = ()
    tool_hint: Optional[str] = None
    tool_arguments: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: Optional[float] = None
    order_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "node_id": self.node_id,
            "description": self.description,
            "expected_outcome": self.expected_outcome,
            "depends_on": list(self.depends_on),
            "requires_verified_dependencies": bool(self.requires_verified_dependencies),
            "order_index": int(self.order_index),
        }
        if self.verification_requirement:
            out["verification_requirement"] = self.verification_requirement
        if self.target_resources:
            out["target_resources"] = list(self.target_resources)
        if self.tool_hint:
            out["tool_hint"] = self.tool_hint
        if self.tool_arguments:
            out["tool_arguments"] = dict(self.tool_arguments)
        if self.timeout_seconds is not None:
            out["timeout_seconds"] = float(self.timeout_seconds)
        return out


@dataclass(frozen=True)
class StructuredPlan:
    """Versioned, sanitized planner representation ready to compile into a TaskGraph."""

    plan_version: int
    plan_id: str
    objective: str
    nodes: Tuple[PlannedTaskNode, ...]
    summary_notes: str = ""
    warnings: Tuple[str, ...] = ()
    plan_digest: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "plan_version": int(self.plan_version),
            "plan_id": self.plan_id,
            "objective": self.objective,
            "nodes": [n.to_dict() for n in self.nodes],
        }
        if self.summary_notes:
            payload["summary_notes"] = self.summary_notes
        if self.plan_digest:
            payload["plan_digest"] = self.plan_digest
        return payload


def compute_structured_plan_digest(plan: Union[StructuredPlan, Dict[str, Any]]) -> str:
    """Compute a deterministic SHA-256 digest over canonical sanitized plan structure."""
    data = plan.to_dict() if isinstance(plan, StructuredPlan) else dict(plan)
    canonical = {
        "plan_version": int(data.get("plan_version", STRUCTURED_PLAN_VERSION)),
        "plan_id": str(data.get("plan_id") or ""),
        "objective": str(data.get("objective") or ""),
        "summary_notes": str(data.get("summary_notes") or ""),
        "nodes": [
            {
                "node_id": str(n.get("node_id") or ""),
                "description": str(n.get("description") or ""),
                "expected_outcome": str(n.get("expected_outcome") or ""),
                "depends_on": list(n.get("depends_on") or []),
                "verification_requirement": n.get("verification_requirement"),
                "requires_verified_dependencies": bool(n.get("requires_verified_dependencies", True)),
                "target_resources": list(n.get("target_resources") or []),
                "tool_hint": n.get("tool_hint"),
                "tool_arguments": n.get("tool_arguments") or {},
                "timeout_seconds": n.get("timeout_seconds"),
                "order_index": int(n.get("order_index", 0)),
            }
            for n in (data.get("nodes") or [])
            if isinstance(n, dict)
        ],
    }
    raw = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode(
        "utf-8", errors="replace"
    )
    return hashlib.sha256(raw).hexdigest()


def extract_plan_json(raw_output: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Strictly extract a single top-level JSON object from planner output."""
    if isinstance(raw_output, dict):
        return dict(raw_output)
    if not isinstance(raw_output, str):
        raise PlanValidationError(f"Planner output must be a JSON string or dict, got {type(raw_output).__name__}.")

    trimmed = raw_output.strip()
    if not trimmed:
        raise PlanValidationError("Planner output is empty and cannot be parsed as a structured plan.")

    # 1. Exact JSON object string
    if trimmed.startswith("{"):
        try:
            parsed = strict_json_loads(trimmed)
        except Exception as exc:
            raise PlanValidationError(f"Malformed plan JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise PlanValidationError(f"Structured plan must be a JSON object, received {type(parsed).__name__}.")
        return parsed

    # 2. Markdown code fence containing a plan JSON object
    fence_matches = list(
        re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", trimmed, flags=re.IGNORECASE)
    )
    fenced_dicts: List[Dict[str, Any]] = []
    for fm in fence_matches:
        inner = fm.group(1).strip()
        if not inner.startswith("{"):
            continue
        try:
            parsed = strict_json_loads(inner)
        except Exception as exc:
            raise PlanValidationError(f"Malformed JSON inside plan code block: {exc}") from exc
        if isinstance(parsed, dict):
            fenced_dicts.append(parsed)
    if len(fenced_dicts) > 1:
        raise PlanValidationError("Multiple plan JSON blocks found; planner must return one cohesive structured plan.")
    if len(fenced_dicts) == 1:
        return fenced_dicts[0]

    # 3. Single unambiguous embedded JSON object after reasoning prose
    first_brace = trimmed.find("{")
    if first_brace != -1:
        decoder = json.JSONDecoder()
        try:
            _, end_offset = decoder.raw_decode(trimmed, first_brace)
        except Exception as exc:
            raise PlanValidationError(f"Malformed JSON in planner response: {exc}") from exc
        candidate_slice = trimmed[first_brace:end_offset]
        try:
            parsed_strict = strict_json_loads(candidate_slice)
        except Exception as exc:
            raise PlanValidationError(f"Malformed plan JSON constant/syntax: {exc}") from exc
        trailing = trimmed[end_offset:].strip()
        if trailing and "{" in trailing:
            raise PlanValidationError("Ambiguous trailing JSON object after plan payload.")
        if isinstance(parsed_strict, dict):
            return parsed_strict

    raise PlanValidationError("Planner response did not contain a valid JSON plan object.")


def _strip_security_overrides(
    mapping: Dict[str, Any],
    *,
    scope_label: str,
    warnings: List[str],
) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for k, v in mapping.items():
        k_low = str(k).strip().lower()
        if k_low in FORBIDDEN_PLAN_SECURITY_KEYS:
            warnings.append(f"Stripped untrusted security override key '{k}' from {scope_label}.")
            continue
        if k_low in FORBIDDEN_PLAN_CONCURRENCY_KEYS:
            warnings.append(f"Ignored untrusted concurrency override '{k}' in {scope_label}; runtime configuration governs concurrency.")
            continue
        if k_low in _IGNORED_MODEL_STATUS_KEYS:
            warnings.append(f"Ignored model completion/status claim '{k}' in {scope_label}.")
            continue
        cleaned[str(k)] = v
    return cleaned


def parse_structured_plan(
    raw_output: Union[str, Dict[str, Any]],
    *,
    fallback_objective: Optional[str] = None,
    workspace_root: Optional[Union[str, Path]] = None,
    max_node_timeout: float = DEFAULT_MAX_NODE_TIMEOUT_SECONDS,
) -> StructuredPlan:
    """Parse and strictly validate raw planner output into a sanitized StructuredPlan."""
    raw_obj = extract_plan_json(raw_output)
    warnings: List[str] = []

    ws_path: Optional[Path] = Path(workspace_root) if workspace_root is not None else None
    top_clean = _strip_security_overrides(raw_obj, scope_label="plan", warnings=warnings)

    raw_ver = top_clean.get("plan_version")
    if raw_ver is None:
        raise PlanValidationError("Structured plan missing required integer 'plan_version'.")
    if isinstance(raw_ver, bool) or not isinstance(raw_ver, int):
        raise PlanValidationError(f"Structured plan 'plan_version' must be an integer, got {raw_ver!r}.")
    if raw_ver not in SUPPORTED_PLAN_VERSIONS:
        raise UnsupportedPlanVersionError(
            f"Unsupported structured plan version {raw_ver}; supported versions: {sorted(SUPPORTED_PLAN_VERSIONS)}."
        )

    raw_obj_text = top_clean.get("objective")
    if raw_obj_text is None and fallback_objective:
        raw_obj_text = fallback_objective
    if not isinstance(raw_obj_text, str) or not raw_obj_text.strip():
        raise PlanValidationError("Structured plan missing non-empty 'objective'.")
    objective = _sanitize_plan_text(raw_obj_text, max_bytes=1024)
    if not objective:
        raise PlanValidationError("Structured plan objective became empty after sanitization.")

    raw_pid = top_clean.get("plan_id")
    plan_id = _normalize_node_id(raw_pid, field_label="plan_id") if raw_pid is not None else f"plan_{uuid.uuid4().hex[:12]}"

    raw_nodes = top_clean.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise PlanValidationError("Structured plan 'nodes' must be a non-empty list of planned tasks.")

    parsed_nodes: List[PlannedTaskNode] = []
    seen_ids: Set[str] = set()
    for idx, entry in enumerate(raw_nodes):
        if not isinstance(entry, dict):
            raise PlanValidationError(f"Plan node at index {idx} must be a JSON object.")
        node_clean = _strip_security_overrides(entry, scope_label=f"node[{idx}]", warnings=warnings)
        nid = _normalize_node_id(node_clean.get("node_id"), field_label="node_id")
        if nid in seen_ids:
            raise PlanValidationError(f"Duplicate node_id '{nid}' in structured plan.")
        seen_ids.add(nid)

        raw_desc = node_clean.get("description")
        if not isinstance(raw_desc, str) or not raw_desc.strip():
            raise PlanValidationError(f"Plan node '{nid}' missing non-empty 'description'.")
        desc = _sanitize_plan_text(raw_desc, max_bytes=512)
        if not desc or _is_vague_description(desc):
            raise PlanValidationError(
                f"Plan node '{nid}' description {desc!r} is too vague; specify an actionable deliverable."
            )

        raw_out = node_clean.get("expected_outcome")
        if not isinstance(raw_out, str) or not raw_out.strip():
            raise PlanValidationError(f"Plan node '{nid}' missing observable 'expected_outcome'.")
        expected_out = _sanitize_plan_text(raw_out, max_bytes=512)
        if not expected_out:
            raise PlanValidationError(f"Plan node '{nid}' expected_outcome became empty after sanitization.")

        raw_deps = node_clean.get("depends_on", [])
        if raw_deps is None:
            raw_deps = []
        if not isinstance(raw_deps, list):
            raise PlanValidationError(f"Plan node '{nid}' field 'depends_on' must be a list.")
        deps: List[str] = []
        for d_item in raw_deps:
            d_norm = _normalize_node_id(d_item, field_label="depends_on entry")
            if d_norm == nid:
                raise PlanValidationError(f"Self-dependency detected on planned node '{nid}'.")
            if d_norm in deps:
                raise PlanValidationError(f"Duplicate dependency '{d_norm}' on planned node '{nid}'.")
            deps.append(d_norm)

        req_ver = node_clean.get("requires_verified_dependencies", True)
        if not isinstance(req_ver, bool):
            raise PlanValidationError(
                f"Plan node '{nid}' field 'requires_verified_dependencies' must be a boolean."
            )

        ver_req = _format_verification_requirement(node_clean.get("verification_requirement"))

        raw_res_list = node_clean.get("target_resources", [])
        if raw_res_list is None:
            raw_res_list = []
        if not isinstance(raw_res_list, list):
            raise PlanValidationError(f"Plan node '{nid}' field 'target_resources' must be a list.")
        norm_resources: List[str] = []
        for r_item in raw_res_list:
            if not isinstance(r_item, str) or not r_item.strip():
                raise PlanValidationError(f"Plan node '{nid}' target_resources entries must be non-empty strings.")
            res_canon = _resolve_resource_within_workspace(r_item, ws_path)
            if res_canon not in norm_resources:
                norm_resources.append(res_canon)

        raw_tool = node_clean.get("tool_hint")
        tool_hint: Optional[str] = None
        if raw_tool is not None:
            if not isinstance(raw_tool, str) or not raw_tool.strip():
                raise PlanValidationError(f"Plan node '{nid}' tool_hint must be a non-empty string or null.")
            tool_hint = _sanitize_plan_text(raw_tool, max_bytes=128)

        raw_targs = node_clean.get("tool_arguments", {})
        if raw_targs is None:
            raw_targs = {}
        if not isinstance(raw_targs, dict):
            raise PlanValidationError(f"Plan node '{nid}' tool_arguments must be a JSON object.")
        reserved_key = find_reserved_authorization_key(raw_targs)
        if reserved_key is not None:
            raise PlanValidationError(
                f"Plan node '{nid}' tool_arguments contains forbidden authorization field '{reserved_key}'."
            )
        clean_targs, _, _ = sanitize_ledger_payload(
            _strip_security_overrides(raw_targs, scope_label=f"node[{nid}].tool_arguments", warnings=warnings),
            max_bytes=2048,
        )
        # Validate any explicit file paths inside tool_arguments against traversal / workspace boundary
        for p_key in ("path", "file_path", "filepath", "target", "dest", "destination"):
            p_val = clean_targs.get(p_key)
            if isinstance(p_val, str) and p_val.strip():
                if ws_path is not None or ".." in Path(p_val.strip()).parts:
                    _resolve_resource_within_workspace(p_val, ws_path)

        timeout_s = _validate_timeout_value(node_clean.get("timeout_seconds"), max_timeout=max_node_timeout)

        raw_ord = node_clean.get("order_index", idx)
        if isinstance(raw_ord, bool) or not isinstance(raw_ord, int) or raw_ord < 0:
            raise PlanValidationError(f"Plan node '{nid}' has invalid order_index {raw_ord!r}.")

        parsed_nodes.append(
            PlannedTaskNode(
                node_id=nid,
                description=desc,
                expected_outcome=expected_out,
                depends_on=tuple(deps),
                verification_requirement=ver_req,
                requires_verified_dependencies=req_ver,
                target_resources=tuple(norm_resources),
                tool_hint=tool_hint,
                tool_arguments=dict(clean_targs),
                timeout_seconds=timeout_s,
                order_index=raw_ord,
            )
        )

    notes = _sanitize_plan_text(top_clean.get("summary_notes") or "", max_bytes=512)
    plan = StructuredPlan(
        plan_version=raw_ver,
        plan_id=plan_id,
        objective=objective,
        nodes=tuple(sorted(parsed_nodes, key=lambda n: (n.order_index, n.node_id))),
        summary_notes=notes,
        warnings=tuple(warnings),
    )
    return StructuredPlan(
        plan_version=plan.plan_version,
        plan_id=plan.plan_id,
        objective=plan.objective,
        nodes=plan.nodes,
        summary_notes=plan.summary_notes,
        warnings=plan.warnings,
        plan_digest=compute_structured_plan_digest(plan),
    )


def _compute_parallel_batches(graph: TaskGraph) -> List[List[str]]:
    """Group nodes into deterministic parallelizable topological waves."""
    node_map = {n.node_id: n for n in graph.nodes}
    in_degree = {n.node_id: len(n.dependency_ids) for n in graph.nodes}
    dependents: Dict[str, List[str]] = {n.node_id: [] for n in graph.nodes}
    for n in graph.nodes:
        for d in n.dependency_ids:
            dependents[d].append(n.node_id)

    batches: List[List[str]] = []
    ready = sorted((nid for nid, deg in in_degree.items() if deg == 0), key=lambda x: (node_map[x].order_index, x))
    while ready:
        batches.append(list(ready))
        next_ready: List[str] = []
        for cur in ready:
            for dep_n in dependents.get(cur, []):
                in_degree[dep_n] -= 1
                if in_degree[dep_n] == 0:
                    next_ready.append(dep_n)
        ready = sorted(next_ready, key=lambda x: (node_map[x].order_index, x))
    return batches


def format_plan_summary(
    graph: TaskGraph,
    *,
    parallel_batches: Optional[List[List[str]]] = None,
    warnings: Sequence[str] = (),
) -> str:
    """Format a concise human-readable plan review summary with dependencies & parallelism."""
    batches = parallel_batches if parallel_batches is not None else _compute_parallel_batches(graph)
    lines = [
        f"Plan Graph Summary (graph_id={graph.graph_id}, execution_id={graph.execution_id}, max_concurrency={graph.max_concurrency}):",
        f"Objective: {graph.objective}",
        f"Nodes ({len(graph.nodes)}):",
    ]
    for node in sorted(graph.nodes, key=lambda n: (n.order_index, n.node_id)):
        deps_s = ",".join(node.dependency_ids) if node.dependency_ids else "none"
        locks = list(node.conflict_resources or [])
        if node.exclusive_resource and node.exclusive_resource not in locks:
            locks.append(node.exclusive_resource)
        res_s = f" resources=[{','.join(locks)}]" if locks else ""
        ver_s = f" verify={node.verification_requirement}" if node.verification_requirement else ""
        exp_s = f" outcome={node.expected_outcome!r}" if node.expected_outcome else ""
        lines.append(
            f"  - [{node.order_index}] {node.node_id}: {node.description} "
            f"(deps=[{deps_s}], requires_verified_deps={node.requires_verified_dependencies}){exp_s}{ver_s}{res_s}"
        )
    if batches:
        lines.append("Parallel Execution Waves:")
        for w_idx, wave in enumerate(batches):
            tag = "concurrent" if len(wave) > 1 else "single"
            lines.append(f"  Wave {w_idx + 1} ({tag}): {', '.join(wave)}")
    if warnings:
        lines.append("Warnings:")
        for w in warnings:
            lines.append(f"  * {w}")
    return "\n".join(lines)


@dataclass
class CompiledPlanBundle:
    """Validated compilation artifact linking StructuredPlan, TaskGraph, proposals, and review summary."""

    structured_plan: StructuredPlan
    task_graph: TaskGraph
    proposals: List[PendingActionProposal] = field(default_factory=list)
    independent_node_ids: List[str] = field(default_factory=list)
    parallel_batches: List[List[str]] = field(default_factory=list)
    requires_confirmation: bool = False
    confirmed: bool = False
    validation_warnings: List[str] = field(default_factory=list)
    attempts: int = 1
    summary_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "structured_plan": self.structured_plan.to_dict(),
            "task_graph": self.task_graph.to_dict(),
            "proposals": [p.to_checkpoint_dict() for p in self.proposals],
            "independent_node_ids": list(self.independent_node_ids),
            "parallel_batches": [list(b) for b in self.parallel_batches],
            "requires_confirmation": bool(self.requires_confirmation),
            "confirmed": bool(self.confirmed),
            "validation_warnings": list(self.validation_warnings),
            "attempts": int(self.attempts),
            "summary_text": self.summary_text,
        }


def _infer_plan_requires_confirmation(plan: StructuredPlan, proposals: Sequence[PendingActionProposal]) -> bool:
    """Determine conservatively whether the compiled plan requests state-mutating or privileged work."""
    mutating_tools = {
        "execute_command",
        "bash",
        "sh",
        "shell",
        "manage_workspace_file",
        "write_file",
        "delete_file",
        "interact_with_ui",
        "gui_actuate",
        "interact_with_browser",
        "playwright_web",
        "manage_system_process",
    }
    for p in proposals:
        if p.tool_id in mutating_tools:
            if p.tool_id == "manage_workspace_file" and str(p.normalized_arguments.get("action") or "").lower() in (
                "read",
                "list",
                "stat",
                "exists",
                "search",
            ):
                continue
            return True
        if p.external_source:
            return True
    for n in plan.nodes:
        if n.exclusive_resource if hasattr(n, "exclusive_resource") else False:
            return True
        if any(r.startswith(("target:", "exclusive:")) for r in n.target_resources):
            return True
    return False


def compile_structured_plan_to_task_graph(
    plan_or_raw: Union[StructuredPlan, str, Dict[str, Any]],
    *,
    execution_id: str,
    graph_id: Optional[str] = None,
    max_concurrency: int = 1,
    fallback_objective: Optional[str] = None,
    workspace_root: Optional[Union[str, Path]] = None,
    tool_schemas: Optional[List[Dict[str, Any]]] = None,
    tool_registry: Optional[Any] = None,
    require_confirmation: Optional[bool] = None,
    confirmed: bool = False,
) -> CompiledPlanBundle:
    """Validate and convert a StructuredPlan into a verified TaskGraph and PendingActionProposals."""
    if not isinstance(execution_id, str) or not execution_id.strip():
        raise PlanValidationError("compile_structured_plan_to_task_graph requires a non-empty 'execution_id'.")
    eff_exec_id = execution_id.strip()

    if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int) or max_concurrency < 1:
        raise PlanValidationError(f"Trusted max_concurrency must be an integer >= 1, got {max_concurrency!r}.")

    plan = (
        plan_or_raw
        if isinstance(plan_or_raw, StructuredPlan)
        else parse_structured_plan(
            plan_or_raw,
            fallback_objective=fallback_objective,
            workspace_root=workspace_root,
        )
    )

    schema_by_name: Dict[str, Dict[str, Any]] = {}
    if isinstance(tool_schemas, list):
        for s in tool_schemas:
            if not isinstance(s, dict):
                continue
            fn = s.get("function") if isinstance(s.get("function"), dict) else {}
            sname = str(fn.get("name") or s.get("name") or "").strip()
            if sname:
                schema_by_name[sname] = s

    proposals: List[PendingActionProposal] = []
    seen_mutating_fps: Dict[str, str] = {}
    seen_desc_targets: Dict[Tuple[str, Tuple[str, ...]], str] = {}
    graph_nodes: List[TaskNode] = []

    for p_node in plan.nodes:
        prop_ids: List[str] = []
        ext_src: Optional[Dict[str, Any]] = None
        reg_tool_obj = None
        if tool_registry is not None and callable(getattr(tool_registry, "get_tool", None)):
            try:
                reg_tool_obj = tool_registry.get_tool(p_node.tool_hint) if p_node.tool_hint else None
            except Exception:
                reg_tool_obj = None
        if tool_registry is not None and p_node.tool_hint and callable(getattr(tool_registry, "get_external_tool_entry", None)):
            try:
                ext_entry = tool_registry.get_external_tool_entry(p_node.tool_hint)
                if ext_entry is not None:
                    ext_src = {
                        "is_external": True,
                        "source_id": getattr(ext_entry, "source_id", "external"),
                        "source_type": getattr(ext_entry, "source_type", "external"),
                        "external_tool_name": p_node.tool_hint,
                    }
            except Exception:
                ext_src = None

        prelim_node = TaskNode(
            node_id=p_node.node_id,
            description=p_node.description,
            dependency_ids=list(p_node.depends_on),
            requires_verified_dependencies=p_node.requires_verified_dependencies,
            order_index=p_node.order_index,
            external_source=ext_src,
            conflict_resources=list(p_node.target_resources),
            expected_outcome=p_node.expected_outcome,
            verification_requirement=p_node.verification_requirement,
            timeout_seconds=p_node.timeout_seconds,
        )

        # Attach conservative conflict & resource metadata reusing TaskGraph conflict detection
        c_profile = extract_node_conflict_profile(
            tool_name=p_node.tool_hint or "",
            arguments=dict(p_node.tool_arguments),
            node=prelim_node,
            external_source=ext_src,
            tool_schema=schema_by_name.get(p_node.tool_hint or ""),
            tool_obj=reg_tool_obj,
        )

        if not c_profile.read_only:
            if c_profile.operation_fingerprint:
                prev_nid = seen_mutating_fps.get(c_profile.operation_fingerprint)
                if prev_nid is not None:
                    raise PlanValidationError(
                        f"Duplicate side-effect operation detected across nodes '{prev_nid}' and '{p_node.node_id}'."
                    )
                seen_mutating_fps[c_profile.operation_fingerprint] = p_node.node_id
            desc_key = (p_node.description.lower(), c_profile.write_targets or tuple(p_node.target_resources))
            if desc_key[1]:
                prev_desc_nid = seen_desc_targets.get(desc_key)
                if prev_desc_nid is not None:
                    raise PlanValidationError(
                        f"Duplicate side-effect task description/targets across nodes '{prev_desc_nid}' and '{p_node.node_id}'."
                    )
                seen_desc_targets[desc_key] = p_node.node_id

        # Merge normalized conflict locks while preserving any explicit non-empty resource hints
        combined_locks = sorted(
            {
                r
                for r in list(p_node.target_resources) + list(c_profile.lock_keys)
                if r and not r.startswith("op:")
            }
        )
        prelim_node.conflict_resources = combined_locks
        prelim_node.exclusive_resource = c_profile.exclusive_resource

        if p_node.tool_hint:
            pid = f"prop_{p_node.node_id}"
            cid = f"call_{p_node.node_id}"
            dep_props = [f"prop_{d}" for d in p_node.depends_on]
            proposals.append(
                PendingActionProposal(
                    proposal_id=pid,
                    execution_id=eff_exec_id,
                    original_call_id=cid,
                    tool_id=p_node.tool_hint,
                    raw_name=p_node.tool_hint,
                    normalized_arguments=dict(p_node.tool_arguments),
                    proposal_order=p_node.order_index,
                    depends_on=dep_props,
                    external_source=ext_src,
                    status=ProposalStatus.PENDING.value,
                    authorization_status="pending_revalidation",
                )
            )
            prop_ids.append(pid)
            prelim_node.proposal_ids = prop_ids

        graph_nodes.append(prelim_node)

    # Check all dependencies exist BEFORE building edges so errors report clearly
    known_ids = {n.node_id for n in graph_nodes}
    for n in graph_nodes:
        for d in n.dependency_ids:
            if d not in known_ids:
                raise PlanValidationError(f"Plan node '{n.node_id}' references unknown dependency '{d}'.")

    # Build TaskGraph with trusted max_concurrency (never model-provided) and reuse TaskGraph.validate()
    tg = TaskGraph(
        graph_id=graph_id or f"tg_{plan.plan_id}",
        execution_id=eff_exec_id,
        objective=plan.objective,
        nodes=graph_nodes,
        edges=[
            TaskGraphEdge(
                from_node_id=d,
                to_node_id=n.node_id,
                requires_verified=bool(n.requires_verified_dependencies),
            )
            for n in graph_nodes
            for d in n.dependency_ids
        ],
        max_concurrency=max_concurrency,
    )
    tg.validate(known_proposals=proposals or None, allow_blocked=False)
    tg.recompute_schedule(known_proposals=proposals or None)

    # Confirm non-empty meaningful root -> leaf path toward objective
    if not any(not n.dependency_ids for n in tg.nodes):
        raise PlanValidationError("Compiled TaskGraph has no root entry node.")

    batches = _compute_parallel_batches(tg)
    indep_root = batches[0] if batches else []
    needs_conf = bool(require_confirmation) if require_confirmation is not None else _infer_plan_requires_confirmation(plan, proposals)
    summary_txt = format_plan_summary(tg, parallel_batches=batches, warnings=plan.warnings)

    return CompiledPlanBundle(
        structured_plan=plan,
        task_graph=tg,
        proposals=proposals,
        independent_node_ids=indep_root,
        parallel_batches=batches,
        requires_confirmation=needs_conf,
        confirmed=bool(confirmed),
        validation_warnings=list(plan.warnings),
        summary_text=summary_txt,
    )


def build_planner_prompt(
    objective: str,
    *,
    available_tools: Optional[Sequence[str]] = None,
    environment_context: Optional[str] = None,
    validation_error: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build concise system + user messages prompting the model for a strict v1 JSON plan."""
    tools_desc = ", ".join(available_tools) if available_tools else "manage_workspace_file, execute_command, read_file, verify_outcome"
    env_line = f"\nEnvironment:\n{environment_context}\n" if environment_context else ""
    sys_msg = (
        "You are Zelyr's task planner. Decompose the user's objective into a minimal, dependency-aware JSON plan.\n"
        "Output ONLY a single valid JSON object matching schema plan_version=1:\n"
        "{\n"
        '  "plan_version": 1,\n'
        '  "objective": "<concise restatement of objective>",\n'
        '  "nodes": [\n'
        "    {\n"
        '      "node_id": "<short_id>",\n'
        '      "description": "<concrete deliverable action>",\n'
        '      "depends_on": ["<prereq_node_id>"],\n'
        '      "expected_outcome": "<observable post-condition>",\n'
        '      "verification_requirement": "<check:target or concrete check description>",\n'
        '      "requires_verified_dependencies": true,\n'
        '      "target_resources": ["<path_or_resource_if_known>"],\n'
        '      "tool_hint": "<optional tool name>",\n'
        '      "tool_arguments": {},\n'
        '      "timeout_seconds": 30\n'
        "    }\n"
        "  ]\n"
        "}\n"
        f"Available tools: {tools_desc}{env_line}"
        "Rules: separate independent work so it can run concurrently; add depends_on only for real prerequisites; "
        "do not include authorization, ring, or concurrency override fields; do not mark tasks completed."
    )
    msgs: List[Dict[str, Any]] = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": objective},
    ]
    if validation_error:
        msgs.append(
            {
                "role": "user",
                "content": (
                    f"[PLAN VALIDATION ERROR]\n{validation_error}\n"
                    "Fix the plan schema/dependencies and emit ONE valid JSON plan object."
                ),
            }
        )
    return msgs


class PlanCompiler:
    """Bounded model-driven planner compiling natural objectives into validated TaskGraphs."""

    def __init__(self, max_plan_retries: int = 2) -> None:
        if isinstance(max_plan_retries, bool) or not isinstance(max_plan_retries, int) or max_plan_retries < 0:
            raise ValueError(f"max_plan_retries must be an integer >= 0, got {max_plan_retries!r}.")
        self.max_plan_retries = int(max_plan_retries)

    @staticmethod
    async def _collect_stream_text(stream_provider: Callable[[dict], Any], payload: Dict[str, Any]) -> str:
        stream = stream_provider(payload)
        parts: List[str] = []
        if hasattr(stream, "__aiter__"):
            async for chunk in stream:
                if not isinstance(chunk, dict):
                    continue
                for ch in chunk.get("choices") or []:
                    if not isinstance(ch, dict):
                        continue
                    delta = ch.get("delta")
                    if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                        parts.append(delta["content"])
                msg = chunk.get("message")
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    parts.append(msg["content"])
            return "".join(parts).strip()
        if callable(stream) or hasattr(stream, "__await__"):
            res = await stream
            if isinstance(res, str):
                return res.strip()
            if isinstance(res, dict):
                return json.dumps(res)
        if isinstance(stream, str):
            return stream.strip()
        if isinstance(stream, dict):
            return json.dumps(stream)
        return ""

    async def compile_from_objective(
        self,
        objective: str,
        *,
        stream_provider: Callable[[dict], Any],
        execution_id: Optional[str] = None,
        central_ledger: Optional[CentralExecutionLedger] = None,
        turn_ledger: Optional[TurnExecutionLedger] = None,
        max_concurrency: int = 1,
        max_plan_retries: Optional[int] = None,
        model: Optional[str] = None,
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
        tool_registry: Optional[Any] = None,
        workspace_root: Optional[Union[str, Path]] = None,
        environment_context: Optional[str] = None,
        require_confirmation: Optional[bool] = None,
        confirmed: bool = False,
        caller_ring: Optional[int] = None,
        task_local_scoped: bool = False,
        persist_checkpoint: bool = True,
    ) -> CompiledPlanBundle:
        """Generate, validate, bound-retry, and compile a structured plan into a TaskGraph."""
        if not isinstance(objective, str) or not objective.strip():
            raise PlanValidationError("Planning objective cannot be empty.")

        eff_retries = self.max_plan_retries if max_plan_retries is None else int(max_plan_retries)
        if eff_retries < 0:
            raise ValueError(f"max_plan_retries must be >= 0, got {eff_retries!r}.")

        eff_central = central_ledger
        eff_exec_id = (execution_id or "").strip()
        if eff_central is not None:
            if not eff_exec_id:
                rec = eff_central.create_execution(
                    task_prompt=objective.strip(),
                    source_component="plan_compiler",
                    caller_ring=caller_ring,
                    task_local_scoped=task_local_scoped,
                    status=ExecutionStatus.INITIALIZED,
                )
                eff_exec_id = rec.execution_id
            elif eff_central.get_execution(eff_exec_id) is None:
                eff_central.create_execution(
                    task_prompt=objective.strip(),
                    execution_id=eff_exec_id,
                    source_component="plan_compiler",
                    caller_ring=caller_ring,
                    task_local_scoped=task_local_scoped,
                    status=ExecutionStatus.INITIALIZED,
                )
        elif not eff_exec_id:
            eff_exec_id = f"exec_{uuid.uuid4().hex[:12]}"

        avail_names: List[str] = []
        for s in tool_schemas or []:
            if not isinstance(s, dict):
                continue
            fn = s.get("function") if isinstance(s.get("function"), dict) else {}
            nm = str(fn.get("name") or s.get("name") or "").strip()
            if nm and nm not in avail_names:
                avail_names.append(nm)

        last_err: Optional[Exception] = None
        last_err_text: Optional[str] = None
        bundle: Optional[CompiledPlanBundle] = None
        attempts_used = 0

        for attempt_idx in range(eff_retries + 1):
            attempts_used = attempt_idx + 1
            messages = build_planner_prompt(
                objective.strip(),
                available_tools=avail_names or None,
                environment_context=environment_context,
                validation_error=last_err_text,
            )
            req_payload: Dict[str, Any] = {
                "messages": messages,
                "stream": True,
                "options": {"temperature": 0.1},
            }
            if model:
                req_payload["model"] = model

            raw_response = await self._collect_stream_text(stream_provider, req_payload)
            try:
                bundle = compile_structured_plan_to_task_graph(
                    raw_response,
                    execution_id=eff_exec_id,
                    max_concurrency=max_concurrency,
                    fallback_objective=objective.strip(),
                    workspace_root=workspace_root,
                    tool_schemas=tool_schemas,
                    tool_registry=tool_registry,
                    require_confirmation=require_confirmation,
                    confirmed=confirmed,
                )
                bundle.attempts = attempts_used
                break
            except (PlanCompilationError, TaskGraphValidationError, ValueError) as exc:
                last_err = exc
                last_err_text = str(exc)
                if eff_central is not None:
                    eff_central.record_event(
                        eff_exec_id,
                        LedgerEventType.PLANNING_DECISION,
                        status="plan_validation_failed",
                        caller_ring=caller_ring,
                        task_local_scoped=task_local_scoped,
                        metadata={
                            "attempt": attempts_used,
                            "max_attempts": eff_retries + 1,
                            "error": last_err_text[:512],
                        },
                        idempotency_key=f"{eff_exec_id}:plan_invalid:{attempts_used}",
                    )

        if bundle is None:
            raise PlanValidationError(
                f"Failed to compile valid TaskGraph after {attempts_used} attempt(s): {last_err_text}"
            ) from last_err

        # Attach & persist accepted plan cleanly without bypassing execution policy
        eff_turn = turn_ledger or TurnExecutionLedger(
            execution_id=eff_exec_id,
            central_ledger=eff_central,
            caller_ring=caller_ring,
            task_local_scoped=task_local_scoped,
            source_component="plan_compiler",
        )
        eff_turn.attach_task_graph(bundle.task_graph)
        for p in bundle.proposals:
            eff_turn._proposal_meta[p.original_call_id] = p.to_dict()

        if eff_central is not None:
            eff_central.record_event(
                eff_exec_id,
                LedgerEventType.PLANNING_DECISION,
                status="plan_compiled",
                caller_ring=caller_ring,
                task_local_scoped=task_local_scoped,
                metadata={
                    "plan_id": bundle.structured_plan.plan_id,
                    "plan_version": bundle.structured_plan.plan_version,
                    "plan_digest": bundle.structured_plan.plan_digest,
                    "graph_id": bundle.task_graph.graph_id,
                    "node_count": len(bundle.task_graph.nodes),
                    "parallel_waves": len(bundle.parallel_batches),
                    "attempts": bundle.attempts,
                    "requires_confirmation": bundle.requires_confirmation,
                    "confirmed": bundle.confirmed,
                    "max_concurrency": bundle.task_graph.max_concurrency,
                },
                idempotency_key=f"{eff_exec_id}:plan_compiled:{bundle.structured_plan.plan_digest[:16]}",
            )
            if persist_checkpoint:
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
                        "plan_confirmed": bundle.confirmed,
                        "requires_confirmation": bundle.requires_confirmation,
                        "max_concurrency": bundle.task_graph.max_concurrency,
                        "model": model,
                    },
                    pending_actions=bundle.proposals,
                    task_graph=bundle.task_graph,
                    emit_event=False,
                )

        return bundle


def replan_task_graph(
    graph: TaskGraph,
    replan_payload: Union[str, Dict[str, Any]],
    *,
    central_ledger: Optional[CentralExecutionLedger] = None,
    turn_ledger: Optional[TurnExecutionLedger] = None,
    known_proposals: Optional[Union[Dict[str, Any], Iterable[PendingActionProposal]]] = None,
    workspace_root: Optional[Union[str, Path]] = None,
    tool_schemas: Optional[List[Dict[str, Any]]] = None,
    tool_registry: Optional[Any] = None,
    checkpoint: bool = True,
) -> TaskGraph:
    """Safely mutate an existing TaskGraph from a structured replan revision.

    Invariants enforced:
    - Preserves original execution_id, objective, max_concurrency, and historical ledger events.
    - Preserves ALL completed_verified nodes verbatim (refusing removal, reset, or modification).
    - Preserves existing valid dependencies (refusing silent dependency removal).
    - Rolls back atomically on cycles, missing dependencies, duplicate IDs, or cross-exec violations.
    """
    raw = extract_plan_json(replan_payload) if isinstance(replan_payload, str) else dict(replan_payload)
    if not isinstance(raw, dict):
        raise PlanValidationError("Replan payload must be a JSON object.")

    warnings: List[str] = []
    clean = _strip_security_overrides(raw, scope_label="replan", warnings=warnings)
    ws_path: Optional[Path] = Path(workspace_root) if workspace_root is not None else None

    if "plan_version" in clean and (
        isinstance(clean["plan_version"], bool)
        or not isinstance(clean["plan_version"], int)
        or clean["plan_version"] not in SUPPORTED_PLAN_VERSIONS
    ):
        raise UnsupportedPlanVersionError(
            f"Unsupported replan schema version: {clean.get('plan_version')!r}."
        )

    if "execution_id" in clean and str(clean["execution_id"]).strip() != graph.execution_id:
        raise PlanValidationError(
            f"Replan cannot change execution_id from '{graph.execution_id}' to '{clean['execution_id']}'."
        )

    existing_props_list: List[Any] = []
    if isinstance(known_proposals, dict):
        existing_props_list = list(known_proposals.values())
    elif known_proposals is not None:
        existing_props_list = list(known_proposals)
    elif turn_ledger is not None and hasattr(turn_ledger, "get_tracked_proposals"):
        existing_props_list = list(turn_ledger.get_tracked_proposals())

    prop_map: Dict[str, Any] = {}
    newly_created_proposals: List[PendingActionProposal] = []
    for item in existing_props_list:
        pid = getattr(item, "proposal_id", None) or (item.get("proposal_id") if isinstance(item, dict) else None)
        cid = getattr(item, "original_call_id", None) or (item.get("original_call_id") if isinstance(item, dict) else None)
        if pid:
            prop_map[str(pid)] = item
        if cid:
            prop_map[str(cid)] = item

    def _build_node_from_entry(entry: Dict[str, Any], default_order: int = 0) -> TaskNode:
        nc = _strip_security_overrides(entry, scope_label="replan.node", warnings=warnings)
        nid = _normalize_node_id(nc.get("node_id"), field_label="node_id")
        desc = _sanitize_plan_text(str(nc.get("description") or ""), max_bytes=512)
        if not desc or _is_vague_description(desc):
            raise PlanValidationError(f"Replan node '{nid}' must define a concrete description.")
        exp_out = _sanitize_plan_text(str(nc.get("expected_outcome") or ""), max_bytes=512)
        ver_req = _format_verification_requirement(nc.get("verification_requirement"))
        timeout_s = _validate_timeout_value(nc.get("timeout_seconds"))
        raw_deps = nc.get("depends_on", []) or []
        if not isinstance(raw_deps, list):
            raise PlanValidationError(f"Replan node '{nid}' depends_on must be a list.")
        deps = [_normalize_node_id(d, field_label="depends_on entry") for d in raw_deps]
        req_ver = nc.get("requires_verified_dependencies", True)
        if not isinstance(req_ver, bool):
            raise PlanValidationError(f"Replan node '{nid}' requires_verified_dependencies must be boolean.")
        raw_res = nc.get("target_resources", []) or []
        if not isinstance(raw_res, list):
            raise PlanValidationError(f"Replan node '{nid}' target_resources must be a list.")
        norm_res = [_resolve_resource_within_workspace(str(r), ws_path) for r in raw_res]

        tool_hint = nc.get("tool_hint")
        tool_args = nc.get("tool_arguments") or {}
        if not isinstance(tool_args, dict):
            raise PlanValidationError(f"Replan node '{nid}' tool_arguments must be a JSON object.")
        if find_reserved_authorization_key(tool_args) is not None:
            raise PlanValidationError(f"Replan node '{nid}' tool_arguments contains forbidden authorization keys.")
        clean_args, _, _ = sanitize_ledger_payload(
            _strip_security_overrides(tool_args, scope_label=f"replan.node[{nid}].tool_arguments", warnings=warnings),
            max_bytes=2048,
        )
        for p_key in ("path", "file_path", "filepath", "target", "dest", "destination"):
            p_val = clean_args.get(p_key)
            if isinstance(p_val, str) and p_val.strip():
                if ws_path is not None or ".." in Path(p_val.strip()).parts:
                    _resolve_resource_within_workspace(p_val, ws_path)

        pids: List[str] = []
        if isinstance(tool_hint, str) and tool_hint.strip():
            pid = f"prop_{nid}"
            cid = f"call_{nid}"
            new_prop = PendingActionProposal(
                proposal_id=pid,
                execution_id=graph.execution_id,
                original_call_id=cid,
                tool_id=tool_hint.strip(),
                raw_name=tool_hint.strip(),
                normalized_arguments=dict(clean_args),
                proposal_order=int(nc.get("order_index", default_order) or default_order),
                depends_on=[f"prop_{d}" for d in deps],
                status=ProposalStatus.PENDING.value,
                authorization_status="pending_revalidation",
            )
            prop_map[pid] = new_prop
            prop_map[cid] = new_prop
            newly_created_proposals.append(new_prop)
            pids.append(pid)

        prelim = TaskNode(
            node_id=nid,
            description=desc,
            proposal_ids=pids,
            dependency_ids=deps,
            requires_verified_dependencies=req_ver,
            order_index=int(nc.get("order_index", default_order) or default_order),
            conflict_resources=norm_res,
            expected_outcome=exp_out,
            verification_requirement=ver_req,
            timeout_seconds=timeout_s,
        )
        prof = extract_node_conflict_profile(
            tool_name=str(tool_hint or ""),
            arguments=clean_args,
            node=prelim,
        )
        prelim.conflict_resources = sorted(
            {r for r in list(norm_res) + list(prof.lock_keys) if r and not r.startswith("op:")}
        )
        prelim.exclusive_resource = prof.exclusive_resource
        return prelim

    # Accept either explicit patch actions (replace_nodes / add_nodes / cancel_nodes)
    # OR a full revised v1 plan whose verified nodes MUST remain untouched.
    before_snapshot = graph.to_dict()

    def _apply_replan(g: TaskGraph) -> None:
        if "nodes" in clean and isinstance(clean["nodes"], list) and "plan_version" in clean:
            # Revised full plan mode: must preserve verified nodes & never silently drop dependencies
            rev_plan = parse_structured_plan(
                {"plan_version": clean["plan_version"], "objective": clean.get("objective") or g.objective, "nodes": clean["nodes"]},
                fallback_objective=g.objective,
                workspace_root=ws_path,
            )
            existing_by_id = {n.node_id: n for n in g.nodes}
            rev_ids = {rn.node_id for rn in rev_plan.nodes}
            for nid, old_n in existing_by_id.items():
                if old_n.execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value:
                    if nid not in rev_ids:
                        raise PlanValidationError(f"Replan cannot drop completed verified node '{nid}'.")
            max_ord = max([n.order_index for n in g.nodes] + [-1])
            for rn in rev_plan.nodes:
                old_n = existing_by_id.get(rn.node_id)
                if old_n is not None:
                    if old_n.execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value:
                        if list(rn.depends_on) != list(old_n.dependency_ids):
                            raise PlanValidationError(
                                f"Replan cannot alter dependencies of completed verified node '{rn.node_id}'."
                            )
                        continue  # Keep completed verified node untouched verbatim
                    if not set(old_n.dependency_ids).issubset(set(rn.depends_on)):
                        missing_orig = sorted(set(old_n.dependency_ids) - set(rn.depends_on))
                        raise PlanValidationError(
                            f"Replan cannot silently remove existing prerequisites {missing_orig} from node '{rn.node_id}'."
                        )
                    new_tn = _build_node_from_entry(rn.to_dict(), default_order=old_n.order_index)
                    g.replace_failed_node(rn.node_id, new_tn, rewire_dependents=False, known_proposals=prop_map)
                else:
                    max_ord += 1
                    new_tn = _build_node_from_entry(rn.to_dict(), default_order=max_ord)
                    g.add_node(new_tn, known_proposals=prop_map)
            return

        # Patch mode
        for cancel_id in clean.get("cancel_nodes") or []:
            nid = _normalize_node_id(cancel_id, field_label="cancel_nodes entry")
            cur = g.get_node(nid)
            if cur is not None and cur.execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value:
                raise PlanValidationError(f"Replan cannot cancel completed verified node '{nid}'.")
            g.cancel_node(nid, reason="Superseded by replan", known_proposals=prop_map)

        for rep in clean.get("replace_nodes") or []:
            if not isinstance(rep, dict):
                raise PlanValidationError("replace_nodes entries must be dicts.")
            target_id = _normalize_node_id(rep.get("target_node_id") or rep.get("node_id"), field_label="target_node_id")
            old_n = g.get_node(target_id)
            if old_n is None:
                raise PlanValidationError(f"Cannot replace unknown node '{target_id}'.")
            if old_n.execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value:
                raise PlanValidationError(f"Replan cannot replace completed verified node '{target_id}'.")
            new_spec = dict(rep.get("replacement") or rep)
            new_tn = _build_node_from_entry(new_spec, default_order=old_n.order_index)
            if old_n.dependency_ids and not set(old_n.dependency_ids).issubset(set(new_tn.dependency_ids)):
                # Only allow preserving prerequisites, never silently dropping prerequisites
                for req_d in old_n.dependency_ids:
                    if req_d not in new_tn.dependency_ids:
                        raise PlanValidationError(
                            f"Replan replacement for '{target_id}' cannot drop existing prerequisite '{req_d}'."
                        )
            g.replace_failed_node(
                target_id,
                new_tn,
                rewire_dependents=bool(rep.get("rewire_dependents", True)),
                known_proposals=prop_map,
            )

        next_ord = max([n.order_index for n in g.nodes] + [-1]) + 1
        for add_entry in clean.get("add_nodes") or []:
            if not isinstance(add_entry, dict):
                raise PlanValidationError("add_nodes entries must be dicts.")
            new_tn = _build_node_from_entry(add_entry, default_order=next_ord)
            next_ord += 1
            g.add_node(new_tn, known_proposals=prop_map)

        for dep_upd in clean.get("add_dependencies") or []:
            if not isinstance(dep_upd, dict):
                raise PlanValidationError("add_dependencies entries must be dicts.")
            nid = _normalize_node_id(dep_upd.get("node_id"), field_label="node_id")
            cur = g.get_node(nid)
            if cur is None:
                raise PlanValidationError(f"Cannot add dependencies to unknown node '{nid}'.")
            extra_deps = [_normalize_node_id(d, field_label="dependency_id") for d in (dep_upd.get("depends_on") or [])]
            merged_deps = list(cur.dependency_ids)
            for d in extra_deps:
                if d not in merged_deps:
                    merged_deps.append(d)
            g.update_dependencies(nid, merged_deps, known_proposals=prop_map)

    try:
        if central_ledger is not None:
            # First validate mutation in-memory before touching turn_ledger or central_ledger
            _apply_replan(graph)
            graph.recompute_schedule(known_proposals=prop_map)
            graph.validate(known_proposals=prop_map, allow_blocked=True)
            if turn_ledger is not None:
                for np in newly_created_proposals:
                    turn_ledger._proposal_meta[np.original_call_id] = np.to_dict()
            graph._restore_from_dict(before_snapshot)
            return CheckpointManager.mutate_task_graph(
                central_ledger,
                graph.execution_id,
                graph,
                mutation_type="plan_replan",
                mutation_fn=_apply_replan,
                turn_ledger=turn_ledger,
                known_proposals=prop_map,
                checkpoint=checkpoint,
            )
        _apply_replan(graph)
        graph.recompute_schedule(known_proposals=prop_map)
        graph.validate(known_proposals=prop_map, allow_blocked=True)
        if turn_ledger is not None:
            for np in newly_created_proposals:
                turn_ledger._proposal_meta[np.original_call_id] = np.to_dict()
        return graph
    except Exception:
        graph._restore_from_dict(before_snapshot)
        raise
