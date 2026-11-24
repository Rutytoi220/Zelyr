"""Dependency-aware task graph model, DAG validation, scheduling, and recovery for Zelyr 1.4."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import uuid

from zelyr.engine.execution_ledger import (
    ActionRecord,
    ActionState,
    CorruptedCheckpointError,
    ExecutionEvent,
    ExecutionRecord,
    ExecutionStatus,
    LedgerEventType,
    LedgerPersistenceError,
    UnsupportedCheckpointVersionError,
    VerificationStatus,
    sanitize_ledger_payload,
    sanitize_ledger_text,
)

TASK_GRAPH_VERSION: int = 1
SUPPORTED_TASK_GRAPH_VERSIONS: frozenset[int] = frozenset({1})
RECOVERY_STATE_VERSION: int = 1
SUPPORTED_RECOVERY_VERSIONS: frozenset[int] = frozenset({1})
_VALID_RECOVERY_DECISIONS: frozenset[str] = frozenset(
    {
        "retry_safe",
        "retry_after_revalidation",
        "replan_node",
        "require_confirmation",
        "blocked",
        "already_completed",
        "unresolved",
    }
)

_FORBIDDEN_GRAPH_KEYS: frozenset[str] = frozenset(
    {
        "api_key",
        "token",
        "secret",
        "password",
        "authorization",
        "credentials",
        "headers",
        "caller_ring_grant",
        "allowed_tools_grant",
        "_caller_ring",
        "_allowed_tools",
        "caller_ring",
        "allowed_tools",
        "task_local_scope",
        "auth_ctx",
        "auth_context",
        "permissions",
        "privileges",
        "bypass_auth",
        "bypass_confirmation",
        "skip_authorization",
        "skip_verification",
        "unrestricted_tools",
        "workspace_root",
        "filesystem_root",
        "sandbox_root",
        "bypass_recovery",
        "skip_recovery_verification",
        "override_retry_limit",
        "force_retry",
    }
)


def _validate_node_recovery_state(raw: Any, *, context_label: str = "recovery_state") -> Dict[str, Any]:
    """Validate and sanitize persisted TaskNode / checkpoint recovery state cleanly (fail-closed)."""
    if not isinstance(raw, dict):
        raise CorruptedTaskGraphError(f"{context_label} must be a JSON object.")

    rver = raw.get("recovery_version", RECOVERY_STATE_VERSION)
    if isinstance(rver, bool) or not isinstance(rver, int):
        raise CorruptedTaskGraphError(f"{context_label} has malformed recovery_version: {rver!r}.")
    if rver not in SUPPORTED_RECOVERY_VERSIONS:
        raise UnsupportedTaskGraphVersionError(
            f"{context_label} uses unsupported recovery_version {rver}; supported: {sorted(SUPPORTED_RECOVERY_VERSIONS)}."
        )

    attempt_count = raw.get("attempt_count", 0)
    if isinstance(attempt_count, bool) or not isinstance(attempt_count, int) or attempt_count < 0:
        raise CorruptedTaskGraphError(f"{context_label} has invalid attempt_count: {attempt_count!r}.")

    max_attempts = raw.get("max_attempts")
    if max_attempts is not None and (
        isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 0
    ):
        raise CorruptedTaskGraphError(f"{context_label} has invalid max_attempts: {max_attempts!r}.")

    for ts_field in ("retry_deadline", "next_retry_after"):
        val = raw.get(ts_field)
        if val is not None and (isinstance(val, bool) or not isinstance(val, (int, float)) or float(val) < 0.0):
            raise CorruptedTaskGraphError(f"{context_label} has invalid {ts_field}: {val!r}.")

    for bool_field in (
        "may_have_side_effect",
        "execution_started",
        "reconciliation_required",
        "verified_outcome",
    ):
        bval = raw.get(bool_field)
        if bval is not None and not isinstance(bval, bool):
            raise CorruptedTaskGraphError(f"{context_label} field '{bool_field}' must be boolean.")

    decision = raw.get("recovery_decision")
    if decision is not None and (not isinstance(decision, str) or decision not in _VALID_RECOVERY_DECISIONS):
        raise CorruptedTaskGraphError(f"{context_label} has invalid recovery_decision: {decision!r}.")

    history = raw.get("recovery_history", [])
    if not isinstance(history, list) or any(not isinstance(h, dict) for h in history):
        raise CorruptedTaskGraphError(f"{context_label} recovery_history must be a list of dicts.")

    forbidden_found = [k for k in raw.keys() if k in _FORBIDDEN_GRAPH_KEYS]
    if forbidden_found:
        raise CorruptedTaskGraphError(
            f"{context_label} contains forbidden policy/auth override keys: {sorted(forbidden_found)}."
        )

    safe_map, _, _ = sanitize_ledger_payload(_strip_forbidden_graph_keys(dict(raw)), max_bytes=4096)
    out = dict(safe_map)
    out["recovery_version"] = int(rver)
    out["attempt_count"] = int(attempt_count)
    out["recovery_history"] = list(history[-24:])
    return out


class TaskGraphError(LedgerPersistenceError):
    """Base error for task graph validation, corruption, and execution continuity."""


class TaskGraphValidationError(TaskGraphError):
    """Raised when a task graph fails deterministic DAG, reference, state, or isolation checks."""


class CorruptedTaskGraphError(CorruptedCheckpointError):
    """Raised when persisted task graph data is malformed or fails integrity verification."""


class UnsupportedTaskGraphVersionError(UnsupportedCheckpointVersionError):
    """Raised when a persisted task graph uses an unsupported schema version."""


class TaskNodeStatus(str, Enum):
    """Execution and readiness status for a single node in a TaskGraph."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED_VERIFIED = "completed_verified"
    COMPLETED_UNVERIFIED = "completed_unverified"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class TaskGraphStatus(str, Enum):
    """Aggregate lifecycle state of a TaskGraph."""

    INITIALIZED = "initialized"
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED_VERIFIED = "completed_verified"
    COMPLETED_UNVERIFIED = "completed_unverified"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    INVALID = "invalid"


import threading as _threading

_VALID_NODE_STATUSES: frozenset[str] = frozenset(s.value for s in TaskNodeStatus)
_VALID_GRAPH_STATUSES: frozenset[str] = frozenset(s.value for s in TaskGraphStatus)
_VALID_VERIFICATION_STATUSES: frozenset[str] = frozenset(
    {
        VerificationStatus.NOT_VERIFIED.value,
        VerificationStatus.VERIFIED.value,
        VerificationStatus.VERIFICATION_FAILED.value,
        VerificationStatus.CLAIMED_UNVERIFIED.value,
        VerificationStatus.INTERRUPTED.value,
        "contradictory",
    }
)


def _strip_forbidden_graph_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            str(k): _strip_forbidden_graph_keys(v)
            for k, v in obj.items()
            if str(k).lower() not in _FORBIDDEN_GRAPH_KEYS
        }
    if isinstance(obj, list):
        return [_strip_forbidden_graph_keys(i) for i in obj]
    return obj


def _sanitize_graph_text(raw: Any, *, max_bytes: int = 512) -> str:
    if not raw:
        return ""
    cleaned, _, _ = sanitize_ledger_text(str(raw), max_bytes=max_bytes, is_external=True, strip_injection=True)
    return (cleaned or "").strip()


def compute_task_graph_digest(payload: Dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 integrity digest over core TaskGraph fields."""
    canonical = {
        "graph_version": int(payload.get("graph_version", TASK_GRAPH_VERSION)),
        "graph_id": str(payload.get("graph_id") or ""),
        "execution_id": str(payload.get("execution_id") or ""),
        "objective": str(payload.get("objective") or ""),
        "status": str(payload.get("status") or ""),
        "nodes": [
            {
                "node_id": str(n.get("node_id") or ""),
                "description": str(n.get("description") or ""),
                "proposal_ids": list(n.get("proposal_ids") or []),
                "dependency_ids": list(n.get("dependency_ids") or []),
                "execution_status": str(n.get("execution_status") or ""),
                "verification_status": str(n.get("verification_status") or ""),
                "failure_info": n.get("failure_info"),
                "observation_refs": list(n.get("observation_refs") or []),
                "requires_verified_dependencies": bool(n.get("requires_verified_dependencies", True)),
                "contradictory": bool(n.get("contradictory", False)),
                "recoverable": n.get("recoverable"),
                "blocked_reason": n.get("blocked_reason"),
                "order_index": int(n.get("order_index", 0)),
                "external_source": n.get("external_source"),
            }
            for n in (payload.get("nodes") or [])
            if isinstance(n, dict)
        ],
        "edges": [
            {
                "from_node_id": str(e.get("from_node_id") or ""),
                "to_node_id": str(e.get("to_node_id") or ""),
                "requires_verified": bool(e.get("requires_verified", True)),
                "edge_type": str(e.get("edge_type") or "depends_on"),
            }
            for e in (payload.get("edges") or [])
            if isinstance(e, dict)
        ],
        "summary": payload.get("summary") or {},
    }
    raw_mc = payload.get("max_concurrency", 1)
    if isinstance(raw_mc, int) and not isinstance(raw_mc, bool) and raw_mc != 1:
        canonical["max_concurrency"] = raw_mc
    if payload.get("active_reservations"):
        canonical["active_reservations"] = payload.get("active_reservations")
    node_res_meta: Dict[str, Dict[str, Any]] = {}
    for n in payload.get("nodes") or []:
        if not isinstance(n, dict):
            continue
        nid = str(n.get("node_id") or "")
        extra_n = {}
        for k in (
            "reserved_by",
            "reservation_id",
            "reserved_at",
            "conflict_resources",
            "exclusive_resource",
            "expected_outcome",
            "verification_requirement",
            "timeout_seconds",
            "recovery_state",
        ):
            val = n.get(k)
            if val is not None and val != "" and val != [] and val != {}:
                extra_n[k] = val
        if nid and extra_n:
            node_res_meta[nid] = extra_n
    if node_res_meta:
        canonical["node_reservations"] = node_res_meta
    raw = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode(
        "utf-8", errors="replace"
    )
    return hashlib.sha256(raw).hexdigest()


class TaskGraphConflictError(TaskGraphValidationError):
    """Raised when a TaskGraph node reservation or concurrent schedule conflicts on a resource."""


@dataclass(frozen=True)
class NodeConflictProfile:
    """Conservative resource and side-effect conflict footprint for a TaskNode."""

    lock_keys: Tuple[str, ...] = ()
    read_targets: Tuple[str, ...] = ()
    write_targets: Tuple[str, ...] = ()
    exclusive_resource: Optional[str] = None
    operation_fingerprint: Optional[str] = None
    read_only: bool = False


_READ_ONLY_TOOLS = frozenset(
    {
        "read_file",
        "list_files",
        "search_files",
        "system_info",
        "inspect_network",
        "query_system_journal",
        "verify_outcome",
        "echo",
    }
)
_READ_FILE_ACTIONS = frozenset({"read", "list", "stat", "exists", "search", "inspect", "grep", "glob"})
_FILE_MUTATION_TOOLS = frozenset(
    {
        "manage_workspace_file",
        "write_file",
        "edit_file",
        "patch_file",
        "delete_file",
        "create_file",
        "append_file",
        "file_write",
        "file_tool",
    }
)
_GUI_EXCLUSIVE_TOOLS = frozenset(
    {"interact_with_ui", "gui_actuate", "gui_inspect", "take_screenshot", "mouse_click", "type_text"}
)
_BROWSER_EXCLUSIVE_TOOLS = frozenset({"interact_with_browser", "playwright_web"})
_HOST_IPC_EXCLUSIVE_TOOLS = frozenset(
    {"manage_desktop_window", "manage_system_clipboard", "manage_media_playback", "manage_system_process"}
)
_SHELL_EXCLUSIVE_TOOLS = frozenset(
    {"execute_command", "bash", "sh", "shell", "shell_command", "run_command", "python_exec", "exec_code"}
)
_TARGET_ARG_KEYS = (
    "path",
    "file_path",
    "filepath",
    "target",
    "filename",
    "dest",
    "destination",
    "dest_path",
    "new_path",
    "output_path",
    "src",
    "source",
    "url",
    "resource",
    "item_id",
    "bucket",
    "table",
    "database",
)


def _canon_conflict_path(raw: Any) -> Optional[str]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    from zelyr.engine.execution_ledger import canonicalize_target

    return canonicalize_target(raw.strip(), check_type="file_exists") or raw.strip()


def extract_node_conflict_profile(
    tool_name: str = "",
    arguments: Optional[Dict[str, Any]] = None,
    *,
    node: Optional["TaskNode"] = None,
    external_source: Optional[Dict[str, Any]] = None,
    tool_schema: Optional[Dict[str, Any]] = None,
    tool_obj: Optional[Any] = None,
) -> NodeConflictProfile:
    """Compute conservative conflict locks and targets for a TaskNode + proposed tool invocation."""
    from zelyr.engine.execution_ledger import _canonicalize_args, _extract_action_target

    tname = str(tool_name or "").strip()
    args = dict(arguments) if isinstance(arguments, dict) else {}
    lock_keys: Set[str] = set()
    read_targets: Set[str] = set()
    write_targets: Set[str] = set()
    excl: Optional[str] = None

    if node is not None:
        if node.exclusive_resource:
            excl = str(node.exclusive_resource).strip()
        for kr in node.conflict_resources or ():
            k_str = str(kr).strip()
            if k_str:
                lock_keys.add(k_str)

    # Schema / object read_only markers
    schema_ro = False
    if isinstance(tool_schema, dict):
        fn_meta = tool_schema.get("function") if isinstance(tool_schema.get("function"), dict) else tool_schema
        schema_ro = bool(tool_schema.get("read_only") is True or fn_meta.get("read_only") is True)
    obj_ro = bool(getattr(tool_obj, "read_only", False) is True) if tool_obj is not None else False

    action_val = str(args.get("action") or args.get("operation") or "").strip().lower()
    is_read_only = bool(
        tname in _READ_ONLY_TOOLS
        or (tname == "manage_workspace_file" and action_val in _READ_FILE_ACTIONS)
        or (schema_ro or obj_ro)
    )
    if tname in _GUI_EXCLUSIVE_TOOLS | _BROWSER_EXCLUSIVE_TOOLS | _HOST_IPC_EXCLUSIVE_TOOLS | _SHELL_EXCLUSIVE_TOOLS:
        is_read_only = False

    # Collect explicit targets from action extractor and target keys
    extracted_targets: List[str] = []
    primary_t = _extract_action_target(tname, args) if tname else None
    if primary_t:
        cp = _canon_conflict_path(primary_t)
        if cp:
            extracted_targets.append(cp)
    for k in _TARGET_ARG_KEYS:
        val = args.get(k)
        if isinstance(val, str) and val.strip():
            cp = _canon_conflict_path(val)
            if cp and cp not in extracted_targets:
                extracted_targets.append(cp)

    if tname in _GUI_EXCLUSIVE_TOOLS:
        excl = excl or "exclusive:desktop_gui"
    elif tname in _BROWSER_EXCLUSIVE_TOOLS:
        excl = excl or "exclusive:browser_session"
    elif tname in _HOST_IPC_EXCLUSIVE_TOOLS:
        excl = excl or "exclusive:host_desktop_ipc"
    elif tname in _SHELL_EXCLUSIVE_TOOLS:
        excl = excl or "exclusive:opaque_shell"
        for et in extracted_targets:
            write_targets.add(et)
            lock_keys.add(f"target:{et}")
    elif tname == "manage_workspace_file" or tname in _FILE_MUTATION_TOOLS:
        if is_read_only:
            for et in extracted_targets:
                read_targets.add(et)
        elif extracted_targets:
            for et in extracted_targets:
                write_targets.add(et)
                lock_keys.add(f"target:{et}")
        else:
            excl = excl or "exclusive:unscoped_filesystem"
    elif tname:
        eff_ext = external_source if isinstance(external_source, dict) else (node.external_source if node else None)
        is_ext = bool(eff_ext and eff_ext.get("is_external", True))
        if is_read_only:
            for et in extracted_targets:
                read_targets.add(et)
        elif extracted_targets or lock_keys:
            for et in extracted_targets:
                lock_keys.add(f"target:{et}")
                if "/" in et or "\\" in et or "." in et or tname != "echo":
                    write_targets.add(et)
            if is_ext and not extracted_targets:
                src_id = str((eff_ext or {}).get("source_id") or "external")
                excl = excl or f"exclusive:external:{src_id}"
        else:
            # Tools whose conflict footprint cannot be reliably disproved serialize conservatively
            if is_ext:
                src_id = str((eff_ext or {}).get("source_id") or "external")
                excl = excl or f"exclusive:external:{src_id}"
            elif not args and not node_has_explicit_resources(node):
                excl = excl or "exclusive:opaque_side_effect"
            elif args:
                # Non-target scalar arguments on an unknown mutator: lock opaque side-effect unless read_only
                excl = excl or "exclusive:opaque_side_effect"

    if excl:
        lock_keys.add(excl)

    op_fp: Optional[str] = None
    if tname and not is_read_only:
        canon_args = _canonicalize_args(args)
        fp_digest = hashlib.sha256(f"{tname}:{canon_args}".encode("utf-8", errors="replace")).hexdigest()[:24]
        op_fp = f"op:{tname}:{fp_digest}"
        lock_keys.add(op_fp)

    return NodeConflictProfile(
        lock_keys=tuple(sorted(lock_keys)),
        read_targets=tuple(sorted(read_targets)),
        write_targets=tuple(sorted(write_targets)),
        exclusive_resource=excl,
        operation_fingerprint=op_fp,
        read_only=is_read_only,
    )


def node_has_explicit_resources(node: Optional["TaskNode"]) -> bool:
    return bool(node is not None and (node.conflict_resources or node.exclusive_resource))


def profiles_conflict(p1: NodeConflictProfile, p2: NodeConflictProfile) -> Optional[str]:
    """Return a human-readable reason if two conflict profiles cannot safely run concurrently."""
    from zelyr.engine.execution_ledger import targets_match

    if p1.operation_fingerprint and p1.operation_fingerprint == p2.operation_fingerprint and not (p1.read_only and p2.read_only):
        return f"duplicate or identical mutating operation '{p1.operation_fingerprint}'"

    if p1.exclusive_resource and p2.exclusive_resource:
        if p1.exclusive_resource == p2.exclusive_resource:
            return f"shared exclusive resource '{p1.exclusive_resource}'"

    for excl_p, other_p in ((p1, p2), (p2, p1)):
        if excl_p.exclusive_resource in ("exclusive:opaque_shell", "exclusive:opaque_side_effect", "exclusive:unscoped_filesystem"):
            if not other_p.read_only or other_p.read_targets:
                return f"conservative serialization for '{excl_p.exclusive_resource}'"

    shared_locks = set(p1.lock_keys) & set(p2.lock_keys)
    if shared_locks:
        return f"overlapping lock resource '{sorted(shared_locks)[0]}'"

    def _targets_overlap(a: str, b: str) -> bool:
        return a == b or targets_match(a, b, allow_dir_containment=True) or targets_match(b, a, allow_dir_containment=True)

    for w1 in p1.write_targets:
        for w2 in p2.write_targets:
            if _targets_overlap(w1, w2):
                return f"conflicting write targets '{w1}' and '{w2}'"
        for r2 in p2.read_targets:
            if _targets_overlap(w1, r2):
                return f"read/write hazard between write '{w1}' and read '{r2}'"

    for w2 in p2.write_targets:
        for r1 in p1.read_targets:
            if _targets_overlap(w2, r1):
                return f"read/write hazard between read '{r1}' and write '{w2}'"

    return None


@dataclass
class TaskGraphEdge:
    """Explicit directed prerequisite edge (from_node_id -> to_node_id)."""

    from_node_id: str
    to_node_id: str
    requires_verified: bool = True
    edge_type: str = "depends_on"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_node_id": self.from_node_id,
            "to_node_id": self.to_node_id,
            "requires_verified": bool(self.requires_verified),
            "edge_type": self.edge_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskGraphEdge":
        if not isinstance(data, dict):
            raise CorruptedTaskGraphError("TaskGraph edge entry must be a JSON object.")
        src = data.get("from_node_id")
        dst = data.get("to_node_id")
        if not isinstance(src, str) or not src.strip() or not isinstance(dst, str) or not dst.strip():
            raise CorruptedTaskGraphError("TaskGraph edge missing non-empty 'from_node_id' or 'to_node_id'.")
        req_ver = data.get("requires_verified", True)
        if not isinstance(req_ver, bool):
            raise CorruptedTaskGraphError("TaskGraph edge 'requires_verified' must be boolean.")
        return cls(
            from_node_id=src.strip(),
            to_node_id=dst.strip(),
            requires_verified=req_ver,
            edge_type=str(data.get("edge_type") or "depends_on"),
        )


@dataclass
class TaskNode:
    """Versioned task graph node referencing existing PendingActionProposal IDs and ledger evidence."""

    node_id: str
    description: str = ""
    proposal_ids: List[str] = field(default_factory=list)
    dependency_ids: List[str] = field(default_factory=list)
    execution_status: str = TaskNodeStatus.PENDING.value
    verification_status: str = VerificationStatus.NOT_VERIFIED.value
    failure_info: Optional[Dict[str, Any]] = None
    observation_refs: List[Dict[str, Any]] = field(default_factory=list)
    requires_verified_dependencies: bool = True
    contradictory: bool = False
    recoverable: Optional[bool] = None
    blocked_reason: Optional[str] = None
    order_index: int = 0
    external_source: Optional[Dict[str, Any]] = None
    reserved_by: Optional[str] = None
    reservation_id: Optional[str] = None
    reserved_at: Optional[float] = None
    conflict_resources: List[str] = field(default_factory=list)
    exclusive_resource: Optional[str] = None
    expected_outcome: str = ""
    verification_requirement: Optional[str] = None
    timeout_seconds: Optional[float] = None
    recovery_state: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        safe_fail, _, _ = sanitize_ledger_payload(_strip_forbidden_graph_keys(self.failure_info), max_bytes=1024) if self.failure_info else ({}, False, False)
        safe_obs_wrap, _, _ = sanitize_ledger_payload({"refs": list(self.observation_refs)[-16:]}, max_bytes=4096)
        safe_obs = list(safe_obs_wrap.get("refs") or []) if isinstance(safe_obs_wrap.get("refs"), list) else []
        out: Dict[str, Any] = {
            "node_id": self.node_id,
            "description": _sanitize_graph_text(self.description, max_bytes=512),
            "proposal_ids": list(self.proposal_ids),
            "dependency_ids": list(self.dependency_ids),
            "execution_status": self.execution_status,
            "verification_status": self.verification_status,
            "failure_info": safe_fail if self.failure_info else None,
            "observation_refs": safe_obs,
            "requires_verified_dependencies": bool(self.requires_verified_dependencies),
            "contradictory": bool(self.contradictory),
            "recoverable": self.recoverable,
            "blocked_reason": _sanitize_graph_text(self.blocked_reason, max_bytes=256) if self.blocked_reason else None,
            "order_index": int(self.order_index),
            "external_source": _strip_forbidden_graph_keys(dict(self.external_source)) if isinstance(self.external_source, dict) else None,
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
        }
        if self.reserved_by is not None:
            out["reserved_by"] = _sanitize_graph_text(self.reserved_by, max_bytes=128)
        if self.reservation_id is not None:
            out["reservation_id"] = _sanitize_graph_text(self.reservation_id, max_bytes=128)
        if self.reserved_at is not None:
            out["reserved_at"] = float(self.reserved_at)
        if self.conflict_resources:
            out["conflict_resources"] = [str(r) for r in self.conflict_resources if str(r).strip()]
        if self.exclusive_resource is not None:
            out["exclusive_resource"] = _sanitize_graph_text(self.exclusive_resource, max_bytes=128)
        if self.expected_outcome:
            out["expected_outcome"] = _sanitize_graph_text(self.expected_outcome, max_bytes=512)
        if self.verification_requirement is not None:
            out["verification_requirement"] = _sanitize_graph_text(self.verification_requirement, max_bytes=256)
        if self.timeout_seconds is not None:
            out["timeout_seconds"] = float(self.timeout_seconds)
        if isinstance(self.recovery_state, dict) and self.recovery_state:
            safe_rec, _, _ = sanitize_ledger_payload(
                _strip_forbidden_graph_keys(dict(self.recovery_state)),
                max_bytes=4096,
            )
            if safe_rec:
                out["recovery_state"] = safe_rec
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskNode":
        if not isinstance(data, dict):
            raise CorruptedTaskGraphError("TaskNode payload must be a JSON object.")
        nid = data.get("node_id")
        if not isinstance(nid, str) or not nid.strip():
            raise CorruptedTaskGraphError("TaskNode missing valid 'node_id'.")
        exec_st = data.get("execution_status", TaskNodeStatus.PENDING.value)
        if not isinstance(exec_st, str) or exec_st not in _VALID_NODE_STATUSES:
            raise CorruptedTaskGraphError(f"TaskNode '{nid}' has invalid execution_status: {exec_st!r}.")
        ver_st = data.get("verification_status", VerificationStatus.NOT_VERIFIED.value)
        if not isinstance(ver_st, str) or ver_st not in _VALID_VERIFICATION_STATUSES:
            raise CorruptedTaskGraphError(f"TaskNode '{nid}' has invalid verification_status: {ver_st!r}.")

        raw_pids = data.get("proposal_ids", [])
        raw_deps = data.get("dependency_ids", [])
        raw_obs = data.get("observation_refs", [])
        if not isinstance(raw_pids, list) or not isinstance(raw_deps, list) or not isinstance(raw_obs, list):
            raise CorruptedTaskGraphError(f"TaskNode '{nid}' list fields must be lists.")
        for item in raw_pids + raw_deps:
            if not isinstance(item, str) or not item.strip():
                raise CorruptedTaskGraphError(f"TaskNode '{nid}' contains malformed identifier entry {item!r}.")
        for ob in raw_obs:
            if not isinstance(ob, dict):
                raise CorruptedTaskGraphError(f"TaskNode '{nid}' observation_refs entries must be dicts.")

        fail_info = data.get("failure_info")
        if fail_info is not None and not isinstance(fail_info, dict):
            raise CorruptedTaskGraphError(f"TaskNode '{nid}' failure_info must be a dict or null.")
        ext_src = data.get("external_source")
        if ext_src is not None and not isinstance(ext_src, dict):
            raise CorruptedTaskGraphError(f"TaskNode '{nid}' external_source must be a dict or null.")
        order_idx = data.get("order_index", 0)
        if not isinstance(order_idx, int) or isinstance(order_idx, bool) or order_idx < 0:
            raise CorruptedTaskGraphError(f"TaskNode '{nid}' has invalid order_index: {order_idx!r}.")

        res_by = data.get("reserved_by")
        if res_by is not None and (not isinstance(res_by, str) or not res_by.strip()):
            raise CorruptedTaskGraphError(f"TaskNode '{nid}' has invalid reserved_by: {res_by!r}.")
        res_id = data.get("reservation_id")
        if res_id is not None and (not isinstance(res_id, str) or not res_id.strip()):
            raise CorruptedTaskGraphError(f"TaskNode '{nid}' has invalid reservation_id: {res_id!r}.")
        res_at = data.get("reserved_at")
        if res_at is not None and (not isinstance(res_at, (int, float)) or isinstance(res_at, bool)):
            raise CorruptedTaskGraphError(f"TaskNode '{nid}' has invalid reserved_at: {res_at!r}.")
        raw_cres = data.get("conflict_resources", [])
        if not isinstance(raw_cres, list) or any(not isinstance(c, str) or not c.strip() for c in raw_cres):
            raise CorruptedTaskGraphError(f"TaskNode '{nid}' conflict_resources must be a list of non-empty strings.")
        excl_res = data.get("exclusive_resource")
        if excl_res is not None and (not isinstance(excl_res, str) or not excl_res.strip()):
            raise CorruptedTaskGraphError(f"TaskNode '{nid}' has invalid exclusive_resource: {excl_res!r}.")
        exp_out = data.get("expected_outcome", "")
        if not isinstance(exp_out, str):
            raise CorruptedTaskGraphError(f"TaskNode '{nid}' expected_outcome must be a string.")
        ver_req = data.get("verification_requirement")
        if ver_req is not None and not isinstance(ver_req, str):
            raise CorruptedTaskGraphError(f"TaskNode '{nid}' verification_requirement must be a string or null.")
        tout_s = data.get("timeout_seconds")
        if tout_s is not None and (
            isinstance(tout_s, bool) or not isinstance(tout_s, (int, float)) or float(tout_s) <= 0.0
        ):
            raise CorruptedTaskGraphError(f"TaskNode '{nid}' has invalid timeout_seconds: {tout_s!r}.")
        raw_rec_state = data.get("recovery_state")
        validated_rec_state: Optional[Dict[str, Any]] = None
        if raw_rec_state is not None:
            validated_rec_state = _validate_node_recovery_state(
                raw_rec_state,
                context_label=f"TaskNode '{nid}' recovery_state",
            )

        now = time.time()
        return cls(
            node_id=nid.strip(),
            description=_sanitize_graph_text(data.get("description") or "", max_bytes=512),
            proposal_ids=[p.strip() for p in raw_pids],
            dependency_ids=[d.strip() for d in raw_deps],
            execution_status=exec_st,
            verification_status=ver_st,
            failure_info=_strip_forbidden_graph_keys(dict(fail_info)) if isinstance(fail_info, dict) else None,
            observation_refs=[_strip_forbidden_graph_keys(dict(o)) for o in raw_obs],
            requires_verified_dependencies=bool(data.get("requires_verified_dependencies", True)),
            contradictory=bool(data.get("contradictory", False)),
            recoverable=data.get("recoverable") if isinstance(data.get("recoverable"), bool) else None,
            blocked_reason=_sanitize_graph_text(data["blocked_reason"], max_bytes=256) if data.get("blocked_reason") else None,
            order_index=order_idx,
            external_source=_strip_forbidden_graph_keys(dict(ext_src)) if isinstance(ext_src, dict) else None,
            reserved_by=_sanitize_graph_text(res_by, max_bytes=128) if res_by else None,
            reservation_id=_sanitize_graph_text(res_id, max_bytes=128) if res_id else None,
            reserved_at=float(res_at) if res_at is not None else None,
            conflict_resources=[str(c).strip() for c in raw_cres],
            exclusive_resource=_sanitize_graph_text(excl_res, max_bytes=128) if excl_res else None,
            expected_outcome=_sanitize_graph_text(exp_out, max_bytes=512) if exp_out else "",
            verification_requirement=_sanitize_graph_text(ver_req, max_bytes=256) if ver_req else None,
            timeout_seconds=float(tout_s) if tout_s is not None else None,
            recovery_state=validated_rec_state,
            created_at=float(data.get("created_at") or now),
            updated_at=float(data.get("updated_at") or now),
        )


@dataclass
class TaskGraph:
    """Explicit, versioned, integrity-verified DAG representing a multi-step Zelyr task."""

    graph_id: str
    execution_id: str
    objective: str = ""
    nodes: List[TaskNode] = field(default_factory=list)
    edges: List[TaskGraphEdge] = field(default_factory=list)
    status: str = TaskGraphStatus.INITIALIZED.value
    graph_version: int = TASK_GRAPH_VERSION
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    summary: Dict[str, Any] = field(default_factory=dict)
    graph_digest: str = ""
    max_concurrency: int = 1
    active_reservations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _lock: _threading.RLock = field(default_factory=_threading.RLock, init=False, repr=False, compare=False)
    _recovery_owners: Dict[str, str] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.max_concurrency, int) or isinstance(self.max_concurrency, bool) or self.max_concurrency < 1:
            raise TaskGraphValidationError(
                f"TaskGraph.max_concurrency must be a positive integer >= 1, got {self.max_concurrency!r}."
            )
        if not isinstance(self.active_reservations, dict):
            raise TaskGraphValidationError("TaskGraph.active_reservations must be a dictionary.")
        if not self.graph_id:
            self.graph_id = f"tg_{uuid.uuid4().hex[:12]}"
        self.objective = _sanitize_graph_text(self.objective, max_bytes=1024)
        for idx, n in enumerate(self.nodes):
            if n.order_index == 0 and idx > 0 and all(x.order_index == 0 for x in self.nodes):
                n.order_index = idx
        if not self.edges and any(n.dependency_ids for n in self.nodes):
            self._sync_edges_from_nodes()
        elif self.edges and not any(n.dependency_ids for n in self.nodes):
            self._sync_node_deps_from_edges()

    def _sync_edges_from_nodes(self) -> None:
        rebuilt: List[TaskGraphEdge] = []
        for node in self.nodes:
            for dep_id in node.dependency_ids:
                rebuilt.append(
                    TaskGraphEdge(
                        from_node_id=dep_id,
                        to_node_id=node.node_id,
                        requires_verified=bool(node.requires_verified_dependencies),
                    )
                )
        self.edges = rebuilt

    def _sync_node_deps_from_edges(self) -> None:
        node_map = {n.node_id: n for n in self.nodes}
        for edge in self.edges:
            target = node_map.get(edge.to_node_id)
            if target is not None:
                target.dependency_ids.append(edge.from_node_id)

    def get_node(self, node_id: str) -> Optional[TaskNode]:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def find_nodes_for_call(self, call_id: str, *, proposal_id: Optional[str] = None) -> List[TaskNode]:
        """Locate nodes bound to a given call_id or proposal_id."""
        refs = {str(r).strip() for r in (call_id, proposal_id) if r and str(r).strip()}
        matched: List[TaskNode] = []
        for node in self.nodes:
            if node.node_id in refs or any(pid in refs for pid in node.proposal_ids):
                matched.append(node)
        return matched

    def validate(
        self,
        *,
        known_proposals: Optional[Union[Dict[str, Any], Iterable[Any]]] = None,
        allow_blocked: bool = True,
    ) -> None:
        """Deterministically validate DAG structure, references, states, cycles, and isolation."""
        if self.graph_version not in SUPPORTED_TASK_GRAPH_VERSIONS:
            raise TaskGraphValidationError(f"Unsupported task graph version: {self.graph_version}")
        if not isinstance(self.graph_id, str) or not self.graph_id.strip():
            raise TaskGraphValidationError("TaskGraph must define a non-empty graph_id.")
        if not isinstance(self.execution_id, str) or not self.execution_id.strip():
            raise TaskGraphValidationError("TaskGraph must define a non-empty execution_id.")
        if self.status not in _VALID_GRAPH_STATUSES:
            raise TaskGraphValidationError(f"Invalid TaskGraph status: {self.status!r}")
        if not isinstance(self.max_concurrency, int) or isinstance(self.max_concurrency, bool) or self.max_concurrency < 1:
            raise TaskGraphValidationError(
                f"TaskGraph max_concurrency must be a positive integer >= 1, got {self.max_concurrency!r}."
            )
        if not isinstance(self.active_reservations, dict):
            raise TaskGraphValidationError("TaskGraph active_reservations must be a dictionary.")

        # Build proposal lookup & enforce single-execution isolation boundary
        prop_map: Dict[str, Any] = {}
        if isinstance(known_proposals, dict):
            prop_map = dict(known_proposals)
        elif known_proposals is not None:
            for item in known_proposals:
                pid = getattr(item, "proposal_id", None) or (item.get("proposal_id") if isinstance(item, dict) else None)
                cid = getattr(item, "original_call_id", None) or (item.get("original_call_id") if isinstance(item, dict) else None)
                if pid:
                    prop_map[str(pid)] = item
                if cid:
                    prop_map[str(cid)] = item

        node_map: Dict[str, TaskNode] = {}
        for node in self.nodes:
            if not isinstance(node.node_id, str) or not node.node_id.strip():
                raise TaskGraphValidationError("TaskNode has empty node_id.")
            if node.node_id in node_map:
                raise TaskGraphValidationError(f"Duplicate node ID detected: '{node.node_id}'.")
            if node.execution_status not in _VALID_NODE_STATUSES:
                raise TaskGraphValidationError(
                    f"Node '{node.node_id}' has invalid execution_status {node.execution_status!r}."
                )
            if node.verification_status not in _VALID_VERIFICATION_STATUSES:
                raise TaskGraphValidationError(
                    f"Node '{node.node_id}' has invalid verification_status {node.verification_status!r}."
                )
            node_map[node.node_id] = node

            # Cross-execution proposal isolation check
            seen_node_pids: Set[str] = set()
            for pid in node.proposal_ids:
                if not isinstance(pid, str) or not pid.strip():
                    raise TaskGraphValidationError(f"Node '{node.node_id}' contains invalid proposal reference {pid!r}.")
                if pid in seen_node_pids:
                    raise TaskGraphValidationError(f"Node '{node.node_id}' references duplicate proposal ID '{pid}'.")
                seen_node_pids.add(pid)
                prop = prop_map.get(pid)
                if prop is not None:
                    prop_exec = (
                        getattr(prop, "execution_id", None)
                        if not isinstance(prop, dict)
                        else prop.get("execution_id")
                    )
                    if prop_exec and str(prop_exec) != self.execution_id:
                        raise TaskGraphValidationError(
                            f"Cross-execution isolation violation: node '{node.node_id}' in '{self.execution_id}' "
                            f"references proposal '{pid}' belonging to '{prop_exec}'."
                        )

            seen_deps: Set[str] = set()
            for dep_id in node.dependency_ids:
                if not isinstance(dep_id, str) or not dep_id.strip():
                    raise TaskGraphValidationError(f"Node '{node.node_id}' has empty dependency ID.")
                if dep_id == node.node_id:
                    raise TaskGraphValidationError(f"Self-dependency detected on node '{node.node_id}'.")
                if dep_id in seen_deps:
                    raise TaskGraphValidationError(
                        f"Duplicate dependency edge detected: '{dep_id}' -> '{node.node_id}'."
                    )
                if dep_id not in node_map and dep_id not in {n.node_id for n in self.nodes}:
                    raise TaskGraphValidationError(
                        f"Node '{node.node_id}' references missing dependency '{dep_id}'."
                    )
                seen_deps.add(dep_id)

        # Validate explicit edge list for duplicates, self-loops, missing nodes, and consistency
        seen_edges: Set[Tuple[str, str]] = set()
        for edge in self.edges:
            if edge.from_node_id == edge.to_node_id:
                raise TaskGraphValidationError(f"Self-dependency edge detected on node '{edge.from_node_id}'.")
            if edge.from_node_id not in node_map:
                raise TaskGraphValidationError(
                    f"Edge '{edge.from_node_id}' -> '{edge.to_node_id}' references missing source node '{edge.from_node_id}'."
                )
            if edge.to_node_id not in node_map:
                raise TaskGraphValidationError(
                    f"Edge '{edge.from_node_id}' -> '{edge.to_node_id}' references missing target node '{edge.to_node_id}'."
                )
            pair = (edge.from_node_id, edge.to_node_id)
            if pair in seen_edges:
                raise TaskGraphValidationError(
                    f"Duplicate dependency edge detected: '{edge.from_node_id}' -> '{edge.to_node_id}'."
                )
            seen_edges.add(pair)
            if edge.from_node_id not in node_map[edge.to_node_id].dependency_ids:
                raise TaskGraphValidationError(
                    f"Edge '{edge.from_node_id}' -> '{edge.to_node_id}' missing from target node dependency_ids."
                )

        # Verify every node dependency_id also exists in edges when edges are populated
        if self.edges:
            for node in self.nodes:
                for dep_id in node.dependency_ids:
                    if (dep_id, node.node_id) not in seen_edges:
                        raise TaskGraphValidationError(
                            f"Node '{node.node_id}' dependency '{dep_id}' missing corresponding TaskGraphEdge."
                        )

        # Deterministic DFS 3-color cycle detection reporting full cycle chain
        color: Dict[str, int] = {}  # 0=unvisited, 1=visiting, 2=visited
        stack: List[str] = []
        adj: Dict[str, List[str]] = {n.node_id: [] for n in self.nodes}
        for node in self.nodes:
            for dep_id in node.dependency_ids:
                adj[dep_id].append(node.node_id)

        def _dfs(nid: str) -> None:
            color[nid] = 1
            stack.append(nid)
            for nxt in adj.get(nid, []):
                st = color.get(nxt, 0)
                if st == 1:
                    cyc_start = stack.index(nxt)
                    cycle_path = " -> ".join(stack[cyc_start:] + [nxt])
                    raise TaskGraphValidationError(f"Cyclic dependency detected: {cycle_path}")
                if st == 0:
                    _dfs(nxt)
            stack.pop()
            color[nid] = 2

        for node in self.nodes:
            if color.get(node.node_id, 0) == 0:
                _dfs(node.node_id)

        # Check ready/running state consistency and optional strict satisfiability check
        unsat = self.find_unsatisfiable_dependencies(known_proposals=prop_map)
        for node in self.nodes:
            if node.execution_status in (TaskNodeStatus.READY.value, TaskNodeStatus.RUNNING.value):
                for dep_id in node.dependency_ids:
                    dep_node = node_map[dep_id]
                    if not self._is_prerequisite_satisfied(dep_node, node.requires_verified_dependencies):
                        raise TaskGraphValidationError(
                            f"Node '{node.node_id}' cannot be '{node.execution_status}' while prerequisite "
                            f"'{dep_id}' is '{dep_node.execution_status}' (verification='{dep_node.verification_status}')."
                        )
        if not allow_blocked and unsat:
            nid, dep_id, reason = unsat[0]
            raise TaskGraphValidationError(
                f"Unsatisfiable dependency for node '{nid}' on '{dep_id}': {reason}"
            )

    @staticmethod
    def _is_prerequisite_satisfied(dep_node: TaskNode, requires_verified: bool = True) -> bool:
        if dep_node.contradictory:
            return False
        if dep_node.execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value:
            return dep_node.verification_status == VerificationStatus.VERIFIED.value
        if not requires_verified and dep_node.execution_status == TaskNodeStatus.COMPLETED_UNVERIFIED.value:
            return True
        return False

    def find_unsatisfiable_dependencies(
        self,
        *,
        known_proposals: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, str, str]]:
        """Return [(node_id, dep_id, reason)] for dependencies that cannot be satisfied without replanning."""
        node_map = {n.node_id: n for n in self.nodes}
        unsatisfiable: List[Tuple[str, str, str]] = []
        for node in self.nodes:
            if node.execution_status in (TaskNodeStatus.COMPLETED_VERIFIED.value, TaskNodeStatus.CANCELLED.value):
                continue
            for dep_id in node.dependency_ids:
                dep = node_map.get(dep_id)
                if dep is None:
                    unsatisfiable.append((node.node_id, dep_id, "missing prerequisite node"))
                    continue
                if dep.contradictory:
                    unsatisfiable.append((node.node_id, dep_id, "prerequisite has contradictory evidence"))
                elif dep.execution_status == TaskNodeStatus.CANCELLED.value:
                    unsatisfiable.append((node.node_id, dep_id, "prerequisite was cancelled"))
                elif dep.execution_status == TaskNodeStatus.FAILED.value and dep.recoverable is False:
                    unsatisfiable.append((node.node_id, dep_id, "prerequisite failed non-recoverably"))
                elif dep.execution_status == TaskNodeStatus.BLOCKED.value:
                    unsatisfiable.append((node.node_id, dep_id, f"prerequisite is blocked ({dep.blocked_reason or 'blocked'})"))
                elif dep.execution_status == TaskNodeStatus.COMPLETED_UNVERIFIED.value and node.requires_verified_dependencies:
                    unsatisfiable.append((node.node_id, dep_id, "prerequisite is completed_unverified and lacks verification"))
                elif known_proposals:
                    for pid in dep.proposal_ids:
                        p = known_proposals.get(pid)
                        if p is not None:
                            p_st = str(getattr(p, "status", None) or (p.get("status") if isinstance(p, dict) else "") or "")
                            if p_st in ("expired", "cancelled"):
                                unsatisfiable.append((node.node_id, dep_id, f"prerequisite proposal '{pid}' is {p_st}"))
        return unsatisfiable

    def topological_order(self) -> List[TaskNode]:
        """Return nodes in deterministic topological order breaking ties by order_index."""
        self.validate(allow_blocked=True)
        node_map = {n.node_id: n for n in self.nodes}
        in_degree: Dict[str, int] = {n.node_id: len(n.dependency_ids) for n in self.nodes}
        dependents: Dict[str, List[str]] = {n.node_id: [] for n in self.nodes}
        for n in self.nodes:
            for d in n.dependency_ids:
                dependents[d].append(n.node_id)

        ready_ids = sorted(
            (nid for nid, deg in in_degree.items() if deg == 0),
            key=lambda nid: (node_map[nid].order_index, nid),
        )
        ordered: List[TaskNode] = []
        while ready_ids:
            cur = ready_ids.pop(0)
            ordered.append(node_map[cur])
            for dep_id in dependents.get(cur, []):
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    ready_ids.append(dep_id)
            ready_ids.sort(key=lambda nid: (node_map[nid].order_index, nid))
        return ordered

    def recompute_schedule(
        self,
        *,
        known_proposals: Optional[Union[Dict[str, Any], Iterable[Any]]] = None,
    ) -> "TaskGraph":
        """Recompute node readiness, blocked states, summary counters, aggregate status, and digest."""
        prop_map: Dict[str, Any] = {}
        if isinstance(known_proposals, dict):
            prop_map = dict(known_proposals)
        elif known_proposals is not None:
            for item in known_proposals:
                pid = getattr(item, "proposal_id", None) or (item.get("proposal_id") if isinstance(item, dict) else None)
                cid = getattr(item, "original_call_id", None) or (item.get("original_call_id") if isinstance(item, dict) else None)
                if pid:
                    prop_map[str(pid)] = item
                if cid:
                    prop_map[str(cid)] = item

        # Validate structure and references first (cycles/missing/cross-exec fail closed)
        # Allow temporarily updating READY/BLOCKED transitions cleanly during recompute.
        self._sync_edges_from_nodes()
        ordered_nodes = self._validate_dag_only(known_proposals=prop_map)
        node_map = {n.node_id: n for n in ordered_nodes}
        now = time.time()

        for node in ordered_nodes:
            # Terminal or actively running/failed states stay unless contradictory/proposal changes apply
            if node.execution_status in (
                TaskNodeStatus.COMPLETED_VERIFIED.value,
                TaskNodeStatus.CANCELLED.value,
                TaskNodeStatus.RUNNING.value,
            ):
                continue

            if node.contradictory:
                node.execution_status = TaskNodeStatus.BLOCKED.value
                node.blocked_reason = node.blocked_reason or "Contradictory observation evidence requires reconciliation."
                node.updated_at = now
                continue

            if node.execution_status in (
                TaskNodeStatus.COMPLETED_UNVERIFIED.value,
                TaskNodeStatus.FAILED.value,
            ):
                continue

            # Check bound proposals for block/cancel/expiration/completion signals
            if node.proposal_ids and prop_map:
                bound_props = [prop_map[pid] for pid in node.proposal_ids if pid in prop_map]
                if bound_props:
                    statuses = {
                        str(getattr(p, "status", None) or (p.get("status") if isinstance(p, dict) else "") or "")
                        for p in bound_props
                    }
                    if statuses <= {"cancelled"}:
                        node.execution_status = TaskNodeStatus.CANCELLED.value
                        first_r = getattr(bound_props[0], "blocked_reason", None) or (
                            bound_props[0].get("blocked_reason") if isinstance(bound_props[0], dict) else None
                        )
                        node.blocked_reason = first_r or "proposal_cancelled"
                        node.updated_at = now
                        continue
                    if any(st in ("blocked", "expired") for st in statuses) and not any(
                        st in ("pending", "validated", "dispatched", "completed") for st in statuses
                    ):
                        blk_p = next(
                            p for p in bound_props
                            if str(getattr(p, "status", None) or (p.get("status") if isinstance(p, dict) else "") or "") in ("blocked", "expired")
                        )
                        node.execution_status = TaskNodeStatus.BLOCKED.value
                        node.blocked_reason = (
                            getattr(blk_p, "blocked_reason", None)
                            or (blk_p.get("blocked_reason") if isinstance(blk_p, dict) else None)
                            or "Associated proposal is blocked or expired."
                        )
                        node.updated_at = now
                        continue

            # Evaluate dependency satisfaction
            blocked_msg: Optional[str] = None
            waiting_on_dep = False
            for dep_id in node.dependency_ids:
                dep = node_map[dep_id]
                if dep.contradictory:
                    blocked_msg = f"Prerequisite '{dep_id}' has contradictory evidence and requires reconciliation."
                    break
                if dep.execution_status == TaskNodeStatus.FAILED.value:
                    blocked_msg = f"Prerequisite '{dep_id}' failed ({(dep.failure_info or {}).get('error') or 'failed'})."
                    break
                if dep.execution_status == TaskNodeStatus.BLOCKED.value:
                    blocked_msg = f"Prerequisite '{dep_id}' is blocked ({dep.blocked_reason or 'blocked'})."
                    break
                if dep.execution_status == TaskNodeStatus.CANCELLED.value:
                    blocked_msg = f"Prerequisite '{dep_id}' was cancelled."
                    break
                if dep.execution_status == TaskNodeStatus.COMPLETED_UNVERIFIED.value:
                    if node.requires_verified_dependencies:
                        blocked_msg = f"Prerequisite '{dep_id}' succeeded without independent verification."
                        break
                    continue
                if dep.execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value:
                    if dep.verification_status != VerificationStatus.VERIFIED.value:
                        blocked_msg = f"Prerequisite '{dep_id}' lacks verified status ({dep.verification_status})."
                        break
                    continue
                # dep is PENDING, READY, or RUNNING
                waiting_on_dep = True

            if blocked_msg is not None:
                node.execution_status = TaskNodeStatus.BLOCKED.value
                node.blocked_reason = _sanitize_graph_text(blocked_msg, max_bytes=256)
                node.updated_at = now
            elif waiting_on_dep:
                node.execution_status = TaskNodeStatus.PENDING.value
                if node.blocked_reason and node.blocked_reason.startswith("Prerequisite "):
                    node.blocked_reason = None
                node.updated_at = now
            else:
                node.execution_status = TaskNodeStatus.READY.value
                node.blocked_reason = None
                node.updated_at = now

        self._refresh_summary_and_digest()
        return self

    def _validate_dag_only(self, *, known_proposals: Optional[Dict[str, Any]] = None) -> List[TaskNode]:
        """Validate structure, isolation, uniqueness, and acyclicity without forbidding blocked states."""
        # Reuse full validator checks except readiness check (which recompute_schedule is about to update)
        if self.graph_version not in SUPPORTED_TASK_GRAPH_VERSIONS:
            raise TaskGraphValidationError(f"Unsupported task graph version: {self.graph_version}")
        if not isinstance(self.graph_id, str) or not self.graph_id.strip():
            raise TaskGraphValidationError("TaskGraph must define a non-empty graph_id.")
        if not isinstance(self.execution_id, str) or not self.execution_id.strip():
            raise TaskGraphValidationError("TaskGraph must define a non-empty execution_id.")
        prop_map = known_proposals or {}
        node_map: Dict[str, TaskNode] = {}
        all_ids = {n.node_id for n in self.nodes}
        for node in self.nodes:
            if not node.node_id or node.node_id in node_map:
                raise TaskGraphValidationError(f"Invalid or duplicate node_id: {node.node_id!r}")
            node_map[node.node_id] = node
            for pid in node.proposal_ids:
                prop = prop_map.get(pid)
                if prop is not None:
                    prop_exec = getattr(prop, "execution_id", None) if not isinstance(prop, dict) else prop.get("execution_id")
                    if prop_exec and str(prop_exec) != self.execution_id:
                        raise TaskGraphValidationError(
                            f"Cross-execution proposal reference '{pid}' belongs to '{prop_exec}', not '{self.execution_id}'."
                        )
            seen_d: Set[str] = set()
            for dep_id in node.dependency_ids:
                if dep_id == node.node_id:
                    raise TaskGraphValidationError(f"Self-dependency detected on node '{node.node_id}'.")
                if dep_id in seen_d:
                    raise TaskGraphValidationError(f"Duplicate dependency edge '{dep_id}' -> '{node.node_id}'.")
                if dep_id not in all_ids:
                    raise TaskGraphValidationError(f"Missing dependency '{dep_id}' on node '{node.node_id}'.")
                seen_d.add(dep_id)

        in_degree: Dict[str, int] = {n.node_id: len(n.dependency_ids) for n in self.nodes}
        dependents: Dict[str, List[str]] = {n.node_id: [] for n in self.nodes}
        for n in self.nodes:
            for d in n.dependency_ids:
                dependents[d].append(n.node_id)
        queue = sorted((nid for nid, deg in in_degree.items() if deg == 0), key=lambda x: (node_map[x].order_index, x))
        ordered: List[TaskNode] = []
        while queue:
            cur = queue.pop(0)
            ordered.append(node_map[cur])
            for nxt in dependents.get(cur, []):
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)
            queue.sort(key=lambda x: (node_map[x].order_index, x))
        if len(ordered) != len(self.nodes):
            # Invoke validate() to emit exact cycle path error
            self.validate(known_proposals=prop_map, allow_blocked=True)
            raise TaskGraphValidationError("Cyclic dependency detected in task graph.")
        return ordered

    def get_next_eligible_node(self) -> Optional[TaskNode]:
        """Return the single highest-priority READY node respecting sequential execution order."""
        with self._lock:
            for node in self.topological_order():
                if node.execution_status == TaskNodeStatus.READY.value and node.reserved_by is None:
                    return node
            return None

    def configure_max_concurrency(self, max_concurrency: int) -> "TaskGraph":
        """Validate and update bounded parallel execution concurrency at runtime."""
        if not isinstance(max_concurrency, int) or isinstance(max_concurrency, bool) or max_concurrency < 1:
            raise TaskGraphValidationError(
                f"TaskGraph.max_concurrency must be a positive integer >= 1, got {max_concurrency!r}."
            )
        with self._lock:
            self.max_concurrency = int(max_concurrency)
            self._refresh_summary_and_digest()
            return self

    def get_running_nodes(self) -> List[TaskNode]:
        """Return nodes currently executing or holding active execution reservations."""
        with self._lock:
            return [
                n
                for n in sorted(self.nodes, key=lambda x: (x.order_index, x.node_id))
                if n.execution_status == TaskNodeStatus.RUNNING.value or n.reserved_by is not None
            ]

    def get_ready_nodes(
        self,
        *,
        limit: Optional[int] = None,
        exclude_conflicting: bool = False,
        profile_fn: Optional[Callable[[TaskNode], NodeConflictProfile]] = None,
    ) -> List[TaskNode]:
        """Return deterministically ordered READY nodes eligible for scheduling."""
        with self._lock:
            if limit is not None and limit <= 0:
                return []
            node_map = {n.node_id: n for n in self.nodes}
            candidates: List[TaskNode] = []
            for node in self.topological_order():
                if node.execution_status != TaskNodeStatus.READY.value:
                    continue
                if node.reserved_by is not None or node.node_id in self.active_reservations:
                    continue
                if not all(
                    d in node_map and self._is_prerequisite_satisfied(node_map[d], node.requires_verified_dependencies)
                    for d in node.dependency_ids
                ):
                    continue
                candidates.append(node)

            if not exclude_conflicting:
                return candidates[:limit] if limit is not None else candidates

            active_profiles: List[NodeConflictProfile] = []
            for r_entry in self.active_reservations.values():
                if isinstance(r_entry, dict):
                    active_profiles.append(self._profile_from_reservation_entry(r_entry))
            for rn in self.nodes:
                if rn.execution_status == TaskNodeStatus.RUNNING.value and rn.node_id not in self.active_reservations:
                    active_profiles.append(
                        NodeConflictProfile(
                            lock_keys=tuple(rn.conflict_resources or ()),
                            exclusive_resource=rn.exclusive_resource,
                            read_only=False,
                        )
                    )

            selected: List[TaskNode] = []
            for cand in candidates:
                c_prof = (
                    profile_fn(cand)
                    if profile_fn is not None
                    else extract_node_conflict_profile(node=cand, external_source=cand.external_source)
                )
                if any(profiles_conflict(c_prof, ap) is not None for ap in active_profiles):
                    continue
                selected.append(cand)
                active_profiles.append(c_prof)
                if limit is not None and len(selected) >= limit:
                    break
            return selected

    @staticmethod
    def _profile_from_reservation_entry(entry: Dict[str, Any]) -> NodeConflictProfile:
        return NodeConflictProfile(
            lock_keys=tuple(str(x) for x in (entry.get("resources") or []) if str(x).strip()),
            read_targets=tuple(str(x) for x in (entry.get("read_targets") or []) if str(x).strip()),
            write_targets=tuple(str(x) for x in (entry.get("write_targets") or []) if str(x).strip()),
            exclusive_resource=str(entry["exclusive_resource"]) if entry.get("exclusive_resource") else None,
            operation_fingerprint=str(entry["operation_fingerprint"]) if entry.get("operation_fingerprint") else None,
            read_only=bool(entry.get("read_only", False)),
        )

    def _active_profiles_excluding(self, exclude_node_id: Optional[str] = None) -> List[Tuple[str, NodeConflictProfile]]:
        active: List[Tuple[str, NodeConflictProfile]] = []
        for nid, r_entry in self.active_reservations.items():
            if nid == exclude_node_id or not isinstance(r_entry, dict):
                continue
            active.append((nid, self._profile_from_reservation_entry(r_entry)))
        for n in self.nodes:
            if n.node_id == exclude_node_id or n.node_id in self.active_reservations:
                continue
            if n.execution_status == TaskNodeStatus.RUNNING.value or n.reserved_by is not None:
                active.append(
                    (
                        n.node_id,
                        NodeConflictProfile(
                            lock_keys=tuple(n.conflict_resources or ()),
                            exclusive_resource=n.exclusive_resource,
                            read_only=False,
                        ),
                    )
                )
        return active

    def can_nodes_run_concurrently(
        self,
        node_a_id: str,
        node_b_id: str,
        *,
        profile_a: Optional[NodeConflictProfile] = None,
        profile_b: Optional[NodeConflictProfile] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Check whether two nodes are independent and conflict-free for parallel execution."""
        with self._lock:
            na = self.get_node(node_a_id)
            nb = self.get_node(node_b_id)
            if na is None or nb is None:
                return False, "unknown_node"
            if na.node_id == nb.node_id:
                return False, "identical_node"
            if nb.node_id in na.dependency_ids or na.node_id in nb.dependency_ids:
                return False, "direct_dependency"
            # Check transitive reachability so ancestor/descendant nodes never run together
            node_map = {n.node_id: n for n in self.nodes}

            def _ancestors(start: TaskNode) -> Set[str]:
                seen: Set[str] = set()
                stack = list(start.dependency_ids)
                while stack:
                    cur = stack.pop()
                    if cur in seen:
                        continue
                    seen.add(cur)
                    cn = node_map.get(cur)
                    if cn is not None:
                        stack.extend(cn.dependency_ids)
                return seen

            if na.node_id in _ancestors(nb) or nb.node_id in _ancestors(na):
                return False, "transitive_dependency"

            pa = profile_a or extract_node_conflict_profile(node=na, external_source=na.external_source)
            pb = profile_b or extract_node_conflict_profile(node=nb, external_source=nb.external_source)
            conflict = profiles_conflict(pa, pb)
            if conflict is not None:
                return False, conflict
            return True, None

    def reserve_node(
        self,
        node_id: str,
        *,
        owner_id: str,
        reservation_id: Optional[str] = None,
        conflict_profile: Optional[NodeConflictProfile] = None,
        resources: Optional[Sequence[str]] = None,
        exclusive_resource: Optional[str] = None,
        enforce_concurrency_cap: bool = True,
        max_concurrency: Optional[int] = None,
    ) -> str:
        """Atomically claim exclusive ownership and canonical sorted resource locks for a READY node."""
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise TaskGraphValidationError("Node reservation requires a non-empty owner_id.")
        eff_owner = owner_id.strip()

        with self._lock:
            eff_cap = max_concurrency if max_concurrency is not None else self.max_concurrency
            if not isinstance(eff_cap, int) or isinstance(eff_cap, bool) or eff_cap < 1:
                raise TaskGraphValidationError(f"Invalid max_concurrency cap: {eff_cap!r}.")

            node_map = {n.node_id: n for n in self.nodes}
            node = node_map.get(node_id)
            if node is None:
                raise TaskGraphValidationError(f"Cannot reserve unknown task node '{node_id}'.")

            if node.reserved_by is not None or node_id in self.active_reservations:
                holder = node.reserved_by or (self.active_reservations.get(node_id) or {}).get("owner_id")
                raise TaskGraphConflictError(f"Node '{node_id}' is already reserved by '{holder}'.")

            if node.execution_status != TaskNodeStatus.READY.value:
                raise TaskGraphConflictError(
                    f"Node '{node_id}' is in status '{node.execution_status}' and cannot be reserved for dispatch."
                )

            for dep_id in node.dependency_ids:
                dep = node_map.get(dep_id)
                if dep is None or not self._is_prerequisite_satisfied(dep, node.requires_verified_dependencies):
                    raise TaskGraphValidationError(
                        f"Node '{node_id}' cannot be reserved while prerequisite '{dep_id}' is unsatisfied."
                    )

            if enforce_concurrency_cap and len(self.active_reservations) >= eff_cap:
                raise TaskGraphConflictError(
                    f"Bounded concurrency limit ({eff_cap}) reached; cannot reserve node '{node_id}'."
                )

            base_prof = conflict_profile or extract_node_conflict_profile(node=node, external_source=node.external_source)
            merged_locks = sorted(
                {str(r).strip() for r in list(base_prof.lock_keys) + list(resources or ()) if str(r).strip()}
                | ({str(exclusive_resource).strip()} if exclusive_resource else set())
            )
            eff_excl = exclusive_resource or base_prof.exclusive_resource
            eff_prof = NodeConflictProfile(
                lock_keys=tuple(merged_locks),
                read_targets=base_prof.read_targets,
                write_targets=base_prof.write_targets,
                exclusive_resource=eff_excl,
                operation_fingerprint=base_prof.operation_fingerprint,
                read_only=base_prof.read_only,
            )

            for active_nid, active_prof in self._active_profiles_excluding(node_id):
                conflict_reason = profiles_conflict(eff_prof, active_prof)
                if conflict_reason is not None:
                    raise TaskGraphConflictError(
                        f"Node '{node_id}' conflicts with active node '{active_nid}': {conflict_reason}."
                    )

            rid = (reservation_id or f"res_{uuid.uuid4().hex[:12]}").strip()
            now = time.time()
            node.reserved_by = eff_owner
            node.reservation_id = rid
            node.reserved_at = now
            node.conflict_resources = list(merged_locks)
            node.exclusive_resource = eff_excl
            node.execution_status = TaskNodeStatus.RUNNING.value
            node.blocked_reason = None
            node.updated_at = now
            self.active_reservations[node_id] = {
                "node_id": node_id,
                "owner_id": eff_owner,
                "reservation_id": rid,
                "resources": list(merged_locks),
                "read_targets": list(eff_prof.read_targets),
                "write_targets": list(eff_prof.write_targets),
                "exclusive_resource": eff_excl,
                "operation_fingerprint": eff_prof.operation_fingerprint,
                "read_only": bool(eff_prof.read_only),
                "reserved_at": now,
            }
            self._refresh_summary_and_digest()
            return rid

    def release_reservation(
        self,
        node_id: str,
        *,
        owner_id: str,
        reservation_id: str,
        final_status: Optional[str] = None,
        blocked_reason: Optional[str] = None,
        known_proposals: Optional[Union[Dict[str, Any], Iterable[Any]]] = None,
    ) -> TaskNode:
        """Release an active node reservation only if caller holds matching owner_id and reservation_id."""
        with self._lock:
            node = self.get_node(node_id)
            if node is None:
                raise TaskGraphValidationError(f"Cannot release unknown task node '{node_id}'.")

            entry = self.active_reservations.get(node_id)
            cur_owner = node.reserved_by or (entry.get("owner_id") if isinstance(entry, dict) else None)
            cur_rid = node.reservation_id or (entry.get("reservation_id") if isinstance(entry, dict) else None)
            if cur_owner is None and entry is None:
                raise TaskGraphValidationError(f"Node '{node_id}' has no active reservation to release.")
            if cur_owner != owner_id or cur_rid != reservation_id:
                raise TaskGraphValidationError(
                    f"Non-owner '{owner_id}' (reservation='{reservation_id}') cannot release lock held by '{cur_owner}' on '{node_id}'."
                )

            now = time.time()
            node.reserved_by = None
            node.reservation_id = None
            node.reserved_at = None
            self.active_reservations.pop(node_id, None)
            if final_status is not None:
                if final_status not in _VALID_NODE_STATUSES:
                    raise TaskGraphValidationError(f"Invalid final_status on release: {final_status!r}.")
                node.execution_status = final_status
            if blocked_reason is not None:
                node.blocked_reason = _sanitize_graph_text(blocked_reason, max_bytes=256)
            node.updated_at = now
            self.recompute_schedule(known_proposals=known_proposals)
            return node

    def reconcile_interrupted_reservations(self, *, release_stale_unstarted: bool = False) -> List[Dict[str, Any]]:
        """Inspect and safely reconcile persisted reservations from a previous interrupted process."""
        with self._lock:
            notes: List[Dict[str, Any]] = []
            node_map = {n.node_id: n for n in self.nodes}
            for nid, entry in list(self.active_reservations.items()):
                node = node_map.get(nid)
                owner = entry.get("owner_id") if isinstance(entry, dict) else None
                if node is None:
                    self.active_reservations.pop(nid, None)
                    notes.append({"node_id": nid, "owner_id": owner, "action": "dropped_orphan_reservation"})
                    continue
                if node.execution_status in (
                    TaskNodeStatus.COMPLETED_VERIFIED.value,
                    TaskNodeStatus.CANCELLED.value,
                    TaskNodeStatus.FAILED.value,
                    TaskNodeStatus.COMPLETED_UNVERIFIED.value,
                ):
                    node.reserved_by = None
                    node.reservation_id = None
                    node.reserved_at = None
                    self.active_reservations.pop(nid, None)
                    notes.append({"node_id": nid, "owner_id": owner, "action": "cleared_terminal_reservation"})
                elif release_stale_unstarted and node.execution_status != TaskNodeStatus.RUNNING.value:
                    node.reserved_by = None
                    node.reservation_id = None
                    node.reserved_at = None
                    self.active_reservations.pop(nid, None)
                    notes.append({"node_id": nid, "owner_id": owner, "action": "cleared_stale_pre_dispatch_reservation"})
                else:
                    notes.append({
                        "node_id": nid,
                        "owner_id": owner or node.reserved_by,
                        "reservation_id": node.reservation_id or (entry.get("reservation_id") if isinstance(entry, dict) else None),
                        "status": node.execution_status,
                        "action": "preserved_interrupted_reservation",
                    })
            self._refresh_summary_and_digest()
            return notes

    def get_blocked_nodes(self) -> List[TaskNode]:
        return [n for n in self.nodes if n.execution_status == TaskNodeStatus.BLOCKED.value]

    def get_failed_nodes(self) -> List[TaskNode]:
        return [n for n in self.nodes if n.execution_status == TaskNodeStatus.FAILED.value]

    def get_unverified_nodes(self) -> List[TaskNode]:
        return [n for n in self.nodes if n.execution_status == TaskNodeStatus.COMPLETED_UNVERIFIED.value]

    def apply_observation(
        self,
        node_id: str,
        structured_observation: Any = None,
        *,
        action_record: Optional[ActionRecord] = None,
        outcome_verifier_passed: Optional[bool] = None,
        is_recoverable: Optional[bool] = None,
        known_proposals: Optional[Union[Dict[str, Any], Iterable[Any]]] = None,
    ) -> TaskNode:
        """Update node state deterministically from StructuredObservation + OutcomeVerifier evidence."""
        node = self.get_node(node_id)
        if node is None:
            raise TaskGraphValidationError(f"Unknown task node '{node_id}'.")
        # Completed and verified nodes cannot be downgraded by stale/prose signals unless independent check fails
        now = time.time()

        ev_level = getattr(structured_observation, "evidence_level", None) if structured_observation is not None else None
        obs_status = getattr(structured_observation, "execution_status", None) if structured_observation is not None else None
        ver_status = getattr(structured_observation, "verification_status", None) if structured_observation is not None else None
        indep_ver = getattr(structured_observation, "independent_verification", None) if structured_observation is not None else None
        rec_guid = getattr(structured_observation, "recovery_guidance", None) if structured_observation is not None else None
        corr_id = getattr(structured_observation, "correlation_id", None) if structured_observation is not None else None

        if isinstance(indep_ver, dict) and "passed" in indep_ver and outcome_verifier_passed is None:
            outcome_verifier_passed = bool(indep_ver["passed"])

        if ev_level == "CONTRADICTED" or obs_status == "contradictory_evidence":
            node.contradictory = True
            node.execution_status = TaskNodeStatus.BLOCKED.value
            node.verification_status = VerificationStatus.VERIFICATION_FAILED.value
            node.blocked_reason = "Contradictory observation evidence requires reconciliation."
            node.failure_info = {"error": "Contradictory evidence", "recoverable": True}
        elif obs_status == "cancelled" or (action_record is not None and action_record.state == ActionState.CANCELLED):
            node.execution_status = TaskNodeStatus.CANCELLED.value
            node.verification_status = VerificationStatus.INTERRUPTED.value
            may_se = bool(rec_guid.get("may_have_side_effect", True)) if isinstance(rec_guid, dict) else True
            err_msg = (
                (action_record.error if action_record else None)
                or (getattr(structured_observation, "error_detail", None) if structured_observation else None)
                or "Operation cancelled"
            )
            node.failure_info = {"error": _sanitize_graph_text(err_msg, max_bytes=256), "may_have_side_effect": may_se}
            node.blocked_reason = "Operation cancelled (possible side effects preserved)."
        elif (
            obs_status in ("executed_error", "failed_before_execution", "timed_out")
            or (action_record is not None and action_record.state == ActionState.FAILED)
        ):
            node.execution_status = TaskNodeStatus.FAILED.value
            node.contradictory = False
            if getattr(action_record, "verification_failed", False) or outcome_verifier_passed is False:
                node.verification_status = VerificationStatus.VERIFICATION_FAILED.value
            else:
                node.verification_status = VerificationStatus.NOT_VERIFIED.value
            eff_recov = is_recoverable
            if eff_recov is None and isinstance(rec_guid, dict) and "is_recoverable" in rec_guid:
                eff_recov = bool(rec_guid["is_recoverable"])
            elif eff_recov is None:
                eff_recov = obs_status != "failed_before_execution"
            node.recoverable = eff_recov
            err_cls = getattr(structured_observation, "error_classification", None) if structured_observation else None
            err_type = err_cls.get("error_type") if isinstance(err_cls, dict) else None
            err_detail = (
                (action_record.error if action_record else None)
                or (getattr(structured_observation, "error_detail", None) if structured_observation else None)
                or obs_status
                or "failed"
            )
            node.failure_info = {
                "error": _sanitize_graph_text(err_detail, max_bytes=256),
                "error_type": err_type,
                "recoverable": eff_recov,
                "may_have_side_effect": bool(rec_guid.get("may_have_side_effect", True)) if isinstance(rec_guid, dict) else True,
            }
            node.blocked_reason = node.failure_info["error"]
        elif (
            outcome_verifier_passed is True
            or ev_level == "CONFIRMED"
            or obs_status == "succeeded_verified"
            or (action_record is not None and action_record.state == ActionState.VERIFIED and not action_record.verification_failed)
        ):
            node.execution_status = TaskNodeStatus.COMPLETED_VERIFIED.value
            node.verification_status = VerificationStatus.VERIFIED.value
            node.contradictory = False
            node.failure_info = None
            node.blocked_reason = None
        elif (
            obs_status in ("succeeded", "succeeded_unverified", "partially_succeeded")
            or ev_level in ("REPORTED", "PARTIAL")
            or (action_record is not None and action_record.state in (ActionState.SUCCESS, ActionState.OBSERVED, ActionState.UNVERIFIED, ActionState.PARTIALLY_COMPLETED))
        ):
            if node.execution_status != TaskNodeStatus.COMPLETED_VERIFIED.value:
                node.execution_status = TaskNodeStatus.COMPLETED_UNVERIFIED.value
                node.verification_status = (
                    VerificationStatus.VERIFICATION_FAILED.value
                    if outcome_verifier_passed is False
                    else (str(ver_status) if ver_status in _VALID_VERIFICATION_STATUSES else VerificationStatus.NOT_VERIFIED.value)
                )
                node.blocked_reason = None

        obs_ref: Dict[str, Any] = {
            "call_id": corr_id or (action_record.call_id if action_record else None),
            "evidence_level": ev_level,
            "execution_status": obs_status,
            "verification_status": ver_status,
            "verified": node.execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value,
            "timestamp": now,
        }
        node.observation_refs.append({k: v for k, v in obs_ref.items() if v is not None})
        node.observation_refs = node.observation_refs[-16:]
        node.updated_at = now
        self.updated_at = now
        self.recompute_schedule(known_proposals=known_proposals)
        return node

    def add_node(
        self,
        node: TaskNode,
        *,
        known_proposals: Optional[Union[Dict[str, Any], Iterable[Any]]] = None,
    ) -> "TaskGraph":
        """Safely add a new node, rolling back atomically if validation fails."""
        snapshot = self.to_dict()
        candidate = TaskNode.from_dict(node.to_dict())
        candidate.order_index = max([n.order_index for n in self.nodes] + [-1]) + 1 if candidate.order_index == 0 and self.nodes else candidate.order_index
        self.nodes.append(candidate)
        try:
            self.recompute_schedule(known_proposals=known_proposals)
            self.validate(known_proposals=known_proposals, allow_blocked=True)
        except Exception:
            self._restore_from_dict(snapshot)
            raise
        return self

    def replace_failed_node(
        self,
        failed_node_id: str,
        replacement: TaskNode,
        *,
        rewire_dependents: bool = True,
        known_proposals: Optional[Union[Dict[str, Any], Iterable[Any]]] = None,
    ) -> "TaskGraph":
        """Replace/supersede a failed or blocked node without altering completed verified nodes."""
        target = self.get_node(failed_node_id)
        if target is None:
            raise TaskGraphValidationError(f"Cannot replace unknown node '{failed_node_id}'.")
        if target.execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value:
            raise TaskGraphValidationError(f"Cannot replace already completed and verified node '{failed_node_id}'.")

        snapshot = self.to_dict()
        new_node = TaskNode.from_dict(replacement.to_dict())
        if new_node.node_id == failed_node_id:
            # In-place retry update preserving historical observation_refs
            new_node.observation_refs = list(target.observation_refs) + list(new_node.observation_refs)
            new_node.execution_status = TaskNodeStatus.PENDING.value
            new_node.verification_status = VerificationStatus.NOT_VERIFIED.value
            new_node.failure_info = None
            new_node.blocked_reason = None
            new_node.contradictory = False
            idx = self.nodes.index(target)
            self.nodes[idx] = new_node
        else:
            target.execution_status = TaskNodeStatus.CANCELLED.value
            target.blocked_reason = f"Superseded by replacement node '{new_node.node_id}'"
            if not new_node.dependency_ids:
                new_node.dependency_ids = list(target.dependency_ids)
            self.nodes.append(new_node)
            if rewire_dependents:
                for n in self.nodes:
                    if n.node_id == new_node.node_id:
                        continue
                    if failed_node_id in n.dependency_ids:
                        n.dependency_ids = [
                            new_node.node_id if d == failed_node_id else d for d in n.dependency_ids
                        ]
                        if n.execution_status == TaskNodeStatus.BLOCKED.value:
                            n.execution_status = TaskNodeStatus.PENDING.value
                            n.blocked_reason = None

        try:
            self.recompute_schedule(known_proposals=known_proposals)
            self.validate(known_proposals=known_proposals, allow_blocked=True)
        except Exception:
            self._restore_from_dict(snapshot)
            raise
        return self

    def update_dependencies(
        self,
        node_id: str,
        new_dependency_ids: Sequence[str],
        *,
        known_proposals: Optional[Union[Dict[str, Any], Iterable[Any]]] = None,
    ) -> "TaskGraph":
        """Safely update dependency_ids on a non-completed node while preventing cycles."""
        node = self.get_node(node_id)
        if node is None:
            raise TaskGraphValidationError(f"Cannot update dependencies on unknown node '{node_id}'.")
        if node.execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value:
            raise TaskGraphValidationError(f"Cannot mutate dependencies of completed verified node '{node_id}'.")

        snapshot = self.to_dict()
        node.dependency_ids = [str(d).strip() for d in new_dependency_ids]
        if node.execution_status == TaskNodeStatus.BLOCKED.value and not node.contradictory:
            node.execution_status = TaskNodeStatus.PENDING.value
            node.blocked_reason = None
        try:
            self.recompute_schedule(known_proposals=known_proposals)
            self.validate(known_proposals=known_proposals, allow_blocked=True)
        except Exception:
            self._restore_from_dict(snapshot)
            raise
        return self

    def cancel_node(
        self,
        node_id: str,
        *,
        reason: str = "Obsolete after replanning",
        known_proposals: Optional[Union[Dict[str, Any], Iterable[Any]]] = None,
    ) -> "TaskGraph":
        """Mark a pending/blocked/failed node as cancelled; refuses cancelling completed_verified nodes."""
        node = self.get_node(node_id)
        if node is None:
            raise TaskGraphValidationError(f"Cannot cancel unknown node '{node_id}'.")
        if node.execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value:
            raise TaskGraphValidationError(f"Cannot cancel completed verified node '{node_id}'.")
        node.execution_status = TaskNodeStatus.CANCELLED.value
        node.blocked_reason = _sanitize_graph_text(reason, max_bytes=256) or "cancelled"
        node.updated_at = time.time()
        return self.recompute_schedule(known_proposals=known_proposals)

    def begin_node_recovery(self, node_id: str, owner_id: str) -> str:
        """Atomically claim exclusive recovery ownership for a single TaskNode."""
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise TaskGraphValidationError("Recovery ownership requires a non-empty owner_id.")
        eff_owner = owner_id.strip()
        node = self.get_node(node_id)
        if node is None:
            raise TaskGraphValidationError(f"Cannot recover unknown task node '{node_id}'.")
        cur_res_owner = node.reserved_by or (self.active_reservations.get(node_id) or {}).get("owner_id")
        if cur_res_owner is not None:
            raise TaskGraphConflictError(
                f"Cannot initiate recovery on node '{node_id}' while reserved by active worker '{cur_res_owner}'."
            )
        cur_rec_owner = self._recovery_owners.get(node_id)
        if cur_rec_owner is not None and cur_rec_owner != eff_owner:
            raise TaskGraphConflictError(
                f"Node '{node_id}' already has an active recovery owner '{cur_rec_owner}'."
            )
        self._recovery_owners[node_id] = eff_owner
        return eff_owner

    def end_node_recovery(self, node_id: str, owner_id: str) -> None:
        """Release per-node recovery ownership only if held by ``owner_id``."""
        eff_owner = str(owner_id or "").strip()
        cur_owner = self._recovery_owners.get(node_id)
        if cur_owner is None:
            return
        if cur_owner != eff_owner:
            raise TaskGraphConflictError(
                f"Non-owner '{eff_owner}' cannot release recovery lock held by '{cur_owner}' on '{node_id}'."
            )
        self._recovery_owners.pop(node_id, None)

    def get_blocked_dependents(self, node_id: str) -> List[Dict[str, Any]]:
        """Return direct and transitive downstream nodes blocked by ``node_id``."""
        node_map = {n.node_id: n for n in self.nodes}
        if node_id not in node_map:
            return []
        children: Dict[str, List[str]] = {n.node_id: [] for n in self.nodes}
        for n in self.nodes:
            for d in n.dependency_ids:
                if d in children:
                    children[d].append(n.node_id)
        seen: Set[str] = set()
        stack: List[str] = list(children.get(node_id, []))
        dependents: List[TaskNode] = []
        while stack:
            cur = stack.pop(0)
            if cur in seen:
                continue
            seen.add(cur)
            cn = node_map.get(cur)
            if cn is not None:
                dependents.append(cn)
                stack.extend(children.get(cur, []))
        dependents.sort(key=lambda x: (x.order_index, x.node_id))
        return [
            {
                "node_id": dn.node_id,
                "execution_status": dn.execution_status,
                "dependency_ids": list(dn.dependency_ids),
                "blocked_reason": dn.blocked_reason,
            }
            for dn in dependents
        ]

    def record_node_recovery_state(
        self,
        node_id: str,
        recovery_state: Dict[str, Any],
        *,
        transition_status: Optional[str] = None,
        verification_status: Optional[str] = None,
        blocked_reason: Optional[str] = None,
        clear_contradiction: bool = False,
        clear_failure_info: bool = False,
        known_proposals: Optional[Union[Dict[str, Any], Iterable[Any]]] = None,
    ) -> TaskNode:
        """Persist validated recovery metadata and optional safe status transition on a node."""
        node = self.get_node(node_id)
        if node is None:
            raise TaskGraphValidationError(f"Unknown task node '{node_id}'.")
        validated = _validate_node_recovery_state(
            recovery_state,
            context_label=f"TaskNode '{node_id}' recovery_state",
        )
        if node.execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value:
            if transition_status is not None and transition_status != TaskNodeStatus.COMPLETED_VERIFIED.value:
                raise TaskGraphValidationError(
                    f"Cannot transition completed verified node '{node_id}' to '{transition_status}'."
                )
            node.recovery_state = validated
            node.updated_at = time.time()
            self._refresh_summary_and_digest()
            return node

        snapshot = self.to_dict()
        node.recovery_state = validated
        if clear_contradiction:
            node.contradictory = False
        if clear_failure_info:
            node.failure_info = None
        if transition_status is not None:
            if transition_status not in _VALID_NODE_STATUSES:
                raise TaskGraphValidationError(f"Invalid transition_status '{transition_status}'.")
            node.execution_status = transition_status
        if verification_status is not None:
            if verification_status not in _VALID_VERIFICATION_STATUSES:
                raise TaskGraphValidationError(f"Invalid verification_status '{verification_status}'.")
            node.verification_status = verification_status
        node.blocked_reason = _sanitize_graph_text(blocked_reason, max_bytes=256) if blocked_reason else None
        node.updated_at = time.time()
        try:
            self.recompute_schedule(known_proposals=known_proposals)
            self.validate(known_proposals=known_proposals, allow_blocked=True)
        except Exception:
            self._restore_from_dict(snapshot)
            raise
        return node

    def prepare_node_for_retry(
        self,
        node_id: str,
        *,
        new_proposal_id: Optional[str] = None,
        recovery_state: Optional[Dict[str, Any]] = None,
        known_proposals: Optional[Union[Dict[str, Any], Iterable[Any]]] = None,
    ) -> TaskNode:
        """Reset a non-reserved recoverable node to PENDING so scheduler validates readiness cleanly."""
        node = self.get_node(node_id)
        if node is None:
            raise TaskGraphValidationError(f"Cannot retry unknown node '{node_id}'.")
        if node.execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value:
            raise TaskGraphValidationError(f"Completed verified node '{node_id}' must not be retried.")
        if node.execution_status == TaskNodeStatus.CANCELLED.value:
            raise TaskGraphValidationError(f"Cancelled node '{node_id}' cannot be prepared for auto-retry.")
        if node.reserved_by is not None or node_id in self.active_reservations:
            raise TaskGraphConflictError(f"Cannot reset reserved node '{node_id}' while still owned.")

        snapshot = self.to_dict()
        if new_proposal_id:
            clean_pid = str(new_proposal_id).strip()
            if not clean_pid:
                raise TaskGraphValidationError("new_proposal_id must be non-empty.")
            node.proposal_ids = [clean_pid]
        if recovery_state is not None:
            node.recovery_state = _validate_node_recovery_state(
                recovery_state,
                context_label=f"TaskNode '{node_id}' recovery_state",
            )
        node.execution_status = TaskNodeStatus.PENDING.value
        node.verification_status = VerificationStatus.NOT_VERIFIED.value
        node.contradictory = False
        node.failure_info = None
        node.blocked_reason = None
        node.updated_at = time.time()
        try:
            self.recompute_schedule(known_proposals=known_proposals)
            self.validate(known_proposals=known_proposals, allow_blocked=True)
        except Exception:
            self._restore_from_dict(snapshot)
            raise
        return node

    def add_corrective_node(
        self,
        failed_node_id: str,
        corrective_node: TaskNode,
        *,
        make_prerequisite_of_failed: bool = True,
        reset_failed_to_pending: bool = True,
        known_proposals: Optional[Union[Dict[str, Any], Iterable[Any]]] = None,
    ) -> "TaskGraph":
        """Insert a corrective node without removing existing dependencies on any node."""
        target = self.get_node(failed_node_id)
        if target is None:
            raise TaskGraphValidationError(f"Cannot add corrective node for unknown node '{failed_node_id}'.")
        if target.execution_status == TaskNodeStatus.COMPLETED_VERIFIED.value:
            raise TaskGraphValidationError(
                f"Cannot attach corrective prerequisite to completed verified node '{failed_node_id}'."
            )
        snapshot = self.to_dict()
        corr = TaskNode.from_dict(corrective_node.to_dict())
        for existing_dep in target.dependency_ids:
            if existing_dep not in corr.dependency_ids and make_prerequisite_of_failed:
                corr.dependency_ids.append(existing_dep)
        if corr.order_index == 0 and target.order_index > 0:
            corr.order_index = target.order_index
        self.nodes.append(corr)
        if make_prerequisite_of_failed:
            if corr.node_id not in target.dependency_ids:
                target.dependency_ids.append(corr.node_id)
            if reset_failed_to_pending:
                target.execution_status = TaskNodeStatus.PENDING.value
                target.verification_status = VerificationStatus.NOT_VERIFIED.value
                target.contradictory = False
                target.failure_info = None
                target.blocked_reason = None
        target.updated_at = time.time()
        try:
            self.recompute_schedule(known_proposals=known_proposals)
            self.validate(known_proposals=known_proposals, allow_blocked=True)
        except Exception:
            self._restore_from_dict(snapshot)
            raise
        return self

    def _restore_from_dict(self, data: Dict[str, Any]) -> None:
        restored = TaskGraph.from_dict(data, default_execution_id=self.execution_id, verify_integrity=False)
        self.nodes = restored.nodes
        self.edges = restored.edges
        self.status = restored.status
        self.summary = restored.summary
        self.updated_at = restored.updated_at
        self.graph_digest = restored.graph_digest
        self.max_concurrency = restored.max_concurrency
        self.active_reservations = dict(restored.active_reservations)

    def _refresh_summary_and_digest(self) -> None:
        counts = {st.value: 0 for st in TaskNodeStatus}
        for n in self.nodes:
            counts[n.execution_status] = counts.get(n.execution_status, 0) + 1

        total = len(self.nodes)
        next_node = None
        for n in sorted(self.nodes, key=lambda x: (x.order_index, x.node_id)):
            if n.execution_status == TaskNodeStatus.READY.value and n.reserved_by is None:
                next_node = n.node_id
                break

        superseded_cancelled = sum(
            1
            for n in self.nodes
            if n.execution_status == TaskNodeStatus.CANCELLED.value
            and str(n.blocked_reason or "").startswith("Superseded by replacement node ")
        )
        non_superseded_cancelled = counts[TaskNodeStatus.CANCELLED.value] - superseded_cancelled
        active_total = total - superseded_cancelled

        if total == 0:
            self.status = TaskGraphStatus.INITIALIZED.value
        elif counts[TaskNodeStatus.COMPLETED_VERIFIED.value] == total or (
            active_total > 0
            and non_superseded_cancelled == 0
            and counts[TaskNodeStatus.COMPLETED_VERIFIED.value] == active_total
        ):
            self.status = TaskGraphStatus.COMPLETED_VERIFIED.value
        elif counts[TaskNodeStatus.RUNNING.value] > 0:
            self.status = TaskGraphStatus.RUNNING.value
        elif counts[TaskNodeStatus.READY.value] > 0:
            self.status = TaskGraphStatus.READY.value
        elif counts[TaskNodeStatus.FAILED.value] > 0:
            self.status = TaskGraphStatus.FAILED.value
        elif counts[TaskNodeStatus.CANCELLED.value] == total:
            self.status = TaskGraphStatus.CANCELLED.value
        elif counts[TaskNodeStatus.BLOCKED.value] > 0:
            self.status = TaskGraphStatus.BLOCKED.value
        elif (counts[TaskNodeStatus.COMPLETED_VERIFIED.value] + counts[TaskNodeStatus.COMPLETED_UNVERIFIED.value]) == total or (
            active_total > 0
            and non_superseded_cancelled == 0
            and (
                counts[TaskNodeStatus.COMPLETED_VERIFIED.value] + counts[TaskNodeStatus.COMPLETED_UNVERIFIED.value]
            )
            == active_total
        ):
            self.status = TaskGraphStatus.COMPLETED_UNVERIFIED.value
        else:
            self.status = TaskGraphStatus.PENDING.value

        self.summary = {
            "total_nodes": total,
            "completed_verified_count": counts[TaskNodeStatus.COMPLETED_VERIFIED.value],
            "completed_unverified_count": counts[TaskNodeStatus.COMPLETED_UNVERIFIED.value],
            "ready_count": counts[TaskNodeStatus.READY.value],
            "pending_count": counts[TaskNodeStatus.PENDING.value],
            "running_count": counts[TaskNodeStatus.RUNNING.value],
            "failed_count": counts[TaskNodeStatus.FAILED.value],
            "blocked_count": counts[TaskNodeStatus.BLOCKED.value],
            "cancelled_count": counts[TaskNodeStatus.CANCELLED.value],
            "next_eligible_node_id": next_node,
        }
        self.updated_at = time.time()
        self.graph_digest = compute_task_graph_digest(self._raw_payload())

    def _raw_payload(self) -> Dict[str, Any]:
        raw: Dict[str, Any] = {
            "graph_id": self.graph_id,
            "execution_id": self.execution_id,
            "objective": _sanitize_graph_text(self.objective, max_bytes=1024),
            "graph_version": self.graph_version,
            "status": self.status,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "summary": dict(self.summary),
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
        }
        if self.max_concurrency != 1:
            raw["max_concurrency"] = int(self.max_concurrency)
        if self.active_reservations:
            raw["active_reservations"] = {
                str(k): dict(v) for k, v in self.active_reservations.items() if isinstance(v, dict)
            }
        return _strip_forbidden_graph_keys(raw)

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            if not self.graph_digest:
                self._refresh_summary_and_digest()
            payload = self._raw_payload()
            payload["graph_digest"] = self.graph_digest
            return payload

    def to_checkpoint_dict(self) -> Dict[str, Any]:
        """Serialize cleanly for task_checkpoints persistence without any auth grants or raw secrets."""
        with self._lock:
            self._refresh_summary_and_digest()
            payload = self._raw_payload()
            payload["graph_digest"] = self.graph_digest
            return _strip_forbidden_graph_keys(payload)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        default_execution_id: Optional[str] = None,
        verify_integrity: bool = True,
    ) -> "TaskGraph":
        if not isinstance(data, dict):
            raise CorruptedTaskGraphError("TaskGraph payload must be a JSON object.")
        raw_ver = data.get("graph_version")
        if not isinstance(raw_ver, int) or isinstance(raw_ver, bool):
            raise CorruptedTaskGraphError(f"Malformed graph_version: {raw_ver!r}")
        if raw_ver not in SUPPORTED_TASK_GRAPH_VERSIONS:
            raise UnsupportedTaskGraphVersionError(f"Unsupported task graph version: {raw_ver}")

        gid = data.get("graph_id")
        if not isinstance(gid, str) or not gid.strip():
            raise CorruptedTaskGraphError("TaskGraph missing valid 'graph_id'.")
        exec_id = data.get("execution_id") or default_execution_id
        if not isinstance(exec_id, str) or not exec_id.strip():
            raise CorruptedTaskGraphError(f"TaskGraph '{gid}' missing valid 'execution_id'.")
        status = data.get("status", TaskGraphStatus.INITIALIZED.value)
        if not isinstance(status, str) or status not in _VALID_GRAPH_STATUSES:
            raise CorruptedTaskGraphError(f"TaskGraph '{gid}' has invalid status: {status!r}.")

        raw_mc = data.get("max_concurrency", 1)
        if not isinstance(raw_mc, int) or isinstance(raw_mc, bool) or raw_mc < 1:
            raise CorruptedTaskGraphError(f"TaskGraph '{gid}' has invalid max_concurrency: {raw_mc!r}.")
        raw_res = data.get("active_reservations", {})
        if not isinstance(raw_res, dict):
            raise CorruptedTaskGraphError(f"TaskGraph '{gid}' active_reservations must be a dictionary.")
        clean_res: Dict[str, Dict[str, Any]] = {}
        for rk, rv in raw_res.items():
            if not isinstance(rk, str) or not rk.strip() or not isinstance(rv, dict):
                raise CorruptedTaskGraphError(f"TaskGraph '{gid}' has malformed reservation entry '{rk}'.")
            clean_res[rk.strip()] = _strip_forbidden_graph_keys(dict(rv))

        raw_nodes = data.get("nodes")
        raw_edges = data.get("edges", [])
        if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
            raise CorruptedTaskGraphError(f"TaskGraph '{gid}' nodes and edges must be lists.")

        nodes = [TaskNode.from_dict(n) for n in raw_nodes]
        edges = [TaskGraphEdge.from_dict(e) for e in raw_edges]
        summary = data.get("summary") or {}
        if not isinstance(summary, dict):
            raise CorruptedTaskGraphError(f"TaskGraph '{gid}' summary must be a dict.")

        stored_digest = str(data.get("graph_digest") or "")
        if verify_integrity:
            expected_digest = compute_task_graph_digest(
                {
                    **data,
                    "graph_version": raw_ver,
                    "graph_id": gid.strip(),
                    "execution_id": exec_id.strip(),
                }
            )
            if not stored_digest or stored_digest != expected_digest:
                raise CorruptedTaskGraphError(
                    f"TaskGraph '{gid}' failed integrity verification (digest mismatch)."
                )

        now = time.time()
        graph = cls(
            graph_id=gid.strip(),
            execution_id=exec_id.strip(),
            objective=_sanitize_graph_text(data.get("objective") or "", max_bytes=1024),
            nodes=nodes,
            edges=edges,
            status=status,
            graph_version=raw_ver,
            created_at=float(data.get("created_at") or now),
            updated_at=float(data.get("updated_at") or now),
            summary=dict(summary),
            graph_digest=stored_digest,
            max_concurrency=raw_mc,
            active_reservations=clean_res,
        )
        return graph


def _sync_graph_method(fn: Any) -> Any:
    def wrapper(self: "TaskGraph", *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return fn(self, *args, **kwargs)
    return wrapper


for _tg_meth_name in (
    "validate",
    "topological_order",
    "recompute_schedule",
    "apply_observation",
    "add_node",
    "replace_failed_node",
    "update_dependencies",
    "cancel_node",
    "get_node",
    "find_nodes_for_call",
    "get_blocked_nodes",
    "get_failed_nodes",
    "get_unverified_nodes",
    "begin_node_recovery",
    "end_node_recovery",
    "get_blocked_dependents",
    "record_node_recovery_state",
    "prepare_node_for_retry",
    "add_corrective_node",
):
    _orig_tg_fn = getattr(TaskGraph, _tg_meth_name, None)
    if callable(_orig_tg_fn):
        _wrapped_tg = _sync_graph_method(_orig_tg_fn)
        _wrapped_tg.__name__ = _orig_tg_fn.__name__
        _wrapped_tg.__doc__ = _orig_tg_fn.__doc__
        setattr(TaskGraph, _tg_meth_name, _wrapped_tg)
del _tg_meth_name, _orig_tg_fn


def reconstruct_task_graph_from_records(
    execution_id: str,
    *,
    objective: str = "",
    pending_actions: Sequence[Any] = (),
    events: Sequence[ExecutionEvent] = (),
    rec: Optional[ExecutionRecord] = None,
) -> Tuple[Optional[TaskGraph], List[str]]:
    """Deterministically reconstruct a TaskGraph from persisted proposals and execution events."""
    if not execution_id:
        return None, ["Cannot reconstruct TaskGraph without execution_id."]

    errors: List[str] = []
    # 1. Gather ordered call/proposal records from events & pending_actions
    ev_by_cid: Dict[str, Dict[str, Any]] = {}
    ev_order: List[str] = []
    for ev in events:
        cid = ev.correlation_id
        if not cid:
            continue
        meta = dict(ev.metadata or {})
        if cid not in ev_by_cid:
            ev_order.append(cid)
            ev_by_cid[cid] = {
                "call_id": cid,
                "proposal_id": None,
                "tool_id": ev.tool_id or "",
                "arguments": {},
                "depends_on": [],
                "proposed": False,
                "dispatched": False,
                "has_result": False,
                "result_status": None,
                "error": None,
                "error_type": None,
                "verified": False,
                "verification_failed": False,
                "contradictory": False,
                "evidence_level": None,
                "execution_status": None,
                "verification_status": None,
                "recoverable": None,
                "is_external": bool(ev.is_external),
                "source_id": ev.source_id,
                "source_type": ev.source_type,
                "order_index": len(ev_order) - 1,
                "created_at": float(ev.timestamp),
                "updated_at": float(ev.timestamp),
            }
        st = ev_by_cid[cid]
        st["updated_at"] = max(float(st["updated_at"]), float(ev.timestamp))
        if ev.tool_id and not st["tool_id"]:
            st["tool_id"] = str(ev.tool_id)
        if ev.is_external:
            st["is_external"] = True
            st["source_id"] = ev.source_id or st["source_id"]
            st["source_type"] = ev.source_type or st["source_type"]
        et = ev.event_type
        if et == LedgerEventType.TOOL_SELECTED.value:
            st["proposed"] = True
            if meta.get("proposal_id"):
                st["proposal_id"] = str(meta["proposal_id"])
            if isinstance(meta.get("depends_on"), list):
                st["depends_on"] = [str(d).strip() for d in meta["depends_on"] if str(d).strip()]
            if meta.get("proposal_order") is not None:
                try:
                    st["order_index"] = int(meta["proposal_order"])
                except Exception:
                    pass
            if isinstance(meta.get("arguments"), dict):
                st["arguments"] = dict(meta["arguments"])
        elif et == LedgerEventType.ARGUMENT_VALIDATED.value and ev.status == "invalid":
            st["has_result"] = True
            st["result_status"] = "rejected"
            st["error_type"] = str(meta.get("error_type") or "SCHEMA_VALIDATION_ERROR")
            st["error"] = str(meta.get("detail") or meta.get("reason") or "invalid_arguments")
            st["recoverable"] = True
        elif et == LedgerEventType.AUTHORIZATION_DECISION.value and ev.status == "denied":
            st["has_result"] = True
            st["result_status"] = "rejected"
            st["error_type"] = str(meta.get("error_type") or "AUTHORIZATION_DENIED")
            st["error"] = str(meta.get("detail") or "Authorization denied")
            st["recoverable"] = False
        elif et == LedgerEventType.TOOL_EXECUTION_START.value:
            st["dispatched"] = True
        elif et == LedgerEventType.TOOL_EXECUTION_RESULT.value:
            st["has_result"] = True
            st["result_status"] = ev.status
            if meta.get("error"):
                st["error"] = str(meta["error"])
            if meta.get("error_type"):
                st["error_type"] = str(meta["error_type"])
            if isinstance(meta.get("verified"), bool):
                st["verified"] = meta["verified"]
        elif et == LedgerEventType.OBSERVATION_UPDATED.value:
            if meta.get("evidence_level"):
                st["evidence_level"] = str(meta["evidence_level"])
            if meta.get("execution_status"):
                st["execution_status"] = str(meta["execution_status"])
            if meta.get("verification_status"):
                st["verification_status"] = str(meta["verification_status"])
            if meta.get("evidence_level") == "CONTRADICTED" or meta.get("execution_status") == "contradictory_evidence":
                st["contradictory"] = True
            if isinstance(meta.get("verification_failed"), bool) and meta["verification_failed"]:
                st["verification_failed"] = True
            guid = meta.get("recovery_guidance")
            if isinstance(guid, dict) and "is_recoverable" in guid:
                st["recoverable"] = bool(guid["is_recoverable"])
        elif et == LedgerEventType.VERIFICATION_RECORDED.value:
            if ev.status == VerificationStatus.VERIFIED.value:
                st["verified"] = True
                st["verification_failed"] = False
            elif ev.status == VerificationStatus.VERIFICATION_FAILED.value:
                st["verification_failed"] = True
        elif et == LedgerEventType.RECOVERY_ATTEMPT.value:
            rs = dict(st.get("recovery_state") or {"recovery_version": RECOVERY_STATE_VERSION, "attempt_count": 0, "recovery_history": []})
            att_n = int(meta.get("attempt") or 0)
            rs["attempt_count"] = max(int(rs.get("attempt_count") or 0), att_n)
            if meta.get("recovery_decision"):
                rs["recovery_decision"] = str(meta["recovery_decision"])
            if ev.status:
                rs["recovery_status"] = str(ev.status)
            if meta.get("failure_category"):
                rs["previous_failure_category"] = str(meta["failure_category"])
            if meta.get("retry_deadline") is not None:
                rs["retry_deadline"] = float(meta["retry_deadline"])
            if isinstance(meta.get("may_have_side_effect"), bool):
                rs["may_have_side_effect"] = meta["may_have_side_effect"]
            if isinstance(meta.get("reconciliation_required"), bool):
                rs["reconciliation_required"] = meta["reconciliation_required"]
            if meta.get("next_recovery_action"):
                rs["next_recovery_action"] = str(meta["next_recovery_action"])
            st["recovery_state"] = rs

    pa_by_cid: Dict[str, Any] = {}
    for pa in pending_actions:
        cid = str(getattr(pa, "original_call_id", None) or (pa.get("original_call_id") if isinstance(pa, dict) else "") or "")
        if cid:
            pa_by_cid[cid] = pa
            if cid not in ev_order:
                ev_order.append(cid)

    if not ev_order:
        return None, []

    nodes: List[TaskNode] = []
    ref_to_node: Dict[str, str] = {}
    raw_items: List[Tuple[str, Optional[Dict[str, Any]], Any]] = []
    for cid in ev_order:
        ev_d = ev_by_cid.get(cid)
        pa = pa_by_cid.get(cid)
        pid = None
        if pa is not None:
            pid = str(getattr(pa, "proposal_id", None) or (pa.get("proposal_id") if isinstance(pa, dict) else "") or "")
        if not pid and ev_d is not None:
            pid = ev_d.get("proposal_id") or f"prop_{cid}"
        nid = f"node_{cid}"
        ref_to_node[nid] = nid
        ref_to_node[cid] = nid
        if pid:
            ref_to_node[pid] = nid
        raw_items.append((cid, ev_d, pa))

    for idx, (cid, ev_d, pa) in enumerate(raw_items):
        nid = f"node_{cid}"
        pid = None
        tool_id = ""
        raw_deps: List[str] = []
        order_idx = idx
        ext_src = None
        pa_status = None
        pa_blk = None
        if pa is not None:
            pid = str(getattr(pa, "proposal_id", None) or (pa.get("proposal_id") if isinstance(pa, dict) else "") or f"prop_{cid}")
            tool_id = str(getattr(pa, "tool_id", None) or (pa.get("tool_id") if isinstance(pa, dict) else "") or "")
            raw_deps = list(getattr(pa, "depends_on", None) or (pa.get("depends_on") if isinstance(pa, dict) else []) or [])
            order_idx = int(getattr(pa, "proposal_order", idx) if not isinstance(pa, dict) else pa.get("proposal_order", idx))
            ext_src = getattr(pa, "external_source", None) if not isinstance(pa, dict) else pa.get("external_source")
            pa_status = str(getattr(pa, "status", None) or (pa.get("status") if isinstance(pa, dict) else "") or "")
            pa_blk = getattr(pa, "blocked_reason", None) if not isinstance(pa, dict) else pa.get("blocked_reason")
        if ev_d is not None:
            tool_id = tool_id or ev_d["tool_id"]
            if not raw_deps and ev_d["depends_on"]:
                raw_deps = list(ev_d["depends_on"])
            if ext_src is None and ev_d["is_external"]:
                ext_src = {
                    "is_external": True,
                    "source_id": ev_d["source_id"] or "external",
                    "source_type": ev_d["source_type"] or "external",
                    "external_tool_name": tool_id,
                }

        resolved_deps: List[str] = []
        for dep_ref in raw_deps:
            mapped = ref_to_node.get(dep_ref)
            if mapped is None:
                errors.append(
                    f"Incomplete reconstruction for node '{nid}': missing prerequisite reference '{dep_ref}'."
                )
                resolved_deps.append(dep_ref)
            elif mapped not in resolved_deps:
                resolved_deps.append(mapped)

        exec_st = TaskNodeStatus.PENDING.value
        ver_st = VerificationStatus.NOT_VERIFIED.value
        fail_info = None
        blk_reason = None
        contradictory = False
        obs_refs: List[Dict[str, Any]] = []

        if ev_d is not None:
            if ev_d["contradictory"] or ev_d["verification_failed"]:
                exec_st = TaskNodeStatus.BLOCKED.value
                ver_st = VerificationStatus.VERIFICATION_FAILED.value
                contradictory = bool(ev_d["contradictory"])
                blk_reason = "Verification failed or contradictory evidence recorded."
                fail_info = {"error": ev_d["error"] or blk_reason, "recoverable": True}
            elif ev_d["dispatched"] and not ev_d["has_result"]:
                exec_st = TaskNodeStatus.RUNNING.value
                ver_st = VerificationStatus.INTERRUPTED.value
                blk_reason = "Dispatched prior to interruption without a persisted result."
            elif ev_d["result_status"] == "cancelled":
                exec_st = TaskNodeStatus.CANCELLED.value
                ver_st = VerificationStatus.INTERRUPTED.value
                blk_reason = ev_d["error"] or "cancelled"
                fail_info = {"error": blk_reason, "may_have_side_effect": bool(ev_d["dispatched"])}
            elif ev_d["result_status"] in ("failed", "timed_out", "rejected") or (ev_d["has_result"] and ev_d["error"] and ev_d["result_status"] != "succeeded"):
                exec_st = TaskNodeStatus.FAILED.value
                ver_st = VerificationStatus.NOT_VERIFIED.value
                blk_reason = ev_d["error"] or ev_d["result_status"]
                fail_info = {
                    "error": blk_reason,
                    "error_type": ev_d["error_type"],
                    "recoverable": ev_d["recoverable"] if ev_d["recoverable"] is not None else (ev_d["error_type"] not in ("AUTHORIZATION_DENIED", "TOOL_NOT_FOUND")),
                }
            elif ev_d["verified"] or ev_d["evidence_level"] == "CONFIRMED":
                exec_st = TaskNodeStatus.COMPLETED_VERIFIED.value
                ver_st = VerificationStatus.VERIFIED.value
            elif ev_d["has_result"] and ev_d["result_status"] == "succeeded":
                exec_st = TaskNodeStatus.COMPLETED_UNVERIFIED.value
                ver_st = VerificationStatus.NOT_VERIFIED.value

            if ev_d["has_result"] or ev_d["evidence_level"]:
                obs_refs.append({
                    k: v
                    for k, v in {
                        "call_id": cid,
                        "evidence_level": ev_d["evidence_level"],
                        "execution_status": ev_d["execution_status"] or ev_d["result_status"],
                        "verification_status": ev_d["verification_status"] or ver_st,
                        "verified": exec_st == TaskNodeStatus.COMPLETED_VERIFIED.value,
                    }.items()
                    if v is not None
                })

        if pa_status == "cancelled" and exec_st == TaskNodeStatus.PENDING.value:
            exec_st = TaskNodeStatus.CANCELLED.value
            ver_st = VerificationStatus.INTERRUPTED.value
            blk_reason = pa_blk or "cancelled"
        elif pa_status in ("blocked", "expired") and exec_st == TaskNodeStatus.PENDING.value:
            exec_st = TaskNodeStatus.BLOCKED.value
            blk_reason = pa_blk or pa_status
        elif pa_status == "failed" and exec_st == TaskNodeStatus.PENDING.value:
            exec_st = TaskNodeStatus.FAILED.value
            blk_reason = pa_blk or "failed"
            fail_info = {"error": blk_reason}

        if rec is not None and exec_st == TaskNodeStatus.PENDING.value:
            if rec.status == ExecutionStatus.ABANDONED.value:
                exec_st = TaskNodeStatus.CANCELLED.value
                blk_reason = "task_abandoned"
            elif rec.status == ExecutionStatus.CANCELLED.value:
                exec_st = TaskNodeStatus.CANCELLED.value
                blk_reason = blk_reason or "task_cancelled"

        nodes.append(
            TaskNode(
                node_id=nid,
                description=_sanitize_graph_text(f"{tool_id or 'action'} ({cid})", max_bytes=256),
                proposal_ids=[pid] if pid else [],
                dependency_ids=resolved_deps,
                execution_status=exec_st,
                verification_status=ver_st,
                failure_info=fail_info,
                observation_refs=obs_refs,
                contradictory=contradictory,
                recoverable=ev_d["recoverable"] if ev_d else None,
                blocked_reason=blk_reason,
                order_index=order_idx,
                external_source=ext_src,
                recovery_state=ev_d.get("recovery_state") if ev_d else None,
            )
        )

    eff_obj = objective or (rec.task_prompt if rec is not None else "")
    graph = TaskGraph(
        graph_id=f"tg_rec_{execution_id[:12]}",
        execution_id=execution_id,
        objective=eff_obj,
        nodes=nodes,
    )
    if errors:
        return graph, errors
    try:
        graph.recompute_schedule(known_proposals={getattr(p, "proposal_id", f"p{i}"): p for i, p in enumerate(pending_actions)})
        graph.validate(allow_blocked=True)
    except TaskGraphValidationError as exc:
        errors.append(str(exc))
    return graph, errors
