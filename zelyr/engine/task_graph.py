"""Dependency-aware task graph model, DAG validation, scheduling, and recovery for Zelyr 1.4."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
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
    }
)


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
    raw = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode(
        "utf-8", errors="replace"
    )
    return hashlib.sha256(raw).hexdigest()


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
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        safe_fail, _, _ = sanitize_ledger_payload(_strip_forbidden_graph_keys(self.failure_info), max_bytes=1024) if self.failure_info else ({}, False, False)
        safe_obs_wrap, _, _ = sanitize_ledger_payload({"refs": list(self.observation_refs)[-16:]}, max_bytes=4096)
        safe_obs = list(safe_obs_wrap.get("refs") or []) if isinstance(safe_obs_wrap.get("refs"), list) else []
        return {
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

    def __post_init__(self) -> None:
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
        for node in self.topological_order():
            if node.execution_status == TaskNodeStatus.READY.value:
                return node
        return None

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

    def _restore_from_dict(self, data: Dict[str, Any]) -> None:
        restored = TaskGraph.from_dict(data, default_execution_id=self.execution_id, verify_integrity=False)
        self.nodes = restored.nodes
        self.edges = restored.edges
        self.status = restored.status
        self.summary = restored.summary
        self.updated_at = restored.updated_at
        self.graph_digest = restored.graph_digest

    def _refresh_summary_and_digest(self) -> None:
        counts = {st.value: 0 for st in TaskNodeStatus}
        for n in self.nodes:
            counts[n.execution_status] = counts.get(n.execution_status, 0) + 1

        total = len(self.nodes)
        next_node = None
        for n in sorted(self.nodes, key=lambda x: (x.order_index, x.node_id)):
            if n.execution_status == TaskNodeStatus.READY.value:
                next_node = n.node_id
                break

        if total == 0:
            self.status = TaskGraphStatus.INITIALIZED.value
        elif counts[TaskNodeStatus.COMPLETED_VERIFIED.value] == total:
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
        elif (counts[TaskNodeStatus.COMPLETED_VERIFIED.value] + counts[TaskNodeStatus.COMPLETED_UNVERIFIED.value]) == total:
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
        return _strip_forbidden_graph_keys({
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
        })

    def to_dict(self) -> Dict[str, Any]:
        if not self.graph_digest:
            self._refresh_summary_and_digest()
        payload = self._raw_payload()
        payload["graph_digest"] = self.graph_digest
        return payload

    def to_checkpoint_dict(self) -> Dict[str, Any]:
        """Serialize cleanly for task_checkpoints persistence without any auth grants or raw secrets."""
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
        )
        return graph


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
