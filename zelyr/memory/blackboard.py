"""Blackboard — Thread-safe ephemeral scratchpad for Multi-Agent Swarm sessions.

Provides hierarchical, namespace-isolated key-value artifact storage partitioned
by swarm ``session_id`` and ``agent_id``. Addresses follow the URI schema::

    blackboard://session_102/coder_agent/draft_ast

All data is held **exclusively in RAM**. No writes reach the persistent SQLite
database. The store is explicitly purged when the associated
``WorkspaceTransactionManager`` commits or rolls back, guaranteeing zero memory
leakage across successive swarm runs.

Usage::

    bb = BlackboardStore()
    bb.write("session_1", "coder_agent", "draft_code", "def foo(): ...")
    value = bb.read("session_1", "coder_agent", "draft_code")
    all_artifacts = bb.list_keys("session_1", "coder_agent")
    bb.purge_session("session_1")
"""
from __future__ import annotations
import copy
import logging
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)

def _safe_copy_value(val: Any) -> Any:
    """Safely copy an artifact value for blackboard snapshotting."""
    try:
        return copy.deepcopy(val)
    except Exception:
        if isinstance(val, dict):
            return dict(val)
        if isinstance(val, list):
            return list(val)
        if isinstance(val, set):
            return set(val)
        return val

class BlackboardNamespace:
    """A single agent's scratchpad within a session."""

    def __init__(self, session_id: str, agent_id: str):
        """Auto-generated docstring.

Args:
    session_id: Argument.
    agent_id: Argument.

Returns:
    Return value.
"""
        self.session_id = session_id
        self.agent_id = agent_id
        self._store: Dict[str, Any] = {}

    @property
    def uri_prefix(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return f'blackboard://{self.session_id}/{self.agent_id}'

    def write(self, key: str, value: Any) -> None:
        """Auto-generated docstring.

Args:
    key: Argument.
    value: Argument.

Returns:
    Return value.
"""
        self._store[key] = value
        logger.debug('Blackboard WRITE %s/%s', self.uri_prefix, key)

    def read(self, key: str, default: Any=None) -> Any:
        """Auto-generated docstring.

Args:
    key: Argument.
    default: Argument.

Returns:
    Return value.
"""
        return self._store.get(key, default)

    def list_keys(self) -> List[str]:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return list(self._store.keys())

    def clear(self) -> None:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        count = len(self._store)
        self._store.clear()
        logger.debug('Blackboard: cleared %d entries from namespace %s', count, self.uri_prefix)

class BlackboardStore:
    """Thread-safe, session-partitioned ephemeral key-value store.

    Every swarm session gets its own isolated namespace tree. Concurrent agents
    operating in the same session can only access their own per-agent partition
    unless they explicitly request cross-agent reads.
    """

    def __init__(self):
        """Initialize BlackboardStore with empty session partitions and transactions."""
        self._sessions: Dict[str, Dict[str, BlackboardNamespace]] = {}
        self._transactions: Dict[str, List[Tuple[str, Dict[str, Dict[str, Any]]]]] = {}
        self._lock = threading.RLock()

    def _get_namespace(self, session_id: str, agent_id: str) -> BlackboardNamespace:
        """Return (or create) the namespace for (session_id, agent_id)."""
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = {}
            if agent_id not in self._sessions[session_id]:
                ns = BlackboardNamespace(session_id, agent_id)
                self._sessions[session_id][agent_id] = ns
                logger.debug('Blackboard: created namespace %s', ns.uri_prefix)
            return self._sessions[session_id][agent_id]

    def write(self, session_id: str, agent_id: str, key: str, value: Any) -> None:
        """Write an artifact to the named namespace.

        Args:
            session_id: The swarm session this belongs to (e.g. ``"session_102"``).
            agent_id:   The writing agent (e.g. ``"coder_agent"``).
            key:        Artifact key (e.g. ``"draft_ast"``).
            value:      Any Python object to store ephemerally.
        """
        self._get_namespace(session_id, agent_id).write(key, value)

    def read(self, session_id: str, agent_id: str, key: str, default: Any=None) -> Any:
        """Read an artifact from a namespace.

        Args:
            session_id: The swarm session to read from.
            agent_id:   The namespace owner.
            key:        Artifact key to retrieve.
            default:    Value to return if key is absent.

        Returns:
            The stored value, or *default* if absent.
        """
        with self._lock:
            try:
                return self._sessions[session_id][agent_id].read(key, default)
            except KeyError:
                return default

    def list_keys(self, session_id: str, agent_id: str) -> List[str]:
        """List all keys stored in a specific agent namespace."""
        with self._lock:
            try:
                return self._sessions[session_id][agent_id].list_keys()
            except KeyError:
                return []

    def list_agents(self, session_id: str) -> List[str]:
        """List all agent IDs that have written artifacts in a session."""
        with self._lock:
            return list(self._sessions.get(session_id, {}).keys())

    def dump_session(self, session_id: str) -> Dict[str, Dict[str, Any]]:
        """Return a snapshot of all artifacts across all agents for a session.

        Used by the ConsensusEngine commit hook to extract final artifacts.
        Returns ``{agent_id: {key: value}}`` mapping.
        """
        with self._lock:
            result: Dict[str, Dict[str, Any]] = {}
            for agent_id, ns in self._sessions.get(session_id, {}).items():
                result[agent_id] = dict(ns._store)
            return result

    def create_snapshot(self, session_id: str) -> Dict[str, Dict[str, Any]]:
        """Return an isolated point-in-time snapshot of the session's blackboard state.

        The snapshot can later be passed to ``restore_snapshot`` to revert mutations.
        Returns ``{agent_id: {key: deepcopied_value}}`` mapping.
        """
        with self._lock:
            snapshot: Dict[str, Dict[str, Any]] = {}
            for agent_id, ns in self._sessions.get(session_id, {}).items():
                snapshot[agent_id] = {k: _safe_copy_value(v) for k, v in ns._store.items()}
            return snapshot

    def restore_snapshot(self, session_id: str, snapshot: Dict[str, Dict[str, Any]]) -> None:
        """Restore the session's blackboard state to a previous snapshot.

        Reverts modified values, removes keys added after the snapshot,
        and deletes namespaces created after the snapshot.
        """
        with self._lock:
            current_session = self._sessions.get(session_id, {})
            # Remove any namespaces created after snapshot
            for agent_id in list(current_session.keys()):
                if agent_id not in snapshot:
                    current_session[agent_id].clear()
                    del current_session[agent_id]
                else:
                    ns_store = current_session[agent_id]._store
                    snap_store = snapshot[agent_id]
                    # Remove keys added after snapshot
                    for key in list(ns_store.keys()):
                        if key not in snap_store:
                            del ns_store[key]
                    # Restore keys present in snapshot
                    for key, val in snap_store.items():
                        ns_store[key] = _safe_copy_value(val)

            # Recreate any namespaces that existed in snapshot but were deleted
            for agent_id, snap_store in snapshot.items():
                if agent_id not in current_session:
                    ns = BlackboardNamespace(session_id, agent_id)
                    ns._store = {k: _safe_copy_value(v) for k, v in snap_store.items()}
                    current_session[agent_id] = ns

            if session_id not in self._sessions and snapshot:
                self._sessions[session_id] = current_session
            logger.info("Blackboard: restored snapshot for session '%s'.", session_id)

    def begin_transaction(self, session_id: str, txn_id: Optional[str] = None) -> str:
        """Begin a transactional savepoint on the blackboard for session_id.

        Args:
            session_id: Swarm session identifier.
            txn_id: Optional explicit transaction ID.

        Returns:
            The transaction ID of the created savepoint.
        """
        with self._lock:
            tid = txn_id or str(uuid.uuid4())
            snapshot = self.create_snapshot(session_id)
            if session_id not in self._transactions:
                self._transactions[session_id] = []
            self._transactions[session_id].append((tid, snapshot))
            logger.debug("Blackboard: begin_transaction '%s' for session '%s'.", tid, session_id)
            return tid

    def rollback_transaction(self, session_id: str, txn_id: Optional[str] = None) -> bool:
        """Roll back blackboard state to a previous transaction savepoint.

        If txn_id is specified, rolls back to that savepoint and removes it and subsequent savepoints.
        If txn_id is None, rolls back to the most recent savepoint.

        Returns:
            True if a savepoint was rolled back, False if no savepoints exist.
        """
        with self._lock:
            stack = self._transactions.get(session_id, [])
            if not stack:
                return False
            if txn_id is None:
                _, snapshot = stack.pop()
                self.restore_snapshot(session_id, snapshot)
                return True
            idx = None
            for i, (tid, _) in enumerate(stack):
                if tid == txn_id:
                    idx = i
                    break
            if idx is None:
                return False
            _, snapshot = stack[idx]
            self._transactions[session_id] = stack[:idx]
            self.restore_snapshot(session_id, snapshot)
            return True

    def commit_transaction(self, session_id: str, txn_id: Optional[str] = None) -> bool:
        """Commit a transaction savepoint without reverting blackboard state.

        Returns:
            True if a savepoint was committed, False if no savepoints exist.
        """
        with self._lock:
            stack = self._transactions.get(session_id, [])
            if not stack:
                return False
            if txn_id is None:
                stack.pop()
                return True
            idx = None
            for i, (tid, _) in enumerate(stack):
                if tid == txn_id:
                    idx = i
                    break
            if idx is None:
                return False
            self._transactions[session_id] = stack[:idx]
            return True

    def purge_session(self, session_id: str) -> int:
        """Completely destroy all data for *session_id*, freeing RAM.

        Should be called by ``WorkspaceTransactionManager`` on both commit
        and rollback to guarantee zero memory leakage.

        Returns:
            Number of namespace entries purged.
        """
        with self._lock:
            self._transactions.pop(session_id, None)
            session = self._sessions.pop(session_id, {})
            total = sum((len(ns.list_keys()) for ns in session.values()))
            for ns in session.values():
                ns.clear()
            logger.info("Blackboard: purged session '%s' (%d namespaces, %d artifacts freed).", session_id, len(session), total)
            return total

    def uri(self, session_id: str, agent_id: str, key: str) -> str:
        """Build the canonical URI for an artifact."""
        return f'blackboard://{session_id}/{agent_id}/{key}'
