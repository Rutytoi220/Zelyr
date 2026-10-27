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
import logging
import threading
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)

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
        """Auto-generated docstring.


Returns:
    Return value.
"""
        self._sessions: Dict[str, Dict[str, BlackboardNamespace]] = {}
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

    def purge_session(self, session_id: str) -> int:
        """Completely destroy all data for *session_id*, freeing RAM.

        Should be called by ``WorkspaceTransactionManager`` on both commit
        and rollback to guarantee zero memory leakage.

        Returns:
            Number of namespace entries purged.
        """
        with self._lock:
            session = self._sessions.pop(session_id, {})
            total = sum((len(ns.list_keys()) for ns in session.values()))
            for ns in session.values():
                ns.clear()
            logger.info("Blackboard: purged session '%s' (%d namespaces, %d artifacts freed).", session_id, len(session), total)
            return total

    def uri(self, session_id: str, agent_id: str, key: str) -> str:
        """Build the canonical URI for an artifact."""
        return f'blackboard://{session_id}/{agent_id}/{key}'
