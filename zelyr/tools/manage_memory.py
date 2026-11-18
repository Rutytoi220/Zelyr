"""Canonical Autonomous Memory Actuator Tool (manage_memory) for Zelyr.

Provides Ring-1 Tier-1 deterministic memory lifecycle management:
- store: Store new facts with category validation and automatic key-based superseding.
- search: Full-text search over memories using FTS5.
- get: Retrieve memory by ID or unique key.
- delete: Soft-delete memory by ID or unique key.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from zelyr.db.memory import MemoryStore, VALID_CATEGORIES, get_db_path
from zelyr.tools.core import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)

REQUIRED_RING = 1
TIER = 1
TUI_HINT = "Updating persistent memory store..."

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "manage_memory",
        "description": (
            "Tier 1 Persistent Memory Lifecycle: Store, search, retrieve, and delete persistent "
            "user preferences, environment details, homelab inventory, and workflow rules. "
            "Actions: 'store' (persist fact, superseding prior active key if provided), "
            "'search' (FTS5 search), 'get' (retrieve by memory_id or key), 'delete' (soft-delete)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["store", "search", "get", "delete"],
                    "description": "The memory lifecycle action to perform.",
                },
                "content": {
                    "type": "string",
                    "description": "Memory text to store (used with action='store').",
                },
                "category": {
                    "type": "string",
                    "enum": ["preference", "environment", "homelab", "workflow", "capability"],
                    "description": "Memory classification category (default: 'preference').",
                },
                "key": {
                    "type": "string",
                    "description": "Optional unique entity key (e.g. 'user.editor', 'server.ip') for superseding.",
                },
                "query": {
                    "type": "string",
                    "description": "Search query string (used with action='search').",
                },
                "memory_id": {
                    "type": "integer",
                    "description": "Memory row ID (used with action='get' or action='delete').",
                },
            },
            "required": ["action"],
        },
    },
}


class ManageMemoryTool(BaseTool):
    """Tier 1 tool for autonomous persistent memory storage, querying, and lifecycle management."""

    def __init__(self, store: Optional[MemoryStore] = None):
        super().__init__(
            tool_id="manage_memory",
            name="manage_memory",
            description=(
                "Manage long-term persistent memory: store preferences and environment facts, "
                "search memories using FTS5, retrieve by key/id, and soft-delete superseded entries."
            ),
            tier=1,
            is_core=True,
            requires_bridge=False,
            requires_window=False,
            token_cost=50,
        )
        self.store = store or MemoryStore()
        self.parameters = [
            ToolParameter(
                name="action",
                type="string",
                description="Memory action: 'store', 'search', 'get', or 'delete'.",
                required=True,
            ),
            ToolParameter(
                name="content",
                type="string",
                description="Text content of the memory to store.",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="category",
                type="string",
                description="Category: 'preference', 'environment', 'homelab', 'workflow', 'capability'.",
                required=False,
                default="preference",
            ),
            ToolParameter(
                name="key",
                type="string",
                description="Identifier key for contradiction resolution and key-based superseding.",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="query",
                type="string",
                description="Search query for action='search'.",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="memory_id",
                type="integer",
                description="Memory row ID for action='get' or action='delete'.",
                required=False,
                default=None,
            ),
        ]

    async def execute(self, params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> ToolResult:
        """Execute memory lifecycle operations."""
        if params is None:
            params = {}
        if kwargs:
            params = {**params, **kwargs}

        action = str(params.get("action", "store")).strip().lower()
        content = (params.get("content") or params.get("fact") or "").strip()
        category = str(params.get("category", "preference")).strip().lower()
        key = params.get("key")
        if isinstance(key, str):
            key = key.strip() or None
        query = (params.get("query") or "").strip()
        memory_id = params.get("memory_id")
        if memory_id is not None:
            try:
                memory_id = int(memory_id)
            except (ValueError, TypeError):
                memory_id = None

        if action == "store":
            if not content:
                return ToolResult(
                    False,
                    error="Parameter 'content' cannot be empty for action='store'.",
                    remedy_hint="Provide non-empty 'content' string describing the fact to remember.",
                    allowed_actions=["manage_memory"],
                )
            if category not in VALID_CATEGORIES:
                category = "preference"

            had_prior = False
            prior_id = None
            if key:
                prior_mem = self.store.get_by_key(key)
                if prior_mem:
                    had_prior = True
                    prior_id = prior_mem["id"]

            try:
                mem_id = self.store.store(content=content, category=category, key=key)
                return ToolResult(
                    True,
                    output={
                        "status": "success",
                        "action": "store",
                        "memory_id": mem_id,
                        "key": key,
                        "category": category,
                        "content": content,
                        "superseded": had_prior,
                        "superseded_id": prior_id,
                        "message": (
                            f"Stored memory #{mem_id} (key='{key}', category='{category}'). "
                            + (f"Superseded prior memory #{prior_id}." if had_prior else "")
                        ).strip(),
                    },
                )
            except Exception as exc:
                return ToolResult(
                    False,
                    error=f"Failed to store memory: {exc}",
                    remedy_hint="Verify database connectivity and parameter types.",
                    allowed_actions=["manage_memory"],
                )

        elif action == "search":
            search_query = query or content
            if not search_query:
                return ToolResult(
                    False,
                    error="Parameter 'query' cannot be empty for action='search'.",
                    remedy_hint="Provide a search term in 'query'.",
                    allowed_actions=["manage_memory"],
                )

            try:
                cat_filter = category if category in VALID_CATEGORIES else None
                results = self.store.search(search_query, category=cat_filter, limit=int(params.get("limit", 10)))
                return ToolResult(
                    True,
                    output={
                        "status": "success",
                        "action": "search",
                        "query": search_query,
                        "count": len(results),
                        "memories": results,
                    },
                )
            except Exception as exc:
                return ToolResult(
                    False,
                    error=f"Search failed: {exc}",
                    remedy_hint="Check search query syntax.",
                    allowed_actions=["manage_memory"],
                )

        elif action == "get":
            found_mem = None
            if memory_id is not None:
                found_mem = self.store.get(memory_id)
            elif key:
                found_mem = self.store.get_by_key(key)
            else:
                return ToolResult(
                    False,
                    error="Either 'memory_id' or 'key' is required for action='get'.",
                    remedy_hint="Provide 'memory_id' or 'key' to retrieve.",
                    allowed_actions=["manage_memory"],
                )

            if found_mem:
                return ToolResult(
                    True,
                    output={
                        "status": "success",
                        "action": "get",
                        "memory": found_mem,
                    },
                )

            target_repr = f"id={memory_id}" if memory_id is not None else f"key='{key}'"
            return ToolResult(
                False,
                error=f"Memory not found for {target_repr}.",
                remedy_hint="Call manage_memory with action='search' to find existing memories.",
                allowed_actions=["manage_memory"],
            )

        elif action == "delete":
            deleted = False
            target_id = memory_id

            if target_id is None and key:
                prior = self.store.get_by_key(key)
                if prior:
                    target_id = prior["id"]

            if target_id is not None:
                deleted = self.store.delete(target_id, hard_delete=bool(params.get("hard", False)))

            if deleted:
                return ToolResult(
                    True,
                    output={
                        "status": "success",
                        "action": "delete",
                        "memory_id": target_id,
                        "key": key,
                        "message": f"Successfully soft-deleted memory #{target_id}.",
                    },
                )

            target_repr = f"id={memory_id}" if memory_id is not None else f"key='{key}'"
            return ToolResult(
                False,
                error=f"Memory not found to delete for {target_repr}.",
                remedy_hint="Call manage_memory with action='search' to find active memory IDs.",
                allowed_actions=["manage_memory"],
            )

        return ToolResult(
            False,
            error=f"Unknown memory action '{action}'. Supported actions: store, search, get, delete.",
            remedy_hint="Use one of action='store', action='search', action='get', action='delete'.",
            allowed_actions=["manage_memory"],
        )


async def execute(action: str = "store", **kwargs: Any) -> str:
    """Standard dynamic tool plugin entry point. Returns serialized JSON string."""
    tool = ManageMemoryTool()
    res = await tool.execute({"action": action, **kwargs})
    if res.success:
        return json.dumps(res.output if isinstance(res.output, (dict, list)) else {"output": res.output})
    return json.dumps({
        "status": "error",
        "error": res.error,
        "remedy_hint": res.remedy_hint,
        "allowed_actions": res.allowed_actions,
        **(res.output if isinstance(res.output, dict) else {}),
    })


__all__ = [
    "ManageMemoryTool",
    "execute",
    "TOOL_SCHEMA",
    "REQUIRED_RING",
    "TIER",
    "TUI_HINT",
]
