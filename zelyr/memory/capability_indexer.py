"""Zelyr Tool Capability Indexer & Reconciliation Engine.

Auto-indexes all registered tools into persistent MemoryStore under category 'capability',
keyed by 'tool.<tool_name>'. Automatically supersedes older entries when tool schemas
or descriptions change, and reconciles capability memory when tools are removed or uninstalled.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from zelyr.db.memory import MemoryStore, get_db_path

logger = logging.getLogger(__name__)


def _sync_disk_unlinked_tools(tools_dir: Optional[Path] = None) -> None:
    """Detect dynamic tools whose files have been removed from disk and unregister them."""
    try:
        from zelyr.core.plugins import get_plugin_file_paths, unregister_plugin
        for name, p in get_plugin_file_paths().items():
            if not p.exists():
                unregister_plugin(name)
    except Exception as e:
        logger.debug("Failed checking tracked plugin file paths: %s", e)

    # Check sys.modules for any loaded dynamic tools
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("zelyr.dynamic_tools."):
            tool_name = mod_name[len("zelyr.dynamic_tools."):]
            mod = sys.modules.get(mod_name)
            file_path = getattr(mod, "__file__", None)
            if file_path and not Path(file_path).exists():
                try:
                    from zelyr.core.plugins import unregister_plugin
                    unregister_plugin(tool_name)
                except Exception:
                    pass


def reconcile_tool_capabilities(
    store: Optional[MemoryStore] = None,
    schemas: Optional[List[Dict[str, Any]]] = None,
    tools_dir: Optional[Path] = None,
) -> Dict[str, int]:
    """Reconcile capability records in MemoryStore with registered active tools.

    - Unregisters any tools whose files were unlinked from disk.
    - Gathers currently valid registered tool names from get_tool_schemas(0).
    - Queries MemoryStore for active records where category = 'capability' AND is_active = 1 AND key LIKE 'tool.%'.
    - Deactivates any capability records whose tool name is not in the active tool names set.
    - Strictly preserves records in all other categories (preference, environment, homelab, workflow).

    Returns a dict with counts, e.g. {"deactivated": count}.
    """
    mem_store = store or MemoryStore()

    _sync_disk_unlinked_tools(tools_dir=tools_dir)

    if schemas is None:
        try:
            from zelyr.core.plugins import get_tool_schemas
            schemas = get_tool_schemas(0)
        except Exception as e:
            logger.warning("Could not load tool schemas for capability reconciliation: %s", e)
            schemas = []

    active_tool_names = set()
    for s in schemas:
        fn = s.get("function", {})
        name = fn.get("name") or s.get("name")
        if name:
            active_tool_names.add(name)

    deactivated_count = 0
    now = time.time()

    with mem_store._write_lock:
        with mem_store._get_conn() as conn:
            cursor = conn.execute(
                """
                SELECT id, key, content FROM memories
                WHERE category = 'capability' AND is_active = 1 AND key LIKE 'tool.%'
                """
            )
            rows = cursor.fetchall()
            for row in rows:
                key = row["key"] or ""
                if key.startswith("tool."):
                    tool_name = key[len("tool."):]
                    if tool_name not in active_tool_names:
                        conn.execute(
                            "UPDATE memories SET is_active = 0, updated_at = ? WHERE id = ?",
                            (now, row["id"]),
                        )
                        deactivated_count += 1
            if deactivated_count > 0:
                conn.commit()

    return {"deactivated": deactivated_count}


def index_tool_capabilities(
    store: Optional[MemoryStore] = None,
    schemas: Optional[List[Dict[str, Any]]] = None,
    tools_dir: Optional[Path] = None,
) -> int:
    """Index registered tool capabilities and reconcile deleted capabilities into MemoryStore.

    Returns the number of newly indexed or updated tools.
    """
    mem_store = store or MemoryStore()

    # Reconcile memory with active tool schemas
    reconcile_tool_capabilities(store=mem_store, schemas=schemas, tools_dir=tools_dir)

    if schemas is None:
        try:
            from zelyr.core.plugins import get_tool_schemas
            schemas = get_tool_schemas(0)
        except Exception as e:
            logger.warning("Could not load tool schemas for capability indexing: %s", e)
            schemas = []

    indexed_count = 0
    for s in schemas:
        try:
            fn = s.get("function", {})
            name = fn.get("name") or s.get("name")
            if not name:
                continue
            desc = fn.get("description", "").strip()
            key = f"tool.{name}"
            content = f"{name}: {desc}" if desc else f"{name}"

            # Check if active capability already exists with identical content
            active_record = mem_store.get_by_key(key)
            if active_record and active_record.get("content") == content:
                continue

            # Store new capability (superseding previous active entry if any)
            mem_store.store(
                content=content,
                category="capability",
                key=key,
                confidence=1.0,
            )
            indexed_count += 1
        except Exception as err:
            logger.warning("Failed to index capability for schema %s: %s", s, err)

    return indexed_count
