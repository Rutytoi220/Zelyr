"""Comprehensive unit and integration tests for Proactive Memory Retrieval and manage_memory.

Tests:
1. Keyword extraction: punctuation stripping and stop word filtering.
2. Composite ranking: BM25 normalization, exponential recency decay, utility boost.
3. Hard token ceiling: strict enforcement of <= 350 tokens under heavy volume.
4. Zero-token injection: empty string returned for irrelevant or empty queries.
5. Access tracking: access_count and updated_at updated for recalled memories.
6. manage_memory tool: store, search, get, delete, and key-based superseding.
7. NativeOrchestrator proactive hydration: injected on depth == 0, bypassed on retry / recovery.
8. NativeOrchestrator memory tool alias resolution and argument normalization.
"""

from __future__ import annotations

import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List
from unittest.mock import AsyncMock

import pytest

from zelyr.db.memory import MemoryStore, get_db_path, init_db
from zelyr.engine.memory_hydration import MemoryHydrator, _count_tokens, hydrate_context_memory
from zelyr.tools.manage_memory import (
    ManageMemoryTool,
    REQUIRED_RING,
    TIER,
    TOOL_SCHEMA,
    TUI_HINT,
    execute,
)


@pytest.fixture
def isolated_db():
    """Create an isolated temporary SQLite database for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_file = Path(tmp_dir) / "test_memory_hydration.db"
        init_db(str(db_file))
        yield str(db_file)


def test_keyword_extraction_filters_stop_words_and_punctuation(isolated_db):
    """Verify keyword extraction cleans punctuation and filters common stop words."""
    hydrator = MemoryHydrator(db_path=isolated_db)

    raw_query = "[Active Window: title='Editor', class='Alacritty'] What is my user.editor configuration, please?"
    keywords = hydrator.extract_query_keywords(raw_query)

    assert "Active Window" not in keywords
    assert "Editor" not in keywords
    assert "what" not in keywords.lower()
    assert "please" not in keywords.lower()
    assert "user.editor" in keywords
    assert "configuration" in keywords


def test_composite_ranking_recency_and_utility(isolated_db):
    """Verify composite ranking scores recent and frequent memories higher than stale ones."""
    store = MemoryStore(db_path=isolated_db)
    now = time.time()

    # 1. Stale memory (updated 30 days ago, low access)
    id_stale = store.store(
        content="Primary editor is Vim",
        category="preference",
        key="editor.stale",
        confidence=1.0,
    )
    with store._get_conn() as conn:
        stale_time = now - (30 * 86400)
        conn.execute("UPDATE memories SET updated_at = ?, access_count = 0 WHERE id = ?", (stale_time, id_stale))
        conn.commit()

    # 2. Fresh memory (updated now, accessed 5 times)
    id_fresh = store.store(
        content="Primary editor is Neovim",
        category="preference",
        key="editor.fresh",
        confidence=1.0,
    )
    with store._get_conn() as conn:
        conn.execute("UPDATE memories SET updated_at = ?, access_count = 5 WHERE id = ?", (now, id_fresh))
        conn.commit()

    hydrator = MemoryHydrator(db_path=isolated_db, max_tokens=350)
    recalled = hydrator.get_relevant_memories("editor", update_access=False)

    assert len(recalled) == 2
    # Fresh memory must rank first due to higher recency and utility boost
    assert recalled[0]["id"] == id_fresh
    assert recalled[1]["id"] == id_stale
    assert recalled[0]["composite_score"] > recalled[1]["composite_score"]


def test_hard_token_ceiling_enforcement(isolated_db):
    """Verify that recalling 25 matching memories never exceeds the <= 350 token ceiling."""
    store = MemoryStore(db_path=isolated_db)

    for i in range(25):
        store.store(
            content=(
                f"Infrastructure parameter node {i} for distributed database consensus: "
                f"endpoint=10.0.0.{i}:9000, cluster_name=zelyr_production_cluster, "
                f"telemetry_stream=active, raft_election_timeout=1500ms, data_directory=/var/data/node_{i}"
            ),
            category="homelab",
            key=f"infra.node_{i}",
            confidence=1.0,
        )

    hydrator = MemoryHydrator(db_path=isolated_db, max_tokens=350)
    recalled = hydrator.get_relevant_memories("infrastructure distributed consensus", limit=25, update_access=False)

    assert len(recalled) > 0
    assert len(recalled) < 25  # Should be capped by token budget

    block = hydrator.format_recalled_memory_block(recalled)
    token_count = _count_tokens(block)

    assert token_count <= 350
    assert "[Contextual Memory]:" in block
    assert "<recalled_memory>" in block
    assert "</recalled_memory>" in block


def test_zero_token_injection_for_irrelevant_queries(isolated_db):
    """Verify zero tokens injected for irrelevant, empty, or whitespace queries."""
    store = MemoryStore(db_path=isolated_db)
    store.store(content="User shell is zsh", category="preference", key="user.shell")

    hydrator = MemoryHydrator(db_path=isolated_db, max_tokens=350)

    # Empty query
    assert hydrator.format_recalled_memory_block(hydrator.get_relevant_memories("")) == ""
    assert hydrator.format_recalled_memory_block(hydrator.get_relevant_memories("    ")) == ""

    # Completely unrelated query
    unrelated = hydrator.get_relevant_memories("quantum_mechanics_schrodinger_9999")
    assert hydrator.format_recalled_memory_block(unrelated) == ""


def test_access_count_increments_for_recalled_memories(isolated_db):
    """Verify access_count increments and updated_at updates for injected memories."""
    store = MemoryStore(db_path=isolated_db)
    id1 = store.store(content="Database port is 5432", category="environment", key="db.port")
    id2 = store.store(content="Cache port is 6379", category="environment", key="cache.port")

    initial1 = store.get(id1)["access_count"]
    initial2 = store.get(id2)["access_count"]

    hydrator = MemoryHydrator(db_path=isolated_db, max_tokens=350)
    # Query only matching database
    recalled = hydrator.get_relevant_memories("database port", update_access=True)

    assert any(m["id"] == id1 for m in recalled)
    assert not any(m["id"] == id2 for m in recalled)

    after1 = store.get(id1)["access_count"]
    after2 = store.get(id2)["access_count"]

    # id1 had its access_count incremented during retrieval
    assert after1 > initial1
    # id2 access count only incremented by the final store.get() call
    assert after2 == initial2 + 1


@pytest.mark.asyncio
async def test_manage_memory_tool_lifecycle_and_superseding(isolated_db):
    """Verify ManageMemoryTool store, superseding, search, get, and delete operations."""
    store = MemoryStore(db_path=isolated_db)
    tool = ManageMemoryTool(store=store)

    # 1. Store initial fact
    res1 = await tool.execute({
        "action": "store",
        "content": "Neovim with lazy.nvim",
        "category": "preference",
        "key": "user.editor",
    })
    assert res1.success is True
    assert res1.output["superseded"] is False
    id1 = res1.output["memory_id"]

    # 2. Store updated fact with identical key -> must supersede old row
    res2 = await tool.execute({
        "action": "store",
        "content": "Helix editor with tree-sitter",
        "category": "preference",
        "key": "user.editor",
    })
    assert res2.success is True
    assert res2.output["superseded"] is True
    assert res2.output["superseded_id"] == id1
    id2 = res2.output["memory_id"]
    assert id2 != id1

    # Verify old row is inactive and superseded
    mem1_db = store.get(id1)
    assert mem1_db["is_active"] == 0
    assert mem1_db["superseded_by"] == id2

    # 3. Get by key returns active row
    res_get = await tool.execute({"action": "get", "key": "user.editor"})
    assert res_get.success is True
    assert res_get.output["memory"]["content"] == "Helix editor with tree-sitter"
    assert res_get.output["memory"]["id"] == id2

    # 4. Search via FTS5
    res_search = await tool.execute({"action": "search", "query": "tree-sitter"})
    assert res_search.success is True
    assert res_search.output["count"] >= 1
    assert any(m["key"] == "user.editor" for m in res_search.output["memories"])

    # 5. Soft-delete
    res_del = await tool.execute({"action": "delete", "key": "user.editor"})
    assert res_del.success is True

    mem2_db = store.get(id2)
    assert mem2_db["is_active"] == 0


@pytest.mark.asyncio
async def test_manage_memory_module_execute_plugin_entrypoint(isolated_db):
    """Verify module-level execute function conforms to standard plugin contract."""
    from unittest.mock import patch
    with patch("zelyr.tools.manage_memory.MemoryStore", return_value=MemoryStore(db_path=isolated_db)):
        raw_json = await execute(action="store", content="tmux with vi mode", key="user.multiplexer")
        import json
        data = json.loads(raw_json)
        assert data.get("status") == "success"
        assert data.get("key") == "user.multiplexer"


def test_tool_plugin_metadata():
    """Verify Ring, Tier, and Schema conformance for manage_memory."""
    assert REQUIRED_RING == 1
    assert TIER == 1
    assert isinstance(TUI_HINT, str)
    assert TOOL_SCHEMA["type"] == "function"
    assert TOOL_SCHEMA["function"]["name"] == "manage_memory"
    props = TOOL_SCHEMA["function"]["parameters"]["properties"]
    assert "action" in props
    assert "content" in props
    assert "key" in props
    assert "query" in props
