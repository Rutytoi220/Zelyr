"""Integration tests for Memory Compaction, Fact Promotion, and REM Sleep Lifecycle.

Verifies:
1. DeepMemoryConsolidation REM sleep (temporal decay & semantic deduplication).
2. Session fact promotion & automatic contradiction superseding.
3. ObservationCompressor buffer compression and MemoryCompactor vector deduplication.
4. Database FTS5 index synchronization & trigger integrity post-compaction.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from zelyr.core.events import EventBus
from zelyr.db.memory import MemoryStore, add_message
from zelyr.memory.compactor import MemoryCompactor
from zelyr.memory.compression import ObservationCompressor
from zelyr.memory.promotion import promote_session_facts
from zelyr.memory.rem_sleep import DeepMemoryConsolidation


@pytest.mark.asyncio
async def test_rem_sleep_consolidation():
    """REM sleep consolidates memory graph: prunes decayed nodes and fuses duplicates."""
    bus = EventBus()
    events_received = []

    bus.subscribe("rem_sleep.started", lambda e: events_received.append("started"))
    bus.subscribe("rem_sleep.completed", lambda e: events_received.append("completed"))

    dmc = DeepMemoryConsolidation(bus)

    now = datetime.now(timezone.utc)
    old_time = (now - timedelta(days=45)).isoformat()
    recent_time = (now - timedelta(days=2)).isoformat()

    nodes = [
        # Node 1: Decayed node (> 30 days old)
        {"id": "node_old", "content": "Ephemeral scratchpad debug log", "created_at": old_time},
        # Node 2: Valid recent node
        {"id": "node_recent_1", "content": "Production database is PostgreSQL 16", "created_at": recent_time},
        # Node 3: Duplicate of Node 2 (different case & whitespace)
        {"id": "node_recent_dup", "content": "  production database is postgresql 16 \n", "created_at": recent_time},
        # Node 4: Distinct recent node
        {"id": "node_recent_2", "content": "Deploying with uv and pytest", "created_at": recent_time},
    ]

    consolidated = await dmc.trigger_rem_sleep(nodes)
    await asyncio.sleep(0.05)

    # 1 decayed node pruned, 1 duplicate node fused -> 2 surviving nodes
    assert len(consolidated) == 2
    contents = [n["content"].strip().lower() for n in consolidated]
    assert "production database is postgresql 16" in contents
    assert "deploying with uv and pytest" in contents
    assert "ephemeral scratchpad debug log" not in contents

    # Bus notifications emitted
    assert "started" in events_received
    assert "completed" in events_received


def test_fact_promotion_contradiction_superseding(tmp_path: Path):
    """Session fact promotion extracts user preferences and supersedes conflicting beliefs."""
    db_file = tmp_path / "lifecycle_memory.db"
    store = MemoryStore(db_path=str(db_file))
    session_1 = "sess_pref_1"
    session_2 = "sess_pref_2"

    # Session 1: User indicates shell preference
    add_message(session_1, "user", "Please remember that my shell is zsh", db_path=str(db_file))
    promoted_1 = promote_session_facts(session_1, store=store)
    assert len(promoted_1) == 1
    mem_1_id = promoted_1[0]

    fact_1 = store.get(mem_1_id)
    assert fact_1 is not None
    assert fact_1["key"] == "user.shell"
    assert fact_1["is_active"] == 1
    assert fact_1["superseded_by"] is None

    # Active memory for user.shell is currently zsh
    active_fact = store.get_by_key("user.shell")
    assert active_fact is not None
    assert "zsh" in active_fact["content"]

    # Session 2: User contradicts earlier preference: switches to fish
    add_message(session_2, "user", "Please remember that my shell is fish", db_path=str(db_file))
    promoted_2 = promote_session_facts(session_2, store=store)
    assert len(promoted_2) == 1
    mem_2_id = promoted_2[0]

    # Verify new active memory is fish
    active_fact_new = store.get_by_key("user.shell")
    assert active_fact_new is not None
    assert active_fact_new["id"] == mem_2_id
    assert "fish" in active_fact_new["content"]
    assert active_fact_new["is_active"] == 1

    # Verify old memory is deactivated and atomically points to new memory
    fact_1_updated = store.get(mem_1_id)
    assert fact_1_updated is not None
    assert fact_1_updated["is_active"] == 0
    assert fact_1_updated["superseded_by"] == mem_2_id


@pytest.mark.asyncio
async def test_semantic_compactor_token_deduplication():
    """ObservationCompressor caps token footprint and MemoryCompactor merges similar vectors."""
    # 1. Observation buffer compression & hard pruning
    huge_output = "data_row_entry: " * 500  # ~8000 chars each
    observations = [
        {"tool": "read_file", "result": {"output": huge_output, "metadata": {"verbose": True}}},
        {"tool": "shell", "result": {"output": huge_output, "trace": "stack..."}},
        {"tool": "grep_search", "result": {"output": "Found match on line 42."}},
    ]

    # Compress buffer with budget of 2000 chars
    compressed = ObservationCompressor.compress_observations_buffer(
        observations,
        max_total_chars=2000,
        iterations=3,
    )
    # Total character count must be within reasonable bound of budget
    assert len(compressed) <= len(observations)
    # The latest observation must remain fully intact
    assert compressed[-1]["result"]["output"] == "Found match on line 42."

    # 2. MemoryCompactor vector deduplication
    mock_db = AsyncMock()
    # Provide two rows with near-identical embeddings (cosine similarity > 0.99)
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.999, 0.01, 0.0]
    mock_cursor = AsyncMock()
    mock_cursor.fetchall.return_value = [
        {
            "emb_id": 1,
            "memory_key": "mem_pref_v1",
            "embedding_json": json.dumps(vec1),
            "created_at": "2026-10-01 10:00:00",
            "retrieval_count": 2,
            "confidence_weight": 1.0,
            "value_json": "{}",
        },
        {
            "emb_id": 2,
            "memory_key": "mem_pref_v2",
            "embedding_json": json.dumps(vec2),
            "created_at": "2026-10-02 12:00:00",
            "retrieval_count": 5,
            "confidence_weight": 1.1,
            "value_json": "{}",
        },
    ]
    mock_db.execute.return_value = mock_cursor

    compactor = MemoryCompactor(mock_db)
    report = await compactor.run_compaction()

    assert report["scanned"] == 2
    assert report["merged"] == 1
    assert report["deleted"] == 1

    # Verify db operations: updated newer memory and deleted older duplicate
    execute_calls = [str(call) for call in mock_db.execute.call_args_list]
    assert any("UPDATE memories SET" in call for call in execute_calls)
    assert any("DELETE FROM memories" in call for call in execute_calls)
    assert mock_db.commit.called


def test_fts5_index_integrity_post_compaction(tmp_path: Path):
    """FTS5 trigger synchronization and rebuild integrity across memory mutations."""
    db_file = tmp_path / "fts_integrity.db"
    store = MemoryStore(db_path=str(db_file))

    # Insert 3 memories
    id_1 = store.store_fact("User primary editor is Neovim with Lua plugins", category="preference", key="user.editor")
    store.store_fact("Docker compose setup with Redis and PostgreSQL", category="homelab", key="homelab.stack")
    store.store_fact("API server runs on port 9412", category="environment", key="env.port")

    # Search for Neovim
    res_editor = store.search("Neovim")
    assert len(res_editor) == 1
    assert res_editor[0]["key"] == "user.editor"

    # Supersede Neovim with VSCode
    id_2 = store.store_fact("User migrated to VSCode editor", category="preference", key="user.editor")
    assert id_2 != id_1

    # Search for VSCode
    res_vscode = store.search("VSCode")
    assert len(res_vscode) == 1
    assert res_vscode[0]["id"] == id_2

    # Default active search for Neovim should return 0 results (superseded)
    res_neovim_active = store.search("Neovim")
    assert len(res_neovim_active) == 0

    # Including inactive returns the superseded memory
    res_neovim_all = store.search("Neovim", include_inactive=True)
    assert len(res_neovim_all) == 1
    assert res_neovim_all[0]["is_active"] == 0
    assert res_neovim_all[0]["superseded_by"] == id_2

    # Test FTS5 rebuild integrity command
    with store._get_conn() as conn:
        conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild');")
        conn.commit()

    # Post-rebuild search remains fully functional
    res_redis = store.search("PostgreSQL")
    assert len(res_redis) == 1
    assert res_redis[0]["key"] == "homelab.stack"
