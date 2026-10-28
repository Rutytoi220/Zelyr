"""Comprehensive test suite for REM Sleep Memory Compaction, Temporal Decay, and FTS5 WAL Concurrency.

Covers:
1. Concurrency Stress Test: Multiple concurrent async tasks reading while compaction merges/purges.
   Asserts zero sqlite3.OperationalError: database is locked errors.
2. Temporal Decay Verification: Ephemeral memories older than 30 days are purged while protected memories
   (preference, environment, capability, core_belief) remain intact.
3. Semantic Deduplication: Duplicate/conflicting memories fuse, inheriting retrieval counts and boosting confidence.
4. FTS5 Index Consistency: Purged memories are absent from full-text search, and merged memories return updated content.
5. Structured Execution Metrics: Verification of structured compaction metrics (total_scanned, merged_count,
   decayed_count, purged_count, duration_seconds).
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from zelyr.core.events import EventBus
from zelyr.db.memory import MemoryStore, get_connection
from zelyr.memory.rem_sleep import DeepMemoryConsolidation


# ---------------------------------------------------------------------------
# Test 1: Concurrency Stress Test under Simultaneous Reads and Compaction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_wal_concurrency_stress_under_simultaneous_reads_and_compaction(tmp_path: Path):
    """Verify high concurrent read/write loads produce zero 'database is locked' errors under WAL mode."""
    db_file = tmp_path / "wal_stress_test.db"
    store = MemoryStore(db_path=str(db_file))

    # Seed baseline records
    for i in range(60):
        store.store(
            content=f"Telemetry record {i} for microservice worker node {i % 5}",
            category="workflow",
            key=f"worker.telemetry.{i}",
        )

    errors = []
    stop_event = asyncio.Event()

    async def continuous_reader(reader_id: int):
        queries = ["microservice", "worker", "telemetry", "record"]
        q_idx = 0
        while not stop_event.is_set():
            try:
                query = queries[q_idx % len(queries)]
                results = store.search(query, limit=5)
                assert isinstance(results, list)
                relevant = store.get_relevant_memories(query, limit=3, update_access=False)
                assert isinstance(relevant, list)
                q_idx += 1
                await asyncio.sleep(0.005)
            except Exception as exc:
                errors.append((reader_id, "reader", exc))

    async def continuous_writer():
        for i in range(25):
            try:
                store.store(
                    content=f"Dynamic runtime heartbeat event {i}",
                    category="environment",
                    key=f"heartbeat.{i % 3}",
                )
                await asyncio.sleep(0.01)
            except Exception as exc:
                errors.append((0, "writer", exc))

    # Launch 10 concurrent reader tasks
    readers = [asyncio.create_task(continuous_reader(r)) for r in range(10)]
    writer_task = asyncio.create_task(continuous_writer())

    # Concurrently execute async and sync compaction sweeps
    await asyncio.sleep(0.03)
    compact_res1 = await store.acompact(older_than_days=30.0)
    assert compact_res1["duration_seconds"] >= 0.0

    compact_res2 = store.compact(older_than_days=30.0)
    assert compact_res2["duration_seconds"] >= 0.0

    await writer_task
    stop_event.set()
    await asyncio.gather(*readers)

    # Assert zero database locked errors
    lock_errors = [e for e in errors if "database is locked" in str(e[2]).lower()]
    assert len(lock_errors) == 0, f"Encountered database locked errors: {lock_errors}"
    assert len(errors) == 0, f"Encountered unexpected concurrency errors: {errors}"


# ---------------------------------------------------------------------------
# Test 2: Temporal Decay Verification & Core Memory Preservation
# ---------------------------------------------------------------------------

def test_temporal_decay_purging_and_core_preservation(tmp_path: Path):
    """Ephemeral logs older than 30 days decay and purge; core categories and core_belief tags survive."""
    db_file = tmp_path / "decay_test.db"
    store = MemoryStore(db_path=str(db_file))

    now = time.time()
    day_seconds = 86400.0
    time_45_days_ago = now - (45.0 * day_seconds)
    time_2_days_ago = now - (2.0 * day_seconds)

    with store._write_lock:
        with store._get_conn() as conn:
            # 1. Ephemeral memory > 30 days old (should decay and purge)
            conn.execute(
                """
                INSERT INTO memories (category, key, content, confidence, is_active, created_at, updated_at)
                VALUES ('workflow', 'build.log.old', 'Ephemeral CI test execution log from previous run', 0.5, 1, ?, ?)
                """,
                (time_45_days_ago, time_45_days_ago),
            )
            # 2. Protected category memory > 30 days old (should survive)
            conn.execute(
                """
                INSERT INTO memories (category, key, content, confidence, is_active, created_at, updated_at)
                VALUES ('preference', 'user.editor', 'User preferred editor is Neovim', 1.0, 1, ?, ?)
                """,
                (time_45_days_ago, time_45_days_ago),
            )
            # 3. Core belief tagged memory > 30 days old in workflow category (should survive)
            conn.execute(
                """
                INSERT INTO memories (category, key, content, confidence, is_active, created_at, updated_at)
                VALUES ('workflow', 'core_belief.audit', 'Always execute behavioral verification before git push', 1.0, 1, ?, ?)
                """,
                (time_45_days_ago, time_45_days_ago),
            )
            # 4. Recent memory in workflow category (should survive)
            conn.execute(
                """
                INSERT INTO memories (category, key, content, confidence, is_active, created_at, updated_at)
                VALUES ('workflow', 'build.log.recent', 'Recent green build scorecard snapshot', 0.8, 1, ?, ?)
                """,
                (time_2_days_ago, time_2_days_ago),
            )
            conn.commit()

    # Run compaction protecting preference, environment, capability
    metrics = store.compact(
        older_than_days=30.0,
        protected_categories={"preference", "environment", "capability"},
    )

    assert metrics["decayed_count"] == 1
    assert metrics["purged_count"] >= 1

    # Verify ephemeral memory is gone
    ephemeral = store.get_by_key("build.log.old")
    assert ephemeral is None

    # Verify protected category memory remains active
    pref = store.get_by_key("user.editor")
    assert pref is not None
    assert "Neovim" in pref["content"]

    # Verify core_belief tagged memory remains active
    core = store.get_by_key("core_belief.audit")
    assert core is not None
    assert "Always execute behavioral verification" in core["content"]

    # Verify recent memory remains active
    recent = store.get_by_key("build.log.recent")
    assert recent is not None


# ---------------------------------------------------------------------------
# Test 3: Semantic Deduplication & Confidence Fusion
# ---------------------------------------------------------------------------

def test_semantic_deduplication_and_confidence_fusion(tmp_path: Path):
    """Duplicate memory entries are fused, inheriting retrieval count and boosting confidence."""
    db_file = tmp_path / "dedup_test.db"
    store = MemoryStore(db_path=str(db_file))

    # Seed duplicate facts with differing whitespace/case and access counts
    id_1 = store.store(
        content="Primary production database is PostgreSQL 16 on port 5432",
        category="homelab",
        key="db.postgres.primary",
        confidence=0.80,
    )
    # Manually assign access count
    with store._write_lock:
        with store._get_conn() as conn:
            conn.execute("UPDATE memories SET access_count = 3 WHERE id = ?", (id_1,))
            conn.commit()

    id_2 = store.store(
        content="  primary production database is postgresql 16 on port 5432 \n",
        category="homelab",
        key="db.postgres.dup",
        confidence=0.75,
    )
    with store._write_lock:
        with store._get_conn() as conn:
            conn.execute("UPDATE memories SET access_count = 8 WHERE id = ?", (id_2,))
            conn.commit()

    metrics = store.compact(older_than_days=30.0, purge_decayed=False)
    assert metrics["merged_count"] == 1

    # Check survivor record
    survivor = store.get(id_1)
    assert survivor is not None
    assert survivor["is_active"] == 1
    # Inherited max access_count + 1 = 8 + 1 = 9
    assert survivor["access_count"] >= 9
    # Boosted confidence: max(0.80, 0.75) + 0.1 = 0.90
    assert survivor["confidence"] >= 0.89

    # Check superseded record
    dup = store.get(id_2)
    assert dup is not None
    assert dup["is_active"] == 0
    assert dup["superseded_by"] == id_1


# ---------------------------------------------------------------------------
# Test 4: FTS5 Index Consistency Post-Compaction
# ---------------------------------------------------------------------------

def test_fts5_index_consistency_post_compaction(tmp_path: Path):
    """FTS5 index automatically purges soft-deleted records and updates merged entries."""
    db_file = tmp_path / "fts5_consistency_test.db"
    store = MemoryStore(db_path=str(db_file))

    now = time.time()
    old_time = now - (60.0 * 86400.0)

    # Insert an ephemeral memory directly
    with store._write_lock:
        with store._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO memories (category, key, content, confidence, is_active, created_at, updated_at)
                VALUES ('workflow', 'temp.scratchpad', 'Volatile ephemeral diagnostic trace info', 0.5, 1, ?, ?)
                """,
                (old_time, old_time),
            )
            conn.commit()

    store.store("Persistent infrastructure cluster definition", category="environment", key="cluster.env")
    store.store("  persistent infrastructure cluster definition  ", category="environment", key="cluster.env.dup")

    # Before compaction: direct query on memories_fts finds the old volatile record
    with get_connection(str(db_file)) as conn:
        before_fts = conn.execute("SELECT * FROM memories_fts WHERE memories_fts MATCH 'Volatile'").fetchall()
        assert len(before_fts) == 1

        before_env = conn.execute("SELECT * FROM memories_fts WHERE memories_fts MATCH 'cluster'").fetchall()
        assert len(before_env) == 2

    # Run compaction
    metrics = store.compact(
        older_than_days=30.0,
        protected_categories={"preference", "environment", "capability"},
        purge_decayed=True,
    )
    assert metrics["decayed_count"] == 1
    assert metrics["merged_count"] == 1

    # After compaction:
    # 1. Purged memory is completely gone from memories_fts
    with get_connection(str(db_file)) as conn:
        after_fts = conn.execute("SELECT * FROM memories_fts WHERE memories_fts MATCH 'Volatile'").fetchall()
        assert len(after_fts) == 0

        # 2. Merged duplicate is purged from memories_fts; exactly 1 active cluster entry exists
        after_env = conn.execute("SELECT * FROM memories_fts WHERE memories_fts MATCH 'cluster'").fetchall()
        assert len(after_env) == 1

    # 3. MemoryStore.search never returns purged or superseded records
    search_decayed = store.search("Volatile")
    assert len(search_decayed) == 0

    search_cluster = store.search("cluster")
    assert len(search_cluster) == 1
    assert search_cluster[0]["key"] == "cluster.env"


# ---------------------------------------------------------------------------
# Test 5: Structured Execution Metrics
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_structured_execution_telemetry_metrics(tmp_path: Path):
    """Compaction returns valid structured telemetry metrics across sync and async APIs."""
    db_file = tmp_path / "telemetry_test.db"
    store = MemoryStore(db_path=str(db_file))

    store.store("Test entry 1", category="preference", key="pref.1")
    store.store("Test entry 2", category="environment", key="env.2")

    # Sync compaction metrics
    res_sync = store.compact(older_than_days=30.0)
    for field in ("total_scanned", "merged_count", "decayed_count", "purged_count", "duration_seconds"):
        assert field in res_sync
    assert isinstance(res_sync["total_scanned"], int)
    assert isinstance(res_sync["duration_seconds"], float)
    assert res_sync["total_scanned"] == 2

    # Async compaction metrics
    res_async = await store.acompact(older_than_days=30.0)
    for field in ("total_scanned", "merged_count", "decayed_count", "purged_count", "duration_seconds"):
        assert field in res_async
    assert isinstance(res_async["total_scanned"], int)
    assert isinstance(res_async["duration_seconds"], float)
    assert res_async["total_scanned"] == 2


# ---------------------------------------------------------------------------
# Test 6: DeepMemoryConsolidation REM Sleep Graph Preservation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_deep_memory_consolidation_rem_sleep_graph():
    """Verify DeepMemoryConsolidation REM sleep preserves core categories and yields execution."""
    bus = EventBus()
    completed_events = []
    bus.subscribe("rem_sleep.completed", lambda e: completed_events.append(e.data))

    dmc = DeepMemoryConsolidation(bus)

    now = datetime.now(timezone.utc)
    old_time = (now - timedelta(days=45)).isoformat()
    recent_time = (now - timedelta(days=2)).isoformat()

    nodes = [
        # Decayed ephemeral log (> 30 days old) -> Should be pruned
        {"id": "n1", "content": "Ephemeral worker debug log", "category": "log", "created_at": old_time},
        # Old preference (> 30 days old) -> Protected category, should survive!
        {"id": "n2", "content": "Shell is zsh", "category": "preference", "created_at": old_time},
        # Old core belief (> 30 days old) -> Tagged core_belief, should survive!
        {"id": "n3", "content": "Safety rule: non-destructive testing only", "category": "workflow", "tags": ["core_belief"], "created_at": old_time},
        # Recent node with duplicate -> Should be fused
        {"id": "n4", "content": "Zelyr version 1.1.0.dev0", "category": "capability", "created_at": recent_time, "access_count": 2},
        {"id": "n5", "content": "  zelyr version 1.1.0.dev0  ", "category": "capability", "created_at": recent_time, "access_count": 4},
    ]

    consolidated = await dmc.trigger_rem_sleep(nodes)

    # n1 pruned, n5 fused into n4 -> 3 surviving nodes (n2, n3, n4)
    assert len(consolidated) == 3
    surviving_contents = [n["content"].strip().lower() for n in consolidated]
    assert "ephemeral worker debug log" not in surviving_contents
    assert "shell is zsh" in surviving_contents
    assert "safety rule: non-destructive testing only" in surviving_contents
    assert "zelyr version 1.1.0.dev0" in surviving_contents

    # Verify fused node inherited access count: max(2, 4) + 1 = 5
    fused_node = next(n for n in consolidated if "version" in n["content"])
    assert fused_node.get("access_count") == 5

    # Verify event emission with structured metrics
    assert len(completed_events) == 1
    assert completed_events[0]["surviving"] == 3
    assert completed_events[0]["pruned"] == 1
    assert completed_events[0]["fused"] == 1
