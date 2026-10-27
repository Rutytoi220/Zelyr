# ADR-003: SQLite WAL + Local Vector Database for Semantic Memory

**Status:** Accepted  
**Date:** 2026-07-18  
**Authors:** Zelyr Core Team

---

## Context

Zelyr's AI agents require persistent semantic memory: the ability to store conversation history, episodic summaries, tool execution traces, and embedding vectors, then retrieve them via both key-value lookup and cosine similarity search. This is the foundation that allows Zelyr to "remember" across sessions and perform contextual retrieval-augmented generation (RAG).

We evaluated the following storage strategies:

1. **Cloud vector databases (Pinecone, Weaviate, Milvus).** Industry-standard for production RAG pipelines. However, they require network connectivity, API keys, and ongoing subscription costs. This fundamentally violates Zelyr's local-first, offline-capable philosophy. A user's private conversation history must never leave their machine.
2. **PostgreSQL + pgvector.** Powerful, but requires a running PostgreSQL server. Unacceptable overhead for a desktop application.
3. **Local SQLite + embedded vector store.** Zero-configuration, single-file persistence, crash-resilient with WAL journaling. The database is just a file in `~/.zelyr/` that the user owns completely.

For the vector similarity layer specifically, we needed a strategy that scales from "first install with zero dependencies" to "power user with optimized HNSW indexing":

- **NumPy brute-force cosine similarity** — zero dependencies beyond NumPy (already required for embeddings). Works for small to medium collections.
- **Qdrant (embedded mode)** — optional `pip install qdrant-client` unlocks HNSW-indexed approximate nearest neighbor search.
- **ChromaDB (embedded mode)** — alternative optional backend with a simpler API.

## Decision

We implemented a three-tier storage architecture:

### Tier 1: SQLite with WAL Journaling ([db/memory.py](../../zelyr/db/memory.py))

All structured data — conversations, messages, events, agent sessions, tool calls, and raw embedding blobs — is stored in a single SQLite database with explicit crash-resilience configuration:

```python
await conn.execute("PRAGMA journal_mode = WAL;")     # Write-Ahead Logging
await conn.execute("PRAGMA synchronous = NORMAL;")   # Safe with WAL, 10x faster than FULL
await conn.execute("PRAGMA cache_size = -10000;")     # 10MB page cache
await conn.execute("PRAGMA busy_timeout = 5000;")     # 5-second retry on lock contention
```

The schema ([schema.sql](../../zelyr/memory/schema.sql)) defines 7 tables with 16 indexes:

| Table | Purpose |
|-------|---------|
| `events` | Audit trail and EventBus persistence |
| `memories` | Key-value store with TTL, tags, and confidence weighting |
| `agent_sessions` | Task execution tracking |
| `tool_calls` | Per-tool telemetry with duration and error capture |
| `conversations` | Multi-turn dialogue containers |
| `messages` | Conversation history with metadata |
| `embeddings` | Raw embedding vectors (JSON-serialized) |

### Tier 2: Vectorized NumPy Cosine Similarity ([semantic.py](../../zelyr/memory/semantic.py))

The fallback search path loads all embeddings from the `embeddings` table into a NumPy matrix and performs fully vectorized cosine similarity in a single pass:

```python
s_matrix = np.array(stored_vectors, dtype=np.float32)
s_norms = np.linalg.norm(s_matrix, axis=1)
similarities = np.dot(s_matrix, q_vec) / (s_norms * q_norm)
```

This approach has zero external dependencies and achieves microsecond retrieval times for collections under ~50,000 vectors.

### Tier 3: Optional HNSW Vector Stores ([vector_store.py](../../zelyr/memory/vector_store.py))

A `BaseVectorStore` protocol defines the contract:

```python
class BaseVectorStore(Protocol):
    def upsert(self, owner_id, owner_type, embedding, payload) -> None: ...
    def search(self, query_embedding, owner_type, top_k) -> List[Dict]: ...
    def delete(self, owner_id) -> None: ...
    def count(self) -> int: ...
```

Two concrete implementations are provided:
- **`QdrantLocalStore`** — Persists to `~/.zelyr/vector_db/` using Qdrant's embedded Rust engine. Handles stale lock file recovery automatically.
- **`ChromaDBStore`** — Persists to `~/.local/share/zelyr/chromadb/` using ChromaDB's embedded engine.

The `SemanticIndex` class auto-detects available backends at initialization:

```python
try:
    from zelyr.memory.vector_store import QdrantLocalStore
    self._vector_store = QdrantLocalStore()
except ImportError:
    logger.warning("Falling back to NumPy+SQLite.")
```

### Memory Lifecycle: The Sleep Cycle

The `SleepCycleDaemon` ([sleep_cycle.py](../../zelyr/memory/sleep_cycle.py)) monitors EventBus activity and triggers two maintenance operations when the system has been idle for 15 minutes:

1. **Episodic Consolidation** — The last 50 messages are compressed by the LLM into a structured JSON summary (`key_facts` + `user_preferences`) and stored as an `episodic_summary` memory. The conversation is then rotated to prevent unbounded context growth.
2. **Memory Compaction** — The `MemoryCompactor` scans for expired TTL entries, duplicate embeddings, and orphaned records, then prunes them.

### Temporal Decay Scoring

Retrieved memories are not ranked purely by cosine similarity. A temporal decay function applies an exponential weight:

```python
decay_factor = math.exp(-0.01 * age_days)
final_score = base_similarity * decay_factor * confidence_weight
```

Memories tagged `core_belief` are exempt from decay, ensuring foundational user preferences are never forgotten.

## Consequences

### Positive

- **Survives `kill -9`.** SQLite WAL journaling guarantees that committed transactions are recoverable after an abrupt daemon termination. The database has never corrupted in testing, even under simulated power-loss scenarios.
- **Zero-configuration first boot.** A new user gets working semantic memory immediately — no database server, no API keys, no cloud signup. The SQLite file is created automatically.
- **Microsecond retrieval.** For collections under 50K vectors, the vectorized NumPy fallback completes full-index cosine similarity in under 1ms. This is faster than the network round-trip to any cloud vector database.
- **Progressive enhancement.** Power users who install `qdrant-client` automatically get HNSW-indexed approximate nearest neighbor search without changing any configuration. The upgrade path is `pip install qdrant-client` and restart.
- **Full user data sovereignty.** All memory lives in `~/.zelyr/` and `~/.local/share/zelyr/`. No data ever leaves the machine. Users can back up, inspect, or delete their entire memory with standard filesystem tools.

### Negative

- **NumPy fallback does not scale past ~100K vectors.** Full-index brute-force cosine similarity loads the entire embedding matrix into RAM. For very large memory stores, the optional Qdrant/ChromaDB backends are required.
- **Dual-write complexity.** When a vector store is available, every `store()` call writes to both SQLite (for schema/metadata) and the vector store (for indexed search). This creates a consistency surface — if one write fails, the stores can diverge. We mitigate this with defensive error handling and log-based monitoring.
- **JSON-serialized embeddings in SQLite.** Storing float arrays as JSON text is space-inefficient (~4x overhead vs. binary encoding). This is acceptable for the fallback path but is the primary reason power users should upgrade to Qdrant.
- **Single-process lock contention.** Although WAL allows concurrent readers, only one writer can hold the database at a time. The `busy_timeout = 5000ms` mitigates this, but heavy concurrent tool execution can still experience brief write stalls.
