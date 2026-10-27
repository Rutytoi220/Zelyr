"""Semantic search index for Zelyr memory.

Stores embedding vectors and performs cosine similarity search.
Uses local vector database (Qdrant) if available, falling back
to in-memory NumPy + SQLite.
"""
from zelyr.compat import state_dir
import json
import math
import time
import asyncio
import logging
from typing import Any, Dict, List, Optional, Protocol
logger = logging.getLogger(__name__)

class EmbeddingProvider(Protocol):
    """Protocol for embedding generation. OllamaClient satisfies this."""

    def embed(self, text: str, model: Optional[str]=None) -> List[float]:
        """Auto-generated docstring.

Args:
    text: Argument.
    model: Argument.

Returns:
    Return value.
"""
        ...

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors (legacy fallback)."""
    import numpy as np
    if len(a) != len(b) or not a:
        return 0.0
    vec_a = np.array(a, dtype=np.float32)
    vec_b = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

class SemanticIndex:
    """Embedding storage and similarity search.
    
    Automatically prefers QdrantLocalStore if `qdrant-client` is installed,
    otherwise falls back to SQLite + NumPy array computations.
    """

    def __init__(self, provider: Optional[EmbeddingProvider]=None, event_bus=None):
        """Initialize semantic index."""
        import collections
        self._provider = provider
        self._event_bus = event_bus
        self._vector_store = None
        self._embedding_cache = collections.OrderedDict()
        self._cache_max_size = 5000
        
        try:
            from zelyr.memory.vector_store import QdrantLocalStore
            import sys
            if 'pytest' in sys.modules:
                self._vector_store = QdrantLocalStore(location=':memory:')
            else:
                self._vector_store = QdrantLocalStore()
        except ImportError:
            logger.warning('qdrant-client not installed. SemanticIndex falling back to NumPy+SQLite.')
        except Exception as e:
            if 'lock' in str(e).lower() or 'accessed by another instance' in str(e).lower():
                import os
                from pathlib import Path
                lock_file = state_dir() / 'vector_db' / '.lock'
                if lock_file.exists():
                    try:
                        os.remove(lock_file)
                        logger.info('Cleared dangling Qdrant lock file.')
                        self._vector_store = QdrantLocalStore()
                        return
                    except Exception as lock_err:
                        logger.warning('Could not clear Qdrant lock file: %s', lock_err)
            logger.warning('Failed to initialize QdrantLocalStore: %s. Falling back to NumPy+SQLite.', e)

    @property
    def has_provider(self) -> bool:
        return self._provider is not None

    async def store(self, db: Any, owner_id: str, owner_type: str, embedding: List[float], model: str='') -> None:
        """Persist an embedding vector."""
        await db.execute('INSERT INTO embeddings (owner_id, owner_type, embedding_json, model) VALUES (?, ?, ?, ?)', (owner_id, owner_type, json.dumps(embedding), model))
        await db.commit()
        if self._vector_store:
            try:
                await asyncio.to_thread(self._vector_store.upsert, owner_id, owner_type, embedding, {'model': model})
            except Exception as e:
                logger.error('Failed to asynchronously upsert to Qdrant: %s', e)

    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        import hashlib
        h = hashlib.sha256(text.encode('utf-8')).hexdigest()
        if h in self._embedding_cache:
            self._embedding_cache.move_to_end(h)
            return self._embedding_cache[h]
        return None

    def _cache_embedding(self, text: str, embedding: List[float]) -> None:
        import hashlib
        h = hashlib.sha256(text.encode('utf-8')).hexdigest()
        self._embedding_cache[h] = embedding
        if len(self._embedding_cache) > self._cache_max_size:
            self._embedding_cache.popitem(last=False)

    def _emit_error(self, message: str):
        if self._event_bus:
            from zelyr.core.events import Event
            try:
                self._event_bus.publish(Event(event_type="system.alert", source="SemanticIndex", data={"message": message, "level": "warning"}))
            except Exception:
                pass

    async def store_text(self, db: Any, owner_id: str, owner_type: str, text: str, model: Optional[str]=None) -> bool:
        """Generate embedding from text via provider, then store. Returns False if no provider."""
        if not self._provider:
            return False
            
        cached = self._get_cached_embedding(text)
        if cached:
            await self.store(db, owner_id, owner_type, cached, model=model or '')
            return True
            
        try:
            embedding = await asyncio.to_thread(self._provider.embed, text, model)
        except Exception as e:
            logger.error('Failed to generate embedding: %s', e)
            self._emit_error(f"Embedding failed: {e}. Falling back to lexical search.")
            return False
            
        if not embedding or not isinstance(embedding, list) or not isinstance(embedding[0], float):
            return False
            
        self._cache_embedding(text, embedding)
        await self.store(db, owner_id, owner_type, embedding, model=model or '')
        return True

    async def store_texts_batch(self, db: Any, items: List[Dict[str, Any]], model: Optional[str]=None) -> bool:
        """Process a batch of texts for embeddings."""
        if not self._provider or not items:
            return False
            
        texts_to_embed = []
        indices_to_embed = []
        
        # Check cache first
        for i, item in enumerate(items):
            cached = self._get_cached_embedding(item['text'])
            if cached:
                item['embedding'] = cached
            else:
                texts_to_embed.append(item['text'])
                indices_to_embed.append(i)
                
        if texts_to_embed:
            try:
                # Use UniversalLLMClient's batch embed
                embeddings = await asyncio.to_thread(self._provider.embed, texts_to_embed, model)
                if embeddings and isinstance(embeddings, list) and isinstance(embeddings[0], list):
                    for idx, emb in zip(indices_to_embed, embeddings):
                        if emb:
                            items[idx]['embedding'] = emb
                            self._cache_embedding(items[idx]['text'], emb)
            except Exception as e:
                logger.error('Failed to generate batch embeddings: %s', e)
                self._emit_error(f"Batch embedding failed: {e}. Falling back to lexical search.")
                return False
                
        # Store all successful ones
        for item in items:
            if 'embedding' in item and item['embedding']:
                await self.store(db, item['owner_id'], item['owner_type'], item['embedding'], model=model or '')
                
        return True

    def _apply_decay(self, base_sim: float, row: dict, now: float) -> dict:
        """Apply temporal decay and confidence weight to a score."""
        import math
        from datetime import datetime
        decay_factor = 1.0
        confidence_weight = row['confidence_weight'] if row['confidence_weight'] is not None else 1.0
        tags = json.loads(row['tags_json']) if row['tags_json'] else []
        if 'core_belief' not in tags:
            created_at_str = row['mem_created_at'] or row['emb_created_at']
            if created_at_str:
                try:
                    created_dt = datetime.strptime(created_at_str, '%Y-%m-%d %H:%M:%S')
                    age_seconds = now - created_dt.timestamp()
                    age_days = max(0.0, age_seconds / 86400.0)
                    decay_rate = 0.01
                    decay_factor = math.exp(-decay_rate * age_days)
                except Exception:
                    pass
        final_score = base_sim * decay_factor * confidence_weight
        return {'similarity': final_score, 'base_similarity': base_sim, 'decay_factor': decay_factor, 'confidence_weight': confidence_weight}

    async def search(self, db: Any, query_embedding: List[float], owner_type: str='', top_k: int=5) -> List[Dict[str, Any]]:
        """Find most similar entries."""
        import time
        now = time.time()
        if self._vector_store:
            try:
                qdrant_results = await asyncio.to_thread(self._vector_store.search, query_embedding, owner_type, top_k * 2)
                if not qdrant_results:
                    return []
                owner_ids = [hit['owner_id'] for hit in qdrant_results]
                placeholders = ','.join(['?'] * len(owner_ids))
                query = f"SELECT e.id, e.owner_id, e.owner_type, e.model, e.created_at as emb_created_at, m.created_at as mem_created_at, m.confidence_weight, m.tags_json FROM embeddings e LEFT JOIN memories m ON e.owner_id = m.key AND e.owner_type = 'memory' WHERE e.owner_id IN ({placeholders})"
                try:
                    cursor = await db.execute(query, tuple(owner_ids))
                except Exception:
                    query = f'SELECT id, owner_id, owner_type, model, created_at as emb_created_at, NULL as mem_created_at, 1.0 as confidence_weight, NULL as tags_json FROM embeddings WHERE owner_id IN ({placeholders})'
                    cursor = await db.execute(query, tuple(owner_ids))
                db_rows = await cursor.fetchall()
                db_lookup = {row['owner_id']: row for row in db_rows}
                scored = []
                for hit in qdrant_results:
                    owner_id = hit['owner_id']
                    if owner_id not in db_lookup:
                        continue
                    row = db_lookup[owner_id]
                    base_sim = hit['score']
                    metrics = self._apply_decay(base_sim, row, now)
                    scored.append({'id': row['id'], 'owner_id': row['owner_id'], 'owner_type': row['owner_type'], 'model': row['model'], **metrics})
                scored.sort(key=lambda x: x['similarity'], reverse=True)
                return scored[:top_k]
            except Exception as e:
                logger.error('Qdrant search failed, falling back to SQLite: %s', e)
        query = "SELECT e.id, e.owner_id, e.owner_type, e.embedding_json, e.model, e.created_at as emb_created_at, m.created_at as mem_created_at, m.confidence_weight, m.tags_json FROM embeddings e LEFT JOIN memories m ON e.owner_id = m.key AND e.owner_type = 'memory'"
        if owner_type:
            query += ' WHERE e.owner_type = ?'
            params: tuple = (owner_type,)
        else:
            params = ()
        try:
            cursor = await db.execute(query, params)
        except Exception:
            query = 'SELECT id, owner_id, owner_type, embedding_json, model, created_at as emb_created_at, NULL as mem_created_at, 1.0 as confidence_weight, NULL as tags_json FROM embeddings'
            if owner_type:
                query += ' WHERE owner_type = ?'
            cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        if not rows:
            return []
        import numpy as np
        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []
            
        # Vectorized numpy math for zero-overhead cosine similarity
        stored_vectors = [json.loads(row['embedding_json']) for row in rows]
        s_matrix = np.array(stored_vectors, dtype=np.float32)
        
        # Calculate norms for all stored vectors simultaneously
        s_norms = np.linalg.norm(s_matrix, axis=1)
        s_norms[s_norms == 0] = 1.0  # Prevent division by zero
        
        # Calculate dot products and divide by norms to get cosine similarities
        similarities = np.dot(s_matrix, q_vec) / (s_norms * q_norm)
        
        scored = []
        for i, row in enumerate(rows):
            base_sim = float(similarities[i])
            metrics = self._apply_decay(base_sim, row, now)
            scored.append({'id': row['id'], 'owner_id': row['owner_id'], 'owner_type': row['owner_type'], 'model': row['model'], **metrics})
        scored.sort(key=lambda x: x['similarity'], reverse=True)
        return scored[:top_k]

    async def search_text(self, db: Any, text: str, owner_type: str='', top_k: int=5) -> List[Dict[str, Any]]:
        """Search by text using the embedding provider."""
        if not self._provider:
            return []
        try:
            embedding = await asyncio.to_thread(self._provider.embed, text)
        except Exception as e:
            logger.error('Failed to generate embedding for search: %s', e)
            return []
        if not embedding:
            return []
        return await self.search(db, embedding, owner_type=owner_type, top_k=top_k)


# ---------------------------------------------------------------------------
# Persistent SQLite Semantic Vector Memory Engine
# ---------------------------------------------------------------------------

import struct
import re
from collections import Counter
import httpx
from zelyr.db.memory import get_db_path, get_connection, init_db


def generate_embedding(
    text: str,
    base_url: Optional[str] = None,
    model: str = "nomic-embed-text",
) -> Optional[List[float]]:
    """Generate embedding vector using local Ollama endpoint."""
    if not text or not text.strip():
        return None

    if base_url is None:
        try:
            from zelyr.config import get_config
            cfg = get_config()
            base_url = getattr(cfg, "ollama_base_url", "http://127.0.0.1:11434").rstrip("/")
        except Exception:
            base_url = "http://127.0.0.1:11434"

    try:
        with httpx.Client(timeout=4.0) as client:
            resp = client.post(
                f"{base_url}/api/embeddings",
                json={"model": model, "prompt": text},
            )
            if resp.status_code == 200:
                data = resp.json()
                emb = data.get("embedding")
                if emb and isinstance(emb, list):
                    return [float(x) for x in emb]
    except Exception as e:
        logger.debug(f"Ollama embedding request failed ({e}), falling back to lexical search.")
    return None


def serialize_embedding(vec: Optional[List[float]]) -> Optional[bytes]:
    """Serialize list of floats into binary BLOB."""
    if not vec:
        return None
    return struct.pack(f"{len(vec)}f", *vec)


def deserialize_embedding(blob: Optional[bytes]) -> List[float]:
    """Deserialize binary BLOB into list of floats."""
    if not blob:
        return []
    n = len(blob) // 4
    try:
        return list(struct.unpack(f"{n}f", blob))
    except Exception:
        return []


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for a, b in zip(v1, v2):
        dot += a * b
        norm1 += a * a
        norm2 += b * b
    if norm1 <= 0.0 or norm2 <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm1) * math.sqrt(norm2))


def bm25_search(query: str, documents: List[tuple[int, str]], top_k: int = 3) -> List[str]:
    """Fallback lexical BM25 search when vector embeddings are unavailable."""
    def tokenize(t: str) -> List[str]:
        return re.findall(r"\w+", t.lower())

    query_tokens = tokenize(query)
    if not query_tokens or not documents:
        return []

    doc_tokens = [tokenize(doc[1]) for doc in documents]
    n_docs = len(documents)

    df: Counter[str] = Counter()
    for tokens in doc_tokens:
        for t in set(tokens):
            df[t] += 1

    idf = {t: math.log((n_docs - df[t] + 0.5) / (df[t] + 0.5) + 1.0) for t in query_tokens}

    k1 = 1.5
    b = 0.75
    avgdl = sum(len(t) for t in doc_tokens) / max(n_docs, 1)

    scored = []
    for idx, (_, text) in enumerate(documents):
        tokens = doc_tokens[idx]
        dl = len(tokens)
        tf = Counter(tokens)
        score = 0.0
        for qt in query_tokens:
            if qt in tf:
                freq = tf[qt]
                numerator = freq * (k1 + 1)
                denominator = freq + k1 * (1 - b + b * (dl / avgdl)) if avgdl > 0 else 1.0
                score += idf.get(qt, 0.0) * (numerator / max(denominator, 1e-6))
        if score > 0.0:
            scored.append((score, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in scored[:top_k]]


def add_memory(fact: str, db_path: Optional[str] = None) -> int:
    """Store a permanent fact in the SQLite vector memory store."""
    fact = (fact or "").strip()
    if not fact:
        return -1

    path = db_path or get_db_path()
    init_db(path)

    emb = generate_embedding(fact)
    blob = serialize_embedding(emb)

    now = time.time()
    with get_connection(path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO memories (category, content, confidence, is_active, created_at, updated_at, embedding)
            VALUES ('preference', ?, 1.0, 1, ?, ?, ?)
            """,
            (fact, now, now, blob),
        )
        conn.commit()
        return cursor.lastrowid or -1


def search_memories(
    query: str,
    top_k: int = 3,
    db_path: Optional[str] = None,
) -> List[str]:
    """Retrieve top_k matching memories using cosine similarity or BM25 fallback."""
    query = (query or "").strip()
    if not query:
        return []

    path = db_path or get_db_path()
    init_db(path)

    with get_connection(path) as conn:
        cursor = conn.execute("SELECT id, content, embedding FROM memories WHERE is_active = 1")
        rows = cursor.fetchall()

    if not rows:
        return []

    q_emb = generate_embedding(query)
    scored: List[tuple[float, str]] = []

    if q_emb:
        for row in rows:
            blob = row["embedding"]
            if blob:
                mem_emb = deserialize_embedding(blob)
                if mem_emb and len(mem_emb) == len(q_emb):
                    sim = cosine_similarity(q_emb, mem_emb)
                    if sim > 0.25:
                        scored.append((sim, row["content"]))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        seen = set()
        results = []
        for _, content in scored:
            if content not in seen:
                seen.add(content)
                results.append(content)
                if len(results) >= top_k:
                    break
        return results

    # Fallback to BM25 lexical search
    docs = [(row["id"], row["content"]) for row in rows]
    return bm25_search(query, docs, top_k=top_k)


def get_all_memories(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve all stored permanent memories ordered by most recent first."""
    path = db_path or get_db_path()
    init_db(path)
    with get_connection(path) as conn:
        cursor = conn.execute(
            "SELECT id, content, created_at FROM memories WHERE is_active = 1 ORDER BY id DESC"
        )
        return [dict(row) for row in cursor.fetchall()]


class SemanticMemoryStore:
    """Manager class for persistent semantic memory operations."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or get_db_path()
        init_db(self.db_path)

    def add_memory(self, fact: str) -> int:
        return add_memory(fact, self.db_path)

    def search_memories(self, query: str, top_k: int = 3) -> List[str]:
        return search_memories(query, top_k=top_k, db_path=self.db_path)

    def get_all_memories(self) -> List[Dict[str, Any]]:
        return get_all_memories(self.db_path)

