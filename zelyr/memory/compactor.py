"""Memory compaction algorithm for Zelyr."""
import asyncio
import json
import logging
from typing import Any, List, Dict
import numpy as np
logger = logging.getLogger(__name__)

class MemoryCompactor:
    """Detects and resolves highly similar memory entries."""

    def __init__(self, db: Any):
        """Initialize MemoryCompactor."""
        self._db = db
        self.similarity_threshold = 0.92

    async def run_compaction(self) -> Dict[str, Any]:
        """Scans the memories and resolves conflicts.
        
        Returns a report of actions taken.
        """
        cursor = await self._db.execute("SELECT e.id as emb_id, e.owner_id as memory_key, e.embedding_json, m.created_at, m.retrieval_count, m.confidence_weight, m.value_json FROM embeddings e JOIN memories m ON e.owner_id = m.key AND e.owner_type = 'memory'")
        rows = await cursor.fetchall()
        if not rows:
            return {'scanned': 0, 'merged': 0, 'deleted': 0}
        vectors = []
        metadata = []
        for row in rows:
            vec = np.array(json.loads(row['embedding_json']), dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
            metadata.append({'emb_id': row['emb_id'], 'key': row['memory_key'], 'created_at': row['created_at'], 'retrieval_count': row['retrieval_count'] or 0, 'confidence_weight': row['confidence_weight'] if row['confidence_weight'] is not None else 1.0, 'value_json': row['value_json']})
        merged_count = 0
        deleted_count = 0
        processed_keys = set()
        from datetime import datetime

        try:
            for i in range(len(vectors)):
                # Yield execution periodically to maintain WAL concurrency and avoid event loop starvation
                if i % 10 == 0:
                    await asyncio.sleep(0.01)

                key_i = metadata[i]['key']
                if key_i in processed_keys:
                    continue
                for j in range(i + 1, len(vectors)):
                    key_j = metadata[j]['key']
                    if key_j in processed_keys:
                        continue
                    sim = float(np.dot(vectors[i], vectors[j]))
                    if sim > self.similarity_threshold:
                        try:
                            dt_i = datetime.strptime(metadata[i]['created_at'], '%Y-%m-%d %H:%M:%S')
                            dt_j = datetime.strptime(metadata[j]['created_at'], '%Y-%m-%d %H:%M:%S')
                        except Exception:
                            dt_i = metadata[i]['created_at']
                            dt_j = metadata[j]['created_at']
                        if dt_i >= dt_j:
                            newer = metadata[i]
                            older = metadata[j]
                        else:
                            newer = metadata[j]
                            older = metadata[i]
                        new_retrieval_count = max(newer['retrieval_count'], older['retrieval_count'])
                        new_confidence = min(10.0, newer['confidence_weight'] + 0.1)
                        await self._db.execute('UPDATE memories SET retrieval_count = ?, confidence_weight = ? WHERE key = ?', (new_retrieval_count, new_confidence, newer['key']))
                        await self._db.execute('DELETE FROM memories WHERE key = ?', (older['key'],))
                        await self._db.execute("DELETE FROM embeddings WHERE owner_id = ? AND owner_type = 'memory'", (older['key'],))
                        processed_keys.add(older['key'])
                        newer['retrieval_count'] = new_retrieval_count
                        newer['confidence_weight'] = new_confidence
                        merged_count += 1
                        deleted_count += 1
            await self._db.commit()
        except Exception as exc:
            if hasattr(self._db, "rollback"):
                await self._db.rollback()
            raise exc

        return {'scanned': len(vectors), 'merged': merged_count, 'deleted': deleted_count}
