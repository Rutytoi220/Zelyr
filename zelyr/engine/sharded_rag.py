import logging
import hashlib
import asyncio
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ShardedRAGManager:
    """Manages distributed vector storage and retrieval across the P2P Mesh."""
    
    def __init__(self, node_ids: List[str] = None):
        # In a full system, node_ids would be dynamically updated via EventBus from MeshNodeServer
        self.node_ids = node_ids or []
        self.local_node_id = "local"
        
        if self.local_node_id not in self.node_ids:
            self.node_ids.append(self.local_node_id)
            
        self.node_ids.sort() # Ensure consistent ordering for hashing
        
    def _consistent_hash(self, doc_id: str) -> str:
        """Determines which node should store this document."""
        if not self.node_ids:
            return self.local_node_id
            
        hash_val = int(hashlib.md5(doc_id.encode('utf-8')).hexdigest(), 16)
        index = hash_val % len(self.node_ids)
        return self.node_ids[index]

    async def distribute_embeddings(self, documents: List[Dict[str, Any]]):
        """Asynchronously routes embedding chunks to the correct mesh nodes."""
        routing_table = {node: [] for node in self.node_ids}
        
        for doc in documents:
            doc_id = doc.get("id", str(hash(doc.get("content", ""))))
            target_node = self._consistent_hash(doc_id)
            routing_table[target_node].append(doc)
            
        for node, docs in routing_table.items():
            if not docs:
                continue
                
            if node == self.local_node_id:
                logger.info(f"ShardedRAG: Storing {len(docs)} documents locally.")
                # We would call vector_store.add() here
            else:
                logger.info(f"ShardedRAG: Broadcasting {len(docs)} documents to Mesh Node {node}")
                # We would use EventBus to publish to `pq_mesh.py` for broadcast
                
    async def query_shards(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Broadcasts a semantic search across all shards and merges results."""
        logger.info(f"ShardedRAG: Broadcasting query '{query}' to {len(self.node_ids)} nodes.")
        
        # Simulate local results
        results = [
            {"id": "doc_1", "score": 0.95, "content": "Local Mock Result", "node": self.local_node_id}
        ]
        
        # In reality, we'd wait for WebSocket responses from peers and merge+sort
        await asyncio.sleep(0.5)
        
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
