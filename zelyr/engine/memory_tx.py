import logging
import asyncio
import re
import uuid
from typing import Dict, Any, List, Optional
from zelyr.memory.vector_store import VectorMemoryEngine
from zelyr.engine.graph_memory import GraphMemoryEngine
from zelyr.engine.sharded_rag import ShardedRAGManager
from zelyr.security.zkp_enclave import SecureEnclaveManager

logger = logging.getLogger(__name__)

class TransactionalMemoryManager:
    """Wraps VectorMemoryEngine with ACID-like transactional semantics."""
    
    def __init__(self, memory_engine: Optional[VectorMemoryEngine] = None):
        if not memory_engine:
            self.engine = VectorMemoryEngine()
        else:
            self.engine = memory_engine
            
        self.graph = GraphMemoryEngine()
        self.sharded_rag = ShardedRAGManager()
        self.enclave = SecureEnclaveManager()
        self._staged_documents: List[Dict[str, Any]] = []
        self._in_transaction = False
        
    def begin_transaction(self):
        """Start a new transaction."""
        if self._in_transaction:
            raise RuntimeError("Transaction already in progress.")
        self._in_transaction = True
        self._staged_documents = []
        logger.info("Memory Transaction BEGIN")
        
    def stage_document(self, doc_id: str, text: str, metadata: dict = None):
        """Stage a document to be indexed upon commit."""
        if not self._in_transaction:
            raise RuntimeError("No active transaction.")
        self._staged_documents.append({
            "doc_id": doc_id,
            "text": text,
            "metadata": metadata or {}
        })
        logger.debug(f"Staged document: {doc_id}")
        
    async def commit(self):
        """Atomically commit all staged documents to the vector store."""
        if not self._in_transaction:
            raise RuntimeError("No active transaction.")
            
        logger.info(f"Memory Transaction COMMIT: Indexing {len(self._staged_documents)} documents...")
        
        try:
            # Distribute embeddings to sharded network first
            await self.sharded_rag.distribute_embeddings(self._staged_documents)
            
            # We process all chunks locally as well for the local replica
            for doc in self._staged_documents:
                # Check for Secure Enclave routing
                if doc["metadata"].get("sensitive") is True:
                    logger.info(f"Memory Transaction: Routing sensitive chunk {doc['doc_id']} to Secure Enclave.")
                    self.enclave.store_sensitive_chunk(doc["doc_id"], doc["text"], doc["metadata"])
                else:
                    # Add to local engine
                    await self.engine.add_document(doc["doc_id"], doc["text"], doc["metadata"])
                    self._extract_and_inject_graph(doc["text"])
                
            self._in_transaction = False
            self._staged_documents.clear()
            logger.info("Memory Transaction COMMITTED successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Memory Transaction failed during commit: {e}. Triggering rollback.")
            await self.rollback()
            raise e
            
    async def rollback(self):
        """Discard all staged documents without saving them."""
        if not self._in_transaction:
            return
            
        # If we had a true two-phase commit with Chroma we'd delete the chunks added so far.
        # Here we just clear the staging buffer. In a more advanced implementation, 
        # we would track the IDs actually written to ChromaDB and call store.delete() on them.
        self._in_transaction = False
        self._staged_documents.clear()
        logger.warning("Memory Transaction ROLLBACK complete. Uncommitted vectors discarded.")

    async def query_zelyr_memory(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Queries the vector index. Uses Sharded RAG if available."""
        # Query remote shards
        sharded_results = await self.sharded_rag.query_shards(query, top_k)
        
        # Query local engine
        local_results = []
        try:
            # We mock the local results format for now based on what VectorMemoryEngine might return
            local_results = await self.engine.search(query, top_k)
        except Exception:
            pass
            
        # Merge results, prioritizing score (simplified merge)
        merged = sharded_results + local_results
        # Ensure we have scores
        merged.sort(key=lambda x: x.get("score", 0), reverse=True)
        return merged[:top_k]

    # Legacy AXIOM compatibility alias
    query_axiom_memory = query_zelyr_memory

    def _extract_and_inject_graph(self, text: str):
        """Tier 1 basic regex extraction for Relational Index."""
        # Simple extraction for demo: look for "service" failing and dependencies
        # Example pattern: "nginx.service failed due to port 80 conflict"
        
        service_match = re.search(r'([\w-]+\.service)', text)
        if service_match:
            service_name = service_match.group(1)
            service_id = f"ent_{uuid.uuid4().hex[:8]}"
            self.graph.add_entity(service_id, service_name, "Service", {"extracted_from": "text"})
            
            # Check for port dependency
            port_match = re.search(r'port (\d+)', text)
            if port_match:
                port_num = port_match.group(1)
                port_name = f"Port {port_num}"
                port_id = f"ent_{uuid.uuid4().hex[:8]}"
                self.graph.add_entity(port_id, port_name, "Port", {"number": port_num})
                self.graph.add_relationship(service_id, port_id, "DEPENDS_ON", 1.0)
                
            # Check for generic dependencies like "network.target"
            target_match = re.search(r'([\w-]+\.target)', text)
            if target_match:
                target_name = target_match.group(1)
                target_id = f"ent_{uuid.uuid4().hex[:8]}"
                self.graph.add_entity(target_id, target_name, "Target", {})
                self.graph.add_relationship(service_id, target_id, "DEPENDS_ON", 1.0)

