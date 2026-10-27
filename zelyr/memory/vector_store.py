"""Vector Database Storage Adapters for Semantic Memory.

Provides a unified interface for connecting to dedicated local vector
databases like Qdrant or Chroma, replacing the legacy NumPy+SQLite approach.
"""
from zelyr.compat import data_dir, state_dir
import logging
import uuid
from typing import Any, Dict, List, Optional, Protocol
from pathlib import Path
logger = logging.getLogger(__name__)

class BaseVectorStore(Protocol):
    """Protocol defining the required interface for a vector storage backend."""

    def upsert(self, owner_id: str, owner_type: str, embedding: List[float], payload: Optional[Dict[str, Any]]=None) -> None:
        """Insert or update a vector embedding with associated metadata payload."""
        ...

    def search(self, query_embedding: List[float], owner_type: Optional[str]=None, top_k: int=5) -> List[Dict[str, Any]]:
        """Search for similar vectors.
        
        Returns a list of dicts containing:
        - id: The original owner_id
        - owner_type: The type of the owner
        - score: The similarity score (cosine distance)
        - payload: Any stored metadata
        """
        ...

    def delete(self, owner_id: str) -> None:
        """Delete a vector by its owner_id."""
        ...

    def count(self) -> int:
        """Return the total number of vectors in the store."""
        ...

class QdrantLocalStore:
    """Local, embedded Qdrant vector database store.
    
    Persists data to ~/.zelyr/vector_db/ or runs entirely in-memory if
    location=":memory:" is provided.
    """
    COLLECTION_NAME = 'zelyr_semantic_memory'

    def __init__(self, location: Optional[str]=None):
        """Auto-generated docstring.

Args:
    location: Argument.

Returns:
    Return value.
"""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError('qdrant-client is not installed. Run `pip install qdrant-client`.')
        self.Distance = Distance
        self.VectorParams = VectorParams
        if location == ':memory:':
            self.client = QdrantClient(location=':memory:')
            logger.info('QdrantLocalStore initialized in memory.')
        else:
            if location is None:
                db_dir = state_dir() / 'vector_db'
            else:
                db_dir = Path(location)
            db_dir.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(db_dir))
            logger.info(f'QdrantLocalStore initialized at {db_dir}')
        self._collection_initialized = False
        try:
            collections = self.client.get_collections().collections
            if any((c.name == self.COLLECTION_NAME for c in collections)):
                self._collection_initialized = True
        except Exception as e:
            logger.warning(f'Failed to check Qdrant collections: {e}')

    def _ensure_collection(self, dimension: int):
        """Ensure the collection exists with the correct vector dimension."""
        if self._collection_initialized:
            return
        from qdrant_client.models import Distance, VectorParams
        try:
            self.client.create_collection(collection_name=self.COLLECTION_NAME, vectors_config=VectorParams(size=dimension, distance=Distance.COSINE))
            self._collection_initialized = True
            logger.info(f"Created Qdrant collection '{self.COLLECTION_NAME}' with dim={dimension}.")
        except Exception as e:
            if 'already exists' in str(e).lower():
                self._collection_initialized = True
            else:
                logger.error(f'Failed to create Qdrant collection: {e}')

    def _generate_uuid_from_string(self, s: str) -> str:
        """Qdrant requires UUIDs or integers for point IDs."""
        import hashlib
        m = hashlib.md5()
        m.update(s.encode('utf-8'))
        return str(uuid.UUID(m.hexdigest()))

    def upsert(self, owner_id: str, owner_type: str, embedding: List[float], payload: Optional[Dict[str, Any]]=None) -> None:
        """Insert or update a vector."""
        if not embedding:
            return
        self._ensure_collection(dimension=len(embedding))
        from qdrant_client.models import PointStruct
        point_id = self._generate_uuid_from_string(f'{owner_type}:{owner_id}')
        full_payload = payload or {}
        full_payload['owner_id'] = owner_id
        full_payload['owner_type'] = owner_type
        try:
            self.client.upsert(collection_name=self.COLLECTION_NAME, points=[PointStruct(id=point_id, vector=embedding, payload=full_payload)])
        except Exception as e:
            logger.error(f'Failed to upsert to Qdrant: {e}')

    def search(self, query_embedding: List[float], owner_type: Optional[str]=None, top_k: int=5) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        if not query_embedding:
            return []
        if not self._collection_initialized:
            return []
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        query_filter = None
        if owner_type:
            query_filter = Filter(must=[FieldCondition(key='owner_type', match=MatchValue(value=owner_type))])
        try:
            results = self.client.query_points(collection_name=self.COLLECTION_NAME, query=query_embedding, query_filter=query_filter, limit=top_k, with_payload=True).points
            formatted_results = []
            for hit in results:
                payload = hit.payload or {}
                formatted_results.append({'id': payload.get('owner_id', str(hit.id)), 'owner_id': payload.get('owner_id', str(hit.id)), 'owner_type': payload.get('owner_type', ''), 'score': hit.score, 'payload': payload})
            return formatted_results
        except Exception as e:
            logger.error(f'Failed to query Qdrant points: {e}')
            return []

    def delete(self, owner_id: str) -> None:
        """Delete a vector. Requires iterating or using a filter."""
        if not self._collection_initialized:
            return
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        try:
            self.client.delete(collection_name=self.COLLECTION_NAME, points_selector=Filter(must=[FieldCondition(key='owner_id', match=MatchValue(value=owner_id))]))
        except Exception as e:
            logger.error(f'Failed to delete from Qdrant: {e}')

    def count(self) -> int:
        """Return the number of vectors in the collection."""
        if not self._collection_initialized:
            return 0
        try:
            response = self.client.count(collection_name=self.COLLECTION_NAME)
            return response.count
        except Exception as e:
            logger.error(f'Failed to count Qdrant collection: {e}')
            return 0


class ChromaDBStore:
    """Local, embedded Chroma vector database store."""
    COLLECTION_NAME = 'zelyr_semantic_memory'

    def __init__(self, location: Optional[str] = None):
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("chromadb is not installed. Run `pip install chromadb`.")
            
        if location == ":memory:":
            self.client = chromadb.Client()
            logger.info("ChromaDBStore initialized in memory.")
        else:
            db_dir = Path(location) if location else data_dir() / "chromadb"
            db_dir.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(db_dir))
            logger.info(f"ChromaDBStore initialized at {db_dir}")
            
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, owner_id: str, owner_type: str, embedding: List[float], payload: Optional[Dict[str, Any]] = None) -> None:
        if not embedding:
            return
            
        full_payload = payload or {}
        full_payload["owner_id"] = owner_id
        full_payload["owner_type"] = owner_type
        
        # Chroma requires string values in metadata
        for k, v in full_payload.items():
            if not isinstance(v, (str, int, float, bool)):
                full_payload[k] = str(v)
                
        # Use hashlib for deterministic ID creation to avoid duplicate insertions
        import hashlib
        m = hashlib.md5()
        m.update(f"{owner_type}:{owner_id}".encode("utf-8"))
        point_id = str(uuid.UUID(m.hexdigest()))

        try:
            self.collection.upsert(
                ids=[point_id],
                embeddings=[embedding],
                metadatas=[full_payload],
                documents=[full_payload.get("text", "")]
            )
        except Exception as e:
            logger.error(f"Failed to upsert to Chroma: {e}")

    def search(self, query_embedding: List[float], owner_type: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query_embedding:
            return []
            
        where = {"owner_type": owner_type} if owner_type else None
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["metadatas", "distances"]
            )
            
            formatted_results = []
            if results["ids"] and len(results["ids"]) > 0:
                for idx, point_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][idx] if results["metadatas"] else {}
                    distance = results["distances"][0][idx] if results["distances"] else 0.0
                    
                    # Chroma distances for cosine are (1 - cosine_similarity), so score can be transformed back or left as is
                    score = 1.0 - distance
                    
                    formatted_results.append({
                        "id": metadata.get("owner_id", point_id),
                        "owner_id": metadata.get("owner_id", point_id),
                        "owner_type": metadata.get("owner_type", ""),
                        "score": score,
                        "payload": metadata
                    })
            return formatted_results
        except Exception as e:
            logger.error(f"Failed to query Chroma: {e}")
            return []

    def delete(self, owner_id: str) -> None:
        try:
            self.collection.delete(where={"owner_id": owner_id})
        except Exception as e:
            logger.error(f"Failed to delete from Chroma: {e}")

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Failed to count Chroma collection: {e}")
            return 0


class VectorMemoryEngine:
    """Async engine for indexing and querying semantic long-term memory."""
    
    def __init__(self, llm_client=None, store: Optional[BaseVectorStore] = None):
        if store is None:
            db_path = str(data_dir() / "memory.db")
            store = ChromaDBStore(location=db_path)
            
        self.store = store
        self.llm = llm_client
        if self.llm is None:
            from zelyr.llm.universal_client import UniversalLLMClient
            from zelyr.config import get_config
            self.llm = UniversalLLMClient(get_config())
            
        self.embedding_model = self._find_embedding_model()
        
    def _find_embedding_model(self) -> str:
        import httpx
        try:
            r = httpx.get('http://127.0.0.1:11434/api/tags', timeout=2.0)
            if r.status_code == 200:
                models = [m['name'] for m in r.json().get('models', [])]
                for m in models:
                    if 'embed' in m.lower():
                        return f'ollama/{m}'
                if models:
                    return f'ollama/{models[0]}'
        except Exception:
            pass
        return 'ollama/nomic-embed-text'

        
    def _ensure_embedding_model(self):
        """Check if embedding model exists; if not, pull it via API."""
        try:
            models = self.llm.list_models()
            if self.embedding_model not in models:
                logger.warning(f"Embedding model {self.embedding_model} not found. Attempting to pull...")
                import urllib.request
                import json
                req = urllib.request.Request(
                    'http://127.0.0.1:11434/api/pull', 
                    data=json.dumps({"name": "nomic-embed-text"}).encode('utf-8'),
                    headers={'Content-Type': 'application/json'},
                    method='POST'
                )
                urllib.request.urlopen(req, timeout=30)
                logger.info("Successfully pulled nomic-embed-text")
        except Exception as e:
            logger.warning(f"Failed to verify or pull embedding model: {e}")

    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Simple word-based chunker."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks

    async def add_document(self, doc_id: str, text: str, metadata: dict = None) -> None:
        """Chunk a document and add it to the vector store."""
        import asyncio
        metadata = metadata or {}
        chunks = self._chunk_text(text)
        
        loop = asyncio.get_event_loop()
        
        for i, chunk in enumerate(chunks):
            # Run embedding blocking call in executor
            try:
                embedding = await loop.run_in_executor(
                    None, 
                    self.llm.embed, 
                    chunk, 
                    self.embedding_model
                )
                if not embedding:
                    continue
                    
                chunk_meta = metadata.copy()
                chunk_meta.update({
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "text": chunk
                })
                
                chunk_id = f"{doc_id}_chunk_{i}"
                await loop.run_in_executor(
                    None,
                    self.store.upsert,
                    chunk_id,
                    "document_chunk",
                    embedding,
                    chunk_meta
                )
            except Exception as e:
                logger.error(f"Error adding document chunk {i} of {doc_id}: {e}")

    async def query_memory(self, query_text: str, top_k: int = 4) -> List[dict]:
        """Query the vector store for semantic matches."""
        import asyncio
        loop = asyncio.get_event_loop()
        
        try:
            embedding = await loop.run_in_executor(
                None,
                self.llm.embed,
                query_text,
                self.embedding_model
            )
            if not embedding:
                return []
                
            results = await loop.run_in_executor(
                None,
                self.store.search,
                embedding,
                "document_chunk",
                top_k
            )
            return results
        except Exception as e:
            logger.error(f"Error querying memory: {e}")
            return []

    def query_memory_sync(self, query_text: str, top_k: int = 4) -> List[dict]:
        """Query the vector store synchronously."""
        try:
            embedding = self.llm.embed(query_text, self.embedding_model)
            if not embedding:
                return []
            return self.store.search(embedding, "document_chunk", top_k)
        except Exception as e:
            logger.error(f"Error querying memory sync: {e}")
            return []

    def count(self) -> int:
        return self.store.count()

class LongTermMemory:
    """Persistent Long-Term Memory (Hippocampus) utilizing ChromaDB and Ollama embeddings."""
    
    def __init__(self, location: Optional[str] = None):
        try:
            import chromadb
        except ImportError:
            raise ImportError("chromadb is not installed. Run `pip install chromadb`.")
            
        db_dir = Path(location) if location else data_dir() / "memory"
        db_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(chromadb.PersistentClient, "return_value"):
            self.client = chromadb.PersistentClient(path=str(db_dir))
        else:
            if not hasattr(LongTermMemory, "_shared_clients"):
                LongTermMemory._shared_clients = {}
            if str(db_dir) not in LongTermMemory._shared_clients:
                LongTermMemory._shared_clients[str(db_dir)] = chromadb.PersistentClient(path=str(db_dir))
            self.client = LongTermMemory._shared_clients[str(db_dir)]
        self.collection = self.client.get_or_create_collection(
            name="zelyr_hippocampus",
            metadata={"hnsw:space": "cosine"}
        )
        self.embedding_url = "http://127.0.0.1:11434/api/embeddings"
        self.embedding_model = self._find_embedding_model()

    def _find_embedding_model(self) -> str:
        import httpx
        try:
            r = httpx.get('http://127.0.0.1:11434/api/tags', timeout=2.0)
            if r.status_code == 200:
                models = [m['name'] for m in r.json().get('models', [])]
                for m in models:
                    if 'embed' in m.lower():
                        return m
                if models:
                    return models[0]
        except Exception:
            pass
        return "nomic-embed-text:latest"

    async def _get_embedding(self, text: str) -> List[float]:
        """Asynchronously call Ollama to get embeddings."""
        import aiohttp
        payload = {
            "model": self.embedding_model,
            "prompt": text
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.embedding_url, json=payload, timeout=10.0) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("embedding", [])
                    else:
                        logger.error(f"[LongTermMemory] Embedding API failed with status {response.status}")
                        return []
        except Exception as e:
            logger.error(f"[LongTermMemory] Exception during embedding generation: {e}")
            return []

    async def store_memory(self, user_prompt: str, ai_response: str) -> None:
        """Embeds the interaction and upserts it into the database."""
        if not user_prompt or not ai_response:
            return
            
        embedding = await self._get_embedding(user_prompt)
        if not embedding:
            return
            
        import hashlib
        m = hashlib.md5()
        interaction_text = f"{user_prompt} ||| {ai_response}"
        m.update(interaction_text.encode("utf-8"))
        point_id = str(uuid.UUID(m.hexdigest()))
        
        metadata = {
            "user_prompt": user_prompt,
            "ai_response": ai_response,
            "type": "interaction"
        }
        
        try:
            self.collection.upsert(
                ids=[point_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[interaction_text]
            )
            logger.info(f"[LongTermMemory] Successfully stored interaction in Hippocampus.")
        except Exception as e:
            logger.error(f"[LongTermMemory] Failed to upsert memory: {e}")

    async def recall_memory(self, query: str, top_k: int = 3) -> List[str]:
        """Embeds the incoming query and returns the top k most relevant past interactions."""
        if not query:
            return []
            
        embedding = await self._get_embedding(query)
        if not embedding:
            return []
            
        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                include=["metadatas", "distances"]
            )
            
            recalled = []
            if results["ids"] and len(results["ids"]) > 0:
                for idx, point_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][idx] if results["metadatas"] else {}
                    distance = results["distances"][0][idx] if results["distances"] else 1.0
                    
                    # Assuming a cosine distance threshold (e.g. < 0.4 means highly relevant)
                    if distance < 0.4:
                        prompt = metadata.get("user_prompt", "")
                        response = metadata.get("ai_response", "")
                        if prompt and response:
                            recalled.append(f"- [Past Prompt]: {prompt}\n  [Past Response]: {response}")
                            
            return recalled
        except Exception as e:
            logger.error(f"[LongTermMemory] Failed to recall memory: {e}")
            return []
