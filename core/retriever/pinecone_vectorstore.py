"""
Pinecone vector store adapter.
Manages embeddings and retrieval using Pinecone as the backend.
Supports fallback to in-memory store if Pinecone is unavailable.
"""
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger("agentic-rag")

try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False


class PineconeVectorStore:
    """
    Vector store using Pinecone as backend.
    Falls back to in-memory storage if Pinecone is unavailable.
    """

    def __init__(
        self,
        api_key: str,
        host: str,
        index_name: str = "rag",
        dimension: int = 1024,
        use_pinecone: bool = True,
    ):
        """
        Initialize Pinecone vector store.

        Args:
            api_key: Pinecone API key
            host: Pinecone host URL (e.g., https://rag-xxx.svc.aped-4627-b74a.pinecone.io)
            index_name: Name of the Pinecone index
            dimension: Dimension of embeddings (must match index)
            use_pinecone: Whether to use Pinecone (falls back to in-memory if False/unavailable)
        """
        self.api_key = api_key
        self.host = host
        self.index_name = index_name
        self.dimension = dimension
        self.use_pinecone = use_pinecone and PINECONE_AVAILABLE
        self.pc = None
        self.index = None

        # In-memory fallback
        self.in_memory_embeddings: Dict[str, np.ndarray] = {}
        self.in_memory_metadata: Dict[str, Dict] = {}
        self.in_memory_docids: Dict[str, str] = {}  # id -> doc_id mapping

        if self.use_pinecone:
            try:
                self._initialize_pinecone()
                logger.info(f"Initialized Pinecone vector store: index={index_name}, dimension={dimension}")
            except Exception as e:
                logger.warning(f"Failed to initialize Pinecone: {e}. Falling back to in-memory store.")
                self.use_pinecone = False

    def _initialize_pinecone(self):
        """Initialize Pinecone client and index."""
        if not PINECONE_AVAILABLE:
            raise Exception("pinecone package not installed. Run: pip install pinecone")

        self.pc = Pinecone(api_key=self.api_key, host=self.host)
        self.index = self.pc.Index(self.index_name)

        # Verify index connection
        stats = self.index.describe_index_stats()
        logger.debug(f"Pinecone index stats: {stats}")

    def add(
        self,
        embeddings: List[np.ndarray],
        ids: List[str],
        metadata: List[Dict],
        doc_id: Optional[str] = None,
    ) -> None:
        """
        Add embeddings to the vector store.

        Args:
            embeddings: List of embedding vectors (numpy arrays)
            ids: List of unique IDs for embeddings
            metadata: List of metadata dicts (one per embedding)
            doc_id: Document ID for tracking (optional)
        """
        if len(embeddings) != len(ids) or len(embeddings) != len(metadata):
            raise ValueError("embeddings, ids, and metadata must have same length")

        if self.use_pinecone:
            try:
                # Prepare vectors for Pinecone: convert numpy arrays to lists
                vectors_to_upsert = []
                for emb, id_, meta in zip(embeddings, ids, metadata):
                    vec = emb.tolist() if isinstance(emb, np.ndarray) else emb
                    vectors_to_upsert.append((id_, vec, meta))

                # Upsert to Pinecone
                self.index.upsert(vectors=vectors_to_upsert)
                logger.debug(f"Upserted {len(vectors_to_upsert)} vectors to Pinecone")
            except Exception as e:
                logger.error(f"Error upserting to Pinecone: {e}")
                # Fall back to in-memory
                self._add_to_memory(embeddings, ids, metadata, doc_id)
        else:
            self._add_to_memory(embeddings, ids, metadata, doc_id)

    def _add_to_memory(
        self,
        embeddings: List[np.ndarray],
        ids: List[str],
        metadata: List[Dict],
        doc_id: Optional[str] = None,
    ) -> None:
        """Add embeddings to in-memory store."""
        for emb, id_, meta in zip(embeddings, ids, metadata):
            self.in_memory_embeddings[id_] = (
                emb if isinstance(emb, np.ndarray) else np.array(emb)
            )
            self.in_memory_metadata[id_] = meta
            if doc_id:
                self.in_memory_docids[id_] = doc_id

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Tuple[str, float, Dict]]:
        """
        Search for top-k similar embeddings.

        Args:
            query_embedding: Query vector (numpy array)
            top_k: Number of results to return

        Returns:
            List of (id, similarity_score, metadata) tuples
        """
        if self.use_pinecone:
            try:
                query_vec = (
                    query_embedding.tolist()
                    if isinstance(query_embedding, np.ndarray)
                    else query_embedding
                )
                results = self.index.query(
                    vector=query_vec, top_k=top_k, include_metadata=True
                )

                # Convert Pinecone results to standard format
                output = []
                for match in results.get("matches", []):
                    output.append(
                        (
                            match["id"],
                            match.get("score", 0.0),
                            match.get("metadata", {}),
                        )
                    )
                return output
            except Exception as e:
                logger.error(f"Error searching Pinecone: {e}")
                # Fall back to in-memory
                return self._search_memory(query_embedding, top_k)
        else:
            return self._search_memory(query_embedding, top_k)

    def _search_memory(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Tuple[str, float, Dict]]:
        """Search in-memory store using cosine similarity."""
        if not self.in_memory_embeddings:
            return []

        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []

        query_normalized = query_embedding / query_norm

        # Compute cosine similarities
        similarities = []
        for id_, emb in self.in_memory_embeddings.items():
            emb_norm = np.linalg.norm(emb)
            if emb_norm > 0:
                emb_normalized = emb / emb_norm
                similarity = np.dot(query_normalized, emb_normalized)
                similarities.append((id_, similarity, self.in_memory_metadata.get(id_, {})))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get(self, id_: str) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Get a specific embedding by ID.

        Args:
            id_: Embedding ID

        Returns:
            Tuple of (embedding, metadata) or None if not found
        """
        if self.use_pinecone:
            try:
                results = self.index.fetch(ids=[id_])
                if results and results.get("vectors"):
                    vec_data = results["vectors"].get(id_)
                    if vec_data:
                        emb = np.array(vec_data.get("values", []))
                        metadata = vec_data.get("metadata", {})
                        return (emb, metadata)
            except Exception as e:
                logger.debug(f"Error fetching from Pinecone: {e}")
                return self._get_from_memory(id_)
        else:
            return self._get_from_memory(id_)

        return None

    def _get_from_memory(self, id_: str) -> Optional[Tuple[np.ndarray, Dict]]:
        """Get from in-memory store."""
        if id_ in self.in_memory_embeddings:
            return (
                self.in_memory_embeddings[id_],
                self.in_memory_metadata.get(id_, {}),
            )
        return None

    def delete(self, ids: List[str]) -> None:
        """Delete embeddings by IDs."""
        if self.use_pinecone:
            try:
                self.index.delete(ids=ids)
                logger.debug(f"Deleted {len(ids)} vectors from Pinecone")
            except Exception as e:
                logger.warning(f"Error deleting from Pinecone: {e}")
                self._delete_from_memory(ids)
        else:
            self._delete_from_memory(ids)

    def _delete_from_memory(self, ids: List[str]) -> None:
        """Delete from in-memory store."""
        for id_ in ids:
            self.in_memory_embeddings.pop(id_, None)
            self.in_memory_metadata.pop(id_, None)
            self.in_memory_docids.pop(id_, None)

    def clear(self) -> None:
        """Clear all embeddings."""
        if self.use_pinecone:
            try:
                # Delete all vectors (if Pinecone supports bulk delete)
                all_ids = [id_ for id_ in self.in_memory_embeddings.keys()]
                if all_ids:
                    self.index.delete(delete_all=True)
                logger.info("Cleared all vectors from Pinecone")
            except Exception as e:
                logger.warning(f"Error clearing Pinecone: {e}")
                self._clear_memory()
        else:
            self._clear_memory()

    def _clear_memory(self) -> None:
        """Clear in-memory store."""
        self.in_memory_embeddings.clear()
        self.in_memory_metadata.clear()
        self.in_memory_docids.clear()

    def size(self) -> int:
        """Get number of embeddings in the store."""
        if self.use_pinecone:
            try:
                stats = self.index.describe_index_stats()
                return stats.get("total_vector_count", 0)
            except Exception as e:
                logger.warning(f"Error getting Pinecone size: {e}")
                return len(self.in_memory_embeddings)
        else:
            return len(self.in_memory_embeddings)

    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        stats = {
            "backend": "pinecone" if self.use_pinecone else "in-memory",
            "dimension": self.dimension,
            "size": self.size(),
        }
        if self.use_pinecone:
            try:
                index_stats = self.index.describe_index_stats()
                stats["pinecone_stats"] = index_stats
            except Exception as e:
                logger.debug(f"Error getting index stats: {e}")
        return stats
