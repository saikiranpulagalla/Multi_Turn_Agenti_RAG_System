"""
Adapter interface to vector DB. Implement methods for FAISS / Pinecone / Weaviate / Chroma.
This is a minimal FAISS-like in-memory adapter example with improved similarity calculations.
"""
from typing import List, Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger("agentic-rag")


class InMemoryVectorStore:
    """
    In-memory vector store using numpy arrays for similarity search.
    Suitable for small to medium datasets. For production, use FAISS, Pinecone, or similar.
    """
    
    def __init__(self):
        # simple dict: id -> (embedding, metadata)
        self.store: Dict[str, Dict[str, Any]] = {}
        self.embedding_dim: Optional[int] = None
    
    def clear(self):
        """Clear all stored vectors and embeddings."""
        self.store.clear()
        self.embedding_dim = None
        logger.debug("InMemoryVectorStore cleared")

    def add(self, id: str, embedding: List[float], metadata: Dict[str, Any]):
        """
        Add a vector to the store with metadata.
        
        Args:
            id: Unique identifier for this vector
            embedding: Vector embedding as list of floats
            metadata: Associated metadata dict
        """
        if not id:
            raise ValueError("Vector ID cannot be empty")
        
        if not embedding:
            raise ValueError("Embedding cannot be empty")
        
        try:
            emb_array = np.array(embedding, dtype=np.float32)
            
            # Validate embedding dimension consistency
            if self.embedding_dim is None:
                self.embedding_dim = len(emb_array)
            elif len(emb_array) != self.embedding_dim:
                logger.warning(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {len(emb_array)}")
                # Pad or truncate to match dimension
                if len(emb_array) < self.embedding_dim:
                    emb_array = np.pad(emb_array, (0, self.embedding_dim - len(emb_array)))
                else:
                    emb_array = emb_array[:self.embedding_dim]
            
            self.store[id] = {
                "embedding": emb_array,
                "metadata": metadata,
                "added_at": np.datetime64('now')
            }
        except Exception as e:
            logger.error(f"Error adding vector {id}: {e}")
            raise

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for most similar vectors using cosine similarity.
        
        Args:
            query_embedding: Query vector embedding
            top_k: Number of results to return
            
        Returns:
            List of dicts with 'id', 'score', and 'metadata' keys
        """
        if not self.store:
            logger.debug("Vector store is empty")
            return []
        
        if not query_embedding:
            logger.warning("Empty query embedding provided")
            return []
        
        try:
            q = np.array(query_embedding, dtype=np.float32)
            
            # Normalize query vector
            q_norm = np.linalg.norm(q)
            if q_norm == 0:
                logger.warning("Query vector has zero norm")
                return []
            
            q_normalized = q / q_norm
            
            results = []
            for id, v_dict in self.store.items():
                try:
                    v = v_dict["embedding"]
                    
                    # Normalize stored vector
                    v_norm = np.linalg.norm(v)
                    if v_norm == 0:
                        score = 0.0
                    else:
                        # Cosine similarity
                        score = float(np.dot(q_normalized, v / v_norm))
                    
                    results.append({
                        "id": id,
                        "score": score,
                        "metadata": v_dict["metadata"]
                    })
                except Exception as e:
                    logger.debug(f"Error computing similarity for {id}: {e}")
                    continue
            
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x["score"], reverse=True)
            
            logger.debug(f"Found {len(results)} results, returning top {top_k}")
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            return []

    def get(self, id: str) -> Optional[Dict[str, Any]]:
        """Get a vector by ID."""
        return self.store.get(id)

    def delete(self, id: str) -> bool:
        """Delete a vector by ID."""
        if id in self.store:
            del self.store[id]
            return True
        return False

    def clear(self):
        """Clear all vectors."""
        self.store.clear()
        self.embedding_dim = None

    def size(self) -> int:
        """Get number of vectors in store."""
        return len(self.store)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "num_vectors": len(self.store),
            "embedding_dim": self.embedding_dim,
            "memory_usage_mb": sum(
                v["embedding"].nbytes for v in self.store.values()
            ) / (1024 * 1024)
        }
