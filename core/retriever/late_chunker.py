"""
Late-chunking & parent-doc retrieval helper.
Basic idea:
- Store coarse parent-doc embeddings (one vector per doc)
- When a query arrives, find relevant parent docs, then perform chunk-level retrieval within top-k parent docs
This file contains improved algorithmic implementations with cosine similarity.
"""
from typing import List, Dict, Any
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    Returns similarity score between -1 and 1.
    """
    # Normalize vectors
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    
    if a_norm == 0 or b_norm == 0:
        return 0.0
    
    return float(np.dot(a, b) / (a_norm * b_norm))


def get_top_parent_docs(query_embedding: List[float], parent_vectors: List[tuple], top_k: int = 3) -> List[str]:
    """
    Find top-k parent documents using cosine similarity.
    
    Args:
        query_embedding: Query vector embedding
        parent_vectors: List of (doc_id, vector) tuples
        top_k: Number of top documents to return
        
    Returns:
        List of top-k doc_ids sorted by relevance score
    """
    if not parent_vectors:
        return []
    
    try:
        q = np.array(query_embedding)
        scores = []
        
        for doc_id, vector in parent_vectors:
            try:
                v = np.array(vector)
                sim = cosine_similarity(q, v)
                scores.append((doc_id, sim))
            except (ValueError, TypeError):
                # Skip invalid vectors
                continue
        
        # Sort by similarity score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k doc_ids
        return [doc_id for doc_id, _ in scores[:top_k]]
    except Exception as e:
        print(f"Error in get_top_parent_docs: {e}")
        # Fallback: return first k docs
        return [doc_id for doc_id, _ in parent_vectors[:top_k]]


def assemble_late_chunks(
    parent_doc_ids: List[str], 
    chunk_index: Dict[str, List[Dict[str, Any]]], 
    max_chunks: int = 8
) -> List[Dict[str, Any]]:
    """
    Assemble chunks from selected parent documents.
    
    Args:
        parent_doc_ids: List of parent document IDs
        chunk_index: Mapping of doc_id to list of chunk dicts
        max_chunks: Maximum number of chunks to return. If None, return all chunks.
        
    Returns:
        List of chunk dicts with metadata
    """
    # Default to 8 if not specified; treat None as unlimited
    if max_chunks is None:
        max_chunks = 999999
    
    selected = []
    
    for doc_id in parent_doc_ids:
        if doc_id not in chunk_index:
            continue
        
        chunks = chunk_index[doc_id]
        # Add chunks from this parent doc
        remaining = max_chunks - len(selected)
        chunks_to_add = chunks[:remaining] if remaining > 0 else []
        selected.extend(chunks_to_add)
        
        # Stop if we've gathered enough chunks
        if len(selected) >= max_chunks:
            break
    
    return selected[:max_chunks]
