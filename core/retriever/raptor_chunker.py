"""
RAPTOR-style helper: select representative chunks for global-intent queries.
This uses a simple MMR (Maximal Marginal Relevance) selection over chunk embeddings
to pick a compact, diverse set of chunks that cover the document. The selected
chunks are returned in original order to preserve local context for synthesis.
"""
from typing import List, Dict, Any
import numpy as np


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def select_representative_chunks(chunks: List[Dict[str, Any]], top_k: int = 8, lambda_mm: float = 0.7) -> List[Dict[str, Any]]:
    """
    Select representative chunks using MMR.

    Args:
        chunks: list of chunk dicts (must include 'embedding' and 'text')
        top_k: maximum number of chunks to select
        lambda_mm: trade-off parameter for MMR (0..1), higher -> relevance, lower -> diversity

    Returns:
        list of selected chunk dicts (preserving original order)
    """
    if not chunks:
        return []

    # Ensure embeddings are numpy arrays
    embs = []
    for c in chunks:
        emb = c.get("embedding")
        if isinstance(emb, list):
            emb = np.array(emb)
        embs.append(emb if emb is not None else np.zeros((1,)))

    embs = np.stack([e if e is not None else np.zeros_like(embs[0]) for e in embs])

    # Query representation for Raptor: use mean of embeddings to represent document intent
    doc_emb = np.mean(embs, axis=0)

    n = len(chunks)
    k = min(top_k, n)
    if k <= 0:
        return []

    # Precompute similarities
    sim_to_doc = np.array([_cosine_sim(doc_emb, embs[i]) for i in range(n)])
    sim_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = _cosine_sim(embs[i], embs[j])

    selected = []
    selected_idx = []

    # Initialize with the most relevant chunk to document mean
    first = int(np.argmax(sim_to_doc))
    selected_idx.append(first)

    # Greedy MMR selection
    while len(selected_idx) < k:
        candidates = [i for i in range(n) if i not in selected_idx]
        if not candidates:
            break
        mmr_scores = []
        for c in candidates:
            sim_doc = sim_to_doc[c]
            sim_selected = 0.0
            if selected_idx:
                sim_selected = max(sim_matrix[c, s] for s in selected_idx)
            mmr_score = lambda_mm * sim_doc - (1 - lambda_mm) * sim_selected
            mmr_scores.append((c, mmr_score))

        # pick candidate with highest MMR score
        next_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected_idx.append(next_idx)

    # Sort selected indices to preserve document order
    selected_idx_sorted = sorted(selected_idx)
    selected_chunks = [chunks[i] for i in selected_idx_sorted]
    return selected_chunks
