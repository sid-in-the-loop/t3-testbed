"""
Diversity utilities for query selection.

Provides:
- Jaccard similarity/distance
- Dense embeddings (sentence-transformers/all-MiniLM-L6-v2) cosine distance
- Greedy max-min diverse subset selection
"""

import re
from typing import List, Set
import numpy as np


def tokenize(text: str) -> Set[str]:
    """Simple tokenization: lowercase, remove punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return set(text.split())


def jaccard_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity between two texts."""
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)
    if not tokens1 and not tokens2:
        return 1.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union if union > 0 else 0.0


def jaccard_distance(text1: str, text2: str) -> float:
    """Compute Jaccard distance (1 - similarity)."""
    return 1.0 - jaccard_similarity(text1, text2)


def compute_jaccard_distance_matrix(queries: List[str]) -> np.ndarray:
    """Compute pairwise Jaccard distance matrix."""
    n = len(queries)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i, j] = jaccard_distance(queries[i], queries[j])
    return matrix


# Lazy singleton for MiniLM (loaded once per process)
_minilm_model = None


def _get_minilm():
    """Return the MiniLM embedding model; load on first use."""
    global _minilm_model
    if _minilm_model is None:
        from sentence_transformers import SentenceTransformer
        _minilm_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _minilm_model


def compute_dense_distance_matrix(queries: List[str]) -> np.ndarray:
    """
    Compute pairwise distance matrix using dense embeddings (all-MiniLM-L6-v2).
    Distance = 1 - cosine_similarity(embeddings).
    """
    model = _get_minilm()
    embeddings = model.encode(queries, convert_to_numpy=True)
    # Cosine similarity between rows
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_norm = embeddings / norms
    sim = np.dot(emb_norm, emb_norm.T).astype(np.float64)
    np.clip(sim, -1.0, 1.0, out=sim)
    return 1.0 - sim


def greedy_diversity_selection(
    queries: List[str], k: int, method: str = "jaccard", seed: int = 42
) -> List[int]:
    """
    Greedy max-min diverse subset selection.

    At each step, pick the candidate whose minimum distance to the already-selected
    set is largest (i.e. most diverse from selected).

    Args:
        queries: List of query strings
        k: Number of queries to select
        method: "jaccard" or "dense"
        seed: Random seed

    Returns:
        List of indices of selected queries
    """
    if k >= len(queries):
        return list(range(len(queries)))

    np.random.seed(seed)

    if method == "dense":
        distance_matrix = compute_dense_distance_matrix(queries)
    else:
        distance_matrix = compute_jaccard_distance_matrix(queries)

    selected = []
    remaining = set(range(len(queries)))

    first_idx = np.random.choice(list(remaining))
    selected.append(first_idx)
    remaining.remove(first_idx)

    while len(selected) < k and remaining:
        best_score = -1
        best_idx = None

        for candidate_idx in remaining:
            min_diversity = min(
                distance_matrix[candidate_idx, selected_idx]
                for selected_idx in selected
            )
            if min_diversity > best_score:
                best_score = min_diversity
                best_idx = candidate_idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)
        else:
            best_idx = np.random.choice(list(remaining))
            selected.append(best_idx)
            remaining.remove(best_idx)

    return selected


def select_diverse_queries(
    queries: List[str], k: int, method: str = "jaccard", seed: int = 42
) -> List[str]:
    """
    Select k diverse queries from a larger set.

    Args:
        queries: List of all candidate queries
        k: Number to select
        method: "jaccard", "dense", or "random"
        seed: Random seed

    Returns:
        List of selected queries
    """
    if method == "random":
        np.random.seed(seed)
        indices = np.random.choice(len(queries), size=min(k, len(queries)), replace=False)
        return [queries[i] for i in indices]
    elif method in ["jaccard", "dense"]:
        indices = greedy_diversity_selection(queries, k, method, seed)
        return [queries[i] for i in indices]
    else:
        raise ValueError(f"Unknown selection method: {method}")
