"""
Diversity utilities for T³ query selection.

Provides functions for:
- Jaccard similarity/distance between text queries
- Greedy DPP approximation for diverse subset selection
- Query diversity scoring
"""

import re
from typing import List, Set
from collections import defaultdict
import numpy as np


def tokenize(text: str) -> Set[str]:
    """Simple tokenization: lowercase, remove punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return set(text.split())


def jaccard_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity between two texts."""
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)

    if not tokens1 and not tokens2:
        return 1.0  # Both empty

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)

    return intersection / union if union > 0 else 0.0


def jaccard_distance(text1: str, text2: str) -> float:
    """Compute Jaccard distance (1 - similarity)."""
    return 1.0 - jaccard_similarity(text1, text2)


def compute_diversity_matrix(queries: List[str]) -> np.ndarray:
    """
    Compute pairwise diversity matrix using Jaccard distance.
    Higher values = more diverse (less similar).
    """
    n = len(queries)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 0.0  # Self-similarity = 0
            else:
                matrix[i, j] = jaccard_distance(queries[i], queries[j])

    return matrix


def greedy_diversity_selection(queries: List[str], k: int, seed: int = 42) -> List[int]:
    """
    Greedy approximation of DPP for diverse subset selection.

    Args:
        queries: List of query strings
        k: Number of queries to select
        seed: Random seed for reproducibility

    Returns:
        List of indices of selected queries (in selection order)
    """
    if k >= len(queries):
        return list(range(len(queries)))

    np.random.seed(seed)
    diversity_matrix = compute_diversity_matrix(queries)

    selected = []
    remaining = set(range(len(queries)))

    # Start with random seed
    first_idx = np.random.choice(list(remaining))
    selected.append(first_idx)
    remaining.remove(first_idx)

    while len(selected) < k and remaining:
        # Score each remaining query by its minimum diversity to already selected
        best_score = -1
        best_idx = None

        for candidate_idx in remaining:
            min_diversity = min(diversity_matrix[candidate_idx, selected_idx]
                              for selected_idx in selected)
            if min_diversity > best_score:
                best_score = min_diversity
                best_idx = candidate_idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)
        else:
            # Fallback: pick randomly if all remaining have same min diversity
            best_idx = np.random.choice(list(remaining))
            selected.append(best_idx)
            remaining.remove(best_idx)

    return selected


def select_diverse_queries(queries: List[str], k: int, method: str = "jaccard",
                          seed: int = 42, question: str = None, lambda_val: float = 0.5) -> List[str]:
    """
    Select k diverse queries from a larger set.

    Args:
        queries: List of all available queries
        k: Number to select
        method: Selection method ("jaccard"/"dpp", "mmr", "random")
        seed: Random seed
        question: Original question (required for MMR)
        lambda_val: Relevance vs diversity weight for MMR (default 0.5)

    Returns:
        List of selected queries (in selection order)
    """
    if method == "random":
        np.random.seed(seed)
        indices = np.random.choice(len(queries), size=min(k, len(queries)),
                                 replace=False)
        return [queries[i] for i in indices]
    elif method in ["jaccard", "dpp"]:
        indices = greedy_diversity_selection(queries, k, seed)
        return [queries[i] for i in indices]
    elif method == "mmr":
        if question is None:
            raise ValueError("Question must be provided for MMR selection.")
        indices = mmr_selection(queries, question, k, lambda_val, seed)
        return [queries[i] for i in indices]
    else:
        raise ValueError(f"Unknown selection method: {method}")


def mmr_selection(queries: List[str], question: str, k: int, lambda_val: float = 0.5, seed: int = 42) -> List[int]:
    """
    Maximal Marginal Relevance (MMR) selection.
    Score(q) = lambda * Sim(q, Question) - (1 - lambda) * max_{s in Selected} Sim(q, s)
    """
    if k >= len(queries):
        return list(range(len(queries)))

    np.random.seed(seed)
    
    # Precompute relevance scores: Sim(q, Question)
    relevance_scores = np.array([jaccard_similarity(q, question) for q in queries])
    
    # Precompute pairwise similarity matrix: Sim(q, s)
    n = len(queries)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = jaccard_similarity(queries[i], queries[j])

    selected = []
    remaining = set(range(len(queries)))

    # Start with the most relevant query
    first_idx = np.argmax(relevance_scores)
    selected.append(first_idx)
    remaining.remove(first_idx)

    while len(selected) < k and remaining:
        best_score = -float('inf')
        best_idx = None

        for candidate_idx in remaining:
            # Diversity term: max similarity to any already selected query
            max_sim_to_selected = max(sim_matrix[candidate_idx, s_idx] for s_idx in selected)
            
            # MMR Score
            score = lambda_val * relevance_scores[candidate_idx] - (1 - lambda_val) * max_sim_to_selected
            
            if score > best_score:
                best_score = score
                best_idx = candidate_idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)
        else:
            break

    return selected


def generate_naive_seeds(question: str, k: int) -> List[str]:
    """
    Generate naive diversity seeds (round-robin from fixed pool).
    This is the baseline from T³ Fixed.
    """
    # Simplified version of the SEED_POOL from t3_fixed.py
    seed_pool = [
        "Search for direct facts about the main topic.",
        "Search for related people or organizations.",
        "Search for historical or temporal information.",
        "Search for geographic locations involved.",
        "Search for numbers, statistics, or measurements.",
        "Search for official sources or authorities.",
        "Search for recent developments or updates.",
        "Search for primary sources or documentation.",
    ]

    return [seed_pool[i % len(seed_pool)] for i in range(k)]


def generate_anchor_seeds(question: str, k: int) -> List[str]:
    """
    Generate anchor-based seeds: focus on different meaningful aspects of the question.
    """
    # Try to find specific entities (capitalized words in original question)
    entities = list(dict.fromkeys(re.findall(r'\b[A-Z][a-z]+\b', question))) # unique preserve order
    
    if not entities:
        # Fallback to tokens
        clean_q = re.sub(r'[^\w\s]', '', question).lower()
        entities = [t for t in clean_q.split() if len(t) > 3][:3] or ["topic"]

    seeds = []
    # 1. Main entity focus
    seeds.append(f"Primary focus on '{entities[0]}'")
    
    # 2. Secondary entity or action focus
    if len(entities) > 1:
        seeds.append(f"Focus on the relationship between '{entities[0]}' and '{entities[1]}'")
    else:
        seeds.append(f"Contextual background for '{entities[0]}'")

    # 3. Numeric/Constraint focus
    seeds.append("Specific focus on numeric values, dates, and quantitative constraints")
    
    # 4. Official/Wiki focus
    seeds.append("Targeted search for official sources or encyclopedic entries")

    # Fill remaining slots with round-robin from SEED_POOL
    from .t3_fixed import SEED_POOL
    while len(seeds) < k:
        seeds.append(SEED_POOL[len(seeds) % len(SEED_POOL)])

    return seeds[:k]
