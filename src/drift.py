"""
Document-level narrative drift computation.

This module:
- Computes cosine-based drift
- Generates temporal drift signal
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List


def compute_cosine_drift(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine-based drift between two vectors.

    Drift = 1 - cosine_similarity

    Args:
        vec1 (np.ndarray): First embedding vector.
        vec2 (np.ndarray): Second embedding vector.

    Returns:
        float: Drift value.
    """
    similarity = cosine_similarity([vec1], [vec2])[0][0]
    return 1 - similarity


def build_drift_signal(aggregated_vectors: List[np.ndarray]) -> List[float]:
    """
    Build temporal drift signal from aggregated vectors.

    Args:
        aggregated_vectors (List[np.ndarray]): Ordered time-window vectors.

    Returns:
        List[float]: Drift values between consecutive windows.
    """
    drift_values = []
    for i in range(len(aggregated_vectors) - 1):
        drift = compute_cosine_drift(
            aggregated_vectors[i],
            aggregated_vectors[i + 1]
        )
        drift_values.append(drift)

    return drift_values