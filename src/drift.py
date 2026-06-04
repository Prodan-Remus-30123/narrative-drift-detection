"""
Document-level narrative drift computation.

This module:
- Computes cosine-based drift
- Generates temporal drift signal
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List


def compute_cosine_drift(vector_a, vector_b):
    similarity = np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))

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
        drift = compute_cosine_drift(aggregated_vectors[i], aggregated_vectors[i + 1])
        drift_values.append(drift)

    return drift_values

def compute_dynamic_threshold(drift_values, method="mean_std"):
    if len(drift_values) < 2:
        return None

    values = np.array(drift_values)

    if method == "mean_std":
        return values.mean() + values.std()

    if method == "median_mad":
        median = np.median(values)
        mad = np.median(np.abs(values - median))

        return median + 1.4826 * mad

    raise ValueError(f"Unknown threshold method: {method}")

def classify_drift(drift_value, threshold):
    if threshold is None:
        return "insufficient_data"

    if drift_value > threshold:
        return "significant"

    return "minor"