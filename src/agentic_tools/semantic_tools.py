"""
semantic_tools.py

Agentic semantic drift tools.
"""

from embeddings import EmbeddingModel
from drift import (
    compute_cosine_drift,
    compute_dynamic_threshold,
    classify_drift
)

from utils.period_sorting import sort_period_key
from agentic_tools.context_registry import get_context
from embedding_model_registry import get_embedding_model


def get_semantic_drift(source):

    context = get_context(source)

    if context.semantic_drift is not None:
        return context.semantic_drift

    grouped = context.get_preprocessed_grouped()

    model = get_embedding_model()

    aggregated_vectors = []

    for period in sorted(
        grouped.keys(),
        key=sort_period_key
    ):

        embeddings = model.encode_documents(
            grouped[period]
        )

        vector = model.aggregate_embeddings(
            embeddings
        )

        aggregated_vectors.append(
            (period, vector)
        )

    results = []
    drift_values = []

    for i in range(
        len(aggregated_vectors) - 1
    ):

        p1, v1 = aggregated_vectors[i]
        p2, v2 = aggregated_vectors[i + 1]

        drift = compute_cosine_drift(
            v1,
            v2
        )

        drift_values.append(
            drift
        )

        results.append({
            "transition": f"{p1}->{p2}",
            "drift": float(drift)
        })

    threshold = compute_dynamic_threshold(
        drift_values,
        method="median_mad"
    )

    for item in results:

        item["threshold"] = float(
            threshold
        )

        item["classification"] = classify_drift(
            item["drift"],
            threshold
        )

    context.semantic_drift = {
        "source": source,
        "semantic_drift": results
    }

    return context.semantic_drift