"""
signature_comparison.py

Compares narrative signatures across sources.

Provides:
- feature vector extraction
- cosine similarity matrix
- clustering
- optional dimensionality reduction
"""

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA


SIGNATURE_FEATURES = [

    # semantic drift

    "mean_semantic_drift",
    "max_semantic_drift",
    "semantic_drift_volatility",

    # frames

    "mean_frame_volatility",
    "max_frame_volatility",
    "mean_frame_persistence",
    "mean_frame_peak_share",

    # ecosystem

    "mean_entity_drift",
    "max_entity_drift",
    "mean_entity_persistence",

    "volatile_core_actor_count",
    "stable_core_actor_count",
    "episodic_disruptor_count",
    "high_impact_shifting_actor_count",

    # migrations

    "mean_migration_ratio",
    "mean_unique_frames_visited",

    # sentiment

    "mean_sentiment",
    "sentiment_volatility",
    "mean_positive",
    "mean_negative"
]


def signature_to_vector(signature):
    """
    Converts narrative signature
    into numeric feature vector.
    """

    vector = []

    for feature in SIGNATURE_FEATURES:

        value = signature.get(
            feature,
            0.0
        )

        if value is None:
            value = 0.0

        vector.append(float(value))

    return np.array(vector)


def build_signature_matrix(signatures):
    """
    Returns:
    - sources
    - matrix
    """

    sources = []
    vectors = []

    for signature in signatures:

        source = signature["source"]

        vector = signature_to_vector(
            signature
        )

        sources.append(source)
        vectors.append(vector)

    return (
        sources,
        np.array(vectors)
    )


def compute_signature_similarity(signatures):
    """
    Computes cosine similarity
    between source signatures.
    """

    sources, matrix = build_signature_matrix(
        signatures
    )

    similarity_matrix = cosine_similarity(
        matrix
    )

    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=sources,
        columns=sources
    )

    return similarity_df


def cluster_signatures(
    signatures,
    n_clusters=2
):
    """
    Clusters outlets by
    narrative behavior.
    """

    sources, matrix = build_signature_matrix(
        signatures
    )

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters
    )

    labels = clustering.fit_predict(
        matrix
    )

    return pd.DataFrame({

        "source": sources,
        "cluster": labels

    })


def compute_signature_pca(
    signatures,
    n_components=2
):
    """
    PCA projection for
    visualization/debugging.
    """

    sources, matrix = build_signature_matrix(
        signatures
    )

    pca = PCA(
        n_components=n_components
    )

    reduced = pca.fit_transform(
        matrix
    )

    df = pd.DataFrame({

        "source": sources,
        "pc1": reduced[:, 0],
        "pc2": reduced[:, 1]

    })

    return df


def print_signature_similarity(
    similarity_df
):
    print(
        "\n=== Signature Similarity Matrix ==="
    )

    print(
        similarity_df.round(3)
    )


def print_signature_clusters(
    cluster_df
):
    print(
        "\n=== Narrative Signature Clusters ==="
    )

    print(cluster_df)


def print_signature_pca(
    pca_df
):
    print(
        "\n=== Narrative Signature PCA ==="
    )

    print(
        pca_df.round(3)
    )