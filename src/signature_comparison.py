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
from semantic_signature_embedding import (
    build_semantic_signature_embeddings
)
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import pairwise_distances

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

    "mean_entity_turnover",
    "max_entity_drift",
    "mean_entity_persistence",

    "volatile_core_actor_ratio",
    "stable_core_actor_ratio",
    "episodic_disruptor_ratio",
    "high_impact_shifting_actor_ratio",

    # migrations

    "mean_migration_ratio",
    "mean_unique_frames_visited",

    # sentiment

    "mean_sentiment",
    "sentiment_volatility",
    "mean_positive",
    "mean_negative",
    "sentiment_shift",

    "mean_narrative_intensity",
    "intensity_volatility",
    "intensity_shift",

    "mean_polarization",
    "polarization_volatility",
    "polarization_shift",

    "emotional_escalation"
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


def build_signature_matrix(
    signatures,
    include_semantic=True,
    semantic_weight=1.0,
    debug=False
):
    semantic_embeddings = (build_semantic_signature_embeddings(signatures) if include_semantic else {})

    sources = []
    numeric_vectors = []
    semantic_vectors = []

    for signature in signatures:
        source = signature["source"]

        numeric_vector = signature_to_vector(signature)
        semantic_vector = semantic_embeddings.get(source)

        if semantic_vector is None:
            semantic_vector = np.zeros(384)

        sources.append(source)
        numeric_vectors.append(numeric_vector)
        semantic_vectors.append(semantic_vector)

        if debug:
            print(source)
            print("numeric norm:", np.linalg.norm(numeric_vector))
            print("semantic norm:", np.linalg.norm(semantic_vector))

    numeric_matrix = np.array(numeric_vectors)
    semantic_matrix = np.array(semantic_vectors)

    numeric_matrix = StandardScaler().fit_transform(numeric_matrix)
    semantic_matrix = normalize(semantic_matrix)

    if include_semantic:
        matrix = np.concatenate(
            [
                numeric_matrix,
                semantic_matrix * semantic_weight
            ],
            axis=1
        )
    else:
        matrix = numeric_matrix

    return sources, matrix


def compute_signature_similarity(signatures, semantic_weight=0.0):
    sources, matrix = build_signature_matrix(
        signatures,
        include_semantic=semantic_weight > 0,
        semantic_weight=semantic_weight,
    )

    distances = pairwise_distances(matrix, metric="euclidean")
    similarity = 1 / (1 + distances)

    return similarity, sources


def compute_semantic_signature_similarity(signatures):
    sources, matrix = build_signature_matrix(
        signatures,
        include_semantic=True,
        semantic_weight=1.0
    )

    return cosine_similarity(matrix), sources

def cluster_signatures(
    signatures,
    n_clusters=2,
    include_semantic=True,
    semantic_weight=1.0
):
    sources, matrix = build_signature_matrix(
        signatures,
        include_semantic=include_semantic,
        semantic_weight=semantic_weight
    )

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters
    )

    labels = clustering.fit_predict(matrix)

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


