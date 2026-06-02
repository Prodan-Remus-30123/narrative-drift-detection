"""
statistical_validation.py

Statistical validation layer for narrative signatures.

Runs fast from analysis_results.json.
Validates:
1. Pairwise signature similarity
2. Permutation-based similarity significance
3. Feature jackknife robustness
4. Noise robustness
5. Cluster separability
"""

import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering

from narrative_signatures import build_all_narrative_signatures
from signature_comparison import SIGNATURE_FEATURES
from semantic_signature_embedding import build_semantic_signature_embeddings


def load_analysis_results(path="analysis_results.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def signature_to_numeric_vector(signature):
    values = []

    for feature in SIGNATURE_FEATURES:
        value = signature.get(feature, 0.0)

        if value is None:
            value = 0.0

        values.append(float(value))

    return np.array(values, dtype=float)


def build_validation_matrix(
    signatures,
    include_semantic=True,
    semantic_weight=1.0
):
    semantic_embeddings = build_semantic_signature_embeddings(
        signatures
    )

    sources = []
    numeric_vectors = []
    semantic_vectors = []

    for signature in signatures:
        source = signature["source"]

        numeric_vector = signature_to_numeric_vector(
            signature
        )

        semantic_vector = semantic_embeddings.get(
            source
        )

        if semantic_vector is None:
            semantic_vector = np.zeros(384)

        sources.append(source)
        numeric_vectors.append(numeric_vector)
        semantic_vectors.append(semantic_vector)

    numeric_matrix = np.array(numeric_vectors)
    semantic_matrix = np.array(semantic_vectors)

    numeric_matrix = StandardScaler().fit_transform(
        numeric_matrix
    )

    semantic_matrix = normalize(
        semantic_matrix,
        norm="l2"
    )

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


def compute_similarity_matrix(sources, matrix):
    similarity = cosine_similarity(matrix)

    return pd.DataFrame(
        similarity,
        index=sources,
        columns=sources
    )


def permutation_similarity_test(
    sources,
    matrix,
    n_iterations=1000,
    random_state=42
):
    rng = np.random.default_rng(random_state)

    observed_similarity = cosine_similarity(matrix)

    pair_results = []

    n_sources = len(sources)

    null_distributions = {
        (i, j): []
        for i in range(n_sources)
        for j in range(i + 1, n_sources)
    }

    for _ in range(n_iterations):
        permuted = matrix.copy()

        for feature_index in range(permuted.shape[1]):
            permuted[:, feature_index] = rng.permutation(
                permuted[:, feature_index]
            )

        null_similarity = cosine_similarity(permuted)

        for i in range(n_sources):
            for j in range(i + 1, n_sources):
                null_distributions[(i, j)].append(
                    null_similarity[i, j]
                )

    for i in range(n_sources):
        for j in range(i + 1, n_sources):
            observed = observed_similarity[i, j]
            null_values = np.array(
                null_distributions[(i, j)]
            )

            p_more_similar = float(
                np.mean(null_values >= observed)
            )

            p_more_different = float(
                np.mean(null_values <= observed)
            )

            pair_results.append({
                "source_a": sources[i],
                "source_b": sources[j],
                "observed_similarity": observed,
                "null_mean": float(np.mean(null_values)),
                "null_std": float(np.std(null_values)),
                "p_more_similar": p_more_similar,
                "p_more_different": p_more_different
            })

    return pd.DataFrame(pair_results)


def jackknife_feature_robustness(
    sources,
    matrix
):
    baseline_similarity = cosine_similarity(matrix)

    results = []

    for removed_index in range(matrix.shape[1]):
        reduced_matrix = np.delete(
            matrix,
            removed_index,
            axis=1
        )

        reduced_similarity = cosine_similarity(
            reduced_matrix
        )

        delta = np.abs(
            baseline_similarity - reduced_similarity
        )

        upper_triangle = delta[
            np.triu_indices_from(delta, k=1)
        ]

        results.append({
            "removed_feature_index": removed_index,
            "mean_similarity_change": float(
                np.mean(upper_triangle)
            ),
            "max_similarity_change": float(
                np.max(upper_triangle)
            )
        })

    return pd.DataFrame(results)


def noise_robustness_test(
    sources,
    matrix,
    noise_levels=(0.01, 0.05, 0.10, 0.20),
    n_iterations=300,
    random_state=42
):
    rng = np.random.default_rng(random_state)

    baseline_similarity = cosine_similarity(matrix)

    results = []

    for noise_level in noise_levels:
        changes = []

        for _ in range(n_iterations):
            noise = rng.normal(
                loc=0.0,
                scale=noise_level,
                size=matrix.shape
            )

            noisy_matrix = matrix + noise
            noisy_similarity = cosine_similarity(
                noisy_matrix
            )

            delta = np.abs(
                baseline_similarity - noisy_similarity
            )

            upper_triangle = delta[
                np.triu_indices_from(delta, k=1)
            ]

            changes.append(
                np.mean(upper_triangle)
            )

        results.append({
            "noise_level": noise_level,
            "mean_similarity_change": float(
                np.mean(changes)
            ),
            "std_similarity_change": float(
                np.std(changes)
            )
        })

    return pd.DataFrame(results)


def cluster_separability(
    sources,
    matrix,
    n_clusters=2
):
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters
    )

    labels = clustering.fit_predict(matrix)

    if len(set(labels)) < 2:
        silhouette = None
    else:
        silhouette = float(
            silhouette_score(matrix, labels)
        )

    return pd.DataFrame({
        "source": sources,
        "cluster": labels
    }), silhouette


def run_statistical_validation(
    analysis_path="analysis_results.json",
    semantic_weight=1.0
):
    analysis_results = load_analysis_results(
        analysis_path
    )

    signatures = build_all_narrative_signatures(
        analysis_results
    )

    sources, matrix = build_validation_matrix(
        signatures,
        include_semantic=True,
        semantic_weight=semantic_weight
    )

    print("\n=== STATISTICAL VALIDATION ===")
    print(f"Sources: {sources}")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Semantic weight: {semantic_weight}")

    similarity_df = compute_similarity_matrix(
        sources,
        matrix
    )

    print("\n=== Normalized Signature Similarity ===")
    print(similarity_df.round(3))

    permutation_df = permutation_similarity_test(
        sources,
        matrix,
        n_iterations=1000
    )

    print("\n=== Permutation Similarity Test ===")
    print(
        permutation_df.round(4)
    )

    jackknife_df = jackknife_feature_robustness(
        sources,
        matrix
    )

    print("\n=== Jackknife Feature Robustness ===")
    print(
        jackknife_df.sort_values(
            "max_similarity_change",
            ascending=False
        ).head(10).round(4)
    )

    noise_df = noise_robustness_test(
        sources,
        matrix
    )

    print("\n=== Noise Robustness ===")
    print(
        noise_df.round(4)
    )

    cluster_df, silhouette = cluster_separability(
        sources,
        matrix,
        n_clusters=2
    )

    print("\n=== Cluster Separability ===")
    print(cluster_df)

    print("\nSilhouette score:")
    print(silhouette)

    return {
        "similarity": similarity_df,
        "permutation": permutation_df,
        "jackknife": jackknife_df,
        "noise": noise_df,
        "clusters": cluster_df,
        "silhouette": silhouette
    }


if __name__ == "__main__":
    run_statistical_validation(
        analysis_path="analysis_results.json",
        semantic_weight=1.0
    )