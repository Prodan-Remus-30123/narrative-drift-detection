"""
frame_normalization.py

Semantic latent frame normalization.

Clusters semantically similar frame labels
using embeddings.
"""

from collections import defaultdict
import numpy as np

from sklearn.cluster import AgglomerativeClustering

from embedding_model_registry import get_embedding_model
from sklearn.metrics.pairwise import cosine_similarity


DISTANCE_THRESHOLD = 0.30


def normalize_frame_labels(frame_labels):
    """
    Normalize semantically similar frame labels.

    Example:
    - International Pressure Framing
    - Global Pressure Framing

    become one canonical frame.
    """

    unique_labels = list(dict.fromkeys(frame_labels))

    if len(unique_labels) == 0:
        return {}

    if len(unique_labels) == 1:
        return {
            unique_labels[0]: unique_labels[0]
        }

    model = get_embedding_model()

    embeddings = np.array(
        model.encode_documents(
            unique_labels
        )
    )

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=DISTANCE_THRESHOLD
    )

    cluster_ids = clustering.fit_predict(
        embeddings
    )

    clusters = defaultdict(list)

    for label, cluster_id in zip(
        unique_labels,
        cluster_ids
    ):
        clusters[int(cluster_id)].append(
            label
        )

    canonical_mapping = {}

    for cluster_id, labels in clusters.items():

        canonical = choose_canonical_label(
            labels
        )

        for label in labels:
            canonical_mapping[label] = canonical

    return canonical_mapping


def choose_canonical_label(labels):
    """
    Choose label closest to
    semantic centroid.
    """

    if len(labels) == 1:
        return labels[0]

    model = get_embedding_model()

    embeddings = np.array(
        model.encode_documents(labels)
    )

    centroid = np.mean(embeddings,axis=0)

    similarities = cosine_similarity(
        embeddings,
        centroid.reshape(1, -1)
    ).flatten()

    best_index = np.argmax(
        similarities
    )

    return labels[best_index]