"""
latent_frames.py

Latent narrative frame discovery.

Discovers verb clusters using embeddings,
labels them with a placeholder label,
and tracks frame evolution over periods.
"""

from collections import defaultdict, Counter
import numpy as np

from sklearn.cluster import AgglomerativeClustering

from temporal_entity_analysis import group_articles_by_period
from entities import analyze_entities
from preprocessing import preprocess_corpus
from database import load_full_articles
import math
from embedding_model_registry import get_embedding_model
from filters.verb_filters import GENERIC_VERBS, SEMANTICALLY_WEAK_VERBS



MIN_VERB_COUNT = 3
# Agglomerative-clustering cosine-distance cutoff for grouping the
# source's *global* verb vocabulary into latent frames (contrast with
# entity_latent_frames.ENTITY_VERB_CLUSTER_DISTANCE_THRESHOLD, which
# clusters a single actor's before/after verbs).
GLOBAL_VERB_CLUSTER_DISTANCE_THRESHOLD = 0.6
MIN_NARRATIVE_CENTRALITY = 2.5
MIN_FRAME_CLUSTER_SIZE = 3

def collect_verbs_by_period(grouped_texts):
    period_verbs = {}

    for period, texts in grouped_texts.items():
        entity_data = analyze_entities(texts)

        verb_counts = Counter()

        for entity, stats in entity_data.items():
            for verb, count in stats["verbs"].items():
                verb_counts[verb] += count

        period_verbs[period] = dict(verb_counts)

    return period_verbs


def collect_global_verbs(period_verbs):
    global_counts = Counter()

    for verbs in period_verbs.values():
        for verb, count in verbs.items():
            global_counts[verb] += count

    return dict(global_counts)

def compute_narrative_specificity(
    period_verbs
):
    """
    Compute TF-IDF-like narrative specificity.

    High specificity:
    - frequent locally
    - rare globally

    Low specificity:
    - generic verbs appearing everywhere
    """

    verb_document_frequency = Counter()

    total_periods = len(
        period_verbs
    )

    # Count in how many periods each verb appears

    for period, verbs in period_verbs.items():

        unique_verbs = set(
            verbs.keys()
        )

        for verb in unique_verbs:

            verb_document_frequency[
                verb
            ] += 1

    specificity_scores = {}

    # Compute specificity

    for period, verbs in period_verbs.items():

        for verb, tf in verbs.items():

            df = verb_document_frequency[
                verb
            ]

            idf = math.log(
                (1 + total_periods)
                /
                (1 + df)
            ) + 1

            specificity = tf * idf

            if verb not in specificity_scores:

                specificity_scores[
                    verb
                ] = []

            specificity_scores[
                verb
            ].append(
                specificity
            )

    averaged_specificity = {}

    for verb, values in specificity_scores.items():

        averaged_specificity[
            verb
        ] = sum(values) / len(values)

    return averaged_specificity

def compute_temporal_variance(period_verbs):
    """
    Measures how unevenly a verb appears across periods.
    Higher = more temporally distinctive.
    """

    all_verbs = set()

    for verbs in period_verbs.values():
        all_verbs.update(verbs.keys())

    temporal_scores = {}

    periods = list(period_verbs.keys())

    for verb in all_verbs:

        values = []

        for period in periods:
            values.append(
                period_verbs[period].get(verb, 0)
            )

        mean_value = np.mean(values)

        if mean_value == 0:
            temporal_scores[verb] = 0
        else:
            temporal_scores[verb] = float(
                np.std(values) / mean_value
            )

    return temporal_scores

def compute_narrative_idf(period_verbs):
    """
    Narrative IDF.

    Penalizes verbs that appear in many periods.
    Rewards temporally distinctive narrative verbs.
    """

    total_periods = len(period_verbs)

    document_frequency = Counter()

    for period, verbs in period_verbs.items():

        unique_verbs = set(
            verbs.keys()
        )

        for verb in unique_verbs:

            document_frequency[
                verb
            ] += 1

    narrative_idf = {}

    for verb, df in document_frequency.items():

        narrative_idf[verb] = math.log(
            total_periods / (1 + df)
        ) + 1

    return narrative_idf

def compute_narrative_centrality(
    specificity_scores,
    temporal_scores,
    narrative_idf
):
    """
    Combines:
    - narrative specificity
    - temporal movement
    - narrative rarity
    """

    centrality = {}

    ALL_FILTERED_VERBS = (
        GENERIC_VERBS |
        SEMANTICALLY_WEAK_VERBS
    )

    for verb, specificity in specificity_scores.items():

        temporal = temporal_scores.get(
            verb,
            0
        )

        idf = narrative_idf.get(
            verb,
            1
        )

        penalty = 1.0

        if verb in SEMANTICALLY_WEAK_VERBS:
            penalty = 0.7

        if verb in GENERIC_VERBS:
            penalty = 0.4

        centrality[verb] = (
            specificity *
            (1 + temporal) *
            idf *
            penalty
        )

    return centrality


def embed_verbs(verbs):
    model = get_embedding_model()

    verb_list = list(verbs)

    embeddings = model.encode_documents(verb_list)

    return verb_list, np.array(embeddings)


def cluster_verbs(verb_list, embeddings):
    if len(verb_list) < 2:
        return {}

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=GLOBAL_VERB_CLUSTER_DISTANCE_THRESHOLD
    )

    labels = clustering.fit_predict(embeddings)

    clusters = defaultdict(list)

    for verb, label in zip(verb_list, labels):
        clusters[int(label)].append(verb)

    return dict(clusters)


def compute_cluster_centroids(clusters, verb_list, embeddings):
    verb_to_embedding = {verb: embeddings[i] for i, verb in enumerate(verb_list)}

    centroids = {}

    for cluster_id, verbs in clusters.items():
        vectors = [verb_to_embedding[verb] for verb in verbs]
        centroids[cluster_id] = np.mean(vectors, axis=0)

    return centroids


def assign_period_verbs_to_clusters(period_verbs, clusters, centrality_scores):
    verb_to_cluster = {}

    for cluster_id, verbs in clusters.items():
        for verb in verbs:
            verb_to_cluster[verb] = cluster_id

    period_frames = {}

    for period, verbs in period_verbs.items():
        frame_counts = Counter()

        for verb, count in verbs.items():
            if verb in verb_to_cluster:
                weighted_count = (count * centrality_scores.get(verb, 1))
                frame_counts[verb_to_cluster[verb]] += weighted_count

        total = sum(frame_counts.values())

        period_frames[period] = {
            cluster_id: {
                "count": count,
                "share": count / total if total > 0 else 0
            }
            for cluster_id, count in frame_counts.items()
        }

    return period_frames


def label_cluster_placeholder(verbs):
    """
    Temporary label until LLM labeling is added.
    """

    top_verbs = verbs[:5]

    return " / ".join(top_verbs)


def label_clusters(clusters):
    labels = {}

    for cluster_id, verbs in clusters.items():
        if len(verbs) < MIN_FRAME_CLUSTER_SIZE:
            continue
        labels[cluster_id] = {
            "label": label_cluster_placeholder(verbs),
            "verbs": verbs
        }

    return labels


def compute_frame_evolution_from_grouped(source,grouped):
    period_verbs = collect_verbs_by_period(grouped)
    specificity_scores = compute_narrative_specificity(period_verbs)
    temporal_scores = compute_temporal_variance(period_verbs)
    narrative_idf = compute_narrative_idf(
    period_verbs
)
    centrality_scores = compute_narrative_centrality(specificity_scores, temporal_scores, narrative_idf)

    filtered_verbs = {
        verb: score
        for verb, score in centrality_scores.items()
        if score >= MIN_NARRATIVE_CENTRALITY
    }

    # print("\n=== NARRATIVE CENTRALITY FILTERING ===")
    # print("\nTop narrative-central verbs:")

    # for verb, score in ranked_centrality[:50]:

    #     print(
    #         f"{verb}: {score:.2f}"
    #     )

    print(
        f"\nSelected verbs: {len(filtered_verbs)} "
        f"out of {len(centrality_scores)}"
    )

    verb_list, embeddings = embed_verbs(filtered_verbs.keys())
    clusters = cluster_verbs(verb_list, embeddings)
    cluster_labels = label_clusters(clusters)
    period_frames = assign_period_verbs_to_clusters(period_verbs, clusters, centrality_scores)

    return {
        "source": source,
        "num_verbs": len(verb_list),
        "num_frames": len(clusters),
        "clusters": cluster_labels,
        "period_frames": period_frames
    }


def main():
    df = load_full_articles()
    source = "cnn.com"
    source_df = df[df["source"] == source]

    grouped = group_articles_by_period(source_df)

    for period in grouped:
        grouped[period] = preprocess_corpus(grouped[period])

    result = compute_frame_evolution_from_grouped(
        source,
        grouped
    )

    print("\n=== LATENT FRAME DISCOVERY ===")
    print("Source:", result["source"])
    print("Number of verbs:", result["num_verbs"])
    print("Number of frames:", result["num_frames"])


if __name__ == "__main__":
    main()