"""
entity_latent_frames.py

Entity-conditioned latent frame discovery.

Clusters a single actor's before/after verb sets (drawn from entity
framing drift, see entity_framing_drift.compute_entity_drift) into
short LLM-labeled narrative frames, so an actor's coverage can be
described as e.g. "moved from public health crisis framing to
geopolitical conflict framing" rather than just a turnover number.
"""

from collections import defaultdict
import numpy as np

from sklearn.cluster import AgglomerativeClustering

from embedding_model_registry import get_embedding_model
from llm_frame_labeler import label_latent_frame


# Agglomerative-clustering cosine-distance cutoff for grouping one
# actor's before/after verbs into latent frames (contrast with
# latent_frames.GLOBAL_VERB_CLUSTER_DISTANCE_THRESHOLD, which clusters
# a source's entire verb vocabulary).
ENTITY_VERB_CLUSTER_DISTANCE_THRESHOLD = 0.55


def cluster_verbs(verbs, entity=None, source=None, transition=None):
    verbs = list(dict.fromkeys(verbs))

    if len(verbs) == 0:
        return []

    if len(verbs) == 1:
        return [
            {
                "cluster_id": 0,
                "verbs": verbs,
                "label": label_cluster_with_llm(verbs, entity=entity, source=source, transition=transition)["label"]
            }
        ]

    model = get_embedding_model()
    embeddings = np.array(
        model.encode_documents(verbs)
    )

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=ENTITY_VERB_CLUSTER_DISTANCE_THRESHOLD
    )

    labels = clustering.fit_predict(embeddings)

    clusters = defaultdict(list)

    for verb, label in zip(verbs, labels):
        clusters[int(label)].append(verb)

    output = []

    for cluster_id, cluster_verbs in clusters.items():
        output.append(
            {
                "cluster_id": cluster_id,
                "verbs": cluster_verbs,
                "label": label_cluster_with_llm(cluster_verbs, entity=entity, source=source, transition=transition)["label"]
            }
        )

    return output


def label_cluster_with_llm(verbs, entity=None, source=None, transition=None):
    if len(verbs) == 0:
        return {
            "label": "empty frame",
            "description": "",
            "confidence": "low"
        }

    return label_latent_frame(
        verbs=verbs,
        entity=entity,
        source=source,
        transition=transition
    )


def infer_transition_label(before_clusters, after_clusters):
    before_labels = [
        cluster["label"]
        for cluster in before_clusters
    ]

    after_labels = [
        cluster["label"]
        for cluster in after_clusters
    ]

    return {
        "before_frame": " / ".join(before_labels[:3]),
        "after_frame": " / ".join(after_labels[:3]),
        "transition_summary": (
            f"Frame shifts from "
            f"{before_labels[:3]} to {after_labels[:3]}"
        )
    }


def _entity_transitions_from_framing_drift(
    framing_drift,
    entity_importance,
    salience_totals,
    entity
):
    """
    Reshape compute_entity_drift's transition-keyed output into a flat,
    single-entity list of before/after verbs plus importance/salience.
    """

    entity_key = entity.lower()
    results = []

    for transition, entities in framing_drift.items():
        for current_entity, stats in entities.items():
            if current_entity.lower() != entity_key:
                continue

            results.append({
                "transition": transition,
                "entity": current_entity,
                "vocabulary_turnover": stats.get("vocabulary_turnover"),
                "shared_similarity": stats.get("shared_similarity"),
                "drift_class": stats.get("drift_class"),
                "before_verbs": list(stats.get("before", {}).keys())[:10],
                "after_verbs": list(stats.get("after", {}).keys())[:10],
                "importance": float(entity_importance.get(current_entity, 0)),
                "salience": float(salience_totals.get(current_entity, 0))
            })

    return results


def compute_entity_latent_frame_transitions(
    source,
    entity,
    framing_drift,
    entity_importance,
    salience_totals
):
    """
    Cluster one entity's before/after verbs per transition into
    LLM-labeled latent frames.

    framing_drift/entity_importance/salience_totals are the outputs
    already computed once per source by the main pipeline
    (entity_framing_drift.compute_entity_drift/compute_entity_importance,
    actor_salience.compute_total_actor_salience) — passed in rather than
    recomputed here.
    """

    framing_results = _entity_transitions_from_framing_drift(
        framing_drift, entity_importance, salience_totals, entity
    )

    transitions = []

    for result in framing_results:
        before_verbs = result["before_verbs"]
        after_verbs = result["after_verbs"]

        before_clusters = cluster_verbs(
            before_verbs,
            entity=entity,
            source=source,
            transition=result["transition"]
        )

        after_clusters = cluster_verbs(
            after_verbs,
            entity=entity,
            source=source,
            transition=result["transition"]
        )

        transition_label = infer_transition_label(
            before_clusters,
            after_clusters
        )

        transitions.append(
            {
                "transition": result["transition"],
                "entity": result["entity"],
                "vocabulary_turnover": result["vocabulary_turnover"],
                "shared_similarity": result["shared_similarity"],
                "drift_class": result["drift_class"],
                "before_verbs": before_verbs,
                "after_verbs": after_verbs,
                "before_clusters": before_clusters,
                "after_clusters": after_clusters,
                "before_frame": transition_label["before_frame"],
                "after_frame": transition_label["after_frame"],
                "frame_transition_summary": transition_label[
                    "transition_summary"
                ],
                "importance": result["importance"],
                "salience": result["salience"]
            }
        )

    return {"source": source, "entity": entity, "latent_frame_transitions": transitions}


def _standalone_source_framing(source):
    """
    Compute framing_drift/entity_importance/salience_totals for a source
    from scratch. Only used by this module's own __main__ demo, so it
    doesn't need the caching the main pipeline already does per-run.
    """

    from database import load_full_articles
    from preprocessing import preprocess_corpus
    from temporal_entity_analysis import group_articles_by_period
    from entity_framing_drift import compute_entity_drift, compute_entity_importance
    from actor_salience import compute_actor_salience, compute_total_actor_salience

    df = load_full_articles()
    source_df = df[df["source"] == source]
    grouped = group_articles_by_period(source_df)

    for period in grouped:
        grouped[period] = preprocess_corpus(grouped[period])

    framing_drift = compute_entity_drift(grouped)
    salience_totals = compute_total_actor_salience(compute_actor_salience(grouped))
    entity_importance = compute_entity_importance(framing_drift, salience_totals)

    return framing_drift, entity_importance, salience_totals


def main():
    source = "cnn.com"
    framing_drift, entity_importance, salience_totals = _standalone_source_framing(source)

    result = compute_entity_latent_frame_transitions(
        source=source,
        entity="china",
        framing_drift=framing_drift,
        entity_importance=entity_importance,
        salience_totals=salience_totals
    )

    print("\n=== ENTITY LATENT FRAME TRANSITIONS ===")
    print("Source:", result["source"])
    print("Entity:", result["entity"])

    for item in result["latent_frame_transitions"]:

        print("\nTransition:", item["transition"])
        print(
            "Vocabulary turnover:",
            round(item["vocabulary_turnover"], 3)
        )

        print(
            "Shared similarity:",
            round(item["shared_similarity"], 3)
            if item["shared_similarity"] is not None
            else "N/A"
        )
        print("Class:", item["drift_class"])

        print("\nBefore verbs:")
        print(item["before_verbs"])

        print("\nAfter verbs:")
        print(item["after_verbs"])

        print("\nBefore latent frame:")
        print(item["before_frame"])

        print("\nAfter latent frame:")
        print(item["after_frame"])

        print("\nSummary:")
        print(item["frame_transition_summary"])


if __name__ == "__main__":
    main()
