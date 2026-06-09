"""
entity_latent_frames.py

Entity-conditioned latent frame discovery.

Uses entity framing drift output and clusters
before/after verbs for each actor transition.
"""

from collections import defaultdict
import numpy as np

from sklearn.cluster import AgglomerativeClustering

from embeddings import EmbeddingModel
from agentic_tools.framing_tools import get_entity_framing
from llm_frame_labeler import label_latent_frame
from embedding_model_registry import get_embedding_model
from agentic_tools.context_registry import get_context



DISTANCE_THRESHOLD = 0.55


def cluster_verbs(verbs, entity=None, source=None, transition=None):
    verbs = list(dict.fromkeys(verbs))

    if len(verbs) == 0:
        return []

    if len(verbs) == 1:
        return [
            {
                "cluster_id": 0,
                "verbs": verbs,
                "label": label_cluster_with_llm(verbs, entity=entity, source=source,transition=transition)["label"]
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
        distance_threshold=DISTANCE_THRESHOLD
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


def compute_entity_latent_frame_transitions(source, entity):

    context = get_context(source)
    entity_key = entity.lower()
    if entity_key in context.entity_latent_frames:
        return context.entity_latent_frames[entity]
    
    framing_data = get_entity_framing(source=source, entity=entity)
    framing_results = framing_data["framing_results"]

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

    result = {"source":source, "entity":entity, "latent_frame_transitions":transitions}
    context.entity_latent_frames[entity_key] = result
    return result


def main():
    result = compute_entity_latent_frame_transitions(
        source="cnn.com",
        entity="china"
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