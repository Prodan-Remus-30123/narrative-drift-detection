"""
entity_frame_alignment.py

Aligns entity-level framing verbs to latent narrative frames.

No fixed thresholds are used.
Each entity-period/transition is ranked against all latent frames using:
- centroid cosine similarity
- max pairwise verb similarity
- lexical overlap
- ensemble rank
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from embedding_model_registry import get_embedding_model


def _safe_verbs(verb_dict_or_list):
    if verb_dict_or_list is None:
        return []

    if isinstance(verb_dict_or_list, dict):
        return [
            verb
            for verb, count in verb_dict_or_list.items()
            if count > 0
        ]

    if isinstance(verb_dict_or_list, list):
        return verb_dict_or_list

    return []


def _mean_embedding(verbs, model):
    if not verbs:
        return None

    embeddings = model.encode_documents(verbs)

    if embeddings is None or len(embeddings) == 0:
        return None

    return np.mean(
        np.array(embeddings),
        axis=0
    )


def _centroid_similarity(entity_verbs, frame_verbs, model):
    entity_embedding = _mean_embedding(entity_verbs, model)
    frame_embedding = _mean_embedding(frame_verbs, model)

    if entity_embedding is None or frame_embedding is None:
        return 0.0

    return float(
        cosine_similarity(
            [entity_embedding],
            [frame_embedding]
        )[0][0]
    )


def _max_pairwise_similarity(entity_verbs, frame_verbs, model):
    if not entity_verbs or not frame_verbs:
        return 0.0

    entity_embeddings = model.encode_documents(entity_verbs)
    frame_embeddings = model.encode_documents(frame_verbs)

    if len(entity_embeddings) == 0 or len(frame_embeddings) == 0:
        return 0.0

    sim_matrix = cosine_similarity(
        entity_embeddings,
        frame_embeddings
    )

    return float(np.max(sim_matrix))


def _lexical_overlap(entity_verbs, frame_verbs):
    entity_set = set(entity_verbs)
    frame_set = set(frame_verbs)

    if not entity_set or not frame_set:
        return 0.0

    intersection = entity_set.intersection(frame_set)
    union = entity_set.union(frame_set)

    return len(intersection) / len(union)


def _rank_scores(scores, key):
    ranked = sorted(
        scores,
        key=lambda x: x[key],
        reverse=True
    )

    for rank, item in enumerate(ranked, start=1):
        item[f"{key}_rank"] = rank

    return ranked


def _compute_ensemble_rank(scores):
    """
    Lower rank is better.
    No thresholds.
    """

    for key in [
        "centroid_similarity",
        "max_pairwise_similarity",
        "lexical_overlap"
    ]:
        scores = _rank_scores(scores, key)

    for item in scores:
        item["ensemble_rank_score"] = (
            item["centroid_similarity_rank"]
            + item["max_pairwise_similarity_rank"]
            + item["lexical_overlap_rank"]
        ) / 3

    return sorted(
        scores,
        key=lambda x: x["ensemble_rank_score"]
    )


def align_entity_verbs_to_frames(entity_verbs, latent_frames, semantic_frames, top_k=3):
    """
    Aligns one entity verb profile to latent frames.

    Args:
        entity_verbs:
            list[str] or dict[str, int]

        latent_frames:
            latent_result["clusters"]

    Returns:
        ranked frame alignments
    """

    model = get_embedding_model()

    entity_verbs = _safe_verbs(entity_verbs)

    scores = []

    for frame_id, frame_data in latent_frames.items():
        frame_verbs = frame_data.get("verbs", [])

        if not frame_verbs:
            continue

        centroid_score = _centroid_similarity(
            entity_verbs,
            frame_verbs,
            model
        )

        max_pairwise_score = _max_pairwise_similarity(
            entity_verbs,
            frame_verbs,
            model
        )

        overlap_score = _lexical_overlap(
            entity_verbs,
            frame_verbs
        )

        scores.append({
            "frame_id": frame_id,
            "frame_label": semantic_frames.get(frame_id, {}).get("frame_label", frame_data.get("label")),
            "frame_description":semantic_frames.get(frame_id,{}).get("frame_description",""),
            "frame_verbs": frame_verbs,
            "entity_verbs": entity_verbs,
            "centroid_similarity": centroid_score,
            "max_pairwise_similarity": max_pairwise_score,
            "lexical_overlap": overlap_score
        })

    ranked = _compute_ensemble_rank(scores)

    return ranked[:top_k]


def build_entity_frame_alignment(framing_drift, latent_result, semantic_frames, top_k=3):
    """
    Builds temporal entity-frame alignment from framing drift.

    For each entity and each transition, aligns:
    - before verbs
    - after verbs

    Output:
        {
            entity: [
                {
                    transition,
                    before_alignment,
                    after_alignment,
                    frame_migration
                }
            ]
        }
    """

    latent_frames = latent_result["clusters"]

    alignment = {}

    for transition, entities in framing_drift.items():
        for entity, stats in entities.items():
            before_verbs = stats.get("before", {})
            after_verbs = stats.get("after", {})

            before_alignment = align_entity_verbs_to_frames(
                before_verbs,
                latent_frames,
                semantic_frames,
                top_k=top_k
            )

            after_alignment = align_entity_verbs_to_frames(
                after_verbs,
                latent_frames,
                semantic_frames,
                top_k=top_k
            )

            before_top = (
                before_alignment[0]
                if before_alignment
                else None
            )

            after_top = (
                after_alignment[0]
                if after_alignment
                else None
            )

            frame_migration = None

            if before_top and after_top:
                frame_migration = {
                    "from_frame_id": before_top["frame_id"],
                    "from_frame": before_top["frame_label"],
                    "to_frame_id": after_top["frame_id"],
                    "to_frame": after_top["frame_label"],
                    "changed": before_top["frame_id"] != after_top["frame_id"]
                }

            if entity not in alignment:
                alignment[entity] = []

            alignment[entity].append({
                "transition": transition,
                "before_alignment": before_alignment,
                "after_alignment": after_alignment,
                "frame_migration": frame_migration
            })

    return alignment


def summarize_entity_frame_migrations(entity_frame_alignment):
    """
    Summarizes frame migrations without fixed thresholds.
    """

    summary = {}

    for entity, timeline in entity_frame_alignment.items():
        migrations = [
            item["frame_migration"]
            for item in timeline
            if item["frame_migration"] is not None
        ]

        if not migrations:
            continue

        changed_count = sum(
            1 for migration in migrations
            if migration["changed"]
        )

        visited_frames = set()

        for migration in migrations:
            visited_frames.add(migration["from_frame_id"])
            visited_frames.add(migration["to_frame_id"])

        summary[entity] = {
            "num_transitions": len(migrations),
            "migration_count": changed_count,
            "migration_ratio": changed_count / len(migrations),
            "unique_frames_visited": len(visited_frames),
            "latest_migration": migrations[-1]
        }

    return summary


def print_top_entity_frame_migrations(
    migration_summary,
    top_n=10
):
    ranked = sorted(
        migration_summary.items(),
        key=lambda x: (
            x[1]["migration_ratio"],
            x[1]["unique_frames_visited"]
        ),
        reverse=True
    )

    for entity, stats in ranked[:top_n]:
        latest = stats["latest_migration"]

        print(f"\nEntity: {entity}")
        print(f"Migration ratio: {stats['migration_ratio']:.3f}")
        print(f"Unique frames visited: {stats['unique_frames_visited']}")
        print(
            "Latest migration: "
            f"{latest['from_frame']} -> {latest['to_frame']}"
        )