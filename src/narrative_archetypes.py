"""
Narrative Archetype Discovery.

Discovers recurring narrative-change archetypes from transition-level signals.

No topic-specific labels or hardcoded domains.
Works across COVID, war, climate, homelessness, etc.
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


def _safe_float(value):
    if value is None:
        return 0.0

    try:
        return float(value)
    except Exception:
        return 0.0


def _top_abs_delta(frames):
    if not frames:
        return 0.0

    return max(
        abs(_safe_float(frame.get("delta_share", 0.0)))
        for frame in frames
    )


def _mean_abs_delta(frames):
    if not frames:
        return 0.0

    return float(
        np.mean([
            abs(_safe_float(frame.get("delta_share", 0.0)))
            for frame in frames
        ])
    )


def _top_entity_turnover(entity_changes):
    if not entity_changes:
        return 0.0

    return max(
        _safe_float(
            entity.get(
                "vocabulary_turnover",
                0.0
            )
        )
        for entity in entity_changes
    )


def _mean_entity_turnover(entity_changes):
    if not entity_changes:
        return 0.0

    return float(
        np.mean([
            _safe_float(
                entity.get(
                    "vocabulary_turnover",
                    0.0
                )
            )
            for entity in entity_changes
        ])
    )

def _mean_entity_js(entity_changes):
    if not entity_changes:
        return 0.0

    values = [
        _safe_float(
            entity.get("framing_drift_js", 0.0)
        )
        for entity in entity_changes
    ]

    if not values:
        return 0.0

    return float(np.mean(values))

def _count_entity_changes(entity_changes):
    return len(entity_changes or [])


def build_transition_archetype_table(
    analysis_results
):
    """
    Converts all source transitions into a transition-level feature table.
    """

    rows = []

    for source, result in analysis_results.items():
        semantic_drift = result.get("semantic_drift", {})
        change_profile = result.get("change_profile", {})

        semantic_labels = semantic_drift.get("labels", [])
        semantic_values = semantic_drift.get("values", [])

        semantic_by_transition = {
            label: value
            for label, value in zip(
                semantic_labels,
                semantic_values
            )
        }

        frame_transitions = {
            item["transition"]: item
            for item in change_profile.get(
                "frame_share_transitions",
                []
            )
        }

        affective_transitions = {
            item["transition"]: item
            for item in change_profile.get(
                "affective_transitions",
                []
            )
        }

        entity_transitions = {
            item["transition"]: item
            for item in change_profile.get(
                "entity_transition_evidence",
                []
            )
        }

        transitions = sorted(
            set(frame_transitions.keys())
            | set(affective_transitions.keys())
            | set(entity_transitions.keys())
            | set(semantic_by_transition.keys())
        )

        for transition in transitions:
            frame_item = frame_transitions.get(
                transition,
                {}
            )

            affective_item = affective_transitions.get(
                transition,
                {}
            )

            entity_item = entity_transitions.get(
                transition,
                {}
            )

            rising_frames = frame_item.get(
                "increasing_frames",
                []
            )

            falling_frames = frame_item.get(
                "decreasing_frames",
                []
            )

            strongest_frames = frame_item.get(
                "strongest_frame_movements",
                []
            )

            entity_changes = entity_item.get(
                "top_entity_changes",
                []
            )

            row = {
                "source": source,
                "transition": transition,

                "semantic_drift":
                    _safe_float(
                        semantic_by_transition.get(
                            transition,
                            0.0
                        )
                    ),

                "top_frame_movement":
                    _top_abs_delta(strongest_frames),

                "mean_frame_movement":
                    _mean_abs_delta(strongest_frames),

                "top_rising_frame":
                    rising_frames[0]["frame_label"]
                    if rising_frames else "",

                "top_falling_frame":
                    falling_frames[0]["frame_label"]
                    if falling_frames else "",

                "top_rising_delta":
                    _safe_float(
                        rising_frames[0].get(
                            "delta_share",
                            0.0
                        )
                    )
                    if rising_frames else 0.0,

                "top_falling_delta":
                    _safe_float(
                        falling_frames[0].get(
                            "delta_share",
                            0.0
                        )
                    )
                    if falling_frames else 0.0,

                "compound_delta":
                    _safe_float(
                        affective_item.get(
                            "compound_delta",
                            0.0
                        )
                    ),

                "intensity_delta":
                    _safe_float(
                        affective_item.get(
                            "intensity_delta",
                            0.0
                        )
                    ),

                "polarization_delta":
                    _safe_float(
                        affective_item.get(
                            "polarization_delta",
                            0.0
                        )
                    ),

                "top_entity_turnover":
                    _top_entity_turnover(entity_changes),

                "mean_entity_turnover":
                    _mean_entity_turnover(entity_changes),
                
                "mean_entity_js": _mean_entity_js(entity_changes),

                "num_entity_changes":
                    _count_entity_changes(entity_changes),

                "top_entity":
                    entity_changes[0]["entity"]
                    if entity_changes else "",

                "top_entity_added_verbs":
                    entity_changes[0].get(
                        "added_verbs",
                        []
                    )
                    if entity_changes else [],

                "top_entity_removed_verbs":
                    entity_changes[0].get(
                        "removed_verbs",
                        []
                    )
                    if entity_changes else []
            }

            rows.append(row)

    return pd.DataFrame(rows)


ARCHETYPE_FEATURES = [
    "semantic_drift",
    "top_frame_movement",
    "mean_frame_movement",
    "top_rising_delta",
    "top_falling_delta",
    "compound_delta",
    "intensity_delta",
    "polarization_delta",
    "top_entity_turnover",
    "mean_entity_turnover",
    "mean_entity_js",
    "num_entity_changes"
]


def choose_best_cluster_count(
    feature_matrix,
    min_k=2,
    max_k=6
):
    """
    Chooses k using silhouette score.
    No domain-specific assumptions.
    """

    n_samples = feature_matrix.shape[0]

    if n_samples < 4:
        return 2, {}

    max_k = min(max_k, n_samples - 1)

    scores = {}

    for k in range(min_k, max_k + 1):
        labels = AgglomerativeClustering(
            n_clusters=k
        ).fit_predict(feature_matrix)

        if len(set(labels)) < 2:
            continue

        score = silhouette_score(
            feature_matrix,
            labels
        )

        scores[k] = float(score)

    if not scores:
        return 2, {}

    best_k = max(
        scores.items(),
        key=lambda x: x[1]
    )[0]

    return best_k, scores


def discover_narrative_archetypes(
    archetype_table,
    n_clusters=None
):
    """
    Discovers archetypes using unsupervised clustering.
    """

    if archetype_table.empty:
        return archetype_table, {}, {}

    feature_df = archetype_table[
        ARCHETYPE_FEATURES
    ].fillna(0.0).astype(float).copy()

    # normalize inside each source first
    for source in archetype_table["source"].unique():

        mask = archetype_table["source"] == source

        source_data = feature_df.loc[mask].astype(float)

        if len(source_data) < 2:
            continue

        feature_df.loc[mask, ARCHETYPE_FEATURES] = (
            StandardScaler().fit_transform(
                source_data
            )
        )

    feature_matrix = feature_df.values

    # then global scaling
    scaled = StandardScaler().fit_transform(
        feature_matrix
    )

    if n_clusters is None:
        n_clusters, silhouette_scores = choose_best_cluster_count(
            scaled
        )
    else:
        silhouette_scores = {}

    labels = AgglomerativeClustering(
        n_clusters=n_clusters
    ).fit_predict(scaled)

    result = archetype_table.copy()
    result["archetype_id"] = labels

    summaries = summarize_archetypes(result)

    return result, summaries, silhouette_scores


def _most_common(values, top_n=5):
    counts = {}

    for value in values:
        if value is None or value == "":
            continue

        counts[value] = counts.get(value, 0) + 1

    return sorted(
        counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]


def summarize_archetypes(
    archetype_table
):
    """
    Builds compact, interpretable archetype summaries.
    """

    summaries = {}

    for archetype_id, group in archetype_table.groupby(
        "archetype_id"
    ):
        summaries[int(archetype_id)] = {
            "num_transitions": int(len(group)),

            "sources": sorted(
                group["source"].unique().tolist()
            ),

            "mean_semantic_drift":
                float(group["semantic_drift"].mean()),

            "mean_frame_movement":
                float(group["mean_frame_movement"].mean()),

            "mean_entity_turnover":
                float(group["mean_entity_turnover"].mean()),

            "mean_entity_js": float(group["mean_entity_js"].mean()),

            "mean_compound_delta":
                float(group["compound_delta"].mean()),

            "mean_intensity_delta":
                float(group["intensity_delta"].mean()),

            "mean_polarization_delta":
                float(group["polarization_delta"].mean()),

            "common_rising_frames":
                _most_common(
                    group["top_rising_frame"].tolist()
                ),

            "common_falling_frames":
                _most_common(
                    group["top_falling_frame"].tolist()
                ),

            "common_top_entities":
                _most_common(
                    group["top_entity"].tolist()
                ),

            "examples":
                group[
                    [
                        "source",
                        "transition",
                        "top_rising_frame",
                        "top_falling_frame",
                        "top_entity"
                    ]
                ].head(5).to_dict("records")
        }

    return summaries


def print_archetype_summaries(
    summaries
):
    print("\n=== NARRATIVE ARCHETYPE DISCOVERY ===")

    for archetype_id, summary in summaries.items():
        print(f"\nArchetype {archetype_id}")
        print(
            f"Transitions: "
            f"{summary['num_transitions']}"
        )
        print(
            f"Sources: "
            f"{summary['sources']}"
        )

        print(
            "Mean signals: "
            f"semantic={summary['mean_semantic_drift']:.4f}, "
            f"frame_move={summary['mean_frame_movement']:.4f}, "
            f"entity_turnover={summary['mean_entity_turnover']:.4f}, "
            f"entity_js={summary['mean_entity_js']:.4f}, "
            f"compound_delta={summary['mean_compound_delta']:.4f}, "
            f"intensity_delta={summary['mean_intensity_delta']:.4f}, "
            f"polarization_delta={summary['mean_polarization_delta']:.4f}"
        )

        print(
            "Common rising frames: "
            f"{summary['common_rising_frames']}"
        )

        print(
            "Common falling frames: "
            f"{summary['common_falling_frames']}"
        )

        print(
            "Common top entities: "
            f"{summary['common_top_entities']}"
        )

        print("Examples:")
        for example in summary["examples"]:
            print(
                f"- {example['source']} | "
                f"{example['transition']} | "
                f"{example['top_falling_frame']} -> "
                f"{example['top_rising_frame']} | "
                f"entity={example['top_entity']}"
            )