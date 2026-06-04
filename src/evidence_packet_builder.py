"""
Builds evidence packets for agentic narrative explanation.

Each packet represents one source-level temporal transition.
Agents consume these packets; they do not recompute analytics.
"""

import numpy as np


def _safe_float(value, default=0.0):
    if value is None:
        return default

    try:
        return float(value)
    except Exception:
        return default


def _split_transition(transition):
    if "->" not in transition:
        return transition, None

    left, right = transition.split("->", 1)
    return left, right


def _rank_transition(value, all_values):
    if not all_values:
        return None

    sorted_values = sorted(
        all_values,
        reverse=True
    )

    return sorted_values.index(value) + 1


def build_evidence_packets_for_source(
    source,
    source_result,
    top_n=3
):
    semantic = source_result.get(
        "semantic_drift",
        {}
    )

    labels = semantic.get("labels", [])
    values = semantic.get("values", [])
    threshold = semantic.get("threshold")

    change_profile = source_result.get(
        "change_profile",
        {}
    )

    affective = source_result.get(
        "affective_dynamics",
        {}
    )

    frame_migrations = source_result.get(
        "frame_migrations",
        {}
    )

    entity_ecosystem = source_result.get(
        "entity_ecosystem",
        {}
    )

    framing_drift = source_result.get(
        "framing_drift",
        {}
    )

    packets = []

    for index, transition in enumerate(labels):
        if index >= len(values):
            continue

        drift_value = values[index]

        period_start, period_end = _split_transition(
            transition
        )

        frame_transitions = change_profile.get(
            "frame_share_transitions",
            []
        )

        affective_transitions = change_profile.get(
            "affective_transitions",
            []
        )

        frame_profile = next(
            (
                item
                for item in frame_transitions
                if item.get("transition") == transition
            ),
            {}
        )

        affective_profile = next(
            (
                item
                for item in affective_transitions
                if item.get("transition") == transition
            ),
            {}
        )

        transition_profile = {
            **frame_profile,
            **affective_profile
        }

        top_rising_frames = transition_profile.get(
            "increasing_frames",
            []
        )

        top_falling_frames = transition_profile.get(
            "decreasing_frames",
            []
        )

        transition_framing = framing_drift.get(
            transition,
            {}
        )

        top_entity_reframings = sorted(
            transition_framing.items(),
            key=lambda item: item[1].get("drift", 0),
            reverse=True
        )[:top_n]

        entity_items = []

        for entity, stats in top_entity_reframings:
            ecosystem_stats = entity_ecosystem.get(
                entity,
                {}
            )

            entity_items.append({
                "entity": entity,
                "drift": _safe_float(
                    stats.get("drift")
                ),
                "drift_class": stats.get(
                    "drift_class"
                ),
                "before_verbs": list(
                    stats.get("before", {}).keys()
                )[:10],
                "after_verbs": list(
                    stats.get("after", {}).keys()
                )[:10],
                "added_verbs": stats.get(
                    "after_only_verbs",
                    []
                )[:10],
                "removed_verbs": stats.get(
                    "before_only_verbs",
                    []
                )[:10],
                "shared_verbs": stats.get(
                    "shared_verbs_list",
                    []
                )[:10],
                "ecosystem_role": ecosystem_stats.get(
                    "ecosystem_type"
                ),
                "importance": _safe_float(
                    ecosystem_stats.get("importance")
                ),
                "persistence_ratio": _safe_float(
                    ecosystem_stats.get("persistence_ratio")
                )
            })

        packet = {
            "source": source,

            "transition": transition,

            "period_start": period_start,

            "period_end": period_end,

            "what_how": {
                "semantic_drift": _safe_float(
                    drift_value
                ),
                "semantic_threshold": _safe_float(
                    threshold
                ),
                "semantic_classification": (
                    "significant"
                    if threshold is not None
                    and drift_value >= threshold
                    else "minor"
                ),
                "relative_drift_rank": _rank_transition(
                    drift_value,
                    values
                ),
                "top_rising_frames": top_rising_frames,
                "top_falling_frames": top_falling_frames,
                "top_entity_reframings": entity_items
            },

            "who": {
                "top_entities": entity_items,
                "frame_migrations": frame_migrations
            },

            "affective": {
                "compound_delta": _safe_float(
                    transition_profile.get("compound_delta")
                ),
                "intensity_delta": _safe_float(
                    transition_profile.get("intensity_delta")
                ),
                "polarization_delta": _safe_float(
                    transition_profile.get("polarization_delta")
                ),
                "source_affective_summary": affective
            },

            "why_context": {
                "available_evidence": [
                    "semantic_drift",
                    "rising_frames",
                    "falling_frames",
                    "entity_reframing",
                    "affective_dynamics"
                ],
                "causal_warning": (
                    "This packet supports evidence-backed "
                    "interpretation, not causal proof."
                )
            }
        }

        packets.append(packet)

    return packets


def build_all_evidence_packets(
    analysis_results,
    top_n=3
):
    packets = []

    for source, source_result in analysis_results.items():
        if not isinstance(source_result, dict):
            continue

        if source.startswith("__"):
            continue

        if "semantic_drift" not in source_result:
            continue

        packets.extend(
            build_evidence_packets_for_source(
                source=source,
                source_result=source_result,
                top_n=top_n
            )
        )

    return packets


def print_evidence_packet_summary(
    packets,
    max_packets=5
):
    print("\n=== EVIDENCE PACKETS ===")

    for packet in packets[:max_packets]:
        print(
            f"\n{packet['source']} | "
            f"{packet['transition']}"
        )

        what_how = packet["what_how"]

        print(
            "Semantic drift:",
            round(
                what_how["semantic_drift"],
                4
            ),
            "|",
            what_how["semantic_classification"]
        )

        print("Top rising frames:")
        for frame in what_how["top_rising_frames"]:
            if isinstance(frame, dict):
                print(
                    "-",
                    frame.get("frame_label"),
                    "| delta=",
                    round(frame.get("delta_share", 0), 4)
                )
            else:
                print("-", frame)

        print("Top falling frames:")
        for frame in what_how["top_falling_frames"]:
            if isinstance(frame, dict):
                print(
                    "-",
                    frame.get("frame_label"),
                    "| delta=",
                    round(frame.get("delta_share", 0), 4)
                )
            else:
                print("-", frame)

        print("Top entities:")
        for entity in packet["who"]["top_entities"]:
            print(
                "-",
                entity["entity"],
                "drift=",
                round(entity["drift"], 3),
                "role=",
                entity["ecosystem_role"]
            )