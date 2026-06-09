"""
Narrative change profile.

Answers:
HOW did the narrative change?

General-purpose, topic-independent layer:
- frame share redistribution
- affective/sentiment movement
- entity verb transition evidence
- transition-level change profile
"""


def _get_frame_label(frame_id, latent_frames, semantic_frames):
    frame_id_str = str(frame_id)

    if frame_id_str in semantic_frames:
        return semantic_frames[frame_id_str].get(
            "frame_label",
            latent_frames[frame_id].get("label", str(frame_id))
        )

    return latent_frames[frame_id].get(
        "label",
        str(frame_id)
    )


def build_frame_share_transitions(
    latent_frames,
    semantic_frames=None,
    top_n=5
):
    if semantic_frames is None:
        semantic_frames = {}

    period_set = set()

    for frame in latent_frames.values():
        for point in frame.get("values", []):
            period_set.add(point["period"])

    periods = sorted(period_set)

    results = []

    for i in range(len(periods) - 1):
        before_period = periods[i]
        after_period = periods[i + 1]

        frame_changes = []

        for frame_id, frame in latent_frames.items():
            values = {
                point["period"]: point.get("share", 0.0)
                for point in frame.get("values", [])
            }

            before_share = values.get(before_period, 0.0)
            after_share = values.get(after_period, 0.0)
            delta = after_share - before_share

            frame_changes.append({
                "frame_id": frame_id,
                "frame_label": _get_frame_label(
                    frame_id,
                    latent_frames,
                    semantic_frames
                ),
                "before_period": before_period,
                "after_period": after_period,
                "before_share": before_share,
                "after_share": after_share,
                "delta_share": delta,
                "abs_delta_share": abs(delta)
            })

        increasing = sorted(
            frame_changes,
            key=lambda x: x["delta_share"],
            reverse=True
        )[:top_n]

        decreasing = sorted(
            frame_changes,
            key=lambda x: x["delta_share"]
        )[:top_n]

        strongest = sorted(
            frame_changes,
            key=lambda x: x["abs_delta_share"],
            reverse=True
        )[:top_n]

        results.append({
            "transition": f"{before_period}->{after_period}",
            "increasing_frames": increasing,
            "decreasing_frames": decreasing,
            "strongest_frame_movements": strongest
        })

    return results


def build_affective_transitions(sentiment_results):
    periods = sorted(sentiment_results.keys())

    results = []

    for i in range(len(periods) - 1):
        before_period = periods[i]
        after_period = periods[i + 1]

        before = sentiment_results[before_period]
        after = sentiment_results[after_period]

        before_intensity = (
            before.get("positive", 0.0)
            + before.get("negative", 0.0)
        )

        after_intensity = (
            after.get("positive", 0.0)
            + after.get("negative", 0.0)
        )

        before_polarization = abs(
            before.get("positive", 0.0)
            - before.get("negative", 0.0)
        )

        after_polarization = abs(
            after.get("positive", 0.0)
            - after.get("negative", 0.0)
        )

        results.append({
            "transition": f"{before_period}->{after_period}",

            "compound_before": before.get("compound", 0.0),
            "compound_after": after.get("compound", 0.0),
            "compound_delta": (
                after.get("compound", 0.0)
                - before.get("compound", 0.0)
            ),

            "intensity_before": before_intensity,
            "intensity_after": after_intensity,
            "intensity_delta": after_intensity - before_intensity,

            "polarization_before": before_polarization,
            "polarization_after": after_polarization,
            "polarization_delta": (
                after_polarization
                - before_polarization
            )
        })

    return results


def entity_change_score(item):
    stats = item[1]

    turnover = stats.get(
        "vocabulary_turnover",
        0.0
    )

    similarity = stats.get(
        "shared_similarity"
    )

    shared_count = stats.get(
        "shared_verbs",
        0
    )

    similarity_signal = (
        0.0
        if similarity is None
        else (1.0 - similarity)
    )

    overlap_bonus = min(
        shared_count / 5.0,
        1.0
    )

    return (
        0.5 * turnover
        + 0.3 * similarity_signal
        + 0.2 * overlap_bonus
    )

def build_entity_transition_evidence(
    framing_drift,
    top_n=5
):
    results = []

    for transition, entities in framing_drift.items():
        ranked_entities = sorted(
        entities.items(),
        key=entity_change_score,
        reverse=True
    )

        transition_entities = []

        for entity, stats in ranked_entities[:top_n]:
            transition_entities.append({
                "entity": entity,
                 "vocabulary_turnover":
                    stats.get("vocabulary_turnover"),

                "shared_similarity":
                    stats.get("shared_similarity"),

                "framing_drift_js": stats.get("framing_drift_js"),

                "drift_class":
                    stats.get("drift_class"),

                "before_top_verbs": list(
                    stats.get("before", {}).keys()
                )[:10],

                "after_top_verbs": list(
                    stats.get("after", {}).keys()
                )[:10],

                "shared_verbs": stats.get(
                    "shared_verbs_list",
                    []
                ),

                "removed_verbs": stats.get(
                    "before_only_verbs",
                    []
                )[:10],

                "added_verbs": stats.get(
                    "after_only_verbs",
                    []
                )[:10]
            })

        results.append({
            "transition": transition,
            "top_entity_changes": transition_entities
        })

    return results


def build_narrative_change_profile(
    analysis_result,
    top_n=5
):
    latent_frames = analysis_result.get(
        "latent_frames",
        {}
    )

    semantic_frames = analysis_result.get(
        "semantic_frames",
        {}
    )

    sentiment_results = analysis_result.get(
        "sentiment",
        {}
    )

    framing_drift = analysis_result.get(
        "framing_drift",
        {}
    )

    return {
        "frame_share_transitions":
            build_frame_share_transitions(
                latent_frames=latent_frames,
                semantic_frames=semantic_frames,
                top_n=top_n
            ),

        "affective_transitions":
            build_affective_transitions(
                sentiment_results
            ),

        "entity_transition_evidence":
            build_entity_transition_evidence(
                framing_drift=framing_drift,
                top_n=top_n
            )
    }


def print_narrative_change_profile(
    change_profile,
    top_n=3
):
    print("\n=== HOW DID THE NARRATIVE CHANGE? ===")

    frame_transitions = change_profile.get(
        "frame_share_transitions",
        []
    )

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

    for transition_data in frame_transitions:
        transition = transition_data["transition"]

        print(f"\nTransition: {transition}")

        affective = affective_transitions.get(
            transition
        )

        if affective:
            print(
                "Affective change: "
                f"compound_delta="
                f"{affective['compound_delta']:.4f}, "
                f"intensity_delta="
                f"{affective['intensity_delta']:.4f}, "
                f"polarization_delta="
                f"{affective['polarization_delta']:.4f}"
            )

        print("\nTop rising frames:")
        for frame in transition_data[
            "increasing_frames"
        ][:top_n]:
            print(
                f"- {frame['frame_label']}: "
                f"{frame['before_share']:.4f} -> "
                f"{frame['after_share']:.4f} "
                f"(delta={frame['delta_share']:.4f})"
            )

        print("\nTop falling frames:")
        for frame in transition_data[
            "decreasing_frames"
        ][:top_n]:
            print(
                f"- {frame['frame_label']}: "
                f"{frame['before_share']:.4f} -> "
                f"{frame['after_share']:.4f} "
                f"(delta={frame['delta_share']:.4f})"
            )

        entity_data = entity_transitions.get(
            transition
        )

        if entity_data:
            print("\nTop entity reframings:")
            for item in entity_data[
                "top_entity_changes"
            ][:top_n]:
                turnover = item.get("vocabulary_turnover")
                similarity = item.get("shared_similarity")

                js = item.get("framing_drift_js")

                turnover_text = (
                    f"{turnover:.3f}"
                    if turnover is not None
                    else "N/A"
                )

                similarity_text = (
                    f"{similarity:.3f}"
                    if similarity is not None
                    else "N/A"
                )

                js_text = (
                    f"{js:.3f}"
                    if js is not None
                    else "N/A"
                )

                print(
                    f"- {item['entity']} "
                    f"(turnover={turnover_text}, "
                    f"js={js_text}, "
                    f"shared_similarity={similarity_text}): "
                    f"{item['before_top_verbs'][:5]} -> "
                    f"{item['after_top_verbs'][:5]}"
                )
                print(
                    f"  added={item['added_verbs'][:5]}, "
                    f"removed={item['removed_verbs'][:5]}"
                )