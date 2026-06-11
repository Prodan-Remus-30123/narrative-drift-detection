from collections import Counter, defaultdict


def _month_to_index(month_label):
    """
    Converts '2020-03' or '2020-03_04' to numeric month index.
    Uses the first month if a bimonthly label is provided.
    """
    if month_label is None:
        return None

    label = str(month_label)

    if "->" in label:
        label = label.split("->")[0]

    year = int(label[:4])
    month = int(label[5:7])

    return year * 12 + month


def _extract_transition_start_month(transition_label):
    """
    Example:
    '2020-03_04->2020-05_06' -> index for 2020-03
    """
    if transition_label is None:
        return None

    left = str(transition_label).split("->")[0]

    return _month_to_index(left)


def _find_regime_for_transition(transition_label, regimes):
    transition_month = _extract_transition_start_month(transition_label)

    if transition_month is None:
        return None

    for regime in regimes:
        if regime["start_index"] <= transition_month <= regime["end_index"]:
            return regime["regime_id"]

    return None


def build_temporal_regimes_from_change_points(source_result):
    """
    Builds temporal narrative regimes using monthly semantic embedding
    change points.

    Expected input:
    source_result["change_points"]["monthly_semantic_embedding_trajectory"]
    """

    monthly_trajectory = source_result.get(
        "monthly_semantic_embedding_trajectory",
        {}
    )

    monthly_labels = monthly_trajectory.get("labels", [])

    cp_result = (
        source_result
        .get("change_points", {})
        .get("monthly_semantic_embedding_trajectory", {})
    )

    change_points = cp_result.get("change_points", [])

    if not monthly_labels:
        return {
            "status": "skipped",
            "reason": "no monthly labels available",
            "regimes": []
        }

    if not change_points:
        return {
            "status": "skipped",
            "reason": "no monthly semantic change points detected",
            "regimes": []
        }

    boundary_labels = [
        point.get("label")
        for point in change_points
        if point.get("label") is not None
    ]

    all_boundaries = [monthly_labels[0]] + boundary_labels

    regimes = []

    for i, start_label in enumerate(all_boundaries):
        start_index = _month_to_index(start_label)

        if i + 1 < len(all_boundaries):
            next_start_index = _month_to_index(all_boundaries[i + 1])
            end_index = next_start_index - 1
            end_label = all_boundaries[i + 1]
        else:
            end_label = monthly_labels[-1]
            end_index = _month_to_index(end_label)

        regimes.append({
            "regime_id": f"regime_{i + 1}",
            "start": start_label,
            "end": end_label,
            "start_index": start_index,
            "end_index": end_index
        })

    return {
        "status": "ok",
        "boundary_labels": boundary_labels,
        "regimes": regimes
    }


def summarize_temporal_regimes(source_result, top_n=5):
    """
    Summarizes each regime using existing bimonthly analysis:
    - semantic drift
    - rising/falling frames
    - top reframed entities
    - sentiment deltas
    """

    regime_result = build_temporal_regimes_from_change_points(source_result)

    if regime_result["status"] != "ok":
        return regime_result

    regimes = regime_result["regimes"]

    semantic = source_result.get("semantic_drift", {})
    semantic_labels = semantic.get("labels", [])
    semantic_values = semantic.get("values", [])

    change_profile = source_result.get(
        "change_profile",
        {}
    )

    frame_transitions = change_profile.get(
        "frame_share_transitions",
        []
    )

    entity_transitions = {
        item["transition"]: item
        for item in change_profile.get(
            "entity_transition_evidence",
            []
        )
    }

    affective = source_result.get("affective_dynamics", {})
    affective_trajectory = affective.get("trajectory", [])

    summaries = []

    for regime in regimes:
        regime_id = regime["regime_id"]

        drift_values = []
        rising_frames = Counter()
        falling_frames = Counter()
        entities = Counter()
        compound_deltas = []
        intensity_deltas = []
        polarization_deltas = []
        transitions = []

        for label, value in zip(semantic_labels, semantic_values):
            assigned_regime = _find_regime_for_transition(label, regimes)

            if assigned_regime == regime_id:
                transitions.append(label)
                drift_values.append(value)

                for transition_data in frame_transitions:
                    transition = transition_data.get(
                        "transition"
                    )

                    assigned_regime = _find_regime_for_transition(
                        transition,
                        regimes
                    )

                    if assigned_regime != regime_id:
                        continue

                    for frame in transition_data.get(
                        "increasing_frames",
                        []
                    )[:top_n]:

                        label = frame.get("frame_label")

                        if label:
                            rising_frames[label] += 1

                    for frame in transition_data.get(
                        "decreasing_frames",
                        []
                    )[:top_n]:

                        label = frame.get("frame_label")

                        if label:
                            falling_frames[label] += 1

                    entity_data = entity_transitions.get(
                        transition
                    )

                    if entity_data:

                        for item in entity_data.get(
                            "top_entity_changes",
                            []
                        )[:top_n]:

                            entity = item.get("entity")

                            if entity:
                                entities[entity] += 1

        for item in affective_trajectory:
            transition = (
                item.get("transition")
                or item.get("label")
                or item.get("period")
            )

            assigned_regime = _find_regime_for_transition(transition, regimes)

            if assigned_regime != regime_id:
                continue

            compound_deltas.append(item.get("compound_delta", 0.0))
            intensity_deltas.append(item.get("intensity_delta", 0.0))
            polarization_deltas.append(item.get("polarization_delta", 0.0))

        summary = {
            **regime,
            "num_transitions": len(transitions),
            "transitions": transitions,
            "mean_semantic_drift": (
                sum(drift_values) / len(drift_values)
                if drift_values else 0.0
            ),
            "max_semantic_drift": (
                max(drift_values)
                if drift_values else 0.0
            ),
            "mean_compound_delta": (
                sum(compound_deltas) / len(compound_deltas)
                if compound_deltas else 0.0
            ),
            "mean_intensity_delta": (
                sum(intensity_deltas) / len(intensity_deltas)
                if intensity_deltas else 0.0
            ),
            "mean_polarization_delta": (
                sum(polarization_deltas) / len(polarization_deltas)
                if polarization_deltas else 0.0
            ),
            "top_rising_frames": rising_frames.most_common(top_n),
            "top_falling_frames": falling_frames.most_common(top_n),
            "top_entities": entities.most_common(top_n)
        }

        summaries.append(summary)

    return {
        "status": "ok",
        "boundary_labels": regime_result["boundary_labels"],
        "regimes": summaries
    }


def print_temporal_regime_summary(regime_result):
    print("\n=== TEMPORAL NARRATIVE REGIMES ===")

    if regime_result.get("status") != "ok":
        print("Skipped:", regime_result.get("reason"))
        return

    print("Detected regime boundaries:")
    for boundary in regime_result.get("boundary_labels", []):
        print(f"- {boundary}")

    for regime in regime_result.get("regimes", []):
        print(
            f"\n{regime['regime_id']}: "
            f"{regime['start']} -> {regime['end']}"
        )

        print(f"Transitions: {regime['num_transitions']}")
        print(
            "Semantic drift mean/max: "
            f"{regime['mean_semantic_drift']:.4f} / "
            f"{regime['max_semantic_drift']:.4f}"
        )

        print(
            "Affective mean deltas: "
            f"compound={regime['mean_compound_delta']:.4f}, "
            f"intensity={regime['mean_intensity_delta']:.4f}, "
            f"polarization={regime['mean_polarization_delta']:.4f}"
        )

        print("Top rising frames:")
        for frame, count in regime["top_rising_frames"]:
            print(f"- {frame} ({count})")

        print("Top falling frames:")
        for frame, count in regime["top_falling_frames"]:
            print(f"- {frame} ({count})")

        print("Top entities:")
        for entity, count in regime["top_entities"]:
            print(f"- {entity} ({count})")