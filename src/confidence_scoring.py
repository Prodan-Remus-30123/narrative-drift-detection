def _clamp(value, min_value=0.0, max_value=1.0):
    return max(min_value, min(max_value, value))


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def compute_confidence_score(source_result):
    """
    Computes source-level confidence based on:
    - semantic signal strength
    - entity framing evidence
    - actor persistence
    - temporal regime evidence
    """

    semantic = source_result.get("semantic_drift", {})
    threshold = semantic.get("threshold")
    values = semantic.get("values", [])

    if values and threshold:
        significant_count = sum(
            1 for value in values
            if value >= threshold
        )

        semantic_strength = significant_count / len(values)
    else:
        semantic_strength = 0.0

    ecosystem = source_result.get("entity_ecosystem", {})

    if ecosystem:
        core_actors = [
            entity for entity, stats in ecosystem.items()
            if stats.get("ecosystem_type") in {
                "stable_core_actor",
                "volatile_core_actor",
                "high_impact_shifting_actor"
            }
        ]

        actor_strength = len(core_actors) / max(len(ecosystem), 1)
    else:
        actor_strength = 0.0

    framing_values = []

    framing_drift = source_result.get("framing_drift", {})

    for transition, entities in framing_drift.items():
        for entity, stats in entities.items():
            js = (
                stats.get("js")
                or stats.get("js_drift")
                or stats.get("jensen_shannon")
            )

            turnover = stats.get("vocabulary_turnover")

            if js is not None:
                framing_values.append(_safe_float(js))

            elif turnover is not None:
                framing_values.append(_safe_float(turnover))

    framing_strength = (
        sum(framing_values) / len(framing_values)
        if framing_values else 0.0
    )

    regimes = source_result.get("temporal_narrative_regimes", {})

    if regimes.get("status") == "ok":
        regime_strength = min(
            len(regimes.get("regimes", [])) / 4,
            1.0
        )
    else:
        regime_strength = 0.0

    confidence = (
        0.30 * semantic_strength
        + 0.30 * framing_strength
        + 0.20 * actor_strength
        + 0.20 * regime_strength
    )

    confidence = _clamp(confidence)

    if confidence >= 0.70:
        label = "high"
    elif confidence >= 0.40:
        label = "medium"
    else:
        label = "low"

    return {
        "confidence_score": confidence,
        "confidence_label": label,
        "semantic_strength": semantic_strength,
        "framing_strength": framing_strength,
        "actor_strength": actor_strength,
        "regime_strength": regime_strength
    }


def print_confidence_score(source, confidence):
    print("\n=== CONFIDENCE SCORE ===")
    print("Source:", source)
    print(
        f"Confidence: {confidence['confidence_label']} "
        f"({confidence['confidence_score']:.3f})"
    )
    print(
        "Components: "
        f"semantic={confidence['semantic_strength']:.3f}, "
        f"framing={confidence['framing_strength']:.3f}, "
        f"actor={confidence['actor_strength']:.3f}, "
        f"regime={confidence['regime_strength']:.3f}"
    )