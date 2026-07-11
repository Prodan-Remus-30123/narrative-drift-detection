from utils.numeric import safe_float as _safe_float
from entity_framing_drift import get_framing_drift_js

# Weights for the four confidence components; must sum to 1.0.
SEMANTIC_WEIGHT = 0.30
FRAMING_WEIGHT = 0.30
ACTOR_WEIGHT = 0.20
REGIME_WEIGHT = 0.20

# A source needs at least this many detected temporal regimes to be
# treated as having full regime-based evidence.
MAX_EXPECTED_REGIMES = 4

CONFIDENCE_HIGH_THRESHOLD = 0.70
CONFIDENCE_MEDIUM_THRESHOLD = 0.40


def _clamp(value, min_value=0.0, max_value=1.0):
    return max(min_value, min(max_value, value))


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
            js = get_framing_drift_js(stats)
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
            len(regimes.get("regimes", [])) / MAX_EXPECTED_REGIMES,
            1.0
        )
    else:
        regime_strength = 0.0

    confidence = (
        SEMANTIC_WEIGHT * semantic_strength
        + FRAMING_WEIGHT * framing_strength
        + ACTOR_WEIGHT * actor_strength
        + REGIME_WEIGHT * regime_strength
    )

    confidence = _clamp(confidence)

    if confidence >= CONFIDENCE_HIGH_THRESHOLD:
        label = "high"
    elif confidence >= CONFIDENCE_MEDIUM_THRESHOLD:
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