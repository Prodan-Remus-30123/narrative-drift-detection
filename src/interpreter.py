"""
interpreter.py

Narrative interpretation layer.

Generates human-readable explanations
based on drift signals and entity-level evidence.
"""


def interpret_shift(
    source,
    period,
    drift,
    entity_stats
):
    """
    Generate narrative interpretation.

    Args:
        source (str)
        period (str)
        drift (float)
        entity_stats (dict)

    Returns:
        str
    """

    interpretation = []

    # 🔹 Drift severity
    if drift > 0.5:
        interpretation.append(
            "Major narrative shift detected."
        )

    elif drift > 0.3:
        interpretation.append(
            "Moderate narrative shift detected."
        )

    else:
        interpretation.append(
            "Minor narrative variation detected."
        )

    # 🔹 Entity-based reasoning
    for entity, stats in entity_stats.items():

        top_verbs = [
            verb
            for verb, _ in stats["verbs"].most_common(3)
        ]

        if not top_verbs:
            continue

        interpretation.append(
            f"{entity} became associated with: "
            f"{', '.join(top_verbs)}."
        )

    return " ".join(interpretation)