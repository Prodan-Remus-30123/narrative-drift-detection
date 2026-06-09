import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_entity_heatmap(framing_drift, top_n=15, min_transitions=1, title="Entity Framing Drift Heatmap"):
    """
    Plots entity-level framing drift across temporal transitions.

    Missing entity-transition pairs are stored as NaN, not 0.
    This prevents missing evidence from being interpreted as stable framing.
    """

    entity_scores = {}

    for transition, entities in framing_drift.items():
        for entity, stats in entities.items():
            if entity not in entity_scores:
                entity_scores[entity] = {"values": [], "count": 0}

            entity_scores[entity]["values"].append(stats.get("vocabulary_turnover", np.nan))
            entity_scores[entity]["count"] += 1

    ranked_entities = sorted(entity_scores.items(), key=lambda x: np.nanmean(x[1]["values"]), reverse=True)
    selected_entities = [entity for entity, stats in ranked_entities if stats["count"] >= min_transitions][:top_n]
    transitions = list(framing_drift.keys())

    matrix = pd.DataFrame(
        np.nan,
        index=selected_entities,
        columns=transitions
    )

    for transition, entities in framing_drift.items():
        for entity in selected_entities:
            if entity in entities:
                matrix.loc[entity, transition] = entities[entity].get("vocabulary_turnover", np.nan)

    if matrix.empty:
        print("No data available for entity heatmap.")
        return matrix

    plt.figure(figsize=(14, max(6, len(selected_entities) * 0.45)))

    plt.imshow(
        matrix.values,
        aspect="auto",
        interpolation="nearest"
    )

    plt.colorbar(label="Framing drift")

    plt.xticks(
        ticks=np.arange(len(matrix.columns)),
        labels=matrix.columns,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        ticks=np.arange(len(matrix.index)),
        labels=matrix.index
    )

    plt.title(title)
    plt.xlabel("Transition")
    plt.ylabel("Entity")

    plt.tight_layout()
    plt.show()

    return matrix