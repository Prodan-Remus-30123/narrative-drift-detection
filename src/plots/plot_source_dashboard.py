import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.period_sorting import sort_period_key


def plot_source_dashboard(source, semantic_labels, semantic_values, sentiment_results, framing_drift, top_n_entities=8):

    fig, axs = plt.subplots(2, 2, figsize=(18, 12))

    fig.suptitle(f"{source} Narrative Evolution Dashboard", fontsize=16)

    # 1. Semantic drift
    axs[0, 0].plot(
        semantic_labels,
        semantic_values,
        marker="o"
    )

    axs[0, 0].set_title("Semantic Drift")
    axs[0, 0].set_ylabel("Cosine Drift")
    axs[0, 0].tick_params(axis="x", rotation=45)

    # 2. Sentiment evolution
    periods = sorted(sentiment_results.keys(), key=sort_period_key)
    compound = [sentiment_results[p]["compound"] for p in periods]
    positive = [sentiment_results[p]["positive"] for p in periods]
    negative = [sentiment_results[p]["negative"] for p in periods]

    axs[0, 1].plot(
        periods,
        compound,
        marker="o",
        label="Compound"
    )

    axs[0, 1].plot(
        periods,
        positive,
        marker="o",
        label="Positive"
    )

    axs[0, 1].plot(
        periods,
        negative,
        marker="o",
        label="Negative"
    )

    axs[0, 1].set_title("Sentiment Evolution")
    axs[0, 1].tick_params(axis="x", rotation=45)
    axs[0, 1].legend()

    # 3. Average framing drift
    framing_labels = sorted(framing_drift.keys(), key=sort_period_key)

    avg_framing = []

    for transition in framing_labels:
        entities = framing_drift[transition]
        if len(entities) == 0:
            avg_framing.append(np.nan)
        else:
            avg = sum(stats["drift"] for stats in entities.values()) / len(entities)
            avg_framing.append(avg)

    axs[1, 0].plot(
        framing_labels,
        avg_framing,
        marker="o"
    )

    axs[1, 0].set_title("Average Entity Framing Drift")
    axs[1, 0].set_ylabel("Average Framing Drift")
    axs[1, 0].tick_params(axis="x", rotation=45)

    # 4. Entity drift heatmap
    entity_scores = {}

    for transition, entities in framing_drift.items():
        for entity, stats in entities.items():
            if entity not in entity_scores:
                entity_scores[entity] = []

            entity_scores[entity].append(stats["drift"])

    ranked_entities = sorted(entity_scores.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True)[:top_n_entities]
    selected_entities = [entity for entity, _ in ranked_entities]
    heatmap_data = []

    for entity in selected_entities:
        row = []

        for transition in framing_labels:
            value = np.nan
            if entity in framing_drift[transition]:
                value = framing_drift[transition][entity]["drift"]

            row.append(value)

        heatmap_data.append(row)

    if len(heatmap_data) > 0:
        heatmap_df = pd.DataFrame(
            heatmap_data,
            index=selected_entities,
            columns=framing_labels
        )

        im = axs[1, 1].imshow(heatmap_df, aspect="auto")

        axs[1, 1].set_xticks(range(len(framing_labels)))

        axs[1, 1].set_xticklabels(
            framing_labels,
            rotation=45,
            ha="right"
        )

        axs[1, 1].set_yticks(range(len(selected_entities)))
        axs[1, 1].set_yticklabels(selected_entities)
        axs[1, 1].set_title("Top Entity Framing Drift")

        fig.colorbar(im, ax=axs[1, 1])

    else:

        axs[1, 1].set_title("Top Entity Framing Drift")

        axs[1, 1].text(
            0.5,
            0.5,
            "No entity drift data",
            ha="center",
            va="center"
        )

    plt.tight_layout()
    plt.show()