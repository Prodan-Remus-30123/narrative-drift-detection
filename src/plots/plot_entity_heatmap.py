import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.period_sorting import (sort_period_key)

def plot_entity_heatmap(
    drift_results,
    top_n=10
):

    entity_scores = {}

    # Collect drift values
    for transition in sorted(drift_results.keys(),key=sort_period_key):

        entities = drift_results[transition]

        for entity, stats in entities.items():

            if entity not in entity_scores:
                entity_scores[entity] = []

            entity_scores[entity].append(
                stats["drift"]
            )

    # SAFETY CHECK
    if len(entity_scores) == 0:

        print(
            "No entity drift data available "
            "for heatmap."
        )

        return

    # Rank entities
    ranked_entities = sorted(

        entity_scores.items(),

        key=lambda x: (
            sum(x[1]) / len(x[1])
        ),

        reverse=True
    )[:top_n]

    top_entities = [
        entity[0]
        for entity in ranked_entities
    ]

    # Build matrix
    matrix = {}

    for transition, entities in drift_results.items():

        matrix[transition] = {}

        for entity in top_entities:

            if entity in entities:

                matrix[transition][entity] = (
                    entities[entity]["drift"]
                )

            else:

                matrix[transition][entity] = 0

    # Convert to DataFrame
    df = pd.DataFrame(matrix).T

    # SECOND SAFETY CHECK
    if df.empty:

        print(
            "Heatmap dataframe is empty."
        )

        return

    # Plot
    plt.figure(figsize=(14, 8))

    sns.heatmap(
        df,
        annot=True,
        cmap="Reds",
        linewidths=0.5
    )

    plt.title(
        "Entity Framing Drift Heatmap"
    )

    plt.xlabel(
        "Entities"
    )

    plt.ylabel(
        "Temporal Transitions"
    )

    plt.tight_layout()

    plt.show()