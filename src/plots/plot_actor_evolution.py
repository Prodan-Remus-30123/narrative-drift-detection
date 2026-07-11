import matplotlib.pyplot as plt
import numpy as np
from utils.period_sorting import sort_period_key
from utils.plot_saving import save_plot, DEFAULT_FIGSIZE

def plot_actor_evolution(drift_results, source, top_n=5, output_dir=None):

    actor_scores = {}

    for transition, entities in drift_results.items():
        for entity, stats in entities.items():
            if entity not in actor_scores:
                actor_scores[entity] = {}

            actor_scores[entity][transition] = stats.get("vocabulary_turnover", 0)

    overall_scores = {}

    for entity, transitions in actor_scores.items():
        overall_scores[entity] = sum(transitions.values())

    top_entities = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    all_transitions = sorted(drift_results.keys(), key=sort_period_key)

    plt.figure(figsize=DEFAULT_FIGSIZE)

    for entity, _ in top_entities:
        y = []

        for transition in all_transitions:
            if transition in actor_scores[entity]:
                y.append(actor_scores[entity][transition])
            else:
                y.append(np.nan)

        plt.plot(
            all_transitions,
            y,
            marker="o",
            label=entity
        )

    plt.xticks(rotation=45)
    plt.ylabel("Framing Drift")
    plt.title("Narrative Actor Evolution")

    plt.legend()

    plt.tight_layout()

    if output_dir:
        save_plot(
            output_dir,
            plot_name=f"{source}_actor_evolution",
            source=source
        )
    else:
        plt.close()