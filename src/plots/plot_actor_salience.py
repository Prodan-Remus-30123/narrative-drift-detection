import matplotlib.pyplot as plt
import numpy as np

from utils.period_sorting import sort_period_key
from utils.plot_saving import save_plot, DEFAULT_FIGSIZE

def plot_actor_salience(salience_results, source, top_n=5, output_dir = None):
    actor_totals = {}
    for period, entities in salience_results.items():
        for entity, score in entities.items():
            actor_totals[entity] = (actor_totals.get(entity, 0) + score)

    top_actors = sorted(actor_totals.items(), key=lambda x: x[1], reverse=True)[:top_n]
    periods = sorted(salience_results.keys(), key=sort_period_key)

    plt.figure(figsize=DEFAULT_FIGSIZE)

    for actor, _ in top_actors:
        values = []
        for period in periods:
            values.append(salience_results[period].get(actor, np.nan))

        plt.plot(
            periods,
            values,
            marker="o",
            label=actor
        )

    plt.title("Actor Salience Evolution")
    plt.xlabel("Period")
    plt.ylabel("Actor Salience")

    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    
    if output_dir:
        save_plot(
            output_dir,
            plot_name=f"{source}_actor_salience_evolution",
            source=source
        )
    else:
        plt.close()