import matplotlib.pyplot as plt


def plot_actor_evolution(
    drift_results,
    top_n=5
):

    actor_scores = {}

    for transition, entities in drift_results.items():

        for entity, stats in entities.items():

            if entity not in actor_scores:

                actor_scores[entity] = {}

            actor_scores[entity][transition] = (
                stats["drift"]
            )

    overall_scores = {}

    for entity, transitions in actor_scores.items():

        overall_scores[entity] = sum(
            transitions.values()
        )

    top_entities = sorted(

        overall_scores.items(),

        key=lambda x: x[1],

        reverse=True
    )[:top_n]

    plt.figure(figsize=(12, 6))

    for entity, _ in top_entities:

        transitions = actor_scores[entity]

        x = list(transitions.keys())

        y = list(transitions.values())

        plt.plot(
            x,
            y,
            marker="o",
            label=entity
        )

    plt.xticks(rotation=45)

    plt.ylabel("Framing Drift")

    plt.title(
        "Narrative Actor Evolution"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()