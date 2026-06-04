import matplotlib.pyplot as plt


def plot_top_entity_drift(drift_results, top_n=10):
    averaged = {}

    for transition in drift_results:
        for entity, stats in drift_results[transition].items():
            if entity not in averaged:
                averaged[entity] = []

            averaged[entity].append(stats["drift"])

    averaged_scores = {}

    for entity, drifts in averaged.items():
        averaged_scores[entity] = (sum(drifts) / len(drifts))

    ranked = sorted(averaged_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    entities = [x[0] for x in ranked]
    scores = [x[1] for x in ranked]

    plt.figure(figsize=(12, 6))

    plt.barh(entities, scores)

    plt.xlabel("Average Framing Drift")
    plt.ylabel("Entities")

    plt.title("Top Drifting Entities")

    plt.tight_layout()

    plt.show()