import matplotlib.pyplot as plt
import numpy as np

def plot_semantic_vs_framing(semantic_labels, semantic_values, framing_values, source):

    semantic_values = np.array(semantic_values)
    framing_values = np.array(framing_values)

    if semantic_values.max() > 0:
        semantic_values = (semantic_values / semantic_values.max())

    if framing_values.max() > 0:
        framing_values = (framing_values / framing_values.max())

    plt.figure(figsize=(12, 6))

    plt.plot(
        semantic_labels,
        semantic_values,
        marker="o",
        label="Semantic Drift"
    )

    plt.plot(
        semantic_labels,
        framing_values,
        marker="o",
        label="Framing Drift"
    )

    plt.xticks(rotation=45)

    plt.ylabel("Drift")

    plt.title(
        f"{source} Semantic vs Framing Drift"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()