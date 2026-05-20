import matplotlib.pyplot as plt
import numpy as np

def plot_semantic_vs_framing(semantic_labels, semantic_values, framing_values, source):

    semantic_values = np.array(semantic_values, dtype=float)
    framing_values = np.array(framing_values, dtype=float)

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

    max_y = max(
        np.nanmax(semantic_values),
        np.nanmax(framing_values)
    )

    plt.ylim(0, max_y * 1.1)
    plt.xticks(rotation=45)

    plt.ylabel("Drift")

    plt.title(
        f"{source} Semantic vs Framing Drift"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()