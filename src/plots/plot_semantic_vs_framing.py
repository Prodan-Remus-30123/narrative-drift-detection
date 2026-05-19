import matplotlib.pyplot as plt


def plot_semantic_vs_framing(
    semantic_labels,
    semantic_values,
    framing_values,
    source
):

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