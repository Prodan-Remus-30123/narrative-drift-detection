"""
visualization.py

Utilities for visualizing narrative drift over time.
"""

import matplotlib.pyplot as plt


def plot_multiple_sources(source_results):
    """
    Plot drift signals for multiple sources.

    Args:
        source_results (dict):
            {
                source_name: {
                    "labels": [...],
                    "values": [...]
                }
            }
    """

    plt.figure(figsize=(10, 6))

    for source, data in source_results.items():
        plt.plot(
            data["labels"],
            data["values"],
            marker="o",
            label=source
        )

    plt.title("Semantic Drift Comparison Across Sources")

    plt.xlabel("Time Window")
    plt.ylabel("Drift Magnitude")

    plt.xticks(rotation=30)

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()