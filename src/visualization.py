"""
visualization.py

Utilities for visualizing narrative drift over time.
"""

import matplotlib.pyplot as plt


def plot_drift(month_labels, drift_values):
    """
    Plot temporal narrative drift signal.

    Args:
        month_labels (list): Labels for time windows.
        drift_values (list): Drift values between windows.
    """

    plt.figure(figsize=(8, 5))

    plt.plot(month_labels, drift_values, marker="o")

    plt.title("Narrative Drift Over Time")
    plt.xlabel("Time Window")
    plt.ylabel("Drift Magnitude")

    plt.grid(True)

    plt.show()