"""
correlation_analysis.py

Cross-layer narrative correlation analysis.
"""

import numpy as np


def compute_sentiment_deltas(sentiment_results):
    periods = list(sentiment_results.keys())
    compound_values = [sentiment_results[p]["compound"] for p in periods]
    deltas = []

    for i in range(len(compound_values) - 1):
        delta = (compound_values[i + 1] - compound_values[i])
        deltas.append(delta)

    return deltas


def compute_average_framing(framing_drift):
    averages = []

    for transition, entities in framing_drift.items():
        if len(entities) == 0:
            averages.append(None)
            continue

        turnover_values = [stats.get("vocabulary_turnover") for stats in entities.values() if stats.get("vocabulary_turnover") is not None]
        avg = np.mean(turnover_values) if turnover_values else None
        averages.append(avg)

    return averages


def compute_correlation(sentiment_deltas, framing_values):

    paired = [(s, f) for s, f in zip(sentiment_deltas, framing_values) if f is not None]

    if len(paired) < 2:
        return None

    s_vals = [x[0] for x in paired]
    f_vals = [x[1] for x in paired]

    correlation = np.corrcoef(s_vals, f_vals)[0, 1]

    return correlation