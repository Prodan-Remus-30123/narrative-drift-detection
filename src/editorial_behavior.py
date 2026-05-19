"""
editorial_behavior.py

Editorial behavior profiling.
"""


def classify_editorial_behavior(
    semantic,
    framing,
    sentiment
):

    if framing > 0.7:

        if sentiment < -0.1:

            return "conflict-driven"

        return "highly reframing"

    if semantic < 0.05 and framing < 0.4:

        return "stable institutional"

    if sentiment < -0.15:

        return "emotionally negative"

    return "mixed"