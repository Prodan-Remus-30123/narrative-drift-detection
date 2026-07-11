"""
editorial_behavior.py

Editorial behavior profiling: labels a source's overall coverage style
from its average semantic drift, framing turnover and sentiment.
"""

# Thresholds are on the same 0-1 scale as the inputs: framing turnover
# (Jaccard-based vocabulary turnover, see entity_framing_drift.py),
# average semantic drift (cosine distance, see drift.py), and VADER
# compound sentiment (-1..1).
HIGH_FRAMING_TURNOVER = 0.7
NEGATIVE_SENTIMENT_UNDER_HIGH_FRAMING = -0.1
STABLE_SEMANTIC_DRIFT = 0.05
STABLE_FRAMING_TURNOVER = 0.4
NEGATIVE_SENTIMENT = -0.15


def classify_editorial_behavior(semantic, framing, sentiment):
    """
    Label a source's editorial style from its average semantic drift,
    framing turnover and sentiment for one analysis run.
    """

    if framing > HIGH_FRAMING_TURNOVER:
        if sentiment < NEGATIVE_SENTIMENT_UNDER_HIGH_FRAMING:
            return "conflict-driven"

        return "highly reframing"

    if semantic < STABLE_SEMANTIC_DRIFT and framing < STABLE_FRAMING_TURNOVER:
        return "stable institutional"

    if sentiment < NEGATIVE_SENTIMENT:
        return "emotionally negative"

    return "mixed"