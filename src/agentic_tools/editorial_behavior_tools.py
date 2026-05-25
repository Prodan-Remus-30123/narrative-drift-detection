"""
editorial_behavior_tools.py

Agentic tools for source-level editorial behavior analysis.
"""

from agentic_tools.semantic_tools import get_semantic_drift
from agentic_tools.framing_tools import get_entity_framing
from agentic_tools.sentiment_tools import get_sentiment_evolution
from agentic_tools.conflict_trust_tools import get_conflict_trust_signals


def get_editorial_behavior_profile(source):
    semantic_data = get_semantic_drift(source)
    framing_data = get_entity_framing(source)
    sentiment_data = get_sentiment_evolution(source)
    conflict_data = get_conflict_trust_signals(source)

    semantic_results = semantic_data["semantic_drift"]
    framing_results = framing_data["framing_results"]
    sentiment_results = sentiment_data["sentiment_evolution"]
    conflict_results = conflict_data["conflict_trust_results"]

    avg_semantic_drift = (
        sum(item["drift"] for item in semantic_results)
        / len(semantic_results)
        if semantic_results else 0
    )

    significant_semantic_shifts = sum(
        1 for item in semantic_results
        if item["classification"] == "significant"
    )

    avg_framing_drift = (
        sum(item["drift"] for item in framing_results)
        / len(framing_results)
        if framing_results else 0
    )

    major_reframing_events = sum(
        1 for item in framing_results
        if item["drift_class"] == "major reframing"
    )

    avg_conflict_score = (
        sum(item["conflict_score"] for item in conflict_results.values())
        / len(conflict_results)
        if conflict_results else 0
    )

    avg_polarization_score = (
        sum(item["polarization_score"] for item in conflict_results.values())
        / len(conflict_results)
        if conflict_results else 0
    )

    avg_trust_score = (
        sum(item["trust_score"] for item in conflict_results.values())
        / len(conflict_results)
        if conflict_results else 0
    )

    return {
        "source": source,
        "avg_semantic_drift": float(avg_semantic_drift),
        "significant_semantic_shifts": int(significant_semantic_shifts),
        "avg_framing_drift": float(avg_framing_drift),
        "major_reframing_events": int(major_reframing_events),
        "avg_sentiment": float(sentiment_data["average_compound"]),
        "emotional_volatility": float(sentiment_data["emotional_volatility"]),
        "avg_conflict_score": float(avg_conflict_score),
        "avg_polarization_score": float(avg_polarization_score),
        "avg_trust_score": float(avg_trust_score),
        "semantic_data": semantic_data,
        "sentiment_data": sentiment_data,
        "conflict_data": conflict_data,
    }