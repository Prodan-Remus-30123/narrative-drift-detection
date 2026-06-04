"""
dynamic_entity_ecosystem.py

Builds dynamic entity ecosystem summaries from entity framing drift.

Goal:
- identify stable core actors
- identify volatile actors
- identify episodic/disruptive actors
- track how entity framing changes across temporal transitions
"""

from collections import Counter
import numpy as np


def _top_verbs(verb_dict, top_n=5):
    if not verb_dict:
        return []

    return [verb for verb, _ in sorted(verb_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]]


def _dominant_verb(verb_dict):
    verbs = _top_verbs(verb_dict, top_n=1)
    return verbs[0] if verbs else None


def build_dynamic_entity_ecosystem(framing_drift, entity_importance=None, min_presence=1):
    """
    Builds temporal ecosystem statistics for entities.

    Input:
        framing_drift:
        {
            "2020-01_02->2020-03_04": {
                "China": {
                    "drift": 0.42,
                    "before": {"spread": 3, ...},
                    "after": {"blame": 4, ...}
                }
            }
        }

    Output:
        {
            "China": {
                "appearances": 4,
                "mean_drift": ...,
                "max_drift": ...,
                "drift_volatility": ...,
                "persistence_ratio": ...,
                "ecosystem_type": ...,
                "timeline": [...]
            }
        }
    """

    entity_timelines = {}

    transitions = sorted(framing_drift.keys())

    for transition in transitions:
        entities = framing_drift[transition]

        for entity, stats in entities.items():
            if entity not in entity_timelines:
                entity_timelines[entity] = []

            before = stats.get("before", {})
            after = stats.get("after", {})

            entity_timelines[entity].append({
                "transition": transition,
                "drift": stats.get("drift", 0),
                "before_top_verbs": _top_verbs(before),
                "before_only_verbs":stats.get("before_only_verbs",[]),
                "shared_verbs": stats.get("shared_verbs_list", []),
                "after_only_verbs": stats.get("after_only_verbs", []),
                "after_top_verbs": _top_verbs(after),
                "before_dominant_verb": _dominant_verb(before),
                "after_dominant_verb": _dominant_verb(after)
            })

    ecosystem = {}

    total_transitions = len(transitions)

    for entity, timeline in entity_timelines.items():
        if len(timeline) < min_presence:
            continue

        drift_values = [item["drift"] for item in timeline]
        dominant_shift_count = sum(1 for item in timeline if item["before_dominant_verb"] != item["after_dominant_verb"])

        persistence_ratio = (len(timeline) / total_transitions if total_transitions > 0 else 0)

        importance = 0

        if entity_importance:
            importance = entity_importance.get(entity, 0)

        mean_drift = float(np.mean(drift_values)) if drift_values else 0
        max_drift = float(np.max(drift_values)) if drift_values else 0
        drift_volatility = float(np.std(drift_values)) if drift_values else 0

        ecosystem_type = classify_entity_ecosystem_role(
            persistence_ratio=persistence_ratio,
            mean_drift=mean_drift,
            max_drift=max_drift,
            importance=importance
        )

        ecosystem[entity] = {
            "appearances": len(timeline),
            "persistence_ratio": persistence_ratio,
            "mean_drift": mean_drift,
            "max_drift": max_drift,
            "drift_volatility": drift_volatility,
            "dominant_shift_count": dominant_shift_count,
            "importance": importance,
            "ecosystem_type": ecosystem_type,
            "timeline": timeline
        }

    return ecosystem


def classify_entity_ecosystem_role(persistence_ratio, mean_drift, max_drift, importance):
    """
    Classifies an entity's temporal ecosystem role.
    """

    if persistence_ratio >= 0.7 and mean_drift < 0.35:
        return "stable_core_actor"

    if persistence_ratio >= 0.7 and mean_drift >= 0.35:
        return "volatile_core_actor"

    if persistence_ratio < 0.5 and max_drift >= 0.45:
        return "episodic_disruptor"

    if importance >= 0.5 and mean_drift >= 0.35:
        return "high_impact_shifting_actor"

    return "peripheral_actor"


def summarize_dynamic_entity_ecosystem(ecosystem):
    """
    Produces aggregate ecosystem-level summary.
    """

    role_counts = Counter()

    drift_values = []
    persistence_values = []

    for entity, stats in ecosystem.items():
        role_counts[stats["ecosystem_type"]] += 1
        drift_values.append(stats["mean_drift"])
        persistence_values.append(stats["persistence_ratio"])

    return {
        "num_entities": len(ecosystem),
        "role_counts": dict(role_counts),
        "mean_entity_drift": float(np.mean(drift_values)) if drift_values else 0,
        "mean_persistence": float(np.mean(persistence_values)) if persistence_values else 0
    }


def print_top_dynamic_entities(ecosystem, top_n=10, sort_by="importance"):
    """
    Compact debug printer.
    """

    ranked = sorted(ecosystem.items(), key=lambda x: x[1].get(sort_by, 0), reverse=True)

    for entity, stats in ranked[:top_n]:
        print(f"\nEntity: {entity}")
        print(f"Role: {stats['ecosystem_type']}")
        print(f"Importance: {stats['importance']:.3f}")
        print(f"Persistence: {stats['persistence_ratio']:.3f}")
        print(f"Mean drift: {stats['mean_drift']:.3f}")
        print(f"Max drift: {stats['max_drift']:.3f}")
        print(f"Volatility: {stats['drift_volatility']:.3f}")

        last_transition = stats["timeline"][-1]

        print(
            "Latest framing: "
            f"{last_transition['before_top_verbs']} -> "
            f"{last_transition['after_top_verbs']}"
        )
        print(
            "Shared verbs: "
            f"{last_transition.get('shared_verbs', [])}"
        )
        print(
            "Removed verbs: "
            f"{last_transition.get('before_only_verbs', [])[:10]}"
        )

        print(
            "Added verbs: "
            f"{last_transition.get('after_only_verbs', [])[:10]}"
        )