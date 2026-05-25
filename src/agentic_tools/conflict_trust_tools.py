"""
conflict_trust_tools.py

Agentic tools for conflict escalation,
institutional trust, and polarization signals.
"""

from database import load_full_articles
from preprocessing import preprocess_corpus
from temporal_entity_analysis import group_articles_by_period
from entities import analyze_entities
from utils.period_sorting import sort_period_key
from agentic_tools.context_registry import (
    get_context
)


CONFLICT_VERBS = {
    "accuse", "blame", "criticize", "attack", "condemn",
    "warn", "threaten", "punish", "ban", "restrict",
    "sanction", "withdraw", "terminate", "pressure",
    "deny", "fight", "clash", "oppose"
}


TRUST_POSITIVE_VERBS = {
    "declare", "recommend", "coordinate", "approve",
    "support", "advise", "inform", "confirm",
    "provide", "guide", "cooperate"
}


TRUST_NEGATIVE_VERBS = {
    "fail", "criticize", "ignore", "withdraw",
    "withhold", "terminate", "pressure",
    "deny", "accuse", "blame"
}


INSTITUTIONAL_ENTITIES = {
    "world health organization",
    "who",
    "cdc",
    "government",
    "white house",
    "state department",
    "congress",
    "fda",
    "johns hopkins university"
}


POLITICAL_ENTITIES = {
    "trump",
    "biden",
    "joe biden",
    "congress",
    "white house",
    "washington",
    "republicans",
    "democrats",
    "state department"
}


def get_conflict_trust_signals(source):
    context = get_context(source)

    if context.conflict_trust_results is not None:
        return context.conflict_trust_results

    grouped = context.get_preprocessed_grouped()

    results = {}

    for period in sorted(grouped.keys(), key=sort_period_key):

        entity_stats = analyze_entities(grouped[period])

        total_verbs = 0
        conflict_count = 0

        trust_positive = 0
        trust_negative = 0
        institutional_mentions = 0

        political_conflict = 0

        top_conflict_actors = {}

        for entity, stats in entity_stats.items():

            entity_lower = entity.lower()
            verbs = stats["verbs"]

            entity_total = sum(verbs.values())
            total_verbs += entity_total

            entity_conflict = 0

            for verb, count in verbs.items():

                verb_lower = verb.lower()

                if verb_lower in CONFLICT_VERBS:
                    conflict_count += count
                    entity_conflict += count

                    if entity_lower in POLITICAL_ENTITIES:
                        political_conflict += count

                if entity_lower in INSTITUTIONAL_ENTITIES:

                    institutional_mentions += count

                    if verb_lower in TRUST_POSITIVE_VERBS:
                        trust_positive += count

                    if verb_lower in TRUST_NEGATIVE_VERBS:
                        trust_negative += count

            if entity_conflict > 0:
                top_conflict_actors[entity] = entity_conflict

        conflict_score = (
            conflict_count / total_verbs
            if total_verbs > 0 else 0
        )

        trust_score = (
            (trust_positive - trust_negative)
            /
            (trust_positive + trust_negative)
            if (trust_positive + trust_negative) > 0 else 0
        )

        polarization_score = (
            political_conflict / total_verbs
            if total_verbs > 0 else 0
        )

        ranked_conflict_actors = sorted(
            top_conflict_actors.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        results[period] = {
            "total_verbs": int(total_verbs),
            "conflict_count": int(conflict_count),
            "conflict_score": float(conflict_score),
            "trust_positive": int(trust_positive),
            "trust_negative": int(trust_negative),
            "trust_score": float(trust_score),
            "institutional_mentions": int(institutional_mentions),
            "political_conflict": int(political_conflict),
            "polarization_score": float(polarization_score),
            "top_conflict_actors": [
                {
                    "entity": actor,
                    "conflict_count": int(score)
                }
                for actor, score in ranked_conflict_actors
            ]
        }

    context.conflict_trust_results = {
        "source": source,
        "conflict_trust_results": results
    }

    return context.conflict_trust_results