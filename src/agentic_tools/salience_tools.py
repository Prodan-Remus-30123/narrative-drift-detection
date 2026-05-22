"""
salience_tools.py

Agentic tools for narrative actor salience analysis.
"""

from database import load_full_articles

from preprocessing import preprocess_corpus

from temporal_entity_analysis import (
    group_articles_by_period
)

from actor_salience import (
    compute_actor_salience,
    compute_total_actor_salience
)

from utils.period_sorting import (
    sort_period_key
)


def get_actor_salience(
    source,
    entity=None
):
    """
    Retrieve actor salience evolution
    for a source and optionally
    a specific entity.

    Args:
        source (str)

        entity (str | None)

    Returns:
        dict
    """

    df = load_full_articles()

    source_df = df[
        df["source"] == source
    ]

    grouped = group_articles_by_period(
        source_df
    )

    for period in grouped:

        grouped[period] = preprocess_corpus(
            grouped[period]
        )

    salience_results = compute_actor_salience(
        grouped
    )

    salience_totals = compute_total_actor_salience(
        salience_results
    )

    periods_sorted = sorted(
        salience_results.keys(),
        key=sort_period_key
    )

    actor_evolution = {}

    for period in periods_sorted:

        entities = salience_results[
            period
        ]

        for current_entity, score in entities.items():

            if entity is not None:

                if current_entity.lower() != entity.lower():
                    continue

            if current_entity not in actor_evolution:

                actor_evolution[current_entity] = {}

            actor_evolution[current_entity][period] = float(score)

    ranked_totals = sorted(

        salience_totals.items(),

        key=lambda x: x[1],

        reverse=True
    )

    if entity is not None:

        ranked_totals = [

            x for x in ranked_totals

            if x[0].lower() == entity.lower()
        ]

    ranked_output = []

    for actor, total_score in ranked_totals:

        ranked_output.append({

            "entity":
                actor,

            "total_salience":
                float(total_score),

            "periods_present":
                len(
                    actor_evolution.get(
                        actor,
                        {}
                    )
                ),

            "salience_evolution":
                actor_evolution.get(
                    actor,
                    {}
                )
        })

    return {

        "source":
            source,

        "entity_filter":
            entity,

        "actor_salience":
            ranked_output
    }