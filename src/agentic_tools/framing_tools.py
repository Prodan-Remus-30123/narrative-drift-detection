"""
framing_tools.py

Agentic tool layer for entity framing analysis.
"""

from database import load_full_articles
from preprocessing import preprocess_corpus

from temporal_entity_analysis import (
    group_articles_by_period
)

from entity_framing_drift import (
    compute_entity_drift,
    compute_entity_importance
)

from actor_salience import (
    compute_actor_salience,
    compute_total_actor_salience
)

from agentic_tools.context_registry import (
    get_context
)

def get_entity_framing(
    source,
    entity=None
):
    """
    Retrieve framing drift information
    for a source and optionally
    a specific entity.

    Args:
        source (str):
            News source.

        entity (str | None):
            Optional entity filter.

    Returns:
        dict
    """

    context = get_context(source)

    grouped = context.get_preprocessed_grouped()

    if context.framing_drift is None:

        context.framing_drift = compute_entity_drift(
            grouped
        )

    if context.salience_results is None:

        context.salience_results = compute_actor_salience(
            grouped
        )

    if context.salience_totals is None:

        context.salience_totals = compute_total_actor_salience(
            context.salience_results
        )

    framing_drift = context.framing_drift

    salience_totals = context.salience_totals

    if context.entity_importance is None:

        context.entity_importance = compute_entity_importance(
            framing_drift,
            salience_totals
        )

    entity_importance = context.entity_importance

    results = []

    for transition, entities in framing_drift.items():

        for current_entity, stats in entities.items():

            if entity is not None:

                if current_entity.lower() != entity.lower():
                    continue

            results.append({

                "transition":
                    transition,

                "entity":
                    current_entity,

                "drift":
                    float(stats["drift"]),

                "drift_class":
                    stats["drift_class"],

                "before_verbs":
                    list(stats["before"].keys())[:10],

                "after_verbs":
                    list(stats["after"].keys())[:10],

                "shared_verbs":
                    int(stats["shared_verbs"]),

                "importance":
                    float(
                        entity_importance.get(
                            current_entity,
                            0
                        )
                    ),

                "salience":
                    float(
                        salience_totals.get(
                            current_entity,
                            0
                        )
                    )
            })

    results = sorted(

        results,

        key=lambda x:
            x["drift"],

        reverse=True
    )

    return {

        "source":
            source,

        "entity_filter":
            entity,

        "framing_results":
            results
    }