import pandas as pd


def build_source_summary(source, semantic_values, framing_drift, entity_importance, num_periods, num_articles, avg_sentiment):

    avg_semantic = (sum(semantic_values) / len(semantic_values))

    framing_values = []
    for transition, entities in framing_drift.items():
        for entity, stats in entities.items():
            framing_values.append(stats["drift"])

    avg_framing = (sum(framing_values) / len(framing_values)) if framing_values else 0

    top_actor = None

    if entity_importance:
        top_actor = max(entity_importance.items(), key=lambda x: x[1])[0]

    return {

        "source": source,
        "avg_semantic_drift": round(avg_semantic, 4),
        "avg_framing_drift": round(avg_framing, 4),
        "top_actor": top_actor,
        "num_periods": num_periods,
        "num_articles": num_articles,
        "avg_sentiment": avg_sentiment
    }