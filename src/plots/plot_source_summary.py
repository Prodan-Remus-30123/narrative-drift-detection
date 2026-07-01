import pandas as pd
from utils.plot_saving import save_plot

def build_source_summary(source, semantic_values, framing_drift, entity_importance, num_periods, num_articles, avg_sentiment):

    avg_semantic = (sum(semantic_values) / len(semantic_values) if semantic_values else 0.0)

    framing_values = []
    for transition, entities in framing_drift.items():
        for entity, stats in entities.items():
            turnover = stats.get("vocabulary_turnover")
            if turnover is not None:
                framing_values.append(turnover)

    avg_framing = (sum(framing_values) / len(framing_values)) if framing_values else 0

    top_actor = None

    if entity_importance:
        top_actor = max(entity_importance.items(), key=lambda x: x[1])[0]

    return {

        "source": source,
        "avg_semantic_drift": round(avg_semantic, 4),
        "avg_framing_turnover": round(avg_framing, 4),
        "top_actor": top_actor,
        "num_periods": num_periods,
        "num_articles": num_articles,
        "avg_sentiment": avg_sentiment
    }