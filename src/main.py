import pandas as pd
from datetime import datetime

from preprocessing import preprocess_corpus
from embeddings import EmbeddingModel
from drift import (compute_cosine_drift, compute_dynamic_threshold, classify_drift)
from plots.plot_semantic_comparison_across_sources import plot_multiple_sources
from changepoints import detect_changepoints
from entities import analyze_entities
from interpreter import interpret_shift
from database import load_full_articles
from agents.drift_agent import analyze_drift

from entity_framing_drift import compute_entity_drift, compute_entity_importance
from plots.plot_entity_drift import plot_top_entity_drift
from plots.plot_entity_heatmap import plot_entity_heatmap

from temporal_entity_analysis import group_articles_by_period
from plots.plot_semantic_vs_framing import plot_semantic_vs_framing
from plots.plot_source_summary import build_source_summary
from plots.plot_actor_evolution import (plot_actor_evolution)

from sentiment_analysis import (analyze_sentiment)
from plots.plot_sentiment_evolution import (plot_sentiment_evolution)

from correlation_analysis import (compute_sentiment_deltas, compute_average_framing, compute_correlation)
from editorial_behavior import (classify_editorial_behavior)
from emotional_volatility import (compute_emotional_volatility)


# def group_by_source_and_month(df):
#     df["date"] = pd.to_datetime(df["date"], format="mixed", utc=True)
#     df["date"] = df["date"].dt.tz_localize(None)
#     df["month"] = df["date"].dt.to_period("M")

#     grouped = {}

#     for (source, month), rows in df.groupby(["source", "month"]):
#         if source not in grouped:
#             grouped[source] = {}

#         grouped[source][month] = rows["text"].tolist()

#     return grouped


def main():
    #  Load CSV
    df = load_full_articles()

    #  Group by month
    grouped = {}

    for source in df["source"].unique():

        source_df = df[df["source"] == source]

        grouped[source] = (group_articles_by_period(source_df))

    print("\n=== GROUPED DATA ===")

    source_summaries = []

    for source in grouped:
        total_docs = sum( len(grouped[source][period]) for period in grouped[source])

        if total_docs < 50:
            print(f"\nSkipping {source}: insufficient documents ({total_docs})")
            continue

        print(f"\nSource: {source}")

        for month in grouped[source]:

            print(
                month,
                len(grouped[source][month])
            )

    #  Preprocess
    for source in grouped:
        for month in grouped[source]:
            grouped[source][month] = preprocess_corpus(grouped[source][month])

    #  Embeddings
    model = EmbeddingModel()

    source_results = {}
    print("\n=== Source-Aware Narrative Drift ===")

    for source in grouped:
        total_docs = sum(len(grouped[source][period]) for period in grouped[source])
        if total_docs < 50:
            print(f"\nSkipping {source}: insufficient documents ({total_docs})")
            continue

        aggregated_vectors = []

        for month in sorted(grouped[source].keys()):
            embeddings = model.encode_documents(grouped[source][month])

            vec = model.aggregate_embeddings(embeddings)

            aggregated_vectors.append((month, vec))

        drift_values = []
        drift_labels = []

        print(f"\nSource: {source}")

        print("\n=== Entity-Level Evidence ===")

        for i in range(len(aggregated_vectors) - 1):

            m1, v1 = aggregated_vectors[i]
            m2, v2 = aggregated_vectors[i + 1]

            drift = compute_cosine_drift(v1, v2)
            
            drift_values.append(drift)
            drift_labels.append(f"{m1}->{m2}")

            print(f"{m1} → {m2} | Drift: {drift:.4f}")
            # drift_interpretation = analyze_drift(
            # source=source,
            # period=f"{m1}->{m2}",
            # drift_value=drift
            # )

            # print("\n[Drift Agent]")
            # print(drift_interpretation)
        
        dynamic_threshold = compute_dynamic_threshold(drift_values,method="median_mad")

        print(f"\nDynamic semantic threshold: " f"{dynamic_threshold:.4f}" if dynamic_threshold is not None else "\nDynamic semantic threshold: insufficient data")

        for label, drift in zip(drift_labels, drift_values):
            classification = classify_drift(drift, dynamic_threshold)

            print(f"{label} | Drift: {drift:.4f} | " f"Classification: {classification}")

        source_results[source] = {
            "labels": drift_labels,
            "values": drift_values
        }

        print("\n=== Sentiment Evolution ===")

        sentiment_results = {}

        for period in grouped[source]:

            sentiment = analyze_sentiment(grouped[source][period])
            sentiment_results[str(period)] = sentiment
            print(f"\n{period}")
            print(f"Compound: " f"{sentiment['compound']:.4f}")
            print(f"Positive: " f"{sentiment['positive']:.4f}")
            print(f"Negative: " f"{sentiment['negative']:.4f}")

            print(f"Neutral: " f"{sentiment['neutral']:.4f}")
        
        plot_sentiment_evolution(sentiment_results, source)

        print("\n=== Entity Framing Drift ===")

        framing_drift = compute_entity_drift(grouped[source])
        entity_importance = compute_entity_importance(framing_drift)

        summary = build_source_summary(
            source= source,
            semantic_values= drift_values,
            framing_drift= framing_drift,
            entity_importance= entity_importance,
            num_periods= len(grouped[source]),
            num_articles= sum(len(grouped[source][period]) for period in grouped[source]),
            avg_sentiment= sum(sentiment_results[p]["compound"] for p in sentiment_results) / len(sentiment_results)
        )

        source_summaries.append(summary)

        behavior = classify_editorial_behavior(
            semantic=summary["avg_semantic_drift"],
            framing=summary["avg_framing_drift"],
            sentiment=summary["avg_sentiment"]
        )

        print(f"\nEditorial behavior: "f"{behavior}")

        volatility = compute_emotional_volatility(sentiment_results)
        print(f"\nEmotional volatility: "f"{volatility:.4f}")

        print("\n=== Top Important Narrative Actors ===")

        ranked_importance = sorted(

            entity_importance.items(),

            key=lambda x: x[1],

            reverse=True
        )

        for entity, score in ranked_importance[:10]:

            print(
                f"{entity}: {score:.3f}"
            )

        average_framing_values = []

        for transition, entities in framing_drift.items():

            if len(entities) == 0:
                average_framing_values.append(0)
                continue

            avg = sum(stats["drift"] for stats in entities.values()) / len(entities)

            average_framing_values.append(avg)

        plot_semantic_vs_framing(
            semantic_labels=drift_labels,
            semantic_values=drift_values,
            framing_values=average_framing_values,
            source=source
        )

        plot_top_entity_drift(framing_drift)
        plot_entity_heatmap(framing_drift)
        plot_actor_evolution(framing_drift)

        print("\n=== Cross-Layer Correlation ===")

        sentiment_deltas = compute_sentiment_deltas(sentiment_results)
        avg_framing = compute_average_framing(framing_drift)
        correlation = compute_correlation(sentiment_deltas, avg_framing)

        print(f"Sentiment ↔ Framing correlation: " f"{correlation}")

        for transition, entities in framing_drift.items():
            print(f"\n{transition}")
            ranked_entities = sorted(entities.items(), key=lambda x: x[1]["drift"], reverse=True)

            for entity, stats in ranked_entities[:5]:
                print(f"\nEntity: {entity}")
                print(f"Drift: " f"{stats['drift']:.3f}")
                print(f"Before: " f"{list(stats['before'].keys())[:5]}")
                print(f"After: " f"{list(stats['after'].keys())[:5]}")

        # Entity analysis per month

        months_sorted = sorted(grouped[source].keys())

        # for month in months_sorted:

        #     print(f"\n{source} | {month}")

        #     entity_stats = analyze_entities(
        #         grouped[source][month]
        #     )

        #     for entity, stats in entity_stats.items():

        #         print(f"\nEntity: {entity}")

        #         print(
        #             f"Subject count: {stats['subject_count']}"
        #         )

        #         print(
        #             f"Object count: {stats['object_count']}"
        #         )

        #         print(
        #             f"Top verbs: {stats['verbs'].most_common(5)}"
        #         )

        #  Narrative interpretation per transition

        # for i in range(len(months_sorted) - 1):

        #     current_month = months_sorted[i + 1]

        #     entity_stats = analyze_entities(
        #         grouped[source][current_month]
        #     )

        #     interpretation = interpret_shift(
        #         source=source,
        #         period=f"{months_sorted[i]}->{current_month}",
        #         drift=drift_values[i],
        #         entity_stats=entity_stats
        #     )

        #     print("\nNarrative Interpretation:")
        #     print(
        #         f"{months_sorted[i]}->{current_month}"
        #     )

        #     print(interpretation)

        # # Change point
        # change_points = detect_changepoints(drift_values)
        # print(f"Detected change points: {change_points}")


    #  Plot drift signal
    plot_multiple_sources(source_results)

    summary_df = pd.DataFrame(
        source_summaries
    )

    print("\n=== SOURCE SUMMARY ===")

    print(summary_df)



if __name__ == "__main__":
    main()