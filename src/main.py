import pandas as pd
from datetime import datetime

from preprocessing import preprocess_corpus
from embeddings import EmbeddingModel
from drift import compute_cosine_drift
from visualization import plot_multiple_sources
from changepoints import detect_changepoints
from entities import analyze_entities
from interpreter import interpret_shift
from database import load_full_articles
from agents.drift_agent import analyze_drift

def group_by_source_and_month(df):
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")

    grouped = {}

    for (source, month), rows in df.groupby(["source", "month"]):
        if source not in grouped:
            grouped[source] = {}

        grouped[source][month] = rows["text"].tolist()

    return grouped


def main():
    # 🔹 Load CSV
    df = load_full_articles()

    # 🔹 Group by month
    grouped = group_by_source_and_month(df)

    print("\n=== GROUPED DATA ===")

    for source in grouped:

        print(f"\nSource: {source}")

        for month in grouped[source]:

            print(
                month,
                len(grouped[source][month])
            )

    # 🔹 Preprocess
    for source in grouped:
        for month in grouped[source]:
            grouped[source][month] = preprocess_corpus(
                grouped[source][month]
            )

    # 🔹 Embeddings
    model = EmbeddingModel()

    source_results = {}
    print("\n=== Source-Aware Narrative Drift ===")

    for source in grouped:

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
            drift_interpretation = analyze_drift(
            source=source,
            period=f"{m1}->{m2}",
            drift_value=drift
            )

            print("\n[Drift Agent]")
            print(drift_interpretation)

            if drift > 0.1:
                print("  → Significant shift detected")
            else:
                print("  → Minor variation")

        source_results[source] = {
            "labels": drift_labels,
            "values": drift_values
        }

        # 🔹 Entity analysis per month

        months_sorted = sorted(grouped[source].keys())

        for month in months_sorted:

            print(f"\n{source} | {month}")

            entity_stats = analyze_entities(
                grouped[source][month]
            )

            for entity, stats in entity_stats.items():

                print(f"\nEntity: {entity}")

                print(
                    f"Subject count: {stats['subject_count']}"
                )

                print(
                    f"Object count: {stats['object_count']}"
                )

                print(
                    f"Top verbs: {stats['verbs'].most_common(5)}"
                )

        # 🔹 Narrative interpretation per transition

        for i in range(len(months_sorted) - 1):

            current_month = months_sorted[i + 1]

            entity_stats = analyze_entities(
                grouped[source][current_month]
            )

            interpretation = interpret_shift(
                source=source,
                period=f"{months_sorted[i]}->{current_month}",
                drift=drift_values[i],
                entity_stats=entity_stats
            )

            print("\nNarrative Interpretation:")
            print(
                f"{months_sorted[i]}->{current_month}"
            )

            print(interpretation)

        # # Change point
        # change_points = detect_changepoints(drift_values)
        # print(f"Detected change points: {change_points}")


    # 🔹 Plot drift signal
    plot_multiple_sources(source_results)


if __name__ == "__main__":
    main()