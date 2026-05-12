import pandas as pd
from datetime import datetime

from preprocessing import preprocess_corpus
from embeddings import EmbeddingModel
from drift import compute_cosine_drift
from visualization import plot_drift


def group_by_month(df):
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")
    return df.groupby("month")["text"].apply(list)


def main():
    # 🔹 Load CSV
    df = pd.read_csv("data/exemplu.csv")

    # 🔹 Group by month
    grouped = group_by_month(df)

    # 🔹 Preprocess
    grouped = {k: preprocess_corpus(v) for k, v in grouped.items()}

    # 🔹 Embeddings
    model = EmbeddingModel()
    aggregated_vectors = []

    for month in sorted(grouped.keys()):
        embeddings = model.encode_documents(grouped[month])
        vec = model.aggregate_embeddings(embeddings)
        aggregated_vectors.append((month, vec))

    # 🔹 Compute drift between consecutive months
    print("\n=== Narrative Drift on Real Data ===")

    drift_values = []
    drift_labels = []

    for i in range(len(aggregated_vectors) - 1):
        m1, v1 = aggregated_vectors[i]
        m2, v2 = aggregated_vectors[i + 1]

        drift = compute_cosine_drift(v1, v2)

        drift_values.append(drift)
        drift_labels.append(f"{m1}->{m2}")

        print(f"{m1} → {m2} | Drift: {drift:.4f}")

        if drift > 0.1:
            print("  → Significant shift detected")
        else:
            print("  → Minor variation")

    # 🔹 Plot drift signal
    plot_drift(drift_labels, drift_values)


if __name__ == "__main__":
    main()