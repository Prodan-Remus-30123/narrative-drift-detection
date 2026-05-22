from embeddings import EmbeddingModel
from drift import compute_cosine_drift, compute_dynamic_threshold, classify_drift
from database import load_full_articles
from preprocessing import preprocess_corpus
from temporal_entity_analysis import group_articles_by_period
from utils.period_sorting import sort_period_key


def get_semantic_drift(source):

    df = load_full_articles()
    source_df = df[df["source"] == source]

    grouped = group_articles_by_period(source_df)

    for period in grouped:
        grouped[period] = preprocess_corpus(grouped[period])

    model = EmbeddingModel()

    aggregated_vectors = []

    for period in sorted(grouped.keys(), key=sort_period_key):

        embeddings = model.encode_documents(grouped[period])
        vector = model.aggregate_embeddings(embeddings)

        aggregated_vectors.append((period, vector))

    results = []

    drift_values = []

    for i in range(len(aggregated_vectors) - 1):

        p1, v1 = aggregated_vectors[i]
        p2, v2 = aggregated_vectors[i + 1]

        drift = compute_cosine_drift(v1, v2)
        drift_values.append(drift)

        results.append({
            "transition": f"{p1}->{p2}",
            "drift": float(drift)
        })

    threshold = compute_dynamic_threshold(
        drift_values,
        method="median_mad"
    )

    for item in results:
        item["threshold"] = float(threshold)
        item["classification"] = classify_drift(
            item["drift"],
            threshold
        )

    return {
        "source": source,
        "semantic_drift": results
    }