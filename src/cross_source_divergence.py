import itertools
import numpy as np


def _cosine_distance(vec_a, vec_b):
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return None

    similarity = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    return 1.0 - similarity


def build_source_period_embedding_table(analysis_results):
    """
    Builds:
    {
        source: {
            period: embedding_vector
        }
    }
    from semantic_embedding_trajectory.
    """

    table = {}

    for source, result in analysis_results.items():
        if source.startswith("__"):
            continue

        trajectory = result.get("semantic_embedding_trajectory", {})
        labels = trajectory.get("labels", [])
        vectors = trajectory.get("vectors", [])

        if not labels or not vectors:
            continue

        table[source] = {
            str(label): vector
            for label, vector in zip(labels, vectors)
        }

    return table


def compute_cross_source_divergence(analysis_results):
    """
    Computes pairwise source divergence per shared period using cosine distance
    between aggregated source-period embeddings.
    """

    table = build_source_period_embedding_table(analysis_results)

    sources = sorted(table.keys())

    pairwise_results = []
    timeline = []

    for source_a, source_b in itertools.combinations(sources, 2):
        periods_a = set(table[source_a].keys())
        periods_b = set(table[source_b].keys())

        shared_periods = sorted(periods_a & periods_b)

        values = []

        for period in shared_periods:
            distance = _cosine_distance(
                table[source_a][period],
                table[source_b][period]
            )

            if distance is None:
                continue

            values.append(distance)

            timeline.append({
                "source_a": source_a,
                "source_b": source_b,
                "pair": f"{source_a} ↔ {source_b}",
                "period": period,
                "divergence": distance
            })

        if not values:
            continue

        pairwise_results.append({
            "source_a": source_a,
            "source_b": source_b,
            "pair": f"{source_a} ↔ {source_b}",
            "shared_periods": len(values),
            "mean_divergence": float(np.mean(values)),
            "max_divergence": float(np.max(values)),
            "min_divergence": float(np.min(values)),
            "divergence_volatility": float(np.std(values)),
            "peak_period": shared_periods[int(np.argmax(values))],
            "values": values
        })

    ranked_pairs = sorted(
        pairwise_results,
        key=lambda x: x["mean_divergence"],
        reverse=True
    )

    return {
        "sources": sources,
        "pairwise": ranked_pairs,
        "timeline": timeline
    }


def print_cross_source_divergence_summary(result, top_n=10):
    print("\n=== CROSS-SOURCE NARRATIVE DIVERGENCE ===")

    sources = result.get("sources", [])

    if len(sources) < 2:
        print("Skipped: fewer than two sources with embedding trajectories.")
        return

    print("Sources:")
    for source in sources:
        print(f"- {source}")

    print("\nMost divergent outlet pairs:")

    for item in result.get("pairwise", [])[:top_n]:
        print(
            f"- {item['pair']} | "
            f"mean={item['mean_divergence']:.4f}, "
            f"max={item['max_divergence']:.4f}, "
            f"volatility={item['divergence_volatility']:.4f}, "
            f"peak={item['peak_period']}, "
            f"periods={item['shared_periods']}"
        )

    if result.get("pairwise"):
        most_aligned = sorted(
            result["pairwise"],
            key=lambda x: x["mean_divergence"]
        )[0]

        print("\nMost aligned outlet pair:")
        print(
            f"- {most_aligned['pair']} | "
            f"mean={most_aligned['mean_divergence']:.4f}, "
            f"max={most_aligned['max_divergence']:.4f}, "
            f"volatility={most_aligned['divergence_volatility']:.4f}"
        )