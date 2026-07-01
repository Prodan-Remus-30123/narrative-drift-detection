import json
from pathlib import Path

import matplotlib.pyplot as plt


ANALYSIS_RESULTS_PATH = Path(
    "outputs/20260616_122240/results/analysis_results.json"
)

OUTPUT_DIR = Path(
    "plots/semantic_drift_transitions"
)


def load_analysis_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_semantic_drift(source_result):
    semantic = source_result.get("semantic_drift", {})

    labels = (
        semantic.get("labels")
        or semantic.get("transitions")
        or []
    )

    values = (
        semantic.get("values")
        or semantic.get("drift_values")
        or []
    )

    return labels, values


def get_threshold(source_result, values):
    semantic = source_result.get("semantic_drift", {})

    threshold = (
        semantic.get("threshold")
        or semantic.get("dynamic_threshold")
        or semantic.get("significance_threshold")
    )

    if threshold is not None:
        return threshold

    if not values:
        return None

    # fallback: mean + std
    mean_value = sum(values) / len(values)
    variance = sum(
        (value - mean_value) ** 2
        for value in values
    ) / len(values)

    return mean_value + variance ** 0.5


def get_significant_transitions(source_result):
    representative = source_result.get(
        "representative_evidence",
        {}
    )

    transitions = representative.get(
        "transitions",
        []
    )

    significant = set()

    for item in transitions:
        if item.get("is_significant"):
            transition = item.get("transition")
            if transition:
                significant.add(transition)

    change_points = (
        source_result
        .get("change_points", {})
        .get("semantic_drift", {})
        .get("change_points", [])
    )

    for point in change_points:
        label = point.get("label")
        if label:
            significant.add(label)

    return significant


def plot_source(source, source_result):
    labels, values = get_semantic_drift(source_result)

    if not labels or not values:
        print(f"Skipped {source}: no semantic drift data")
        return

    threshold = get_threshold(source_result, values)
    significant = get_significant_transitions(source_result)

    x = list(range(len(labels)))

    plt.figure(figsize=(16, 6))

    plt.plot(
        x,
        values,
        marker="o",
        linewidth=2,
        label="Semantic drift"
    )

    if threshold is not None:
        plt.axhline(
            threshold,
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold ({threshold:.4f})"
        )

    for i, label in enumerate(labels):
        if label in significant:
            plt.scatter(
                i,
                values[i],
                s=120,
                marker="D",
                label="Detected transition"
                if i == 0 else None
            )

            plt.annotate(
                label,
                (i, values[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                rotation=30
            )

    plt.title(
        f"Example Semantic Drift Signal with Detected Narrative Transitions — {source}"
    )

    plt.xlabel("Temporal transition")
    plt.ylabel("Semantic drift")
    plt.xticks(
        x,
        labels,
        rotation=45,
        ha="right"
    )

    plt.legend()
    plt.tight_layout()

    safe_source = (
        source
        .replace(".", "_")
        .replace("/", "_")
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = OUTPUT_DIR / (
        f"semantic_drift_detected_transitions_{safe_source}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")


def main():
    results = load_analysis_results(
        ANALYSIS_RESULTS_PATH
    )

    for source, source_result in results.items():
        if source.startswith("__"):
            continue

        plot_source(
            source,
            source_result
        )


if __name__ == "__main__":
    main()