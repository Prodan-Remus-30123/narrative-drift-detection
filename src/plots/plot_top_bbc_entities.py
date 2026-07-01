import json
from pathlib import Path

import matplotlib.pyplot as plt


ANALYSIS_RESULTS_PATH = Path("outputs/20260616_122240/results/analysis_results.json")
SOURCE = "bbc.co.uk"
TOP_N = 15

OUTPUT_DIR = Path("plots/entity_importance")


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_entity_importance(source_result):
    ecosystem = source_result.get("entity_ecosystem", {})

    entity_scores = {}

    for entity, data in ecosystem.items():
        if not isinstance(data, dict):
            continue

        score = data.get("importance")

        if score is None:
            continue

        try:
            entity_scores[entity] = float(score)
        except Exception:
            continue

    return entity_scores


def plot_top_entities(entity_scores):
    if not entity_scores:
        print("No entity importance data found.")
        return
    
    top_entities = sorted(
        entity_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:TOP_N]

    entities = [item[0] for item in top_entities][::-1]
    scores = [item[1] for item in top_entities][::-1]

    plt.figure(figsize=(10, 7))

    plt.barh(
        entities,
        scores
    )

    plt.title(
        f"Top {TOP_N} BBC Entities by Importance"
    )

    plt.xlabel("Entity importance score")
    plt.ylabel("Entity")

    plt.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = OUTPUT_DIR / (
        f"top_{TOP_N}_entities_by_importance_{SOURCE.replace('.', '_')}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")


def main():
    results = load_results(ANALYSIS_RESULTS_PATH)

    source_result = results.get(SOURCE)

    if source_result is None:
        print(f"Source not found: {SOURCE}")
        print("Available sources:")
        for key in results.keys():
            print("-", key)
        return

    entity_scores = extract_entity_importance(source_result)

    plot_top_entities(entity_scores)


if __name__ == "__main__":
    main()