"""
Analyze extracted verb frequency distribution.

Input:
- unique_verbs.json

Output:
- frequency threshold statistics
- verbs_above_threshold_*.json
"""

import json
from pathlib import Path


INPUT_FILE = Path("unique_verbs.json")

THRESHOLDS = [1, 2, 3, 5, 10, 20, 50]


def load_verbs():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    verbs = load_verbs()

    total_unique = len(verbs)
    total_occurrences = sum(item["count"] for item in verbs)

    print("\n====================")
    print("VERB FREQUENCY ANALYSIS")
    print("====================")

    print(f"Total unique verbs: {total_unique}")
    print(f"Total verb occurrences: {total_occurrences}")

    print("\n====================")
    print("THRESHOLD COVERAGE")
    print("====================")

    for threshold in THRESHOLDS:
        kept = [
            item
            for item in verbs
            if item["count"] >= threshold
        ]

        kept_unique = len(kept)
        kept_occurrences = sum(item["count"] for item in kept)

        unique_pct = (
            kept_unique / total_unique * 100
            if total_unique else 0
        )

        occurrence_pct = (
            kept_occurrences / total_occurrences * 100
            if total_occurrences else 0
        )

        print(
            f"count >= {threshold:<2} | "
            f"unique verbs: {kept_unique:<5} "
            f"({unique_pct:6.2f}%) | "
            f"occurrences kept: {kept_occurrences:<7} "
            f"({occurrence_pct:6.2f}%)"
        )

        output_file = Path(
            f"verbs_count_ge_{threshold}.json"
        )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                kept,
                f,
                indent=2,
                ensure_ascii=False
            )

    print("\n====================")
    print("TOP 50 VERBS")
    print("====================")

    for item in verbs[:50]:
        print(f"{item['verb']:<25} {item['count']}")

    print("\n====================")
    print("LOW FREQUENCY EXAMPLES")
    print("====================")

    low_frequency = [
        item
        for item in verbs
        if item["count"] <= 2
    ]

    for item in low_frequency[:100]:
        print(f"{item['verb']:<25} {item['count']}")


if __name__ == "__main__":
    main()