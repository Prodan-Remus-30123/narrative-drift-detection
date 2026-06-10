import json
from collections import Counter

from database import load_full_articles
from temporal_entity_analysis import group_articles_by_period
from preprocessing import preprocess_corpus
from entities import analyze_entities


OUTPUT_FILE = "unique_verbs.json"


def main():
    df = load_full_articles()

    all_verbs = Counter()

    for source in df["source"].unique():
        source_df = df[df["source"] == source]
        grouped = group_articles_by_period(source_df)

        for period, texts in grouped.items():
            print(f"Processing {source} | {period}")

            texts = preprocess_corpus(texts)
            entity_data = analyze_entities(texts)

            for entity, stats in entity_data.items():
                for verb, count in stats["verbs"].items():
                    all_verbs[verb] += count

    output = [
        {
            "verb": verb,
            "count": count
        }
        for verb, count in all_verbs.most_common()
    ]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {OUTPUT_FILE}")
    print(f"Unique verbs: {len(output)}")


if __name__ == "__main__":
    main()