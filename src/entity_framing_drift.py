"""
entity_framing_drift.py

Temporal entity framing drift analysis.
"""

from collections import defaultdict
from scipy.spatial.distance import cosine

from database import (load_full_articles_with_dates)
from temporal_entity_analysis import (group_articles_by_period)
from entities import (analyze_entities)


MIN_VERB_COUNT = 5

def build_entity_vectors(
    grouped_texts
):

    period_vectors = {}

    for period, texts in grouped_texts.items():

        entity_data = analyze_entities(
            texts
        )

        vectors = {}

        for entity, stats in entity_data.items():

            total_verbs = sum(
                stats["verbs"].values()
            )

            if total_verbs < MIN_VERB_COUNT:
                continue

            vectors[entity] = dict(
                stats["verbs"]
            )

        period_vectors[period] = vectors

    return period_vectors


def vector_similarity(
    vector_a,
    vector_b
):

    all_keys = set(
        vector_a.keys()
    ).union(
        vector_b.keys()
    )

    a = []
    b = []

    for key in all_keys:

        a.append(
            vector_a.get(key, 0)
        )

        b.append(
            vector_b.get(key, 0)
        )

    if sum(a) == 0 or sum(b) == 0:
        return 0

    return 1 - cosine(a, b)


def compare_periods(
    period_vectors
):

    periods = sorted(
        period_vectors.keys()
    )

    for i in range(
        len(periods) - 1
    ):

        period_a = periods[i]
        period_b = periods[i + 1]

        print(
            f"\n\n===================="
        )

        print(
            f"{period_a} -> {period_b}"
        )

        print(
            f"===================="
        )

        vectors_a = period_vectors[
            period_a
        ]

        vectors_b = period_vectors[
            period_b
        ]

        shared_entities = set(
            vectors_a.keys()
        ).intersection(
            vectors_b.keys()
        )

        for entity in sorted(shared_entities):

            similarity = vector_similarity(

                vectors_a[entity],

                vectors_b[entity]
            )

            drift = 1 - similarity

            if drift < 0.4:
                continue

            print(f"\nEntity: {entity}")
            print(f"Framing drift: " f"{drift:.3f}")
            print(f"{period_a} verbs:")
            print(vectors_a[entity])
            print(f"{period_b} verbs:")
            print(vectors_b[entity])

def compute_entity_drift(
    grouped_texts
):

    period_vectors = build_entity_vectors(
        grouped_texts
    )

    periods = sorted(
        period_vectors.keys()
    )

    drift_results = {}

    for i in range(
        len(periods) - 1
    ):

        period_a = periods[i]
        period_b = periods[i + 1]

        transition = (
            f"{period_a}->{period_b}"
        )

        drift_results[
            transition
        ] = {}

        vectors_a = period_vectors[
            period_a
        ]

        vectors_b = period_vectors[
            period_b
        ]

        shared_entities = set(
            vectors_a.keys()
        ).intersection(
            vectors_b.keys()
        )

        for entity in shared_entities:

            similarity = vector_similarity(

                vectors_a[entity],

                vectors_b[entity]
            )

            drift = 1 - similarity

            drift_results[
                transition
            ][entity] = {

                "drift": drift,

                "before":
                    vectors_a[entity],

                "after":
                    vectors_b[entity]
            }

    return drift_results

def main():
    df = load_full_articles_with_dates()
    grouped = group_articles_by_period(df)
    period_vectors = build_entity_vectors(grouped)
    compare_periods(period_vectors)

if __name__ == "__main__":

    main()