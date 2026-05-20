"""
entity_framing_drift.py

Temporal entity framing drift analysis.
"""

from collections import defaultdict
from scipy.spatial.distance import cosine

from database import (load_full_articles_with_dates)
from temporal_entity_analysis import (group_articles_by_period)
from entities import (analyze_entities)
import math


MIN_VERB_COUNT = 5
TOP_K_VERBS = 15
MIN_SHARED_VERBS = 2


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

def compute_verb_document_frequency(period_vectors):
    verb_df = defaultdict(int)
    total_entity_vectors = 0

    for period, entity_vectors in period_vectors.items():

        for entity, verb_vector in entity_vectors.items():

            total_entity_vectors += 1

            for verb in verb_vector.keys():
                verb_df[verb] += 1

    return verb_df, total_entity_vectors

def weight_entity_vectors(
    period_vectors,
    verb_df,
    total_entity_vectors
):

    weighted_vectors = {}

    for period, entity_vectors in period_vectors.items():

        weighted_vectors[period] = {}

        for entity, verb_vector in entity_vectors.items():

            weighted_verb_vector = {}

            for verb, count in verb_vector.items():

                idf = math.log(
                    (1 + total_entity_vectors) /
                    (1 + verb_df[verb])
                ) + 1

                weighted_verb_vector[verb] = count * idf

            top_verbs = sorted(
                weighted_verb_vector.items(),
                key=lambda x: x[1],
                reverse=True
            )[:TOP_K_VERBS]

            weighted_vectors[period][entity] = dict(top_verbs)

    return weighted_vectors

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

            shared_verbs = set(vectors_a[entity].keys()).intersection(set(vectors_b[entity].keys()))

            if len(shared_verbs) < 2:
                continue

            similarity = vector_similarity(vectors_a[entity], vectors_b[entity])

            drift = 1 - similarity

            # if drift < 0.4:
            #     continue

            print(f"\nEntity: {entity}")
            print(f"Framing drift: " f"{drift:.3f}")
            print(f"{period_a} verbs:")
            print(vectors_a[entity])
            print(f"{period_b} verbs:")
            print(vectors_b[entity])

def compute_entity_drift(grouped_texts):

    period_vectors = build_entity_vectors(grouped_texts)

    verb_df, total_entity_vectors = compute_verb_document_frequency(
        period_vectors
    )

    period_vectors = weight_entity_vectors(
        period_vectors,
        verb_df,
        total_entity_vectors
    )

    periods = sorted(period_vectors.keys())

    drift_results = {}

    for i in range(
        len(periods) - 1
    ):

        period_a = periods[i]
        period_b = periods[i + 1]

        transition = (
            f"{period_a}->{period_b}"
        )

        drift_results[transition] = {}
        vectors_a = period_vectors[period_a]

        vectors_b = period_vectors[period_b]

        shared_entities = set(
            vectors_a.keys()
        ).intersection(
            vectors_b.keys()
        )

        print(
            f"{transition} "
            f"shared entities: "
            f"{len(shared_entities)}"
        )

        for entity in shared_entities:

            shared_verbs = set(
                vectors_a[entity].keys()
            ).intersection(
                set(vectors_b[entity].keys())
            )

            if len(shared_verbs) < MIN_SHARED_VERBS:
                continue

            similarity = vector_similarity(

                vectors_a[entity],

                vectors_b[entity]
            )

            drift = 1 - similarity

            if drift >= 0.7:
                drift_class = "major reframing"
            elif drift >= 0.4:
                drift_class = "moderate reframing"
            else:
                drift_class = "stable framing"

            drift_results[transition][entity] = {
                "drift": drift,
                "drift_class": drift_class,
                "before": vectors_a[entity],
                "after": vectors_b[entity],
                "shared_verbs": len(shared_verbs)
            }

    return drift_results

def compute_entity_importance(
    framing_drift,
    salience_totals
):

    importance_scores = {}

    for transition, entities in framing_drift.items():

        for entity, stats in entities.items():

            if entity not in importance_scores:

                importance_scores[entity] = {
                    "drifts": [],
                    "transitions": 0
                }

            importance_scores[entity]["drifts"].append(
                stats["drift"]
            )

            importance_scores[entity]["transitions"] += 1

    final_scores = {}

    for entity, data in importance_scores.items():

        avg_drift = (sum(data["drifts"])/len(data["drifts"]))

        transition_count = data["transitions"]
        salience = salience_totals.get(entity, 1)
        importance = (avg_drift * math.log(salience + 1) * transition_count)

        final_scores[entity] = importance

    return final_scores


def main():
    df = load_full_articles_with_dates()
    grouped = group_articles_by_period(df)

    period_vectors = build_entity_vectors(grouped)
    verb_df, total_entity_vectors = compute_verb_document_frequency(period_vectors)

    period_vectors = weight_entity_vectors(period_vectors, verb_df, total_entity_vectors)
    compare_periods(period_vectors) 

if __name__ == "__main__":

    main()