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
import numpy as np


MIN_VERB_COUNT = 3
TOP_K_VERBS = 30
MIN_SHARED_VERBS = 2
MIN_VERB_LENGTH = 3
MIN_GLOBAL_VERB_COUNT = 10

def is_valid_verb(verb):
    if not isinstance(verb, str):
        return False

    verb = verb.strip().lower()

    if len(verb) < MIN_VERB_LENGTH:
        return False

    if not verb.isalpha():
        return False

    return True

def build_entity_vectors(grouped_texts):

    period_vectors = {}

    for period, texts in grouped_texts.items():

        entity_data = analyze_entities(
            texts
        )

        vectors = {}

        for entity, stats in entity_data.items():

            clean_verbs = {
                verb: count
                for verb, count in stats["verbs"].items()
                if is_valid_verb(verb)
            }

            total_verbs = sum(clean_verbs.values())

            if total_verbs < MIN_VERB_COUNT:
                continue

            vectors[entity] = clean_verbs

        period_vectors[period] = vectors

    return period_vectors

def compute_verb_document_frequency(period_vectors):
    verb_df = defaultdict(int)
    verb_global_count = defaultdict(int)
    total_entity_vectors = 0

    for period, entity_vectors in period_vectors.items():

        for entity, verb_vector in entity_vectors.items():

            total_entity_vectors += 1

            for verb, count in verb_vector.items():
                verb_df[verb] += 1
                verb_global_count[verb] += count

    return verb_df, verb_global_count, total_entity_vectors

def weight_entity_vectors(
    period_vectors,
    verb_df,
    verb_global_count,
    total_entity_vectors
):

    
    weighted_vectors = {}

    for period, entity_vectors in period_vectors.items():

        weighted_vectors[period] = {}

        for entity, verb_vector in entity_vectors.items():

            weighted_verb_vector = {}

            for verb, count in verb_vector.items():

                if verb_global_count[verb] < MIN_GLOBAL_VERB_COUNT:
                    continue

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

def shared_verb_similarity(vector_a, vector_b):
    shared_keys = set(vector_a.keys()).intersection(
        set(vector_b.keys())
    )

    if len(shared_keys) < MIN_SHARED_VERBS:
        return None

    a = []
    b = []

    for key in sorted(shared_keys):
        a.append(vector_a.get(key, 0))
        b.append(vector_b.get(key, 0))

    if sum(a) == 0 or sum(b) == 0:
        return None

    similarity = 1 - cosine(a, b)

    if math.isnan(similarity):
        return None

    return similarity


def vocabulary_turnover(vector_a, vector_b):
    before_keys = set(vector_a.keys())
    after_keys = set(vector_b.keys())

    union = before_keys.union(after_keys)

    if not union:
        return None

    intersection = before_keys.intersection(after_keys)

    jaccard_similarity = len(intersection) / len(union)

    return 1 - jaccard_similarity

def normalize_distribution(vector):
    total = sum(vector.values())

    if total == 0:
        return {}

    return {
        key: value / total
        for key, value in vector.items()
    }


def jensen_shannon_distance(vector_a, vector_b):
    keys = sorted(
        set(vector_a.keys()).union(vector_b.keys())
    )

    if not keys:
        return None

    p = np.array([
        vector_a.get(key, 0.0)
        for key in keys
    ], dtype=float)

    q = np.array([
        vector_b.get(key, 0.0)
        for key in keys
    ], dtype=float)

    if p.sum() == 0 or q.sum() == 0:
        return None

    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)

    def kl_divergence(a, b):
        mask = a > 0
        return np.sum(
            a[mask] * np.log2(a[mask] / b[mask])
        )

    js_divergence = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

    return float(np.sqrt(js_divergence))

def compare_periods(period_vectors):

    periods = sorted(period_vectors.keys())

    for i in range(len(periods) - 1):

        period_a = periods[i]
        period_b = periods[i + 1]

        print("\n\n====================")
        print(f"{period_a} -> {period_b}")
        print("====================")

        vectors_a = period_vectors[period_a]
        vectors_b = period_vectors[period_b]

        shared_entities = set(vectors_a.keys()).intersection(
            vectors_b.keys()
        )

        for entity in sorted(shared_entities):

            shared_similarity = shared_verb_similarity(
                vectors_a[entity],
                vectors_b[entity]
            )

            turnover = vocabulary_turnover(
                vectors_a[entity],
                vectors_b[entity]
            )

            if turnover is None:
                continue

            print(f"\nEntity: {entity}")
            print(f"Vocabulary turnover: {turnover:.3f}")

            if shared_similarity is not None:
                print(f"Shared verb similarity: {shared_similarity:.3f}")
            else:
                print("Shared verb similarity: insufficient shared verbs")

            print(f"{period_a} verbs:")
            print(vectors_a[entity])

            print(f"{period_b} verbs:")
            print(vectors_b[entity])

def compute_entity_drift(grouped_texts):

    period_vectors = build_entity_vectors(grouped_texts)
    raw_period_vectors = period_vectors

    verb_df, verb_global_count, total_entity_vectors = compute_verb_document_frequency(
        period_vectors
    )

    # print("\n=== GLOBAL VERB FILTER ===")

    kept = sum(
        1
        for count in verb_global_count.values()
        if count >= MIN_GLOBAL_VERB_COUNT
    )

    # print(
    #     f"Kept verbs: {kept} / "
    #     f"{len(verb_global_count)}"
    # )

    period_vectors = weight_entity_vectors(
        period_vectors,
        verb_df,
        verb_global_count,
        total_entity_vectors
    )

    for period, entities in period_vectors.items():
        for entity, verbs in entities.items():

            if "m" in verbs:

                print("\n====================")
                print("FOUND VERB M")
                print("====================")
                print("period:", period)
                print("entity:", entity)
                print("verbs:", verbs)

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

        # print(
        #     f"{transition} "
        #     f"shared entities: "
        #     f"{len(shared_entities)}"
        # )

        for entity in shared_entities:
            before_verbs = set(vectors_a[entity].keys())
            after_verbs = set(vectors_b[entity].keys())

            shared_verbs = before_verbs.intersection(after_verbs)

            shared_verbs_list = sorted(list(shared_verbs))
            before_only = sorted(list(before_verbs - shared_verbs))
            after_only = sorted(list(after_verbs - shared_verbs))

            shared_similarity = shared_verb_similarity(
                vectors_a[entity],
                vectors_b[entity]
            )

            framing_drift_js = jensen_shannon_distance(
                raw_period_vectors[period_a][entity],
                raw_period_vectors[period_b][entity]
            )

            turnover = vocabulary_turnover(
                vectors_a[entity],
                vectors_b[entity]
            )

            if turnover is None:
                continue

            if turnover >= 0.75:
                drift_class = "high vocabulary turnover"
            elif turnover >= 0.4:
                drift_class = "moderate vocabulary turnover"
            else:
                drift_class = "low vocabulary turnover"


            drift_results[transition][entity] = {
                "drift": None,
                "drift_class": None,
                "shared_similarity": shared_similarity,
                "vocabulary_turnover": turnover,
                "framing_drift_js": framing_drift_js,

                "before": vectors_a[entity],
                "after": vectors_b[entity],

                "shared_verbs": len(shared_verbs),

                "shared_verbs_list": shared_verbs_list,

                "before_only_verbs": before_only,

                "after_only_verbs": after_only
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

            turnover = stats.get("vocabulary_turnover")

            if turnover is None:
                continue

            importance_scores[entity]["drifts"].append(turnover)

            importance_scores[entity]["transitions"] += 1

    final_scores = {}

    for entity, data in importance_scores.items():

        if len(data["drifts"]) == 0:
            continue

        avg_drift = sum(data["drifts"]) / len(data["drifts"])

        transition_count = data["transitions"]
        salience = salience_totals.get(entity, 1)
        importance = (avg_drift * math.log(salience + 1) * transition_count)

        final_scores[entity] = importance

    return final_scores


def main():
    df = load_full_articles_with_dates()
    grouped = group_articles_by_period(df)

    period_vectors = build_entity_vectors(grouped)
    verb_df, verb_global_count, total_entity_vectors = compute_verb_document_frequency(period_vectors)

    period_vectors = weight_entity_vectors(period_vectors, verb_df, verb_global_count, total_entity_vectors)
    compare_periods(period_vectors) 

if __name__ == "__main__":

    main()