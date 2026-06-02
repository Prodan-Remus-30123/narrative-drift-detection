"""
signature_explanation.py

Explains why two narrative signatures are similar or different.

Uses:
- standardized numeric feature differences
- semantic frame label comparison
- source-level narrative signature summaries
"""

import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from narrative_signatures import build_all_narrative_signatures
from signature_comparison import SIGNATURE_FEATURES
from semantic_signature_embedding import build_semantic_signature_embeddings


def load_analysis_results(path="analysis_results.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_signature_lookup(analysis_results):
    signatures = build_all_narrative_signatures(
        analysis_results
    )

    return {
        signature["source"]: signature
        for signature in signatures
    }


def build_standardized_feature_table(signatures):
    rows = []

    for signature in signatures:
        row = {
            "source": signature["source"]
        }

        for feature in SIGNATURE_FEATURES:
            value = signature.get(feature, 0.0)

            if value is None:
                value = 0.0

            row[feature] = float(value)

        rows.append(row)

    df = pd.DataFrame(rows)

    scaler = StandardScaler()

    scaled_values = scaler.fit_transform(
        df[SIGNATURE_FEATURES]
    )

    scaled_df = pd.DataFrame(
        scaled_values,
        columns=SIGNATURE_FEATURES
    )

    scaled_df.insert(
        0,
        "source",
        df["source"]
    )

    return df, scaled_df


def explain_numeric_difference(
    source_a,
    source_b,
    scaled_df,
    top_n=8
):
    row_a = scaled_df[
        scaled_df["source"] == source_a
    ].iloc[0]

    row_b = scaled_df[
        scaled_df["source"] == source_b
    ].iloc[0]

    differences = []

    for feature in SIGNATURE_FEATURES:
        value_a = row_a[feature]
        value_b = row_b[feature]

        diff = value_a - value_b

        differences.append({
            "feature": feature,
            "source_a_value": float(value_a),
            "source_b_value": float(value_b),
            "absolute_difference": float(abs(diff)),
            "direction": (
                source_a
                if diff > 0
                else source_b
            )
        })

    return sorted(
        differences,
        key=lambda x: x["absolute_difference"],
        reverse=True
    )[:top_n]


def extract_top_frames(signature, top_n=5):
    frames = signature.get(
        "top_volatile_frames",
        []
    )

    return frames[:top_n]


def explain_semantic_frame_overlap(
    signature_a,
    signature_b
):
    frames_a = set(
        extract_top_frames(signature_a)
    )

    frames_b = set(
        extract_top_frames(signature_b)
    )

    shared = frames_a.intersection(frames_b)

    only_a = frames_a.difference(frames_b)
    only_b = frames_b.difference(frames_a)

    union = frames_a.union(frames_b)

    overlap_score = (
        len(shared) / len(union)
        if union
        else 0.0
    )

    return {
        "shared_frames": sorted(shared),
        "only_source_a": sorted(only_a),
        "only_source_b": sorted(only_b),
        "frame_overlap_score": overlap_score
    }


def explain_embedding_similarity(
    signatures,
    source_a,
    source_b
):
    semantic_embeddings = build_semantic_signature_embeddings(
        signatures
    )

    embedding_a = semantic_embeddings.get(source_a)
    embedding_b = semantic_embeddings.get(source_b)

    if embedding_a is None or embedding_b is None:
        return None

    similarity = cosine_similarity(
        [embedding_a],
        [embedding_b]
    )[0][0]

    return float(similarity)


def classify_relationship(similarity):
    if similarity >= 0.65:
        return "strongly similar"

    if similarity >= 0.30:
        return "moderately similar"

    if similarity >= -0.30:
        return "weakly related / mixed"

    if similarity >= -0.65:
        return "moderately different"

    return "strongly different"


def explain_pair(
    source_a,
    source_b,
    analysis_path="analysis_results.json",
    top_n=8
):
    analysis_results = load_analysis_results(
        analysis_path
    )

    signatures = build_all_narrative_signatures(
        analysis_results
    )

    lookup = {
        signature["source"]: signature
        for signature in signatures
    }

    if source_a not in lookup:
        raise ValueError(
            f"Unknown source: {source_a}"
        )

    if source_b not in lookup:
        raise ValueError(
            f"Unknown source: {source_b}"
        )

    raw_df, scaled_df = build_standardized_feature_table(
        signatures
    )

    numeric_differences = explain_numeric_difference(
        source_a,
        source_b,
        scaled_df,
        top_n=top_n
    )

    semantic_overlap = explain_semantic_frame_overlap(
        lookup[source_a],
        lookup[source_b]
    )

    embedding_similarity = explain_embedding_similarity(
        signatures,
        source_a,
        source_b
    )

    relationship = classify_relationship(
        embedding_similarity
        if embedding_similarity is not None
        else 0.0
    )

    explanation = {
        "source_a": source_a,
        "source_b": source_b,
        "semantic_embedding_similarity": embedding_similarity,
        "relationship": relationship,
        "top_numeric_differences": numeric_differences,
        "semantic_frame_overlap": semantic_overlap
    }

    return explanation


def print_pair_explanation(explanation):
    print(
        "\n=== NARRATIVE SIGNATURE EXPLANATION ==="
    )

    print(
        f"\nPair: {explanation['source_a']} "
        f"vs {explanation['source_b']}"
    )

    print(
        "Semantic embedding similarity: "
        f"{explanation['semantic_embedding_similarity']:.3f}"
    )

    print(
        "Relationship: "
        f"{explanation['relationship']}"
    )

    print("\nTop numeric differences:")

    for item in explanation["top_numeric_differences"]:
        print(
            f"- {item['feature']}: "
            f"{explanation['source_a']}={item['source_a_value']:.3f}, "
            f"{explanation['source_b']}={item['source_b_value']:.3f}, "
            f"abs_diff={item['absolute_difference']:.3f}, "
            f"higher={item['direction']}"
        )

    overlap = explanation["semantic_frame_overlap"]

    print("\nSemantic frame overlap:")
    print(
        f"Overlap score: "
        f"{overlap['frame_overlap_score']:.3f}"
    )

    print("\nShared frames:")
    for frame in overlap["shared_frames"]:
        print(f"- {frame}")

    print(
        f"\nFrames unique to "
        f"{explanation['source_a']}:"
    )

    for frame in overlap["only_source_a"]:
        print(f"- {frame}")

    print(
        f"\nFrames unique to "
        f"{explanation['source_b']}:"
    )

    for frame in overlap["only_source_b"]:
        print(f"- {frame}")


def explain_all_pairs(
    analysis_path="analysis_results.json"
):
    analysis_results = load_analysis_results(
        analysis_path
    )

    signatures = build_all_narrative_signatures(
        analysis_results
    )

    sources = [
        signature["source"]
        for signature in signatures
    ]

    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            explanation = explain_pair(
                sources[i],
                sources[j],
                analysis_path=analysis_path
            )

            print_pair_explanation(
                explanation
            )


if __name__ == "__main__":
    explanation = explain_pair(
    "bbc.co.uk",
    "theguardian.com"
)

    print_pair_explanation(
        explanation
    )