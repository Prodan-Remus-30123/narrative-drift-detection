"""
temporal_frame_evolution.py

Tracks temporal evolution of latent narrative frames produced by latent_frames.py.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from embedding_model_registry import get_embedding_model
from utils.period_sorting import sort_period_key


STABLE_THRESHOLD = 0.85
TRANSFORMED_THRESHOLD = 0.65
WEAK_THRESHOLD = 0.50


def build_cluster_embeddings(clusters):
    model = get_embedding_model()

    cluster_embeddings = {}

    for cluster_id, data in clusters.items():
        verbs = data.get("verbs", [])

        if not verbs:
            continue

        embeddings = model.encode_documents(verbs)

        if len(embeddings) == 0:
            continue

        cluster_embeddings[cluster_id] = np.mean(
            np.array(embeddings),
            axis=0
        )

    return cluster_embeddings


def classify_transition(similarity):
    if similarity >= STABLE_THRESHOLD:
        return "stable"

    if similarity >= TRANSFORMED_THRESHOLD:
        return "transformed"

    if similarity >= WEAK_THRESHOLD:
        return "weak_transition"

    return "discontinuous"


def build_period_active_frames(period_frames):
    active = {}

    for period, frames in period_frames.items():
        active[period] = set(frames.keys())

    return active


def build_temporal_frame_evolution(latent_result):
    clusters = latent_result["clusters"]
    period_frames = latent_result["period_frames"]

    cluster_embeddings = build_cluster_embeddings(clusters)
    active_frames = build_period_active_frames(period_frames)

    periods = sorted(
        period_frames.keys(),
        key=sort_period_key
    )

    transitions = []

    for i in range(len(periods) - 1):
        from_period = periods[i]
        to_period = periods[i + 1]

        from_frames = active_frames[from_period]
        to_frames = active_frames[to_period]

        matched_to_frames = set()

        for from_frame_id in from_frames:
            if from_frame_id not in cluster_embeddings:
                continue

            best_match = None
            best_similarity = -1.0

            from_embedding = cluster_embeddings[from_frame_id]

            for to_frame_id in to_frames:
                if to_frame_id not in cluster_embeddings:
                    continue

                to_embedding = cluster_embeddings[to_frame_id]

                similarity = cosine_similarity(
                    [from_embedding],
                    [to_embedding]
                )[0][0]

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = to_frame_id

            if best_match is None:
                transitions.append({
                    "from_period": from_period,
                    "to_period": to_period,
                    "from_frame_id": from_frame_id,
                    "to_frame_id": None,
                    "from_frame": clusters[from_frame_id]["label"],
                    "to_frame": None,
                    "from_share": period_frames[from_period][from_frame_id]["share"],
                    "to_share": 0,
                    "similarity": None,
                    "evolution_type": "disappeared"
                })

                continue

            matched_to_frames.add(best_match)

            transitions.append({
                "from_period": from_period,
                "to_period": to_period,
                "from_frame_id": from_frame_id,
                "to_frame_id": best_match,
                "from_frame": clusters[from_frame_id]["label"],
                "to_frame": clusters[best_match]["label"],
                "from_share": period_frames[from_period][from_frame_id]["share"],
                "to_share": period_frames[to_period][best_match]["share"],
                "similarity": float(best_similarity),
                "evolution_type": classify_transition(best_similarity)
            })

        for to_frame_id in to_frames:
            if to_frame_id not in matched_to_frames:
                transitions.append({
                    "from_period": from_period,
                    "to_period": to_period,
                    "from_frame_id": None,
                    "to_frame_id": to_frame_id,
                    "from_frame": None,
                    "to_frame": clusters[to_frame_id]["label"],
                    "from_share": 0,
                    "to_share": period_frames[to_period][to_frame_id]["share"],
                    "similarity": None,
                    "evolution_type": "emerged"
                })

    return transitions


def summarize_frame_evolution(transitions):
    summary = {
        "stable": 0,
        "transformed": 0,
        "weak_transition": 0,
        "discontinuous": 0,
        "emerged": 0,
        "disappeared": 0,
        "mean_similarity": None
    }

    similarities = []

    for transition in transitions:
        evolution_type = transition["evolution_type"]

        if evolution_type in summary:
            summary[evolution_type] += 1

        if transition["similarity"] is not None:
            similarities.append(transition["similarity"])

    if similarities:
        summary["mean_similarity"] = float(np.mean(similarities))

    return summary

def build_frame_trajectories(latent_result):
    clusters = latent_result["clusters"]
    period_frames = latent_result["period_frames"]

    periods = sorted(
        period_frames.keys(),
        key=sort_period_key
    )
    

    trajectories = {}

    for cluster_id, cluster_data in clusters.items():
        values = []

        for period in periods:
            frame_stats = period_frames.get(period, {}).get(
                cluster_id,
                {
                    "count": 0,
                    "share": 0
                }
            )

            values.append({
                "period": period,
                "count": frame_stats["count"],
                "share": frame_stats["share"]
            })

        shares = [
            item["share"]
            for item in values
        ]

        active_periods = sum(
            1 for s in shares if s > 0.01
        )

        persistence_ratio = (
            active_periods / len(shares)
        )

        trajectories[cluster_id] = {
            "label": cluster_data["label"],
            "verbs": cluster_data["verbs"],
            "values": values,
            "peak_period": values[int(np.argmax(shares))]["period"] if shares else None,
            "peak_share": float(max(shares)) if shares else 0,
            "mean_share": float(np.mean(shares)) if shares else 0,
            "volatility": float(np.std(shares)) if shares else 0,
            "first_share": float(shares[0]) if shares else 0,
            "last_share": float(shares[-1]) if shares else 0,
            "net_change": float(shares[-1] - shares[0]) if shares else 0,
            "persistence_ratio": float(persistence_ratio)
        }



    return trajectories