"""
narrative_signatures.py

Builds source-level narrative behavior signatures.

A narrative signature is a compact feature vector describing how a source behaves
over time in terms of semantic drift, latent frames, entity ecosystems,
frame migrations, and sentiment.
"""

import numpy as np
from collections import Counter


def _safe_mean(values):
    values = [v for v in values if v is not None]

    if not values:
        return 0.0

    return float(np.mean(values))


def _safe_std(values):
    values = [v for v in values if v is not None]

    if not values:
        return 0.0

    return float(np.std(values))


def _safe_max(values):
    values = [v for v in values if v is not None]

    if not values:
        return 0.0

    return float(np.max(values))


def compute_frame_signature(latent_frames):
    """
    latent_frames = output from build_frame_trajectories()
    """

    if not latent_frames:
        return {
            "num_latent_frames": 0,
            "mean_frame_volatility": 0.0,
            "max_frame_volatility": 0.0,
            "mean_frame_persistence": 0.0,
            "mean_frame_peak_share": 0.0,
            "top_frame_labels": []
        }

    volatilities = []
    peak_shares = []
    persistence_values = []
    labels = []

    for frame_id, frame in latent_frames.items():
        volatilities.append(
            frame.get("volatility", 0.0)
        )

        peak_shares.append(
            frame.get("peak_share", 0.0)
        )

        values = frame.get("values", [])

        if values:
            active_periods = sum(
                1
                for point in values
                if point.get("share", 0.0) > 0
            )

            persistence = active_periods / len(values)
        else:
            persistence = 0.0

        persistence_values.append(persistence)

        labels.append(
            frame.get("label", "unknown_frame")
        )

    ranked_frames = sorted(
        latent_frames.items(),
        key=lambda x: x[1].get("volatility", 0.0),
        reverse=True
    )

    top_frame_labels = [
        frame.get("label", "unknown_frame")
        for _, frame in ranked_frames[:5]
    ]

    return {
        "num_latent_frames": len(latent_frames),
        "mean_frame_volatility": _safe_mean(volatilities),
        "max_frame_volatility": _safe_max(volatilities),
        "mean_frame_persistence": _safe_mean(persistence_values),
        "mean_frame_peak_share": _safe_mean(peak_shares),
        "top_frame_labels": top_frame_labels
    }


def compute_semantic_drift_signature(semantic_drift):
    values = semantic_drift.get("values", [])

    return {
        "mean_semantic_drift": _safe_mean(values),
        "max_semantic_drift": _safe_max(values),
        "semantic_drift_volatility": _safe_std(values),
        "semantic_threshold": semantic_drift.get("threshold")
    }


def compute_sentiment_signature(sentiment_results):
    compounds = [
        result.get("compound", 0.0)
        for result in sentiment_results.values()
    ]

    positives = [
        result.get("positive", 0.0)
        for result in sentiment_results.values()
    ]

    negatives = [
        result.get("negative", 0.0)
        for result in sentiment_results.values()
    ]

    return {
        "mean_sentiment": _safe_mean(compounds),
        "sentiment_volatility": _safe_std(compounds),
        "mean_positive": _safe_mean(positives),
        "mean_negative": _safe_mean(negatives)
    }


def compute_ecosystem_signature(entity_ecosystem):
    if not entity_ecosystem:
        return {
            "num_ecosystem_entities": 0,
            "mean_entity_drift": 0.0,
            "max_entity_drift": 0.0,
            "mean_entity_persistence": 0.0,
            "volatile_core_actor_count": 0,
            "stable_core_actor_count": 0,
            "episodic_disruptor_count": 0,
            "high_impact_shifting_actor_count": 0,
            "peripheral_actor_count": 0
        }

    role_counts = Counter()

    mean_drifts = []
    max_drifts = []
    persistence_values = []

    for entity, stats in entity_ecosystem.items():
        role = stats.get("ecosystem_type", "unknown")
        role_counts[role] += 1

        mean_drifts.append(
            stats.get("mean_drift", 0.0)
        )

        max_drifts.append(
            stats.get("max_drift", 0.0)
        )

        persistence_values.append(
            stats.get("persistence_ratio", 0.0)
        )

    return {
        "num_ecosystem_entities": len(entity_ecosystem),
        "mean_entity_drift": _safe_mean(mean_drifts),
        "max_entity_drift": _safe_max(max_drifts),
        "mean_entity_persistence": _safe_mean(persistence_values),
        "volatile_core_actor_count": role_counts.get("volatile_core_actor", 0),
        "stable_core_actor_count": role_counts.get("stable_core_actor", 0),
        "episodic_disruptor_count": role_counts.get("episodic_disruptor", 0),
        "high_impact_shifting_actor_count": role_counts.get("high_impact_shifting_actor", 0),
        "peripheral_actor_count": role_counts.get("peripheral_actor", 0)
    }


def compute_migration_signature(frame_migrations):
    if not frame_migrations:
        return {
            "mean_migration_ratio": 0.0,
            "max_migration_ratio": 0.0,
            "mean_unique_frames_visited": 0.0,
            "max_unique_frames_visited": 0.0
        }

    migration_ratios = []
    unique_frames = []

    for entity, stats in frame_migrations.items():
        migration_ratios.append(
            stats.get("migration_ratio", 0.0)
        )

        unique_frames.append(
            stats.get("unique_frames_visited", 0)
        )

    return {
        "mean_migration_ratio": _safe_mean(migration_ratios),
        "max_migration_ratio": _safe_max(migration_ratios),
        "mean_unique_frames_visited": _safe_mean(unique_frames),
        "max_unique_frames_visited": _safe_max(unique_frames)
    }


def build_narrative_signature(source, analysis_result):
    """
    Builds one source-level narrative signature.
    """

    signature = {
        "source": source
    }

    signature.update(
        compute_semantic_drift_signature(
            analysis_result.get("semantic_drift", {})
        )
    )

    signature.update(
        compute_frame_signature(
            analysis_result.get("latent_frames", {})
        )
    )

    signature.update(
        compute_sentiment_signature(
            analysis_result.get("sentiment", {})
        )
    )

    signature.update(
        compute_ecosystem_signature(
            analysis_result.get("entity_ecosystem", {})
        )
    )

    signature.update(
        compute_migration_signature(
            analysis_result.get("frame_migrations", {})
        )
    )

    return signature


def build_all_narrative_signatures(analysis_results):
    signatures = []

    for source, result in analysis_results.items():
        signatures.append(
            build_narrative_signature(
                source,
                result
            )
        )

    return signatures


def print_narrative_signature(signature):
    print(f"\n=== Narrative Signature: {signature['source']} ===")

    print(
        f"Semantic drift mean/max: "
        f"{signature['mean_semantic_drift']:.4f} / "
        f"{signature['max_semantic_drift']:.4f}"
    )

    print(
        f"Frame volatility mean/max: "
        f"{signature['mean_frame_volatility']:.4f} / "
        f"{signature['max_frame_volatility']:.4f}"
    )

    print(
        f"Entity drift mean/max: "
        f"{signature['mean_entity_drift']:.4f} / "
        f"{signature['max_entity_drift']:.4f}"
    )

    print(
        f"Mean migration ratio: "
        f"{signature['mean_migration_ratio']:.4f}"
    )

    print(
        f"Sentiment mean/volatility: "
        f"{signature['mean_sentiment']:.4f} / "
        f"{signature['sentiment_volatility']:.4f}"
    )

    print(
        "Actor roles: "
        f"volatile_core={signature['volatile_core_actor_count']}, "
        f"episodic={signature['episodic_disruptor_count']}, "
        f"high_impact={signature['high_impact_shifting_actor_count']}"
    )

    print(
        "Top volatile frames: "
        f"{signature['top_frame_labels']}"
    )