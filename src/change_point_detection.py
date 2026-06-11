"""
change_point_detection.py

Detects temporal change points in narrative drift signals.
"""

import numpy as np
from sklearn.decomposition import PCA


try:
    import ruptures as rpt
except ImportError:
    rpt = None


MIN_SERIES_LENGTH = 4


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def detect_change_points(
    values,
    labels=None,
    model="rbf",
    penalty=1.0
):
    """
    Detect change points in a 1D temporal signal.

    Args:
        values: list of numeric values
        labels: optional temporal labels
        model: ruptures model, e.g. "rbf", "l2"
        penalty: penalty controlling number of change points

    Returns:
        list of change point dictionaries
    """

    if rpt is None:
        return {
            "status": "skipped",
            "reason": "ruptures is not installed",
            "change_points": []
        }

    clean_values = [
        _safe_float(value)
        for value in values
    ]

    if len(clean_values) < MIN_SERIES_LENGTH:
        return {
            "status": "skipped",
            "reason": "time series too short",
            "change_points": []
        }

    signal = np.array(clean_values).reshape(-1, 1)

    algorithm = rpt.Pelt(
        model=model
    ).fit(signal)

    indices = algorithm.predict(
        pen=penalty
    )

    # ruptures returns the final endpoint too; remove it
    indices = [
        index
        for index in indices
        if index < len(clean_values)
    ]

    change_points = []

    for index in indices:
        before_index = max(0, index - 1)

        change_points.append({
            "index": index,
            "label": (
                labels[index]
                if labels and index < len(labels)
                else str(index)
            ),
            "previous_label": (
                labels[before_index]
                if labels and before_index < len(labels)
                else str(before_index)
            ),
            "previous_value": clean_values[before_index],
            "value": clean_values[index],
            "delta": clean_values[index] - clean_values[before_index]
        })

    return {
        "status": "ok",
        "model": model,
        "penalty": penalty,
        "values": clean_values,
        "labels": labels,
        "change_points": change_points
    }


def detect_source_change_points(source_result):
    """
    Detects change points for one source using available signals.
    Uses the actual structures produced by main.py:
    - semantic_drift
    - affective_dynamics
    - framing_drift
    """

    results = {}

    # ==========================
    # 1. Semantic drift
    # ==========================
    semantic = source_result.get("semantic_drift", {})

    semantic_labels = semantic.get("labels", [])
    semantic_values = semantic.get("values", [])

    results["semantic_drift"] = detect_change_points(
        values=semantic_values,
        labels=semantic_labels,
        model="l2",
        penalty=0.001
    )

    # ==========================
    # 2. Affective dynamics
    # ==========================
    affective = source_result.get("affective_dynamics", {})
    affective_trajectory = affective.get("trajectory", [])

    affective_labels = []
    compound_values = []
    intensity_values = []
    polarization_values = []

    for item in affective_trajectory:
        label = item.get("transition") or item.get("label") or item.get("period")

        affective_labels.append(label)
        compound_values.append(item.get("compound_delta", 0.0))
        intensity_values.append(item.get("intensity_delta", 0.0))
        polarization_values.append(item.get("polarization_delta", 0.0))

    results["sentiment_compound"] = detect_change_points(
        values=compound_values,
        labels=affective_labels,
        model="l2",
        penalty=0.001
    )

    results["sentiment_intensity"] = detect_change_points(
        values=intensity_values,
        labels=affective_labels,
        model="l2",
        penalty=0.001
    )

    results["sentiment_polarization"] = detect_change_points(
        values=polarization_values,
        labels=affective_labels,
        model="l2",
        penalty=0.001
    )

    # ==========================
    # 3. Entity framing drift
    # ==========================
    framing_drift = source_result.get("framing_drift", {})

    framing_labels = []
    turnover_values = []
    js_values = []

    for transition, entities in framing_drift.items():
        if not isinstance(entities, dict) or len(entities) == 0:
            continue

        transition_turnovers = []
        transition_js = []

        for entity, stats in entities.items():
            turnover = stats.get("vocabulary_turnover")
            js = (
                stats.get("js_drift")
                or stats.get("jensen_shannon")
                or stats.get("js")
            )

            if turnover is not None:
                transition_turnovers.append(_safe_float(turnover))

            if js is not None:
                transition_js.append(_safe_float(js))

        if transition_turnovers:
            framing_labels.append(transition)
            turnover_values.append(
                sum(transition_turnovers) / len(transition_turnovers)
            )

            if transition_js:
                js_values.append(
                    sum(transition_js) / len(transition_js)
                )
            else:
                js_values.append(0.0)

    results["entity_turnover"] = detect_change_points(
        values=turnover_values,
        labels=framing_labels,
        model="l2",
        penalty=0.001
    )

    results["entity_js"] = detect_change_points(
        values=js_values,
        labels=framing_labels,
        model="l2",
        penalty=0.001
    )

    return results

def detect_embedding_change_points(
    vectors,
    labels=None,
    model="rbf",
    penalty=3.0
):
    """
    Detect change points directly on semantic embedding trajectories.

    Args:
        vectors: list of embedding vectors, one per period
        labels: period labels, e.g. ["2020-01_02", "2020-03_04"]
        model: ruptures model; "rbf" is recommended for embeddings
        penalty: controls number of detected change points

    Returns:
        change point result dictionary
    """

    if rpt is None:
        return {
            "status": "skipped",
            "reason": "ruptures is not installed",
            "change_points": []
        }

    if vectors is None or len(vectors) < MIN_SERIES_LENGTH:
        return {
            "status": "skipped",
            "reason": "embedding trajectory too short",
            "change_points": []
        }

    signal = np.array(vectors)

    if signal.ndim != 2:
        return {
            "status": "skipped",
            "reason": f"invalid embedding signal shape: {signal.shape}",
            "change_points": []
        }

    algorithm = rpt.Pelt(
        model=model
    ).fit(signal)

    indices = algorithm.predict(
        pen=penalty
    )

    indices = [
        index
        for index in indices
        if index < len(signal)
    ]

    change_points = []

    for index in indices:
        before_index = max(0, index - 1)

        change_points.append({
            "index": index,
            "label": (
                labels[index]
                if labels and index < len(labels)
                else str(index)
            ),
            "previous_label": (
                labels[before_index]
                if labels and before_index < len(labels)
                else str(before_index)
            )
        })

    return {
        "status": "ok",
        "model": model,
        "penalty": penalty,
        "signal_shape": signal.shape,
        "labels": labels,
        "change_points": change_points
    }


def detect_monthly_semantic_change_points(
    monthly_vectors,
    labels=None,
    n_components=3,
    model="rbf",
    penalty=0.5
):
    """
    Detects semantic regime shifts on monthly SBERT embedding trajectories.

    monthly_vectors:
        list of monthly aggregated embedding vectors

    labels:
        monthly period labels, e.g. ["2020-01", "2020-02", ...]

    The high-dimensional SBERT vectors are first reduced with PCA,
    then PELT is applied to the reduced semantic trajectory.
    """

    if rpt is None:
        return {
            "status": "skipped",
            "reason": "ruptures is not installed",
            "change_points": []
        }

    if monthly_vectors is None or len(monthly_vectors) < MIN_SERIES_LENGTH:
        return {
            "status": "skipped",
            "reason": "monthly semantic trajectory too short",
            "change_points": []
        }

    signal = np.array(monthly_vectors)

    if signal.ndim != 2:
        return {
            "status": "skipped",
            "reason": f"invalid monthly embedding shape: {signal.shape}",
            "change_points": []
        }

    actual_components = min(
        n_components,
        signal.shape[0] - 1,
        signal.shape[1]
    )

    reduced_signal = PCA(
        n_components=actual_components
    ).fit_transform(signal)

    algorithm = rpt.Pelt(
        model=model
    ).fit(reduced_signal)

    indices = algorithm.predict(
        pen=penalty
    )

    indices = [
        index
        for index in indices
        if index < len(reduced_signal)
    ]

    change_points = []

    for index in indices:
        before_index = max(0, index - 1)

        change_points.append({
            "index": index,
            "label": (
                labels[index]
                if labels and index < len(labels)
                else str(index)
            ),
            "previous_label": (
                labels[before_index]
                if labels and before_index < len(labels)
                else str(before_index)
            )
        })

    return {
        "status": "ok",
        "model": model,
        "penalty": penalty,
        "pca_components": actual_components,
        "original_shape": signal.shape,
        "reduced_shape": reduced_signal.shape,
        "labels": labels,
        "change_points": change_points
    }


def print_change_point_summary(source, change_points):
    print("\n=== CHANGE POINT DETECTION ===")
    print("Source:", source)

    for signal_name, result in change_points.items():
        print(f"\nSignal: {signal_name}")

        if result["status"] != "ok":
            print("Skipped:", result.get("reason"))
            continue

        points = result["change_points"]

        if not points:
            print("No change points detected.")
            continue

        for point in points:
            if "previous_value" in point and "value" in point:
                print(
                    f"- {point['previous_label']} -> {point['label']} | "
                    f"{point['previous_value']:.4f} -> "
                    f"{point['value']:.4f} "
                    f"(delta={point['delta']:.4f})"
                )
            else:
                print(
                    f"- regime boundary: "
                    f"{point['previous_label']} -> {point['label']}"
                )