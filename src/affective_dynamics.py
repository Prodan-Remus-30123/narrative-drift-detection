"""
affective_dynamics.py

Advanced affective dynamics layer.

Computes:
- emotional volatility
- narrative intensity
- polarization score
- sentiment shift
- emotional escalation
- affective trajectory
"""

import numpy as np


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


def _safe_delta(values):
    if len(values) < 2:
        return 0.0

    return float(values[-1] - values[0])


def compute_affective_dynamics(sentiment_results):
    """
    sentiment_results:
    {
        "2020-01_02": {
            "compound": ...,
            "positive": ...,
            "negative": ...,
            "neutral": ...
        }
    }
    """

    if not sentiment_results:
        return {
            "mean_compound": 0.0,
            "compound_volatility": 0.0,
            "sentiment_shift": 0.0,
            "mean_narrative_intensity": 0.0,
            "intensity_volatility": 0.0,
            "intensity_shift": 0.0,
            "mean_polarization": 0.0,
            "polarization_volatility": 0.0,
            "polarization_shift": 0.0,
            "emotional_escalation": 0.0,
            "trajectory": []
        }

    periods = sorted(sentiment_results.keys())

    compounds = []
    positives = []
    negatives = []
    neutrals = []
    intensities = []
    polarizations = []

    trajectory = []

    for period in periods:
        result = sentiment_results[period]

        compound = float(result.get("compound", 0.0))
        positive = float(result.get("positive", 0.0))
        negative = float(result.get("negative", 0.0))
        neutral = float(result.get("neutral", 0.0))

        intensity = positive + negative

        polarization = abs(positive - negative)

        compounds.append(compound)
        positives.append(positive)
        negatives.append(negative)
        neutrals.append(neutral)
        intensities.append(intensity)
        polarizations.append(polarization)

        trajectory.append({
            "period": period,
            "compound": compound,
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "narrative_intensity": intensity,
            "polarization": polarization
        })

    emotional_escalation = 0.0

    if len(intensities) >= 2:
        emotional_escalation = float(
            np.mean(
                np.diff(intensities)
            )
        )

    return {
        "mean_compound": _safe_mean(compounds),
        "compound_volatility": _safe_std(compounds),
        "sentiment_shift": _safe_delta(compounds),

        "mean_positive": _safe_mean(positives),
        "mean_negative": _safe_mean(negatives),
        "mean_neutral": _safe_mean(neutrals),

        "mean_narrative_intensity": _safe_mean(intensities),
        "intensity_volatility": _safe_std(intensities),
        "intensity_shift": _safe_delta(intensities),

        "mean_polarization": _safe_mean(polarizations),
        "polarization_volatility": _safe_std(polarizations),
        "polarization_shift": _safe_delta(polarizations),

        "emotional_escalation": emotional_escalation,

        "trajectory": trajectory
    }


def print_affective_dynamics(affective_result):
    print("\n=== Affective Dynamics ===")

    print(
        "Compound mean/volatility/shift: "
        f"{affective_result['mean_compound']:.4f} / "
        f"{affective_result['compound_volatility']:.4f} / "
        f"{affective_result['sentiment_shift']:.4f}"
    )

    print(
        "Narrative intensity mean/volatility/shift: "
        f"{affective_result['mean_narrative_intensity']:.4f} / "
        f"{affective_result['intensity_volatility']:.4f} / "
        f"{affective_result['intensity_shift']:.4f}"
    )

    print(
        "Polarization mean/volatility/shift: "
        f"{affective_result['mean_polarization']:.4f} / "
        f"{affective_result['polarization_volatility']:.4f} / "
        f"{affective_result['polarization_shift']:.4f}"
    )

    print(
        "Emotional escalation: "
        f"{affective_result['emotional_escalation']:.4f}"
    )