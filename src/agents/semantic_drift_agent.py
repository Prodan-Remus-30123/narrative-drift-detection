"""
semantic_drift_agent.py

Agentic Semantic Drift Agent.

This agent investigates whether the overall topic
remains semantically stable or shifts over time.
"""

from agentic_tools.semantic_tools import (
    get_semantic_drift
)


class SemanticDriftAgent:

    def __init__(self):
        self.name = "Semantic Drift Agent"

    def analyze(
        self,
        source
    ):
        semantic_data = get_semantic_drift(
            source=source
        )

        drift_results = semantic_data[
            "semantic_drift"
        ]

        if len(drift_results) == 0:

            return {
                "agent": self.name,
                "source": source,
                "status": "insufficient_data",
                "interpretation": (
                    "No semantic drift data was available "
                    "for this source."
                ),
                "semantic_results": [],
                "confidence": "low"
            }

        interpretation = self._interpret(
            source=source,
            drift_results=drift_results
        )

        confidence = self._estimate_confidence(
            drift_results
        )

        return {
            "agent": self.name,
            "source": source,
            "status": "ok",
            "interpretation": interpretation,
            "semantic_results": drift_results,
            "confidence": confidence
        }

    def _interpret(
        self,
        source,
        drift_results
    ):
        significant = [
            item for item in drift_results
            if item["classification"] == "significant"
        ]

        avg_drift = (
            sum(item["drift"] for item in drift_results)
            /
            len(drift_results)
        )

        max_shift = max(
            drift_results,
            key=lambda x: x["drift"]
        )

        threshold = drift_results[0].get(
            "threshold",
            None
        )

        interpretation = (
            f"{source} shows an average semantic drift "
            f"of {avg_drift:.3f}. "
        )

        if len(significant) == 0:

            interpretation += (
                "The source remains semantically stable "
                "across the observed periods, suggesting "
                "that the broad topic remains consistent. "
            )

        elif len(significant) == 1:

            interpretation += (
                "The source is mostly semantically stable, "
                "but one significant topic shift is detected. "
            )

        else:

            interpretation += (
                "The source shows multiple significant "
                "semantic shifts, suggesting stronger topic "
                "evolution over time. "
            )

        interpretation += (
            f"The strongest semantic transition occurs during "
            f"{max_shift['transition']} with drift "
            f"{max_shift['drift']:.3f}. "
        )

        if threshold is not None:

            interpretation += (
                f"The dynamic threshold for significance is "
                f"{threshold:.3f}."
            )

        return interpretation

    def _estimate_confidence(
        self,
        drift_results
    ):
        num_transitions = len(
            drift_results
        )

        if num_transitions >= 4:
            return "high"

        if num_transitions >= 2:
            return "medium"

        return "low"