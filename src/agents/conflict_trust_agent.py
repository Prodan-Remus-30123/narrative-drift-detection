"""
conflict_trust_agent.py

Agentic Conflict & Trust Agent.

Analyzes conflict escalation, institutional trust,
and political polarization signals.
"""

from agentic_tools.conflict_trust_tools import (
    get_conflict_trust_signals
)


class ConflictTrustAgent:

    def __init__(self):
        self.name = "Conflict & Trust Agent"

    def analyze(self, source):

        signal_data = get_conflict_trust_signals(
            source=source
        )

        results = signal_data[
            "conflict_trust_results"
        ]

        if len(results) == 0:
            return {
                "agent": self.name,
                "source": source,
                "status": "insufficient_data",
                "interpretation": (
                    "No conflict or trust data was available."
                ),
                "confidence": "low"
            }

        interpretation = self._interpret(
            source=source,
            results=results
        )

        confidence = self._estimate_confidence(
            results
        )

        return {
            "agent": self.name,
            "source": source,
            "status": "ok",
            "interpretation": interpretation,
            "conflict_trust_results": results,
            "confidence": confidence
        }

    def _interpret(self, source, results):

        periods = list(results.keys())

        first = periods[0]
        last = periods[-1]

        first_conflict = results[first]["conflict_score"]
        last_conflict = results[last]["conflict_score"]

        conflict_change = last_conflict - first_conflict

        max_conflict_period = max(
            results.items(),
            key=lambda x: x[1]["conflict_score"]
        )

        max_polarization_period = max(
            results.items(),
            key=lambda x: x[1]["polarization_score"]
        )

        lowest_trust_period = min(
            results.items(),
            key=lambda x: x[1]["trust_score"]
        )

        interpretation = (
            f"{source} shows a conflict score change "
            f"from {first_conflict:.3f} in {first} "
            f"to {last_conflict:.3f} in {last}. "
        )

        if conflict_change > 0.02:
            interpretation += (
                "This suggests increasing conflict framing "
                "over time. "
            )
        elif conflict_change < -0.02:
            interpretation += (
                "This suggests decreasing conflict framing "
                "over time. "
            )
        else:
            interpretation += (
                "Conflict framing remains relatively stable "
                "over time. "
            )

        interpretation += (
            f"The highest conflict period is "
            f"{max_conflict_period[0]} "
            f"with score {max_conflict_period[1]['conflict_score']:.3f}. "
        )

        interpretation += (
            f"The strongest political polarization signal occurs in "
            f"{max_polarization_period[0]} "
            f"with score {max_polarization_period[1]['polarization_score']:.3f}. "
        )

        interpretation += (
            f"The lowest institutional trust signal occurs in "
            f"{lowest_trust_period[0]} "
            f"with trust score {lowest_trust_period[1]['trust_score']:.3f}."
        )

        return interpretation

    def _estimate_confidence(self, results):

        num_periods = len(results)

        total_verbs = sum(
            period_data["total_verbs"]
            for period_data in results.values()
        )

        if num_periods >= 5 and total_verbs >= 500:
            return "high"

        if num_periods >= 3 and total_verbs >= 200:
            return "medium"

        return "low"