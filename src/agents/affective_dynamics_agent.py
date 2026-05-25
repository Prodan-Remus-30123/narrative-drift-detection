"""
affective_dynamics_agent.py

Agentic Affective Dynamics Agent.

This agent analyzes emotional evolution,
emotional volatility, baseline-adjusted sentiment,
and narrative intensity for a news source.
"""

from agentic_tools.sentiment_tools import (
    get_sentiment_evolution
)


class AffectiveDynamicsAgent:

    def __init__(self):
        self.name = "Affective Dynamics Agent"

    def analyze(
        self,
        source
    ):
        sentiment_data = get_sentiment_evolution(
            source=source
        )

        sentiment_results = sentiment_data[
            "sentiment_evolution"
        ]

        if len(sentiment_results) == 0:

            return {
                "agent": self.name,
                "source": source,
                "status": "insufficient_data",
                "interpretation": (
                    "No sentiment data was available "
                    "for this source."
                ),
                "confidence": "low"
            }

        interpretation = self._interpret(
            source=source,
            sentiment_data=sentiment_data
        )

        confidence = self._estimate_confidence(
            sentiment_results
        )

        return {
            "agent": self.name,
            "source": source,
            "status": "ok",
            "interpretation": interpretation,
            "sentiment_results": sentiment_results,
            "emotional_volatility": sentiment_data[
                "emotional_volatility"
            ],
            "average_compound": sentiment_data[
                "average_compound"
            ],
            "adjusted_compounds": sentiment_data[
                "adjusted_compounds"
            ],
            "intensity": sentiment_data.get(
                "intensity",
                None
            ),
            "confidence": confidence
        }

    def _interpret(
        self,
        source,
        sentiment_data
    ):
        sentiment_results = sentiment_data[
            "sentiment_evolution"
        ]

        adjusted = sentiment_data[
            "adjusted_compounds"
        ]

        volatility = sentiment_data[
            "emotional_volatility"
        ]

        avg_compound = sentiment_data[
            "average_compound"
        ]

        periods = list(
            sentiment_results.keys()
        )

        first_period = periods[0]
        last_period = periods[-1]

        first_compound = sentiment_results[
            first_period
        ]["compound"]

        last_compound = sentiment_results[
            last_period
        ]["compound"]

        sentiment_change = (
            last_compound
            -
            first_compound
        )

        most_negative_period = min(
            sentiment_results.items(),
            key=lambda x: x[1]["compound"]
        )

        most_positive_period = max(
            sentiment_results.items(),
            key=lambda x: x[1]["compound"]
        )

        strongest_negative_deviation = min(
            adjusted.items(),
            key=lambda x: x[1]
        )

        strongest_positive_deviation = max(
            adjusted.items(),
            key=lambda x: x[1]
        )

        interpretation = (
            f"{source} shows an average compound sentiment "
            f"of {avg_compound:.3f}. "
        )

        if sentiment_change > 0.15:

            interpretation += (
                "The emotional trajectory becomes noticeably "
                "less negative or more positive over time. "
            )

        elif sentiment_change < -0.15:

            interpretation += (
                "The emotional trajectory becomes more negative "
                "over time. "
            )

        else:

            interpretation += (
                "The emotional trajectory remains relatively "
                "stable over time. "
            )

        if volatility >= 0.20:

            interpretation += (
                f"Emotional volatility is relatively high "
                f"({volatility:.3f}), suggesting meaningful "
                "fluctuation in affective tone across periods. "
            )

        elif volatility >= 0.10:

            interpretation += (
                f"Emotional volatility is moderate "
                f"({volatility:.3f}), suggesting some affective "
                "movement across periods. "
            )

        else:

            interpretation += (
                f"Emotional volatility is low "
                f"({volatility:.3f}), suggesting a stable "
                "affective tone. "
            )

        interpretation += (
            f"The most negative period is "
            f"{most_negative_period[0]} "
            f"with compound sentiment "
            f"{most_negative_period[1]['compound']:.3f}. "
        )

        interpretation += (
            f"The most positive period is "
            f"{most_positive_period[0]} "
            f"with compound sentiment "
            f"{most_positive_period[1]['compound']:.3f}. "
        )

        interpretation += (
            f"Relative to the source baseline, the strongest "
            f"negative deviation occurs in "
            f"{strongest_negative_deviation[0]} "
            f"({strongest_negative_deviation[1]:.3f}), "
            f"while the strongest positive deviation occurs in "
            f"{strongest_positive_deviation[0]} "
            f"({strongest_positive_deviation[1]:.3f})."
        )

        return interpretation

    def _estimate_confidence(
        self,
        sentiment_results
    ):
        num_periods = len(
            sentiment_results
        )

        if num_periods >= 5:
            return "high"

        if num_periods >= 3:
            return "medium"

        return "low"