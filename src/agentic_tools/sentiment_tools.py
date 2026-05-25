"""
sentiment_tools.py

Agentic sentiment analysis tools.
"""

from database import load_full_articles

from preprocessing import preprocess_corpus

from temporal_entity_analysis import (
    group_articles_by_period
)

from sentiment_analysis import (
    analyze_sentiment
)

from emotional_volatility import (
    compute_emotional_volatility
)

from utils.period_sorting import (
    sort_period_key
)

from agentic_tools.context_registry import get_context


def get_sentiment_evolution(source):
    """
    Retrieve temporal sentiment evolution
    for a news source.

    Args:
        source (str)

    Returns:
        dict
    """

    context = get_context(source)

    if context.sentiment_results is not None:
        return context.sentiment_results

    grouped = context.get_preprocessed_grouped()

    sentiment_results = {}

    for period in sorted(
        grouped.keys(),
        key=sort_period_key
    ):

        sentiment = analyze_sentiment(
            grouped[period]
        )

        sentiment_results[
            period
        ] = {

            "compound":
                float(
                    sentiment["compound"]
                ),

            "positive":
                float(
                    sentiment["positive"]
                ),

            "negative":
                float(
                    sentiment["negative"]
                ),

            "neutral":
                float(
                    sentiment["neutral"]
                )
        }

    volatility = compute_emotional_volatility(
        sentiment_results
    )

    compounds = [

        sentiment_results[p]["compound"]

        for p in sentiment_results
    ]

    avg_compound = (
        sum(compounds)
        /
        len(compounds)
    )

    adjusted_compounds = {}

    for period in sentiment_results:

        adjusted_compounds[period] = float(

            sentiment_results[period]["compound"]

            -
            avg_compound
        )
    
    intensity = (sentiment["positive"] + sentiment["negative"])

    context.sentiment_results = {

        "source":
            source,

        "sentiment_evolution":
            sentiment_results,

        "emotional_volatility":
            float(volatility),

        "average_compound":
            float(avg_compound),

        "adjusted_compounds":
            adjusted_compounds,
        "intensity":
            float(intensity)
    }

    return context.sentiment_results