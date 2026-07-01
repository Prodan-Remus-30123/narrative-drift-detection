"""
sentiment_analysis.py

Temporal sentiment evolution analysis.
"""

from vaderSentiment.vaderSentiment import (
    SentimentIntensityAnalyzer
)

import numpy as np


analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(
    texts
):

    sentiment_scores = []

    for text in texts:

        score = analyzer.polarity_scores(
            text
        )

        sentiment_scores.append(
            score
        )

    if len(sentiment_scores) == 0:

        return {
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "compound": 0
        }

    return {

        "positive":
            np.mean([
                s["pos"]
                for s in sentiment_scores
            ]),

        "negative":
            np.mean([
                s["neg"]
                for s in sentiment_scores
            ]),

        "neutral":
            np.mean([
                s["neu"]
                for s in sentiment_scores
            ]),

        "compound":
            np.mean([
                s["compound"]
                for s in sentiment_scores
            ])
    }