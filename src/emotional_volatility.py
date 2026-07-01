"""
emotional_volatility.py
"""

import numpy as np


def compute_emotional_volatility(
    sentiment_results
):

    compounds = [

        sentiment_results[p]["compound"]

        for p in sentiment_results
    ]

    return np.std(compounds)