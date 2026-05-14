"""
changepoints.py

Change-point detection over narrative drift signals.
"""

import ruptures as rpt
import numpy as np


def detect_changepoints(
    drift_signal,
    model="l2",
    penalty=0.01
):
    """
    Detect structural breaks in drift signal.

    Args:
        drift_signal (list): Temporal drift values
        model (str): Cost model for ruptures
        penalty (float): Controls sensitivity

    Returns:
        list: Indices of detected change points
    """

    signal = np.array(drift_signal).reshape(-1, 1)

    algo = rpt.Pelt(model=model).fit(signal)

    breakpoints = algo.predict(pen=penalty)

    # Remove final endpoint
    breakpoints = [
        bp for bp in breakpoints
        if bp < len(drift_signal)
    ]

    return breakpoints