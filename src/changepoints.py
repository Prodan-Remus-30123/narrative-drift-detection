"""
Statistical change-point detection over drift signal.

Planned integration:
- CUSUM
- PELT (via ruptures library)
- Other time-series structural break methods
"""

from typing import List


def detect_changepoints(drift_signal: List[float]):
    """
    Placeholder for change-point detection logic.

    Args:
        drift_signal (List[float]): Temporal drift values.

    Returns:
        List[int]: Indices of detected change points.
    """
    # TODO: # implement PELT-based change point detection using the 'ruptures' library.
# The drift signal will be treated as a 1D time series and structural breaks will be detected.
    return []