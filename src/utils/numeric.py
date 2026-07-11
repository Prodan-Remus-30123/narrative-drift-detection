"""
numeric.py

Small numeric coercion helpers shared across the analysis modules.
"""


def safe_float(value, default=0.0):
    """
    Coerce value to float, falling back to `default` for None or
    anything that can't be converted (e.g. NaN-producing strings).
    """

    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default
