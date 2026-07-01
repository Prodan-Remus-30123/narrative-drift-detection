"""
date_utils.py

Utilities for temporal batching.
"""

from datetime import datetime
from dateutil.relativedelta import relativedelta


def generate_monthly_ranges(start_date, end_date):
    ranges = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while current < end:
        next_month = (current + relativedelta(months=1))
        ranges.append({
            "start_date": current.strftime("%Y-%m-%d"),
            "end_date": next_month.strftime("%Y-%m-%d")
        })

        current = next_month

    return ranges