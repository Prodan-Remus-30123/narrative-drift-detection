"""
text_normalization.py

Utilities for title normalization.
"""

import re


def normalize_title(title):
    title = title.lower()
    title = re.sub(r"[^a-z0-9 ]", "",title)
    title = " ".join(title.split())

    return title