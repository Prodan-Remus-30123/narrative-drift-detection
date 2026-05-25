"""
frame_label_cache.py
"""

_CACHE = {}


def get_cached_label(key):

    return _CACHE.get(key)


def store_cached_label(
    key,
    value
):

    _CACHE[key] = value