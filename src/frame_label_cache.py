"""
frame_label_cache.py

Persistent cache for LLM-generated semantic frame labels.
"""

import json
from pathlib import Path


CACHE_PATH = Path("data/frame_label_cache.json")


def _ensure_cache_dir():
    CACHE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )


def load_cache():
    _ensure_cache_dir()

    if not CACHE_PATH.exists():
        return {}

    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(cache):
    _ensure_cache_dir()

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(
            cache,
            f,
            indent=2,
            ensure_ascii=False
        )


def make_frame_key(verbs):
    normalized = [
        str(verb).strip().lower()
        for verb in verbs
        if str(verb).strip()
    ]

    normalized = sorted(set(normalized))

    return "|".join(normalized)


def get_cached_label(key):
    cache = load_cache()

    return cache.get(key)


def store_cached_label(key, value):
    cache = load_cache()

    cache[key] = value

    save_cache(cache)