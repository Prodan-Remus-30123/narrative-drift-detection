"""
frame_cache_db.py

Persistent SQLite cache for LLM-generated latent-frame labels.

Two callers, two key schemes, one backing store:
- llm_frame_labeler.py labels an entity-conditioned verb cluster, keyed
  by a hash of its embedding centroid (frame_cache table).
- semantic_frame_labeling.py labels a source-level latent frame, keyed
  by its sorted/lowercased verb list (verb_key_frame_cache table).
"""

import sqlite3


DB_PATH = "frame_cache.db"


def _connect():
    return sqlite3.connect(DB_PATH)


def initialize_frame_cache():

    connection = _connect()
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS frame_cache (
            frame_hash TEXT PRIMARY KEY,
            label TEXT,
            description TEXT,
            confidence TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS verb_key_frame_cache (
            verb_key TEXT PRIMARY KEY,
            frame_label TEXT,
            frame_description TEXT
        )
        """
    )

    connection.commit()
    connection.close()


def get_cached_frame(frame_hash):

    connection = _connect()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT label, description, confidence
        FROM frame_cache
        WHERE frame_hash = ?
        """,
        (frame_hash,)
    )

    row = cursor.fetchone()
    connection.close()

    if row is None:
        return None

    return {
        "label": row[0],
        "description": row[1],
        "confidence": row[2]
    }


def store_cached_frame(frame_hash, result):

    connection = _connect()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT OR REPLACE INTO frame_cache (
            frame_hash, label, description, confidence
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            frame_hash,
            result["label"],
            result["description"],
            result["confidence"]
        )
    )

    connection.commit()
    connection.close()


def make_frame_key(verbs):
    """
    Build a stable cache key for a verb list: order/case independent.
    """

    normalized = [
        str(verb).strip().lower()
        for verb in verbs
        if str(verb).strip()
    ]

    normalized = sorted(set(normalized))

    return "|".join(normalized)


def get_cached_verb_frame_label(verb_key):

    initialize_frame_cache()

    connection = _connect()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT frame_label, frame_description
        FROM verb_key_frame_cache
        WHERE verb_key = ?
        """,
        (verb_key,)
    )

    row = cursor.fetchone()
    connection.close()

    if row is None:
        return None

    return {
        "frame_label": row[0],
        "frame_description": row[1]
    }


def store_cached_verb_frame_label(verb_key, result):

    initialize_frame_cache()

    connection = _connect()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT OR REPLACE INTO verb_key_frame_cache (
            verb_key, frame_label, frame_description
        )
        VALUES (?, ?, ?)
        """,
        (
            verb_key,
            result["frame_label"],
            result["frame_description"]
        )
    )

    connection.commit()
    connection.close()
