"""
frame_cache_db.py

Persistent latent frame cache.
"""

import sqlite3
import json


DB_PATH = "frame_cache.db"


def initialize_frame_cache():

    connection = sqlite3.connect(
        DB_PATH
    )

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

    connection.commit()
    connection.close()


def get_cached_frame(
    frame_hash
):

    connection = sqlite3.connect(
        DB_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            label,
            description,
            confidence

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


def store_cached_frame(
    frame_hash,
    result
):

    connection = sqlite3.connect(
        DB_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT OR REPLACE INTO frame_cache (

            frame_hash,
            label,
            description,
            confidence

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