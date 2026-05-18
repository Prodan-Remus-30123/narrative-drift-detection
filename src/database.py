"""
database.py

SQLite storage layer for article ingestion and analysis.
"""

import sqlite3
import pandas as pd


DB_PATH = "database/articles.db"


def get_connection():
    """
    Create SQLite connection.
    """
    return sqlite3.connect(DB_PATH)


def initialize_database():
    """
    Create articles table if it does not exist.
    """

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS articles (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        provider TEXT,
        source TEXT,

        date TEXT,

        title TEXT,
        url TEXT UNIQUE,

        text TEXT,

        ingestion_status TEXT,
        extraction_status TEXT,

        extraction_attempts INTEGER DEFAULT 0,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


def load_full_articles():
    """
    Load articles with full extracted text.
    """

    conn = get_connection()

    query = """
    SELECT
        source,
        date,
        text
    FROM articles
    WHERE ingestion_status='full_text'
    """

    df = pd.read_sql_query(query, conn)

    conn.close()

    return df


def get_latest_article_date(
    provider,
    source
):
    """
    Get latest collected article date
    for provider/source pair.
    """

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
    SELECT MAX(date)
    FROM articles
    WHERE provider=?
    AND source=?
    """, (
        provider,
        source
    ))

    result = cursor.fetchone()

    conn.close()

    return result[0]


if __name__ == "__main__":
    initialize_database()

    print("Database initialized.")