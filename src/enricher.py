"""
enricher.py

Fetch full article text and update DB.
"""

import time

from newspaper import Article
from database import get_connection
import os
import tempfile


MIN_ARTICLE_LENGTH = 500
MAX_EXTRACTION_ATTEMPTS = 3

temp_dir = os.path.join(
    tempfile.gettempdir(),
    ".newspaper_scraper",
    "article_resources"
)

os.makedirs(
    temp_dir,
    exist_ok=True
)


def fetch_pending_articles():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT id, url, extraction_attempts
    FROM articles
    WHERE extraction_status IN ('pending', 'failed')
    AND extraction_attempts < ?
    """, (
        MAX_EXTRACTION_ATTEMPTS,
    ))

    rows = cursor.fetchall()
    conn.close()

    return rows


def update_article_text(
    article_id,
    text,
    ingestion_status,
    extraction_status
):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
    UPDATE articles
    SET
        text=?,
        ingestion_status=?,
        extraction_status=?,
        extraction_attempts =
            extraction_attempts + 1,
        updated_at=CURRENT_TIMESTAMP
    WHERE id=?
    """, (
        text,
        ingestion_status,
        extraction_status,
        article_id
    ))

    conn.commit()

    conn.close()


def extract_text(url):

    try:

        article = Article(url, memoize_articles=False)

        article.download()

        article.parse()

        return article.text.strip()

    except Exception as e:

        print(f"Extraction error: {e}")

        return ""


def enrich_articles():

    rows = fetch_pending_articles()

    print(f"Pending articles: {len(rows)}")

    for article_id, url, attempts in rows:

        print(
            f"\nExtracting "
            f"(attempt {attempts + 1}):"
        )

        print(url)

        text = extract_text(url)

        if len(text) >= MIN_ARTICLE_LENGTH:

            update_article_text(
                article_id,
                text,
                "full_text",
                "success"
            )

            print(
                f"Success "
                f"({len(text)} chars)"
            )

        else:

            update_article_text(
                article_id,
                "",
                "metadata_collected",
                "failed"
            )

            print("Failed extraction.")

        time.sleep(3)


if __name__ == "__main__":
    enrich_articles()