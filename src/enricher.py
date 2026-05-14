"""
enricher.py

Fetch full article text and update DB.
"""

import time

from newspaper import Article

from database import get_connection


def fetch_pending_articles():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
    SELECT id, url
    FROM articles
    WHERE extraction_status='pending'
    """)

    rows = cursor.fetchall()

    conn.close()

    return rows


def update_article_text(
    article_id,
    text,
    status
):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
    UPDATE articles
    SET
        text=?,
        ingestion_status=?,
        extraction_attempts = extraction_attempts + 1
    WHERE id=?
    """, (
        text,
        status,
        article_id
    ))

    conn.commit()
    conn.close()


def extract_text(url):

    try:

        article = Article(url)

        article.download()
        article.parse()

        return article.text.strip()

    except Exception:

        return ""


def enrich_articles():

    rows = fetch_pending_articles()

    print(f"Pending articles: {len(rows)}")

    for article_id, url in rows:

        print(f"Extracting: {url}")

        text = extract_text(url)

        if len(text) > 500:

            update_article_text(
                article_id,
                text,
                "full_text"
            )

            print("Success.")

        else:

            update_article_text(
                article_id,
                "",
                "failed"
            )

            print("Failed.")

        time.sleep(3)


if __name__ == "__main__":
    enrich_articles()