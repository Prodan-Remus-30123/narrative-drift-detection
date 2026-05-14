"""
data_collection.py

Collects real news articles using GDELT and extracts article text.
"""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd
from gdeltdoc import GdeltDoc, Filters
from newspaper import Article
from gdeltdoc.errors import RateLimitError


def extract_article_text(url: str, timeout: int = 10) -> str:
    """
    Extract full article text from a URL.

    Args:
        url: Article URL.
        timeout: Request timeout.

    Returns:
        Extracted article text, or empty string if extraction fails.
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text.strip()
    except Exception:
        return ""


def collect_gdelt_articles(
    query: str,
    start_date: str,
    end_date: str,
    max_records: int = 10,
    source_country: Optional[str] = None,
    output_path: str = "data/real_articles.csv",
) -> pd.DataFrame:
    """
    Collect news article metadata from GDELT and extract article text.

    Args:
        query: Search query, e.g. "COVID-19 China WHO".
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
        max_records: Maximum number of articles.
        source_country: Optional source country filter, e.g. "US".
        output_path: CSV output path.

    Returns:
        DataFrame with columns: date, source, url, title, text.
    """
    gd = GdeltDoc()

    filters = Filters(
    keyword=query,
    start_date=start_date,
    end_date=end_date,
    num_records=max_records,
    )

    while True:

        try:
            results = gd.article_search(filters)
            break

        except RateLimitError:

            print("Rate limited. Waiting 60 seconds...")

            time.sleep(60)

    rows = []

    for _, row in results.iterrows():
        url = row.get("url", "")
        title = row.get("title", "")
        source = row.get("domain", "unknown")
        date = row.get("seendate", "")

        if not url:
            continue

        text = extract_article_text(url)

        if len(text) < 500:
            continue

        rows.append(
            {
                "date": date,
                "source": source,
                "url": url,
                "title": title,
                "text": text,
            }
        )

        time.sleep(3)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    df = collect_gdelt_articles(
        query="COVID-19 China WHO",
        start_date="2020-01-01",
        end_date="2020-06-30",
        max_records=30,
        source_country="US",
        output_path="data/real_articles.csv",
    )

    print(df.head())
    print(f"\nSaved {len(df)} articles.")