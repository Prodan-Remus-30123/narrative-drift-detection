"""
deduplication.py

Article deduplication utilities.
"""

from utils.text_normalization import normalize_title


def deduplicate_articles(articles):
    seen_urls = set()
    seen_titles = set()
    unique_articles = []

    for article in articles:
        url = article.get("url", "")
        title = article.get("title", "")
        normalized_title = normalize_title(title)

        if not url:
            continue

        if url in seen_urls:
            continue

        if normalized_title in seen_titles:
            continue

        seen_urls.add(url)

        seen_titles.add(normalized_title)
        unique_articles.append(article)

    return unique_articles