"""
filtering.py

Article quality filtering.
"""

BAD_PATTERNS = [
    "briefing",
    "watch live",
    "live updates",
    "podcast",
    "newsletter",
    "quiz",
    "crossword",
    "photo",
    "gallery"
]


def is_low_quality(article):
    title = article.get("title", "").lower()

    for pattern in BAD_PATTERNS:
        if pattern in title:
            return True

    return False


def filter_articles(articles):
    filtered_articles = []
    removed_count = 0

    for article in articles:
        if is_low_quality(article):
            removed_count += 1
            continue

        filtered_articles.append(article)

    print(f"Filtered out " f"{removed_count} low-quality articles.")
    return filtered_articles