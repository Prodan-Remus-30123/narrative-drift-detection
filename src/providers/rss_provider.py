"""
RSS-based news provider.
"""

import feedparser

from providers.base_provider import BaseProvider


class RSSProvider(BaseProvider):

    name = "rss"

    FEEDS = {
        "bbc.co.uk": "http://feeds.bbci.co.uk/news/world/rss.xml",
        "cnn.com": "http://rss.cnn.com/rss/edition_world.rss",
    }

    def collect(
        self,
        query,
        start_date,
        end_date,
        domains=None,
        num_records=30
    ):

        collected_articles = []

        query_terms = (
            query.replace('"', "")
            .replace("AND", "")
            .split()
        )

        for source, feed_url in self.FEEDS.items():

            if domains and source not in domains:
                continue

            feed = feedparser.parse(feed_url)

            for entry in feed.entries:

                title = entry.get("title", "")
                summary = entry.get("summary", "")
                text = f"{title} {summary}".lower()

                if not any(
                    term.lower() in text
                    for term in query_terms
                ):
                    continue

                collected_articles.append({
                    "provider": self.name,
                    "source": source,
                    "date": entry.get("published", ""),
                    "title": title,
                    "url": entry.get("link", ""),
                    "raw_summary": summary,
                    "language": "en"
                })

                if len(collected_articles) >= num_records:
                    return {
                        "status": "success",
                        "articles": collected_articles
                    }

        return {
            "status": "success",
            "articles": collected_articles
        }