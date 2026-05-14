"""
gdelt_provider.py

GDELT-based news collection provider.
"""

import time

from gdeltdoc import GdeltDoc, Filters
from gdeltdoc.errors import RateLimitError

from providers.base_provider import BaseProvider
from database import get_latest_article_date


class GDELTProvider(BaseProvider):

    def __init__(self):

        self.gd = GdeltDoc()

    def collect(
        self,
        query,
        start_date,
        end_date,
        domains,
        num_records=3
    ):

        collected_articles = []

        for domain in domains:

            print(f"\n=== GDELT: {domain} ===")

            latest_date = get_latest_article_date(
            provider="gdelt",
            source=domain
            )

            if latest_date:
                start_date = latest_date[:10]
                print(
                    f"Incremental collection from "
                    f"{start_date}"
                )

            filters = Filters(
                keyword=query,
                domain=domain,
                start_date=start_date,
                end_date=end_date,
                num_records=num_records
            )

            max_retries = 3
            retry_count = 0

            results = None

            while retry_count < max_retries:
                try:
                    results = self.gd.article_search(
                        filters
                    )
                    break

                except RateLimitError:
                    retry_count += 1
                    print(
                        f"Rate limited. Retry "
                        f"{retry_count}/{max_retries}"
                    )
                    time.sleep(20)

                except Exception as e:
                    print(f"GDELT fatal error: {e}")
                    break
            
            if results is None:
                if len(collected_articles) > 0:

                    return {
                        "status": "partial_success",
                        "articles": collected_articles
                    }

                return {
                    "status": "failed",
                    "articles": []
                }

            print(
                f"Articles returned: {len(results)}"
            )

            for _, row in results.iterrows():

                article = {

                    "provider": "gdelt",

                    "source": row.get(
                        "domain",
                        "unknown"
                    ),

                    "date": row.get(
                        "seendate",
                        ""
                    ),

                    "title": row.get(
                        "title",
                        ""
                    ),

                    "url": row.get(
                        "url",
                        ""
                    )
                }

                collected_articles.append(
                    article
                )

                time.sleep(3)

        return {
            "status": "success",
            "articles": collected_articles
        }