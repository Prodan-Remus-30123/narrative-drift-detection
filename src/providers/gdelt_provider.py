"""
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

            provider_start_date = start_date

            # latest_date = get_latest_article_date(
            #     provider="gdelt",
            #     source=domain
            # )

            # if latest_date:

            #     provider_start_date = (
            #         latest_date[:8]
            #     )

            #     provider_start_date = (
            #         f"{provider_start_date[:4]}-"
            #         f"{provider_start_date[4:6]}-"
            #         f"{provider_start_date[6:8]}"
            #     )

            #     print(
            #         f"Incremental collection from "
            #         f"{provider_start_date}"
            #     )

            filters = Filters(
                keyword=query,
                domain=domain,
                start_date=provider_start_date,
                end_date=end_date,
                num_records=num_records
            )

            max_retries = 3
            retry_count = 0

            results = None

            while retry_count < max_retries:

                try:

                    results = self.gd.article_search(filters)
                    print( f"Articles returned: " f"{len(results)}" )
                    break

                except RateLimitError:

                    retry_count += 1

                    print(
                        f"Rate limited. Retry "
                        f"{retry_count}/{max_retries}"
                    )

                    wait_time = 20 * retry_count
                    print(f"Waiting {wait_time} seconds...")

                    time.sleep(wait_time)

                except Exception as e:

                    print(f"GDELT fatal error: {e}")

                    break

            if results is None:
                print( f"Skipping domain: {domain}")
                continue

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

        if len(collected_articles) == 0:

            return {
                "status": "failed",
                "articles": []
            }

        return {
            "status": "success",
            "articles": collected_articles
        }