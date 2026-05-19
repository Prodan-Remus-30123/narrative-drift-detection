"""
raw_gdelt_provider.py

Direct GDELT API provider using raw HTTP requests.
"""

import time
import requests

from providers.base_provider import BaseProvider


class RawGDELTProvider(BaseProvider):

    BASE_URL = (
        "https://api.gdeltproject.org/api/v2/doc/doc"
    )

    def collect(
        self,
        query,
        start_date,
        end_date,
        domains,
        num_records=10
    ):

        collected_articles = []

        for domain in domains:

            print(
                f"\n=== RAW GDELT: {domain} ==="
            )

            gdelt_query = ( f'{query} ' f'domain:{domain}')
            print(
                f"GDELT query: "
                f"{gdelt_query}"
            )

            params = {

                "query":
                    gdelt_query,

                "mode":
                    "artlist",

                "format":
                    "json",

                "maxrecords":
                    num_records,

                "startdatetime":
                    self._format_gdelt_datetime(
                        start_date
                    ),

                "enddatetime":
                    self._format_gdelt_datetime(
                        end_date
                    )
            }

            results = self._execute_request(
                params
            )

            if results is None:

                print(
                    f"Skipping domain: "
                    f"{domain}"
                )

                continue

            articles = results.get(
                "articles",
                []
            )

            print(
                f"Articles returned: "
                f"{len(articles)}"
            )

            for article in articles:

                normalized_article = {

                    "provider":
                        "raw_gdelt",

                    "source":
                        domain,

                    "date":
                        article.get(
                            "seendate",
                            ""
                        ),

                    "title":
                        article.get(
                            "title",
                            ""
                        ),

                    "url":
                        article.get(
                            "url",
                            ""
                        )
                }

                collected_articles.append(
                    normalized_article
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

    def _execute_request(
        self,
        params
    ):

        max_retries = 3

        retry_count = 0

        while retry_count < max_retries:

            try:

                print(
                    f"Sending request..."
                )

                response = requests.get(

                    self.BASE_URL,

                    params=params,

                    timeout=30
                )

                print(
                    f"Status code: "
                    f"{response.status_code}"
                )

                if response.status_code == 429:

                    retry_count += 1

                    wait_time = (
                        20 * retry_count
                    )

                    print(
                        f"Rate limited. "
                        f"Retry "
                        f"{retry_count}/"
                        f"{max_retries}"
                    )

                    print(
                        f"Waiting "
                        f"{wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                    continue

                response.raise_for_status()

                print(f"Response preview: " f"{response.text[:500]}")
                data = response.json()

                print(
                    f"Response keys: "
                    f"{data.keys()}"
                )

                return data

            except requests.exceptions.Timeout:

                retry_count += 1

                print(
                    "Request timed out."
                )

                wait_time = (
                    20 * retry_count
                )

                print(
                    f"Waiting "
                    f"{wait_time} seconds..."
                )

                time.sleep(wait_time)

            except Exception as e:

                retry_count += 1

                print(
                    f"GDELT request error: {e}"
                )

                wait_time = (
                    20 * retry_count
                )

                time.sleep(wait_time)

        return None

    def _format_gdelt_datetime(
        self,
        date_string
    ):

        return (
            date_string
                .replace("-", "")
            + "000000"
        )