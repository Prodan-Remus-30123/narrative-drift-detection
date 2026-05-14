"""
guardian_provider.py

Guardian API news provider.
"""

import requests

from providers.base_provider import BaseProvider


class GuardianProvider(BaseProvider):

    BASE_URL = "https://content.guardianapis.com/search"

    def __init__(self, api_key):

        self.api_key = api_key

    def collect(
        self,
        query,
        start_date,
        end_date,
        domains=None,
        num_records=30
    ):

        params = {

            "q": query,
            "from-date": start_date,
            "to-date": end_date,
            "page": 2,
            "page-size": num_records,
            "api-key": self.api_key
        }

        response = requests.get(
            self.BASE_URL,
            params=params
        )

        if response.status_code == 429:

            return {
                "status": "rate_limited",
                "articles": []
            }

        if response.status_code != 200:

            return {
                "status": "failed",
                "articles": []
            }

        data = response.json()

        results = data["response"]["results"]

        articles = []

        for row in results:

            articles.append({

                "provider": "guardian",

                "source": "theguardian.com",

                "date": row.get(
                    "webPublicationDate",
                    ""
                ),

                "title": row.get(
                    "webTitle",
                    ""
                ),

                "url": row.get(
                    "webUrl",
                    ""
                )
            })

        return {
            "status": "success",
            "articles": articles
        }