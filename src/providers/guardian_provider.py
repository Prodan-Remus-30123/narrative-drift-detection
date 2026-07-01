"""
Guardian API news provider.
"""

import requests

from providers.base_provider import BaseProvider


class GuardianProvider(BaseProvider):

    BASE_URL = "https://content.guardianapis.com/search"

    def __init__(self, api_key):
        self.api_key = api_key

    def collect(self, query, start_date, end_date, domains=None, num_records=30):
        articles = []

        for page in range(1, 6):
            params = {
                "q": query,
                "from-date": start_date,
                "to-date": end_date,
                "page": page,
                "page-size": min(num_records, 50),
                "api-key": self.api_key
            }

            response = requests.get(self.BASE_URL, params=params)

            if response.status_code == 429:
                break

            if response.status_code != 200:
                continue

            data = response.json()
            results = data["response"]["results"]

            for row in results:
                articles.append({
                    "provider": "guardian",
                    "source": "theguardian.com",
                    "date": row.get("webPublicationDate", ""),
                    "title": row.get("webTitle", ""),
                    "url": row.get("webUrl", "")
                })

            if len(results) == 0:
                break

        return {
            "status": "success",
            "articles": articles
        }