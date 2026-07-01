"""
base_provider.py

Base interface for news providers.
"""


class BaseProvider:

    def collect(self, query, start_date, end_date):
        raise NotImplementedError