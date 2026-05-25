"""
context.py

Shared analysis context and cache for agentic tools.
"""

from database import load_full_articles
from preprocessing import preprocess_corpus
from temporal_entity_analysis import group_articles_by_period


class AnalysisContext:

    def __init__(self, source):
        self.source = source

        self.df = None
        self.source_df = None
        self.grouped = None
        self.preprocessed_grouped = None

        self.semantic_drift = None
        self.framing_drift = None
        self.salience_results = None
        self.salience_totals = None
        self.entity_importance = None
        self.sentiment_results = None
        self.conflict_trust_results = None

    def get_source_df(self):

        if self.source_df is None:

            self.df = load_full_articles()

            self.source_df = self.df[
                self.df["source"] == self.source
            ].copy()

        return self.source_df

    def get_grouped(self):

        if self.grouped is None:

            source_df = self.get_source_df()

            self.grouped = group_articles_by_period(
                source_df
            )

        return self.grouped

    def get_preprocessed_grouped(self):

        if self.preprocessed_grouped is None:

            grouped = self.get_grouped()

            self.preprocessed_grouped = {}

            for period, texts in grouped.items():

                self.preprocessed_grouped[period] = preprocess_corpus(
                    texts
                )

        return self.preprocessed_grouped