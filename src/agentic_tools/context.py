"""
context.py

Shared analysis context and cache for agentic tools.
"""

from database import load_full_articles
from preprocessing import preprocess_corpus
from temporal_entity_analysis import group_articles_by_period
import hashlib


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
        self.entity_latent_frames = {}

        self.sentiment_results = None
        self.conflict_trust_results = None

        self.source_fingerprint = None

    def get_source_df(self):

        self.df = load_full_articles()

        source_df = self.df[self.df["source"] == self.source].copy()

        new_fingerprint = self.compute_source_fingerprint(source_df)

        if (self.source_fingerprint is not None and self.source_fingerprint != new_fingerprint):
            print(
                f"\n[Context] Data changed for {self.source}. "
                "Invalidating analysis cache."
            )

            self.invalidate_analysis_cache()

        self.source_fingerprint = new_fingerprint
        self.source_df = source_df

        return self.source_df

    def get_grouped(self):

        if self.grouped is None:
            source_df = self.get_source_df()
            self.grouped = group_articles_by_period(source_df)

        return self.grouped

    def get_preprocessed_grouped(self):

        if self.preprocessed_grouped is None:
            grouped = self.get_grouped()
            self.preprocessed_grouped = {}
            for period, texts in grouped.items():
                self.preprocessed_grouped[period] = preprocess_corpus(texts)

        return self.preprocessed_grouped
    
    def compute_source_fingerprint(self, source_df):
        """
        Computes a lightweight fingerprint for source data.
        If articles are added/removed or latest date changes,
        the fingerprint changes.
        """

        article_count = len(source_df)

        latest_date = None

        if "date" in source_df.columns and article_count > 0:
            latest_date = str(source_df["date"].max())

        raw = f"{self.source}|{article_count}|{latest_date}"

        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def invalidate_analysis_cache(self):
        """
        Clears source-dependent cached computations.
        Does not clear persistent LLM frame label cache.
        """

        self.grouped = None
        self.preprocessed_grouped = None

        self.semantic_drift = None
        self.framing_drift = None

        self.salience_results = None
        self.salience_totals = None
        self.entity_importance = None

        self.sentiment_results = None
        self.conflict_trust_results = None

        self.entity_latent_frames = {}