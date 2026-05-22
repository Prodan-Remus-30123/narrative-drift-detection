"""
evidence_tools.py

Agentic evidence retrieval tools.

Retrieves sentence-level evidence from articles
for entities and optionally verbs.
"""

import re
import pandas as pd

from database import load_full_articles

from utils.period_sorting import (
    sort_period_key
)


def assign_period(date_value):
    date = pd.to_datetime(
        date_value,
        errors="coerce"
    )

    if pd.isna(date):
        return None

    year = date.year
    month = date.month

    if month in [1, 2]:
        return f"{year}-01_02"

    if month in [3, 4]:
        return f"{year}-03_04"

    if month in [5, 6]:
        return f"{year}-05_06"

    if month in [7, 8]:
        return f"{year}-07_08"

    if month in [9, 10]:
        return f"{year}-09_10"

    return f"{year}-11_12"


def split_sentences(text):
    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) > 30
    ]


def sentence_matches(
    sentence,
    entity,
    verbs=None
):
    sentence_lower = sentence.lower()
    entity_lower = entity.lower()

    if entity_lower not in sentence_lower:
        return False

    if verbs is None:
        return True

    for verb in verbs:

        pattern = r"\b" + re.escape(
            verb.lower()
        ) + r"\b"

        if re.search(pattern, sentence_lower):
            return True

    return False


def retrieve_entity_evidence(
    source,
    entity,
    verbs=None,
    max_snippets_per_period=3
):
    df = load_full_articles()

    source_df = df[
        df["source"] == source
    ].copy()

    if source_df.empty:

        return {
            "source": source,
            "entity": entity,
            "verbs": verbs or [],
            "status": "no_source_data",
            "evidence": {}
        }

    source_df["period"] = source_df["date"].apply(
        assign_period
    )

    source_df = source_df.dropna(
        subset=["period"]
    )

    evidence = {}

    periods = sorted(
        source_df["period"].unique(),
        key=sort_period_key
    )

    for period in periods:

        period_df = source_df[
            source_df["period"] == period
        ]

        snippets = []

        for _, row in period_df.iterrows():

            text = row.get("text", "")

            for sentence in split_sentences(text):

                if sentence_matches(
                    sentence,
                    entity,
                    verbs
                ):

                    snippets.append({

                        "sentence":
                            sentence,

                        "title":
                            row.get("title", ""),

                        "url":
                            row.get("url", ""),

                        "date":
                            str(row.get("date", "")),

                        "matched_entity":
                            entity,

                        "matched_verbs":
                            verbs or []
                    })

                if len(snippets) >= max_snippets_per_period:
                    break

            if len(snippets) >= max_snippets_per_period:
                break

        evidence[period] = snippets

    return {
        "source": source,
        "entity": entity,
        "verbs": verbs or [],
        "status": "ok",
        "evidence": evidence
    }