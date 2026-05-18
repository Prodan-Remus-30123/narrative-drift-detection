"""
corpus_statistics.py

Basic corpus analytics and dataset statistics.
"""

import pandas as pd

from database import get_connection


def load_articles():

    conn = get_connection()

    query = """
    SELECT
        provider,
        source,
        date,
        title,
        ingestion_status,
        extraction_status
    FROM articles
    """

    df = pd.read_sql_query(
        query,
        conn
    )

    conn.close()

    return df


def print_basic_statistics(df):

    print("\n====================")
    print("CORPUS STATISTICS")
    print("====================")

    print(
        f"\nTotal articles: "
        f"{len(df)}"
    )

    print(
        f"\nProviders: "
        f"{df['provider'].nunique()}"
    )

    print(
        f"\nSources: "
        f"{df['source'].nunique()}"
    )


def print_provider_distribution(df):

    print("\n====================")
    print("ARTICLES PER PROVIDER")
    print("====================")

    provider_counts = (
        df["provider"]
        .value_counts()
    )

    print(provider_counts)


def print_source_distribution(df):

    print("\n====================")
    print("TOP SOURCES")
    print("====================")

    source_counts = (
        df["source"]
        .value_counts()
        .head(20)
    )

    print(source_counts)


def print_monthly_distribution(df):

    print("\n====================")
    print("ARTICLES PER MONTH")
    print("====================")

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["year_month"] = (
        df["date"]
        .dt
        .to_period("M")
    )

    monthly_counts = (
        df["year_month"]
        .value_counts()
        .sort_index()
    )

    print(monthly_counts)


def print_extraction_statistics(df):

    print("\n====================")
    print("EXTRACTION STATUS")
    print("====================")

    extraction_counts = (
        df["extraction_status"]
        .value_counts()
    )

    print(extraction_counts)


def main():

    df = load_articles()

    print_basic_statistics(df)

    print_provider_distribution(df)

    print_source_distribution(df)

    print_monthly_distribution(df)

    print_extraction_statistics(df)


if __name__ == "__main__":

    main()