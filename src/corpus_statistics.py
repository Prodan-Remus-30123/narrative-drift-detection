"""
corpus_statistics.py

Basic corpus analytics and dataset statistics.
"""

import pandas as pd
from database import get_connection
import matplotlib.pyplot as plt
from pathlib import Path

def load_articles():
    conn = get_connection()

    query = """
        SELECT
            provider,
            source,
            date,
            title,
            topic,
            ingestion_status,
            extraction_status,
            text
        FROM articles
        """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


def print_basic_statistics(df):

    print("\n====================")
    print("CORPUS STATISTICS")
    print("====================")

    print(f"\nTotal articles: " f"{len(df)}")
    print(f"\nProviders: " f"{df['provider'].nunique()}")
    print(f"\nSources: " f"{df['source'].nunique()}")


def print_provider_distribution(df):

    print("\n====================")
    print("ARTICLES PER PROVIDER")
    print("====================")

    provider_counts = (df["provider"].value_counts())
    print(provider_counts)


def print_source_distribution(df):

    print("\n====================")
    print("TOP SOURCES")
    print("====================")

    source_counts = (df["source"].value_counts().head(20))
    print(source_counts)


def print_monthly_distribution(df):
    print("\n====================")
    print("ARTICLES PER MONTH")
    print("====================")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year_month"] = (df["date"].dt.to_period("M"))
    monthly_counts = (df["year_month"].value_counts().sort_index())

    print(monthly_counts)


def print_extraction_statistics(df):

    print("\n====================")
    print("EXTRACTION STATUS")
    print("====================")

    extraction_counts = (df["extraction_status"].value_counts())
    print(extraction_counts)

def print_topic_statistics(df):

    print("\n====================")
    print("TOPIC DISTRIBUTION")
    print("====================")

    print(df["topic"].value_counts())

    print("\n====================")
    print("UKRAINE SOURCES")
    print("====================")

    ukraine_df = df[
        df["topic"] == "ukraine_war"
    ]

    print(
        ukraine_df["source"]
        .value_counts()
    )

    print("\n====================")
    print("UKRAINE FULL TEXT")
    print("====================")

    full_text = ukraine_df[
        ukraine_df["text"]
        .fillna("")
        .str.len() > 1000
    ]

    print(
        f"Full text articles: "
        f"{len(full_text)}"
    )

    print(
        full_text["source"]
        .value_counts()
    )

def plot_monthly_distribution(df):

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

    plt.figure(figsize=(16, 6))

    monthly_counts.plot(
        kind="bar"
    )

    plt.title(
        "Corpus Distribution Over Time"
    )

    plt.xlabel(
        "Month"
    )

    plt.ylabel(
        "Number of Articles"
    )

    plt.tight_layout()

    output_dir = Path(
        "plots/corpus_statistics"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        output_dir /
        "articles_per_month.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

def main():
    df = load_articles()

    print_basic_statistics(df)
    print_provider_distribution(df)
    print_source_distribution(df)
    print_monthly_distribution(df)
    print_extraction_statistics(df)
    print_topic_statistics(df)
    plot_monthly_distribution(df)



if __name__ == "__main__":
    main()