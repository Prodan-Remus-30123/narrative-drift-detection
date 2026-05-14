"""
collector.py

Multi-provider ingestion orchestrator.
"""

import os
import time

from dotenv import load_dotenv

from database import (
    initialize_database,
    get_connection
)

from providers.gdelt_provider import (
    GDELTProvider
)

from providers.guardian_provider import (
    GuardianProvider
)


load_dotenv()


def insert_article(article):

    conn = get_connection()

    cursor = conn.cursor()

    try:

        cursor.execute("""
        INSERT INTO articles (

            provider,
            source,
            date,

            title,
            url,

            ingestion_status,
            extraction_status

        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (

            article["provider"],
            article["source"],
            article["date"],

            article["title"],
            article["url"],

            "metadata_collected",
            "pending"
        ))

        conn.commit()

        print(
            f"Inserted: {article['title']}"
        )

    except Exception:
        pass

    conn.close()


def run_collection():

    initialize_database()

    domains = [
        "bbc.co.uk",
        "cnn.com",
        "reuters.com",
        "nytimes.com"
    ]

    providers = [

        GDELTProvider(),

        GuardianProvider(
            api_key=os.getenv(
                "GUARDIAN_API_KEY"
            )
        )
    ]

    query = '"COVID-19" AND WHO AND China'

    start_date = "2020-01-01"

    end_date = "2020-12-31"

    all_articles = []

    for provider in providers:

        provider_name = provider.__class__.__name__

        print(
            f"\n=== Trying provider: {provider_name} ==="
        )

        try:

            result = provider.collect(
                query=query,
                start_date=start_date,
                end_date=end_date,
                domains=domains,
                num_records=10
            )

            if result["status"] == "partial_success":
                print(
                    f"{provider_name} partially succeeded."
                )

            if result["status"] == "rate_limited":
                print(
                    f"{provider_name} rate limited."
                )
                continue

            if result["status"] == "failed":
                print(
                    f"{provider_name} failed."
                )
                continue

            articles = result["articles"]

            print(
                f"{provider_name} returned "
                f"{len(articles)} articles."
            )

            all_articles.extend(articles)

        except Exception as e:

            print(
                f"{provider_name} exception: {e}"
            )

    print(
        f"\nTotal collected articles: "
        f"{len(all_articles)}"
    )

    for article in all_articles:

        insert_article(article)

        time.sleep(1)


if __name__ == "__main__":
    run_collection()