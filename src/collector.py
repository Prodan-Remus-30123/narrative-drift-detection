"""
Multi-provider ingestion orchestrator.
"""

import os
import time
from dotenv import load_dotenv
from database import (initialize_database, get_connection)
from providers.gdelt_provider import (GDELTProvider)
from providers.guardian_provider import (GuardianProvider)
from utils.deduplication import (deduplicate_articles)
from utils.filtering import (filter_articles)
from utils.date_utils import (generate_monthly_ranges)


load_dotenv()

provider_configs = {

    "GDELTProvider": {

         "queries": [

        {
            "query": "COVID-19",
            "num_records": 20
        },

        {
            "query": "coronavirus",
            "num_records": 20
        },

        {
            "query": "COVID WHO",
            "num_records": 5
        },

        {
            "query": "Coronavirus China",
            "num_records": 5
        },

        {
            "query": "Pandemic Beijing",
            "num_records": 5
        }
    ],

        "domains": [
            "bbc.co.uk",
            "cnn.com",
            "reuters.com",
            "nytimes.com"
        ]

    },

    "GuardianProvider": {

        "queries": [

        {
            "query": '"COVID-19" AND WHO',
            "num_records": 20
        },

        {
            "query": '"COVID-19" AND China',
            "num_records": 20
        },

        {
            "query": '"pandemic" AND Beijing',
            "num_records": 20
        }
    ],

        "domains": None,
    }
}


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

            text,

            ingestion_status,
            extraction_status

        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (

            article["provider"],
            article["source"],
            article["date"],

            article["title"],
            article["url"],

            "",

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

    providers = [

        GDELTProvider(),

        GuardianProvider(
            api_key=os.getenv(
                "GUARDIAN_API_KEY"
            )
        )
    ]

    global_start_date = "2020-01-01"

    global_end_date = "2020-12-31"

    monthly_ranges = generate_monthly_ranges(
        global_start_date,
        global_end_date
    )

    minimum_articles = 10

    all_articles = []

    for monthly_range in monthly_ranges:

        start_date = monthly_range[
            "start_date"
        ]

        end_date = monthly_range[
            "end_date"
        ]

        print(
            f"\n======================"
        )

        print(
            f"Batch: "
            f"{start_date} -> {end_date}"
        )

        print(
            f"======================"
        )
    
        for provider in providers:

            provider_name = provider.__class__.__name__

            print(
                f"\n=== Trying provider: {provider_name} ==="
            )

            config = provider_configs.get(
                provider_name,
                {}
            )

            queries = config.get("queries", [])

            provider_articles = []

            try:

                for query_config in queries:

                    query = query_config["query"]
                    query_num_records = query_config["num_records"]

                    print(f"\nQuery: {query}")

                    result = provider.collect(
                        query=query,
                        start_date=start_date,
                        end_date=end_date,
                        domains=config.get("domains"),
                        num_records=query_num_records
                    )

                    status = result["status"]

                    if status == "partial_success":
                        print(f"{provider_name} partially succeeded.")

                    if status == "rate_limited":
                        print(f"{provider_name} rate limited.")
                        continue

                    if status == "failed":
                        print(f"{provider_name} failed.")
                        continue

                    articles = result["articles"]

                    print(f"{provider_name} returned " f"{len(articles)} articles.")

                    provider_articles.extend(articles)

                    time.sleep(2)

                provider_articles = deduplicate_articles(provider_articles)
                provider_articles = filter_articles(provider_articles)

                print(
                    f"{provider_name} unique articles: "
                    f"{len(provider_articles)}"
                )

                if len(provider_articles) < minimum_articles:

                    print(
                        f"{provider_name} returned too few "
                        f"articles."
                    )

                    print(
                        "Fallback retrieval needed later."
                    )

                all_articles.extend(
                    provider_articles
                )

            except Exception as e:

                print(
                    f"{provider_name} exception: {e}"
                )

    all_articles = deduplicate_articles(
        all_articles
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