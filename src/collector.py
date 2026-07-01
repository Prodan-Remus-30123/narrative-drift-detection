"""
Multi-provider ingestion orchestrator.
"""

import os
import time
from dotenv import load_dotenv
from database import (initialize_database, get_connection)
from providers.raw_gdelt_provider import (RawGDELTProvider)
from providers.guardian_provider import (GuardianProvider)
from providers.gdelt_query_builder import (build_boolean_query)
from utils.deduplication import (deduplicate_articles)
from utils.filtering import (filter_articles)
from utils.date_utils import (generate_monthly_ranges)


# COLLECTION_TOPIC = "covid"
COLLECTION_TOPIC = "ukraine_war"

load_dotenv()

COVID_PROVIDER_CONFIGS = {

    "RawGDELTProvider": {

        "queries": [

    {
        "query":
            "coronavirus AND China "
            "AND theme:HEALTH_PANDEMIC "
            "AND sourcelang:english",

        "num_records": 100
    },

    {
        "query":
            "coronavirus AND WHO "
            "AND theme:HEALTH_PANDEMIC "
            "AND sourcelang:english",

        "num_records": 100
    },

    {
        "query":
            "coronavirus AND outbreak "
            "AND theme:HEALTH_PANDEMIC "
            "AND sourcelang:english",

        "num_records": 100
    },

    {
        "query":
            "coronavirus AND Wuhan "
            "AND theme:HEALTH_PANDEMIC "
            "AND sourcelang:english",

        "num_records": 100
    },

    {
        "query":
            "COVID AND vaccine "
            "AND theme:HEALTH_PANDEMIC "
            "AND sourcelang:english",

        "num_records": 100
    },

    {
    "query":
        "COVID AND lockdown "
        "AND theme:HEALTH_PANDEMIC "
        "AND sourcelang:english",

    "num_records": 100
},
{
    "query":
        "COVID AND vaccination "
        "AND theme:HEALTH_PANDEMIC "
        "AND sourcelang:english",

    "num_records": 100
},
{
    "query":
        "COVID AND omicron "
        "AND theme:HEALTH_PANDEMIC "
        "AND sourcelang:english",

    "num_records": 100
},
{
    "query":
        "COVID AND delta variant "
        "AND theme:HEALTH_PANDEMIC "
        "AND sourcelang:english",

    "num_records": 100
},
{
    "query":
        "COVID AND reopening "
        "AND theme:HEALTH_PANDEMIC "
        "AND sourcelang:english",

    "num_records": 100
}
],

        "domains": [

            "bbc.co.uk",

            "cnn.com",

            "nytimes.com",

            "washingtonpost.com"
        ]
    },

    "GuardianProvider": {

        "queries": [

            # INSTITUTIONAL RESPONSE

            {
                "query":
                    '"World Health Organization" AND COVID',
                "num_records": 15
            },

            {
                "query":
                    '"CDC" AND pandemic',
                "num_records": 10
            },

            # GEOPOLITICAL

            {
                "query":
                    '"China" AND pandemic',
                "num_records": 15
            },

            {
                "query":
                    '"US China" AND COVID',
                "num_records": 10
            },

            {
                "query":
                    '"Beijing" AND coronavirus',
                "num_records": 10
            },

            # VACCINES

            {
                "query":
                    '"COVID vaccine" AND WHO',
                "num_records": 15
            },

            {
                "query":
                    '"vaccine diplomacy" AND China',
                "num_records": 10
            },

            # PUBLIC HEALTH

            {
                "query":
                    '"lockdown" AND pandemic',
                "num_records": 10
            },

            {
                "query":
                    '"hospital system" AND COVID',
                "num_records": 10
            }
        ],

        "domains": None
    }}

UKRAINE_WAR_PROVIDER_CONFIGS = {
    "RawGDELTProvider": {
        "queries": [
            {
                "query":
                    "Ukraine AND Russia AND invasion "
                    "AND theme:ARMEDCONFLICT "
                    "AND sourcelang:english",
                "num_records": 100
            },
            {
                "query":
                    "Ukraine AND Russia AND war "
                    "AND theme:ARMEDCONFLICT "
                    "AND sourcelang:english",
                "num_records": 100
            },
            {
                "query":
                    "Ukraine AND Russian forces "
                    "AND theme:ARMEDCONFLICT "
                    "AND sourcelang:english",
                "num_records": 100
            },
            {
                "query":
                    "Kyiv AND Russia "
                    "AND theme:ARMEDCONFLICT "
                    "AND sourcelang:english",
                "num_records": 100
            },
            {
                "query":
                    "Ukraine AND Zelensky "
                    "AND theme:ARMEDCONFLICT "
                    "AND sourcelang:english",
                "num_records": 100
            },
            {
                "query":
                    "Ukraine AND Putin "
                    "AND theme:ARMEDCONFLICT "
                    "AND sourcelang:english",
                "num_records": 100
            },
            {
                "query":
                    "Ukraine AND NATO "
                    "AND theme:ARMEDCONFLICT "
                    "AND sourcelang:english",
                "num_records": 100
            },
            {
                "query":
                    "Ukraine AND sanctions AND Russia "
                    "AND theme:SANCTIONS "
                    "AND sourcelang:english",
                "num_records": 100
            },
            {
                "query":
                    "Ukraine AND refugees "
                    "AND theme:REFUGEES "
                    "AND sourcelang:english",
                "num_records": 100
            },
            {
                "query":
                    "Ukraine AND weapons "
                    "AND theme:ARMEDCONFLICT "
                    "AND sourcelang:english",
                "num_records": 100
            }
        ],
        "domains": [
            "bbc.co.uk",
            "cnn.com",
            "washingtonpost.com",
            "nytimes.com"
        ]
    },

    "GuardianProvider": {
        "queries": [
            {
                "query": "Ukraine Russia invasion",
                "num_records": 50
            },
            {
                "query": "Ukraine war",
                "num_records": 50
            },
            {
                "query": "Russia sanctions Ukraine",
                "num_records": 50
            },
            {
                "query": "Zelensky Putin Ukraine",
                "num_records": 50
            },
            {
                "query": "Ukraine refugees",
                "num_records": 50
            },
            {
                "query": "Ukraine NATO weapons",
                "num_records": 50
            }
        ],
        "domains": None
    }
}

TOPIC_PROVIDER_CONFIGS = {
    "covid": COVID_PROVIDER_CONFIGS,
    "ukraine_war": UKRAINE_WAR_PROVIDER_CONFIGS
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
            topic,
            ingestion_status,
            extraction_status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (

            article["provider"],
            article["source"],
            article["date"],
            article["title"],
            article["url"],
            "",
            article.get("topic", COLLECTION_TOPIC),
            "metadata_collected",
            "pending"
        ))

        conn.commit()

        print(
            f"Inserted: {article['title']}"
        )

    except Exception as e:
        print(f"Insert skipped/error: {e}")

    conn.close()


def run_collection():

    initialize_database()

    providers = [RawGDELTProvider(), GuardianProvider( api_key=os.getenv("GUARDIAN_API_KEY"))]

    TOPIC_DATE_RANGES = {
        "covid": {
            "start": "2020-01-01",
            "end": "2022-12-31"
        },
        "ukraine_war": {
            "start": "2022-02-01",
            "end": "2025-06-01"
        }
    }

    global_start_date = TOPIC_DATE_RANGES[COLLECTION_TOPIC]["start"]
    global_end_date = TOPIC_DATE_RANGES[COLLECTION_TOPIC]["end"]


    monthly_ranges = generate_monthly_ranges(global_start_date, global_end_date)

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

            provider_configs = TOPIC_PROVIDER_CONFIGS[COLLECTION_TOPIC]
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
                # provider_articles = filter_articles(provider_articles)

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
        article["topic"] = COLLECTION_TOPIC
        insert_article(article)
        time.sleep(1)


if __name__ == "__main__":
    run_collection()