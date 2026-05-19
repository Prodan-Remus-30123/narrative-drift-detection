"""
temporal_entity_analysis.py

Monthly entity framing analysis.
"""

import pandas as pd

from database import (
    load_full_articles_with_dates
)

from entities import (
    analyze_entities
)


def group_articles_by_period(df):

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    grouped = {}

    for _, row in df.iterrows():

        date = row["date"]

        if pd.isna(date):
            continue

        month = date.month
        year = date.year

        if month in [1, 2]:
            period = f"{year}-01_02"

        elif month in [3, 4]:
            period = f"{year}-03_04"

        elif month in [5, 6]:
            period = f"{year}-05_06"

        elif month in [7, 8]:
            period = f"{year}-07_08"

        elif month in [9, 10]:
            period = f"{year}-09_10"

        else:
            period = f"{year}-11_12"

        if period not in grouped:
            grouped[period] = []

        grouped[period].append(
            row["text"]
        )

    return grouped

def print_top_entities(
    entity_data,
    top_n=10
):

    MIN_ENTITY_FREQUENCY = 3

    filtered_entities = {

    entity: stats

    for entity, stats in entity_data.items()

    if (
        stats["subject_count"]
        +
        stats["object_count"]
        +
        sum(stats["verbs"].values())
    ) >= MIN_ENTITY_FREQUENCY
}
    ranked_entities = sorted(

        filtered_entities.items(),

        key=lambda x:
            (
                x[1]["subject_count"]
                +
                x[1]["object_count"]
                +
                sum(
                    x[1]["verbs"].values()
                )
            ),

        reverse=True
    )

    for entity, stats in ranked_entities[:top_n]:

        print("\n====================")

        print(f"Entity: {entity}")

        print(
            f"Subject count: "
            f"{stats['subject_count']}"
        )

        print(
            f"Object count: "
            f"{stats['object_count']}"
        )

        print(
            f"Top verbs: "
            f"{stats['verbs'].most_common(5)}"
        )


def main():

    df = load_full_articles_with_dates()

    period_groups = (
        group_articles_by_period(df)
    )

    for period, texts in period_groups.items():

        print(
            f"\n\n################################"
        )

        print(
            f"MONTH: {period}"
        )

        print(
            f"################################"
        )

        print(
            f"Documents: {len(texts)}"
        )

        entity_data = analyze_entities(
            texts
        )

        print_top_entities(
            entity_data
        )


if __name__ == "__main__":

    main()