import json
import re
from pathlib import Path

import pandas as pd

from database import get_connection


REQUIRED_COVID_TERMS = [
    "covid",
    "coronavirus",
    "pandemic",
    "lockdown",
    "vaccine",
    "vaccination",
    "omicron",
    "hospital",
    "infection",
    "cases",
    "variant",
    "public health",
    "nhs",
    "mask",
    "testing",
    "quarantine",
]

BAD_TITLE_PATTERNS = [
    "live",
    "as it happened",
    "briefing",
    "morning mail",
    "first thing",
    "newsletter",
    "transcript",
    "learning english",
    "6 minute english",
    "sport",
    "grand prix",
    "formula 1",
    "f1",
    "disney",
    "tourists",
    "tourism",
    "destinations",
    "cookbooks",
    "wildfires",
    "election",
    "trudeau",
    "podcast",
    "quiz",
]

ANALYSIS_RESULTS_PATH = Path(
    "outputs/20260616_122240/results/analysis_results.json"
)

OUTPUT_PATH = Path(
    "outputs/covid/true_low_drift_validation_cases.xlsx"
)

TOPIC_FILTER = "covid"
TOP_N_CASES = 10
ARTICLES_PER_PERIOD = 3
MIN_TEXT_LENGTH = 800


BAD_TITLE_PATTERNS = [
    "live",
    "as it happened",
    "briefing",
    "morning mail",
    "first thing",
    "newsletter",
    "today's headlines",
    "what we know",
    "what we learned",
    "updates",
    "latest",
    "quiz",
    "podcast",
]


def normalize_text(value):
    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value)
    ).strip()


def is_bad_title(title):
    title = normalize_text(title).lower()

    return any(
        pattern in title
        for pattern in BAD_TITLE_PATTERNS
    )


def period_to_months(period_label):
    """
    Example:
    2020-07_08 -> ["2020-07", "2020-08"]
    """

    year = period_label[:4]
    months = period_label[5:].split("_")

    return [
        f"{year}-{month}"
        for month in months
    ]


def split_transition(transition):
    before, after = transition.split("->")

    return before, after


def load_analysis_results():
    with open(
        ANALYSIS_RESULTS_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def load_articles():
    conn = get_connection()

    query = """
    SELECT
        source,
        date,
        title,
        url,
        text,
        topic,
        ingestion_status
    FROM articles
    WHERE topic = ?
    AND ingestion_status = 'full_text'
    AND text IS NOT NULL
    AND LENGTH(text) >= ?
    """

    df = pd.read_sql_query(
        query,
        conn,
        params=(
            TOPIC_FILTER,
            MIN_TEXT_LENGTH
        )
    )

    conn.close()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["year_month"] = (
        df["date"]
        .dt
        .to_period("M")
        .astype(str)
    )

    df["title_clean"] = (
        df["title"]
        .fillna("")
        .astype(str)
    )

    return df


def is_relevant_covid_article(title, text):
    content = (
        normalize_text(title) + " " +
        normalize_text(text[:2000])
    ).lower()

    return any(
        term in content
        for term in REQUIRED_COVID_TERMS
    )

def get_true_low_drift_transitions(results):
    rows = []

    for source, source_result in results.items():
        if source.startswith("__"):
            continue

        semantic = source_result.get(
            "semantic_drift",
            {}
        )

        labels = semantic.get("labels", [])
        values = semantic.get("values", [])

        for label, value in zip(labels, values):
            rows.append({
                "source": source,
                "transition": label,
                "semantic_drift": float(value)
            })

    rows = sorted(
        rows,
        key=lambda item: item["semantic_drift"]
    )

    return rows


def select_articles_for_period(df, source, period):
    months = period_to_months(period)

    period_df = df[
        (df["source"] == source) &
        (df["year_month"].isin(months))
    ].copy()

    if period_df.empty:
        return []

    period_df = period_df[
        ~period_df["title_clean"]
        .apply(is_bad_title)
    ].copy()

    if period_df.empty:
        return []
    
    period_df = period_df[
        period_df.apply(
            lambda row: is_relevant_covid_article(
                row["title"],
                row["text"]
            ),
            axis=1
        )
    ].copy()

    period_df["text_length"] = (
        period_df["text"]
        .fillna("")
        .str
        .len()
    )

    # Prefer normal articles with substantial text.
    period_df["covid_relevance"] = period_df.apply(
        lambda row: sum(
            term in (
                normalize_text(row["title"]) + " " +
                normalize_text(row["text"][:2000])
            ).lower()
            for term in REQUIRED_COVID_TERMS
        ),
        axis=1
    )

    period_df = period_df.sort_values(
        by=["covid_relevance", "text_length"],
        ascending=[False, False]
    )

    selected = []

    seen_urls = set()

    for _, row in period_df.iterrows():
        url = normalize_text(row["url"])

        if url in seen_urls:
            continue

        selected.append({
            "title": normalize_text(row["title"]),
            "url": url,
            "date": str(row["date"].date())
            if pd.notnull(row["date"])
            else "",
            "extract": normalize_text(row["text"])[:900]
        })

        seen_urls.add(url)

        if len(selected) >= ARTICLES_PER_PERIOD:
            break

    return selected


def build_validation_cases(results, df):
    low_transitions = get_true_low_drift_transitions(
        results
    )

    cases = []

    for item in low_transitions:
        source = item["source"]
        transition = item["transition"]
        before_period, after_period = split_transition(
            transition
        )

        before_articles = select_articles_for_period(
            df,
            source,
            before_period
        )

        after_articles = select_articles_for_period(
            df,
            source,
            after_period
        )

        if (
            len(before_articles) < 2 or
            len(after_articles) < 2
        ):
            continue

        cases.append({
            "case_id": f"LOW_{len(cases) + 1:02d}",
            "source": source,
            "transition": transition,
            "before_period": before_period,
            "after_period": after_period,
            "semantic_drift": item["semantic_drift"],
            "before_articles": before_articles,
            "after_articles": after_articles
        })

        if len(cases) >= TOP_N_CASES:
            break

    return cases


def export_cases(cases):
    rows = []

    for case in cases:
        base = {
            "case_id": case["case_id"],
            "source": case["source"],
            "transition": case["transition"],
            "semantic_drift": case["semantic_drift"],
            "before_period": case["before_period"],
            "after_period": case["after_period"]
        }

        for side in ["before", "after"]:
            articles = case[f"{side}_articles"]

            for i, article in enumerate(
                articles,
                start=1
            ):
                row = base.copy()

                row.update({
                    "period_type": side.upper(),
                    "article_no": i,
                    "title": article["title"],
                    "url": article["url"],
                    "date": article["date"],
                    "extract": article["extract"]
                })

                rows.append(row)

    output_df = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_df.to_excel(
        OUTPUT_PATH,
        index=False
    )

    print(f"Saved: {OUTPUT_PATH}")

    print("\nSelected low-drift cases:")

    for case in cases:
        print(
            f"{case['case_id']} | "
            f"{case['source']} | "
            f"{case['transition']} | "
            f"drift={case['semantic_drift']:.6f}"
        )

        print("  BEFORE:")
        for article in case["before_articles"]:
            print(f"   - {article['title']}")

        print("  AFTER:")
        for article in case["after_articles"]:
            print(f"   - {article['title']}")

        print()


def main():
    results = load_analysis_results()
    df = load_articles()

    cases = build_validation_cases(
        results,
        df
    )

    export_cases(cases)


if __name__ == "__main__":
    main()