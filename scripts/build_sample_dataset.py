"""
build_sample_dataset.py

One-off local utility: builds a small, redistribution-safe sample
database from the full local article corpus (database/articles.db),
for bundling with the Hugging Face Space demo.

Why a sample instead of the full corpus:
- The full corpus is ~40k full-text scraped news articles; publishing
  that verbatim on a public Space raises real copyright concerns.
- A live demo doesn't need 40k articles -- a few hundred, concentrated
  into a handful of bimonthly periods (temporal_entity_analysis.py
  requires at least MIN_DOCS_PER_PERIOD=20 documents per period or
  the period is dropped), is enough to show every pipeline stage
  producing real signal, and runs in seconds instead of minutes.

The sample window (Jan-Aug 2022) was chosen because all three source
have dense coverage there; cnn.com in particular has a real gap in the
raw collected data after September 2022 that would otherwise starve
several periods.

Article text is truncated to TEXT_EXCERPT_CHARS characters per
article, which keeps the redistributed excerpt short (well under a
single paragraph) while leaving enough content for spaCy/SBERT to
extract meaningful entities, verbs and embeddings.

Run locally (never in the Space itself):
    python scripts/build_sample_dataset.py
"""

import sqlite3
from pathlib import Path

import pandas as pd

SOURCE_DB_PATH = "database/articles.db"
SAMPLE_DB_PATH = "data/sample_articles.db"

TOPIC = "ukraine_war"
SOURCES = ["bbc.co.uk", "cnn.com", "theguardian.com"]
ARTICLES_PER_PERIOD = 40
TEXT_EXCERPT_CHARS = 800

# Must match temporal_entity_analysis.group_articles_by_period's
# bimonthly bucketing so every bucket clears MIN_DOCS_PER_PERIOD=20.
PERIODS = [
    ("2022-01-01", "2022-02-28"),
    ("2022-03-01", "2022-04-30"),
    ("2022-05-01", "2022-06-30"),
    ("2022-07-01", "2022-08-31"),
]


def build_sample_database():
    Path(SAMPLE_DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    source_conn = sqlite3.connect(SOURCE_DB_PATH)
    source_cursor = source_conn.cursor()

    sample_path = Path(SAMPLE_DB_PATH)
    if sample_path.exists():
        sample_path.unlink()

    sample_conn = sqlite3.connect(SAMPLE_DB_PATH)
    sample_cursor = sample_conn.cursor()

    sample_cursor.execute("""
    CREATE TABLE articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        provider TEXT,
        source TEXT,
        date TEXT,
        title TEXT,
        url TEXT UNIQUE,
        text TEXT,
        topic TEXT DEFAULT 'covid',
        ingestion_status TEXT,
        extraction_status TEXT,
        extraction_attempts INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    total_inserted = 0

    for source in SOURCES:
        # The real DB has inconsistent date string formats across rows
        # (some ISO-with-dashes, some compact GDELT-style without), so
        # raw SQL string date-range comparison silently misses rows.
        # Load everything for this source/topic and let pandas
        # normalize dates instead, matching how the rest of the
        # pipeline (main.py, actor_graph.py, ...) already handles this.
        source_df = pd.read_sql_query("""
            SELECT provider, source, date, title, url, text, topic
            FROM articles
            WHERE ingestion_status = 'full_text'
              AND topic = ?
              AND source = ?
              AND text IS NOT NULL
              AND TRIM(text) != ''
        """, source_conn, params=(TOPIC, source))

        source_df["parsed_date"] = pd.to_datetime(
            source_df["date"], format="mixed", utc=True
        ).dt.tz_localize(None)

        for period_start, period_end in PERIODS:
            period_df = source_df[
                (source_df["parsed_date"] >= pd.Timestamp(period_start))
                & (source_df["parsed_date"] <= pd.Timestamp(period_end))
            ].sort_values("parsed_date")

            if len(period_df) < ARTICLES_PER_PERIOD:
                print(
                    f"WARNING: {source} {period_start}..{period_end} only has "
                    f"{len(period_df)} articles (< {ARTICLES_PER_PERIOD})"
                )

            # Evenly spaced picks within the period, rather than only
            # the earliest articles.
            step = max(1, len(period_df) // ARTICLES_PER_PERIOD)
            picked = period_df.iloc[::step].iloc[:ARTICLES_PER_PERIOD]

            for _, row in picked.iterrows():
                excerpt = row["text"].strip()[:TEXT_EXCERPT_CHARS]

                sample_cursor.execute("""
                    INSERT OR IGNORE INTO articles (
                        provider, source, date, title, url, text, topic,
                        ingestion_status, extraction_status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'full_text', 'success')
                """, (
                    row["provider"], row["source"], row["date"], row["title"],
                    row["url"], excerpt, row["topic"]
                ))

            print(f"{source} {period_start}..{period_end}: sampled {len(picked)} / {len(period_df)} articles")
            total_inserted += len(picked)

    sample_conn.commit()
    sample_conn.close()
    source_conn.close()

    print(f"\nWrote {total_inserted} articles to {SAMPLE_DB_PATH}")


if __name__ == "__main__":
    build_sample_database()
