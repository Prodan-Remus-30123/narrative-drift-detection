import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from database import get_connection


def column_exists(cursor, table_name, column_name):
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    return column_name in columns


def main():
    conn = get_connection()
    cursor = conn.cursor()

    if not column_exists(cursor, "articles", "topic"):
        cursor.execute(
            "ALTER TABLE articles ADD COLUMN topic TEXT DEFAULT 'covid'"
        )
        print("Added column: topic")
    else:
        print("Column already exists: topic")

    cursor.execute(
        """
        UPDATE articles
        SET topic = 'covid'
        WHERE topic IS NULL OR topic = ''
        """
    )

    conn.commit()
    conn.close()

    print("Migration completed.")


if __name__ == "__main__":
    main()