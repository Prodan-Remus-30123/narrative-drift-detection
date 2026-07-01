from database import get_connection


conn = get_connection()
cursor = conn.cursor()

cursor.execute("PRAGMA table_info(articles)")
print("Columns:")
for row in cursor.fetchall():
    print(row)

cursor.execute(
    """
    SELECT topic, COUNT(*)
    FROM articles
    GROUP BY topic
    """
)

print("\nTopic counts:")
for row in cursor.fetchall():
    print(row)

conn.close()