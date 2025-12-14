import psycopg2
from storage.db.connection import get_connection
import os

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")

def init_schema():
    """Initialize the database schema by executing the SQL in schema.sql."""
    conn = get_connection()
    with conn.cursor() as cur:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            sql = f.read()
            cur.execute(sql)
    conn.commit()
    conn.close()
    print("✅ Database schema created successfully.")

if __name__ == "__main__":
    init_schema()
