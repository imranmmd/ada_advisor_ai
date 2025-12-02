import psycopg2
from .config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

def get_connection():
    """
    Standard psycopg2 connection.
    Imported by save_to_postgres.py and other modules.
    """
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
