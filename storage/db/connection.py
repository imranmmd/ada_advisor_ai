import psycopg2
from .config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD


def _ensure_credentials():
    """Ensure that required DB credentials are set."""
    missing = [name for name, value in {
        "DB_USER": DB_USER,
        "DB_PASSWORD": DB_PASSWORD,
    }.items() if not value]
    if missing:
        raise RuntimeError(
            f"Database credentials missing: {', '.join(missing)}. "
            "Set the variables in your environment or .env file."
        )


def get_connection():
    """
    Standard psycopg2 connection.
    Imported by save_to_postgres.py and other modules.
    """
    _ensure_credentials()
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
