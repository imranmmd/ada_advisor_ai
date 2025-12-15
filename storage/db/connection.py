import psycopg2
from .config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_URL


def _ensure_credentials():
    """Ensure that required DB credentials are set (remote-first)."""
    if DB_URL:
        return
    missing = [
        name
        for name, value in {
            "DB_HOST": DB_HOST,
            "DB_PORT": DB_PORT,
            "DB_NAME": DB_NAME,
            "DB_USER": DB_USER,
            "DB_PASSWORD": DB_PASSWORD,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Remote database configuration is required. "
            "Set DATABASE_URL (preferred) or provide all DB_* variables. "
            f"Missing: {', '.join(missing)}"
        )


def get_connection():
    """
    Standard psycopg2 connection.
    Imported by save_to_postgres.py and other modules.
    """
    _ensure_credentials()
    if DB_URL:
        return psycopg2.connect(DB_URL)
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
