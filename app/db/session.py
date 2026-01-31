import sqlite3
from contextlib import contextmanager
from pathlib import Path
from config.settings import settings
from app.db.models import MIGRATIONS, run_schema_upgrades


def init_wal_mode(conn: sqlite3.Connection) -> None:
    """Enable Write-Ahead Logging mode for better concurrency."""
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.commit()


def get_db_connection() -> sqlite3.Connection:
    """Create and return a new database connection with WAL mode enabled."""
    conn = sqlite3.connect(settings.sqlite_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    init_wal_mode(conn)
    return conn


def get_conn() -> sqlite3.Connection:
    """
    Get database connection.

    Note: Returns a fresh connection. Callers should use context managers
    or ensure proper connection cleanup.
    """
    return get_db_connection()


@contextmanager
def get_db_context():
    """
    Context manager for database connections with automatic cleanup.

    Usage:
        with get_db_context() as conn:
            conn.execute("SELECT ...")
            conn.commit()
    """
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()


def init_db(sqlite_path: str) -> None:
    """Initialize database with schema and WAL mode."""
    # Ensure DB file path directory exists
    p = Path(sqlite_path)
    if p.parent and str(p.parent) != ".":
        p.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(sqlite_path, check_same_thread=False)
    conn.executescript(MIGRATIONS)
    conn.commit()

    # Run schema upgrades for existing databases
    run_schema_upgrades(conn)

    # Enable WAL mode
    init_wal_mode(conn)

    conn.close()
