import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path


DATABASE_PATH = os.getenv("DATABASE_PATH", "./data/app.db")


def _ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_conn():
    _ensure_parent_dir(DATABASE_PATH)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tenants (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                api_key TEXT,
                webhook_secret TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS inbound_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                external_user_id TEXT NOT NULL,
                external_message_id TEXT,
                text TEXT NOT NULL,
                status TEXT DEFAULT 'received',
                intent TEXT,
                ai_reply TEXT,
                lead_score INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (tenant_id) REFERENCES tenants(id)
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL,
                message_id INTEGER NOT NULL,
                intent TEXT NOT NULL,
                score INTEGER NOT NULL,
                status TEXT DEFAULT 'new',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (tenant_id) REFERENCES tenants(id),
                FOREIGN KEY (message_id) REFERENCES inbound_messages(id)
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS crm_outbox (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL,
                lead_id INTEGER NOT NULL,
                crm_target TEXT NOT NULL,
                payload TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                synced_at TEXT,
                FOREIGN KEY (tenant_id) REFERENCES tenants(id),
                FOREIGN KEY (lead_id) REFERENCES leads(id)
            )
            """
        )

        cursor.execute("PRAGMA table_info(tenants)")
        columns = {row[1] for row in cursor.fetchall()}

        if "api_key" not in columns:
            cursor.execute("ALTER TABLE tenants ADD COLUMN api_key TEXT")

        if "webhook_secret" not in columns:
            cursor.execute("ALTER TABLE tenants ADD COLUMN webhook_secret TEXT")
