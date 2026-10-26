import sqlite3
from pathlib import Path

DB_PATH = Path("data/sentiment.db")


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_text TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

    print("Database initialized successfully.")


def save_prediction(review, sentiment, confidence):
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO predictions
                (review_text, sentiment, confidence)
            VALUES (?, ?, ?)
        """, (review, sentiment, confidence))

        return cursor.lastrowid


def get_predictions(limit=1000):
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT id, review_text, sentiment,
                   confidence, created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

    return [dict(row) for row in rows]


if __name__ == "__main__":
    initialize_database()
