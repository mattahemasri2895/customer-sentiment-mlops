import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = "data/sentiment.db"
OUTPUT_PATH = "data/predictions_export.csv"

Path("data").mkdir(exist_ok=True)

with sqlite3.connect(DB_PATH) as conn:
    df = pd.read_sql_query(
        """
        SELECT id, review_text, sentiment, confidence, created_at
        FROM predictions
        ORDER BY created_at DESC
        """,
        conn
    )

df.to_csv(OUTPUT_PATH, index=False)

print("Export completed successfully!")
print("Total predictions:", len(df))
print("Saved to:", OUTPUT_PATH)
print(df.head())
