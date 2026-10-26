import pandas as pd
from pathlib import Path

INPUT_FILE = Path(
    "data/raw/amazon_reviews_multilingual_US_v1_00.tsv"
)

OUTPUT_FILE = Path(
    "data/processed/reviews_cleaned.csv"
)

CHUNK_SIZE = 100_000


def preprocess_dataset():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    total_read = 0
    total_saved = 0
    first_chunk = True

    # Read the large TSV file in manageable chunks
    for chunk in pd.read_csv(
        INPUT_FILE,
        sep="\t",
        chunksize=CHUNK_SIZE,
        dtype=str,
        on_bad_lines="warn"
    ):
        total_read += len(chunk)

        # Keep only the columns required for sentiment analysis
        chunk = chunk[["star_rating", "review_body"]].copy()

        # Remove missing values
        chunk = chunk.dropna(
            subset=["star_rating", "review_body"]
        )

        chunk["review_body"] = (
            chunk["review_body"].astype(str).str.strip()
        )

        # Remove empty reviews
        chunk = chunk[chunk["review_body"] != ""]

        # Convert ratings to numeric values
        chunk["star_rating"] = pd.to_numeric(
            chunk["star_rating"],
            errors="coerce"
        )

        chunk = chunk[
            chunk["star_rating"].between(1, 5)
        ]

        # Map ratings to sentiment labels
        chunk["sentiment"] = chunk["star_rating"].map({
            1: "negative",
            2: "negative",
            3: "neutral",
            4: "positive",
            5: "positive"
        })

        # Rename review column
        chunk = chunk.rename(
            columns={"review_body": "text"}
        )

        chunk = chunk[["text", "sentiment"]]

        # Append each processed chunk to the output CSV
        chunk.to_csv(
            OUTPUT_FILE,
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            encoding="utf-8"
        )

        first_chunk = False
        total_saved += len(chunk)

        print(
            f"Read: {total_read:,} | "
            f"Saved: {total_saved:,}"
        )

    if total_saved == 0:
        raise ValueError(
            "No valid reviews found. Check the dataset."
        )

    print("\nPreprocessing completed!")
    print(f"Rows read: {total_read:,}")
    print(f"Rows saved: {total_saved:,}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    preprocess_dataset()
