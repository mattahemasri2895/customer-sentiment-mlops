import pandas as pd
from pathlib import Path

INPUT_FILE = "data/processed/reviews_cleaned.csv"
OUTPUT_FILE = "data/processed/training_sample.csv"

CHUNK_SIZE = 100_000
SAMPLES_PER_CLASS = 2_000
FINAL_PER_CLASS = 30_000

Path("data/processed").mkdir(parents=True, exist_ok=True)

collected = {
    "positive": [],
    "negative": [],
    "neutral": []
}

for chunk in pd.read_csv(
    INPUT_FILE,
    chunksize=CHUNK_SIZE
):
    for label in collected:
        group = chunk[chunk["sentiment"] == label]

        if not group.empty:
            sample = group.sample(
                n=min(len(group), SAMPLES_PER_CLASS),
                random_state=42
            )
            collected[label].append(sample)

    print("Processed a chunk")

balanced_parts = []

for label, parts in collected.items():
    if not parts:
        raise ValueError(f"No reviews found for {label}")

    class_df = pd.concat(parts, ignore_index=True)

    if len(class_df) < FINAL_PER_CLASS:
        raise ValueError(
            f"Not enough {label} reviews: {len(class_df)}"
        )

    class_df = class_df.sample(
        n=FINAL_PER_CLASS,
        random_state=42
    )

    balanced_parts.append(class_df)

result = pd.concat(
    balanced_parts,
    ignore_index=True
)

result = result.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

result.to_csv(OUTPUT_FILE, index=False)

print("\nBalanced sample created!")
print(result["sentiment"].value_counts())
print("Total rows:", len(result))
print("Saved to:", OUTPUT_FILE)
