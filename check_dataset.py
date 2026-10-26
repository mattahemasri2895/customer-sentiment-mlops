import pandas as pd

file_path = "data/raw/amazon_reviews_multilingual_US_v1_00.tsv"

df = pd.read_csv(
    file_path,
    sep="\t",
    nrows=5,
    dtype=str
)

print("Column names:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.to_string(index=False))

print("\nDataset preview completed.")
