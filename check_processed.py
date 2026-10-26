import pandas as pd

file_path = "data/processed/reviews_cleaned.csv"

df = pd.read_csv(file_path, nrows=10)

print("Columns:", df.columns.tolist())
print("\nSample reviews:")
print(df.head())

print("\nSentiment labels:")
print(df["sentiment"].value_counts())