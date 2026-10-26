import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

DATA_PATH = "data/processed/training_sample.csv"
MODEL_PATH = "models/sentiment_pipeline.joblib"
METRICS_PATH = "models/metrics.json"

Path("models").mkdir(exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

X = df["text"].astype(str)
y = df["sentiment"].astype(str)

print("Total reviews:", len(df))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training reviews:", len(X_train))
print("Testing reviews:", len(X_test))

# Build the machine learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=20000,
        min_df=2,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=500,
        random_state=42
    ))
])

print("\nTraining model...")
model.fit(X_train, y_train)

print("Making predictions...")
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", round(accuracy, 4))

print("\nClassification Report:")
report_text = classification_report(
    y_test,
    predictions,
    zero_division=0
)
print(report_text)

print("Confusion Matrix:")
print(confusion_matrix(
    y_test,
    predictions,
    labels=model.classes_
))

# Save the trained model
joblib.dump(model, MODEL_PATH)

# Save evaluation metrics
metrics = {
    "accuracy": float(accuracy),
    "test_samples": int(len(y_test)),
    "classes": [str(c) for c in model.classes_],
    "random_state": 42
}

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

print("\nModel saved to:", MODEL_PATH)
print("Metrics saved to:", METRICS_PATH)
print("Training completed successfully!")
