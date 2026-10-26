
from pathlib import Path

import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

from database import (
    initialize_database,
    save_prediction,
    get_predictions
)

# Flask application setup
app = Flask(__name__)

CORS(app, resources={
    r"/predict": {"origins": "*"},
    r"/predictions": {"origins": "*"},
    r"/dashboard-data": {"origins": "*"}
})

# Load the trained sentiment model
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "sentiment_pipeline.joblib"

model = joblib.load(MODEL_PATH)
initialize_database()


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Customer Sentiment Analysis API is running",
        "endpoints": {
            "predict": "POST /predict",
            "predictions": "GET /predictions",
            "dashboard": "GET /dashboard-data"
        }
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Send a valid JSON object"
        }), 400

    review = data.get("review")

    if not isinstance(review, str) or not review.strip():
        return jsonify({
            "error": "Please provide a non-empty review"
        }), 400

    review = review.strip()

    if len(review) > 5000:
        return jsonify({
            "error": "Review cannot exceed 5000 characters"
        }), 400

    probabilities = model.predict_proba([review])[0]
    sentiment = str(model.predict([review])[0])
    confidence = float(max(probabilities))

    prediction_id = save_prediction(
        review,
        sentiment,
        confidence
    )

    return jsonify({
        "id": prediction_id,
        "review": review,
        "sentiment": sentiment,
        "confidence": round(confidence, 4)
    }), 200


@app.route("/predictions", methods=["GET"])
def predictions():
    return jsonify(get_predictions())


@app.route("/dashboard-data", methods=["GET"])
def dashboard_data():
    records = get_predictions()

    if not isinstance(records, list):
        records = []

    total_reviews = len(records)
    positive_count = 0
    negative_count = 0
    confidence_values = []

    for record in records:
        if not isinstance(record, dict):
            continue

        # Support either common review field name
        sentiment = str(record.get("sentiment", "")).strip().lower()

        if sentiment == "positive":
            positive_count += 1
        elif sentiment == "negative":
            negative_count += 1

        confidence = record.get("confidence")

        try:
            if confidence is not None:
                value = float(confidence)
                if 0 <= value <= 1:
                    confidence_values.append(value)
        except (TypeError, ValueError):
            pass

    average_confidence = (
        sum(confidence_values) / len(confidence_values)
        if confidence_values else 0.0
    )

    # Return the most recent records first.
    recent_predictions = list(reversed(records[-10:]))

    return jsonify({
        "total_reviews": total_reviews,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "average_confidence": round(average_confidence, 4),
        "recent_predictions": recent_predictions
    }), 200


if __name__ == "__main__":
    app.run(debug=False, port=5000)
