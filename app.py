from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd   # ✅ FIXED
import os

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model & scaler
model = pickle.load(open(os.path.join(BASE_DIR, "models", "rain_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "models", "scaler.pkl"), "rb"))

# ✅ IMPORTANT: load training columns
columns = pickle.load(open(os.path.join(BASE_DIR, "models", "columns.pkl"), "rb"))

@app.route("/")
def home():
    return "🌧️ RainCast API is running successfully!"

# 🔥 FINAL CLEAN PREDICT FUNCTION
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Handle missing values
        input_df = input_df.fillna(method='ffill')

        # Encoding
        input_df = pd.get_dummies(input_df)

        # 🔥 VERY IMPORTANT (column alignment)
        input_df = input_df.reindex(columns=columns, fill_value=0)

        # Scaling
        input_scaled = scaler.transform(input_df)

        # Prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

if __name__ == "__main__":
    app.run(debug=True)