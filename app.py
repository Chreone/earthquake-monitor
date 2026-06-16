"""
Flask backend for EQWatch - Seismic Pattern Analysis
"""

import joblib
import pandas as pd
import requests
import numpy as np
import os
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from datetime import datetime

# Get the directory where app.py is located
BASE_DIR = Path(__file__).resolve().parent

# Import utils
from earthquake_utils import (
    compute_cell_features, cell_bounds, get_all_cells,
    SIGNIFICANT_MAG
)

app = Flask(__name__)

# Use absolute paths for Render
DATA_PATH = os.path.join(BASE_DIR, "data", "mindanao_earthquakes_clean.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Groq API config
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

print("=" * 50)
print("Loading EQWatch...")
print("=" * 50)
print(f"Base directory: {BASE_DIR}")
print(f"Data path: {DATA_PATH}")
print(f"Models directory: {MODEL_DIR}")

# Load data
try:
    df = pd.read_csv(DATA_PATH, parse_dates=["Datetime"])
    print(f"✓ Loaded {len(df)} earthquake records")
    print(f"✓ Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")
    data_loaded = True
except Exception as e:
    print(f"✗ Error loading data: {e}")
    df = None
    data_loaded = False

# Load models
try:
    clf = joblib.load(os.path.join(MODEL_DIR, "risk_classifier.pkl"))
    reg = joblib.load(os.path.join(MODEL_DIR, "magnitude_regressor.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    models_loaded = True
    print("✓ ML Models loaded")
except Exception as e:
    models_loaded = False
    print(f"✗ Models not found: {e}")

NOW = df["Datetime"].max() if data_loaded else datetime.now()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/earthquakes")
def api_earthquakes():
    """Get all earthquake points for reference"""
    if not data_loaded:
        return jsonify([])
    points = df[["Latitude", "Longitude", "Magnitude"]].values.tolist()
    return jsonify(points)


@app.route("/api/significant")
def api_significant():
    """Get significant earthquakes (M >= 4.5) for map markers"""
    if not data_loaded:
        return jsonify([])
    sig = df[df["Magnitude"] >= 4.5].copy()
    if len(sig) == 0:
        sig = df.nlargest(20, 'Magnitude').copy()
    sig["Datetime"] = sig["Datetime"].astype(str)
    cols = ["Datetime", "Latitude", "Longitude", "Magnitude", "Depth_In_Km",
            "Specific_Location", "General_Location"]
    cols = [c for c in cols if c in sig.columns]
    return jsonify(sig[cols].to_dict(orient="records"))


@app.route("/api/stats")
def api_stats():
    """Return statistics and chart data for analytics tab"""
    if not data_loaded:
        return jsonify({
            "total_records": 0,
            "date_range": ["N/A", "N/A"],
            "avg_magnitude": 0,
            "max_magnitude": 0,
            "by_province": {},
            "monthly_trends": {"months": [], "counts": []},
            "magnitude_distribution": {},
            "depth_avg": 0,
            "depth_max": 0
        })
    
    # Monthly trends - ALL DATA
    df_copy = df.copy()
    df_copy['Month'] = df_copy['Datetime'].dt.to_period('M').astype(str)
    monthly = df_copy.groupby('Month').size().reset_index(name='count')
    
    # Magnitude distribution
    mag_bins = [0, 2, 3, 4, 5, 6, 7, 10]
    mag_labels = ['0-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7+']
    df_copy['mag_range'] = pd.cut(df_copy['Magnitude'], bins=mag_bins, labels=mag_labels, right=False)
    mag_dist = df_copy['mag_range'].value_counts().to_dict()
    
    return jsonify({
        "total_records": int(len(df)),
        "date_range": [str(df["Datetime"].min()), str(df["Datetime"].max())],
        "avg_magnitude": float(df["Magnitude"].mean()),
        "max_magnitude": float(df["Magnitude"].max()),
        "by_province": df["General_Location"].value_counts().head(10).to_dict(),
        "depth_avg": float(df["Depth_In_Km"].mean()),
        "depth_max": float(df["Depth_In_Km"].max()),
        "monthly_trends": {
            "months": monthly['Month'].tolist(),
            "counts": monthly['count'].tolist()
        },
        "magnitude_distribution": mag_dist
    })


@app.route("/api/predictions")
def api_predictions():
    """Run trained models to identify pattern similarity"""
    if not models_loaded or not data_loaded:
        print("⚠ Models not loaded, using fallback patterns")
        return generate_fallback_patterns()
    
    try:
        cells = get_all_cells(df)
        feature_rows = []
        geo = []

        for cell_lat, cell_lon in cells:
            cell_df = df[(df["cell_lat"] == cell_lat) & (df["cell_lon"] == cell_lon)]
            if len(cell_df) < 3:
                continue
            feats = compute_cell_features(cell_df, cell_df, NOW)
            feature_rows.append({
                "cell_lat": cell_lat, "cell_lon": cell_lon,
                "month": int(NOW.month), **feats,
            })
            geo.append((cell_lat, cell_lon))

        if len(feature_rows) == 0:
            return generate_fallback_patterns()

        feat_df = pd.DataFrame(feature_rows)[feature_columns]
        similarity_scores = clf.predict_proba(feat_df)[:, 1]
        expected_mag = reg.predict(feat_df)

        cells_out = []
        for (cell_lat, cell_lon), sim, mag in zip(geo, similarity_scores, expected_mag):
            cells_out.append({
                "bounds": cell_bounds(cell_lat, cell_lon),
                "similarity_score": round(float(sim), 3),
                "reference_magnitude": round(float(mag), 2),
            })

        return jsonify({
            "as_of": str(NOW),
            "pattern_similarity_threshold": SIGNIFICANT_MAG,
            "cells": cells_out
        })
    except Exception as e:
        print(f"✗ Pattern error: {e}")
        return generate_fallback_patterns()


def generate_fallback_patterns():
    """Generate fallback pattern data when models unavailable"""
    cells_out = []
    for lat in range(5, 10):
        for lon in range(121, 126):
            similarity = abs(lat - 7.5) * abs(lon - 124.5) / 30
            similarity = min(max(similarity, 0.05), 0.85)
            cells_out.append({
                "bounds": [[lat, lon], [lat + 0.5, lon + 0.5]],
                "similarity_score": round(similarity, 3),
                "reference_magnitude": round(4.0 + similarity * 3, 2)
            })
    return jsonify({
        "as_of": str(NOW),
        "pattern_similarity_threshold": 4.0,
        "cells": cells_out[:30]
    })


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Chatbot using Groq API with disclaimer"""
    user_message = (request.json or {}).get("message", "").strip()
    if not user_message:
        return jsonify({"reply": "Please type a message."})

    # Get context from data
    total_quakes = len(df) if data_loaded else 0
    avg_mag = df["Magnitude"].mean() if data_loaded else 0
    max_mag = df["Magnitude"].max() if data_loaded else 0
    
    top_quakes = ""
    if data_loaded:
        top = df.nlargest(3, 'Magnitude')
        top_quakes = f" The largest recorded was M{top.iloc[0]['Magnitude']:.1f} in {top.iloc[0]['General_Location'] if 'General_Location' in top.columns else 'Mindanao'}."

    system_prompt = (
        f"You are an earthquake information assistant focused on Mindanao, Philippines. "
        f"Our database has {total_quakes} earthquake records from 2016 to present. "
        f"Average magnitude is {avg_mag:.1f}. Maximum recorded is M{max_mag:.1f}.{top_quakes} "
        f"IMPORTANT: You cannot predict earthquakes. That is scientifically impossible. "
        f"Provide helpful information about recorded earthquakes, safety tips, and how to interpret historical patterns. "
        f"Always remind users to follow PHIVOLCS for official alerts. "
        f"Be concise and clear. Do not say anything that sounds like a prediction."
    )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.5,
        "max_tokens": 500,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        reply = "The assistant is temporarily unavailable. Please try again later."

    return jsonify({"reply": reply})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)