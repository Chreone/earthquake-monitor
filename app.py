"""
Flask backend:
- Serves the map UI
- Serves historical earthquake data and stats
- Runs the trained models to forecast 30-day risk per grid cell
- Proxies chat messages to Groq's API for the earthquake chatbot
"""

import joblib
import pandas as pd
import requests
import numpy as np
from flask import Flask, jsonify, render_template, request

from earthquake_utils import (
    compute_cell_features, cell_bounds, get_all_cells,
    SIGNIFICANT_MAG,
)

app = Flask(__name__)

DATA_PATH = "data/mindanao_earthquakes_clean.csv"

# --- Groq API config ---
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

print("Loading cleaned data...")
df = pd.read_csv(DATA_PATH, parse_dates=["Datetime"])

print("Loading trained models...")
clf = joblib.load("models/risk_classifier.pkl")
reg = joblib.load("models/magnitude_regressor.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

NOW = df["Datetime"].max()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/earthquakes")
def api_earthquakes():
    """Lightweight [lat, lon, magnitude] points for the heatmap layer."""
    points = df[["Latitude", "Longitude", "Magnitude"]].values.tolist()
    return jsonify(points)


@app.route("/api/significant")
def api_significant():
    """Detailed info for notable earthquakes (M >= 4.5), for map markers."""
    sig = df[df["Magnitude"] >= 4.5].copy()
    sig["Datetime"] = sig["Datetime"].astype(str)
    cols = ["Datetime", "Latitude", "Longitude", "Magnitude", "Depth_In_Km",
            "Specific_Location", "General_Location"]
    return jsonify(sig[cols].to_dict(orient="records"))


@app.route("/api/stats")
def api_stats():
    """Return statistics AND chart data for analytics tab"""
    
    # Monthly trends for chart
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
            "months": monthly['Month'].tolist()[-24:],
            "counts": monthly['count'].tolist()[-24:]
        },
        "magnitude_distribution": mag_dist
    })


@app.route("/api/predictions")
def api_predictions():
    """Run the trained models on the latest data to forecast the next 30 days, per grid cell."""
    cells = get_all_cells(df)
    feature_rows = []
    geo = []

    for cell_lat, cell_lon in cells:
        cell_df = df[(df["cell_lat"] == cell_lat) & (df["cell_lon"] == cell_lon)]
        if len(cell_df) < 5:
            continue
        feats = compute_cell_features(cell_df, cell_df, NOW)
        feature_rows.append({
            "cell_lat": cell_lat, "cell_lon": cell_lon,
            "month": int(NOW.month), **feats,
        })
        geo.append((cell_lat, cell_lon))

    feat_df = pd.DataFrame(feature_rows)[feature_columns]
    risk_proba = clf.predict_proba(feat_df)[:, 1]
    expected_mag = reg.predict(feat_df)

    cells_out = []
    for (cell_lat, cell_lon), risk, mag in zip(geo, risk_proba, expected_mag):
        cells_out.append({
            "bounds": cell_bounds(cell_lat, cell_lon),
            "risk_probability": round(float(risk), 3),
            "expected_max_magnitude": round(float(mag), 2),
        })

    return jsonify({
        "as_of": str(NOW),
        "forecast_days": 30,
        "significant_threshold": SIGNIFICANT_MAG,
        "cells": cells_out,
    })


@app.route("/api/chat", methods=["POST"])
def api_chat():
    user_message = (request.json or {}).get("message", "").strip()
    if not user_message:
        return jsonify({"reply": "Please type a message."})

    top_quakes = df.sort_values("Magnitude", ascending=False).head(3)
    recent_summary = "; ".join(
        f"M{r.Magnitude} near {r.Specific_Location.strip()}, {r.General_Location} on {r.Datetime.date()}"
        for r in top_quakes.itertuples()
    )

    system_prompt = (
        "You are an earthquake-awareness assistant focused on Mindanao, Philippines. "
        "You sit on top of a PHIVOLCS earthquake dataset and a machine learning model "
        "that estimates earthquake risk per grid cell for the next 30 days based on "
        f"recent seismic activity. Some of the strongest recorded earthquakes in the "
        f"dataset are: {recent_summary}. "
        "Prioritize earthquake safety, preparedness, and PHIVOLCS/NDRRMC guidance, and "
        "explain how the data/model works when asked. You can answer general questions "
        "too, but where it makes sense, relate things back to earthquake awareness and "
        "safety. Be clear that earthquake forecasting here is statistical/probabilistic, "
        "not an exact prediction of when or where a quake will strike."
    )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.5,
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
        reply = f"Sorry, the chatbot is temporarily unavailable. Please try again later. Error: {str(e)[:100]}"

    return jsonify({"reply": reply})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)