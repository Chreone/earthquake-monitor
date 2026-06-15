"""
Shared utilities for loading/cleaning the PHIVOLCS earthquake data,
building the spatial-grid training dataset, and computing
"as of today" features for prediction.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------

# Provinces / island groups that belong to Mindanao, as they appear
# in the PHIVOLCS "General_Location" column
MINDANAO_PROVINCES = [
    "Sarangani", "Davao Occidental", "Davao Oriental", "Davao Del Sur",
    "Davao Del Norte", "Davao De Oro", "City Of Davao", "South Cotabato",
    "Cotabato", "North Cotabato", "Sultan Kudarat", "Maguindanao Del Sur",
    "Maguindanao Del Norte", "Surigao Del Sur", "Surigao Del Norte",
    "Agusan Del Sur", "Agusan Del Norte", "Misamis Oriental",
    "Misamis Occidental", "Bukidnon", "Zamboanga Del Norte",
    "Zamboanga Del Sur", "Zamboanga Sibugay", "Lanao Del Sur",
    "Lanao Del Norte", "Dinagat Islands", "Basilan", "Sulu",
    "Tawi-tawi", "Camiguin",
]

# Bounding box used to drop a handful of clearly mis-keyed coordinates
LAT_MIN, LAT_MAX = 4.0, 11.0
LON_MIN, LON_MAX = 121.0, 128.0

# Grid cell size in degrees (~111km at the equator)
CELL_SIZE = 1.0

# "Significant" earthquake threshold used for the risk model
SIGNIFICANT_MAG = 4.0

# How many days of history to look back when building features
LOOKBACK_DAYS = 90

# How many days ahead the model predicts
FORECAST_DAYS = 30

FEATURE_COLUMNS = [
    "cell_lat", "cell_lon",
    "quake_count_90d", "mean_mag_90d", "max_mag_90d",
    "mean_depth_90d", "days_since_significant", "month",
]


# ---------------------------------------------------------------
# DATA LOADING / CLEANING
# ---------------------------------------------------------------

def _clean_depth(value):
    """PHIVOLCS sometimes reports depth as '<001', '--', '-' etc."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def load_and_clean_data(csv_path):
    """Load the raw PHIVOLCS CSV and return a cleaned Mindanao-only dataframe."""
    df = pd.read_csv(csv_path, low_memory=False)

    df["Datetime"] = pd.to_datetime(df["Date_Time_PH"], errors="coerce")
    df = df.dropna(subset=["Datetime", "Latitude", "Longitude", "Magnitude"])

    df["Depth_In_Km"] = df["Depth_In_Km"].apply(_clean_depth)
    df["Depth_In_Km"] = df["Depth_In_Km"].fillna(df["Depth_In_Km"].median())

    # Keep only Mindanao provinces/areas
    df = df[df["General_Location"].isin(MINDANAO_PROVINCES)]

    # Drop a handful of obviously mis-keyed coordinates
    df = df[df["Latitude"].between(LAT_MIN, LAT_MAX) & df["Longitude"].between(LON_MIN, LON_MAX)]

    df = df.sort_values("Datetime").reset_index(drop=True)

    df["cell_lat"] = (np.floor(df["Latitude"] / CELL_SIZE) * CELL_SIZE).astype(float)
    df["cell_lon"] = (np.floor(df["Longitude"] / CELL_SIZE) * CELL_SIZE).astype(float)

    return df


# ---------------------------------------------------------------
# GRID HELPERS
# ---------------------------------------------------------------

def get_all_cells(df):
    """Return every (cell_lat, cell_lon) pair that has recorded a quake."""
    pairs = df[["cell_lat", "cell_lon"]].drop_duplicates()
    return sorted(pairs.itertuples(index=False, name=None))


def cell_bounds(cell_lat, cell_lon):
    """Return [[south, west], [north, east]] bounds for a grid cell (for Leaflet)."""
    return [[cell_lat, cell_lon], [cell_lat + CELL_SIZE, cell_lon + CELL_SIZE]]


# ---------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------

def compute_cell_features(cell_df, history_df, as_of, lookback_days=LOOKBACK_DAYS):
    """
    Compute the feature row for one grid cell as of a given date.

    cell_df    : rows belonging to this grid cell only (any order)
    history_df : same rows, used to look further back for
                 "days since last significant quake"
    as_of      : pandas Timestamp - features use data strictly
                 before/at this date
    """
    window_start = as_of - pd.Timedelta(days=lookback_days)

    window = cell_df[(cell_df["Datetime"] > window_start) & (cell_df["Datetime"] <= as_of)]

    count = len(window)
    mean_mag = float(window["Magnitude"].mean()) if count else 0.0
    max_mag = float(window["Magnitude"].max()) if count else 0.0
    mean_depth = float(window["Depth_In_Km"].mean()) if count else 0.0

    past_sig = history_df[
        (history_df["Datetime"] <= as_of) & (history_df["Magnitude"] >= SIGNIFICANT_MAG)
    ]
    if len(past_sig):
        days_since_sig = (as_of - past_sig["Datetime"].max()).days
    else:
        days_since_sig = 9999

    return {
        "quake_count_90d": count,
        "mean_mag_90d": mean_mag,
        "max_mag_90d": max_mag,
        "mean_depth_90d": mean_depth,
        "days_since_significant": min(days_since_sig, 9999),
    }


def build_training_dataset(df):
    """
    Build a (cell, time-step) dataset for training.
    For every grid cell and every ~30-day time step, compute lookback
    features and the forward-looking targets.
    """
    rows = []
    cells = get_all_cells(df)

    data_start = df["Datetime"].min()
    data_end = df["Datetime"].max()

    step_dates = pd.date_range(
        data_start + pd.Timedelta(days=LOOKBACK_DAYS),
        data_end - pd.Timedelta(days=FORECAST_DAYS),
        freq="30D",
    )

    for cell_lat, cell_lon in cells:
        cell_df = df[(df["cell_lat"] == cell_lat) & (df["cell_lon"] == cell_lon)]
        if len(cell_df) < 5:
            continue  # skip cells with almost no history

        for as_of in step_dates:
            feats = compute_cell_features(cell_df, cell_df, as_of)

            future_window = cell_df[
                (cell_df["Datetime"] > as_of) &
                (cell_df["Datetime"] <= as_of + pd.Timedelta(days=FORECAST_DAYS))
            ]
            max_mag_next = float(future_window["Magnitude"].max()) if len(future_window) else 0.0
            significant_next = int(max_mag_next >= SIGNIFICANT_MAG)

            rows.append({
                "cell_lat": cell_lat,
                "cell_lon": cell_lon,
                "as_of": as_of,
                "month": as_of.month,
                **feats,
                "max_mag_next": max_mag_next,
                "significant_next": significant_next,
            })

    return pd.DataFrame(rows)