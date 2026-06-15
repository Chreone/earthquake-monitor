"""
Run this once to clean the raw PHIVOLCS CSV and produce a cleaned
Mindanao-only CSV that the Flask app and training script will use.
"""

import os
from earthquake_utils import load_and_clean_data

RAW_PATH = "data/phivolcs_earthquake_data.csv"
CLEAN_PATH = "data/mindanao_earthquakes_clean.csv"


def main():
    df = load_and_clean_data(RAW_PATH)
    os.makedirs("data", exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)

    print(f"Saved {len(df)} cleaned Mindanao records to {CLEAN_PATH}")
    print("\nTop provinces by number of records:")
    print(df["General_Location"].value_counts().head(10))
    print("\nDate range:", df["Datetime"].min(), "to", df["Datetime"].max())


if __name__ == "__main__":
    main()