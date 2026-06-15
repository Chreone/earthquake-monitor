"""
Trains:
1. A classifier: will a M>=4.0 earthquake hit a given grid cell in the
   next 30 days?
2. A regressor: what is the expected maximum magnitude in that cell
   over the next 30 days?

Run this after preprocess.py. Saves models into models/.
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, roc_auc_score

from earthquake_utils import load_and_clean_data, build_training_dataset, FEATURE_COLUMNS

DATA_PATH = "data/phivolcs_earthquake_data.csv"


def main():
    print("Loading and cleaning data...")
    df = load_and_clean_data(DATA_PATH)
    print(f"Mindanao records: {len(df)}")

    print("Building grid training dataset (this may take a bit)...")
    train_df = build_training_dataset(df)
    print(f"Training samples: {len(train_df)}")

    train_df = train_df.sort_values("as_of").reset_index(drop=True)

    X = train_df[FEATURE_COLUMNS]
    y_class = train_df["significant_next"]
    y_reg = train_df["max_mag_next"]

    # Time-based split: oldest 80% trains, most recent 20% tests
    split_idx = int(len(train_df) * 0.8)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    yc_train, yc_test = y_class.iloc[:split_idx], y_class.iloc[split_idx:]
    yr_train, yr_test = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]

    print("\n--- Risk classifier (M>=4.0 in next 30 days?) ---")
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=12, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    clf.fit(X_train, yc_train)
    print(classification_report(yc_test, clf.predict(X_test)))
    try:
        proba = clf.predict_proba(X_test)[:, 1]
        print("ROC AUC:", roc_auc_score(yc_test, proba))
    except ValueError:
        print("ROC AUC: not enough class variety in test set")

    print("\n--- Magnitude regressor (expected max magnitude next 30 days) ---")
    reg = RandomForestRegressor(
        n_estimators=300, max_depth=12, random_state=42, n_jobs=-1
    )
    reg.fit(X_train, yr_train)
    preds = reg.predict(X_test)
    print("MAE:", mean_absolute_error(yr_test, preds))

    print("\nFeature importances (classifier):")
    for col, imp in sorted(zip(FEATURE_COLUMNS, clf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {col}: {imp:.3f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/risk_classifier.pkl")
    joblib.dump(reg, "models/magnitude_regressor.pkl")
    joblib.dump(FEATURE_COLUMNS, "models/feature_columns.pkl")
    print("\nModels saved to models/")


if __name__ == "__main__":
    main()