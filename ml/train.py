import pandas as pd
import joblib
import os
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from ml.preprocess import load_data_from_db, prepare_data
from ml.evaluate import evaluate_model


def engineer_features(df):
    """Add extra features to help the model find patterns."""
    df = df.copy()

    # Billing amount per day of stay
    df["billing_per_day"] = df["billing_amount"] / (df["length_of_stay"] + 1)

    # Age groups (child, young adult, adult, senior)
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 18, 35, 60, 120],
        labels=[0, 1, 2, 3]
    ).astype(int)

    return df


def train():
    
    print("\n Starting model training pipeline...")

    # ── 1. Load data from PostgreSQL ──────────────────────────────────────────
    df = load_data_from_db()

    # ── 2. Engineer extra features ────────────────────────────────────────────
    df = engineer_features(df)
    print(" Feature engineering done")

    # ── 3. Preprocess ─────────────────────────────────────────────────────────
    # Update feature cols to include new engineered features
    from ml.preprocess import FEATURE_COLS, TARGET_COL, encode_features

    extended_features = FEATURE_COLS + ["billing_per_day", "age_group"]

    df_encoded, encoders = encode_features(df, fit=True)

    X = df_encoded[extended_features]
    y = df_encoded[TARGET_COL]

    # Save extended feature list
    joblib.dump(extended_features, "models/feature_cols.joblib")

    # Encode target labels
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    joblib.dump(target_encoder, "models/target_encoder.joblib")
    print(f" Target classes: {target_encoder.classes_}")

    # ── 4. Train/test split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"\n   Train size: {len(X_train):,} | Test size: {len(X_test):,}")

    # ── 5. Train both models ──────────────────────────────────────────────────
    results = []

    # — Random Forest (tuned) —
    print("\n Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_results = evaluate_model(rf, X_test, y_test, "Random Forest")
    results.append(("Random Forest", rf, rf_results))

    # — XGBoost (tuned) —
    print("\n Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
        verbosity=0
    )
    xgb.fit(X_train, y_train)
    xgb_results = evaluate_model(xgb, X_test, y_test, "XGBoost")
    results.append(("XGBoost", xgb, xgb_results))

    # ── 6. Pick the best model by F1-score ────────────────────────────────────
    best_name, best_model, best_metrics = max(
        results, key=lambda x: x[2]["f1_score"]
    )
    print(f"\n Best model: {best_name} (F1: {best_metrics['f1_score']})")

    # ── 7. Save best model + metadata ─────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/model.joblib")

    metadata = {
        "best_model":  best_name,
        "metrics":     best_metrics,
        "all_results": [r[2] for r in results],
        "features":    extended_features,
    }
    with open("models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f" Model saved    → models/model.joblib")
    print(f" Metadata saved → models/metadata.json")
    print("\n Training complete!\n")

    return best_model, best_metrics


if __name__ == "__main__":
    train()