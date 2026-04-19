import pandas as pd
import numpy as np
from sqlalchemy import text
from database.db_connection import get_engine
from sklearn.preprocessing import LabelEncoder
import joblib
import os

FEATURE_COLS = [
    "age", "gender", "blood_type", "medical_condition",
    "insurance_provider", "billing_amount", "admission_type",
    "medication", "length_of_stay"
]
TARGET_COL = "test_results"


def load_data_from_db():
    """Pull cleaned data straight from PostgreSQL."""
    print(" Loading data from PostgreSQL...")
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(
            text(f"SELECT {', '.join(FEATURE_COLS + [TARGET_COL])} FROM patients"),
            conn
        )
    print(f"   Loaded {len(df):,} records")
    return df


def encode_features(df, fit=True, encoders=None):
    """
    Convert categorical columns to numbers.
    fit=True  → learn the encoding (training time)
    fit=False → apply existing encoding (prediction time)
    """
    df = df.copy()

    cat_cols = [
        "gender", "blood_type", "medical_condition",
        "insurance_provider", "admission_type", "medication"
    ]

    if fit:
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

        # Save encoders so we can reuse them at prediction time
        os.makedirs("models", exist_ok=True)
        joblib.dump(encoders, "models/encoders.joblib")
        print(" Encoders saved → models/encoders.joblib")

    else:
        # Use existing encoders — handle unseen values gracefully
        for col in cat_cols:
            le = encoders[col]
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0]
                if x in le.classes_ else -1
            )

    return df, encoders


def prepare_data(df, fit=True, encoders=None):
    """Full preprocessing pipeline — encoding + split X/y."""
    df, encoders = encode_features(df, fit=fit, encoders=encoders)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    return X, y, encoders


if __name__ == "__main__":
    df = load_data_from_db()
    X, y, encoders = prepare_data(df)
    print(f"\n Features shape: {X.shape}")
    print(f"   Target classes: {y.unique()}")