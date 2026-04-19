import joblib
import os

_model          = None
_encoders       = None
_target_encoder = None
_feature_cols   = None


def load_artifacts():
    global _model, _encoders, _target_encoder, _feature_cols

    print(" Loading model artifacts...")

    _model          = joblib.load("models/model.joblib")
    _encoders       = joblib.load("models/encoders.joblib")
    _target_encoder = joblib.load("models/target_encoder.joblib")

    # Load saved feature list if it exists, else use default
    if os.path.exists("models/feature_cols.joblib"):
        _feature_cols = joblib.load("models/feature_cols.joblib")
    else:
        _feature_cols = [
            "age", "gender", "blood_type", "medical_condition",
            "insurance_provider", "billing_amount", "admission_type",
            "medication", "length_of_stay"
        ]

    print(" Model artifacts loaded!")


def get_artifacts():
    if _model is None:
        load_artifacts()
    return _model, _encoders, _target_encoder, _feature_cols