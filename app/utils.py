import pandas as pd


def prepare_input(patient, encoders, feature_cols):
    """
    Transform raw API input into a model-ready dataframe.
    Handles unseen categorical values gracefully.
    """

    # ── Map API field names → internal column names ───────────────────────────
    raw = {
        "age":               patient.Age,
        "gender":            patient.Gender.strip().title(),
        "blood_type":        patient.Blood_Type.strip().title(),
        "medical_condition": patient.Medical_Condition.strip().title(),
        "insurance_provider":patient.Insurance_Provider.strip().title(),
        "billing_amount":    patient.Billing_Amount,
        "admission_type":    patient.Admission_Type.strip().title(),
        "medication":        patient.Medication.strip().title(),
        "length_of_stay":    patient.Length_of_Stay,
    }

    df = pd.DataFrame([raw])

    # ── Engineer the same extra features used during training ─────────────────
    df["billing_per_day"] = df["billing_amount"] / (df["length_of_stay"] + 1)
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 18, 35, 60, 120],
        labels=[0, 1, 2, 3]
    ).astype(int)

    # ── Encode categoricals using saved encoders ──────────────────────────────
    cat_cols = [
        "gender", "blood_type", "medical_condition",
        "insurance_provider", "admission_type", "medication"
    ]
    for col in cat_cols:
        le = encoders[col]
        val = df[col].astype(str).iloc[0]
        # If value wasn't seen during training, default to 0
        if val in le.classes_:
            df[col] = le.transform([val])
        else:
            df[col] = 0

    # ── Return only the features the model was trained on ─────────────────────
    return df[feature_cols]