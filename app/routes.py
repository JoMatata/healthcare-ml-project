from fastapi import APIRouter, HTTPException, Body
from app.schemas import PatientInput, PredictionOutput, HealthResponse
from app.model_loader import get_artifacts
from app.utils import prepare_input
from database.db_connection import get_engine
from sqlalchemy import text
import json
import os

router = APIRouter()


from fastapi import APIRouter, HTTPException, Body

@router.post("/predict", response_model=PredictionOutput)
def predict(patient: PatientInput = Body(..., 
    examples={
        "Age": 45,
        "Gender": "Male",
        "Blood_Type": "O+",
        "Medical_Condition": "Diabetes",
        "Billing_Amount": 2000.50,
        "Admission_Type": "Emergency",
        "Insurance_Provider": "Cigna",
        "Medication": "Aspirin",
        "Length_of_Stay": 5
    }
)):
    try:
        model, encoders, target_encoder, feature_cols = get_artifacts()
        X = prepare_input(patient, encoders, feature_cols)
        pred_encoded = model.predict(X)[0]
        pred_label   = target_encoder.inverse_transform([pred_encoded])[0]
        return PredictionOutput(predicted_test_result=pred_label)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
def health():
    try:
        # Get record count from DB
        engine = get_engine()
        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM patients")).scalar()

        # Get model metadata
        metadata = {}
        if os.path.exists("models/metadata.json"):
            with open("models/metadata.json") as f:
                metadata = json.load(f)

        return HealthResponse(
            status       = "healthy",
            model        = metadata.get("best_model", "Unknown"),
            accuracy     = metadata.get("metrics", {}).get("accuracy", 0.0),
            records_in_db= count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))