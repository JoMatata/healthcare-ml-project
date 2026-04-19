from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    Age:               float  = Field(..., example=45)
    Gender:            str    = Field(..., example="Male")
    Blood_Type:        str    = Field(..., example="O+")
    Medical_Condition: str    = Field(..., example="Diabetes")
    Billing_Amount:    float  = Field(..., example=2000.50)
    Admission_Type:    str    = Field(..., example="Emergency")
    Insurance_Provider:str    = Field(..., example="Cigna")
    Medication:        str    = Field(..., example="Aspirin")
    Length_of_Stay:    int    = Field(..., example=5)

    class Config:
        # Allow the exact field names from the brief
        populate_by_name = True


class PredictionOutput(BaseModel):
    predicted_test_result: str


class HealthResponse(BaseModel):
    status:     str
    model:      str
    accuracy:   float
    records_in_db: int