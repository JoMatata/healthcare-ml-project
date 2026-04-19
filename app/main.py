from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router
import os

app = FastAPI(
    title="🏥 Healthcare ML API",
    description="Predicts patient test results as Normal, Abnormal, or Inconclusive.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    # Only load model if it exists
    if os.path.exists("models/model.joblib"):
        from app.model_loader import load_artifacts
        load_artifacts()
    else:
        print("⚠️  No model found — run ml/train.py first")

app.include_router(router)

@app.get("/")
def root():
    return {
        "message": "Healthcare ML API is running 🏥",
        "docs":    "/docs",
        "predict": "/predict",
        "health":  "/health"
    }