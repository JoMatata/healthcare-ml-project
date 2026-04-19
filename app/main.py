from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routes import router
from app.model_loader import load_artifacts
import os

app = FastAPI(
    title=" Healthcare ML API",
    description="Predicts patient test results as Normal, Abnormal, or Inconclusive.",
    version="1.0.0",
)

# Allow frontend to talk to the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model into memory when the app starts
@app.on_event("startup")
def startup_event():
    load_artifacts()

# Include all routes
app.include_router(router)

@app.get("/")
def root():
    return {
        "message": "Healthcare ML API is running 🏥",
        "docs":    "/docs",
        "predict": "/predict",
        "health":  "/health"
    }