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
    """Run setup pipeline if model doesn't exist, then load artifacts."""

    if not os.path.exists("models/model.joblib"):
        print("🚀 No model found — running full setup pipeline...")
        try:
            import sys
            sys.path.insert(0, ".")

            # Set up Kaggle credentials from env
            import json
            username = os.getenv("KAGGLE_USERNAME")
            key      = os.getenv("KAGGLE_KEY")

            if username and key:
                kaggle_dir = os.path.expanduser("~/.kaggle")
                os.makedirs(kaggle_dir, exist_ok=True)
                creds_path = os.path.join(kaggle_dir, "kaggle.json")
                with open(creds_path, "w") as f:
                    json.dump({"username": username, "key": key}, f)
                os.chmod(creds_path, 0o600)
                print("✅ Kaggle credentials configured")

                # Run pipeline
                from scripts.ingest import download_dataset
                download_dataset()

                from scripts.clean import clean_data
                clean_data()

                from scripts.load import load_to_db
                load_to_db()

                from ml.train import train
                train()

                print("✅ Pipeline complete!")

            else:
                print("❌ KAGGLE_USERNAME or KAGGLE_KEY not set in environment")

        except Exception as e:
            print(f"❌ Setup failed: {e}")
            import traceback
            traceback.print_exc()

    # Load model artifacts
    if os.path.exists("models/model.joblib"):
        from app.model_loader import load_artifacts
        load_artifacts()
        print("✅ Model loaded and API is ready!")
    else:
        print("⚠️  Model still not found — /predict will not work")

app.include_router(router)

@app.get("/")
def root():
    return {
        "message": "Healthcare ML API is running 🏥",
        "docs":    "/docs",
        "predict": "/predict",
        "health":  "/health"
    }