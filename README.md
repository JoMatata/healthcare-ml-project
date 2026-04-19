# 🏥 Healthcare ML Pipeline & Prediction API

A production-grade healthcare analytics system that cleans and stores patient
data in PostgreSQL, retrains a machine learning model every Saturday at 12:00
noon using Apache Airflow, and serves predictions via a FastAPI REST API.

---

## 📋 Project Overview

| Component        | Technology                        |
|------------------|-----------------------------------|
| Data Source      | Kaggle Healthcare Dataset (55,500 records) |
| Database         | PostgreSQL                        |
| ML Models        | Random Forest + XGBoost           |
| API Framework    | FastAPI                           |
| Scheduler        | Apache Airflow 3                  |
| Package Manager  | uv                                |
| Deployment       | Railway                           |

---

## 🗂️ Project Structure
healthcare-ml-project/
├── app/                    # FastAPI application
│   ├── main.py             # App entry point
│   ├── routes.py           # API endpoints
│   ├── schemas.py          # Request/response models
│   ├── model_loader.py     # Model loading logic
│   └── utils.py            # Helper functions
├── data/                   # Data storage
│   └── cleaned_healthcare.csv
├── database/               # Database connection
│   └── db_connection.py
├── ml/                     # Machine learning
│   ├── train.py            # Training pipeline
│   ├── preprocess.py       # Feature engineering
│   └── evaluate.py         # Model evaluation
├── models/                 # Saved model artifacts
├── scripts/                # Data pipeline scripts
│   ├── ingest.py           # Kaggle download
│   ├── clean.py            # Data cleaning
│   └── load.py             # Load to PostgreSQL
├── airflow/
│   └── dags/
│       └── retrain_dag.py  # Weekly retraining DAG
├── frontend/
│   └── index.html          # Simple prediction UI
├── .env.example
├── pyproject.toml
└── README.md

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/healthcare-ml-project.git
cd healthcare-ml-project
```

### 2. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Create virtual environment and install dependencies
```bash
uv venv
source .venv/bin/activate
uv sync
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Edit .env with your database credentials
```

### 5. Set up PostgreSQL
```bash
sudo -u postgres psql
```
```sql
CREATE DATABASE healthcare_db;
CREATE USER healthcare_user WITH PASSWORD 'yourpassword';
GRANT ALL PRIVILEGES ON DATABASE healthcare_db TO healthcare_user;
\c healthcare_db
GRANT ALL ON SCHEMA public TO healthcare_user;
\q
```

### 6. Run the data pipeline
```bash
python scripts/ingest.py    # Download dataset from Kaggle
python scripts/clean.py     # Clean the data
python scripts/load.py      # Load into PostgreSQL
```

### 7. Train the model
```bash
python ml/train.py
```

### 8. Start the API
```bash
uvicorn app.main:app --reload
```

---

## 🔁 Airflow Retraining Schedule

The model retrains automatically every **Saturday at 12:00 noon**.

```bash
export AIRFLOW_HOME=$(pwd)/airflow
airflow standalone
```

Pipeline order:
clean_data → load_to_database → retrain_model → notify_complete

---

## 🚀 API Usage

### Base URL
### Base URL
https://web-production-aceb5.up.railway.app

### Endpoints

| Method | Endpoint   | Description              |
|--------|------------|--------------------------|
| GET    | `/`        | API status               |
| GET    | `/health`  | Model + DB health check  |
| POST   | `/predict` | Predict test result      |

### Example Request

```bash
curl -X POST "https://your-app.railway.app/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Age": 45,
       "Gender": "Male",
       "Blood_Type": "O+",
       "Medical_Condition": "Diabetes",
       "Billing_Amount": 2000.50,
       "Admission_Type": "Emergency",
       "Insurance_Provider": "Cigna",
       "Medication": "Aspirin",
       "Length_of_Stay": 5
     }'
```

### Example Response

```json
{
  "predicted_test_result": "Normal"
}
```

---

## 📊 Model Performance

| Model         | Accuracy | F1-Score |
|---------------|----------|----------|
| Random Forest | 43.15%   | 0.4313   |
| XGBoost       | 36.10%   | 0.3610   |

> **Note:** The dataset is synthetically generated, meaning test results were
> randomly assigned rather than derived from real clinical patterns. This limits
> model accuracy. The pipeline architecture is production-ready.

---

## 👤 Author

Built as a Pre-Internship project for LuxDevHQ Data Engineering program.