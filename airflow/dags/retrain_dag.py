"""
DAG: Healthcare ML Model — Weekly Retraining
Schedule: Every Saturday at 12:00 noon
Author: Your Name
"""

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# ── Make sure our project root is on the Python path ─────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load environment variables
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


# ── Task functions ─────────────────────────────────────────────────────────────

def task_clean_data():
    """Clean the raw dataset."""
    from scripts.clean import clean_data
    result = clean_data()
    print(f"Cleaned data shape: {result.shape}")


def task_load_to_db():
    """Load cleaned data into PostgreSQL."""
    from scripts.load import load_to_db
    load_to_db()


def task_train_model():
    """Retrain the ML model and save the best one."""
    from ml.train import train
    _, metrics = train()
    print(f"Retraining complete — F1: {metrics['f1_score']}")


def task_notify_complete(**context):
    """Log completion summary."""
    execution_date = context.get("logical_date", "N/A")
    print(f"""
    ╔══════════════════════════════════════════╗
    ║   ✅ Weekly Retraining Complete!          ║
    ║   📅 Run date: {execution_date}
    ║   🏥 Healthcare ML Pipeline              ║
    ╚══════════════════════════════════════════╝
    """)


# ── DAG Definition ─────────────────────────────────────────────────────────────

default_args = {
    "owner":            "healthcare-data-team",
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry":   False,
}

with DAG(
    dag_id="healthcare_weekly_retrain",
    description="Every Saturday at noon: clean → load → retrain ML model",
    schedule="0 12 * * 6",        # Cron: minute=0, hour=12, day=*, month=*, weekday=6(Saturday)
    start_date=datetime(2025, 1, 1),
    catchup=False,                 # Don't backfill missed runs
    default_args=default_args,
    tags=["healthcare", "ml", "retraining", "weekly"],
) as dag:

    clean = PythonOperator(
        task_id="clean_data",
        python_callable=task_clean_data,
    )

    load = PythonOperator(
        task_id="load_to_database",
        python_callable=task_load_to_db,
    )

    train = PythonOperator(
        task_id="retrain_model",
        python_callable=task_train_model,
    )

    notify = PythonOperator(
        task_id="notify_complete",
        python_callable=task_notify_complete,
    )

    # ── Pipeline order ──────────────────────────────────────────────────────
    # clean → load → train → notify
    clean >> load >> train >> notify