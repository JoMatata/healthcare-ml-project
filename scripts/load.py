import pandas as pd
from sqlalchemy import text
from database.db_connection import get_engine

def load_to_db():
    print(" Loading cleaned data into PostgreSQL...")

    df = pd.read_csv("data/cleaned_healthcare.csv")
    engine = get_engine()

    with engine.connect() as conn:
        # ── Create table if it doesn't exist ─────────────────────────────────
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS patients (
                id                 SERIAL PRIMARY KEY,
                age                INTEGER,
                gender             VARCHAR(10),
                blood_type         VARCHAR(5),
                medical_condition  VARCHAR(50),
                insurance_provider VARCHAR(50),
                billing_amount     FLOAT,
                admission_type     VARCHAR(20),
                medication         VARCHAR(50),
                test_results       VARCHAR(20),
                length_of_stay     INTEGER
            );
        """))

        # ── Avoid duplicate records using a staging approach ──────────────────
        conn.execute(text("CREATE TEMP TABLE staging (LIKE patients INCLUDING ALL);"))

        # Load into staging first
        df.to_sql("staging", conn, if_exists="append", index=False,
                  method="multi", chunksize=1000)

        # Insert only records not already in patients
        result = conn.execute(text("""
            INSERT INTO patients (age, gender, blood_type, medical_condition,
                                  insurance_provider, billing_amount, admission_type,
                                  medication, test_results, length_of_stay)
            SELECT age, gender, blood_type, medical_condition,
                   insurance_provider, billing_amount, admission_type,
                   medication, test_results, length_of_stay
            FROM staging
            ON CONFLICT DO NOTHING;
        """))

        conn.execute(text("DROP TABLE staging;"))
        conn.commit()

    # Verify
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM patients")).scalar()

    print(f" Done! Total records in database: {count:,}")

if __name__ == "__main__":
    load_to_db()