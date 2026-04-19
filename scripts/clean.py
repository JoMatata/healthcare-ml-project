import pandas as pd
import os

def clean_data():
    raw_path = "data/raw/healthcare.csv"
    clean_path = "data/cleaned_healthcare.csv"

    print(" Loading raw data...")
    df = pd.read_csv(raw_path)
    print(f"   Raw shape: {df.shape}")

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    print(" Column names standardized")

    df.drop(columns=["name", "doctor", "hospital", "room_number"], inplace=True)
    print(" Irrelevant columns dropped")

    df["date_of_admission"] = pd.to_datetime(df["date_of_admission"])
    df["discharge_date"]    = pd.to_datetime(df["discharge_date"])
    print(" Date columns converted")

    df["length_of_stay"] = (df["discharge_date"] - df["date_of_admission"]).dt.days
    df.drop(columns=["date_of_admission", "discharge_date"], inplace=True)
    print(" length_of_stay feature created")

    cat_cols = ["gender", "blood_type", "medical_condition",
                "admission_type", "insurance_provider", "medication"]
    for col in cat_cols:
        df[col] = df[col].str.strip().str.title()
    print(" Categorical values standardized")

    df["test_results"] = df["test_results"].str.strip().str.title()
    valid = ["Normal", "Abnormal", "Inconclusive"]
    before = len(df)
    df = df[df["test_results"].isin(valid)]
    print(f" Target column cleaned — removed {before - len(df)} invalid rows")

    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f" Duplicates removed — dropped {before - len(df)} rows")

    df.reset_index(drop=True, inplace=True)

    os.makedirs("data", exist_ok=True)
    df.to_csv(clean_path, index=False)

    print(f"\n Cleaned data saved → {clean_path}")
    print(f"   Final shape: {df.shape}")
    print(f"\n   Columns: {df.columns.tolist()}")
    print(f"\n   Test Results distribution:\n{df['test_results'].value_counts()}")

    return df

if __name__ == "__main__":
    clean_data()