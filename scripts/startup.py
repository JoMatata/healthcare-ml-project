"""
Railway startup script.
Runs once on deployment to set up the database and train the model.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

def setup():
    print("🚀 Running startup setup...")

    # Step 1 — Download dataset
    print("\n📥 Step 1: Downloading dataset...")
    from scripts.ingest import download_dataset
    download_dataset()

    # Step 2 — Clean data
    print("\n🧹 Step 2: Cleaning data...")
    from scripts.clean import clean_data
    clean_data()

    # Step 3 — Load to database
    print("\n🗄️  Step 3: Loading to database...")
    from scripts.load import load_to_db
    load_to_db()

    # Step 4 — Train model
    print("\n🤖 Step 4: Training model...")
    from ml.train import train
    train()

    print("\n✅ Startup setup complete!")

if __name__ == "__main__":
    setup()