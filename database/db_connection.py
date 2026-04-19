import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

def get_engine():
    # Try all possible Railway database URL variable names
    database_url = (
        os.getenv("DATABASE_URL") or
        os.getenv("DATABASE_PUBLIC_URL")
    )

    if database_url:
        database_url = database_url.replace("postgres://", "postgresql://")
        print("✅ Connected via DATABASE_URL")
        engine = create_engine(database_url)

    elif os.getenv("PGHOST"):
        # Use individual PG variables Railway provides
        host     = os.getenv("PGHOST")
        port     = os.getenv("PGPORT", "5432")
        db       = os.getenv("PGDATABASE")
        user     = os.getenv("PGUSER")
        password = os.getenv("PGPASSWORD")
        url      = f"postgresql://{user}:{password}@{host}:{port}/{db}"
        print(f"✅ Connected via PG variables: {user}@{host}:{port}/{db}")
        engine   = create_engine(url)

    else:
        # Local development fallback
        user     = os.getenv("DB_USER")
        password = os.getenv("DB_PASSWORD")
        host     = os.getenv("DB_HOST", "localhost")
        port     = os.getenv("DB_PORT", "5432")
        db       = os.getenv("DB_NAME")
        print(f"⚠️  Using local vars: {user}@{host}:{port}/{db}")
        url      = f"postgresql://{user}:{password}@{host}:{port}/{db}"
        engine   = create_engine(url)

    return engine

def test_connection():
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Database connection successful!")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    test_connection()