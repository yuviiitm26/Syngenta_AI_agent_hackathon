# app/utils/db.py

from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)

def test_connection():
    with engine.connect() as conn:
        result = conn.execute("SELECT 1;")
        print("PostgreSQL connected:", result.fetchone())
