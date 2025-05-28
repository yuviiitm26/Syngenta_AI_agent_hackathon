from sqlalchemy import create_engine
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

def load_csv_to_postgres():
    try:
        csv_path = "data/dataco.csv"
        table_name = "orders"

        df = pd.read_csv(csv_path, encoding='latin1')

        # Clean column names
        df.columns = [col.lower().replace(" ", "_").replace("(", "").replace(")", "") for col in df.columns]

        df.to_sql(table_name, con=engine, if_exists="replace", index=False)
        print(f"Loaded {len(df)} rows into table '{table_name}'.")
    except Exception as e:
        print(f"Error loading CSV: {e}")
