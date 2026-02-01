import os
from urllib.parse import quote_plus
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set. Please configure it in .env")

# Check if it's an ODBC connection string (Azure SQL format) and convert to SQLAlchemy URL
if DATABASE_URL.startswith("Driver=") or DATABASE_URL.startswith("{") or "ODBC Driver" in DATABASE_URL:
    # Azure SQL with ODBC - construct proper SQLAlchemy URL
    # Format: mssql+pyodbc:///?odbc_connect=<encoded_connection_string>
    encoded_conn = quote_plus(DATABASE_URL)
    DATABASE_URL = f"mssql+pyodbc:///?odbc_connect={encoded_conn}"

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
