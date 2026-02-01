import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Get DATABASE_URL from environment, default to SQLite
_env_url = os.getenv("DATABASE_URL", "")

# Ignore Azure SQL / ODBC connection strings - use SQLite instead
if not _env_url or "ODBC" in _env_url or "Driver=" in _env_url or "mssql" in _env_url:
    DATABASE_URL = "sqlite:///./voice_ai.db"
    print("[DB] Using local SQLite database: voice_ai.db")
else:
    DATABASE_URL = _env_url

# SQLite requires check_same_thread=False for FastAPI compatibility
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
