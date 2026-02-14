"""
Pytest configuration — use SQLite in-memory DB so tests run without MySQL.

Strategy: Set DATABASE_URL=sqlite:// BEFORE app.db is imported.
db.py now detects SQLite and skips MySQL-specific connect_args.
"""
import os

# MUST be set before any app module is imported
os.environ["DATABASE_URL"] = "sqlite://"
os.environ.setdefault("GOOGLE_API_KEY", "")
os.environ.setdefault("APP_BASE_URL", "http://localhost:8000")

# Now safe to import — db.py will use SQLite
from app.db import engine
from app.models import Base

# Create all tables in the in-memory SQLite DB
Base.metadata.create_all(bind=engine)
