# Run with: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.twilio_routes import router as twilio_router
from app.upload_routes import router as upload_router
from .db import Base, engine
from . import models
from .seed import seed_data
from .logging_config import get_logger

logger = get_logger("main")

app = FastAPI(title="Sears Home Services Voice AI Agent")

app.include_router(twilio_router, prefix="/twilio")
app.include_router(upload_router)

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


@app.on_event("startup")
def on_startup():
    logger.info("🚀 Starting Voice Agent application...")
    Base.metadata.create_all(bind=engine)
    seed_data()
    logger.info("✅ Database initialized and seeded")


@app.get("/health")
async def health_check():
    return {"status": "ok"}
