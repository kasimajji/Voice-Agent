# Run with: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

from fastapi import FastAPI

from app.twilio_routes import router as twilio_router
from .db import Base, engine
from . import models
from .seed import seed_data

app = FastAPI(title="Sears Home Services Voice AI Agent")

app.include_router(twilio_router, prefix="/twilio")


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    seed_data()


@app.get("/health")
async def health_check():
    return {"status": "ok"}
