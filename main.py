"""WatchLog — FastAPI application entry point."""
import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import health, cameras, alerts, telegram_webhook
from config import settings
from storage.database import init_db

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="WatchLog",
    description="AI-powered CCTV monitoring — RTSP → AI analysis → Telegram alerts",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(cameras.router)
app.include_router(alerts.router)
app.include_router(telegram_webhook.router)


@app.on_event("startup")
async def startup():
    logger.info("WatchLog starting up…")
    await init_db()
    logger.info("Database initialised")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
