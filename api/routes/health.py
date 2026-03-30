from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: str
    version: str


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    return HealthResponse(
        status="ok",
        service="WatchLog",
        timestamp=datetime.utcnow().isoformat() + "Z",
        version="0.1.0",
    )
