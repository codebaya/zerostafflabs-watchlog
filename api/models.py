"""Pydantic request/response models."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, HttpUrl


class CameraCreate(BaseModel):
    name: str
    rtsp_url: str
    location: Optional[str] = None


class CameraResponse(BaseModel):
    id: int
    name: str
    location: Optional[str]
    enabled: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class AlertResponse(BaseModel):
    id: int
    camera_id: int
    camera_name: str
    event_type: str
    description: str
    confidence: float
    snapshot_path: Optional[str]
    telegram_sent: bool
    false_positive: Optional[bool]
    detected_at: datetime

    model_config = {"from_attributes": True}


class AlertFeedback(BaseModel):
    false_positive: bool


class TelegramCallback(BaseModel):
    update_id: int
    callback_query: Optional[dict] = None


class FrameAnalysisResponse(BaseModel):
    camera_id: int
    should_alert: bool
    alert_reason: str
    alert_id: Optional[int] = None
    event_type: Optional[str] = None
    confidence: Optional[float] = None
