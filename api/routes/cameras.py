"""Camera management routes."""
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from ai.pipeline import AnalysisPipeline
from api.models import CameraCreate, CameraResponse, FrameAnalysisResponse
from config import settings
from notifications.telegram import send_alert
from rtsp.client import RTSPError, probe_stream
from storage.database import (
    create_alert,
    create_camera,
    get_camera,
    get_db,
    list_cameras,
    mark_alert_sent,
)

logger = logging.getLogger(__name__)
_pipeline: Optional[AnalysisPipeline] = None


def _get_pipeline() -> AnalysisPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AnalysisPipeline()
    return _pipeline

router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.get("", response_model=List[CameraResponse])
async def get_cameras(db: AsyncSession = Depends(get_db)):
    cameras = await list_cameras(db)
    return cameras


@router.get("/{camera_id}", response_model=CameraResponse)
async def get_camera_detail(camera_id: int, db: AsyncSession = Depends(get_db)):
    cam = await get_camera(db, camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    return cam


@router.post("", response_model=CameraResponse, status_code=201)
async def add_camera(body: CameraCreate, db: AsyncSession = Depends(get_db)):
    cam = await create_camera(db, name=body.name, rtsp_url=body.rtsp_url, location=body.location)
    return cam


@router.post("/{camera_id}/probe", tags=["cameras"])
async def probe_camera(camera_id: int, db: AsyncSession = Depends(get_db)):
    """Test RTSP connectivity and return stream metadata."""
    cam = await get_camera(db, camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    try:
        loop = asyncio.get_running_loop()
        info = await loop.run_in_executor(None, probe_stream, cam.rtsp_url)
        return {"status": "ok", "stream_info": info}
    except RTSPError as exc:
        return {"status": "error", "detail": str(exc)}


@router.post("/frame", response_model=FrameAnalysisResponse, tags=["cameras"])
async def submit_frame(
    camera_id: int = Form(...),
    frame: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Accept a JPEG frame from a browser webcam and run it through the AI pipeline.

    Browser sends frames via multipart/form-data at 1–5 second intervals.
    If an anomaly is detected, an alert is created in DB and a Telegram notification is sent.
    """
    cam = await get_camera(db, camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    raw = await frame.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=422, detail="Invalid image data — expected JPEG/PNG")

    pipeline = _get_pipeline()
    result = await pipeline.analyze(img)

    alert_id: Optional[int] = None
    event_type: Optional[str] = None
    confidence: Optional[float] = None

    if result.should_alert:
        event_type = (
            result.classification.event_type if result.classification
            else ("person" if result.detection.has_person else "anomaly")
        )
        confidence = (
            result.classification.confidence if result.classification
            else (result.detection.detections[0].confidence if result.detection.detections else 0.5)
        )

        # Save annotated snapshot
        snap_dir = Path(settings.clip_storage_dir) / "frames"
        snap_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow()
        annotated = pipeline.detector.draw_boxes(img, result.detection)
        snap_path = snap_dir / f"cam{camera_id}_{ts.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        cv2.imwrite(str(snap_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])

        alert = await create_alert(
            db,
            camera_id=camera_id,
            camera_name=cam.name,
            event_type=event_type,
            description=result.alert_reason,
            confidence=confidence,
            snapshot_path=str(snap_path),
        )
        alert_id = alert.id

        sent = await send_alert(
            camera_name=cam.name,
            event_type=event_type,
            description=result.alert_reason,
            confidence=confidence,
            snapshot_path=snap_path,
            alert_id=alert.id,
        )
        if sent:
            await mark_alert_sent(db, alert.id)

        logger.warning("ALERT (frame) cam=%d type=%s conf=%.2f", camera_id, event_type, confidence)

    desc = None
    if result.classification:
        desc = result.classification.description
    return FrameAnalysisResponse(
        camera_id=camera_id,
        should_alert=result.should_alert,
        description=desc,
        alert_reason=result.alert_reason,
        alert_id=alert_id,
        event_type=event_type,
        confidence=confidence,
    )


@router.post("/virtual/snapshot", response_model=FrameAnalysisResponse, tags=["cameras"])
async def virtual_snapshot(camera_id: int = 0):
    """Generate a synthetic noise frame and run it through the AI pipeline.

    Useful for testing the analysis pipeline without a real camera.
    No alert is persisted — this is a dry-run endpoint.
    """
    # 640×480 BGR noise image — same shape the pipeline expects
    img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    pipeline = _get_pipeline()
    result = await pipeline.analyze(img)

    event_type: Optional[str] = None
    confidence: Optional[float] = None
    if result.should_alert:
        event_type = (
            result.classification.event_type if result.classification
            else ("person" if result.detection.has_person else "anomaly")
        )
        confidence = (
            result.classification.confidence if result.classification
            else (result.detection.detections[0].confidence if result.detection.detections else 0.5)
        )

    desc = None
    if result.classification:
        desc = result.classification.description
    return FrameAnalysisResponse(
        camera_id=camera_id,
        should_alert=result.should_alert,
        description=desc,
        alert_reason=result.alert_reason,
        alert_id=None,
        event_type=event_type,
        confidence=confidence,
    )
