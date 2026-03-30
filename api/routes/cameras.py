"""Camera management routes."""
import asyncio
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from api.models import CameraCreate, CameraResponse
from rtsp.client import RTSPError, probe_stream
from storage.database import (
    create_camera,
    get_camera,
    get_db,
    list_cameras,
)

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
