"""Screen share AI analysis routes — POST frame, WebSocket stream, GET timeline."""
import base64
import io
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ai.classifier import QwenClassifier
from storage.database import (
    ScreenCapture,
    create_screen_capture,
    get_db,
    list_screen_captures,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["screen"])

_SCREEN_PROMPT = (
    "이 화면에서 사용자가 지금 무엇을 하고 있는지 한국어로 간결하게 설명해줘. "
    "앱 이름, 작업 내용, 특이사항을 포함해서 2-3문장으로."
)

_classifier: QwenClassifier | None = None


def _get_classifier() -> QwenClassifier:
    global _classifier
    if _classifier is None:
        _classifier = QwenClassifier()
    return _classifier


class ScreenFrameRequest(BaseModel):
    frame_b64: str


class ScreenCaptureResponse(BaseModel):
    id: int
    description: str
    app_context: str
    captured_at: str
    snapshot_path: Optional[str] = None

    @classmethod
    def from_orm(cls, capture: ScreenCapture) -> "ScreenCaptureResponse":
        return cls(
            id=capture.id,
            description=capture.description,
            app_context=capture.app_context,
            captured_at=capture.captured_at.isoformat(),
            snapshot_path=capture.snapshot_path,
        )


@router.post("/api/analysis/screen-frame", response_model=ScreenCaptureResponse)
async def analyze_screen_frame(
    req: ScreenFrameRequest,
    db: AsyncSession = Depends(get_db),
):
    """Analyze a base64-encoded JPEG screen frame with AI and store the result."""
    try:
        img_data = base64.b64decode(req.frame_b64)
        from PIL import Image
        img = Image.open(io.BytesIO(img_data))
        img.verify()
    except Exception as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=f"Invalid image data: {exc}")

    classifier = _get_classifier()
    description = await classifier.analyze_with_prompt(req.frame_b64, _SCREEN_PROMPT)

    capture = await create_screen_capture(db, description=description)
    return ScreenCaptureResponse.from_orm(capture)


@router.websocket("/ws/screen-stream")
async def screen_stream(websocket: WebSocket, db: AsyncSession = Depends(get_db)):
    """WebSocket endpoint for continuous screen frame analysis."""
    await websocket.accept()
    classifier = _get_classifier()

    try:
        while True:
            import json
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                frame_b64 = msg.get("frame_b64", "")
                if not frame_b64:
                    await websocket.send_json({"error": "frame_b64 required"})
                    continue

                description = await classifier.analyze_with_prompt(frame_b64, _SCREEN_PROMPT)
                capture = await create_screen_capture(db, description=description)

                await websocket.send_json({
                    "id": capture.id,
                    "description": capture.description,
                    "app_context": capture.app_context,
                    "captured_at": capture.captured_at.isoformat(),
                })
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
            except Exception as exc:
                logger.error("screen-stream error: %s", exc)
                await websocket.send_json({"error": str(exc)})
    except WebSocketDisconnect:
        logger.info("Screen stream WebSocket disconnected")


@router.get("/api/analysis/timeline", response_model=List[ScreenCaptureResponse])
async def get_timeline(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """Return recent screen analysis records."""
    captures = await list_screen_captures(db, limit=limit, offset=offset)
    return [ScreenCaptureResponse.from_orm(c) for c in captures]
