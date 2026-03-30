"""Real-time camera watcher.

Connects to an RTSP stream, runs the AI pipeline on each sampled frame,
persists alerts to DB, and sends Telegram notifications.

Run alongside the FastAPI server or as a standalone process.
"""
import asyncio
import logging
from datetime import datetime
from pathlib import Path

from config import settings
from ai.pipeline import AnalysisPipeline
from notifications.telegram import send_alert
from rtsp.client import RTSPError
from rtsp.frame_extractor import FrameExtractor
from storage.database import async_session, create_alert, list_cameras, mark_alert_sent

logger = logging.getLogger(__name__)


async def watch_camera(camera_id: int, camera_name: str, rtsp_url: str, sample_interval: float = 5.0):
    """Continuously watch one camera and raise alerts."""
    pipeline = AnalysisPipeline()
    extractor = FrameExtractor(rtsp_url, sample_interval=sample_interval)

    logger.info("Watching camera %d (%s) @ %s", camera_id, camera_name, rtsp_url)

    async for ts, frame in extractor.stream_frames():
        try:
            result = await pipeline.analyze(frame)

            if not result.should_alert:
                continue

            # Save annotated snapshot
            annotated = pipeline.detector.draw_boxes(frame, result.detection)
            snapshot_path = extractor.save_frame(annotated, ts, camera_id)

            event_type = (
                result.classification.event_type
                if result.classification
                else ("person" if result.detection.has_person else "anomaly")
            )
            confidence = (
                result.classification.confidence if result.classification
                else (result.detection.detections[0].confidence if result.detection.detections else 0.5)
            )

            async with async_session() as db:
                alert = await create_alert(
                    db,
                    camera_id=camera_id,
                    camera_name=camera_name,
                    event_type=event_type,
                    description=result.alert_reason,
                    confidence=confidence,
                    snapshot_path=str(snapshot_path),
                )

                sent = await send_alert(
                    camera_name=camera_name,
                    event_type=event_type,
                    description=result.alert_reason,
                    confidence=confidence,
                    snapshot_path=snapshot_path,
                    alert_id=alert.id,
                )
                if sent:
                    await mark_alert_sent(db, alert.id)

            logger.warning(
                "ALERT cam=%d type=%s conf=%.2f desc=%s",
                camera_id, event_type, confidence, result.alert_reason[:80],
            )

        except Exception as exc:
            logger.error("Error processing frame from camera %d: %s", camera_id, exc)


async def watch_all_cameras(sample_interval: float = 5.0):
    """Start watchers for all enabled cameras in DB."""
    from storage.database import init_db
    await init_db()

    async with async_session() as db:
        cameras = await list_cameras(db)

    if not cameras:
        logger.warning("No cameras configured — add cameras via POST /cameras")
        return

    tasks = [
        asyncio.create_task(
            watch_camera(cam.id, cam.name, cam.rtsp_url, sample_interval)
        )
        for cam in cameras if cam.enabled
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(watch_all_cameras())
