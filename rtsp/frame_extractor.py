"""Frame extraction pipeline.

Batch mode: extract frames from RTSP at a configured interval.
Designed for night-time batch runs (02:00–06:00).
"""
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional

import cv2
import numpy as np

from config import settings
from rtsp.client import RTSPFrameCapture, RTSPError

logger = logging.getLogger(__name__)


class FrameExtractor:
    """Extract frames from an RTSP stream at a given interval (seconds)."""

    def __init__(
        self,
        rtsp_url: str,
        sample_interval: float = 5.0,
        output_dir: Optional[Path] = None,
    ):
        self.rtsp_url = rtsp_url
        self.sample_interval = sample_interval
        self.output_dir = output_dir or Path(settings.clip_storage_dir) / "frames"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def stream_frames(self) -> AsyncGenerator[tuple[datetime, np.ndarray], None]:
        """Yield (timestamp, frame) pairs at the configured sample interval."""
        cap = RTSPFrameCapture(self.rtsp_url)
        try:
            cap.open()
        except RTSPError as exc:
            logger.error("Cannot open RTSP for extraction: %s", exc)
            return

        try:
            while True:
                ts = datetime.utcnow()
                frame = await cap.async_read_frame()
                if frame is not None:
                    yield ts, frame
                else:
                    logger.warning("Null frame; attempting reconnect in %.1fs", self.sample_interval)

                await asyncio.sleep(self.sample_interval)
        finally:
            cap.close()

    def save_frame(self, frame: np.ndarray, ts: datetime, camera_id: int) -> Path:
        fname = f"cam{camera_id}_{ts.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        path = self.output_dir / fname
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return path

    async def extract_batch(
        self,
        camera_id: int,
        duration_seconds: float,
        sample_interval: Optional[float] = None,
    ) -> list[Path]:
        """Extract frames for a fixed duration. Used by the night batch scheduler."""
        interval = sample_interval or self.sample_interval
        collected: list[Path] = []
        deadline = asyncio.get_running_loop().time() + duration_seconds

        async for ts, frame in self.stream_frames():
            path = self.save_frame(frame, ts, camera_id)
            collected.append(path)
            logger.debug("Saved frame %s", path.name)
            if asyncio.get_running_loop().time() >= deadline:
                break

        logger.info("Batch extraction done: %d frames for camera %d", len(collected), camera_id)
        return collected
