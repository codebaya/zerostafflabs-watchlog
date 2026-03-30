"""RTSP client — ffmpeg-based frame capture.

Supports Hikvision, Dahua, and Hanwha NVR/DVR streams using standard RTSP.
ffmpeg is invoked as a subprocess for maximum compatibility.
"""
import asyncio
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
_FFPROBE = shutil.which("ffprobe") or "ffprobe"


class RTSPError(Exception):
    pass


def probe_stream(rtsp_url: str, timeout: int = 5) -> dict:
    """Return basic stream info (width, height, fps) via ffprobe.

    Raises RTSPError on failure.
    """
    cmd = [
        _FFPROBE, "-v", "quiet", "-print_format", "json",
        "-show_streams", "-rtsp_transport", "tcp",
        "-timeout", str(timeout * 1_000_000),
        rtsp_url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 2)
    except FileNotFoundError:
        raise RTSPError("ffprobe not found — install ffmpeg")
    except subprocess.TimeoutExpired:
        raise RTSPError(f"ffprobe timed out connecting to {rtsp_url}")

    if result.returncode != 0:
        raise RTSPError(f"ffprobe error: {result.stderr[:200]}")

    import json
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not video:
        raise RTSPError("No video stream found")

    fps_raw = video.get("r_frame_rate", "25/1").split("/")
    fps = int(fps_raw[0]) / max(int(fps_raw[1]), 1)
    return {
        "width": video.get("width", 0),
        "height": video.get("height", 0),
        "fps": round(fps, 2),
        "codec": video.get("codec_name", "unknown"),
    }


class RTSPFrameCapture:
    """Async-friendly RTSP frame reader via OpenCV.

    Uses TCP transport for reliability with NVR/DVR streams.
    """

    def __init__(self, rtsp_url: str, reconnect_delay: float = 3.0):
        self.rtsp_url = rtsp_url
        self.reconnect_delay = reconnect_delay
        self._cap: Optional[cv2.VideoCapture] = None

    def _open(self) -> cv2.VideoCapture:
        # Force TCP transport — more reliable than UDP with NVRs
        cap = cv2.VideoCapture(
            f"rtspsrc location={self.rtsp_url} protocols=tcp latency=0 ! "
            "decodebin ! videoconvert ! appsink",
            cv2.CAP_GSTREAMER,
        )
        if not cap.isOpened():
            # Fallback to plain OpenCV RTSP (no GStreamer required)
            os_url = self.rtsp_url
            cap = cv2.VideoCapture(os_url)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        if not cap.isOpened():
            raise RTSPError(f"Cannot open RTSP stream: {self.rtsp_url}")
        return cap

    def open(self) -> None:
        self._cap = self._open()
        logger.info("RTSP stream opened: %s", self.rtsp_url)

    def close(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None

    def read_frame(self) -> Optional[np.ndarray]:
        """Read one frame. Returns None if the stream is unavailable."""
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok:
            logger.warning("Frame read failed on %s", self.rtsp_url)
            return None
        return frame

    async def async_read_frame(self) -> Optional[np.ndarray]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.read_frame)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()


async def capture_snapshot(rtsp_url: str, output_path: Path) -> Path:
    """Capture a single snapshot from an RTSP stream via ffmpeg subprocess.

    More reliable for one-shot captures than keeping a persistent connection.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _FFMPEG, "-y",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-frames:v", "1",
        "-q:v", "2",
        str(output_path),
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
    except asyncio.TimeoutError:
        raise RTSPError(f"Snapshot capture timed out: {rtsp_url}")
    except FileNotFoundError:
        raise RTSPError("ffmpeg not found — install ffmpeg")

    if proc.returncode != 0:
        raise RTSPError(f"ffmpeg snapshot error: {stderr.decode()[:200]}")

    return output_path
