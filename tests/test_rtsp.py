"""RTSP client unit tests (no real camera required)."""
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from rtsp.client import RTSPError, RTSPFrameCapture


def test_rtsp_frame_capture_open_failure():
    with patch("rtsp.client.cv2.VideoCapture") as mock_cap:
        instance = mock_cap.return_value
        instance.isOpened.return_value = False
        cap = RTSPFrameCapture("rtsp://fake")
        with pytest.raises(RTSPError, match="Cannot open RTSP"):
            cap.open()


def test_rtsp_frame_capture_read_when_closed():
    cap = RTSPFrameCapture("rtsp://fake")
    assert cap.read_frame() is None


def test_rtsp_frame_capture_read_ok():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with patch("rtsp.client.cv2.VideoCapture") as mock_cap:
        instance = mock_cap.return_value
        instance.isOpened.return_value = True
        instance.read.return_value = (True, frame)
        cap = RTSPFrameCapture("rtsp://fake")
        cap._cap = instance
        result = cap.read_frame()
        assert result is not None
        assert result.shape == (480, 640, 3)
