"""AI pipeline unit tests — mock mode only."""
import os
os.environ.setdefault("QWEN_BACKEND", "mock")

import numpy as np
import pytest

from ai.classifier import QwenClassifier
from ai.detector import YOLODetector
from ai.pipeline import AnalysisPipeline


def make_blank_frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


@pytest.mark.asyncio
async def test_qwen_classifier_mock():
    clf = QwenClassifier()
    frame = make_blank_frame()
    result = await clf.classify(frame)
    assert result.event_type in {"normal", "person", "intrusion", "anomaly", "theft", "fire", "vandalism"}
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.description, str)


def test_yolo_detector_mock():
    det = YOLODetector()
    frame = make_blank_frame()
    result = det.detect(frame)
    assert isinstance(result.has_person, bool)
    assert isinstance(result.person_count, int)
    assert isinstance(result.labels, list)


@pytest.mark.asyncio
async def test_pipeline_mock():
    pipeline = AnalysisPipeline()
    frame = make_blank_frame()
    result = await pipeline.analyze(frame)
    assert isinstance(result.should_alert, bool)
    assert isinstance(result.alert_reason, str)
    assert result.detection is not None
