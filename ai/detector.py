"""YOLO object detector.

Uses ultralytics (YOLOv8/v11 API) — the model path/name is configurable.
When YOLO_MODEL_PATH is empty, downloads the default model automatically.
Falls back to a mock detector if ultralytics is not installed.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class DetectionResult:
    detections: List[Detection] = field(default_factory=list)
    has_person: bool = False
    person_count: int = 0
    labels: List[str] = field(default_factory=list)


class YOLODetector:
    def __init__(self):
        self._model = None
        self._model_path = settings.yolo_model_path or "yolo11n.pt"  # default lightweight model
        self._conf = settings.yolo_confidence

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)
            logger.info("YOLO model loaded: %s", self._model_path)
        except ImportError:
            logger.warning("ultralytics not installed — using mock YOLO detector")
            self._model = "mock"

    def detect(self, frame: np.ndarray) -> DetectionResult:
        self._load_model()
        if self._model == "mock":
            return self._mock_detect()
        return self._yolo_detect(frame)

    def _yolo_detect(self, frame: np.ndarray) -> DetectionResult:
        import cv2
        try:
            results = self._model(frame, conf=self._conf, verbose=False)
            detections: List[Detection] = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = self._model.names.get(cls_id, str(cls_id))
                    detections.append(Detection(label=label, confidence=conf, bbox=(x1, y1, x2, y2)))

            persons = [d for d in detections if d.label == "person"]
            return DetectionResult(
                detections=detections,
                has_person=len(persons) > 0,
                person_count=len(persons),
                labels=list({d.label for d in detections}),
            )
        except Exception as exc:
            logger.error("YOLO detect error: %s", exc)
            return DetectionResult()

    @staticmethod
    def _mock_detect() -> DetectionResult:
        import random
        if random.random() < 0.3:
            det = Detection(label="person", confidence=0.87, bbox=(100, 80, 300, 480))
            return DetectionResult(
                detections=[det], has_person=True, person_count=1, labels=["person"]
            )
        return DetectionResult()

    def draw_boxes(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        import cv2
        out = frame.copy()
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            color = (0, 0, 255) if det.label == "person" else (0, 255, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                out, f"{det.label} {det.confidence:.2f}",
                (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )
        return out
