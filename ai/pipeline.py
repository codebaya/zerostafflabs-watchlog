"""Combined AI analysis pipeline.

Runs YOLO detection first (fast), then Qwen classification (slow) only when
YOLO finds a person or the frame looks interesting.
"""
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ai.classifier import ClassificationResult, QwenClassifier
from ai.detector import DetectionResult, YOLODetector

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    detection: DetectionResult
    classification: Optional[ClassificationResult]
    should_alert: bool
    alert_reason: str


class AnalysisPipeline:
    """Run YOLO → Qwen on a single frame and decide whether to raise an alert."""

    def __init__(self):
        self.detector = YOLODetector()
        self.classifier = QwenClassifier()

    async def analyze(self, frame: np.ndarray) -> AnalysisResult:
        # Step 1 — fast YOLO pass (sync but cheap)
        loop = asyncio.get_running_loop()
        detection = await loop.run_in_executor(None, self.detector.detect, frame)

        classification: Optional[ClassificationResult] = None
        should_alert = False
        alert_reason = ""

        # Step 2 — Qwen VL if YOLO found something or always-on mode
        trigger_qwen = detection.has_person or detection.detections
        if trigger_qwen:
            classification = await self.classifier.classify(frame)
            if classification.is_anomaly and classification.confidence >= 0.5:
                should_alert = True
                alert_reason = classification.description
        elif detection.has_person:
            should_alert = True
            alert_reason = f"사람 {detection.person_count}명 감지 (YOLO)"

        return AnalysisResult(
            detection=detection,
            classification=classification,
            should_alert=should_alert,
            alert_reason=alert_reason,
        )
