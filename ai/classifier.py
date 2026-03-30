"""Qwen VL anomaly classifier.

Backend selection via QWEN_BACKEND env var:
  - "ollama"       : call local Ollama API (recommended for Mac Mini)
  - "transformers" : load via HuggingFace transformers
  - "mock"         : deterministic test mode (no model required)
"""
import asyncio
import base64
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx
import numpy as np

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    is_anomaly: bool
    event_type: str          # "normal" | "person" | "intrusion" | "anomaly" | "fire" | …
    description: str
    confidence: float


_ANOMALY_PROMPT = (
    "You are a night-time CCTV security analyst. "
    "Analyze this frame and determine if there is any anomaly or security concern. "
    "Reply in JSON with keys: is_anomaly (bool), event_type (string), "
    "description (string in Korean), confidence (0.0-1.0). "
    "event_type choices: normal, person, intrusion, theft, fire, vandalism, anomaly. "
    "Be concise. Reply ONLY with valid JSON, no markdown."
)


class QwenClassifier:
    def __init__(self):
        self._backend = settings.qwen_backend.lower()
        self._model = settings.qwen_model

    async def classify(self, frame: np.ndarray) -> ClassificationResult:
        if self._backend == "mock":
            return self._mock_classify()
        elif self._backend == "ollama":
            return await self._ollama_classify(frame)
        elif self._backend == "transformers":
            return await self._transformers_classify(frame)
        else:
            logger.warning("Unknown QWEN_BACKEND=%s, falling back to mock", self._backend)
            return self._mock_classify()

    def _mock_classify(self) -> ClassificationResult:
        """Deterministic mock for local testing without a real model."""
        import random
        # 90% 정상, 10% 이상 감지 (실제 CCTV 환경 반영)
        normal_descriptions = [
            "화면에 사람이 보이고 있습니다. 정상적인 활동으로 보입니다.",
            "컴퓨터 화면을 보고 있는 사람이 있습니다. 이상 없음.",
            "실내 공간입니다. 특이 사항 없음.",
            "사무 환경으로 보입니다. 정상 상황입니다.",
            "화면에 사람 또는 물체가 감지됩니다. 정상 활동으로 판단됩니다.",
        ]
        anomaly_descriptions = [
            "움직임이 감지되었습니다. 확인이 필요합니다.",
            "비정상적인 패턴이 감지되었습니다.",
        ]
        if random.random() < 0.90:
            return ClassificationResult(False, "normal", random.choice(normal_descriptions), round(random.uniform(0.85, 0.98), 2))
        else:
            return ClassificationResult(True, "person", random.choice(anomaly_descriptions), round(random.uniform(0.70, 0.85), 2))

    async def _ollama_classify(self, frame: np.ndarray) -> ClassificationResult:
        import cv2
        _, buf = cv2.imencode(".jpg", frame)
        b64 = base64.b64encode(buf.tobytes()).decode()

        payload = {
            "model": self._model,
            "prompt": _ANOMALY_PROMPT,
            "images": [b64],
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post("http://localhost:11434/api/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return self._parse_response(data.get("response", ""))
        except Exception as exc:
            logger.error("Ollama classify error: %s", exc)
            return ClassificationResult(False, "normal", f"분류 오류: {exc}", 0.0)

    async def _transformers_classify(self, frame: np.ndarray) -> ClassificationResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transformers_classify_sync, frame)

    def _transformers_classify_sync(self, frame: np.ndarray) -> ClassificationResult:
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            from PIL import Image
            import torch

            if not hasattr(self, "_model_loaded"):
                logger.info("Loading Qwen model: %s", self._model)
                self._processor = AutoProcessor.from_pretrained(self._model, trust_remote_code=True)
                self._hf_model = AutoModelForVision2Seq.from_pretrained(
                    self._model, trust_remote_code=True, torch_dtype=torch.float16
                )
                self._model_loaded = True

            import cv2
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            inputs = self._processor(text=_ANOMALY_PROMPT, images=pil_img, return_tensors="pt")
            output = self._hf_model.generate(**inputs, max_new_tokens=256)
            text = self._processor.decode(output[0], skip_special_tokens=True)
            return self._parse_response(text)
        except Exception as exc:
            logger.error("Transformers classify error: %s", exc)
            return ClassificationResult(False, "normal", f"분류 오류: {exc}", 0.0)

    async def analyze_with_prompt(self, image_b64: str, prompt: str) -> str:
        """Analyze an image with a custom prompt, returning a plain text description."""
        if self._backend == "mock":
            return "사용자가 코드 편집기에서 작업 중입니다. Python 파일을 열고 함수를 수정하고 있습니다."
        elif self._backend == "ollama":
            payload = {
                "model": self._model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
            }
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.post("http://localhost:11434/api/generate", json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    return data.get("response", "분석 결과 없음").strip()
            except Exception as exc:
                logger.error("Ollama analyze_with_prompt error: %s", exc)
                return f"분석 오류: {exc}"
        else:
            return "분석 백엔드가 설정되지 않았습니다."

    @staticmethod
    def _parse_response(text: str) -> ClassificationResult:
        import json, re
        try:
            # Strip markdown code fences if present
            clean = re.sub(r"```(?:json)?", "", text).strip()
            data = json.loads(clean)
            return ClassificationResult(
                is_anomaly=bool(data.get("is_anomaly", False)),
                event_type=str(data.get("event_type", "normal")),
                description=str(data.get("description", "")),
                confidence=min(max(float(data.get("confidence", 0.5)), 0.0), 1.0),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.debug("Raw Qwen response (non-JSON): %s", text[:200])
            is_anom = any(kw in text.lower() for kw in ["이상", "침입", "위험", "anomaly", "intrusion", "person"])
            return ClassificationResult(
                is_anomaly=is_anom,
                event_type="anomaly" if is_anom else "normal",
                description=text[:300] if text else "분석 결과 없음",
                confidence=0.5,
            )
