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
    "당신은 CCTV 영상 분석 전문가입니다. "
    "화면에 보이는 모든 것을 구체적으로 묘사하세요. "
    "인물이 있다면: 추정 성별, 나이대, 옷 색상과 스타일, 머리 모양, 안경 착용 여부, 현재 동작과 자세. "
    "배경: 실내/실외 여부, 보이는 가구나 사물 (책상, 의자, 벽 색상 등), 조명 상태. "
    "반드시 JSON으로만 답하세요 (마크다운 금지): "
    '{"is_anomaly": false, "event_type": "scene", "description": "한국어 2-3문장 상황 묘사", "confidence": 0.9}' "
    "description은 반드시 구체적인 장면 묘사여야 합니다. 절대 '정상'이라는 단어만 쓰지 마세요."
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
        """Realistic scene description mock (no AI model required)."""
        import random
        # 실제 CCTV 기록처럼 구체적인 장면 묘사
        scenes = [
            "30대 추정 남성 1명이 화면 중앙에 위치. 흰색 반팔 티셔츠 착용, 짧은 검은 머리, 검은 뿔테 안경 착용. 나무 재질 책상 앞에 앉아 정면을 응시하고 있음.",
            "안경 착용 남성이 고개를 약간 왼쪽으로 돌린 상태. 흰 상의 착용. 배경은 밝은 회색 벽. 검은 사무용 의자 등받이 보임.",
            "남성 1명이 책상에 양손을 올려놓고 앉아 있음. 흰 반팔 티셔츠, 안경 착용. 실내 환경, 조명 밝음. 뒤쪽으로 빈 공간 보임.",
            "인물이 몸을 약간 앞으로 숙이고 있음. 흰 상의, 안경. 책상 위 특정 물체를 바라보는 듯한 자세. 실내, 조명 양호.",
            "카메라 앵글 기준 인물이 약간 오른쪽에 위치. 흰 티셔츠 착용 남성. 편안한 자세로 앉아 있음. 배경에 회색 벽과 문 틀 일부 보임.",
            "남성이 손을 들어올려 얼굴 근처에 위치시킴. 흰 반팔 티셔츠, 안경. 무언가를 만지거나 조작하는 동작으로 보임.",
            "실내 공간. 책상과 사무 의자 배치. 남성 1명 착석 중. 조명 밝고 배경 깔끔함. 특이 사항 없음.",
        ]
        return ClassificationResult(False, "scene", random.choice(scenes), round(random.uniform(0.88, 0.97), 2))

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
