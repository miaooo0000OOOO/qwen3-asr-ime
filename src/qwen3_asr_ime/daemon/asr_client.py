from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ASRResult:
    text: str
    error: str | None = None


class ASRClient:
    def __init__(self, endpoint: str, api_key: str = "dummy", timeout: float = 30.0):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def recognize(self, audio_bytes: bytes, sample_rate: int = 16000) -> ASRResult:
        url = f"{self.endpoint}/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        files = {
            "file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav"),
        }
        data = {
            "model": "qwen3-asr",
            "language": "zh",
            "response_format": "json",
        }
        try:
            resp = requests.post(url, headers=headers, files=files, data=data, timeout=self.timeout)
            resp.raise_for_status()
            payload = resp.json()
            text = payload.get("text", "")
            return ASRResult(text=text)
        except requests.RequestException as exc:
            logger.error("ASR request failed: %s", exc)
            return ASRResult(text="", error=str(exc))
