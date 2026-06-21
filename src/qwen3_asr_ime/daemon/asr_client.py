from __future__ import annotations

import base64
import logging
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ASRResult:
    text: str
    language: str | None = None
    error: str | None = None


class ASRClient:
    """Client for Qwen3-ASR vLLM server (OpenAI-compatible chat completions API)."""

    def __init__(self, endpoint: str, api_key: str = "dummy", timeout: float = 30.0):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def recognize(self, audio_bytes: bytes, sample_rate: int = 16000) -> ASRResult:
        url = f"{self.endpoint}/v1/chat/completions"
        # Qwen3-ASR vLLM expects audio as base64 data URL
        b64 = base64.b64encode(audio_bytes).decode("utf-8")
        audio_url = f"data:audio/wav;base64,{b64}"

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": audio_url},
                        }
                    ],
                }
            ],
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        try:
            resp = requests.post(
                url, headers=headers, json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            language, text = _parse_asr_output(content)
            return ASRResult(text=text, language=language)
        except requests.RequestException as exc:
            logger.error("ASR request failed: %s", exc)
            return ASRResult(text="", error=str(exc))
        except (KeyError, IndexError) as exc:
            return ASRResult(text="", error=f"Bad response: {exc}")


def _parse_asr_output(content: str) -> tuple[str | None, str]:
    """Parse Qwen3-ASR vLLM output: <|language|>...<|/language|><|text|>...<|/text|>"""
    language = None
    text = ""
    for tag in content.split("<|language|>")[1:]:
        lang, rest = tag.split("<|/language|>", 1)
        language = lang.strip()
        if "<|text|>" in rest:
            text = rest.split("<|text|>")[1].split("<|/text|>")[0]
    return language, text
