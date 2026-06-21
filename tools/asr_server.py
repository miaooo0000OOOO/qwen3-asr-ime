#!/usr/bin/env python3
"""Lightweight Qwen3-ASR server using transformers backend."""

from __future__ import annotations

import base64
import io
import logging
import time
import wave

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("asr-server")

app = FastAPI(title="Qwen3-ASR Server")

_model = None


def _load_model():
    global _model
    if _model is not None:
        return
    import torch
    from qwen_asr import Qwen3ASRModel

    logger.info("Loading Qwen3-ASR-1.7B ...")
    t0 = time.time()
    _model = Qwen3ASRModel.from_pretrained(
        "/Data2/Models/Qwen3-ASR-1.7B",
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_new_tokens=256,
    )
    logger.info(
        "Loaded in %.1fs, GPU: %.1fGB", time.time() - t0, torch.cuda.memory_allocated() / 1e9
    )


def _decode_audio(raw: bytes) -> np.ndarray:
    """Decode WAV bytes to float32 mono array."""
    with wave.open(io.BytesIO(raw), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        dtype = {1: np.int8, 2: np.int16}.get(wf.getsampwidth(), np.int16)
        data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
        if wf.getnchannels() > 1:
            data = data.reshape(-1, wf.getnchannels()).mean(axis=1)
        data /= np.iinfo(dtype).max
    return data.astype(np.float32)


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    _load_model()
    body = await req.json()

    try:
        content = body["messages"][0]["content"]
        audio_block = next(c for c in content if c["type"] == "audio_url")
        audio_url = audio_block["audio_url"]["url"]
    except (KeyError, IndexError, StopIteration):
        raise HTTPException(400, "Missing audio_url in messages")

    # Decode base64 data URL
    if "," in audio_url:
        audio_url = audio_url.split(",", 1)[1]
    raw = base64.b64decode(audio_url)
    audio = _decode_audio(raw)

    language = body.get("language")

    t0 = time.time()
    results = _model.transcribe(
        audio=(audio, 16000),
        language=language,
        return_time_stamps=False,
    )
    elapsed = time.time() - t0

    r = results[0]
    text = r.text
    lang = r.language if hasattr(r, "language") else (language or "auto")
    logger.info("Recognized [%.1fs] %s: %s", elapsed, lang, text)

    return {
        "choices": [
            {"message": {"content": f"<|language|>{lang}<|/language|><|text|>{text}<|/text|>"}}
        ]
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    _load_model()
    uvicorn.run(app, host="127.0.0.1", port=8000)
