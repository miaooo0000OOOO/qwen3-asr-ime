#!/usr/bin/env python3
"""Qwen3-ASR streaming server (WebSocket-only, requires vLLM backend)."""

from __future__ import annotations

import base64
import io
import logging
import os
import time
import wave
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("asr-server")

app = FastAPI(title="Qwen3-ASR Streaming Server")

_model = None
_model_backend: str | None = None

MODEL_PATH = os.environ.get("QWEN3_ASR_MODEL", "/Data2/Models/Qwen3-ASR-0.6B")


def _load_model() -> None:
    global _model, _model_backend
    if _model is not None:
        return

    backend = os.environ.get("QWEN3_ASR_BACKEND", "vllm").lower()
    if backend != "vllm":
        raise RuntimeError("Streaming server requires vLLM backend")

    from qwen_asr import Qwen3ASRModel

    gpu_mem = float(os.environ.get("QWEN3_ASR_GPU_MEM", "0.9"))
    max_tokens = int(os.environ.get("QWEN3_ASR_MAX_TOKENS", "256"))
    max_model_len = int(os.environ.get("QWEN3_ASR_MAX_MODEL_LEN", "4096"))
    enforce_eager = os.environ.get("QWEN3_ASR_ENFORCE_EAGER", "1").lower() in ("1", "true", "yes")
    enable_prefix_caching = os.environ.get("QWEN3_ASR_PREFIX_CACHING", "0").lower() in ("1", "true", "yes")
    logger.info(
        "Loading Qwen3-ASR (%s) with vLLM backend (gpu_mem=%.2f, max_model_len=%d) ...",
        MODEL_PATH,
        gpu_mem,
        max_model_len,
    )
    t0 = time.time()
    _model = Qwen3ASRModel.LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=gpu_mem,
        max_new_tokens=max_tokens,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
        enable_prefix_caching=enable_prefix_caching,
    )
    _model_backend = "vllm"
    logger.info("Loaded vLLM backend in %.1fs", time.time() - t0)


def _decode_audio(raw: bytes) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Decode WAV bytes to float32 mono array."""
    with wave.open(io.BytesIO(raw), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        dtype_map: dict[int, type] = {1: np.int8, 2: np.int16}
        dtype = dtype_map.get(wf.getsampwidth(), np.int16)
        data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
        if wf.getnchannels() > 1:
            data = data.reshape(-1, wf.getnchannels()).mean(axis=1)
        data /= np.iinfo(dtype).max
    return data.astype(np.float32)


@app.websocket("/v1/asr/stream")
async def asr_stream(websocket: WebSocket) -> None:
    global _model
    _load_model()
    assert _model is not None

    await websocket.accept()
    state = _model.init_streaming_state()
    last_text = ""
    await websocket.send_json({"type": "ready"})

    try:
        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type")

            if msg_type == "chunk":
                audio_b64 = msg.get("audio", "")
                fmt = msg.get("format", "wav")
                if not audio_b64:
                    continue
                try:
                    raw = base64.b64decode(audio_b64)
                    if fmt == "pcm":
                        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    else:
                        pcm = _decode_audio(raw)
                    _model.streaming_transcribe(pcm, state)
                    if state.text != last_text:
                        last_text = state.text
                        await websocket.send_json(
                            {
                                "type": "partial",
                                "text": state.text,
                                "language": state.language,
                            }
                        )
                except Exception as exc:
                    logger.exception("Streaming chunk failed")
                    await websocket.send_json(
                        {"type": "error", "message": f"Chunk processing failed: {exc}"}
                    )

            elif msg_type == "finish":
                _model.finish_streaming_transcribe(state)
                await websocket.send_json(
                    {
                        "type": "final",
                        "text": state.text,
                        "language": state.language,
                    }
                )
                await websocket.close(code=1000)
                return

            elif msg_type == "config":
                # Re-initialize state if language/config provided before any chunk.
                language = msg.get("language")
                state = _model.init_streaming_state(language=language)
                last_text = ""
                await websocket.send_json({"type": "ready"})

            else:
                await websocket.send_json(
                    {"type": "error", "message": f"Unknown message type: {msg_type}"}
                )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.exception("WebSocket streaming error")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
            await websocket.close(code=1011)
        except Exception:
            pass


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "backend": _model_backend}


if __name__ == "__main__":
    _load_model()
    uvicorn.run(app, host="127.0.0.1", port=8000)
