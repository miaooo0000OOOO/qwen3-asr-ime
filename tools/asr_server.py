#!/usr/bin/env python3
"""Qwen3-ASR server — supports streaming (vLLM) and non-streaming (transformers/vLLM)."""

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
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("asr-server")

app = FastAPI(title="Qwen3-ASR Server")

_model = None
_model_backend: str | None = None
_server_mode: str = "offline"

MODEL_PATH = os.environ.get("QWEN3_ASR_MODEL", "/Data2/Models/Qwen3-ASR-1.7B")
SERVER_MODE = os.environ.get("QWEN3_ASR_MODE", "offline")
SERVER_BACKEND = os.environ.get("QWEN3_ASR_BACKEND", "transformers")
SERVER_DEVICE = os.environ.get("QWEN3_ASR_DEVICE", "auto")
SERVER_PORT = int(os.environ.get("QWEN3_ASR_PORT", "8000"))


def _load_model() -> None:
    global _model, _model_backend, _server_mode
    if _model is not None:
        return

    _server_mode = SERVER_MODE
    backend = SERVER_BACKEND

    if _server_mode == "streaming" and backend != "vllm":
        raise RuntimeError(
            f"streaming mode requires vllm backend, got {backend}"
        )
    if _server_mode == "offline" and backend not in ("transformers", "vllm"):
        raise RuntimeError(
            f"offline mode requires transformers or vllm backend, got {backend}"
        )

    if backend == "vllm":
        _load_vllm()
    elif backend == "transformers":
        _load_transformers()
    else:
        raise RuntimeError(f"Unsupported backend: {backend}")


def _load_vllm() -> None:
    global _model, _model_backend
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


def _load_transformers() -> None:
    global _model, _model_backend
    from qwen_asr import Qwen3ASRModel

    device = SERVER_DEVICE

    logger.info(
        "Loading Qwen3-ASR (%s) with transformers backend (device=%s) ...",
        MODEL_PATH,
        device,
    )
    t0 = time.time()
    if device == "auto":
        _model = Qwen3ASRModel.from_pretrained(MODEL_PATH)
    else:
        _model = Qwen3ASRModel.from_pretrained(MODEL_PATH, device=device)
    _model_backend = "transformers"
    logger.info("Loaded transformers backend in %.1fs", time.time() - t0)


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


@app.post("/v1/asr/transcribe")
async def asr_transcribe(request: Request) -> dict[str, Any]:
    """Non-streaming (offline) ASR endpoint.

    Accepts raw WAV bytes in the request body. Returns a JSON object with
    ``text`` and ``language`` fields.

    Query params:
        language: Optional language hint (e.g. ``"zh"``, ``"en"``).
    """
    global _model
    _load_model()
    assert _model is not None
    assert _model_backend is not None

    language = request.query_params.get("language")

    try:
        raw_body = await request.body()
        if not raw_body:
            return {"text": "", "language": language, "error": "Empty request body"}

        # Decode WAV to float32 numpy array
        pcm = _decode_audio(raw_body)

        # transcribe accepts (np.ndarray, sample_rate) tuple
        results = _model.transcribe(
            (pcm, 16000), language=language
        )
        if results:
            first = results[0]
            return {"text": first.text, "language": first.language}
        else:
            return {"text": "", "language": language}
    except Exception as exc:
        logger.exception("Non-streaming recognition failed")
        return {"text": "", "language": language, "error": str(exc)}


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
    return {"status": "ok", "backend": _model_backend, "mode": _server_mode}


if __name__ == "__main__":
    _load_model()
    uvicorn.run(app, host="127.0.0.1", port=SERVER_PORT)
