"""End-to-end integration test.

Mocks only what's inaccessible in a test environment:
1. ``pynput.keyboard.Listener`` — controllable press/release callbacks
2. ``sd.InputStream`` — captures callback & injects test audio data
3. ``ASRStreamClient`` — returns canned streaming transcription

All mock setup happens inside the test function to avoid cross-test contamination.
Hotkey event handling and state machine run real code.
"""

from __future__ import annotations

import asyncio
import io
import wave
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from qwen3_asr_ime.common.config import IMEConfig
from qwen3_asr_ime.common.protocol import RecognizedText, StateUpdate, parse_message
from qwen3_asr_ime.daemon.asr_client import ASRResult

# ---- Helpers (pure, no side effects at module level) ----


def _make_wav(duration: float = 0.3) -> bytes:
    fs = 16000
    n = int(fs * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    env = np.clip(1 - np.abs(np.linspace(-1, 1, n)) ** 2, 0, 1)
    audio = (
        0.3 * np.sin(2 * np.pi * 150 * t)
        + 0.2 * np.sin(2 * np.pi * 300 * t)
        + 0.1 * np.sin(2 * np.pi * 450 * t)
    ) * env
    pcm = (audio * 8000).astype(np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(pcm)
    return buf.getvalue()


async def _stream_results() -> AsyncIterator[ASRResult]:
    yield ASRResult(text="你好世界", language="Chinese", final=True)


class _FakeKey:
    def __init__(self, name: str):
        self.name = name


@pytest.mark.xfail(
    strict=False,
    reason="Known issue: runs in same process as other tests; run in isolation: -m e2e",
)
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_full_daemon_flow(tmp_path: Path) -> None:
    # Shared state for the virtual pynput listener
    press_callback: list = [None]
    release_callback: list = [None]

    def listener_constructor(*, on_press=None, on_release=None, **kwargs):
        press_callback[0] = on_press
        release_callback[0] = on_release
        ms = MagicMock()
        ms.start = MagicMock()
        ms.stop = MagicMock()
        return ms

    # ---- Set up sd.InputStream mock ----
    stream_callback: list = [None]

    def stream_constructor(*, callback=None, **_kw):
        stream_callback[0] = callback
        ms = MagicMock()
        ms.start = MagicMock()
        ms.stop = MagicMock()
        ms.close = MagicMock()
        return ms

    with (
        patch("sounddevice.InputStream", side_effect=stream_constructor),
        patch("pynput.keyboard.Listener", side_effect=listener_constructor),
    ):
        # Import the daemon module after mocks are in place
        from qwen3_asr_ime.daemon.service import VoiceInputDaemon

        socket_path = tmp_path / "e2e.sock"
        config = IMEConfig(
            hotkey_device="pynput",
            hotkey_key="CTRL",
            audio_sample_rate=16000,
            audio_channels=1,
            audio_format="int16",
            audio_chunk_ms=20,
            asr_endpoint="http://127.0.0.1:8000",
            asr_mode="streaming",
            asr_model="1.7B",
            asr_backend="vllm",
            asr_device="cpu",
            asr_quantization="none",
            asr_api_key="dummy",
            asr_timeout=5.0,
            asr_auto_sleep_time=300,
            asr_backend_wait_timeout=120,
            ipc_socket_path=str(socket_path),
            log_level="DEBUG",
        )

        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.send_chunk = AsyncMock()
        mock_client.send_json = AsyncMock()
        mock_client.close = AsyncMock()
        mock_client.iterate = MagicMock(return_value=_stream_results())

        test_wav = _make_wav(duration=0.3)

        with patch("qwen3_asr_ime.daemon.service.ASRStreamClient", return_value=mock_client):
            daemon = VoiceInputDaemon(config)
            daemon_task = asyncio.create_task(daemon.start())

            for _ in range(60):
                if socket_path.exists():
                    break
                await asyncio.sleep(0.05)
            assert socket_path.exists(), "IPC socket not created"

            await asyncio.sleep(0.2)

            # ---- IPC client ----
            received: list[str] = []
            client_ready = asyncio.Event()

            async def ipc_client() -> None:
                reader, writer = await asyncio.open_unix_connection(str(socket_path))
                client_ready.set()
                while True:
                    line = await reader.readline()
                    if not line:
                        break
                    received.append(line.decode("utf-8").strip())

            client_task = asyncio.create_task(ipc_client())
            await asyncio.wait_for(client_ready.wait(), timeout=3.0)
            await asyncio.sleep(0.2)

            # ---- Ctrl press ----
            assert press_callback[0] is not None
            press_callback[0](_FakeKey("ctrl_l"))
            await asyncio.sleep(0.3)

            assert daemon._state == "recording", f"Expected recording, got {daemon._state}"

            # Feed audio
            if stream_callback[0]:
                chunk = np.frombuffer(
                    test_wav[test_wav.find(b"data") + 8 : test_wav.find(b"data") + 8 + 1600],
                    dtype=np.int16,
                ).reshape(-1, 1)
                stream_callback[0](chunk, len(chunk), None, None)

            await asyncio.sleep(0.1)

            # ---- Ctrl release ----
            assert release_callback[0] is not None
            release_callback[0](_FakeKey("ctrl_l"))
            await asyncio.sleep(1.0)

            # ---- Cleanup ----
            daemon._shutdown()
            daemon_task.cancel()
            client_task.cancel()
            for t in (daemon_task, client_task):
                try:
                    await t
                except (asyncio.CancelledError, GeneratorExit):
                    pass

    # ---- Assert ----
    assert daemon._state == "idle", f"Expected idle, got {daemon._state}"
    assert len(received) >= 2, f"Expected >=2 IPC messages, got {len(received)}: {received}"

    found_text = None
    found_states = set()
    for msg in received:
        p = parse_message(msg)
        if isinstance(p, StateUpdate):
            found_states.add(p.state)
        if isinstance(p, RecognizedText) and p.text:
            found_text = p.text

    assert found_text == "你好世界", f"Expected '你好世界', got '{found_text}'. All: {received}"
    assert "recording" in found_states, f"State 'recording' missing. Seen: {found_states}"
