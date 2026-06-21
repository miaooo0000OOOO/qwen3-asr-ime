"""End-to-end integration test.

Mocks only what's inaccessible in a test environment:
1. ``evdev.list_devices()`` / ``InputDevice`` / ``read()`` — controllable events
2. ``sd.InputStream`` — captures callback & injects test audio data
3. ``ASRClient.recognize`` — returns canned transcription

All mock setup happens inside the test function to avoid cross-test contamination
of ``sys.modules`` (other test files may overwrite the evdev mock).
Hotkey event handling and state machine run real code.
"""
import asyncio
import io
import sys
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

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


def _make_event(code: int, value: int) -> MagicMock:
    e = MagicMock()
    e.type = 1
    e.code = code
    e.value = value
    return e


@pytest.mark.xfail(
    strict=False,
    reason="Known issue: runs in same process as test_integration.py which pollutes sys.modules['evdev']; run in isolation: -m e2e",
)
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_full_daemon_flow(tmp_path):
    # ---- Set up evdev mock ----
    evdev_mock = MagicMock()
    ecodes_mock = MagicMock()
    ecodes_mock.EV_KEY = 1
    ecodes_mock.ecodes = {
        "KEY_LEFTCTRL": 29,
        "KEY_RIGHTCTRL": 97,
        "KEY_LEFTMETA": 125,
        "KEY_LEFTSHIFT": 42,
        "KEY_LEFTALT": 56,
    }
    sys.modules["evdev"] = evdev_mock
    sys.modules["evdev.ecodes"] = ecodes_mock

    # Patch hotkey's evdev reference in case hotkey was already imported
    # (it will be cached across test files in the full suite).
    import qwen3_asr_ime.daemon.hotkey as _hotkey_mod
    _hotkey_mod.evdev = evdev_mock

    # Shared state for the virtual device
    event_queue: list = []
    device_closed = False

    def device_read():
        if device_closed:
            raise BlockingIOError
        items = list(event_queue)
        event_queue.clear()
        if not items:
            raise BlockingIOError
        return items

    virtual_device = MagicMock()
    virtual_device.fn = "/dev/input/event0"
    virtual_device.read = device_read
    virtual_device.grab = MagicMock()
    virtual_device.ungrab = MagicMock()

    evdev_mock.list_devices = MagicMock(return_value=["/dev/input/event0"])
    evdev_mock.InputDevice = MagicMock(return_value=virtual_device)

    # ---- Set up sd.InputStream mock ----
    stream_callback: list = [None]

    def stream_constructor(*, callback=None, **_kw):
        stream_callback[0] = callback
        ms = MagicMock()
        ms.start = MagicMock()
        ms.stop = MagicMock()
        ms.close = MagicMock()
        return ms

    # Patch sd before daemon imports reference it
    with patch("sounddevice.InputStream", side_effect=stream_constructor):
        # Import the daemon module after mocks are in place
        from qwen3_asr_ime.daemon.service import VoiceInputDaemon

        socket_path = tmp_path / "e2e.sock"
        config = IMEConfig(
            hotkey_device="evdev",
            hotkey_key="CTRL",
            audio_sample_rate=16000,
            audio_channels=1,
            audio_format="int16",
            audio_chunk_ms=20,
            asr_endpoint="http://127.0.0.1:8000",
            asr_model="Qwen/Qwen3-ASR",
            asr_device="cpu",
            asr_quantization="none",
            asr_api_key="dummy",
            ipc_socket_path=str(socket_path),
            log_level="DEBUG",
        )

        test_wav = _make_wav(duration=0.3)

        daemon = VoiceInputDaemon(config)
        daemon_task = asyncio.create_task(daemon.start())
        daemon.asr.recognize = MagicMock(return_value=ASRResult(text="你好世界"))

        for _ in range(60):
            if socket_path.exists():
                break
            await asyncio.sleep(0.05)
        assert socket_path.exists(), "IPC socket not created"

        await asyncio.sleep(1.0)  # let listener thread enter read loop

        # ---- IPC client ----
        received: list[str] = []
        client_ready = asyncio.Event()

        async def ipc_client():
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
        event_queue.append(_make_event(29, 1))
        await asyncio.sleep(0.5)

        assert daemon._state == "recording", f"Expected recording, got {daemon._state}"

        # Feed audio
        if stream_callback[0]:
            chunk = np.frombuffer(
                test_wav[
                    test_wav.find(b"data")
                    + 8 : test_wav.find(b"data")
                    + 8
                    + 1600
                ],
                dtype=np.int16,
            ).reshape(-1, 1)
            stream_callback[0](chunk, len(chunk), None, None)

        await asyncio.sleep(0.1)

        # ---- Ctrl release ----
        event_queue.append(_make_event(29, 0))
        await asyncio.sleep(1.5)

        # ---- Cleanup ----
        device_closed = True
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
        assert len(received) >= 2, (
            f"Expected >=2 IPC messages, got {len(received)}: {received}"
        )

        found_text = None
        found_states = set()
        for msg in received:
            p = parse_message(msg)
            if isinstance(p, StateUpdate):
                found_states.add(p.state)
            if isinstance(p, RecognizedText) and p.text:
                found_text = p.text

        assert found_text == "你好世界", (
            f"Expected '你好世界', got '{found_text}'. All: {received}"
        )
        assert "recording" in found_states, (
            f"State 'recording' missing. Seen: {found_states}"
        )
