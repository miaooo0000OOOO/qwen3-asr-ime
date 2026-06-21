import asyncio
import io
import sys
import wave
from unittest.mock import MagicMock, patch

import pytest

# Mock evdev and sounddevice at module level before any qwen3_asr_ime imports
sys.modules["evdev"] = MagicMock()
sys.modules["evdev.ecodes"] = MagicMock()
sys.modules["evdev.ecodes"].EV_KEY = 1
sys.modules["evdev.ecodes"].ecodes = {}
sys.modules["sounddevice"] = MagicMock()
sys.modules["sounddevice"].InputStream = MagicMock()

import responses

from qwen3_asr_ime.common.config import IMEConfig
from qwen3_asr_ime.common.protocol import RecognizedText, parse_message


def _make_silent_wav(duration_sec: float = 0.2) -> bytes:
    sample_rate = 16000
    samples = int(sample_rate * duration_sec)
    pcm = b"\x00\x00" * samples
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


@pytest.mark.asyncio
@responses.activate
async def test_daemon_recognize_flow(tmp_path):
    """Integration test: daemon broadcasts recognized text via IPC."""
    socket_path = tmp_path / "test.sock"
    config = IMEConfig(
        hotkey_device="evdev",
        hotkey_key="<Super>+<Shift>+R",
        audio_sample_rate=16000,
        audio_channels=1,
        audio_format="int16",
        audio_chunk_ms=20,
        asr_endpoint="http://localhost:8000",
        asr_model="Qwen/Qwen3-ASR",
        asr_device="cpu",
        asr_quantization="none",
        asr_api_key="dummy",
        ipc_socket_path=str(socket_path),
        log_level="DEBUG",
    )

    responses.post(
        "http://localhost:8000/v1/audio/transcriptions",
        json={"text": "测试文本"},
        status=200,
    )

    with patch("qwen3_asr_ime.daemon.service.create_hotkey_listener") as mock_hotkey:
        mock_hotkey.return_value.start = MagicMock()
        mock_hotkey.return_value.stop = MagicMock()

        from qwen3_asr_ime.daemon.service import VoiceInputDaemon
        daemon = VoiceInputDaemon(config)

        daemon.recorder._frames = [_make_silent_wav(duration_sec=0.2)]

        received = []
        connected = asyncio.Event()

        async def client():
            reader, writer = await asyncio.open_unix_connection(str(socket_path))
            connected.set()
            while True:
                line = await reader.readline()
                if not line:
                    break
                msg = line.decode("utf-8").strip()
                received.append(msg)

        # Start daemon
        daemon_task = asyncio.create_task(daemon.start())

        # Wait for IPC socket to exist
        for _ in range(50):
            if socket_path.exists():
                break
            await asyncio.sleep(0.05)
        assert socket_path.exists(), "IPC socket not created"

        # Start client, wait for connection
        client_task = asyncio.create_task(client())
        await asyncio.wait_for(connected.wait(), timeout=3.0)

        # Now trigger recognition
        await daemon._recognize(daemon.recorder.stop())

        # Wait for recognized text to arrive
        async def wait_for_text():
            for _ in range(50):
                for msg in received:
                    parsed = parse_message(msg)
                    if isinstance(parsed, RecognizedText) and parsed.text == "测试文本":
                        return
                await asyncio.sleep(0.05)
            raise AssertionError(f"Test text not received. Got: {received}")

        await asyncio.wait_for(wait_for_text(), timeout=5.0)

        daemon._shutdown()
        for t in [daemon_task, client_task]:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, GeneratorExit):
                pass

    # Verify we got the recognized text via parse_message (handles \\uXXXX encoding)
    parsed_messages = [parse_message(m) for m in received]
    assert any(
        isinstance(p, RecognizedText) and p.text == "测试文本"
        for p in parsed_messages
    ), f"Received messages: {received}"
