import asyncio
import io
import sys
import wave
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock sounddevice at module level before any qwen3_asr_ime imports
sys.modules["sounddevice"] = MagicMock()
sys.modules["sounddevice"].InputStream = MagicMock()

from qwen3_asr_ime.common.config import IMEConfig  # noqa: E402
from qwen3_asr_ime.common.protocol import RecognizedText, parse_message  # noqa: E402
from qwen3_asr_ime.daemon.asr_client import ASRResult  # noqa: E402
from qwen3_asr_ime.daemon.hotkey import HotkeyEvent  # noqa: E402


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


async def _stream_results() -> AsyncIterator[ASRResult]:
    yield ASRResult(text="测", language="Chinese", final=False)
    yield ASRResult(text="测试文本", language="Chinese", final=True)


@pytest.mark.asyncio
async def test_daemon_streaming_recognize_flow(tmp_path: Path) -> None:
    """Integration test: daemon streams audio and broadcasts final text via IPC."""
    socket_path = tmp_path / "test.sock"
    config = IMEConfig(
        hotkey_device="pynput",
        hotkey_key="<Super>+<Shift>+R",
        audio_sample_rate=16000,
        audio_channels=1,
        audio_format="int16",
        audio_chunk_ms=20,
        asr_endpoint="http://localhost:8000",
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

    from qwen3_asr_ime.daemon.service import VoiceInputDaemon

    daemon = VoiceInputDaemon(config)

    # Mock BackendManager to avoid spawning real backend process
    daemon._backend_mgr = AsyncMock()
    daemon._backend_mgr.is_running = True
    daemon._backend_mgr.spawn = AsyncMock()
    daemon._backend_mgr.wait_ready = AsyncMock()
    daemon._backend_mgr.touch_activity = MagicMock()
    daemon._backend_mgr.check_idle = AsyncMock(return_value=False)
    daemon._backend_mgr.stop = AsyncMock()

    silent_wav = _make_silent_wav(duration_sec=0.2)

    mock_client = AsyncMock()
    mock_client.connect = AsyncMock()
    mock_client.send_chunk = AsyncMock()
    mock_client.send_json = AsyncMock()
    mock_client.close = AsyncMock()
    mock_client.iterate = MagicMock(return_value=_stream_results())

    with (
        patch("qwen3_asr_ime.daemon.service.ASRStreamClient", return_value=mock_client),
        patch("qwen3_asr_ime.daemon.service.create_hotkey_listener") as mock_hotkey,
    ):
        mock_hotkey.return_value.start = MagicMock()
        mock_hotkey.return_value.stop = MagicMock()

        received: list[str] = []
        connected = asyncio.Event()

        async def client() -> None:
            reader, writer = await asyncio.open_unix_connection(str(socket_path))
            connected.set()
            while True:
                line = await reader.readline()
                if not line:
                    break
                received.append(line.decode("utf-8").strip())

        daemon_task = asyncio.create_task(daemon.start())

        for _ in range(50):
            if socket_path.exists():
                break
            await asyncio.sleep(0.05)
        assert socket_path.exists(), "IPC socket not created"

        client_task = asyncio.create_task(client())
        await asyncio.wait_for(connected.wait(), timeout=3.0)

        # Trigger streaming recognition
        with patch.object(daemon.recorder, "stop", return_value=silent_wav):
            daemon._on_hotkey(HotkeyEvent(action="press"))
            await asyncio.sleep(0.1)
            assert daemon._state == "recording"

            daemon._on_hotkey(HotkeyEvent(action="release"))
            await asyncio.sleep(0.3)

        async def wait_for_text() -> None:
            for _ in range(100):
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

    parsed_messages = [parse_message(m) for m in received]
    assert any(isinstance(p, RecognizedText) and p.text == "测试文本" for p in parsed_messages), (
        f"Received messages: {received}"
    )
