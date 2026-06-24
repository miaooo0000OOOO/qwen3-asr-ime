"""Tests for streaming Qwen3-ASR client/server."""

from __future__ import annotations

import asyncio
import io
import sys
import wave
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.modules["sounddevice"] = MagicMock()
sys.modules["sounddevice"].InputStream = MagicMock()

from qwen3_asr_ime.common.config import IMEConfig  # noqa: E402
from qwen3_asr_ime.common.protocol import RecognizedText, parse_message  # noqa: E402
from qwen3_asr_ime.daemon.asr_client import ASRResult, ASRStreamClient  # noqa: E402
from qwen3_asr_ime.daemon.hotkey import HotkeyEvent  # noqa: E402


def _make_silent_wav(duration_sec: float = 0.1) -> bytes:
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
async def test_stream_client_partial_and_final() -> None:
    """ASRStreamClient yields partials and a final result."""
    mock_ws = AsyncMock()
    mock_ws.recv.side_effect = [
        '{"type": "ready"}',
        '{"type": "partial", "text": "你好", "language": "Chinese"}',
        '{"type": "partial", "text": "你好世界", "language": "Chinese"}',
        '{"type": "final", "text": "你好世界", "language": "Chinese"}',
    ]

    connect_mock = AsyncMock(return_value=mock_ws)
    with patch("qwen3_asr_ime.daemon.asr_client.websockets.connect", connect_mock):
        client = ASRStreamClient("http://localhost:8000")
        await client.connect()
        await client.send_chunk(b"fake pcm", fmt="pcm")

        results = [r async for r in client.iterate()]

    assert len(results) == 3
    assert results[0] == ASRResult(text="你好", language="Chinese", final=False)
    assert results[1] == ASRResult(text="你好世界", language="Chinese", final=False)
    assert results[2] == ASRResult(text="你好世界", language="Chinese", final=True)


@pytest.mark.asyncio
async def test_stream_client_error() -> None:
    """ASRStreamClient surfaces server-side errors."""
    mock_ws = AsyncMock()
    mock_ws.recv.side_effect = [
        '{"type": "ready"}',
        '{"type": "error", "message": "backend not vllm"}',
    ]

    connect_mock = AsyncMock(return_value=mock_ws)
    with patch("qwen3_asr_ime.daemon.asr_client.websockets.connect", connect_mock):
        client = ASRStreamClient("http://localhost:8000")
        await client.connect()
        results = [r async for r in client.iterate()]

    assert len(results) == 1
    assert results[0].error == "backend not vllm"


@pytest.mark.asyncio
async def test_daemon_streaming_flow(tmp_path: Path) -> None:
    """Daemon with streaming=True broadcasts partial and final results over IPC."""
    socket_path = tmp_path / "stream.sock"
    config = IMEConfig(
        hotkey_device="pynput",
        hotkey_key="CTRL",
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

    # Mock the streaming client
    mock_client = AsyncMock()
    mock_client.connect = AsyncMock()
    mock_client.send_chunk = AsyncMock()
    mock_client.send_json = AsyncMock()
    mock_client.close = AsyncMock()
    mock_client.iterate = MagicMock(return_value=async_iter_stream_results())

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

        with (
            patch.object(daemon.recorder, "start") as mock_start,
            patch.object(daemon.recorder, "stop", return_value=_make_silent_wav()) as mock_stop,
        ):
            daemon_task = asyncio.create_task(daemon.start())
            for _ in range(50):
                if socket_path.exists():
                    break
                await asyncio.sleep(0.05)
            assert socket_path.exists()

            client_task = asyncio.create_task(client())
            await asyncio.wait_for(connected.wait(), timeout=3.0)

            # Simulate press → release → finish
            daemon._on_hotkey(HotkeyEvent(action="press"))
            await asyncio.sleep(0.1)
            assert daemon._state == "recording"

            daemon._on_hotkey(HotkeyEvent(action="release"))
            await asyncio.sleep(0.3)

            # Wait for final result to be broadcast
            async def wait_for_final() -> None:
                for _ in range(100):
                    for msg in received:
                        parsed = parse_message(msg)
                        if isinstance(parsed, RecognizedText) and parsed.text == "你好世界":
                            return
                    await asyncio.sleep(0.05)
                raise AssertionError(f"Final text not received. Got: {received}")

            await asyncio.wait_for(wait_for_final(), timeout=5.0)

            daemon._shutdown()
            for t in [daemon_task, client_task]:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, GeneratorExit):
                    pass

            assert mock_start.called
            assert mock_stop.called

    parsed = [parse_message(m) for m in received]
    texts = [p.text for p in parsed if isinstance(p, RecognizedText)]
    assert "你好" in texts
    assert "你好世界" in texts


async def async_iter_stream_results() -> AsyncIterator[ASRResult]:
    yield ASRResult(text="你好", language="Chinese", final=False)
    yield ASRResult(text="你好世界", language="Chinese", final=True)


async def async_iter_stream_results_two_partials() -> AsyncIterator[ASRResult]:
    yield ASRResult(text="你好", language="Chinese", final=False)
    yield ASRResult(text="你好世", language="Chinese", final=False)
    yield ASRResult(text="你好世界", language="Chinese", final=True)


@pytest.mark.asyncio
async def test_daemon_streaming_types_partial_results(tmp_path: Path) -> None:
    """Without IPC clients, partial results are typed incrementally."""
    socket_path = tmp_path / "stream_type.sock"
    config = IMEConfig(
        hotkey_device="pynput",
        hotkey_key="CTRL",
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

    mock_client = AsyncMock()
    mock_client.connect = AsyncMock()
    mock_client.send_chunk = AsyncMock()
    mock_client.send_json = AsyncMock()
    mock_client.close = AsyncMock()
    mock_client.iterate = MagicMock(return_value=async_iter_stream_results_two_partials())

    with (
        patch("qwen3_asr_ime.daemon.service.ASRStreamClient", return_value=mock_client),
        patch("qwen3_asr_ime.daemon.service.create_hotkey_listener") as mock_hotkey,
    ):
        mock_hotkey.return_value.start = MagicMock()
        mock_hotkey.return_value.stop = MagicMock()

        typed_calls: list[tuple[int, str]] = []

        def capture_type(to_delete: int, text: str) -> None:
            typed_calls.append((to_delete, text))

        with (
            patch.object(daemon.recorder, "start") as mock_start,
            patch.object(daemon.recorder, "stop", return_value=_make_silent_wav()) as mock_stop,
            patch("qwen3_asr_ime.daemon.service._type_incremental_x11", side_effect=capture_type),
        ):
            daemon_task = asyncio.create_task(daemon.start())
            for _ in range(50):
                if socket_path.exists():
                    break
                await asyncio.sleep(0.05)
            assert socket_path.exists()

            # No IPC client connected, so pynput should handle output.
            daemon._on_hotkey(HotkeyEvent(action="press"))
            await asyncio.sleep(0.1)
            assert daemon._state == "recording"

            daemon._on_hotkey(HotkeyEvent(action="release"))
            await asyncio.sleep(0.3)

            async def wait_for_typed() -> None:
                for _ in range(100):
                    if typed_calls:
                        return
                    await asyncio.sleep(0.05)
                raise AssertionError("No typing occurred")

            await asyncio.wait_for(wait_for_typed(), timeout=5.0)

            daemon._shutdown()
            for t in [daemon_task]:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, GeneratorExit):
                    pass

            assert mock_start.called
            assert mock_stop.called

    # Incremental typing should only emit the newly recognized characters.
    assert typed_calls == [(0, "你好"), (0, "世"), (0, "界")]


@pytest.mark.asyncio
async def test_daemon_interrupt_aborts_recognition(tmp_path: Path) -> None:
    """An interrupt event while recording aborts the session without typing."""
    socket_path = tmp_path / "stream_interrupt.sock"
    config = IMEConfig(
        hotkey_device="pynput",
        hotkey_key="CTRL",
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

    mock_client = AsyncMock()
    mock_client.connect = AsyncMock()
    mock_client.send_chunk = AsyncMock()
    mock_client.send_json = AsyncMock()
    mock_client.close = AsyncMock()
    mock_client.iterate = MagicMock(return_value=async_iter_stream_results())

    with (
        patch("qwen3_asr_ime.daemon.service.ASRStreamClient", return_value=mock_client),
        patch("qwen3_asr_ime.daemon.service.create_hotkey_listener") as mock_hotkey,
    ):
        mock_hotkey.return_value.start = MagicMock()
        mock_hotkey.return_value.stop = MagicMock()

        typed_texts: list[str] = []

        def capture_type(to_delete: int, text: str) -> None:
            typed_texts.append(text)

        with (
            patch.object(daemon.recorder, "start") as mock_start,
            patch.object(daemon.recorder, "stop", return_value=_make_silent_wav()) as mock_stop,
            patch("qwen3_asr_ime.daemon.service._type_incremental_x11", side_effect=capture_type),
        ):
            daemon_task = asyncio.create_task(daemon.start())
            for _ in range(50):
                if socket_path.exists():
                    break
                await asyncio.sleep(0.05)
            assert socket_path.exists()

            daemon._on_hotkey(HotkeyEvent(action="press"))
            await asyncio.sleep(0.1)
            assert daemon._state == "recording"

            daemon._on_hotkey(HotkeyEvent(action="interrupt"))
            await asyncio.sleep(0.2)

            assert daemon._state == "idle"
            assert mock_start.called
            assert mock_stop.called

            # The partial that arrived before interrupt may have been typed,
            # but the final result must not be emitted after abort.
            assert "世界" not in "".join(typed_texts)

            # Subsequent release should be a no-op after interrupt.
            daemon._on_hotkey(HotkeyEvent(action="release"))
            await asyncio.sleep(0.1)
            assert daemon._state == "idle"

            daemon._shutdown()
            for t in [daemon_task]:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, GeneratorExit):
                    pass
