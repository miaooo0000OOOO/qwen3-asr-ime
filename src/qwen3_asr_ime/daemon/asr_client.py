"""WebSocket client for streaming Qwen3-ASR recognition.

The client speaks the JSON protocol exposed by ``tools/asr_server.py`` on the
``/v1/asr/stream`` endpoint. It sends audio chunks as they are recorded and
receives incremental (``partial``) and final (``final``) recognition results.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator
from urllib.parse import urlparse

import websockets

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ASRResult:
    """A single recognition result from the streaming ASR server.

    Attributes:
        text: Recognized text content.
        language: Detected language, if available.
        error: Error message when recognition fails.
        final: ``True`` for the final result after ``finish`` is sent;
            ``False`` for incremental partial results.
    """

    text: str
    language: str | None = None
    error: str | None = None
    final: bool = False


class ASRStreamClient:
    """Async WebSocket client for streaming Qwen3-ASR recognition.

    The server must be running with the vLLM backend; otherwise the WebSocket
    endpoint will return an error and ``connect()`` will raise.

    Typical usage::

        async with ASRStreamClient("http://127.0.0.1:8000") as client:
            await client.send_chunk(pcm_bytes, fmt="pcm")
            async for result in client.iterate():
                print(result.text)
    """

    def __init__(self, endpoint: str, api_key: str = "dummy", timeout: float = 30.0):
        """Initialize the client.

        Args:
            endpoint: HTTP endpoint of the ASR server (e.g.
                ``"http://127.0.0.1:8000"``). It is converted to a WebSocket URL
                internally.
            api_key: Bearer token sent in the WebSocket handshake.
            timeout: Connection open timeout in seconds.
        """
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._ws: websockets.WebSocketClientProtocol | None = None

    @staticmethod
    def _http_to_ws(url: str) -> str:
        """Convert an HTTP(S) URL to the corresponding WebSocket URL."""
        parsed = urlparse(url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        return f"{scheme}://{parsed.netloc}/v1/asr/stream"

    async def connect(self, language: str | None = None) -> None:
        """Open the WebSocket connection and wait for the server ``ready`` message.

        Args:
            language: Optional language hint sent to the ASR model.

        Raises:
            RuntimeError: If the server does not respond with ``{"type": "ready"}``.
        """
        ws_url = self._http_to_ws(self.endpoint)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        logger.info("Connecting to streaming ASR endpoint: %s", ws_url)
        self._ws = await websockets.connect(
            ws_url, extra_headers=headers, open_timeout=self.timeout
        )
        config: dict[str, str | None] = {"type": "config"}
        if language is not None:
            config["language"] = language
        await self._ws.send(json.dumps(config))
        ready = json.loads(await self._ws.recv())
        if ready.get("type") != "ready":
            raise RuntimeError(f"Streaming ASR handshake failed: {ready}")
        logger.info("Streaming ASR connected")

    async def send_chunk(self, audio_bytes: bytes, fmt: str = "wav") -> None:
        """Send an audio chunk to the server.

        Args:
            audio_bytes: Raw audio bytes (WAV container or int16 PCM).
            fmt: ``"wav"`` or ``"pcm"``. The server decodes accordingly.

        Raises:
            RuntimeError: If called before ``connect()``.
        """
        if self._ws is None:
            raise RuntimeError("WebSocket not connected")
        b64 = base64.b64encode(audio_bytes).decode("utf-8")
        await self._ws.send(json.dumps({"type": "chunk", "format": fmt, "audio": b64}))

    async def send_json(self, data: dict[str, Any]) -> None:
        """Send an arbitrary JSON message to the server.

        Used internally for ``finish`` and can be reused for future protocol
        messages.

        Raises:
            RuntimeError: If called before ``connect()``.
        """
        if self._ws is None:
            raise RuntimeError("WebSocket not connected")
        await self._ws.send(json.dumps(data))

    async def finish(self) -> ASRResult:
        """Signal the end of the audio stream and return the final result.

        Sends ``{"type": "finish"}`` and consumes messages until the server
        returns ``final`` or ``error``.

        Returns:
            The final ``ASRResult``.
        """
        if self._ws is None:
            raise RuntimeError("WebSocket not connected")
        await self.send_json({"type": "finish"})
        async for result in self._iter_messages():
            if result.final or result.error:
                return result
        return ASRResult(text="", error="Connection closed before final result")

    async def iterate(self) -> AsyncIterator[ASRResult]:
        """Yield partial and final results until the stream is finished.

        This is the main consumption API. It yields ``partial`` results during
        recording and stops after the first ``final`` or ``error`` message.
        """
        async for result in self._iter_messages():
            yield result
            if result.final or result.error:
                return

    async def _iter_messages(self) -> AsyncIterator[ASRResult]:
        """Low-level message parser: read JSON messages and yield ``ASRResult``."""
        if self._ws is None:
            return
        try:
            while True:
                raw = await self._ws.recv()
                msg = json.loads(raw)
                msg_type = msg.get("type")
                if msg_type == "partial":
                    yield ASRResult(
                        text=msg.get("text", ""),
                        language=msg.get("language"),
                        final=False,
                    )
                elif msg_type == "final":
                    yield ASRResult(
                        text=msg.get("text", ""),
                        language=msg.get("language"),
                        final=True,
                    )
                    return
                elif msg_type == "error":
                    yield ASRResult(
                        text="",
                        error=msg.get("message", "Unknown streaming error"),
                    )
                    return
        except websockets.exceptions.ConnectionClosed as exc:
            yield ASRResult(text="", error=f"WebSocket closed: {exc}")

    async def close(self) -> None:
        """Close the WebSocket connection gracefully."""
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception as exc:
                logger.warning("Failed to close WebSocket cleanly: %s", exc)
            finally:
                self._ws = None

    async def __aenter__(self) -> "ASRStreamClient":
        """Async context manager entry: connect and return the client."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Async context manager exit: close the connection."""
        await self.close()


class ASRHttpClient:
    """Async HTTP client for non-streaming (offline) ASR recognition.

    Sends the complete WAV audio as a POST to ``/v1/asr/transcribe`` and
    returns a single ``ASRResult`` with the final transcription.

    Typical usage::

        client = ASRHttpClient("http://127.0.0.1:8000")
        result = await client.transcribe(wav_bytes)
        print(result.text)
    """

    def __init__(self, endpoint: str, api_key: str = "dummy", timeout: float = 30.0):
        """Initialize the client.

        Args:
            endpoint: HTTP endpoint of the ASR server (e.g.
                ``"http://127.0.0.1:8000"``).
            api_key: Bearer token sent in the Authorization header.
            timeout: Total request timeout in seconds.
        """
        self._transcribe_url = f"{endpoint.rstrip('/')}/v1/asr/transcribe"
        self._api_key = api_key
        self._timeout = timeout

    async def transcribe(
        self, wav_bytes: bytes, language: str | None = None
    ) -> ASRResult:
        """Send a complete WAV recording for recognition.

        Args:
            wav_bytes: WAV-encoded audio bytes (full recording).
            language: Optional language hint (e.g. ``"zh"``, ``"en"``).

        Returns:
            ``ASRResult`` with ``final=True`` on success, or ``error`` set
            on failure. Never raises — errors are returned in the result.
        """
        import aiohttp

        headers = {"Authorization": f"Bearer {self._api_key}"}
        params = {}
        if language:
            params["language"] = language

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._transcribe_url,
                    data=wav_bytes,
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=self._timeout),
                ) as resp:
                    if resp.status == 200:
                        body = await resp.json()
                        return ASRResult(
                            text=body.get("text", ""),
                            language=body.get("language"),
                            final=True,
                        )
                    else:
                        body_text = await resp.text()
                        return ASRResult(
                            text="",
                            error=f"ASR server returned {resp.status}: {body_text[:200]}",
                        )
        except Exception as exc:
            return ASRResult(text="", error=f"ASR request failed: {exc}")
