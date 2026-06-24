"""Microphone audio recording for the streaming ASR daemon.

This module provides:

- ``AudioConfig``: immutable configuration for audio capture (sample rate,
  channels, sample format, chunk duration).
- ``Recorder``: a thin wrapper around ``sounddevice.InputStream`` that collects
  audio frames, optionally forwards PCM chunks in real-time to a streaming ASR
  client, and returns the full recording as a WAV byte buffer on stop.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from typing import Callable

import numpy as np
import sounddevice as sd


@dataclass(frozen=True, slots=True)
class AudioConfig:
    """Configuration for audio input capture.

    Attributes:
        sample_rate: Capture sample rate in Hz (default 16kHz for ASR).
        channels: Number of input channels (default 1 = mono).
        dtype: NumPy sample format passed to sounddevice (default ``"int16"``).
        chunk_ms: Duration of each callback chunk in milliseconds. The
            ``Recorder`` uses this to compute ``blocksize`` for the input stream.
    """

    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "int16"
    chunk_ms: int = 20

    @property
    def chunk_samples(self) -> int:
        """Number of samples per channel in one chunk."""
        return int(self.sample_rate * self.chunk_ms / 1000)


class Recorder:
    """Record microphone audio and expose both streaming chunks and full WAV output.

    In the streaming ASR flow:

    1. ``start(chunk_callback)`` opens the input stream. Each audio chunk is
       appended to an internal frame buffer and also forwarded to
       ``chunk_callback`` as little-endian int16 bytes for real-time streaming.
    2. ``stop()`` closes the stream and returns the complete recording as a WAV
       byte buffer (used for finalization/logging).
    3. ``close()`` is a safety net for cleanup on shutdown.

    Args:
        config: Audio capture parameters. Uses defaults if omitted.
    """

    def __init__(self, config: AudioConfig | None = None):
        """Create a Recorder with optional AudioConfig (uses defaults if omitted)."""
        self.config = config or AudioConfig()
        self._stream: sd.InputStream | None = None
        self._frames: list[np.ndarray] = []
        self._is_recording = False
        self._chunk_callback: Callable[[bytes], None] | None = None

    def start(self, chunk_callback: Callable[[bytes], None] | None = None) -> None:
        """Open the microphone input stream and begin recording.

        Args:
            chunk_callback: Optional callback invoked on every audio chunk from
                the sounddevice thread. Receives int16 PCM bytes. Must not raise.
        """
        if self._is_recording:
            return
        self._frames = []
        self._chunk_callback = chunk_callback
        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=self.config.dtype,
            blocksize=self.config.chunk_samples,
            callback=self._callback,
        )
        self._stream.start()
        self._is_recording = True

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """sounddevice input callback: store chunk and forward to stream client.

        Runs on a dedicated audio thread. Any exception from ``chunk_callback``
        is swallowed to avoid crashing the audio stream.
        """
        self._frames.append(indata.copy())
        if self._chunk_callback is not None:
            try:
                self._chunk_callback(self._pcm_to_bytes(indata.copy()))
            except Exception:
                # Never raise from sounddevice callback.
                pass

    def stop(self) -> bytes:
        """Stop recording and return the captured audio as a WAV byte buffer.

        Returns:
            WAV-encoded bytes containing the full recording. Empty bytes if
            called while not recording.
        """
        if not self._is_recording or self._stream is None:
            return b""
        self._stream.stop()
        self._stream.close()
        self._stream = None
        self._is_recording = False
        self._chunk_callback = None
        raw = np.concatenate(self._frames, axis=0).tobytes() if self._frames else b""
        return self._to_wav(raw)

    def _to_wav(self, raw_pcm: bytes) -> bytes:
        """Wrap raw int16 PCM bytes in a WAV container."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.config.channels)
            wf.setsampwidth(2)  # int16 = 2 bytes
            wf.setframerate(self.config.sample_rate)
            wf.writeframes(raw_pcm)
        return buf.getvalue()

    def _pcm_to_bytes(self, pcm: np.ndarray) -> bytes:
        """Convert a numpy chunk to little-endian int16 bytes."""
        return pcm.astype(np.int16).tobytes()

    def close(self) -> None:
        """Release the input stream if it is still open."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
