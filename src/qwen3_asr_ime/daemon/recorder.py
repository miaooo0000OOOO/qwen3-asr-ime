from __future__ import annotations

import io
import wave
from dataclasses import dataclass

import sounddevice as sd
import numpy as np


@dataclass(frozen=True, slots=True)
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "int16"
    chunk_ms: int = 20

    @property
    def chunk_samples(self) -> int:
        return int(self.sample_rate * self.chunk_ms / 1000)


class Recorder:
    def __init__(self, config: AudioConfig | None = None):
        self.config = config or AudioConfig()
        self._stream: sd.InputStream | None = None
        self._frames: list[np.ndarray] = []
        self._is_recording = False

    def start(self) -> None:
        if self._is_recording:
            return
        self._frames = []
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
        self._frames.append(indata.copy())

    def stop(self) -> bytes:
        if not self._is_recording or self._stream is None:
            return b""
        self._stream.stop()
        self._stream.close()
        self._stream = None
        self._is_recording = False
        raw = np.concatenate(self._frames, axis=0).tobytes() if self._frames else b""
        return self._to_wav(raw)

    def _to_wav(self, raw_pcm: bytes) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.config.channels)
            wf.setsampwidth(2)  # int16 = 2 bytes
            wf.setframerate(self.config.sample_rate)
            wf.writeframes(raw_pcm)
        return buf.getvalue()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
