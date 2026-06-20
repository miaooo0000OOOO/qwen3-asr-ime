from __future__ import annotations

import io
import wave
from dataclasses import dataclass

import pyaudio


@dataclass(frozen=True, slots=True)
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    format: int = pyaudio.paInt16
    chunk_ms: int = 20

    @property
    def chunk_samples(self) -> int:
        return int(self.sample_rate * self.chunk_ms / 1000)


class Recorder:
    def __init__(self, config: AudioConfig | None = None):
        self.config = config or AudioConfig()
        self._audio = pyaudio.PyAudio()
        self._stream = None
        self._frames: list[bytes] = []
        self._is_recording = False

    def start(self) -> None:
        if self._is_recording:
            return
        self._frames = []
        self._stream = self._audio.open(
            format=self.config.format,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            input=True,
            frames_per_buffer=self.config.chunk_samples,
            stream_callback=self._callback,
        )
        self._is_recording = True

    def _callback(self, in_data, frame_count, time_info, status_flags):
        self._frames.append(in_data)
        return (None, pyaudio.paContinue)

    def stop(self) -> bytes:
        if not self._is_recording or self._stream is None:
            return b""
        self._stream.stop_stream()
        self._stream.close()
        self._stream = None
        self._is_recording = False
        return self._to_wav(b"".join(self._frames))

    def _to_wav(self, raw_pcm: bytes) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.config.channels)
            wf.setsampwidth(self._audio.get_sample_size(self.config.format))
            wf.setframerate(self.config.sample_rate)
            wf.writeframes(raw_pcm)
        return buf.getvalue()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        self._audio.terminate()
