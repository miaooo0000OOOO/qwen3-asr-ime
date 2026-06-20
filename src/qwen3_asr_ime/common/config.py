from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class IMEConfig:
    hotkey_device: str
    hotkey_key: str
    audio_sample_rate: int
    audio_channels: int
    audio_format: str
    audio_chunk_ms: int
    asr_endpoint: str
    asr_model: str
    asr_device: str
    asr_quantization: str
    asr_api_key: str
    ipc_socket_path: str
    log_level: str

    @classmethod
    def defaults(cls, uid: int | None = None) -> "IMEConfig":
        if uid is None:
            uid = os.getuid()
        return cls(
            hotkey_device="evdev",
            hotkey_key="<Super>+<Shift>+R",
            audio_sample_rate=16000,
            audio_channels=1,
            audio_format="int16",
            audio_chunk_ms=20,
            asr_endpoint="http://127.0.0.1:8000/v1/audio/transcriptions",
            asr_model="Qwen/Qwen3-ASR",
            asr_device="auto",
            asr_quantization="auto",
            asr_api_key="dummy",
            ipc_socket_path=f"{os.environ.get('XDG_RUNTIME_DIR', f'/run/user/{uid}')}/qwen3-asr-ime.sock",
            log_level="INFO",
        )

    @classmethod
    def load(cls, path: Path | None = None) -> "IMEConfig":
        if path is None:
            path = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "qwen3-asr-ime" / "config.yaml"
        defaults = cls.defaults()
        data = {
            "hotkey_device": defaults.hotkey_device,
            "hotkey_key": defaults.hotkey_key,
            "audio_sample_rate": defaults.audio_sample_rate,
            "audio_channels": defaults.audio_channels,
            "audio_format": defaults.audio_format,
            "audio_chunk_ms": defaults.audio_chunk_ms,
            "asr_endpoint": defaults.asr_endpoint,
            "asr_model": defaults.asr_model,
            "asr_device": defaults.asr_device,
            "asr_quantization": defaults.asr_quantization,
            "asr_api_key": defaults.asr_api_key,
            "ipc_socket_path": defaults.ipc_socket_path,
            "log_level": defaults.log_level,
        }
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            known_keys = set(data.keys())
            data.update({k: v for k, v in loaded.items() if k in known_keys})
        return cls(**data)
