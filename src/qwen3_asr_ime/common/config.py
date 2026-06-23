from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    asr_timeout: float
    ipc_socket_path: str
    log_level: str

    @classmethod
    def defaults(cls, uid: int | None = None) -> "IMEConfig":
        if uid is None:
            uid = os.getuid()
        return cls(
            hotkey_device="auto",
            hotkey_key="CTRL",
            audio_sample_rate=16000,
            audio_channels=1,
            audio_format="int16",
            audio_chunk_ms=20,
            asr_endpoint="http://127.0.0.1:8000",
            asr_model="Qwen/Qwen3-ASR-0.6B",
            asr_device="auto",
            asr_quantization="auto",
            asr_api_key="dummy",
            asr_timeout=30.0,
            ipc_socket_path=f"{os.environ.get('XDG_RUNTIME_DIR', f'/run/user/{uid}')}/qwen3-asr-ime.sock",
            log_level="INFO",
        )

    @classmethod
    def load(cls, path: Path | None = None) -> "IMEConfig":
        if path is None:
            path = (
                Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
                / "qwen3-asr-ime"
                / "config.yaml"
            )
        defaults = cls.defaults()
        data: dict[str, Any] = {
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
            "asr_timeout": defaults.asr_timeout,
            "ipc_socket_path": defaults.ipc_socket_path,
            "log_level": defaults.log_level,
        }

        # Mapping from flat config key -> list of nested keys in the YAML file.
        _KEY_PATHS: dict[str, list[str]] = {
            "hotkey_device": ["hotkey", "device"],
            "hotkey_key": ["hotkey", "key"],
            "audio_sample_rate": ["audio", "sample_rate"],
            "audio_channels": ["audio", "channels"],
            "audio_format": ["audio", "format"],
            "audio_chunk_ms": ["audio", "chunk_ms"],
            "asr_endpoint": ["asr", "endpoint"],
            "asr_model": ["asr", "model"],
            "asr_device": ["asr", "device"],
            "asr_quantization": ["asr", "quantization"],
            "asr_api_key": ["asr", "api_key"],
            "asr_timeout": ["asr", "timeout"],
            "ipc_socket_path": ["ipc", "socket_path"],
            "log_level": ["logging", "level"],
        }

        def _get_nested(root: dict[str, Any], path: list[str]) -> Any:
            node: Any = root
            for key in path:
                if not isinstance(node, dict):
                    return None
                node = node.get(key)
            return node

        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            for flat_key, nested_path in _KEY_PATHS.items():
                value = _get_nested(loaded, nested_path)
                if value is not None:
                    data[flat_key] = value

        return cls(**data)
