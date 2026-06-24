from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, NoReturn

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
    asr_mode: str              # "offline" | "streaming"
    asr_model: str             # "0.6B" | "1.7B"
    asr_backend: str           # "transformers" | "vllm"
    asr_device: str
    asr_quantization: str
    asr_api_key: str
    asr_timeout: float
    asr_auto_sleep_time: int   # seconds idle before stopping backend (0 = never)
    asr_backend_wait_timeout: int  # seconds to wait for backend /health
    ipc_socket_path: str
    log_level: str

    _MODEL_PATH_MAP: dict[str, str] = field(
        default_factory=lambda: {
            "0.6B": "/Data2/Models/Qwen3-ASR-0.6B",
            "1.7B": "/Data2/Models/Qwen3-ASR-1.7B",
        },
        repr=False,
        init=False,
    )

    @property
    def model_path(self) -> str:
        return self._MODEL_PATH_MAP.get(self.asr_model, self._MODEL_PATH_MAP["1.7B"])

    @classmethod
    def _validate(cls, data: dict[str, Any]) -> None:
        """Cross-validate mode+backend combination. Exits on invalid combo."""
        mode = data.get("asr_mode", "offline")
        backend = data.get("asr_backend", "transformers")
        valid_combos = {
            ("offline", "transformers"),
            ("streaming", "vllm"),
        }
        if (mode, backend) not in valid_combos:
            import sys
            print(
                f"ERROR: 不支持的 mode+backend 组合: mode={mode}, backend={backend}. "
                f"offline 需 transformers, streaming 需 vllm",
                file=sys.stderr,
            )
            sys.exit(1)

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
            asr_mode="offline",
            asr_model="1.7B",
            asr_backend="transformers",
            asr_device="auto",
            asr_quantization="auto",
            asr_api_key="dummy",
            asr_timeout=30.0,
            asr_auto_sleep_time=300,
            asr_backend_wait_timeout=120,
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
            "asr_mode": defaults.asr_mode,
            "asr_model": defaults.asr_model,
            "asr_backend": defaults.asr_backend,
            "asr_device": defaults.asr_device,
            "asr_quantization": defaults.asr_quantization,
            "asr_api_key": defaults.asr_api_key,
            "asr_timeout": defaults.asr_timeout,
            "asr_auto_sleep_time": defaults.asr_auto_sleep_time,
            "asr_backend_wait_timeout": defaults.asr_backend_wait_timeout,
            "ipc_socket_path": defaults.ipc_socket_path,
            "log_level": defaults.log_level,
        }

        _KEY_PATHS: dict[str, list[str]] = {
            "hotkey_device": ["hotkey", "device"],
            "hotkey_key": ["hotkey", "key"],
            "audio_sample_rate": ["audio", "sample_rate"],
            "audio_channels": ["audio", "channels"],
            "audio_format": ["audio", "format"],
            "audio_chunk_ms": ["audio", "chunk_ms"],
            "asr_endpoint": ["asr", "endpoint"],
            "asr_mode": ["asr", "mode"],
            "asr_model": ["asr", "model"],
            "asr_backend": ["asr", "backend"],
            "asr_device": ["asr", "device"],
            "asr_quantization": ["asr", "quantization"],
            "asr_api_key": ["asr", "api_key"],
            "asr_timeout": ["asr", "timeout"],
            "asr_auto_sleep_time": ["asr", "auto_sleep_time"],
            "asr_backend_wait_timeout": ["asr", "backend_wait_timeout"],
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

        cls._validate(data)
        return cls(**data)


class ConfigWatcher:
    """Periodically polls config file mtime; reloads on change.

    If the config file does not exist, creates a default one via
    ``IMEConfig.defaults()`` serialized to YAML. Failure to create or
    first-load the config results in ``sys.exit(1)``.

    On subsequent reload failures, the old config is retained and a
    warning is logged — the program does NOT exit.

    Attributes:
        config: The currently-effective ``IMEConfig``.
    """

    _POLL_INTERVAL: float = 5.0  # seconds (hardcoded, not in config file)

    def __init__(self, path: Path | None = None) -> None:
        import logging
        import sys

        self._logger = logging.getLogger(__name__)
        if path is None:
            path = (
                Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
                / "qwen3-asr-ime"
                / "config.yaml"
            )
        self._path = path

        # Ensure config file exists
        if not self._path.exists():
            self._create_default()

        # First load — must succeed or exit
        try:
            self._config = IMEConfig.load(self._path)
        except Exception as exc:
            print(
                f"ERROR: 配置文件解析失败: {self._path}: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

        self._mtime = self._path.stat().st_mtime

    @property
    def config(self) -> IMEConfig:
        return self._config

    def _create_default(self) -> None:
        """Create parent directory and default config YAML. Exits on failure."""
        import sys
        import yaml

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(
                f"ERROR: 无法创建配置目录: {self._path.parent}: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

        defaults = IMEConfig.defaults()
        yaml_data = {
            "hotkey": {
                "device": defaults.hotkey_device,
                "key": defaults.hotkey_key,
            },
            "audio": {
                "sample_rate": defaults.audio_sample_rate,
                "channels": defaults.audio_channels,
                "format": defaults.audio_format,
                "chunk_ms": defaults.audio_chunk_ms,
            },
            "asr": {
                "endpoint": defaults.asr_endpoint,
                "mode": defaults.asr_mode,
                "model": defaults.asr_model,
                "backend": defaults.asr_backend,
                "device": defaults.asr_device,
                "quantization": defaults.asr_quantization,
                "api_key": defaults.asr_api_key,
                "timeout": defaults.asr_timeout,
                "auto_sleep_time": defaults.asr_auto_sleep_time,
                "backend_wait_timeout": defaults.asr_backend_wait_timeout,
            },
            "ipc": {
                "socket_path": defaults.ipc_socket_path,
            },
            "logging": {
                "level": defaults.log_level,
            },
        }
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                yaml.safe_dump(yaml_data, f, allow_unicode=True, default_flow_style=False)
        except OSError as exc:
            print(
                f"ERROR: 无法创建默认配置文件: {self._path}: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)
        self._logger.info("Created default config at %s", self._path)

    def _reload(self) -> None:
        """Attempt to reload config from disk.

        On failure, retains the old config and logs a warning — does NOT exit.
        Called only for runtime reloads, not initial load.
        """
        try:
            new_config = IMEConfig.load(self._path)
            self._config = new_config
            self._logger.info("Config reloaded from %s", self._path)
        except Exception as exc:
            self._logger.warning(
                "Failed to reload config from %s: %s — keeping previous config",
                self._path,
                exc,
            )

    async def watch_loop(self, on_change: Callable[[IMEConfig], None]) -> NoReturn:
        """Run forever: poll config mtime every _POLL_INTERVAL seconds.

        When the mtime changes, reload the config and invoke ``on_change``
        with the new ``IMEConfig`` if it differs from the previous one.

        If the file is deleted, re-create the default config.

        Args:
            on_change: Async callback receiving the new config when it changes.
                Must not raise.
        """
        import asyncio
        while True:
            await asyncio.sleep(self._POLL_INTERVAL)
            try:
                stat = self._path.stat()
            except FileNotFoundError:
                self._logger.warning("Config file deleted; re-creating default")
                self._create_default()
                self._mtime = self._path.stat().st_mtime
                continue

            if stat.st_mtime != self._mtime:
                old = self._config
                self._reload()
                self._mtime = stat.st_mtime
                if self._config != old:
                    try:
                        on_change(self._config)
                    except Exception as exc:
                        self._logger.exception(
                            "Config change callback failed: %s", exc
                        )
