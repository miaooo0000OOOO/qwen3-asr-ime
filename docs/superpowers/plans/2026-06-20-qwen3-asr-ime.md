# Qwen3-ASR Linux 语音输入法实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个基于 Qwen3-ASR 后端的 IBus 语音输入法，支持全局热键录音、本地 ASR API 识别、自动 GPU/CPU 选择。

**Architecture:** IBus 引擎负责与输入法框架交互；独立守护进程负责录音、调用 ASR 服务、管理状态；两者通过 Unix Socket 通信；ASR 服务由 vLLM/SGLang/llama.cpp 等本地 API 提供。

**Tech Stack:** Python 3.13, IBus (PyGObject), PyAudio/SoundDevice, evdev/pynput, requests, systemd user service.

---

## 文件结构

```
qwen3-asr-ime/
├── pyproject.toml
├── README.md
├── bin/
│   └── install.sh
├── systemd/
│   └── qwen3-asr-ime.service
├── src/
│   └── qwen3_asr_ime/
│       ├── __init__.py
│       ├── common/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   ├── protocol.py
│       │   └── logger.py
│       ├── daemon/
│       │   ├── __init__.py
│       │   ├── service.py
│       │   ├── recorder.py
│       │   ├── hotkey.py
│       │   ├── asr_client.py
│       │   └── process_manager.py
│       └── ibus/
│           ├── __init__.py
│           ├── engine.py
│           ├── factory.py
│           └── main.py
└── tests/
    ├── __init__.py
    ├── test_protocol.py
    ├── test_asr_client.py
    ├── test_recorder.py
    └── test_integration.py
```

---

## Task 1: 项目骨架与依赖配置

**Files:**
- Create: `/Data2/Code/python/qwen3-asr-ime/pyproject.toml`
- Create: `/Data2/Code/python/qwen3-asr-ime/src/qwen3_asr_ime/__init__.py`
- Create: `/Data2/Code/python/qwen3-asr-ime/src/qwen3_asr_ime/common/__init__.py`
- Create: `/Data2/Code/python/qwen3-asr-ime/src/qwen3_asr_ime/daemon/__init__.py`
- Create: `/Data2/Code/python/qwen3-asr-ime/src/qwen3_asr_ime/ibus/__init__.py`
- Create: `/Data2/Code/python/qwen3-asr-ime/tests/__init__.py`

- [ ] **Step 1: 创建 pyproject.toml**

```toml
[project]
name = "qwen3-asr-ime"
version = "0.1.0"
description = "Qwen3-ASR based Linux voice input method for IBus"
requires-python = ">=3.10"
dependencies = [
    "pygobject>=3.42",
    "pyaudio>=0.2.13",
    "requests>=2.31",
    "evdev>=1.6",
    "pynput>=1.7",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "responses>=0.25",
    "httpx>=0.27",
    "ruff>=0.4",
    "mypy>=1.9",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100

[tool.mypy]
python_version = "3.10"
strict = true
```

- [ ] **Step 2: 创建空模块文件**

```bash
touch src/qwen3_asr_ime/__init__.py
mkdir -p src/qwen3_asr_ime/common src/qwen3_asr_ime/daemon src/qwen3_asr_ime/ibus
touch src/qwen3_asr_ime/common/__init__.py src/qwen3_asr_ime/daemon/__init__.py src/qwen3_asr_ime/ibus/__init__.py
touch tests/__init__.py
```

- [ ] **Step 3: 验证目录结构**

Run: `tree -L 4`
Expected: 目录结构与文件结构一致。

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml src tests
git commit -m "chore: scaffold project structure"
```

---

## Task 2: 公共模块 - 协议、配置、日志

**Files:**
- Create: `src/qwen3_asr_ime/common/protocol.py`
- Create: `src/qwen3_asr_ime/common/config.py`
- Create: `src/qwen3_asr_ime/common/logger.py`
- Create: `tests/test_protocol.py`

- [ ] **Step 1: 编写协议模块**

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class RecognizedText:
    type: Literal["recognized"] = "recognized"
    text: str = ""
    confidence: float | None = None
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps({
            "type": self.type,
            "text": self.text,
            "confidence": self.confidence,
            "error": self.error,
        })

    @classmethod
    def from_dict(cls, data: dict) -> "RecognizedText":
        return cls(
            type=data.get("type", "recognized"),
            text=data.get("text", ""),
            confidence=data.get("confidence"),
            error=data.get("error"),
        )


@dataclass(frozen=True, slots=True)
class StateUpdate:
    type: Literal["state"] = "state"
    state: Literal["idle", "recording", "recognizing", "error"] = "idle"
    message: str | None = None

    def to_json(self) -> str:
        return json.dumps({
            "type": self.type,
            "state": self.state,
            "message": self.message,
        })

    @classmethod
    def from_dict(cls, data: dict) -> "StateUpdate":
        return cls(
            type=data.get("type", "state"),
            state=data.get("state", "idle"),
            message=data.get("message"),
        )


def parse_message(line: str) -> RecognizedText | StateUpdate:
    data = json.loads(line)
    msg_type = data.get("type")
    if msg_type == "recognized":
        return RecognizedText.from_dict(data)
    if msg_type == "state":
        return StateUpdate.from_dict(data)
    raise ValueError(f"Unknown message type: {msg_type}")
```

- [ ] **Step 2: 编写配置模块**

```python
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
            ipc_socket_path=f"/run/user/{uid}/qwen3-asr-ime.sock",
            log_level="INFO",
        )

    @classmethod
    def load(cls, path: Path | None = None) -> "IMEConfig":
        if path is None:
            path = Path.home() / ".config" / "qwen3-asr-ime" / "config.yaml"
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
            data.update(loaded)
        return cls(**data)
```

- [ ] **Step 3: 编写日志模块**

```python
import logging
import sys


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    return logger
```

- [ ] **Step 4: 编写协议测试**

```python
import pytest

from qwen3_asr_ime.common.protocol import (
    RecognizedText,
    StateUpdate,
    parse_message,
)


def test_recognized_text_roundtrip():
    msg = RecognizedText(text="你好 world", confidence=0.95)
    parsed = parse_message(msg.to_json())
    assert isinstance(parsed, RecognizedText)
    assert parsed.text == "你好 world"
    assert parsed.confidence == pytest.approx(0.95)


def test_state_update_roundtrip():
    msg = StateUpdate(state="recording", message="开始录音")
    parsed = parse_message(msg.to_json())
    assert isinstance(parsed, StateUpdate)
    assert parsed.state == "recording"
    assert parsed.message == "开始录音"


def test_parse_unknown_message_type():
    with pytest.raises(ValueError, match="Unknown message type"):
        parse_message('{"type": "unknown"}')
```

- [ ] **Step 5: 运行协议测试**

Run: `pytest tests/test_protocol.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/qwen3_asr_ime/common tests/test_protocol.py
git commit -m "feat(common): add protocol, config, logger"
```

---

## Task 3: ASR HTTP 客户端

**Files:**
- Create: `src/qwen3_asr_ime/daemon/asr_client.py`
- Create: `tests/test_asr_client.py`

- [ ] **Step 1: 编写 ASR 客户端**

```python
from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ASRResult:
    text: str
    error: str | None = None


class ASRClient:
    def __init__(self, endpoint: str, api_key: str = "dummy", timeout: float = 30.0):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def recognize(self, audio_bytes: bytes, sample_rate: int = 16000) -> ASRResult:
        url = f"{self.endpoint}/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        files = {
            "file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav"),
        }
        data = {
            "model": "qwen3-asr",
            "language": "zh",
            "response_format": "json",
        }
        try:
            resp = requests.post(url, headers=headers, files=files, data=data, timeout=self.timeout)
            resp.raise_for_status()
            payload = resp.json()
            text = payload.get("text", "")
            return ASRResult(text=text)
        except requests.RequestException as exc:
            logger.error("ASR request failed: %s", exc)
            return ASRResult(text="", error=str(exc))
```

- [ ] **Step 2: 编写 ASR 客户端测试**

```python
import pytest
import responses

from qwen3_asr_ime.daemon.asr_client import ASRClient


@responses.activate
def test_recognize_success():
    responses.post(
        "http://localhost:8000/v1/audio/transcriptions",
        json={"text": "你好"},
        status=200,
    )
    client = ASRClient("http://localhost:8000")
    result = client.recognize(b"fake wav data")
    assert result.text == "你好"
    assert result.error is None


@responses.activate
def test_recognize_failure():
    responses.post(
        "http://localhost:8000/v1/audio/transcriptions",
        status=500,
    )
    client = ASRClient("http://localhost:8000", timeout=2.0)
    result = client.recognize(b"fake wav data")
    assert result.text == ""
    assert result.error is not None
```

- [ ] **Step 3: 运行测试**

Run: `pytest tests/test_asr_client.py -v`
Expected: 2 passed.

- [ ] **Step 4: Commit**

```bash
git add src/qwen3_asr_ime/daemon/asr_client.py tests/test_asr_client.py
git commit -m "feat(daemon): add ASR HTTP client"
```

---

## Task 4: 录音模块

**Files:**
- Create: `src/qwen3_asr_ime/daemon/recorder.py`
- Create: `tests/test_recorder.py`

- [ ] **Step 1: 安装 PyAudio 依赖并验证**

Run: `uv pip install pyaudio`
Expected: 安装成功。

- [ ] **Step 2: 编写录音模块**

```python
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
```

- [ ] **Step 3: 编写录音测试（使用 mock）**

```python
from unittest.mock import MagicMock, patch

from qwen3_asr_ime.daemon.recorder import AudioConfig, Recorder


@patch("qwen3_asr_ime.daemon.recorder.pyaudio.PyAudio")
def test_recorder_start_stop(mock_pyaudio_cls):
    mock_audio = MagicMock()
    mock_stream = MagicMock()
    mock_pyaudio_cls.return_value = mock_audio
    mock_audio.open.return_value = mock_stream
    mock_audio.get_sample_size.return_value = 2

    recorder = Recorder(AudioConfig(sample_rate=16000, chunk_ms=20))
    recorder.start()
    assert recorder._is_recording is True

    mock_stream.stop_stream.assert_not_called()
    recorder.stop()
    assert recorder._is_recording is False
    mock_stream.stop_stream.assert_called_once()


@patch("qwen3_asr_ime.daemon.recorder.pyaudio.PyAudio")
def test_recorder_wav_output(mock_pyaudio_cls):
    mock_audio = MagicMock()
    mock_stream = MagicMock()
    mock_pyaudio_cls.return_value = mock_audio
    mock_audio.open.return_value = mock_stream
    mock_audio.get_sample_size.return_value = 2

    recorder = Recorder(AudioConfig(sample_rate=16000, chunk_ms=20))
    recorder.start()
    # simulate callback
    callback = mock_audio.open.call_args[1]["stream_callback"]
    callback(b"\x00\x01", None, None, None)
    wav_bytes = recorder.stop()

    assert wav_bytes.startswith(b"RIFF")
```

- [ ] **Step 4: 运行测试**

Run: `pytest tests/test_recorder.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qwen3_asr_ime/daemon/recorder.py tests/test_recorder.py
git commit -m "feat(daemon): add audio recorder"
```

---

## Task 5: 全局热键监听

**Files:**
- Create: `src/qwen3_asr_ime/daemon/hotkey.py`

- [ ] **Step 1: 编写 evdev 热键监听**

```python
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import evdev


@dataclass(frozen=True, slots=True)
class HotkeyEvent:
    action: Literal["press", "release"]


class EvdevHotkeyListener:
    def __init__(
        self,
        key_combo: str,
        on_event: Callable[[HotkeyEvent], None],
    ):
        self.key_combo = key_combo
        self.on_event = on_event
        self._pressed: set[int] = set()
        self._target_codes = self._parse_combo(key_combo)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._stop = threading.Event()

    @staticmethod
    def _parse_combo(combo: str) -> set[int]:
        name_map = {
            "SUPER": evdev.ecodes.ecodes["KEY_LEFTMETA"],
            "SHIFT": evdev.ecodes.ecodes["KEY_LEFTSHIFT"],
            "CTRL": evdev.ecodes.ecodes["KEY_LEFTCTRL"],
            "ALT": evdev.ecodes.ecodes["KEY_LEFTALT"],
        }
        codes: set[int] = set()
        for part in combo.upper().replace("<", "").replace(">", "").split("+"):
            part = part.strip()
            if part in name_map:
                codes.add(name_map[part])
            else:
                codes.add(evdev.ecodes.ecodes.get(f"KEY_{part}", 0))
        return codes

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        for dev in devices:
            dev.grab()
        try:
            while not self._stop.is_set():
                for dev in devices:
                    try:
                        for event in dev.read():
                            if event.type == evdev.ecodes.EV_KEY:
                                self._handle(event)
                    except BlockingIOError:
                        continue
        finally:
            for dev in devices:
                dev.ungrab()

    def _handle(self, event: evdev.InputEvent) -> None:
        code = event.code
        if event.value == 1:  # key down
            self._pressed.add(code)
            if self._pressed == self._target_codes:
                self.on_event(HotkeyEvent("press"))
        elif event.value == 0:  # key up
            if code in self._pressed:
                self._pressed.remove(code)
                if code in self._target_codes:
                    self.on_event(HotkeyEvent("release"))
```

- [ ] **Step 2: 编写 pynput 热键监听（备选）**

```python
from pynput import keyboard

from qwen3_asr_ime.daemon.hotkey import HotkeyEvent


class PynputHotkeyListener:
    def __init__(self, key_combo: str, on_event):
        self.on_event = on_event
        self._pressed = set()
        self._target = self._parse_combo(key_combo)
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )

    @staticmethod
    def _parse_combo(combo: str) -> set:
        parts = set(p.strip().upper() for p in combo.replace("<", "").replace(">", "").split("+"))
        return parts

    def _on_press(self, key):
        name = self._name(key)
        self._pressed.add(name)
        if self._pressed == self._target:
            self.on_event(HotkeyEvent("press"))

    def _on_release(self, key):
        name = self._name(key)
        if name in self._pressed:
            self._pressed.remove(name)
            if name in self._target:
                self.on_event(HotkeyEvent("release"))

    @staticmethod
    def _name(key):
        try:
            return key.name.upper()
        except AttributeError:
            return str(key).upper()

    def start(self):
        self._listener.start()

    def stop(self):
        self._listener.stop()
```

注：将 PynputHotkeyListener 也放入 `hotkey.py` 中。

- [ ] **Step 3: Commit**

```bash
git add src/qwen3_asr_ime/daemon/hotkey.py
git commit -m "feat(daemon): add global hotkey listener"
```

---

## Task 6: 守护进程主服务

**Files:**
- Create: `src/qwen3_asr_ime/daemon/service.py`
- Modify: `src/qwen3_asr_ime/daemon/hotkey.py`（导出工厂函数）

- [ ] **Step 1: 在 hotkey.py 增加工厂函数**

```python
def create_hotkey_listener(device: str, key_combo: str, on_event):
    if device == "evdev":
        return EvdevHotkeyListener(key_combo, on_event)
    if device == "pynput":
        return PynputHotkeyListener(key_combo, on_event)
    raise ValueError(f"Unsupported hotkey device: {device}")
```

- [ ] **Step 2: 编写守护进程主服务**

```python
from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

from qwen3_asr_ime.common.config import IMEConfig
from qwen3_asr_ime.common.logger import get_logger
from qwen3_asr_ime.common.protocol import RecognizedText, StateUpdate
from qwen3_asr_ime.daemon.asr_client import ASRClient
from qwen3_asr_ime.daemon.hotkey import HotkeyEvent, create_hotkey_listener
from qwen3_asr_ime.daemon.recorder import AudioConfig, Recorder

logger = get_logger(__name__)


class VoiceInputDaemon:
    def __init__(self, config: IMEConfig):
        self.config = config
        self.recorder = Recorder(
            AudioConfig(
                sample_rate=config.audio_sample_rate,
                channels=config.audio_channels,
                chunk_ms=config.audio_chunk_ms,
            )
        )
        self.asr = ASRClient(config.asr_endpoint, api_key=config.asr_api_key)
        self.hotkey = create_hotkey_listener(
            config.hotkey_device,
            config.hotkey_key,
            self._on_hotkey,
        )
        self._clients: set[asyncio.StreamWriter] = set()
        self._state: str = "idle"
        self._server = None

    async def start(self) -> None:
        socket_path = Path(self.config.ipc_socket_path)
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        if socket_path.exists():
            socket_path.unlink()

        self._server = await asyncio.start_unix_server(
            self._on_client_connected,
            path=str(socket_path),
        )
        os.chmod(socket_path, 0o600)
        self.hotkey.start()
        logger.info("Daemon started, listening on %s", socket_path)

    async def run_forever(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._shutdown)
        async with self._server:
            await self._server.serve_forever()

    def _on_hotkey(self, event: HotkeyEvent) -> None:
        if event.action == "press" and self._state == "idle":
            self._state = "recording"
            self.recorder.start()
            self._broadcast_state("recording", "开始录音")
            logger.info("Recording started")
        elif event.action == "release" and self._state == "recording":
            self._state = "recognizing"
            self._broadcast_state("recognizing", "识别中...")
            logger.info("Recording stopped, recognizing")
            audio_bytes = self.recorder.stop()
            asyncio.create_task(self._recognize(audio_bytes))

    async def _recognize(self, audio_bytes: bytes) -> None:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self.asr.recognize, audio_bytes)
        if result.error:
            self._broadcast_state("error", f"识别失败: {result.error}")
            self._broadcast_recognized("", error=result.error)
        else:
            self._broadcast_state("idle", None)
            self._broadcast_recognized(result.text)
            logger.info("Recognized: %s", result.text)

    def _broadcast_state(self, state: str, message: str | None) -> None:
        msg = StateUpdate(state=state, message=message).to_json()
        self._broadcast(msg)

    def _broadcast_recognized(self, text: str, error: str | None = None) -> None:
        msg = RecognizedText(text=text, error=error).to_json()
        self._broadcast(msg)

    def _broadcast(self, msg: str) -> None:
        data = (msg + "\n").encode("utf-8")
        for writer in list(self._clients):
            try:
                writer.write(data)
                asyncio.create_task(writer.drain())
            except Exception as exc:
                logger.warning("Failed to send to client: %s", exc)

    def _on_client_connected(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self._clients.add(writer)
        logger.info("IBus engine connected")

        async def read_loop():
            while True:
                try:
                    line = await reader.readline()
                    if not line:
                        break
                except Exception:
                    break

        asyncio.create_task(read_loop())

    def _shutdown(self) -> None:
        logger.info("Shutting down daemon")
        if self._server:
            self._server.close()
        self.hotkey.stop()
        self.recorder.close()


async def main():
    config = IMEConfig.load()
    daemon = VoiceInputDaemon(config)
    await daemon.start()
    await daemon.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 3: Commit**

```bash
git add src/qwen3_asr_ime/daemon/service.py src/qwen3_asr_ime/daemon/hotkey.py
git commit -m "feat(daemon): add voice input daemon service"
```

---

## Task 7: ASR 服务进程管理

**Files:**
- Create: `src/qwen3_asr_ime/daemon/process_manager.py`

- [ ] **Step 1: 编写 ASR 进程管理器**

```python
from __future__ import annotations

import logging
import shutil
import subprocess
import time

import requests

logger = logging.getLogger(__name__)


class ASRProcessManager:
    def __init__(self, model: str, device: str = "auto", quantization: str = "auto"):
        self.model = model
        self.device = device
        self.quantization = quantization
        self._process: subprocess.Popen | None = None

    def _detect_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            if torch.cuda.is_available():
                mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if mem >= 6:
                    return "cuda"
        except Exception:
            pass
        return "cpu"

    def _build_command(self) -> list[str]:
        device = self._detect_device()
        if shutil.which("vllm"):
            cmd = [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", self.model,
                "--port", "8000",
            ]
            if device == "cpu":
                cmd.extend(["--device", "cpu"])
            return cmd
        if shutil.which("llama-server"):
            return ["llama-server", "-m", self.model, "--port", "8000"]
        raise RuntimeError("No supported ASR server backend found (vllm or llama-server)")

    def start(self) -> None:
        if self.is_running():
            logger.info("ASR service already running")
            return
        cmd = self._build_command()
        logger.info("Starting ASR service: %s", " ".join(cmd))
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._wait_for_ready()

    def _wait_for_ready(self, timeout: float = 120.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = requests.get("http://127.0.0.1:8000/health", timeout=1)
                if resp.status_code == 200:
                    logger.info("ASR service ready")
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        raise TimeoutError("ASR service did not become ready")

    def stop(self) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    @staticmethod
    def is_running() -> bool:
        try:
            resp = requests.get("http://127.0.0.1:8000/health", timeout=1)
            return resp.status_code == 200
        except requests.RequestException:
            return False
```

- [ ] **Step 2: Commit**

```bash
git add src/qwen3_asr_ime/daemon/process_manager.py
git commit -m "feat(daemon): add ASR process manager"
```

---

## Task 8: IBus 引擎

**Files:**
- Create: `src/qwen3_asr_ime/ibus/engine.py`
- Create: `src/qwen3_asr_ime/ibus/factory.py`
- Create: `src/qwen3_asr_ime/ibus/main.py`

- [ ] **Step 1: 编写 IBus 引擎**

```python
from __future__ import annotations

import asyncio
from pathlib import Path

import gi

gi.require_version("IBus", "1.0")
from gi.repository import GLib, IBus

from qwen3_asr_ime.common.config import IMEConfig
from qwen3_asr_ime.common.logger import get_logger
from qwen3_asr_ime.common.protocol import RecognizedText, StateUpdate, parse_message

logger = get_logger(__name__)


class Qwen3ASREngine(IBus.Engine):
    __gtype_name__ = "Qwen3ASREngine"

    def __init__(self):
        super().__init__()
        self.config = IMEConfig.load()
        self._reader = None
        self._writer = None
        self._prop_list = IBus.PropList()
        self._prop_list.append(
            IBus.Property(
                key="status",
                type=IBus.PropType.NORMAL,
                label=IBus.Text.new_from_string("Qwen3-ASR 就绪"),
            )
        )
        GLib.timeout_add(100, self._connect_to_daemon)

    def _connect_to_daemon(self):
        socket_path = Path(self.config.ipc_socket_path)
        if not socket_path.exists():
            return True
        try:
            asyncio.ensure_future(self._connect(socket_path))
            asyncio.ensure_future(self._read_loop())
            return False
        except Exception as exc:
            logger.warning("Failed to connect to daemon: %s", exc)
            return True

    async def _connect(self, socket_path: Path):
        reader, writer = await asyncio.open_unix_connection(str(socket_path))
        self._reader = reader
        self._writer = writer
        logger.info("Connected to daemon")

    async def _read_loop(self):
        while True:
            try:
                line = await self._reader.readline()
                if not line:
                    break
                msg = parse_message(line.decode("utf-8"))
                GLib.idle_add(self._handle_message, msg)
            except Exception as exc:
                logger.error("Read loop error: %s", exc)
                break

    def _handle_message(self, msg):
        if isinstance(msg, RecognizedText):
            if msg.error:
                self.update_property(
                    "status",
                    IBus.Property(
                        key="status",
                        type=IBus.PropType.NORMAL,
                        label=IBus.Text.new_from_string(f"错误: {msg.error[:20]}"),
                    ),
                )
            elif msg.text:
                self.commit_text(IBus.Text.new_from_string(msg.text))
                self.update_property(
                    "status",
                    IBus.Property(
                        key="status",
                        type=IBus.PropType.NORMAL,
                        label=IBus.Text.new_from_string("Qwen3-ASR 就绪"),
                    ),
                )
        elif isinstance(msg, StateUpdate):
            label = msg.message or msg.state
            self.update_property(
                "status",
                IBus.Property(
                    key="status",
                    type=IBus.PropType.NORMAL,
                    label=IBus.Text.new_from_string(label),
                ),
            )
        return False

    def do_focus_in(self):
        self.register_properties(self._prop_list)

    def do_focus_out(self):
        pass

    def do_property_activate(self, prop_name: str, prop_state: int):
        logger.info("Property activated: %s", prop_name)
```

- [ ] **Step 2: 编写工厂和入口**

`factory.py`:

```python
import gi

gi.require_version("IBus", "1.0")
from gi.repository import IBus

from qwen3_asr_ime.ibus.engine import Qwen3ASREngine


class EngineFactory(IBus.Factory):
    __gtype_name__ = "Qwen3ASREngineFactory"

    def __init__(self, bus):
        super().__init__(connection=bus.get_connection())
        self.bus = bus

    def do_create_engine(self, engine_name):
        if engine_name == "qwen3-asr-ime":
            return Qwen3ASREngine()
        return None
```

`main.py`:

```python
import sys

import gi

gi.require_version("IBus", "1.0")
from gi.repository import GLib, GObject, IBus

from qwen3_asr_ime.common.logger import get_logger
from qwen3_asr_ime.ibus.factory import EngineFactory

logger = get_logger(__name__)


def main():
    IBus.init()
    bus = IBus.Bus()
    if not bus.is_connected():
        logger.error("Cannot connect to IBus daemon")
        sys.exit(1)

    factory = EngineFactory(bus)
    factory.add_engine("qwen3-asr-ime", GObject.type_from_name("Qwen3ASREngine"))

    loop = GLib.MainLoop()
    logger.info("Qwen3-ASR IME engine started")
    loop.run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add src/qwen3_asr_ime/ibus
git commit -m "feat(ibus): add IBus engine, factory and entrypoint"
```

---

## Task 9: 安装脚本与 systemd 服务

**Files:**
- Create: `bin/install.sh`
- Create: `systemd/qwen3-asr-ime.service`

- [ ] **Step 1: 编写 install.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_DIR="${HOME}/.config/qwen3-asr-ime"
IBUS_COMPONENT_DIR="${HOME}/.ibus/components"
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"

mkdir -p "${CONFIG_DIR}" "${IBUS_COMPONENT_DIR}" "${SYSTEMD_USER_DIR}"

# Install Python package in editable mode
python3 -m pip install -e "${PROJECT_DIR}"

# Create default config if missing
if [[ ! -f "${CONFIG_DIR}/config.yaml" ]]; then
cat > "${CONFIG_DIR}/config.yaml" <<'EOF'
hotkey:
  device: "evdev"
  key: "<Super>+<Shift>+R"
audio:
  sample_rate: 16000
  channels: 1
  format: "int16"
  chunk_ms: 20
asr:
  endpoint: "http://127.0.0.1:8000/v1/audio/transcriptions"
  model: "Qwen/Qwen3-ASR"
  device: "auto"
  quantization: "auto"
  api_key: "dummy"
ipc:
  socket_path: "/run/user/${UID}/qwen3-asr-ime.sock"
logging:
  level: "INFO"
EOF
fi

# Register IBus component
cat > "${IBUS_COMPONENT_DIR}/qwen3-asr-ime.xml" <<EOF
<?xml version="1.0" encoding="utf-8"?>
<component>
  <name>qwen3-asr-ime</name>
  <description>Qwen3-ASR Voice Input Method</description>
  <exec>$(which python3) -m qwen3_asr_ime.ibus.main</exec>
  <version>0.1.0</version>
  <author>Assistant</author>
  <license>MIT</license>
  <homepage>https://github.com/example/qwen3-asr-ime</homepage>
  <textdomain>qwen3-asr-ime</textdomain>
  <engines>
    <engine>
      <name>qwen3-asr-ime</name>
      <language>zh</language>
      <author>Assistant</author>
      <icon>microphone</icon>
      <display_name>Qwen3-ASR</display_name>
      <symbol>🎤</symbol>
      <setup></setup>
    </engine>
  </engines>
</component>
EOF

# Install systemd user service
cat > "${SYSTEMD_USER_DIR}/qwen3-asr-ime.service" <<EOF
[Unit]
Description=Qwen3-ASR Voice Input Daemon
After=graphical-session.target

[Service]
Type=simple
ExecStart=$(which python3) -m qwen3_asr_ime.daemon.service
Restart=on-failure
Environment="PYTHONUNBUFFERED=1"

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now qwen3-asr-ime || true

echo "Installation complete. Please re-login or restart IBus:"
echo "  ibus restart"
echo "Then add 'Qwen3-ASR' in IBus preferences."
```

- [ ] **Step 2: 编写 systemd 服务**

```ini
[Unit]
Description=Qwen3-ASR Voice Input Daemon
After=graphical-session.target

[Service]
Type=simple
ExecStart=$(which python3) -m qwen3_asr_ime.daemon.service
Restart=on-failure
Environment="PYTHONUNBUFFERED=1"

[Install]
WantedBy=default.target
```

- [ ] **Step 3: 使 install.sh 可执行并验证**

Run:
```bash
chmod +x bin/install.sh
bash -n bin/install.sh
```
Expected: 无语法错误。

- [ ] **Step 4: Commit**

```bash
git add bin/install.sh systemd/qwen3-asr-ime.service
git commit -m "feat(deploy): add install script and systemd user service"
```

---

## Task 10: 集成测试

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: 编写集成测试**

```python
import asyncio
import io
import wave
from pathlib import Path

import pytest
import responses

from qwen3_asr_ime.common.config import IMEConfig
from qwen3_asr_ime.daemon.service import VoiceInputDaemon


def _make_silent_wav(duration_sec: float = 0.5) -> bytes:
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

    daemon = VoiceInputDaemon(config)
    # Simulate hotkey press/release without real hardware
    daemon._state = "recording"
    daemon.recorder._frames = [_make_silent_wav()]
    daemon._state = "recognizing"
    audio = daemon.recorder.stop()

    received = []

    async def client():
        await asyncio.sleep(0.1)
        reader, writer = await asyncio.open_unix_connection(str(socket_path))
        async for line in reader:
            received.append(line.decode("utf-8").strip())

    daemon_task = asyncio.create_task(daemon.start())
    client_task = asyncio.create_task(client())

    # Wait for socket to be ready
    await asyncio.sleep(0.2)
    await daemon._recognize(audio)

    await asyncio.wait_for(client_task, timeout=2)
    daemon._shutdown()
    await daemon_task

    assert any("测试文本" in msg for msg in received)
```

- [ ] **Step 2: 运行集成测试**

Run: `pytest tests/test_integration.py -v`
Expected: 1 passed。

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add daemon integration test"
```

---

## Task 11: README 与最终验证

**Files:**
- Create: `README.md`

- [ ] **Step 1: 编写 README**

```markdown
# Qwen3-ASR Linux 语音输入法

基于 Qwen3-ASR 的 IBus 语音输入法。按住全局热键录音，松开后自动输入识别文本。

## 特性

- IBus 输入法前端
- 本地 Qwen3-ASR API 服务
- 自动 GPU/CPU 选择
- 中文+英文自动识别
- systemd user service 自启动

## 安装

```bash
cd /Data2/Code/python/qwen3-asr-ime
./bin/install.sh
```

安装完成后重新登录或重启 IBus：

```bash
ibus restart
```

然后在 IBus 设置中添加 "Qwen3-ASR"。

## 配置

编辑 `~/.config/qwen3-asr-ime/config.yaml`：

```yaml
hotkey:
  device: "evdev"  # evdev 或 pynput
  key: "<Super>+<Shift>+R"
asr:
  endpoint: "http://127.0.0.1:8000/v1/audio/transcriptions"
  model: "Qwen/Qwen3-ASR"
  device: "auto"
```

## 运行测试

```bash
pytest -q
```

## 系统要求

- Ubuntu 24.04 / Linux with IBus
- Python 3.10+
- PortAudio / PyAudio
- NVIDIA GPU + CUDA（可选，用于 GPU 推理）
```

- [ ] **Step 2: 运行全部测试**

Run: `pytest -q`
Expected: 所有测试通过。

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add README"
```

---

## Self-Review

- [ ] **Spec coverage:** 设计文档中所有需求（IBus 前端、本地 API、自动 GPU/CPU、全局热键、中英识别）都有对应任务。
- [ ] **Placeholder scan:** 计划中没有 TBD/TODO/"实现 later" 等占位符。
- [ ] **Type consistency:** `StateUpdate.state`, `RecognizedText.type` 等类型在 protocol、engine、service 中一致。
- [ ] **Command验证:** 每个测试任务都包含具体 `pytest` 命令和期望输出。

---

## 执行选项

Plan complete and saved to `/Data2/Code/python/qwen3-asr-ime/docs/superpowers/plans/2026-06-20-qwen3-asr-ime.md`. Two execution options:

**1. Subagent-Driven (recommended)** - 为每个 Task 分派独立子代理，我负责审阅和合并。

**2. Inline Execution** - 在本会话中按 Task 顺序直接实现。

Which approach?
