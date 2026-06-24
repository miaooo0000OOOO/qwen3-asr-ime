# Non-Streaming ASR Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add non-streaming (offline) ASR mode with transformers backend, config file hot-reloading, and auto-sleep backend lifecycle management.

**Architecture:** New `ConfigWatcher` polls config mtime every 5s; `BackendManager` spawns/monitors/kills the backend child process with health-polling and idle timeout; `ASRHttpClient` sends WAV via HTTP POST for non-streaming recognition; `VoiceInputDaemon` routes between streaming and non-streaming modes.

**Tech Stack:** Python 3.10+, asyncio, httpx, PyYAML, qwen-asr 0.0.6, transformers, FastAPI/uvicorn

## Global Constraints

- Non-streaming is the default mode
- Config file: `~/.config/qwen3-asr-ime/config.yaml`
- Config poll interval: 5 seconds (hardcoded)
- All unexpected errors MUST call `sys.exit(1)` — never silently continue
- `mode="offline"` requires `backend="transformers"`; `mode="streaming"` requires `backend="vllm"`
- `auto_sleep_time: 0` means backend never sleeps
- Model mapping: `"0.6B"` → `/Data2/Models/Qwen3-ASR-0.6B`, `"1.7B"` → `/Data2/Models/Qwen3-ASR-1.7B`

---

### Task 1: Extend IMEConfig with new fields and validation

**Files:**
- Modify: `src/qwen3_asr_ime/common/config.py`

**Interfaces:**
- Produces: `IMEConfig` gains 5 new frozen fields; `IMEConfig._validate()` classmethod; updated `defaults()` and `load()` classmethods; new `_KEY_PATHS` entries

- [ ] **Step 1: Add new fields to IMEConfig and update defaults()**

In `src/qwen3_asr_ime/common/config.py`, replace the `IMEConfig` dataclass and its `defaults()` method:

```python
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
```

Note: `field` requires importing `from dataclasses import dataclass, field`.

- [ ] **Step 2: Add _validate() classmethod**

Add the validation method to `IMEConfig`:

```python
    @classmethod
    def _validate(cls, data: dict[str, Any]) -> None:
        """Cross-validate mode+backend combination. Exits on invalid combo."""
        mode = data.get("asr_mode", "offline")
        backend = data.get("asr_backend", "transformers")
        valid_combos = {
            ("offline", "transformers"),
            ("offline", "vllm"),
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
```

- [ ] **Step 3: Update _KEY_PATHS and load() to include new fields + call _validate**

In `load()`, add the new key paths to `_KEY_PATHS` and call `_validate` before constructing:

```python
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
```

And in the `load()` method, add defaults for the new fields in `data`, then call validate:

```python
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
```

- [ ] **Step 4: Run existing tests to confirm no regressions**

```bash
cd /Data2/Code/python/qwen3-asr-ime && python3 -m pytest tests/ -v -m "not e2e"
```
Expected: all existing tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/qwen3_asr_ime/common/config.py
git commit -m "feat(config): add mode, model, backend, sleep/wait fields with cross-validation"
```

---

### Task 2: Create ConfigWatcher class

**Files:**
- Modify: `src/qwen3_asr_ime/common/config.py`

**Interfaces:**
- Produces: `ConfigWatcher` class with `__init__(path)`, `config` property, `watch_loop(on_change)` async method, `_create_default()`, `_reload()`
- Consumes: `IMEConfig.load(path)` from Task 1

- [ ] **Step 1: Add ConfigWatcher class to common/config.py**

Append the following class after `IMEConfig` at the end of `src/qwen3_asr_ime/common/config.py`:

```python
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
```

The `Callable` type requires adding at the top:
```python
from typing import Any, Callable, NoReturn
```

- [ ] **Step 2: Extend test_config.py with ConfigWatcher tests**

Create `tests/test_config_watcher.py`:

```python
"""Tests for ConfigWatcher."""
import asyncio
import tempfile
import time
from pathlib import Path

import pytest

from qwen3_asr_ime.common.config import ConfigWatcher, IMEConfig


def test_create_default_config():
    """ConfigWatcher creates a default config file when none exists."""
    with tempfile.TemporaryDirectory() as td:
        config_path = Path(td) / "config.yaml"
        watcher = ConfigWatcher(path=config_path)
        assert config_path.exists()
        assert watcher.config.asr_mode == "offline"
        assert watcher.config.asr_model == "1.7B"


def test_load_existing_config():
    """ConfigWatcher loads an existing valid config."""
    with tempfile.TemporaryDirectory() as td:
        config_path = Path(td) / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("""
asr:
  mode: "streaming"
  model: "0.6B"
  backend: "vllm"
""")
        watcher = ConfigWatcher(path=config_path)
        assert watcher.config.asr_mode == "streaming"
        assert watcher.config.asr_model == "0.6B"
        assert watcher.config.asr_backend == "vllm"


def test_invalid_mode_backend_combo_exits():
    """Config with invalid mode+backend combo exits immediately."""
    with tempfile.TemporaryDirectory() as td:
        config_path = Path(td) / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("""
asr:
  mode: "streaming"
  backend: "transformers"
""")
        with pytest.raises(SystemExit):
            ConfigWatcher(path=config_path)


def test_reload_detects_mtime_change():
    """ConfigWatcher re-reads config when mtime changes."""
    with tempfile.TemporaryDirectory() as td:
        config_path = Path(td) / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("""
asr:
  mode: "offline"
  auto_sleep_time: 60
""")
        watcher = ConfigWatcher(path=config_path)
        assert watcher.config.asr_auto_sleep_time == 60

        # Update the file
        time.sleep(0.01)  # ensure mtime changes
        config_path.write_text("""
asr:
  mode: "offline"
  auto_sleep_time: 120
""")

        # Manually trigger reload via the internal method
        watcher._mtime = 0  # force reload
        watcher._reload()
        assert watcher.config.asr_auto_sleep_time == 120


def test_reload_failure_keeps_old_config():
    """When reload fails, the old config is retained."""
    with tempfile.TemporaryDirectory() as td:
        config_path = Path(td) / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("""
asr:
  mode: "offline"
  auto_sleep_time: 60
""")
        watcher = ConfigWatcher(path=config_path)
        old_sleep = watcher.config.asr_auto_sleep_time

        # Corrupt the file
        time.sleep(0.01)
        config_path.write_text("this is not valid: yaml: :::")

        watcher._mtime = 0
        watcher._reload()
        # Config should be unchanged
        assert watcher.config.asr_auto_sleep_time == old_sleep
```

- [ ] **Step 3: Run the new tests**

```bash
cd /Data2/Code/python/qwen3-asr-ime && python3 -m pytest tests/test_config_watcher.py -v
```
Expected: 5 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/qwen3_asr_ime/common/config.py tests/test_config_watcher.py
git commit -m "feat(config): add ConfigWatcher with mtime polling and default config creation"
```

---

### Task 3: Create BackendManager

**Files:**
- Create: `src/qwen3_asr_ime/daemon/backend_manager.py`
- Test: `tests/test_backend_manager.py`

**Interfaces:**
- Consumes: `IMEConfig` (asr_endpoint, asr_backend, asr_mode, model_path, asr_auto_sleep_time, asr_backend_wait_timeout, asr_api_key, asr_device)
- Produces: `BackendManager` class with `spawn(config)`, `wait_ready()`, `ensure_running()`, `touch_activity()`, `check_idle()`, `stop()`, `restart(config)`, `is_running` property

- [ ] **Step 0: Install aiohttp dependency**

```bash
pip install aiohttp>=3.9
```

- [ ] **Step 1: Create the module skeleton with docstring and imports**

Write `src/qwen3_asr_ime/daemon/backend_manager.py`:

```python
"""Backend process lifecycle manager.

``BackendManager`` spawns the ASR server as a child process, monitors its
health endpoint, enforces idle-timeout auto-sleep, and detects fatal errors
in the backend's stderr (OOM, port conflict, model not found).

All unexpected failures cause ``sys.exit(1)``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qwen3_asr_ime.common.config import IMEConfig

logger = logging.getLogger(__name__)

# Project root directory (parent of tools/)
_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
_SERVER_SCRIPT = _PROJECT_DIR / "tools" / "asr_server.py"
```

- [ ] **Step 2: Write BackendManager class with spawn and wait_ready**

```python
class BackendManager:
    """Manages the full lifecycle of the ASR backend child process.

    Usage::

        mgr = BackendManager()
        await mgr.spawn(config)
        await mgr.wait_ready()
        # ... backend is running ...
        mgr.touch_activity()
        await mgr.check_idle()   # stops backend if idle timeout reached
    """

    # Patterns in stderr that indicate a fatal startup error.
    _FATAL_PATTERNS: dict[str, str] = {
        "out of memory": "GPU 显存不足，无法加载模型",
        "cuda out of memory": "GPU 显存不足，无法加载模型",
        "address already in use": "端口已被占用",
        "cannot access": "模型路径不存在或无效",
        "no such file": "模型路径不存在或无效",
    }

    def __init__(self) -> None:
        self._process: asyncio.subprocess.Process | None = None
        self._last_activity: float = 0.0
        self._stderr_task: asyncio.Task[None] | None = None
        self._health_url: str = ""
        self._auto_sleep_time: float = 0.0
        self._wait_timeout: float = 120.0
        self._port: int = 8000

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def spawn(self, config: IMEConfig) -> None:
        """Launch the ASR server subprocess.

        Args:
            config: Current IMEConfig providing mode, backend, model, endpoint.

        Exits with ``sys.exit(1)`` if the process fails to start.
        """
        if self.is_running:
            logger.warning("Backend already running; stopping first")
            await self.stop()

        from urllib.parse import urlparse
        parsed = urlparse(config.asr_endpoint)
        self._port = parsed.port or 8000
        self._health_url = f"{config.asr_endpoint.rstrip('/')}/health"
        self._auto_sleep_time = float(config.asr_auto_sleep_time)
        self._wait_timeout = float(config.asr_backend_wait_timeout)

        env = os.environ.copy()
        env["QWEN3_ASR_MODE"] = config.asr_mode
        env["QWEN3_ASR_BACKEND"] = config.asr_backend
        env["QWEN3_ASR_MODEL"] = config.model_path
        env["QWEN3_ASR_DEVICE"] = config.asr_device
        env["QWEN3_ASR_PORT"] = str(self._port)

        cmd = [sys.executable, str(_SERVER_SCRIPT)]

        logger.info(
            "Starting backend: %s (mode=%s, backend=%s, model=%s, port=%d)",
            cmd, config.asr_mode, config.asr_backend, config.model_path, self._port,
        )
        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as exc:
            print(
                f"ERROR: 后端进程启动失败: {' '.join(cmd)}: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

        self._stderr_task = asyncio.create_task(self._monitor_stderr())

    async def wait_ready(self) -> None:
        """Poll /health until the backend responds 200 OK with ``status=="ok"``.

        While polling, monitors stderr for fatal patterns. If the backend
        process exits before becoming ready, exits immediately.

        Raises/Exits:
            ``sys.exit(1)`` on timeout or fatal error detected in stderr.
        """
        import aiohttp

        if self._process is None:
            print("ERROR: 后端进程未启动", file=sys.stderr)
            sys.exit(1)

        deadline = asyncio.get_running_loop().time() + self._wait_timeout
        last_log = asyncio.get_running_loop().time()

        while asyncio.get_running_loop().time() < deadline:
            # Check if process died
            if self._process.returncode is not None:
                # Give stderr monitor a moment to report the actual cause
                await asyncio.sleep(0.5)
                print(
                    f"ERROR: 后端进程异常退出 (code={self._process.returncode})",
                    file=sys.stderr,
                )
                sys.exit(1)

            # Try the health endpoint
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        self._health_url, timeout=aiohttp.ClientTimeout(total=2)
                    ) as resp:
                        if resp.status == 200:
                            body = await resp.json()
                            if body.get("status") == "ok":
                                logger.info(
                                    "Backend healthy (backend=%s) in %.1fs",
                                    body.get("backend", "unknown"),
                                    self._wait_timeout - (deadline - asyncio.get_running_loop().time()),
                                )
                                self._last_activity = asyncio.get_running_loop().time()
                                return
            except Exception:
                pass  # not ready yet

            if asyncio.get_running_loop().time() - last_log > 10:
                elapsed = self._wait_timeout - (deadline - asyncio.get_running_loop().time())
                logger.info("Waiting for backend... (%.0f/%.0fs)", elapsed, self._wait_timeout)
                last_log = asyncio.get_running_loop().time()

            await asyncio.sleep(1.0)

        print(
            f"ERROR: 后端启动超时 ({self._wait_timeout}s): {self._health_url} 无响应",
            file=sys.stderr,
        )
        sys.exit(1)
```

- [ ] **Step 3: Add stderr monitor, ensure_running, touch_activity, check_idle, stop, restart**

```python
    async def _monitor_stderr(self) -> None:
        """Continuously read backend stderr; detect fatal patterns and exit.

        Runs as a background task. Non-fatal output is logged at DEBUG level.
        """
        if self._process is None or self._process.stderr is None:
            return
        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                # Check for fatal patterns
                lower = text.lower()
                for pattern, message in self._FATAL_PATTERNS.items():
                    if pattern in lower:
                        print(f"ERROR: {message}: {text}", file=sys.stderr)
                        sys.exit(1)
                logger.debug("Backend stderr: %s", text)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("stderr monitor error: %s", exc)

    async def _read_remaining_stderr(self) -> str:
        """Drain any remaining stderr lines after process exits. Returns last few lines."""
        if self._process is None or self._process.stderr is None:
            return ""
        lines: list[str] = []
        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                lines.append(line.decode("utf-8", errors="replace").strip())
        except Exception:
            pass
        return "\n".join(lines[-10:])

    async def ensure_running(self) -> None:
        """Call before each recognition request.

        If the backend is already running, resets the idle timer and returns immediately.
        Otherwise spawns and waits for readiness.
        """
        if self.is_running:
            self._last_activity = asyncio.get_running_loop().time()
            return
        logger.info("Backend not running; re-spawning...")
        # Need config to re-spawn — stored from last spawn
        # This is handled by the daemon which holds the config reference
        # For standalone use, this is a no-op if the process died unexpectedly
        if not self.is_running:
            # The daemon should call spawn() with current config instead
            pass

    def touch_activity(self) -> None:
        """Mark the backend as recently used, resetting the idle timer.

        Called after each successful ASR recognition completes.
        """
        self._last_activity = asyncio.get_running_loop().time()

    async def check_idle(self) -> bool:
        """Check if the backend has been idle beyond auto_sleep_time.

        If idle timeout reached and backend is running, stops the backend.

        Returns:
            True if the backend was stopped due to idle timeout.
        """
        if not self.is_running:
            return False
        if self._auto_sleep_time <= 0:
            return False
        idle_duration = asyncio.get_running_loop().time() - self._last_activity
        if idle_duration >= self._auto_sleep_time:
            logger.info(
                "Backend idle for %.0fs (limit: %.0fs); stopping to save resources",
                idle_duration,
                self._auto_sleep_time,
            )
            await self.stop()
            return True
        return False

    async def stop(self) -> None:
        """Gracefully stop the backend process.

        Sends SIGTERM, waits up to 5 seconds, then SIGKILL if still alive.
        """
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass
            self._stderr_task = None

        if self._process is None:
            return

        if self._process.returncode is None:
            logger.info("Stopping backend (pid=%d)...", self._process.pid)
            try:
                self._process.send_signal(signal.SIGTERM)
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("Backend did not stop; sending SIGKILL")
                    self._process.kill()
                    await self._process.wait()
            except ProcessLookupError:
                pass  # already exited
            logger.info("Backend stopped")

        self._process = None

    async def restart(self, config: IMEConfig) -> None:
        """Stop the current backend and spawn a new one with updated config.

        Used when config-watcher detects backend-relevant changes.
        """
        logger.info("Restarting backend due to config change")
        await self.stop()
        await self.spawn(config)
        await self.wait_ready()
```

- [ ] **Step 4: Write tests**

Write `tests/test_backend_manager.py`:

```python
"""Tests for BackendManager."""
import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from qwen3_asr_ime.common.config import IMEConfig


class TestBackendManager:
    """Unit tests for BackendManager without a real backend process."""

    @pytest.fixture
    def config(self):
        return IMEConfig.defaults()

    @pytest.fixture
    def manager(self):
        from qwen3_asr_ime.daemon.backend_manager import BackendManager
        return BackendManager()

    def test_initial_state(self, manager):
        """Manager starts with no process and not running."""
        assert not manager.is_running

    @pytest.mark.asyncio
    async def test_touch_activity_resets_timer(self, manager):
        """touch_activity updates last_activity timestamp."""
        old = manager._last_activity
        await asyncio.sleep(0.01)
        manager.touch_activity()
        assert manager._last_activity > old

    @pytest.mark.asyncio
    async def test_check_idle_not_running(self, manager):
        """check_idle returns False when no process is running."""
        result = await manager.check_idle()
        assert result is False

    @pytest.mark.asyncio
    async def test_check_idle_zero_sleep_time(self, manager):
        """check_idle returns False when auto_sleep_time is 0."""
        manager._auto_sleep_time = 0
        # Mock is_running
        manager._process = MagicMock()
        manager._process.returncode = None
        result = await manager.check_idle()
        manager._process = None
        assert result is False
```

- [ ] **Step 5: Run tests**

```bash
cd /Data2/Code/python/qwen3-asr-ime && python3 -m pytest tests/test_backend_manager.py -v
```
Expected: 4 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/qwen3_asr_ime/daemon/backend_manager.py tests/test_backend_manager.py
git commit -m "feat(backend): add BackendManager for child process lifecycle"
```

---

### Task 4: Create ASRHttpClient

**Files:**
- Modify: `src/qwen3_asr_ime/daemon/asr_client.py`

**Interfaces:**
- Consumes: `ASRResult` (existing dataclass in same file)
- Produces: `ASRHttpClient` class with `transcribe(wav_bytes, language=None) -> ASRResult`

- [ ] **Step 1: Add ASRHttpClient class to asr_client.py**

Append after the `ASRStreamClient` class in `src/qwen3_asr_ime/daemon/asr_client.py`:

```python
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
```

- [ ] **Step 2: Write tests**

Write `tests/test_asr_http_client.py`:

```python
"""Tests for ASRHttpClient."""
import json
from unittest.mock import AsyncMock, patch

import pytest

from qwen3_asr_ime.daemon.asr_client import ASRHttpClient, ASRResult


class TestASRHttpClient:
    """Unit tests for ASRHttpClient."""

    @pytest.fixture
    def client(self):
        return ASRHttpClient("http://127.0.0.1:8000", api_key="test", timeout=5.0)

    def test_transcribe_url(self, client):
        """URL is constructed correctly from endpoint."""
        assert client._transcribe_url == "http://127.0.0.1:8000/v1/asr/transcribe"

    @pytest.mark.asyncio
    async def test_transcribe_success(self, client):
        """Successful transcription returns text with final=True."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"text": "你好世界", "language": "zh"}
        )
        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.transcribe(b"fake_wav_data")
            assert result.text == "你好世界"
            assert result.language == "zh"
            assert result.final is True
            assert result.error is None

    @pytest.mark.asyncio
    async def test_transcribe_server_error(self, client):
        """Server error returns result with error set."""
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.transcribe(b"fake_wav_data")
            assert result.error is not None
            assert "500" in result.error
            assert result.final is False
```

- [ ] **Step 3: Run tests**

```bash
cd /Data2/Code/python/qwen3-asr-ime && python3 -m pytest tests/test_asr_http_client.py -v
```
Expected: 3 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/qwen3_asr_ime/daemon/asr_client.py tests/test_asr_http_client.py
git commit -m "feat(client): add ASRHttpClient for non-streaming HTTP POST recognition"
```

---

### Task 5: Update asr_server.py with non-streaming endpoint

**Files:**
- Modify: `tools/asr_server.py`

**Interfaces:**
- Consumes: Environment variables `QWEN3_ASR_MODE`, `QWEN3_ASR_BACKEND`, `QWEN3_ASR_MODEL`, `QWEN3_ASR_DEVICE`, `QWEN3_ASR_PORT`
- Produces: `POST /v1/asr/transcribe` endpoint (JSON response `{"text", "language"}`); `GET /health` now includes `"mode"` field

- [ ] **Step 1: Refactor server to support both backends based on env vars**

Replace the current `_load_model()` and global variables with the following:

In `tools/asr_server.py`, replace:

```python
_model = None
_model_backend: str | None = None

MODEL_PATH = os.environ.get("QWEN3_ASR_MODEL", "/Data2/Models/Qwen3-ASR-0.6B")
```

With:

```python
_model = None
_model_backend: str | None = None
_server_mode: str = "offline"

MODEL_PATH = os.environ.get("QWEN3_ASR_MODEL", "/Data2/Models/Qwen3-ASR-1.7B")
SERVER_MODE = os.environ.get("QWEN3_ASR_MODE", "offline")
SERVER_BACKEND = os.environ.get("QWEN3_ASR_BACKEND", "transformers")
SERVER_DEVICE = os.environ.get("QWEN3_ASR_DEVICE", "auto")
SERVER_PORT = int(os.environ.get("QWEN3_ASR_PORT", "8000"))
```

Replace `_load_model()`:

```python
def _load_model() -> None:
    global _model, _model_backend, _server_mode
    if _model is not None:
        return

    _server_mode = SERVER_MODE
    backend = SERVER_BACKEND

    if _server_mode == "streaming" and backend != "vllm":
        raise RuntimeError(
            f"streaming mode requires vllm backend, got {backend}"
        )
    if _server_mode == "offline" and backend not in ("transformers", "vllm"):
        raise RuntimeError(
            f"offline mode requires transformers or vllm backend, got {backend}"
        )

    if backend == "vllm":
        _load_vllm()
    elif backend == "transformers":
        _load_transformers()
    else:
        raise RuntimeError(f"Unsupported backend: {backend}")


def _load_vllm() -> None:
    global _model, _model_backend
    from qwen_asr import Qwen3ASRModel

    gpu_mem = float(os.environ.get("QWEN3_ASR_GPU_MEM", "0.9"))
    max_tokens = int(os.environ.get("QWEN3_ASR_MAX_TOKENS", "256"))
    max_model_len = int(os.environ.get("QWEN3_ASR_MAX_MODEL_LEN", "4096"))
    enforce_eager = os.environ.get("QWEN3_ASR_ENFORCE_EAGER", "1").lower() in ("1", "true", "yes")
    enable_prefix_caching = os.environ.get("QWEN3_ASR_PREFIX_CACHING", "0").lower() in ("1", "true", "yes")
    logger.info(
        "Loading Qwen3-ASR (%s) with vLLM backend (gpu_mem=%.2f, max_model_len=%d) ...",
        MODEL_PATH,
        gpu_mem,
        max_model_len,
    )
    t0 = time.time()
    _model = Qwen3ASRModel.LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=gpu_mem,
        max_new_tokens=max_tokens,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
        enable_prefix_caching=enable_prefix_caching,
    )
    _model_backend = "vllm"
    logger.info("Loaded vLLM backend in %.1fs", time.time() - t0)


def _load_transformers() -> None:
    global _model, _model_backend
    from qwen_asr import Qwen3ASRModel

    device = SERVER_DEVICE

    logger.info(
        "Loading Qwen3-ASR (%s) with transformers backend (device=%s) ...",
        MODEL_PATH,
        device,
    )
    t0 = time.time()
    _model = Qwen3ASRModel.from_pretrained(
        MODEL_PATH,
        device=device if device != "auto" else None,
    )
    _model_backend = "transformers"
    logger.info("Loaded transformers backend in %.1fs", time.time() - t0)
```

- [ ] **Step 2: Add POST /v1/asr/transcribe endpoint**

Add after the existing `asr_stream` WebSocket endpoint and before `/health`:

```python
@app.post("/v1/asr/transcribe")
async def asr_transcribe(request: Request) -> dict[str, Any]:
    """Non-streaming (offline) ASR endpoint.

    Accepts raw WAV bytes in the request body. Returns a JSON object with
    ``text`` and ``language`` fields.

    Query params:
        language: Optional language hint (e.g. ``"zh"``, ``"en"``).
    """
    global _model
    _load_model()
    assert _model is not None
    assert _model_backend is not None

    language = request.query_params.get("language")

    try:
        raw_body = await request.body()
        if not raw_body:
            return {"text": "", "language": language, "error": "Empty request body"}

        # Decode WAV to float32 numpy array
        pcm = _decode_audio(raw_body)

        # transcribe accepts (np.ndarray, sample_rate) tuple
        results = _model.transcribe(
            (pcm, 16000), language=language
        )
        if results:
            first = results[0]
            return {"text": first.text, "language": first.language}
        else:
            return {"text": "", "language": language}
    except Exception as exc:
        logger.exception("Non-streaming recognition failed")
        return {"text": "", "language": language, "error": str(exc)}
```

The `Request` import needs to be added at the top:
```python
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
```

- [ ] **Step 3: Update /health endpoint to include mode**

```python
@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "backend": _model_backend, "mode": _server_mode}
```

- [ ] **Step 4: Update main to use env vars for port**

```python
if __name__ == "__main__":
    _load_model()
    uvicorn.run(app, host="127.0.0.1", port=SERVER_PORT)
```

- [ ] **Step 5: Run a quick syntax check**

```bash
cd /Data2/Code/python/qwen3-asr-ime && python3 -c "import py_compile; py_compile.compile('tools/asr_server.py', doraise=True); print('OK')"
```
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add tools/asr_server.py
git commit -m "feat(server): add POST /v1/asr/transcribe with transformers backend support"
```

---

### Task 6: Integrate into VoiceInputDaemon

**Files:**
- Modify: `src/qwen3_asr_ime/daemon/service.py`

**Interfaces:**
- Consumes: `ConfigWatcher` (Task 2), `BackendManager` (Task 3), `ASRHttpClient` (Task 4), `IMEConfig`
- Produces: Updated `VoiceInputDaemon` with mode routing, backend lifecycle, config watching, idle loop

- [ ] **Step 1: Add imports for new components**

In `src/qwen3_asr_ime/daemon/service.py`, update imports:

```python
from qwen3_asr_ime.common.config import ConfigWatcher, IMEConfig
from qwen3_asr_ime.common.logger import get_logger
from qwen3_asr_ime.common.protocol import HotkeyCommand, RecognizedText, StateUpdate, parse_message
from qwen3_asr_ime.daemon.asr_client import ASRHttpClient, ASRResult, ASRStreamClient
from qwen3_asr_ime.daemon.backend_manager import BackendManager
from qwen3_asr_ime.daemon.hotkey import HotkeyEvent, create_hotkey_listener
from qwen3_asr_ime.daemon.recorder import AudioConfig, Recorder
```

Remove the direct `IMEConfig` import (replaced by `ConfigWatcher` usage):
```python
# Remove: from qwen3_asr_ime.common.config import IMEConfig
```

- [ ] **Step 2: Update VoiceInputDaemon.__init__ to accept ConfigWatcher and create BackendManager**

```python
class VoiceInputDaemon:
    def __init__(self, config_watcher: ConfigWatcher):
        self._config_watcher = config_watcher
        self._config = config_watcher.config
        self._backend_mgr = BackendManager()

        self.recorder = Recorder(
            AudioConfig(
                sample_rate=self._config.audio_sample_rate,
                channels=self._config.audio_channels,
                chunk_ms=self._config.audio_chunk_ms,
            )
        )
        self.hotkey = create_hotkey_listener(
            self._config.hotkey_device,
            self._config.hotkey_key,
            self._on_hotkey,
        )
        self._clients: set[asyncio.StreamWriter] = set()
        self._lock = threading.Lock()
        self._state: str = "idle"
        self._server: asyncio.AbstractServer | None = None

        # Streaming ASR state
        self._stream_client: ASRStreamClient | None = None
        self._streaming_task: asyncio.Task[None] | None = None
        self._sender_task: asyncio.Task[None] | None = None
        self._serve_task: asyncio.Task[None] | None = None
        self._pending_chunks: collections.deque[bytes] = collections.deque()
        self._max_pending_chunks = 200
        self._streaming_error: str | None = None
        self._current_text: str = ""
        self._typed_text: str = ""
        self._stream_final_result: ASRResult | None = None
        self._stream_final_event: asyncio.Event | None = None

        # Offline ASR state
        self._offline_recording: bytes = b""

        # Background tasks
        self._config_watch_task: asyncio.Task[None] | None = None
        self._idle_check_task: asyncio.Task[None] | None = None

        # Error counter for consecutive failures
        self._consecutive_errors: int = 0
        self._max_consecutive_errors: int = 5
```

- [ ] **Step 3: Update start() to spawn backend and launch background tasks**

```python
    async def start(self) -> None:
        """Create IPC socket, start backend, launch hotkey, begin config/idle watchers."""
        socket_path = Path(self._config.ipc_socket_path)
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        if socket_path.exists():
            socket_path.unlink()
        self._loop = asyncio.get_running_loop()

        # Spawn backend and wait for health
        await self._backend_mgr.spawn(self._config)
        await self._backend_mgr.wait_ready()

        # Start IPC server
        self._server = await asyncio.start_unix_server(
            self._on_client_connected,
            path=str(socket_path),
        )
        os.chmod(socket_path, 0o600)

        # Start hotkey listener
        self.hotkey.start()

        # Launch background watchers
        self._config_watch_task = asyncio.create_task(
            self._config_watcher.watch_loop(self._on_config_change)
        )
        self._idle_check_task = asyncio.create_task(self._idle_check_loop())

        logger.info(
            "Daemon started (mode=%s, backend=%s, model=%s)",
            self._config.asr_mode,
            self._config.asr_backend,
            self._config.asr_model,
        )
```

- [ ] **Step 4: Update _on_hotkey dispatch and _handle_hotkey to route between offline and streaming**

First, change the dispatch in `_on_hotkey` from `call_soon_threadsafe` (sync-only) to `run_coroutine_threadsafe` (supports async):

```python
    def _on_hotkey(self, event: HotkeyEvent) -> None:
        """Handle a hotkey event (may arrive from a non-asyncio thread)."""
        asyncio.run_coroutine_threadsafe(self._handle_hotkey(event), self._loop)
```

Then replace the existing `_handle_hotkey` method:

```python
    async def _handle_hotkey(self, event: HotkeyEvent) -> None:
        """State machine driven by press/release hotkey events.

        Routes between offline (non-streaming) and streaming modes based on config.
        """
        try:
            if event.action == "press" and self._state == "idle":
                # Wake up backend if it was sleeping
                if not self._backend_mgr.is_running:
                    await self._backend_mgr.spawn(self._config)
                    await self._backend_mgr.wait_ready()
                else:
                    self._backend_mgr.touch_activity()

                self._state = "recording"
                self._current_text = ""
                self._typed_text = ""
                self._streaming_error = None
                self._stream_final_result = None
                self._offline_recording = b""
                self._pending_chunks.clear()

                if self._config.asr_mode == "streaming":
                    # Existing streaming flow
                    self._stream_final_event = asyncio.Event()
                    self.recorder.start(chunk_callback=self._on_audio_chunk)
                    self._stream_client = ASRStreamClient(
                        self._config.asr_endpoint,
                        api_key=self._config.asr_api_key,
                        timeout=self._config.asr_timeout,
                    )
                    self._streaming_task = asyncio.create_task(self._run_stream())
                    self._broadcast_state("recording", "🔴 录音中 (流式)")
                    logger.info("⬇ Ctrl 按下 → 开始录音并建立流式 ASR 连接")
                else:
                    # Offline mode: just record, no WebSocket
                    self.recorder.start()  # no chunk_callback needed
                    self._broadcast_state("recording", "🔴 录音中 (离线)")
                    logger.info("⬇ Ctrl 按下 → 开始录音 (离线模式)")

            elif event.action == "release" and self._state == "recording":
                self._state = "recognizing"
                self._broadcast_state("recognizing", "🔄 识别中...")
                audio_bytes = self.recorder.stop()
                dur = len(audio_bytes) / 32000
                logger.info(
                    "⬆ Ctrl 松开 → 停止录音 (%.1f 秒, %d KB)", dur, len(audio_bytes) // 1024
                )
                if self._config.asr_mode == "offline":
                    self._offline_recording = audio_bytes
                    asyncio.create_task(self._run_offline_recognition())
                else:
                    asyncio.create_task(self._finish_stream())

            elif event.action == "interrupt" and self._state == "recording":
                logger.info("⛔ 检测到组合键，中断语音输入")
                self._state = "idle"
                self._broadcast_state("idle", "🎤 就绪")
                self.recorder.stop()
                self._pending_chunks.clear()
                if self._config.asr_mode == "streaming":
                    asyncio.create_task(self._cleanup_streaming())
        except Exception:
            logger.exception("❌ 热键处理错误")
            self._consecutive_errors += 1
            if self._consecutive_errors >= self._max_consecutive_errors:
                import sys
                logger.critical(
                    "连续 %d 次错误，退出程序", self._consecutive_errors
                )
                sys.exit(1)
```

- [ ] **Step 5: Add offline recognition flow and support methods**

```python
    async def _run_offline_recognition(self) -> None:
        """Send the complete recording to the HTTP ASR endpoint and handle result."""
        if not self._offline_recording:
            self._state = "idle"
            self._broadcast_state("idle", "🎤 就绪")
            return

        try:
            client = ASRHttpClient(
                self._config.asr_endpoint,
                api_key=self._config.asr_api_key,
                timeout=self._config.asr_timeout,
            )
            result = await client.transcribe(self._offline_recording)

            if result.error:
                self._streaming_error = result.error
                self._consecutive_errors += 1
                logger.error("❌ 离线 ASR 识别失败: %s", result.error)
                self._state = "idle"
                self._broadcast_state("error", "⚠️ ASR 错误")
                self._broadcast_recognized("", error=result.error)
                if self._consecutive_errors >= self._max_consecutive_errors:
                    import sys
                    logger.critical(
                        "连续 %d 次识别失败，退出程序", self._consecutive_errors
                    )
                    sys.exit(1)
            else:
                self._consecutive_errors = 0
                self._state = "idle"
                self._broadcast_state("idle", "🎤 就绪")
                self._broadcast_recognized(result.text)
                logger.info('✅ ASR 识别完成: "%s"', result.text)
                if not self._clients:
                    self._type_text_final(result.text)
        except Exception as exc:
            logger.exception("❌ 离线 ASR 未预期错误")
            self._consecutive_errors += 1
            self._state = "idle"
            self._broadcast_state("error", "⚠️ ASR 错误")
            if self._consecutive_errors >= self._max_consecutive_errors:
                import sys
                sys.exit(1)
        finally:
            self._backend_mgr.touch_activity()

    def _type_text_final(self, text: str) -> None:
        """Type the final recognized text into the active X11 window.

        Unlike streaming incremental typing, this types the entire text at once.
        """
        if self._clients:
            return
        if not text:
            return
        self._typed_text = text
        self._loop.run_in_executor(None, _type_text_x11, text)
```

- [ ] **Step 6: Add config change handler and idle check loop**

```python
    def _on_config_change(self, new_config: IMEConfig) -> None:
        """Handle config file changes detected by ConfigWatcher.

        Compares old and new config. If backend-relevant fields changed,
        schedules a backend restart. Daemon-local fields are applied
        immediately.
        """
        old = self._config
        self._config = new_config

        # Check if backend-relevant config has changed
        backend_keys = (
            "asr_mode", "asr_model", "asr_backend", "asr_device",
            "asr_endpoint", "asr_auto_sleep_time", "asr_backend_wait_timeout",
        )
        needs_restart = any(
            getattr(old, k) != getattr(new_config, k) for k in backend_keys
        )

        if needs_restart:
            logger.info("Backend-relevant config changed; scheduling restart")
            asyncio.create_task(self._backend_mgr.restart(new_config))

    async def _idle_check_loop(self) -> None:
        """Periodically check if the backend should be put to sleep."""
        while True:
            await asyncio.sleep(5.0)  # check every 5s
            try:
                await self._backend_mgr.check_idle()
            except Exception as exc:
                logger.warning("Idle check error: %s", exc)
```

- [ ] **Step 7: Update main() entry point**

```python
async def main() -> None:
    """Entry point: init ConfigWatcher, create daemon, run forever."""
    watcher = ConfigWatcher()
    daemon = VoiceInputDaemon(watcher)
    await daemon.start()
    await daemon.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 8: Run full test suite**

```bash
cd /Data2/Code/python/qwen3-asr-ime && python3 -m pytest tests/ -v -m "not e2e"
```
Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add src/qwen3_asr_ime/daemon/service.py
git commit -m "feat(daemon): integrate ConfigWatcher, BackendManager, offline ASR flow"
```

---

### Task 7: Update dependencies, systemd, and install script

**Files:**
- Modify: `pyproject.toml`
- Modify: `systemd/qwen3-asr-server.service`
- Modify: `systemd/qwen3-asr-ime.service`
- Modify: `bin/install.sh`

**Interfaces:**
- Consumes: All previous tasks
- Produces: Updated deployment configuration

- [ ] **Step 1: Add httpx dependency to pyproject.toml**

In `pyproject.toml`, add to `dependencies`:

```toml
dependencies = [
    "sounddevice>=0.5",
    "pynput>=1.7",
    "pyyaml>=6.0",
    "websockets>=12.0",
    "aiohttp>=3.9",
]
```

Note: `httpx` removed in favor of `aiohttp` (already implicitly available; used in BackendManager and ASRHttpClient). Actually, the plan already uses aiohttp throughout — update the dependency list to explicitly include it.

- [ ] **Step 2: Add transformers extra to pyproject.toml**

In `[project.optional-dependencies]`, add:

```toml
transformers = ["transformers>=4.45", "accelerate>=0.34"]
```

- [ ] **Step 3: Update systemd server service template**

In `systemd/qwen3-asr-server.service`:

```ini
[Unit]
Description=Qwen3-ASR Model Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart={{PYTHON}} {{PROJECT_DIR}}/tools/asr_server.py
Restart=on-failure
RestartSec=5
Environment="PYTHONUNBUFFERED=1"
Environment="QWEN3_ASR_MODE=offline"
Environment="QWEN3_ASR_BACKEND=transformers"
Environment="QWEN3_ASR_MODEL=/Data2/Models/Qwen3-ASR-1.7B"
Environment="QWEN3_ASR_DEVICE=auto"
Environment="QWEN3_ASR_PORT=8000"

[Install]
WantedBy=default.target
```

- [ ] **Step 4: Update systemd daemon service template**

In `systemd/qwen3-asr-ime.service` — remove the `Requires=` dependency since the daemon now manages the backend itself:

```ini
[Unit]
Description=Qwen3-ASR Voice Input Daemon
After=graphical-session.target

[Service]
Type=simple
ExecStart={{PYTHON}} -m qwen3_asr_ime.daemon.service
Restart=on-failure
Environment="PYTHONUNBUFFERED=1"

[Install]
WantedBy=default.target
```

- [ ] **Step 5: Update install.sh with new default config template**

In `bin/install.sh`, update the default config YAML template:

```bash
if [[ ! -f "${CONFIG_DIR}/config.yaml" ]]; then
cat > "${CONFIG_DIR}/config.yaml" <<'EOF'
hotkey:
  device: "pynput"
  key: "CTRL"
audio:
  sample_rate: 16000
  channels: 1
  format: "int16"
  chunk_ms: 20
asr:
  endpoint: "http://127.0.0.1:8000"
  mode: "offline"
  model: "1.7B"
  backend: "transformers"
  device: "auto"
  quantization: "auto"
  api_key: "dummy"
  timeout: 30.0
  auto_sleep_time: 300
  backend_wait_timeout: 120
ipc:
  socket_path: "/run/user/${UID}/qwen3-asr-ime.sock"
logging:
  level: "INFO"
EOF
fi
```

Also update the pip install line to include the transformers extra:

```bash
python3 -m pip install -e "${PROJECT_DIR}[vllm,transformers]"
```

- [ ] **Step 6: Run syntax checks**

```bash
cd /Data2/Code/python/qwen3-asr-ime && python3 -m pytest tests/ -v -m "not e2e"
```
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml systemd/qwen3-asr-server.service systemd/qwen3-asr-ime.service bin/install.sh
git commit -m "chore: update deps (aiohttp), systemd templates, and install script for offline mode"
```

---

## Verification Checklist

Before marking implementation complete, verify:

- [ ] `python3 -m pytest tests/ -v -m "not e2e"` — all unit tests pass
- [ ] Config file missing → default created with offline/1.7B/transformers defaults
- [ ] Invalid mode+backend combo → `sys.exit(1)` with clear message
- [ ] Config file edited during runtime → detected within 5s, backend restarted
- [ ] Backend idle timeout → backend process stops
- [ ] Hotkey press during sleep → backend re-spawned and ready before recording starts
- [ ] Backend OOM during startup → `sys.exit(1)` with GPU memory message
- [ ] `python3 tools/asr_server.py` starts and `/health` returns `{"status":"ok","backend":"transformers","mode":"offline"}`
