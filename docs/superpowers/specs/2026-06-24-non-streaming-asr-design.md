# Non-Streaming ASR Support Design

**Date**: 2026-06-24
**Status**: Approved

## Overview

当前项目仅支持基于 vLLM 的流式 WebSocket ASR。本设计新增：
- 非流式模式（默认），基于 transformers 后端推理，使用 Qwen3-ASR 1.7B
- 配置文件热监听（5s 轮询 mtime）
- 后端自动休眠（空闲超时关闭后端进程释放 GPU/CPU）
- Fail-loudly 错误处理策略

---

## 1. Configuration File

扩展 `~/.config/qwen3-asr-ime/config.yaml`：

```yaml
hotkey:
  device: "pynput"
  key: "CTRL"
audio:
  sample_rate: 16000
  channels: 1
  format: "int16"
  chunk_ms: 20
ipc:
  socket_path: "/run/user/${UID}/qwen3-asr-ime.sock"
logging:
  level: "INFO"
asr:
  endpoint: "http://127.0.0.1:8000"
  api_key: "dummy"

  # --- 新增字段 ---
  mode: "offline"            # "offline" | "streaming"
  model: "1.7B"              # "0.6B" | "1.7B"
  backend: "transformers"    # "transformers" | "vllm"
  auto_sleep_time: 300       # 空闲秒数后关闭后端 (0 = 永不休眠)
  backend_wait_timeout: 120  # 等待后端启动就绪的超时秒数
  # --- 保留字段 ---
  device: "auto"             # "auto" | "cuda" | "cpu"
  quantization: "auto"       # "auto" | "int8" | "fp16"
  timeout: 30.0              # 单次识别超时秒数
```

**约束**：
- `mode="offline"` 必须配 `backend="transformers"`（或 vllm offline）
- `mode="streaming"` 必须配 `backend="vllm"`
- 配置文件缺失时自动创建含以上默认值的文件，创建失败则 `sys.exit(1)`
- 监听周期硬编码为 5 秒（不暴露到配置文件）

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    VoiceInputDaemon                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐             │
│  │ Hotkey   │ │ Recorder │ │ ASR Client   │             │
│  │ Listener │ │          │ │ (router)     │             │
│  └──────────┘ └──────────┘ └──┬───┬───────┘             │
│                               │   │                      │
│                    ┌──────────┘   └──────────┐           │
│                    ▼                         ▼           │
│         ┌─────────────────┐    ┌─────────────────────┐  │
│         │ASRHttpClient    │    │ASRStreamClient      │  │
│         │(POST /transcribe)│   │(WebSocket /stream)  │  │
│         └────────┬────────┘    └──────────┬──────────┘  │
│                  │                        │              │
│  ┌───────────────┼────────────────────────┼──────────┐  │
│  │ ConfigWatcher │         │              │          │  │
│  │ (5s poll)     │         │              │          │  │
│  └───────────────┘         │              │          │  │
│                             │              │          │  │
│  ┌──────────────────────────┼──────────────┼───────┐  │  │
│  │ BackendManager           │              │       │  │  │
│  │ (spawn/monitor/kill)     │              │       │  │  │
│  └──────────────────────────┼──────────────┼───────┘  │  │
└─────────────────────────────┼──────────────┼──────────┘
                              │              │
                              ▼              ▼
┌─────────────────────────────────────────────────────────┐
│                   asr_server.py                           │
│  ┌──────────────────┐  ┌──────────────────────────┐      │
│  │ /v1/asr/transcribe│  │ /v1/asr/stream (WS)     │      │
│  │ (HTTP POST)       │  │                          │      │
│  │ transformers 后端  │  │ vLLM 后端                │      │
│  └──────────────────┘  └──────────────────────────┘      │
│  GET /health ← 启动就绪检查                               │
└─────────────────────────────────────────────────────────┘
```

### 新增模块

| 模块 | 文件路径 | 职责 |
|------|---------|------|
| ConfigWatcher | `common/config.py` (扩展) | 5s 轮询 mtime, 解析失败保留旧配置, 创建默认配置 |
| BackendManager | `daemon/backend_manager.py` | spawn/kill 子进程, 轮询 `/health` 等待就绪, 追踪空闲计时, 异常检测 |
| ASRHttpClient | `daemon/asr_client.py` (扩展) | 异步 HTTP POST 完整音频, 返回识别结果 |
| NonStreamingASR endpoint | `tools/asr_server.py` (扩展) | FastAPI `POST /v1/asr/transcribe`, transformers 模型加载与推理 |

### 修改模块

| 模块 | 变更 |
|------|------|
| IMEConfig | 新增 5 个字段 (`mode`, `model`, `backend`, `auto_sleep_time`, `backend_wait_timeout`)，新增 `_validate()` 交叉校验 |
| VoiceInputDaemon | 集成 BackendManager 和 ConfigWatcher, 根据 mode 路由到不同 client |
| asr_server.py | 新增非流式端点, 根据环境变量选择后端类型, 命令行参数支持 |

---

## 3. Key Lifecycle

```
Daemon 启动
  → ConfigWatcher 加载配置 (不存在则创建默认, 失败则 exit)
  → BackendManager.spawn() (根据配置启动后端子进程)
  → BackendManager.wait_ready() (轮询 /health, 超时 exit)
  → Daemon 进入运行循环
      → 用户按热键 → 重置空闲计时器
      → 用户松热键 → ASR 识别 → 文本输入 → 开始空闲倒计时
      → ConfigWatcher 检测配置变更 → 标记 dirty
      → 后端相关变更 + 当前空闲 → 重启后端
      → 空闲超时 → BackendManager.stop() → 等待下次热键唤醒
```

---

## 4. Component Details

### 4.1 ConfigWatcher (扩展 `common/config.py`)

```python
class ConfigWatcher:
    """定时监听配置文件变更，线程安全。"""
    _path: Path
    _mtime: float          # 上次读取时的修改时间
    _config: IMEConfig     # 当前配置（运行时可能更新）
    _interval: float = 5.0 # 轮询间隔（硬编码）

    def __init__(self, path: Path | None = None):
        # 1. 确定配置文件路径
        # 2. 若文件不存在 → _create_default() → 失败则 sys.exit(1)
        # 3. 首次加载: _reload() → 失败则 sys.exit(1)
        # 4. 记录 self._mtime = path.stat().st_mtime

    def _create_default(self) -> None:
        # 创建目录 → 创建默认YAML → 失败则 sys.exit(1)

    def _reload(self) -> None:
        # 读取+解析YAML → 失败则:
        #   首次加载: sys.exit(1)
        #   运行时重载: logger.warning + 保留旧配置（不退出）

    async def watch_loop(self, on_change: Callable[[IMEConfig], None]) -> NoReturn:
        """异步循环，每5秒检查mtime，变更时调用回调。"""
        while True:
            await asyncio.sleep(self._interval)
            try:
                stat = self._path.stat()
                if stat.st_mtime != self._mtime:
                    old = self._config
                    self._reload()
                    self._mtime = stat.st_mtime
                    if self._config != old:
                        on_change(self._config)
            except FileNotFoundError:
                # 文件被删除 → 重新创建默认
                self._create_default()
```

### 4.2 BackendManager (`daemon/backend_manager.py`)

```python
class BackendManager:
    """管理后端子进程的完整生命周期。"""
    _process: asyncio.subprocess.Process | None
    _last_activity: float       # 最后活跃时间戳
    _auto_sleep_time: float
    _wait_timeout: float
    _health_url: str

    async def spawn(self, config: IMEConfig) -> None:
        """启动后端子进程。失败 sys.exit(1)。"""

    async def wait_ready(self) -> None:
        """轮询 /health 直到 200 OK+status=="ok" 或超时 exit。
           同时监控子进程 stderr，检测 OOM/端口占用等错误。"""

    async def ensure_running(self) -> None:
        """有请求时调用：若进程在运行则重置空闲计时；否则 spawn+wait_ready。"""

    def touch_activity(self) -> None:
        """每次 ASR 识别完成后调用，更新 _last_activity。"""

    async def check_idle(self) -> None:
        """若 _auto_sleep_time > 0 且 (now - _last_activity) > timeout → stop()。"""

    async def stop(self) -> None:
        """SIGTERM → wait(timeout=5s) → SIGKILL。"""

    async def restart(self, config: IMEConfig) -> None:
        """stop() + spawn(config) + wait_ready()。用于配置变更时。"""
```

### 4.3 ASRHttpClient (扩展 `daemon/asr_client.py`)

```python
class ASRHttpClient:
    """异步 HTTP 客户端，发送完整 WAV 并等待识别结果。"""

    def __init__(self, endpoint: str, api_key: str, timeout: float):
        self._transcribe_url = f"{endpoint.rstrip('/')}/v1/asr/transcribe"
        ...

    async def transcribe(self, wav_bytes: bytes, language: str | None = None) -> ASRResult:
        """POST /v1/asr/transcribe，发送 WAV body，返回 ASRResult。
           失败时 ASRResult.error 非 None，不抛异常。"""
        # 使用 httpx 或 aiohttp 发送 POST，body 为 WAV bytes
        # 超时由 self.timeout 控制
        # 返回: ASRResult(text=..., language=..., final=True)
```

### 4.4 asr_server.py 扩展

```python
# 新增非流式端点
@app.post("/v1/asr/transcribe")
async def asr_transcribe(request: Request):
    """接收 WAV 上传，用 transformers 模型做一次完整推理。"""

# 启动逻辑改为根据环境变量或命令行参数选择后端
SERVER_MODE = os.environ.get("QWEN3_ASR_MODE", "offline")
SERVER_BACKEND = os.environ.get("QWEN3_ASR_BACKEND", "transformers")
SERVER_MODEL_SIZE = os.environ.get("QWEN3_ASR_MODEL_SIZE", "1.7B")

def _load_model():
    if SERVER_BACKEND == "vllm":
        model = Qwen3ASRModel.LLM(...)
    elif SERVER_BACKEND == "transformers":
        model = Qwen3ASRModel.from_pretrained(model_path, device=...)
```

### 4.5 VoiceInputDaemon 集成变更

```python
class VoiceInputDaemon:
    _config_watcher: ConfigWatcher
    _backend_mgr: BackendManager
    _idle_task: asyncio.Task | None  # 空闲检查循环

    async def start(self):
        # 1. ConfigWatcher 初始化 (不存在则创建默认)
        # 2. BackendManager.spawn() + wait_ready()
        # 3. 启动 IPC server + hotkey listener
        # 4. asyncio.create_task(config_watcher.watch_loop(self._on_config_change))
        # 5. asyncio.create_task(self._idle_check_loop())

    def _handle_hotkey(self, event):
        if event.action == "press" and self._state == "idle":
            await self._backend_mgr.ensure_running()
            if self._config.asr_mode == "offline":
                # 开始录音 (不建立WS连接)
            else:
                # 现有流式逻辑
        elif event.action == "release" and self._state == "recording":
            if self._config.asr_mode == "offline":
                await self._run_offline_recognition()
            else:
                # 现有流式 finish 逻辑

    async def _run_offline_recognition(self):
        """停止录音 → POST /v1/asr/transcribe → 拿到结果 → 输入文本 → touch_activity"""

    def _on_config_change(self, new_config: IMEConfig):
        """配置变更回调。比较新旧配置，若后端相关项变更且当前空闲，重启后端。"""

    async def _idle_check_loop(self):
        """周期检查空闲时间，触发后端休眠。"""
```

---

## 5. Error Handling (Fail-Loudly)

### Hard-Exit Scenarios (`sys.exit(1)`)

| Scenario | Detection | Exit Message |
|----------|-----------|-------------|
| 配置文件不存在且创建失败 | `OSError` / `PermissionError` | `"无法创建默认配置文件: {path}: {error}"` |
| 首次启动配置解析失败 | `yaml.YAMLError` / `ValueError` | `"配置文件解析失败: {path}: {error}"` |
| 后端进程启动失败 | `subprocess.Popen` 抛异常 | `"后端进程启动失败: {cmd}: {error}"` |
| 后端启动超时 | `/health` 轮询超时 | `"后端启动超时 ({timeout}s): {url}/health 无响应"` |
| 后端异常状态 | `/health` 返回非 200 | `"后端异常状态: {response}"` |
| 后端意外崩溃 | 子进程 `poll()` 返回非零 | `"后端进程异常退出 (code={code})"` |
| GPU 显存不足 | stderr 检测 OOM | `"GPU 显存不足，无法加载模型: {model}"` |
| 端口被占用 | stderr 检测 address in use | `"端口 {port} 已被占用"` |
| 模型路径不存在 | 加载异常 | `"模型路径不存在或无效: {model_path}"` |
| mode+backend 组合无效 | 配置交叉校验 | `"不支持的组合: mode={mode}, backend={backend}"` |

### Runtime Error Handling (non-exit)

- 运行时配置解析失败：`logger.warning` + 保留旧配置
- 识别请求失败 (网络/超时)：`ASRResult.error` 非 None，daemon 广播错误但不退出
- 连续错误阈值：连续 N 次识别失败 → `sys.exit(1)`

### General Principle

```
任何未被以上列表覆盖的异常 → 记录完整 traceback → sys.exit(1)
程序永远不静默运行。
```

---

## 6. Non-Streaming Recognition Flow

```
用户按下热键 (Ctrl)
  → ensure_running() (如后端已休眠，重新 spawn + wait_ready)
  → 开始录音 (Recorder.start)
  → 状态 → "recording"
  → (无 WebSocket 连接, 无 partial 结果, 无实时打字)

用户松开热键
  → 停止录音 (Recorder.stop → WAV bytes)
  → 状态 → "recognizing"
  → ASRHttpClient.transcribe(wav_bytes)
      → POST /v1/asr/transcribe
      → 后端 transformers 模型推理
      → 返回 {"text": "识别结果", "language": "zh"}
  → 状态 → "idle"
  → 输入文本到 X11
  → BackendManager.touch_activity() → 开始空闲倒计时
  → 广播结果到 IPC 客户端
```

与流式模式的关键区别：
- 录音期间无网络通信
- 无 partial 结果 (松开热键前看不到任何文字)
- 松键后需等待完整推理 (延迟更高)
- 但模型加载更快 (transformers 无 vLLM 预热开销)

---

## 7. File Change Summary

| File | Action |
|------|--------|
| `src/qwen3_asr_ime/common/config.py` | Extend: ConfigWatcher, new fields, validation |
| `src/qwen3_asr_ime/daemon/backend_manager.py` | **New**: BackendManager |
| `src/qwen3_asr_ime/daemon/asr_client.py` | Extend: ASRHttpClient |
| `src/qwen3_asr_ime/daemon/service.py` | Modify: integrate BackendManager + ConfigWatcher, offline flow |
| `tools/asr_server.py` | Modify: add POST /v1/asr/transcribe, transformers loading, CLI args |
| `systemd/qwen3-asr-server.service` | Modify: template with mode/env vars |
| `systemd/qwen3-asr-ime.service` | Modify: remove Requires= dependency (daemon manages backend) |
| `bin/install.sh` | Modify: update config template, env vars |
| `pyproject.toml` | Modify: add `httpx` dependency, optional `transformers` extra |
