# Qwen3-ASR Linux 语音输入法设计文档

> 状态：待实现  
> 创建日期：2026-06-20

## 1. 目标

实现一个基于 Qwen3-ASR 的 Linux 语音输入法，用户按住全局热键即可录音，松开后将识别出的中文/英文文本自动输入到当前聚焦的输入框。

关键约束：
- 前端采用 IBus 输入法框架。
- Qwen3-ASR 以本地 API 服务形式运行。
- 根据本机 8GB 显存自动选择 GPU 或 CPU 推理。
- 支持中文+英文自动识别。

## 2. 架构概述

采用三层架构，职责清晰：

```
┌─────────────────────────────────────────────────────────────┐
│  IBus Engine (ibus/engine.py)                               │
│  - 与 IBus 框架交互                                          │
│  - 接收守护进程识别结果并 commit 到当前输入框                    │
│  - 显示录音/识别状态                                          │
└──────────────┬──────────────────────────────────────────────┘
               │ Unix Socket / D-Bus
┌──────────────▼──────────────────────────────────────────────┐
│  Voice Daemon (daemon/service.py)                           │
│  - 监听全局热键（evdev/pynput）                               │
│  - 录音（PyAudio / SoundDevice）                              │
│  - 调用本地 ASR HTTP API                                     │
│  - 管理录音状态，向 IBus Engine 发送识别结果                    │
└──────────────┬──────────────────────────────────────────────┘
               │ HTTP / OpenAI-compatible API
┌──────────────▼──────────────────────────────────────────────┐
│  ASR Service (vLLM / SGLang / llama.cpp server)             │
│  - 加载 Qwen3-ASR 模型                                       │
│  - 提供 /v1/audio/transcriptions 接口                        │
└─────────────────────────────────────────────────────────────┘
```

设计理由：
- IBus 引擎保持简单，避免在输入法进程里做重计算或访问硬件。
- 守护进程可独立运行、测试，未来可复用到 Fcitx5 或其他前端。
- ASR 服务独立，便于更换推理后端或部署到远程。

## 3. 组件说明

| 文件 | 职责 |
|------|------|
| `ibus/engine.py` | IBus 引擎主类，处理 focus 事件、commit 文本、状态显示。 |
| `ibus/factory.py` | IBus 工厂，创建引擎实例，注册到 IBus。 |
| `ibus/main.py` | IBus 组件入口。 |
| `daemon/service.py` | 守护进程主循环，协调热键、录音、ASR、IPC。 |
| `daemon/recorder.py` | 音频录制模块，输出 16kHz 单声道 PCM。 |
| `daemon/hotkey.py` | 全局热键监听（基于 evdev 或 pynput）。 |
| `daemon/asr_client.py` | ASR HTTP 客户端，封装 `/v1/audio/transcriptions`。 |
| `daemon/process_manager.py` | 检测/启动/监控 ASR 服务子进程。 |
| `common/protocol.py` | IPC 消息协议（TypedDict / dataclass + JSON）。 |
| `common/config.py` | 配置加载（YAML/JSON），含热键、模型、设备、阈值等。 |
| `common/logger.py` | 统一日志。 |
| `bin/install.sh` | 安装脚本：复制文件、注册 IBus 组件、安装 systemd user service。 |
| `systemd/qwen3-asr-ime.service` | systemd user service 单元文件。 |

## 4. 数据流

1. 用户按住全局热键。
2. `hotkey.py` 通知 `service.py` 开始录音。
3. `recorder.py` 以 16kHz 单声道录制 PCM，存入内存 ring buffer。
4. 用户松开热键。
5. `service.py` 停止录音，将音频编码为 WAV/FLAC（或按 ASR 服务要求 raw PCM）。
6. `asr_client.py` 调用 ASR 服务 `/v1/audio/transcriptions`。
7. ASR 服务返回识别文本（`{ "text": "..." }`）。
8. `service.py` 通过 Unix Socket 发送 `RecognizedText` 消息给 IBus Engine。
9. IBus Engine 调用 `commit_text()` 将文本写入当前输入框。

## 5. IPC 协议

IBus Engine 与 Daemon 之间使用 Unix Domain Socket 通信，消息为 JSON Line。

消息类型：

```python
# Daemon -> Engine
class RecognizedText(TypedDict):
    type: Literal["recognized"]
    text: str
    confidence: float | None
    error: str | None

class StateUpdate(TypedDict):
    type: Literal["state"]
    state: Literal["idle", "recording", "recognizing", "error"]
    message: str | None

# Engine -> Daemon
class StartRecording(TypedDict):
    type: Literal["start"]

class StopRecording(TypedDict):
    type: Literal["stop"]
```

## 6. ASR 服务集成

ASR 服务应实现 OpenAI-compatible 的 `/v1/audio/transcriptions`：

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer dummy" \
  -F file=@recording.wav \
  -F model=qwen3-asr \
  -F language=zh
```

本地服务选型与显存策略：

| 场景 | 策略 |
|------|------|
| 显存 >= 6GB | 优先 GPU 推理（vLLM/SGLang bf16/int8） |
| 显存 < 6GB 或 GPU 不可用 | CPU 推理（llama.cpp / transformers + bitsandbytes int4） |
| 自动检测 | `process_manager.py` 启动前调用 `nvidia-smi` 或 `torch.cuda` 判断 |

配置项：
- `asr.service_command`: 启动 ASR 服务的命令模板。
- `asr.endpoint`: HTTP 端点。
- `asr.model`: 模型路径或 Hugging Face repo ID。
- `asr.device`: `auto`/`cuda`/`cpu`。
- `asr.quantization`: `none`/`int8`/`int4`。

## 7. 全局热键

守护进程直接监听系统输入事件，推荐方案：

- 首选 `evdev`（读取 `/dev/input/event*`），无需 root 时当前用户需在 `input` 组。
- 备选 `pynput`（X11/Wayland 兼容性好，但需运行图形会话）。

默认热键：`<Super>` + `<Shift>` + `R`（可在配置中修改）。

## 8. 配置

配置文件路径：`~/.config/qwen3-asr-ime/config.yaml`

```yaml
hotkey:
  device: "evdev"  # evdev | pynput
  key: "<Super>+<Shift>+R"

audio:
  sample_rate: 16000
  channels: 1
  format: "int16"
  chunk_ms: 20

asr:
  endpoint: "http://127.0.0.1:8000/v1/audio/transcriptions"
  model: "Qwen/Qwen3-ASR"
  device: "auto"  # auto | cuda | cpu
  quantization: "auto"  # auto | none | int8 | int4
  api_key: "dummy"

ipc:
  socket_path: "/run/user/{UID}/qwen3-asr-ime.sock"

logging:
  level: "INFO"
```

## 9. 部署

安装流程：

```bash
cd /Data2/Code/python/qwen3-asr-ime
./bin/install.sh
```

`install.sh` 执行：
1. 安装 Python 依赖（`pygobject`, `ibus`, `pyaudio`, `requests`, `evdev` 等）。
2. 复制 IBus 组件到 `~/.ibus/components/` 或系统路径。
3. 创建 systemd user service 文件到 `~/.config/systemd/user/`。
4. 启用并启动服务：`systemctl --user enable --now qwen3-asr-ime`。
5. 提示用户重新登录或重启 IBus。

运行依赖：
- IBus
- Python 3.10+
- PortAudio / PyAudio
- 可选：NVIDIA 驱动 + CUDA（用于 GPU 推理）

## 10. 错误处理

| 错误场景 | 处理 |
|----------|------|
| ASR 服务未启动 | 守护进程尝试自动拉起；失败时发送 `StateUpdate(state="error", message="ASR 服务未启动")` 给引擎，引擎在状态栏显示错误图标。 |
| 录音设备不可用 | 守护进程启动时检测，记录日志并通知引擎；热键触发时给出提示。 |
| 识别为空 | 不 commit 任何文本。 |
| ASR 超时 | 设置 HTTP 超时（默认 30s），超时后返回错误状态。 |
| IPC 断开 | 引擎和守护进程均实现重连机制。 |

## 11. 测试策略

- 单元测试：
  - `recorder.py` 用模拟音频流测试开始/停止/数据完整性。
  - `asr_client.py` 用 `responses`/`httpx` mock 测试 API 调用和重试。
  - `protocol.py` 测试消息序列化/反序列化。
- 集成测试：
  - 启动守护进程 + mock ASR 服务，模拟热键事件，验证 IPC 消息。
- 端到端：
  - 在测试 IBus 环境中验证 commit 文本行为（可录制屏幕或读取 dummy 输入框内容）。

## 12. 后续扩展

- 支持 Fcitx5 前端。
- 支持流式识别（按住热键时边说边识别）。
- 支持多语言切换和自定义提示词（prompt）。
- 支持识别结果候选词（多候选）和纠错。

## 13. 参考

- Qwen3-ASR: https://huggingface.co/Qwen
- IBus Python Engine 示例: https://github.com/ibus/ibus-tmpl
- OpenAI Audio Transcriptions API: https://platform.openai.com/docs/api-reference/audio/createTranscription
