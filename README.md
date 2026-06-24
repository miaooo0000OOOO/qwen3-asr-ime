# Qwen3-ASR Voice Input

Linux 语音输入工具，基于 Qwen3-ASR。按住 Ctrl 录音，松开自动识别并输入文本到当前焦点窗口。

## 特性

- **全局热键**：按住 Ctrl 录音，松开输入（pynput 全局监听，不拦截其他输入）
- **本地推理**：默认使用 Qwen3-ASR-0.6B，RTX 4060 8GB 显存可行（可切换到 1.7B）
- **流式识别**：按住即开始识别，录音过程中实时返回 partial 文本
- **全应用兼容**：终端、浏览器、编辑器均可输入中文/英文
- **零配置**：systemd user service，`./bin/install.sh` 一键安装

## 系统要求

- Ubuntu 24.04 / Linux（X11）
- Python 3.10+
- NVIDIA GPU（RTX 4060 8GB 即可运行 0.6B；1.7B 需要更大显存）
- 必须安装 vLLM：`pip install qwen-asr[vllm]`

## 快速开始

```bash
# 1. 安装系统依赖
sudo apt-get install -y portaudio19-dev python3-dev

# 2. 安装（流式识别依赖 vLLM）
cd qwen3-asr-ime
pip install -e ".[vllm]"

# 3. 下载模型（从 ModelScope，国内快）
pip install modelscope

# 8GB 显存推荐 0.6B（默认）
modelscope download --model Qwen/Qwen3-ASR-0.6B --local_dir /Data2/Models/Qwen3-ASR-0.6B

# 显存充足可下载 1.7B，并修改服务中的 QWEN3_ASR_MODEL
# modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir /Data2/Models/Qwen3-ASR-1.7B
# 编辑 ~/.config/systemd/user/qwen3-asr-ime.service 中的 Environment=QWEN3_ASR_MODEL

# 4. 启动服务
# install.sh 已注册 systemd user service；守护进程会按需启动 ASR backend。
# 如需手动调试后端，可单独运行：python tools/asr_server.py
systemctl --user start qwen3-asr-ime

# 5. 按下 Ctrl 录音，松开输入到当前焦点窗口
```

## 卸载

```bash
# 停止并禁用服务，卸载 Python 包，保留用户配置
./bin/uninstall.sh

# 同时删除 ~/.config/qwen3-asr-ime/
./bin/uninstall.sh --purge
```

## 架构

```
Ctrl (pynput) → 守护进程 → 录音 (sounddevice) → WebSocket /v1/asr/stream → pynput.type → 文本框
```

## 服务端

`tools/asr_server.py` 提供两个接口：

- `POST /v1/asr/transcribe`：非流式识别，返回完整录音的一次性转写结果（当前默认配置使用此接口）。
- `WebSocket /v1/asr/stream`：流式识别，按住热键时守护进程实时发送 PCM16 音频 chunk，服务端调用 `Qwen3ASRModel.streaming_transcribe()` 增量识别并推送 partial/final 结果。

### `WebSocket /v1/asr/stream`

**消息协议（JSON）：**

- 客户端 → 服务端：
  - `{"type": "config", "language": "zh"}`（可选，第一条消息）
  - `{"type": "chunk", "format": "pcm", "audio": "<base64-int16-mono-16khz>"}`
  - `{"type": "chunk", "format": "wav", "audio": "<base64-wav>"}`
  - `{"type": "finish"}`
- 服务端 → 客户端：
  - `{"type": "ready"}`
  - `{"type": "partial", "text": "...", "language": "..."}`
  - `{"type": "final", "text": "...", "language": "..."}`
  - `{"type": "error", "message": "..."}`

### 环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `QWEN3_ASR_MODEL` | 模型路径或 HuggingFace repo | `/Data2/Models/Qwen3-ASR-0.6B` |
| `QWEN3_ASR_GPU_MEM` | vLLM GPU 内存占用 | `0.9` |
| `QWEN3_ASR_MAX_TOKENS` | vLLM max_new_tokens | `256` |
| `QWEN3_ASR_MAX_MODEL_LEN` | 限制上下文长度，降低显存占用 | `4096` |
| `QWEN3_ASR_ENFORCE_EAGER` | 禁用 vLLM 编译，降低启动显存 | `1` |
| `QWEN3_ASR_PREFIX_CACHING` | 是否启用 prefix caching | `0` |

## 日志

```bash
journalctl --user -u qwen3-asr-ime -f
```

## 测试

```bash
pytest -v
```

## 配置

`~/.config/qwen3-asr-ime/config.yaml`（默认自动生成）:

```yaml
hotkey:
  key: "CTRL"           # 自定义热键
asr:
  endpoint: "http://127.0.0.1:8000"
  model: "Qwen/Qwen3-ASR-0.6B"
  timeout: 30.0         # 识别超时（秒）
```

## 许可

