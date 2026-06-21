# Qwen3-ASR IME

Linux 语音输入法，基于 Qwen3-ASR-1.7B。按住 Ctrl 录音，松开自动识别并输入文本。

## 特性

- **全局热键**：按住 Ctrl 录音，松开输入（evdev 被动只读，不拦截其他输入）
- **本地推理**：Qwen3-ASR-1.7B 模型，RTX 4060 8GB 显存可行
- **全应用兼容**：终端、浏览器、编辑器均可输入中文/英文
- **零配置**：systemd user service，`./bin/install.sh` 一键安装

## 系统要求

- Ubuntu 24.04 / Linux（X11）
- Python 3.10+
- NVIDIA GPU（RTX 4060 8GB 或更高，如无 GPU 可尝试 0.6B 版本）
- IBus（可选，用于系统输入法集成）

## 快速开始

```bash
# 1. 安装系统依赖
sudo apt-get install -y libgirepository-2.0-dev libglib2.0-dev portaudio19-dev python3-dev xdotool xclip

# 2. 安装
cd qwen3-asr-ime
./bin/install.sh

# 3. 下载模型（从 ModelScope，国内快）
pip install modelscope
modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir /Data2/Models/Qwen3-ASR-1.7B

# 4. 启动 ASR 服务
python tools/asr_server.py

# 5. 守护进程已由 install.sh 启动（systemd user service）
#    按下 Ctrl 录音，松开输入
```

## 架构

```
Ctrl (evdev) → 守护进程 → 录音 (sounddevice) → Qwen3-ASR API → xclip+Ctrl+Shift+V → 文本框
                   ↕ Unix Socket IPC
             IBus Engine（可选）
```

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
  model: "Qwen/Qwen3-ASR-1.7B"
```

## 许可

MIT
