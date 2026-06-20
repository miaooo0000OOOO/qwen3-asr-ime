# Qwen3-ASR Linux 语音输入法

基于 Qwen3-ASR 的 IBus 语音输入法。按住全局热键录音，松开后自动输入识别文本。

## 特性

- IBus 输入法前端
- 本地 Qwen3-ASR API 服务（vLLM/SGLang/llama.cpp）
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

## 项目结构

```
src/qwen3_asr_ime/
├── common/         # 公共模块（配置、协议、日志）
├── daemon/         # 语音输入守护进程
│   ├── service.py        # 主服务（asyncio 事件循环）
│   ├── recorder.py       # 音频录制
│   ├── hotkey.py         # 全局热键监听
│   ├── asr_client.py     # ASR HTTP 客户端
│   └── process_manager.py# ASR 服务进程管理
├── ibus/           # IBus 引擎
│   ├── engine.py         # IBus Engine 实现
│   ├── factory.py        # IBus Factory
│   └── main.py           # 入口点
```
