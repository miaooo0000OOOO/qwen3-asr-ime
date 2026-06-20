#!/usr/bin/env bash
set -euo pipefail

# Check system dependencies
for pkg in python3-dev libgirepository2.0-dev libglib2.0-dev portaudio19-dev; do
    if ! dpkg -l "$pkg" 2>/dev/null | grep -q "^ii"; then
        echo "Missing system package: $pkg"
        echo "Install: sudo apt-get install -y libgirepository2.0-dev libglib2.0-dev portaudio19-dev python3-dev"
        exit 1
    fi
done

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_DIR="${HOME}/.config/qwen3-asr-ime"
IBUS_COMPONENT_DIR="${HOME}/.ibus/components"
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"

mkdir -p "${CONFIG_DIR}" "${IBUS_COMPONENT_DIR}" "${SYSTEMD_USER_DIR}"

python3 -m pip install -e "${PROJECT_DIR}[ibus]"

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
