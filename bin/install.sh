#!/usr/bin/env bash
set -euo pipefail

# Check system dependencies
for pkg in python3-dev portaudio19-dev; do
    if ! dpkg -l "$pkg" 2>/dev/null | grep -q "^ii"; then
        echo "Missing system package: $pkg"
        echo "Install: sudo apt-get install -y portaudio19-dev python3-dev"
        exit 1
    fi
done

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_DIR="${HOME}/.config/qwen3-asr-ime"
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"

mkdir -p "${CONFIG_DIR}" "${SYSTEMD_USER_DIR}"

# Install Python package
python3 -m pip install -e "${PROJECT_DIR}[vllm,transformers]"

# Create default config if missing
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

# Install systemd user services (template → instantiate)
PYTHON="$(which python3)"

# Install only the daemon service. The daemon itself starts the ASR backend
# subprocess on demand via BackendManager, so a separate server service is not
# needed and would conflict on port 8000.
for svc in qwen3-asr-ime; do
    sed -e "s|{{PYTHON}}|${PYTHON}|g" \
        -e "s|{{PROJECT_DIR}}|${PROJECT_DIR}|g" \
        "${PROJECT_DIR}/systemd/${svc}.service" \
        > "${SYSTEMD_USER_DIR}/${svc}.service"
done

systemctl --user daemon-reload
systemctl --user enable --now qwen3-asr-ime || true

echo ""
echo "==================== Installation Summary ===================="
echo ""
echo "  ✓ sudo apt-get install -y portaudio19-dev python3-dev"
echo "  ✓ Python package installed"
echo "  ✓ qwen3-asr-ime systemd user service enabled"
echo ""
echo "  Hold the configured hotkey (default: Ctrl) to record,"
echo "  release to input recognized text into the focused window."
echo ""
echo "============================================================="
