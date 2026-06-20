#!/usr/bin/env bash
set -euo pipefail

# Check system dependencies
for pkg in python3-dev libgirepository-2.0-dev libglib2.0-dev portaudio19-dev; do
    if ! dpkg -l "$pkg" 2>/dev/null | grep -q "^ii"; then
        echo "Missing system package: $pkg"
        echo "Install: sudo apt-get install -y libgirepository-2.0-dev libglib2.0-dev portaudio19-dev python3-dev"
        exit 1
    fi
done

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_DIR="${HOME}/.config/qwen3-asr-ime"
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"

mkdir -p "${CONFIG_DIR}" "${SYSTEMD_USER_DIR}"

# Install Python package
python3 -m pip install -e "${PROJECT_DIR}[ibus]"

# Create default config if missing
if [[ ! -f "${CONFIG_DIR}/config.yaml" ]]; then
cat > "${CONFIG_DIR}/config.yaml" <<'EOF'
hotkey:
  device: "evdev"
  key: "CTRL"
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

# Generate IBus component XML
IBUS_XML_CONTENT=$(cat << EOF
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
      <license>MIT</license>
      <author>Assistant</author>
      <icon>microphone</icon>
      <layout>default</layout>
      <longname>Qwen3-ASR</longname>
      <description>Qwen3-ASR Voice Input Method</description>
      <rank>50</rank>
      <symbol>🎤</symbol>
      <textdomain>qwen3-asr-ime</textdomain>
    </engine>
  </engines>
</component>
EOF
)

# Install to system component directory (requires sudo)
echo ""
echo "== Installing IBus component (requires sudo) =="
echo "This will copy the component XML to /usr/share/ibus/component/"
echo "Please enter your sudo password when prompted."
echo ""
COMPONENT_FILE=$(mktemp)
cat > "$COMPONENT_FILE" << INNEREOF
$IBUS_XML_CONTENT
INNEREOF

if sudo cp "$COMPONENT_FILE" /usr/share/ibus/component/qwen3-asr-ime.xml 2>/dev/null; then
    sudo chmod 644 /usr/share/ibus/component/qwen3-asr-ime.xml
    echo "IBus component installed to /usr/share/ibus/component/"
    rm -f "$COMPONENT_FILE"

    # Write IBus cache
    ibus write-cache 2>/dev/null || true

    # Restart IBus to pick up new engine
    echo "Restarting IBus..."
    ibus restart 2>/dev/null || true
    sleep 2

    # Register input source via gsettings
    if command -v gsettings &>/dev/null; then
        echo "Adding input source to GNOME settings..."
        python3 -c "
import subprocess, ast
try:
    result = subprocess.run(['gsettings', 'get', 'org.gnome.desktop.input-sources', 'sources'],
        capture_output=True, text=True, timeout=5)
    sources = ast.literal_eval(result.stdout.strip())
    if ('ibus', 'qwen3-asr-ime') not in sources:
        sources.append(('ibus', 'qwen3-asr-ime'))
        subprocess.run(['gsettings', 'set', 'org.gnome.desktop.input-sources', 'sources', str(sources)],
            check=True, timeout=5)
        print('Input source added successfully.')
    else:
        print('Input source already exists.')
except Exception as e:
    print(f'Note: could not auto-add input source: {e}')
"
    fi
else
    echo ""
    echo "WARNING: Could not install to system directory (need sudo)."
    echo "To fix manually, run these commands:"
    echo ""
    echo "  cp '$COMPONENT_FILE' /tmp/qwen3-asr-ime.xml"
    echo "  sudo cp /tmp/qwen3-asr-ime.xml /usr/share/ibus/component/"
    echo "  ibus write-cache"
    echo "  ibus restart"
    echo "  gsettings set org.gnome.desktop.input-sources sources \"[('xkb', 'us'), ('ibus', 'libpinyin'), ('ibus', 'qwen3-asr-ime')]\""
fi

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

echo ""
echo "==================== Installation Summary ===================="
echo ""
echo "  ✓ sudo apt-get install -y libgirepository-2.0-dev libglib2.0-dev portaudio19-dev python3-dev"
echo "  ✓ Python package installed"
echo "  ✓ systemd user service enabled"
echo ""
echo "  If IBus component registration failed above, run manually:"
echo "    sudo cp /usr/share/ibus/component/qwen3-asr-ime.xml  # (if not already there)"
echo "    ibus write-cache && ibus restart"
echo ""
echo "  Then switch to Qwen3-ASR input method:"
echo "    Super+Space  (or your IBus switch key)"
echo "    Hold Super+Shift+R to record, release to input"
echo ""
echo "============================================================="