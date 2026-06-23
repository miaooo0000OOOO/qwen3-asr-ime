#!/usr/bin/env bash
set -euo pipefail

# Uninstall script for qwen3-asr-ime.
# Reverses the actions performed by bin/install.sh.
#
# Usage:
#     ./bin/uninstall.sh         # keep user config
#     ./bin/uninstall.sh --purge # also remove ~/.config/qwen3-asr-ime

PURGE=false
if [[ "${1:-}" == "--purge" ]]; then
    PURGE=true
fi

CONFIG_DIR="${HOME}/.config/qwen3-asr-ime"
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"

echo "Stopping and disabling systemd user services..."
systemctl --user stop qwen3-asr-ime qwen3-asr-server 2>/dev/null || true
systemctl --user disable qwen3-asr-ime qwen3-asr-server 2>/dev/null || true

echo "Removing systemd user service files..."
for svc in qwen3-asr-ime qwen3-asr-server; do
    svc_file="${SYSTEMD_USER_DIR}/${svc}.service"
    if [[ -f "${svc_file}" ]]; then
        rm -f "${svc_file}"
        echo "  removed ${svc_file}"
    fi
done

systemctl --user daemon-reload 2>/dev/null || true

if [[ "${PURGE}" == true ]]; then
    if [[ -d "${CONFIG_DIR}" ]]; then
        rm -rf "${CONFIG_DIR}"
        echo "Removed user config: ${CONFIG_DIR}"
    fi
else
    echo "Kept user config: ${CONFIG_DIR} (use --purge to remove it)"
fi

echo "Uninstalling Python package qwen3-asr-ime..."
python3 -m pip uninstall -y qwen3-asr-ime 2>/dev/null || true

echo ""
echo "==================== Uninstallation Summary ===================="
echo ""
echo "  ✓ systemd user services stopped, disabled, and removed"
if [[ "${PURGE}" == true ]]; then
    echo "  ✓ user configuration removed"
else
    echo "  ✓ user configuration kept (run with --purge to remove)"
fi
echo "  ✓ Python package uninstalled"
echo ""
echo "==============================================================="
