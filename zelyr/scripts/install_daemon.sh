#!/usr/bin/env bash
# Zelyr Systemd Daemon Installer

set -e

SERVICE_NAME="zelyr.service"
USER_SYSTEMD_DIR="${HOME}/.config/systemd/user"
SERVICE_PATH="${USER_SYSTEMD_DIR}/${SERVICE_NAME}"
ZELYR_EXECUTABLE="$(which python3) -m zelyr.api.cli daemon start"

echo "============================================="
echo "   Zelyr Daemon Installer (User systemd)     "
echo "============================================="

# Ensure systemd user directory exists
mkdir -p "${USER_SYSTEMD_DIR}"

echo "[1/3] Generating service file at ${SERVICE_PATH}..."

cat <<EOF > "${SERVICE_PATH}"
[Unit]
Description=Zelyr AI Orchestration Daemon
After=network.target

[Service]
Type=simple
ExecStart=${ZELYR_EXECUTABLE}
Restart=always
RestartSec=3
Environment=PATH=${PATH}
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
EOF

echo "[2/3] Reloading systemd daemon..."
systemctl --user daemon-reload

echo "[3/3] Enabling and starting Zelyr service..."
systemctl --user enable "${SERVICE_NAME}"
systemctl --user start "${SERVICE_NAME}"

echo "============================================="
echo " ✓ Zelyr Daemon installed and running!       "
echo " You can check status with: systemctl --user status zelyr"
echo " You can view logs with: journalctl --user -fu zelyr"
echo "============================================="
