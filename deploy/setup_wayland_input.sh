#!/bin/bash
set -e

echo "==============================================="
echo " Zelyr Linux User-Space Input Setup Script"
echo "==============================================="
echo "This script configures udev rules and user services to allow"
echo "Zelyr to control the mouse and keyboard on Linux (Wayland/X11)"
echo "without requiring root/sudo privileges during execution."
echo ""

# 1. Check if ydotool is installed
if ! command -v ydotool &> /dev/null; then
    echo "ERROR: 'ydotool' is not installed."
    echo "Please install it using your package manager (e.g. 'sudo apt install ydotool' or 'sudo pacman -S ydotool') and run this script again."
    exit 1
fi

# 2. Setup Udev Rules
echo "[1/3] Setting up udev rules for /dev/uinput..."
UDEV_RULE_FILE="/etc/udev/rules.d/99-uinput.rules"
UDEV_RULE='KERNEL=="uinput", MODE="0666", OPTIONS+="static_node=uinput"'

if [ ! -f "$UDEV_RULE_FILE" ] || ! grep -q "0666" "$UDEV_RULE_FILE"; then
    echo "Requesting sudo to create udev rule: $UDEV_RULE_FILE"
    echo "$UDEV_RULE" | sudo tee "$UDEV_RULE_FILE" > /dev/null
    sudo udevadm control --reload-rules && sudo udevadm trigger
    echo "Udev rules configured."
else
    echo "Udev rule already exists at $UDEV_RULE_FILE."
fi

# 3. Apply immediately for current session
echo "[2/3] Applying uinput permissions for current session..."
if [ -e /dev/uinput ]; then
    sudo chmod 666 /dev/uinput
    echo "Permissions applied to /dev/uinput."
else
    echo "WARNING: /dev/uinput does not exist. The uinput kernel module might need to be loaded."
    sudo modprobe uinput || true
fi

# 4. Setup Systemd User Service for ydotoold
echo "[3/3] Configuring ydotoold systemd user service..."
SERVICE_DIR="$HOME/.config/systemd/user"
mkdir -p "$SERVICE_DIR"
SERVICE_FILE="$SERVICE_DIR/ydotoold.service"

cat <<EOF > "$SERVICE_FILE"
[Unit]
Description=ydotool daemon
After=graphical-session.target

[Service]
ExecStart=$(command -v ydotoold || echo "/usr/bin/ydotoold")
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now ydotoold.service

echo ""
echo "==============================================="
echo "Setup Complete!"
echo "ydotoold is now running as a user service."
echo "Zelyr can now physically interact with your OS in user-space."
echo "==============================================="
