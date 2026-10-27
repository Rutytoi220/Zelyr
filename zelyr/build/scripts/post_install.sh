#!/bin/bash
# Zelyr post-install hook — reload systemd user daemon
set -e

echo "[Zelyr] Reloading systemd user daemons…"
systemctl --user daemon-reload 2>/dev/null || true
systemctl --user enable zelyr.service 2>/dev/null || true

echo "[Zelyr] Post-install complete. Start with: systemctl --user start zelyr.service"
