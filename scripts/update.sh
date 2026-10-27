#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  Zelyr Autonomous Updater
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

echo "[updater] Waiting 2 seconds for PySide6 application to exit cleanly..."
sleep 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}" || exit 1

echo "[updater] Starting repository sync..."
git fetch --all
git reset --hard origin/main

echo "[updater] Updating dependencies..."
VENV_DIR="${HOME}/.local/share/zelyr/venv"
if [ -d "${VENV_DIR}" ]; then
    source "${VENV_DIR}/bin/activate"
    pip install -r requirements.txt
    pip install -e .
else
    echo "[updater] WARNING: Virtual environment not found at ${VENV_DIR}"
fi

echo "[updater] Update complete. Relaunching Zelyr..."
if [ -f "${HOME}/.local/share/applications/Zelyr.desktop" ]; then
    DESKTOP_ID="Zelyr.desktop"
elif [ -f "${HOME}/.local/share/applications/zelyr.desktop" ]; then
    DESKTOP_ID="zelyr.desktop"
else
    DESKTOP_ID=""
fi
if [ -n "${DESKTOP_ID}" ]; then
    if command -v gtk-launch &>/dev/null; then
        gtk-launch "${DESKTOP_ID}" > /dev/null 2>&1 &
    else
        nohup "${VENV_DIR}/bin/python" main.py --gui > /dev/null 2>&1 &
    fi
else
    if [ -f "${VENV_DIR}/bin/python" ]; then
        nohup "${VENV_DIR}/bin/python" main.py --gui > /dev/null 2>&1 &
    else
        nohup python3 main.py --gui > /dev/null 2>&1 &
    fi
fi

exit 0
