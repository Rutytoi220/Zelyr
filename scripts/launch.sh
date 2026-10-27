#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  Zelyr Desktop — Launch Wrapper
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

# Ensure Qt correctly falls back between Wayland and X11
export QT_QPA_PLATFORM="wayland;xcb"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${HOME}/.local/share/zelyr/venv"

if [ ! -d "${VENV_DIR}" ]; then
    echo -e "\033[0;31m[ERROR]\033[0m Virtual environment not found at ${VENV_DIR}."
    echo "Please run ./install.sh first."
    exit 1
fi

echo -e "\033[1;36m[INFO]\033[0m Activating Zelyr environment..."
source "${VENV_DIR}/bin/activate"

echo -e "\033[1;32m[OK]\033[0m Launching Zelyr Engine..."
exec python3 "${SCRIPT_DIR}/main.py" "$@"
