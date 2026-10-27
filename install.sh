#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  Zelyr Desktop v8.2 — Native Linux Installer
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

# ── Colors ──────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[  OK]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[FAIL]${NC}  $*"; }

# ── Paths ───────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="${SCRIPT_DIR}/deploy"

CONFIG_DIR="${HOME}/.config/zelyr"
DATA_DIR="${HOME}/.local/share/zelyr"
VENV_DIR="${DATA_DIR}/venv"
DESKTOP_DIR="${HOME}/.local/share/applications"
ICON_DIR="${HOME}/.local/share/icons/hicolor/256x256/apps"

ICON_SRC="${SCRIPT_DIR}/zelyr/gui/assets/logo.png"

# ── Banner ──────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}"
echo "  ╔═══════════════════════════════════════════════════════╗"
echo "  ║          Zelyr Desktop v8.2 — Linux Installer         ║"
echo "  ║       Local-First AI Orchestration for Linux          ║"
echo "  ╚═══════════════════════════════════════════════════════╝"
echo -e "${NC}"

info "Running preflight checks..."

# Check Python version
if ! command -v python3 &>/dev/null; then
    error "python3 is not installed."
    exit 1
fi

if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    error "Python 3.10 or higher is required."
    exit 1
fi
success "Python 3.10+ detected."

# ── Step 1: Directories & Virtual Environment ───────────────────
echo ""
info "${BOLD}Step 1/3: Setting up virtual environment...${NC}"

mkdir -p "${CONFIG_DIR}" "${DATA_DIR}"
success "Created local directories."

if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
    success "Created virtual environment at ${VENV_DIR}."
else
    info "Virtual environment already exists."
fi

info "Installing requirements..."
"${VENV_DIR}/bin/pip" install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/pip" install -r "${SCRIPT_DIR}/requirements.txt"
"${VENV_DIR}/bin/pip" install -e "${SCRIPT_DIR}"
success "Dependencies installed."

# ── Step 2: Application Icon ──────────────────────────────────
echo ""
info "${BOLD}Step 2/3: Installing application icon...${NC}"

mkdir -p "${ICON_DIR}"
if [ -f "${ICON_SRC}" ]; then
    cp "${ICON_SRC}" "${ICON_DIR}/zelyr.png"
    success "Installed icon → ${ICON_DIR}/zelyr.png"
    ICON_DEST="${ICON_DIR}/zelyr.png"
else
    warn "Icon not found at ${ICON_SRC}. Will use generic icon."
    ICON_DEST="utilities-terminal"
fi

# ── Step 3: Desktop Entry ──────────────────────────────────────
echo ""
info "${BOLD}Step 3/3: Installing desktop entry...${NC}"

mkdir -p "${DESKTOP_DIR}"
DESKTOP_SRC="${DEPLOY_DIR}/zelyr.desktop.template"
DESKTOP_DEST="${DESKTOP_DIR}/Zelyr.desktop"

if [ ! -f "${DESKTOP_SRC}" ]; then
    error "Desktop template not found at ${DESKTOP_SRC}"
    exit 1
fi

EXEC_CMD="${SCRIPT_DIR}/scripts/launch.sh --gui"

sed -e "s|{{EXEC_PATH}}|${EXEC_CMD}|g" \
    -e "s|{{ICON_PATH}}|${ICON_DEST}|g" \
    "${DESKTOP_SRC}" > "${DESKTOP_DEST}"

chmod +x "${DESKTOP_DEST}"
success "Created Zelyr.desktop → ${DESKTOP_DEST}"

# ── Update Desktop Database ───────────────────────────────────
if command -v update-desktop-database &>/dev/null; then
    update-desktop-database "${DESKTOP_DIR}" 2>/dev/null || true
    success "Desktop database updated."
fi

if command -v gtk-update-icon-cache &>/dev/null; then
    gtk-update-icon-cache -f -t "${HOME}/.local/share/icons/hicolor" 2>/dev/null || true
    success "Icon cache updated."
fi

# ── Summary ────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}"
echo "  ╔═══════════════════════════════════════════════════════╗"
echo "  ║            Installation Complete!                     ║"
echo "  ╠═══════════════════════════════════════════════════════╣"
echo "  ║  GUI:      Search 'Zelyr' in your app launcher       ║"
echo "  ║  Binary:   ${SCRIPT_DIR}/scripts/launch.sh"
echo "  ╚═══════════════════════════════════════════════════════╝"
echo -e "${NC}"
