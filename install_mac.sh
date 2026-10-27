#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  Zelyr Desktop v6.8 — macOS Installer
#  Installs the launchd background daemon and Applications shortcut
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

LAUNCHD_DIR="${HOME}/Library/LaunchAgents"
APP_DIR="${HOME}/Applications"

PLIST_SRC="${DEPLOY_DIR}/com.zelyr.daemon.plist"
ICON_SRC="${SCRIPT_DIR}/zelyr/gui/assets/logo.png"

# ── Banner ──────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}"
echo "  ╔═══════════════════════════════════════════════════════╗"
echo "  ║          Zelyr Desktop v6.8 — macOS Installer         ║"
echo "  ║       Local-First AI Orchestration for Apple Mac      ║"
echo "  ╚═══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── Preflight Checks ───────────────────────────────────────────
info "Running preflight checks..."

if [[ "$OSTYPE" != "darwin"* ]]; then
    error "This script is for macOS only. Please run install.sh on Linux."
    exit 1
fi

if ! command -v launchctl &>/dev/null; then
    error "launchctl not found. Is this a valid macOS environment?"
    exit 1
fi
success "macOS launchd detected."

if ! command -v zelyr &>/dev/null; then
    warn "'zelyr' command not found in PATH."
    warn "Make sure the package is installed (pip install -e .) before enabling the service."
fi

# ── Step 1: Launchd User Agent ───────────────────────────────
echo ""
info "Step 1/2: Installing launchd agent..."

if [ ! -f "${PLIST_SRC}" ]; then
    error "Plist file not found at ${PLIST_SRC}"
    exit 1
fi

mkdir -p "${LAUNCHD_DIR}"
cp "${PLIST_SRC}" "${LAUNCHD_DIR}/com.zelyr.daemon.plist"
success "Copied com.zelyr.daemon.plist → ${LAUNCHD_DIR}/"

# Unload if it already exists, then load
launchctl unload "${LAUNCHD_DIR}/com.zelyr.daemon.plist" 2>/dev/null || true
launchctl load "${LAUNCHD_DIR}/com.zelyr.daemon.plist"
success "Loaded and started launchd daemon."

# ── Step 2: AppleScript Application Shortcut ───────────────────
echo ""
info "Step 2/2: Creating Zelyr.app shortcut..."

mkdir -p "${APP_DIR}"
APP_BUNDLE="${APP_DIR}/Zelyr.app"
APP_MACOS="${APP_BUNDLE}/Contents/MacOS"
APP_RESOURCES="${APP_BUNDLE}/Contents/Resources"

# Create standard app bundle structure
mkdir -p "${APP_MACOS}"
mkdir -p "${APP_RESOURCES}"

# Create the launcher script
cat > "${APP_MACOS}/zelyr_launcher" << 'EOF'
#!/bin/bash
export PATH="/usr/local/bin:/opt/homebrew/bin:$HOME/.local/bin:$PATH"
exec zelyr-gui
EOF
chmod +x "${APP_MACOS}/zelyr_launcher"

# Create minimal Info.plist
cat > "${APP_BUNDLE}/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>zelyr_launcher</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIdentifier</key>
    <string>com.zelyr.desktop</string>
    <key>CFBundleName</key>
    <string>Zelyr</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>6.8.0</string>
    <key>LSUIElement</key>
    <false/>
</dict>
</plist>
EOF

# Convert PNG to ICNS (macOS format) via sips if available
if command -v sips &>/dev/null && command -v iconutil &>/dev/null && [ -f "${ICON_SRC}" ]; then
    ICONSET_DIR="/tmp/zelyr.iconset"
    mkdir -p "${ICONSET_DIR}"
    sips -z 16 16     "${ICON_SRC}" --out "${ICONSET_DIR}/icon_16x16.png" &>/dev/null
    sips -z 32 32     "${ICON_SRC}" --out "${ICONSET_DIR}/icon_16x16@2x.png" &>/dev/null
    sips -z 32 32     "${ICON_SRC}" --out "${ICONSET_DIR}/icon_32x32.png" &>/dev/null
    sips -z 64 64     "${ICON_SRC}" --out "${ICONSET_DIR}/icon_32x32@2x.png" &>/dev/null
    sips -z 128 128   "${ICON_SRC}" --out "${ICONSET_DIR}/icon_128x128.png" &>/dev/null
    sips -z 256 256   "${ICON_SRC}" --out "${ICONSET_DIR}/icon_128x128@2x.png" &>/dev/null
    sips -z 256 256   "${ICON_SRC}" --out "${ICONSET_DIR}/icon_256x256.png" &>/dev/null
    sips -z 512 512   "${ICON_SRC}" --out "${ICONSET_DIR}/icon_256x256@2x.png" &>/dev/null
    sips -z 512 512   "${ICON_SRC}" --out "${ICONSET_DIR}/icon_512x512.png" &>/dev/null
    sips -z 1024 1024 "${ICON_SRC}" --out "${ICONSET_DIR}/icon_512x512@2x.png" &>/dev/null
    iconutil -c icns "${ICONSET_DIR}" -o "${APP_RESOURCES}/AppIcon.icns"
    rm -rf "${ICONSET_DIR}"
    success "Generated AppIcon.icns from logo.png."
else
    warn "sips or iconutil not found (or missing logo.png). Skipping icon generation."
fi

success "Created ${APP_BUNDLE}."

# ── Summary ────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}"
echo "  ╔═══════════════════════════════════════════════════════╗"
echo "  ║            Installation Complete!                     ║"
echo "  ╠═══════════════════════════════════════════════════════╣"
echo "  ║  Daemon:   launchctl list | grep zelyr               ║"
echo "  ║  Logs:     tail -f /tmp/zelyr-daemon.log             ║"
echo "  ║  GUI:      Open ~/Applications/Zelyr.app             ║"
echo "  ╚═══════════════════════════════════════════════════════╝"
echo -e "${NC}"
